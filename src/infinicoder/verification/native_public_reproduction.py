"""The same initial public observation in both arms;optional trace in one arm."""

from contextlib import contextmanager
import copy
import json
import time

from infinicoder.verification.native_public_failure_slice import NativePublicFailureSlice
from infinicoder.verification.public_reproduction import PublicDevelopmentReproduction
from infinicoder.verification.public_replay import digest


class NativePublicReproductionBootstrap(NativePublicFailureSlice):
    identity = "native_common_public_reproduction_v1"
    expose_tools = True
    command_names = ("ic-public-reproduction",)
    HELP = "ic-public-reproduction output START END retrieves the exact historical public output;at most2000 characters;never re-executes a command."

    def __init__(self, count_messages, *, tokenizer_identity, reproduction, diagnostics=True):
        if (
            not isinstance(reproduction, PublicDevelopmentReproduction)
            or type(diagnostics) is not bool
        ):
            raise ValueError("explicit common reproduction and diagnostic switch required")
        super().__init__(
            count_messages,
            tokenizer_identity=tokenizer_identity,
            capability=reproduction.diagnostic,
        )
        self.reproduction = reproduction
        self.diagnostics = diagnostics

    @contextmanager
    def bind(self, request, model, environment, deadline):
        self.bootstrap_report = None
        self.bootstrap_attempted = False
        try:
            with super().bind(request, model, environment, deadline):
                with self.reproduction.ordinary.operation.bind(request, deadline) as lease:
                    self.bootstrap_lease = lease
                    yield self
        finally:
            self._persist_bootstrap()

    def _persist_bootstrap(self):
        if self.bootstrap_report is None:
            return
        try:
            path = self.state / "native-public-bootstrap.json"
            temporary = path.with_suffix(".next")
            temporary.write_text(
                json.dumps(self.bootstrap_report, ensure_ascii=False, indent=2), encoding="utf8"
            )
            temporary.replace(path)
        except (OSError, ValueError, TypeError) as error:
            self.bootstrap_report["persistence_error"] = type(error).__name__

    def before_query(self, messages):
        if self.bootstrap_attempted:
            if self.diagnostics:
                super().before_query(messages)
            return
        self.bootstrap_attempted = True
        spec = self.capability.spec
        operation = self.reproduction.ordinary.operation
        start = time.monotonic()
        self.bootstrap_report = dict(
            status="unavailable",
            views=[],
            instrumented=False,
            public_origin=spec.public_origin,
            auxiliary_model_calls=0,
            private_final_used=False,
        )
        try:
            self._namespace()
            before = self._snapshot()
            if any(v.get("kind") != "file" for v in before.values()):
                raise ValueError("incomplete public source coverage")
            if any(before[p].get("sha256") != pin for p, pin in spec.public_input_hashes.items()):
                raise ValueError("public input changed")
            if self._remaining() <= spec.final_reserve_seconds + 1:
                raise TimeoutError("public bootstrap final reserve unavailable")
            dispatch = dict(
                kind="common_public_reproduction",
                implementation_identity=operation.implementation_identity,
                scope=operation.public_scope,
                coverage=operation.public_coverage,
            )
            cutoff = min(
                self.deadline - spec.final_reserve_seconds, time.monotonic() + spec.timeout_seconds
            )
            with self.controller_public_dispatch(dispatch):
                result = self.bootstrap_lease.invoke(cutoff)
            after = self._snapshot()
            self.record_controller_return(dispatch, before=before, after=after, report=result)
            if after != before:
                raise ValueError("public bootstrap changed observed source")
            trace = result.get("trace") if isinstance(result, dict) else None
            if not isinstance(trace, dict) or trace.get("instrumented") is not False:
                raise ValueError("ordinary public report unavailable")
            pins = {p: v["sha256"] for p, v in before.items()}
            if (
                trace.get("source_after") != pins
                or {x["path"]: x["sha256"] for x in trace["sources"]} != pins
                or tuple(trace["public_argv"]) != spec.argv
            ):
                raise ValueError("ordinary public report version mismatch")
            process = result.get("process", {})
            if process.get("timed_out") or process.get("truncated"):
                status = "unavailable"
            elif trace.get("exception_type") in ("ImportError", "ModuleNotFoundError", "OSError"):
                status = "environment_or_input_error"
            else:
                status = (
                    "observed_exit_zero" if trace["exit_code"] == 0 else "observed_nonzero_exit"
                )
            self.bootstrap_report.update(
                status=status, result=result, tested_sources=pins, source_versions="current"
            )
            if status == "observed_nonzero_exit" and self.diagnostics:
                self.pending_failure = dict(
                    id="bootstrap-0",
                    origin="controller_public_bootstrap",
                    command=spec.command,
                    before=copy.deepcopy(before),
                    after=copy.deepcopy(after),
                )
                super().before_query(messages)
        except Exception as error:
            self.bootstrap_report["reason"] = type(error).__name__ + ": " + str(error)
        finally:
            if not self.diagnostics:
                self.attempted = True
            self.bootstrap_report["controller_wall_seconds"] = time.monotonic() - start
            self._persist_bootstrap()

    def _bootstrap_text(self):
        report = self.bootstrap_report
        if report is None or "result" not in report:
            return None
        current = {p: v.get("sha256") for p, v in self._snapshot().items()}
        freshness = "current" if current == report["tested_sources"] else "historical"
        report["source_versions"] = freshness
        output = report["result"].get("process", {}).get("output", "")
        data = dict(
            authority="ordinary public development observation;not final acceptance",
            version_state=freshness,
            status=report["status"],
            command=self.capability.spec.command,
            output_tail=output[-1400:],
            output_characters=len(output),
            output_omitted_characters=max(0, len(output) - 1400),
            retrieve_full_output="ic-public-reproduction output START END",
            source_set_sha256=digest(json.dumps(report["tested_sources"], sort_keys=True).encode()),
        )
        return "PUBLIC DEVELOPMENT REPRODUCTION\n" + json.dumps(
            data, ensure_ascii=False, separators=(",", ":")
        )

    def project(self, messages):
        original = super().project(messages)
        return self.render_bootstrap(original, messages)

    def render_bootstrap(self, original, messages, *, fallback=None):
        """Render the common observation without preparing another history."""
        text = self._bootstrap_text()
        if text is None:
            return original
        view = dict(role="user", content=text)
        projected = [*original, view]
        if self._count(projected) > self.limit:
            # Keep the common report if only the optional diagnostic no longer
            # fits alongside it;the ordinary task/history are never shortened.
            projected = [*(messages if fallback is None else fallback), view]
        if self._count(projected) > self.limit:
            return messages if fallback is None else fallback
        self.bootstrap_report["views"].append(
            dict(view_sha256=digest(text.encode()), canonical_messages=len(messages))
        )
        self._persist_bootstrap()
        return projected

    def tool(self, args):
        if len(args) != 3 or args[0] != "output":
            raise ValueError(self.HELP)
        start, end = map(int, args[1:])
        if not 0 <= start < end <= start + 2000:
            raise ValueError("public output range must be at most2000 characters")
        text = (self.bootstrap_report or {}).get("result", {}).get("process", {}).get("output", "")
        return json.dumps(
            dict(
                historical_public_output=True,
                start=start,
                end=min(end, len(text)),
                characters=len(text),
                text=text[start:end],
                commands_replayed=False,
            ),
            ensure_ascii=False,
        )
