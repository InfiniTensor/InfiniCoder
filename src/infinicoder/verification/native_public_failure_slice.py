"""Public execution evidence at one ordinary SDK query boundary;default off."""

from contextlib import contextmanager
import ast
import copy
import json
from pathlib import Path
import time

from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.runtime.native_plain import NativePlainObserver
from infinicoder.verification.public_replay import PublicReplayCapability, digest


class NativePublicFailureSlice(NativePlainObserver):
    identity = "native_public_failure_slice_v1"

    def __init__(self, count_messages, *, tokenizer_identity, capability, implementation_paths=()):
        if not isinstance(capability, PublicReplayCapability):
            raise ValueError("caller-owned public replay capability required")
        super().__init__(
            count_messages,
            tokenizer_identity=tokenizer_identity,
            implementation_paths=capability.spec.source_paths,
        )
        self.capability = capability

    def validate(self, request):
        super().validate(request)
        if (
            request.context_adapter is not self
            or request.resume
            or any(
                getattr(request, k) is not None
                for k in (
                    "planning_adapter",
                    "repair_adapter",
                    "candidate_search",
                    "implementation_proposal",
                )
            )
        ):
            raise UnsupportedCore(
                "public replay requires an ordinary single run without other mechanisms"
            )
        if getattr(request.transport_adapter, "delivery_policy", None) is not None:
            raise UnsupportedCore("public replay requires the common ordinary transport")
        if Path(self.capability.operation.workspace).resolve() != Path(request.workspace).resolve():
            raise UnsupportedCore("public replay workspace mismatch")
        validate_public_sources(self.capability, request)

    @contextmanager
    def bind(self, request, model, environment, deadline):
        if getattr(self, "bound_once", False):
            raise UnsupportedCore("public replay cannot resume a prior episode")
        self.bound_once = True
        self.request = request
        self.environment = environment
        self.pending_failure = None
        self.attempted = False
        self.report = None
        self.binding = None
        try:
            with super().bind(request, model, environment, deadline):
                with self.capability.operation.bind(request, deadline) as lease:
                    self.replay_lease = lease
                    yield self
        finally:
            self._persist()
            try:
                path = self.state / "native-memory-info.json"
                if path.exists():
                    info = json.loads(path.read_bytes())
                    info.update(
                        identity=self.identity,
                        public_failure_slice=True,
                        automatic_compression=False,
                        canonical_history_changed=False,
                        auxiliary_model_calls=0,
                        projection="Original canonical SDK history plus a bounded current public execution slice",
                        prepared_diagnostic_views=len(self.report["views"]) if self.report else 0,
                    )
                    path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf8")
            except (OSError, ValueError, TypeError, AttributeError):
                pass

    def _persist(self):
        if self.report is None:
            return
        path = self.state / "native-public-failure-slice.json"
        temporary = path.with_suffix(".next")
        try:
            temporary.write_text(
                json.dumps(self.report, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf8",
            )
            temporary.replace(path)
        except (OSError, ValueError, TypeError) as error:
            self.report["persistence_error"] = type(error).__name__

    def _namespace(self):
        describe = self.capability.namespace_binding or getattr(
            self.environment, "implementation_workspace_binding", None
        )
        if not callable(describe):
            raise UnsupportedCore("no exact live workspace mapping")
        binding = describe()
        if (
            binding.get("root") != str(self.workspace)
            or binding.get("model_root") != self.capability.model_root
        ):
            raise UnsupportedCore("public replay source mapping mismatch")
        if self.binding is not None and self.binding != binding:
            raise UnsupportedCore("source namespace changed")
        self.binding = copy.deepcopy(binding)

    def _snapshot(self):
        result = super()._snapshot()
        for relative in self.capability.readonly_sources or {}:
            path = self.capability.source_path(relative)
            try:
                if (
                    any(p.is_symlink() for p in (path, *path.parents))
                    or not path.is_file()
                    or path.stat().st_size > 1024**2
                ):
                    result[relative] = dict(kind="unknown", reason="invalid public source binding")
                    continue
                raw = path.read_bytes()
                result[relative] = dict(kind="file", sha256=digest(raw))
            except OSError as error:
                result[relative] = dict(kind="unknown", reason=type(error).__name__)
        return result

    def execution_completed(self, event):
        if (
            not self.attempted
            and event["command"] == self.capability.spec.command
            and type(event.get("returncode")) is int
            and event["returncode"] != 0
            and event.get("observer_complete")
            and not event.get("error")
            and event["before"] == event["after"]
        ):
            self.pending_failure = copy.deepcopy(event)

    def before_query(self, messages):
        if self.attempted or self.pending_failure is None:
            return
        self.attempted = True
        spec = self.capability.spec
        failure = self.pending_failure
        started = time.monotonic()
        self.report = dict(
            identity=self.identity,
            status="not_applicable",
            trigger_execution=failure["id"],
            trigger_origin=failure.get("origin", "ordinary_sdk_execution"),
            ordinary_command=failure["command"],
            public_origin=spec.public_origin,
            views=[],
            auxiliary_model_calls=0,
            private_final_used=False,
        )
        try:
            self._namespace()
            before = self._snapshot()
            if before != failure["after"]:
                raise ValueError("source changed after the public failure")
            if any(before[p].get("sha256") != pin for p, pin in spec.public_input_hashes.items()):
                raise ValueError("caller-public input changed")
            if any(v.get("kind") != "file" for v in before.values()):
                raise ValueError("source coverage is incomplete")
            if self._remaining() <= spec.final_reserve_seconds + 1:
                raise TimeoutError("caller final reserve unavailable")
            cutoff = min(
                self.deadline - spec.final_reserve_seconds, time.monotonic() + spec.timeout_seconds
            )
            result = self.replay_lease.invoke(cutoff)
            if self._snapshot() != before:
                raise ValueError("diagnostic changed observed source")
            trace = result.get("trace") if isinstance(result, dict) else None
            if not isinstance(trace, dict) or trace.get("collector_error"):
                raise ValueError("no usable bounded public trace")
            pins = {p: v["sha256"] for p, v in before.items()}
            if (
                {x["path"]: x["sha256"] for x in trace["sources"]} != pins
                or trace["source_after"] != pins
                or tuple(trace["public_argv"]) != spec.argv
            ):
                raise ValueError("trace version or public argv mismatch")
            if trace.get("capture_mode", "prefix") != spec.capture_mode:
                raise ValueError("public trace capture policy mismatch")
            if not trace["events"] and not trace.get("terminal_anchors"):
                raise ValueError("no observed implementation frames or terminal positions")
            self.report.update(
                status="observed",
                tested_sources=pins,
                namespace=self.binding,
                result=result,
                source_versions="current",
                diagnostic_wall_seconds=time.monotonic() - started,
            )
            self._save_event(
                "public_failure_slice",
                dict(
                    trigger_execution=failure["id"],
                    tested_sources=pins,
                    wall_seconds=self.report["diagnostic_wall_seconds"],
                    ordinary_history_unchanged=True,
                ),
            )
        except Exception as error:
            self.report.update(
                status="unavailable", reason=type(error).__name__ + ": " + str(error)
            )
        finally:
            self.report["controller_wall_seconds"] = time.monotonic() - started
            self._persist()

    def project(self, messages):
        original = super().project(messages)
        return self.render_view(original, messages)

    def render_view(self, original, messages):
        """Render into an existing owner's packet;no query/journal hook."""
        if self.report is None or self.report["status"] != "observed":
            return original
        pins = {p: v.get("sha256") for p, v in self._snapshot().items()}
        if pins != self.report["tested_sources"]:
            self.report["source_versions"] = "historical"
            self._persist()
            return original
        trace = self.report["result"]["trace"]
        events = trace["events"]
        tail = trace.get("capture_mode") == "failure_tail"
        exceptions = [x for x in events if x["kind"] == "exception"]
        implementation_errors = [
            x for x in exceptions if x["path"] not in self.capability.spec.public_input_hashes
        ]
        anchors = trace.get("terminal_anchors", []) if tail else []
        if tail and not anchors and not exceptions:
            self._save_event(
                "public_slice_omitted",
                dict(reason="No recorded failure position;ordinary returns are not failures"),
            )
            return original
        focus = anchors[-1] if anchors else (implementation_errors or exceptions or events)[-1]
        identities = {v for v in (focus.get("frame"), focus.get("parent")) if v is not None}
        wanted = [x for x in events if x["kind"] == "call" and x["frame"] in identities]
        selected = []
        priority = (
            [focus, *([anchors[0]] if anchors and anchors[0] != focus else []), *wanted[-2:]]
            if tail
            else [*wanted[-2:], focus]
        )
        for item in priority:
            if item in selected:
                continue
            compact = {k: v for k, v in item.items() if k != "sha256"}
            for name in ("parameters", "locals"):
                if name in compact:
                    values = compact[name]
                    compact[name] = dict(list(values.items())[:4])
                    compact["omitted_" + name] = list(values)[4:]
            selected.append(compact)
        data = dict(
            authority="observed public execution;not acceptance;hypotheses still require testing",
            sources=self.report["tested_sources"],
            events=selected,
            truncated=self.report["result"]["trace"]["truncated"],
            limitations=self.report["result"]["trace"]["limitations"],
        )
        if tail:
            data["failure_capture"] = dict(
                terminal_status=trace["terminal_status"],
                failure_relation="unhandled terminal positions;not proven root cause"
                if anchors
                else "observed exception;terminal relation unknown",
                evicted_events=trace["evicted_events"],
                coverage_exhausted=trace["coverage_exhausted"],
                stop_reason=trace["stop_reason"],
                call_ancestry=trace["call_ancestry"],
                terminal_depth_limited=trace["terminal_depth_limited"],
                unknown_frame_calls=trace["unknown_frame_calls"],
                omitted_terminal_anchors=trace["terminal_anchor_omissions"],
            )
        spec = self.capability.spec
        encode = lambda value: (
            "PUBLIC FAILURE EXECUTION SLICE\n"
            + json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        )
        text = encode(data)
        if tail:
            # Preserve the terminal position before optional older calls.
            while len(selected) > 1 and (
                self._count([dict(role="user", content=text)]) > spec.context_tokens
                or self._count([*original, dict(role="user", content=text)]) > self.limit
            ):
                selected.pop()
                text = encode(data)
        if (
            self._count([dict(role="user", content=text)]) > spec.context_tokens
            or self._count([*original, dict(role="user", content=text)]) > self.limit
        ):
            self._save_event(
                "public_slice_omitted",
                dict(reason="actual tokenizer budget;original input retained"),
            )
            return original
        # Complete current local definitions only;omit whole definitions when
        # they do not fit,never silently clip a function or the canonical task.
        definitions = []
        for item in reversed(selected):
            if len(definitions) == 2:
                break
            path = self.capability.source_path(item["path"])
            raw = path.read_bytes()
            if digest(raw) != pins[item["path"]]:
                continue
            source = raw.decode("utf8")
            try:
                node = next(
                    (
                        n
                        for n in ast.walk(ast.parse(source))
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and n.name == item["function"]
                        and n.lineno <= item["line"] <= n.end_lineno
                    ),
                    None,
                )
            except (SyntaxError, ValueError):
                continue
            if node is None:
                continue
            definition = dict(
                path=item["path"], line=node.lineno, source=ast.get_source_segment(source, node)
            )
            if definition in definitions:
                continue
            proposed = encode(dict(data, definitions=[*definitions, definition]))
            if (
                self._count([dict(role="user", content=proposed)]) <= spec.context_tokens
                and self._count([*original, dict(role="user", content=proposed)]) <= self.limit
            ):
                definitions.append(definition)
                text = proposed
        if {p: v.get("sha256") for p, v in self._snapshot().items()} != pins:
            self.report["source_versions"] = "historical"
            self._persist()
            return original
        self.report["views"].append(
            dict(view_sha256=digest(text.encode()), canonical_messages=len(messages))
        )
        self._persist()
        return [*original, dict(role="user", content=text)]


def validate_public_sources(capability, request):
    """Shared source-only admission,without installing an observer or model."""
    root = Path(request.workspace).resolve()
    if Path(capability.operation.workspace).resolve() != root:
        raise UnsupportedCore("public replay workspace mismatch")
    for relative in capability.spec.source_paths:
        path = capability.source_path(relative)
        external = relative in (capability.readonly_sources or {})
        if (
            any(p.is_symlink() for p in (path, *path.parents))
            or not external
            and not path.resolve().is_relative_to(root)
            or not path.is_file()
            or path.stat().st_size > 1024**2
        ):
            raise UnsupportedCore("public replay requires bounded regular caller-readable sources")
        pin = capability.spec.public_input_hashes.get(relative)
        if pin is not None and digest(path.read_bytes()) != pin:
            raise UnsupportedCore(
                "public reproduction input differs from its frozen caller contract"
            )
