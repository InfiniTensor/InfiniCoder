"""At most two actual candidate-bound public probes in the original SDK run."""

from contextlib import contextmanager
import hashlib
import json
import time

from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.runtime.native_plain import NativePlainObserver
from infinicoder.verification.public_transition_probe import digest, encoded
from infinicoder.verification.transition_reports import PublicTransitionCapability


def _brief(value, depth=0):
    if depth > 3:
        return dict(omitted=True)
    if isinstance(value, dict):
        if value.get("kind") == "tensor":
            values = value["values"]
            return dict(
                dtype=value["dtype"],
                shape=value["shape"],
                sample=values[:4],
                sample_omitted=max(0, len(values) - 4),
                element_strides=value["element_strides"],
                storage_offset=value["storage_offset"],
            )
        keys = list(value)[:4]
        result = {k: _brief(value[k], depth + 1) for k in keys}
        if len(value) > 4:
            result["other_fields_omitted"] = len(value) - 4
        return result
    if isinstance(value, list):
        return [_brief(v, depth + 1) for v in value[:4]]
    return value


class NativeTransitionProbe(NativePlainObserver):
    identity = "native_public_transition_probe_v1"

    def __init__(
        self, count_messages, *, tokenizer_identity, implementation_paths=(), capability=None
    ):
        super().__init__(
            count_messages,
            tokenizer_identity=tokenizer_identity,
            implementation_paths=implementation_paths,
        )
        if capability is not None and not isinstance(capability, PublicTransitionCapability):
            raise ValueError("explicit caller public transition capability required")
        self.capability = capability

    def validate(self, request):
        super().validate(request)
        if self.capability is not None:
            self.capability.validate(request)
            if not set(self.capability.operation.observed_paths).issubset(self.paths):
                raise UnsupportedCore(
                    "transition paths must be in the original explicitly observed source set"
                )

    def _save_probe(self):
        (self.state / "native-transition-probe.json").write_bytes(
            (encoded(self.report) + "\n").encode()
        )

    @contextmanager
    def bind(self, request, model, environment, deadline):
        with super().bind(request, model, environment, deadline):
            self.last_handled = -1
            self.latest = None
            self.report = dict(
                identity=self.identity,
                records=[],
                deliveries=[],
                extra_model_calls=0,
                canonical_history_changed=False,
                no_source_prefix=True,
                no_automatic_edit=True,
                status="N/A;no caller public reference"
                if self.capability is None
                else "waiting_for_actual_edit",
            )
            self.index.db.execute(
                "CREATE TABLE native_transition_inputs(id INTEGER PRIMARY KEY,messages TEXT,accounting TEXT)"
            )
            self.index.db.commit()
            try:
                yield self
            finally:
                self._save_probe()
        path = self.state / "native-memory-info.json"
        info = json.loads(path.read_bytes())
        info.update(
            projection="Canonical SDK history unchanged;optional actual public transition event appended to input",
            automatic_source=False,
            automatic_compression=False,
            transition_probe=self.capability is not None,
        )
        path.write_bytes((encoded(info) + "\n").encode())

    def before_query(self, messages):
        if self.capability is None or not self.executions:
            return
        event = self.executions[-1]
        if event["id"] <= self.last_handled:
            return
        self.last_handled = event["id"]
        paths = set(self.capability.operation.observed_paths)
        if (
            not paths.intersection(event.get("changed", ()))
            or not event.get("message_ids")
            or not event.get("observer_complete")
            or event.get("error") is not None
        ):
            return
        before = self._snapshot()
        selected = {p: before[p] for p in paths}
        if paths.intersection(self.unattributed) or any(
            v.get("kind") != "file" for v in selected.values()
        ):
            return
        store = self.capability.store
        if store is None:
            raise UnsupportedCore("public probe has no caller-bound run/read reports")
        started = time.monotonic()
        cutoff = min(self.deadline, started + 30, started + max(0, 60 - store.spent_seconds))
        if cutoff <= started:
            return
        dispatch = dict(
            id="transition:" + str(len(self.report["records"])),
            implementation_identity=self.capability.operation.implementation_identity,
            scope=self.capability.operation.public_scope,
            coverage=self.capability.operation.public_coverage,
            source_revision=digest(selected),
            trigger_execution_id=event["id"],
        )
        try:
            with self.controller_public_dispatch(dispatch):
                reply, result = self.capability.run(cutoff)
        except Exception as error:
            reply = dict(probe_executed=False, error=type(error).__name__ + ": " + str(error)[:300])
            result = dict(
                status="env_error", reason="public_probe_dispatch_error", detail=reply["error"]
            )
        after = self._snapshot()
        if any(before[p] != after[p] for p in paths):
            result = dict(status="unknown", reason="sources_changed_during_probe")
        if reply.get("probe_executed"):
            self.record_controller_return(dispatch, before=before, after=after, report=result)
        record = dict(
            trigger_execution_id=event["id"],
            source_versions=selected,
            reply=reply,
            result=result,
            wall_s=time.monotonic() - started,
            attempts=store.attempts,
            total_probe_wall_s=store.spent_seconds,
            origin="actual_caller_public_operation;not an SDK assistant action",
        )
        self.report["records"].append(record)
        self.latest = record
        self.report["status"] = "actual_public_feedback_available"
        self._save_probe()

    def project(self, messages):
        original = super().project(messages)
        if self.latest is None:
            return original
        current = self._snapshot()
        old = self.latest["source_versions"]
        currency = (
            "current_on_declared_paths" if all(current[p] == v for p, v in old.items()) else "stale"
        )
        result = self.latest["result"]
        reply = self.latest["reply"]
        contract = self.capability.contract.manifest()
        summary = dict(
            status=result.get("status", "unknown"),
            source_currency=currency,
            report_id=reply.get("report_id"),
            candidate=contract["candidate_entry_binding"],
            reference=contract["public_reference_identity"]["id"],
            candidate_source_versions=old,
            reference_artifact_sha256=contract["public_reference_identity"]["artifact_sha256"],
            contract_sha256=self.capability.contract.identity,
            actual_calls=result.get("actual_calls"),
            step=result.get("step"),
            case_id=result.get("case_id"),
            input_summary=_brief(result.get("input")),
            initial_state_summary=_brief(result.get("initial_state")),
            difference=result.get("difference"),
            exception=result.get("exception"),
            reason=result.get("reason"),
            detail=_brief(result.get("detail")),
            coverage=contract["coverage"],
            authority="Actual finite public diagnostic;not final acceptance",
            read="python -B /tmp/ic-transition-probe.py read "
            + reply["report_id"]
            + " --offset 0 --limit 2000"
            if reply.get("report_id")
            else None,
        )
        variants = [
            summary,
            {k: v for k, v in summary.items() if k not in ("coverage", "initial_state_summary")},
            {
                k: v
                for k, v in summary.items()
                if k
                not in (
                    "candidate",
                    "reference",
                    "coverage",
                    "contract_sha256",
                    "candidate_source_versions",
                    "reference_artifact_sha256",
                    "initial_state_summary",
                )
            },
        ]
        base = self.prepare(original)
        base_tokens = self._count(base)
        chosen = None
        record = None
        for value in variants:
            projected = base + [
                dict(
                    role="user",
                    content="CONTROLLER PUBLIC TRANSITION OBSERVATION\n" + encoded(value),
                )
            ]
            tokens = self._count(projected)
            if tokens - base_tokens <= 600 and tokens <= self.limit:
                chosen = projected
                record = dict(
                    status="prepared;actual HTTP proves delivery",
                    input_tokens=tokens,
                    feedback_tokens=tokens - base_tokens,
                    report_id=reply.get("report_id"),
                    source_currency=currency,
                    model_input_sha256=hashlib.sha256(encoded(projected).encode()).hexdigest(),
                    canonical_history_changed=False,
                )
                break
        if chosen is None:
            self.report["deliveries"].append(
                dict(
                    status="omitted;original input preserved",
                    report_id=reply.get("report_id"),
                    source_currency=currency,
                )
            )
            self._save_probe()
            return original
        self.report["deliveries"].append(record)
        self._save_probe()
        with self.index.db:
            self.index.db.execute(
                "INSERT INTO native_transition_inputs(messages,accounting) VALUES(?,?)",
                (encoded(chosen), encoded(record)),
            )
        return chosen
