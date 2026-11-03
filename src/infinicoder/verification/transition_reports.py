"""The existing immutable run/read store with finite revision probe budgets."""

from contextlib import contextmanager
import copy
import math
from pathlib import Path
import threading
import time

from infinicoder.runtime.agent_execution import CallerPublicOperation, UnsupportedCore
from infinicoder.verification.public_check_reports import PublicCheckReports
from infinicoder.verification.public_transition_probe import PublicTransitionContract, digest


class TransitionReports(PublicCheckReports):
    def __init__(self, operation, directory, request, deadline, contract):
        super().__init__(operation, directory, request, deadline)
        self.transition_contract = contract
        self.attempts = 0
        self.spent_seconds = 0.0
        self.revisions = {}
        self.probe_gate = threading.Lock()

    def run(self, *, absolute_deadline=None):
        self._guard()
        if absolute_deadline is not None and (
            type(absolute_deadline) not in (int, float) or not math.isfinite(absolute_deadline)
        ):
            raise ValueError("finite absolute probe deadline required")
        entered = time.monotonic()
        cutoff = min(
            self.deadline,
            entered + 30,
            entered + max(0, 60 - self.spent_seconds),
            self.deadline if absolute_deadline is None else absolute_deadline,
        )
        if not self.probe_gate.acquire(timeout=max(0, cutoff - time.monotonic())):
            raise TimeoutError("transition probe busy")
        try:
            versions = self._versions(cutoff)
            if any(r["kind"] != "file" for r in versions.values()):
                return dict(
                    error="source_binding_unknown", source_currency="unknown", probe_executed=False
                )
            revision = digest(versions)
            cached = self.revisions.get(revision)
            if cached is not None:
                return dict(copy.deepcopy(cached), reused_revision=True, probe_executed=False)
            if self.attempts >= 2 or self.spent_seconds >= 60:
                return dict(
                    error="transition_probe_budget_exhausted",
                    source_currency="unknown",
                    probe_executed=False,
                )
            cutoff = min(cutoff, time.monotonic() + 60 - self.spent_seconds)
            self.attempts += 1
            try:
                result = super().run(absolute_deadline=cutoff)
                result.update(probe_executed=True, reused_revision=False, source_revision=revision)
            except Exception as error:
                result = dict(
                    error=type(error).__name__ + ": " + str(error)[:500],
                    source_currency="unknown",
                    probe_executed=False,
                    reused_revision=False,
                    source_revision=revision,
                )
            finally:
                self.spent_seconds += time.monotonic() - entered
            self.revisions[revision] = copy.deepcopy(result)
            return result
        finally:
            self.probe_gate.release()


class PublicTransitionCapability:
    """Caller supplies the public operation;the product supplies bounded reuse.

    bind_reports() is opened by the caller's public acceptance context before
    the original agent runs. ON/OFF share this same store and manual run/read
    endpoint;only NativeTransitionProbe activates automatic edit feedback.
    """

    def __init__(self, contract, operation):
        if not isinstance(contract, PublicTransitionContract) or not isinstance(
            operation, CallerPublicOperation
        ):
            raise ValueError("explicit public transition contract and caller operation required")
        data = contract.manifest()
        if (
            operation.source_mutation_allowed
            or not operation.observed_paths
            or tuple(operation.observed_paths) != tuple(data["observed_paths"])
            or operation.timeout_seconds > 30
            or operation.result_token_bound > 600
            or operation.implementation_identity.get("transition_contract_sha256")
            != contract.identity
            or data["execution_operation"].get("id") != operation.operation_id
        ):
            raise ValueError(
                "probe operation must preserve sources and match the bounded public contract"
            )
        self.contract, self.operation = contract, operation
        self.store = None

    def validate(self, request):
        self.contract.manifest()
        if Path(self.operation.workspace).resolve() != Path(request.workspace).resolve():
            raise UnsupportedCore("public transition operation belongs to another workspace")

    @contextmanager
    def bind_reports(self, request, deadline, directory):
        self.validate(request)
        if self.store is not None:
            raise ValueError("transition reports cannot nest")
        self.store = TransitionReports(self.operation, directory, request, deadline, self.contract)
        try:
            yield self.store
        finally:
            self.store.active = False
            from infinicoder.verification.public_transition_probe import encoded

            (self.store.directory / "probe-lifecycle.json").write_bytes(
                (
                    encoded(
                        dict(
                            attempts=self.store.attempts,
                            spent_seconds=self.store.spent_seconds,
                            revisions=self.store.revisions,
                            closed=True,
                            independent_final_invoked=False,
                        )
                    )
                    + "\n"
                ).encode()
            )
            self.store = None

    def run(self, cutoff):
        if self.store is None:
            raise UnsupportedCore("caller public transition reports are not bound")
        reply = self.store.run(absolute_deadline=cutoff)
        if not reply.get("report_id"):
            return reply, dict(
                status="unknown", reason=reply.get("error", "no actual public report")
            )
        record, _ = self.store._load(reply["report_id"])
        if record.get("error"):
            return reply, dict(
                status="env_error",
                reason="caller_diagnostic_operation_failed",
                detail=record["error"],
            )
        current = self.store._versions(cutoff)
        if self.store._currency(record, current) != "current_on_declared_paths":
            return reply, dict(status="unknown", reason="candidate_source_changed_or_unknown")
        raw = record.get("result")
        process = raw.get("process", {}) if isinstance(raw, dict) else {}
        if process.get("timed_out") or process.get("truncated"):
            return reply, dict(status="env_error", reason="diagnostic_runner_timeout_or_truncation")
        report = raw.get("transition_report") if isinstance(raw, dict) else None
        if not isinstance(report, dict) or report.get("contract_sha256") != self.contract.identity:
            return reply, dict(
                status="unknown", reason="actual_transition_report_missing_or_wrong_contract"
            )
        if report.get("source_versions") != record["source_before"]:
            return reply, dict(
                status="unknown", reason="runner_candidate_binding_differs_from_observed_revision"
            )
        if report.get("status") not in (
            "unknown",
            "env_error",
            "candidate_error",
            "counterexample",
            "observed_values_match",
        ):
            return reply, dict(status="unknown", reason="unsupported_diagnostic_status")
        return reply, copy.deepcopy(report)
