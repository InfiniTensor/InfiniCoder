"""One native coding loop:existing producer,public measurement,best restore."""

from contextlib import contextmanager
import copy
from dataclasses import replace
from pathlib import Path

from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.memory.source.evidence_source import ReadOnlyTaskSources
from infinicoder.domains.operator.native_implementation_provider import NativeImplementationProvider
from infinicoder.optimization.native_measured_optimization import (
    PublicPerformanceController,
    PublicPerformancePolicy,
    PublicPerformanceSample,
)
from infinicoder.memory.native_memory import _json


class NativeOptimizingImplementationProvider(NativeImplementationProvider):
    """A caller decoder converts the original external report,never model text.

    The first measurement reuses the provider's exact public execution. Later
    genuinely changed versions invoke that same original operation. There is
    one SDK/model/context owner;the measurement controller has no model calls.
    """

    identity = "native_producer_public_measurement_retention_v1"

    def __init__(self, *args, performance_policy, performance_decoder, decoder_identity, **kwargs):
        super().__init__(*args, **kwargs)
        if (
            not isinstance(performance_policy, PublicPerformancePolicy)
            or performance_policy.operation is not self.public_check.operation
        ):
            raise ValueError("performance must use this exact original public operation")
        if self.contract.target not in performance_policy.rollback_paths:
            raise ValueError("explicit rollback permission for the produced target required")
        if (
            not callable(performance_decoder)
            or not isinstance(decoder_identity, str)
            or not decoder_identity.strip()
        ):
            raise ValueError("identified caller-owned external performance decoder required")
        if not set(performance_policy.operation.observed_paths).issubset(self.paths):
            raise ValueError(
                "every public measurement dependency must be observed by the native owner"
            )
        self.original_performance_policy = performance_policy
        self.performance_decoder, self.decoder_identity = performance_decoder, decoder_identity
        operation = performance_policy.operation
        self.sources = ReadOnlyTaskSources(Path(operation.workspace), operation.observed_paths)
        # This is explicitly a decoding/reuse capability. Its records never
        # pretend the cached first report is a second underlying GPU execution.
        decoded = replace(
            operation,
            operation_id=operation.operation_id + ".decoded-performance",
            implementation_identity=dict(
                original=operation.implementation_identity,
                decoder=decoder_identity,
                execution="reuse_exact_provider_report_once;otherwise_original_bound_public_operation",
            ),
            executor=self._measure,
        )
        self.performance = PublicPerformanceController(
            replace(performance_policy, operation=decoded), self.sources
        )

    def _namespace(self):
        self._workspace()
        return self.binding

    def validate(self, request):
        super().validate(request)
        self.sources.validate(request.workspace, self.sources.paths)

    def _pins(self):
        return {p: self.sources.read(p)[2] for p in self.sources.paths}

    @contextmanager
    def bind(self, request, model, environment, deadline):
        self.reused_provider_measurements = 0
        self.additional_public_executions = 0
        self.performance_active = False
        self.provider_dependency_pins = None
        previous_remaining = self.sources.remaining
        self.sources.remaining = self._remaining
        try:
            with super().bind(request, model, environment, deadline):
                with self.performance.bind(request, self, deadline):
                    try:
                        yield self
                    finally:
                        (self.state / "native-producer-performance.json").write_text(
                            _json(
                                dict(
                                    decoder_identity=self.decoder_identity,
                                    original_operation=self.public_check.operation.manifest(),
                                    reused_provider_measurements=self.reused_provider_measurements,
                                    additional_public_executions=self.additional_public_executions,
                                    original_provider_executions=self.report.get(
                                        "public_executions", 0
                                    )
                                    if self.report
                                    else 0,
                                    extra_model_calls=0,
                                    private_final_used=False,
                                    authority="Original public workload only;independent final remains separate",
                                )
                            ),
                            encoding="utf8",
                        )
        finally:
            self.sources.remaining = previous_remaining

    def _measure(self, deadline):
        operation = self.public_operation
        if operation is None or operation.operation is not self.public_check.operation:
            raise UnsupportedCore("original public operation is not bound")
        current = self._pins()
        if not self.reused_provider_measurements:
            report = self.report
            if (
                report is None
                or report.get("status") != "public_checked"
                or report.get("tested_sha256") != current[self.contract.target]
                or any(current[p] != pin for p, pin in self.provider_dependency_pins.items())
            ):
                raise ValueError(
                    "initial external measurement no longer binds the produced source/dependencies"
                )
            raw = report["public_result"]
            self.reused_provider_measurements += 1
            self._save_event(
                "provider_public_measurement_reused",
                dict(
                    source_hashes=current,
                    decoder_identity=self.decoder_identity,
                    original_operation_id=operation.operation.operation_id,
                    additional_underlying_executions=0,
                    original_wall_seconds=report["public_wall_seconds"],
                ),
            )
        else:
            self.additional_public_executions += 1
            raw = operation.invoke(deadline)
        sample = self.performance_decoder(copy.deepcopy(raw))
        if (
            isinstance(sample, PublicPerformanceSample)
            and sample.correct
            and self.public_check.decoder(raw).status != "public_pass"
        ):
            raise ValueError("performance decoder contradicts original public correctness failure")
        return sample

    def _view(self, current, status, freshness, evidence):
        if status in ("public_pass", "public_fail"):
            try:
                pins = self._pins()
                changed = self.provider_dependency_pins is None or any(
                    pins[p] != h for p, h in self.provider_dependency_pins.items()
                )
            except (ValueError, OSError):
                changed = True
            if changed:
                status, freshness = "unknown", "stale_or_unverified"
                evidence = (
                    "Observed dependency changed;the earlier public report does not verify this source set. "
                    + evidence
                )
        return super()._view(current, status, freshness, evidence)

    def before_query(self, messages):
        first = not self.attempted
        if first:
            try:
                self._namespace()
                self.provider_dependency_pins = {
                    p: h for p, h in self._pins().items() if p != self.contract.target
                }
            except (OSError, ValueError, UnsupportedCore):
                self.provider_dependency_pins = None
        super().before_query(messages)
        if first:
            self.performance_active = bool(
                self.report
                and self.report.get("status") == "public_checked"
                and self.provider_dependency_pins is not None
            )
        if self.performance_active:
            self.performance.before_query()

    def execution_completed(self, event):
        super().execution_completed(event)
        if set(event.get("changed", ())).intersection(self.sources.paths):
            self.performance.source_changed()

    def project(self, messages):
        packet = super().project(messages)
        if not self.performance_active:
            return packet
        view = self.performance.view()
        view.update(
            reused_provider_measurements=self.reused_provider_measurements,
            additional_public_executions=self.additional_public_executions,
            first_measurement_cost="Already charged to original producer public execution;not retimed",
        )
        extra = dict(role="user", content=_json(view))
        return [*packet, extra] if self._count([*packet, extra]) <= self.limit else packet


def optimize_provider_request(
    request,
    *,
    performance_decoder,
    decoder_identity,
    exclusive_workspace,
    rollback_paths=None,
    max_candidates=2,
    final_reserve_seconds=30,
    min_speedup=1.02,
):
    """Attach retention to an unused provider request without changing its scope.

    Works with create_cpu_request as well as the original isolated GPU caller.
    A callable external decoder is required;it cannot be supplied by model text.
    """
    from infinicoder.runtime.agent_execution import CoreRequest

    if (
        not isinstance(request, CoreRequest)
        or type(request.context_adapter) is not NativeImplementationProvider
    ):
        raise ValueError("an existing ordinary NativeImplementationProvider request required")
    old = request.context_adapter
    if old.bound_once:
        raise ValueError("cannot replace a running or previously bound context")
    policy = PublicPerformancePolicy(
        old.public_check.operation,
        tuple(rollback_paths) if rollback_paths is not None else (old.contract.target,),
        exclusive_workspace,
        max_candidates,
        final_reserve_seconds,
        min_speedup,
    )
    context = NativeOptimizingImplementationProvider(
        old.count_messages,
        tokenizer_identity=old.tokenizer_identity,
        implementation_paths=old.paths,
        contract=old.contract,
        public_check=old.public_check,
        performance_policy=policy,
        performance_decoder=performance_decoder,
        decoder_identity=decoder_identity,
    )
    return replace(request, context_adapter=context)
