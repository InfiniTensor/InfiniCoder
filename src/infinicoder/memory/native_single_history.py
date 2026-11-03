"""Expose the existing per-phase history budget in one ordinary SDK run."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.memory.native_phase_history import NativePhaseHistoryContext, PhaseHistoryBinding
from infinicoder.runtime.native_plain import NativePlainObserver


class NativeSinglePhaseHistory:
    identity = "native_single_phase_history_v1"

    def __init__(self, count_messages, *, tokenizer_identity, implementation_paths=()):
        self.counter = count_messages
        self.tokenizer_identity = tokenizer_identity
        self.paths = tuple(implementation_paths)
        self.history = None
        self._used = self._active = False

    def validate(self, request):
        if any(
            value is not None
            for value in (
                request.candidate_search,
                request.repair_adapter,
                request.planning_adapter,
            )
        ):
            raise UnsupportedCore("single-run history cannot enable search,repair or planning")
        NativePlainObserver(
            self.counter,
            tokenizer_identity=self.tokenizer_identity,
            implementation_paths=self.paths,
        ).validate(request)

    def bind(self, *args, **kwargs):
        raise UnsupportedCore(
            "single-run history requires the resolved original public task binding"
        )

    @contextmanager
    def bind_for_task(self, request, model, environment, deadline, *, public_task):
        self.validate(request)
        if (
            self._used
            or request.context_adapter is not self
            or not isinstance(public_task, str)
            or not public_task.strip()
        ):
            raise UnsupportedCore("fresh single-run context and resolved public task required")
        self._used = True
        state = Path(request.state_dir).resolve()
        phase = state / "plain"
        phase.mkdir(exist_ok=False)
        descriptor = PhaseHistoryBinding(
            state,
            phase,
            "plain",
            Path(request.workspace).resolve(),
            hashlib.sha256(request.spec.task.encode()).hexdigest(),
            hashlib.sha256(public_task.encode()).hexdigest(),
            self.paths,
            deadline,
            self.counter,
            self.tokenizer_identity,
        )
        self.history = NativePhaseHistoryContext(descriptor)
        report = dict(
            identity=self.identity,
            phase=descriptor.manifest(),
            original_trajectory_ref=str(state / "native-trajectory.json"),
            original_public_task_ref=str(state / "native-public-task.json"),
            history_report_ref=str(phase / "native-phase-history.json"),
            extra_model_calls=0,
            max_original_SDK_runs=1,
            search_enabled=False,
            source_enabled=False,
            planning_enabled=False,
            capability_improvement_proven=False,
        )
        try:
            with self.history.bind_phase(request, model, environment):
                self._active = True
                try:
                    yield self
                finally:
                    self._active = False
        finally:
            (state / "native-single-history.json").write_bytes(
                (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode()
            )

    @contextmanager
    def bind_public_operation(self, operation):
        if not self._active:
            raise UnsupportedCore("single-run public observation requires an active original phase")
        with self.history.bind_public_operation(operation):
            yield self

    def _observer(self):
        if not self._active:
            raise UnsupportedCore(
                "trusted evaluator callback requires the active single-run observer"
            )
        return self.history

    def begin_check(self, **kwargs):
        return self._observer().begin_check(**kwargs)

    def end_check(self, token, **kwargs):
        return self._observer().end_check(token, **kwargs)

    def observer_failed(self, error):
        return self._observer().observer_failed(error)
