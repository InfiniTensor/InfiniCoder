"""Explicit execution-core selection; the existing Runtime remains the default.

Native adapters are supplied by the caller, not inferred from model output.
They own sandbox setup, physical-request accounting and independent acceptance.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from contextlib import contextmanager
import hashlib
import json
import math
import threading
import time

from infinicoder.models.budget import ModelProfile
from infinicoder.planning.task_contract import TaskSpec, validate_task_spec

NATIVE_CORE = "native_infinilm_v1"


class UnsupportedCore(ValueError):
    """The selected core cannot preserve the requested execution contract."""


@dataclass(frozen=True)
class CoreBudget:
    max_model_calls: int = 15
    wall_seconds: float = 300
    request_timeout: float = 600

    def __post_init__(self):
        if type(self.max_model_calls) is not int or self.max_model_calls < 1:
            raise ValueError("max_model_calls must be a positive integer")
        for name in ("wall_seconds", "request_timeout"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")


class AuxiliaryCallBudget:
    """One attached query shares the current relay and reduces SDK step credit.

    Raw SDK completion is captured before context hooks. It bypasses the SDK's
    action parser/retry loop;the original relay remains the cost/budget authority.
    """

    def __init__(
        self, request, agent, query, deadline, *, raw_query=None, prepare=None, public_task=None
    ):
        self.request, self.agent, self.query, self.deadline = request, agent, query, deadline
        if public_task is not None and (
            not isinstance(public_task, str) or not public_task.strip()
        ):
            raise ValueError("original public task must be nonempty text")
        self.public_task = public_task
        self.raw_query, self.prepare = raw_query, prepare
        if (raw_query is None) != (prepare is None):
            raise ValueError("raw completion and preparation must be supplied together")
        self.call_units = 0
        self.original_steps = agent.config.step_limit

    def query_once(self, messages, *, output_tokens, absolute_deadline):
        if self.call_units:
            raise UnsupportedCore("auxiliary query cannot repeat")
        transport = self.request.transport_adapter
        if transport.remaining_physical_calls() < 3:
            raise ValueError("review requires three remaining physical call slots")
        cutoff = min(self.deadline, absolute_deadline)
        if cutoff <= time.monotonic():
            raise TimeoutError("auxiliary deadline expired")
        self.call_units = 1
        self.agent.config.step_limit = max(0, self.original_steps - self.call_units)
        with transport.limit_request(
            output_tokens=output_tokens, absolute_deadline=cutoff, max_physical_calls=1
        ):
            kwargs = dict(
                max_tokens=output_tokens,
                timeout=max(0.001, cutoff - time.monotonic()),
                num_retries=0,
            )
            if self.raw_query is None:
                return self.query(messages, **kwargs)
            response = self.raw_query(self.prepare(messages), **kwargs)
            raw = response.model_dump(mode="json")
            message = raw["choices"][0]["message"]
            return dict(
                role="assistant", content=message.get("content", ""), extra=dict(response=raw)
            )


@dataclass(frozen=True)
class FinalAcceptance:
    passed: bool
    evidence_ref: str

    def __post_init__(self):
        if (
            type(self.passed) is not bool
            or not isinstance(self.evidence_ref, str)
            or not self.evidence_ref
        ):
            raise ValueError("independent acceptance needs a bool and an evidence reference")


@dataclass(frozen=True)
class ModelBinding:
    """Yielded by a budgeted transport context; usage may finish after SDK exit."""

    model_name: str
    model_kwargs: dict
    usage_ledger_ref: str


@dataclass(frozen=True)
class NativeTaskBinding:
    """Caller-owned verbatim public task, with no fabricated public checker.

    The task must equal TaskSpec.task. Its caller states the real environment
    path and owns independent final verification; this object proves no result.
    """

    task: str
    identity: str

    def __post_init__(self):
        if (
            not isinstance(self.task, str)
            or not self.task.strip()
            or not isinstance(self.identity, str)
            or not self.identity.strip()
        ):
            raise ValueError("verbatim public task and caller binding identity required")


@dataclass(frozen=True)
class CallerPublicOperation:
    """An explicitly public caller capability, separate from final acceptance.

    The caller owns the fixed executor, its shared HTTP/controller lock and its
    resource isolation. Metadata describes that contract; it does not prove it.
    No command, URL or permission is inferred from repository/model text.
    """

    operation_id: str
    workspace: Path
    implementation_identity: dict
    public_scope: str
    public_coverage: Any
    timeout_seconds: float
    environment_identity: dict
    observed_paths: tuple
    side_effects: tuple
    result_token_bound: int
    executor: Any
    source_mutation_allowed: bool = False

    def __post_init__(self):
        for name in ("operation_id", "public_scope"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError("explicit public operation identity and scope required")
        if (
            not self.implementation_identity
            or not self.environment_identity
            or not self.public_coverage
        ):
            raise ValueError("public implementation, environment and coverage metadata required")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("finite positive public operation timeout required")
        if type(self.result_token_bound) is not int or not 1 <= self.result_token_bound <= 8192:
            raise ValueError("bounded public result size required")
        if not callable(self.executor):
            raise ValueError("caller must supply a fixed bounded public executor")
        if not isinstance(self.observed_paths, tuple) or len(set(self.observed_paths)) != len(
            self.observed_paths
        ):
            raise ValueError("distinct caller-observed paths required")
        from pathlib import PurePosixPath

        for value in self.observed_paths:
            if (
                not isinstance(value, str)
                or not value
                or "\\" in value
                or ":" in value
                or "\0" in value
                or PurePosixPath(value).is_absolute()
                or ".." in PurePosixPath(value).parts
                or str(PurePosixPath(value)) != value
            ):
                raise ValueError("normalized workspace-relative observation paths required")
        if not isinstance(self.side_effects, tuple) or any(
            not isinstance(x, str) for x in self.side_effects
        ):
            raise ValueError("caller must explicitly declare public-operation side effects")
        if type(self.source_mutation_allowed) is not bool:
            raise ValueError("explicit source mutation policy required")
        # Reject non-data metadata now, before a caller opens its task.
        json.dumps(self.manifest(), allow_nan=False)

    def manifest(self):
        return dict(
            operation_id=self.operation_id,
            workspace=str(Path(self.workspace).resolve()),
            implementation_identity=self.implementation_identity,
            public_scope=self.public_scope,
            public_coverage=self.public_coverage,
            timeout_seconds=self.timeout_seconds,
            environment_identity=self.environment_identity,
            observed_paths=self.observed_paths,
            side_effects=self.side_effects,
            result_token_bound=self.result_token_bound,
            source_mutation_allowed=self.source_mutation_allowed,
            authority="caller-declared public operation;not independent final acceptance",
        )

    @contextmanager
    def bind(self, request, deadline):
        if Path(self.workspace).resolve() != Path(request.workspace).resolve():
            raise UnsupportedCore("public operation belongs to a different live workspace")
        if not math.isfinite(deadline) or deadline <= time.monotonic():
            raise TimeoutError("public operation task deadline exhausted")
        lease = _BoundPublicOperation(self, deadline)
        try:
            yield lease
        finally:
            lease.active = False


class _BoundPublicOperation:
    """Non-serializable capability, valid only in its original caller context."""

    def __init__(self, operation, deadline):
        self.operation, self.deadline = operation, deadline
        self.active, self.owner_thread = True, threading.get_ident()
        self.manifest = operation.manifest()
        self.contract_sha256 = self._identity()

    def _identity(self):
        text = json.dumps(
            self.operation.manifest(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(text.encode()).hexdigest()

    def invoke(self, absolute_deadline):
        if not self.active or threading.get_ident() != self.owner_thread:
            raise UnsupportedCore("public operation lease is expired or on another thread")
        if self._identity() != self.contract_sha256:
            raise UnsupportedCore("public operation contract changed while bound")
        if (
            isinstance(absolute_deadline, bool)
            or not isinstance(absolute_deadline, (int, float))
            or not math.isfinite(absolute_deadline)
        ):
            raise ValueError("absolute public-operation deadline required")
        now = time.monotonic()
        cutoff = min(self.deadline, absolute_deadline, now + self.operation.timeout_seconds)
        if cutoff <= now:
            raise TimeoutError("public operation deadline exhausted before dispatch")
        # The trusted executor clamps its subprocess and shared lock wait to
        # this absolute deadline. It must not call private final verification.
        return self.operation.executor(cutoff)


@dataclass(frozen=True)
class PublicTaskContext:
    """Original public task/command plus an optional caller-issued capability."""

    binding: str | NativeTaskBinding
    operation: CallerPublicOperation | None = None
    candidate_check: Any = None

    def __post_init__(self):
        if not (
            isinstance(self.binding, NativeTaskBinding)
            or isinstance(self.binding, str)
            and self.binding.strip()
        ):
            raise ValueError("original public command or verbatim task binding required")
        if self.operation is not None and not isinstance(self.operation, CallerPublicOperation):
            raise ValueError("explicit CallerPublicOperation required")
        if self.candidate_check is not None:
            from infinicoder.verification.candidate_public_check import CandidatePublicCheck

            if (
                not isinstance(self.candidate_check, CandidatePublicCheck)
                or self.candidate_check.operation is not self.operation
            ):
                raise ValueError("public selection must belong to this exact caller operation")


@dataclass(frozen=True)
class CoreRequest:
    spec: TaskSpec
    workspace: Path
    state_dir: Path
    model_profile: ModelProfile
    budget: CoreBudget
    execution_adapter: Any = None
    transport_adapter: Any = None
    acceptance_adapter: Any = None
    resume: bool = False
    pause_required: bool = False
    context_adapter: Any = None
    repair_adapter: Any = None
    planning_adapter: Any = None
    candidate_search: Any = None
    implementation_proposal: Any = None


@dataclass(frozen=True)
class CoreOutcome:
    backend: str
    exit_reason: str
    candidate_ref: str
    native_trajectory_ref: str
    usage_ledger_ref: str
    native_result: dict
    final_acceptance: FinalAcceptance | None
    agent_calls: int
    error: str | None = None
    acceptance_error: str | None = None

    @property
    def success(self):
        return self.final_acceptance is not None and self.final_acceptance.passed


def validate_native_request(request):
    validate_task_spec(request.spec)
    if request.resume or request.pause_required:
        raise UnsupportedCore("native core is one-shot; use Runtime for pause/resume")
    spec = request.spec
    if (
        spec.milestones
        or spec.amendments
        or spec.capabilities
        or spec.target_facts
        or spec.tensor_layouts
    ):
        raise UnsupportedCore(
            "native core cannot consume stages,amendments or domain fields;use Runtime"
        )
    if spec.allowed_paths:
        from infinicoder.execution.native_scoped_execution import ScopedNativeExecution
        from infinicoder.runtime.native_plain import NativePlainObserver
        from infinicoder.memory.native_evidence_route import NativeEvidenceRoute

        if type(request.execution_adapter) is not ScopedNativeExecution:
            raise UnsupportedCore(
                "scoped native tasks need explicit OS-enforced ScopedNativeExecution"
            )
        context = request.context_adapter
        if type(context) is NativeEvidenceRoute:
            context.validate_scoped(request)
        if (
            any(
                getattr(request, field) is not None
                for field in (
                    "repair_adapter",
                    "planning_adapter",
                    "candidate_search",
                    "implementation_proposal",
                )
            )
            or context is not None
            and type(context) not in (NativePlainObserver, NativeEvidenceRoute)
        ):
            raise UnsupportedCore(
                "scoped Native supports only plain or explicitly bounded post-edit public feedback"
            )
    workspace, state = Path(request.workspace).resolve(), Path(request.state_dir).resolve()
    if not workspace.is_dir() or state.is_relative_to(workspace) or workspace.is_relative_to(state):
        raise ValueError("existing workspace and separate state directory required")
    if not isinstance(request.model_profile, ModelProfile) or not isinstance(
        request.budget, CoreBudget
    ):
        raise ValueError("explicit ModelProfile and CoreBudget required")
    for adapter, methods in (
        (request.execution_adapter, ("validate", "open")),
        (request.transport_adapter, ("validate", "open")),
        (request.acceptance_adapter, ("validate", "public_command", "verify")),
    ):
        if not getattr(adapter, "identity", None) or any(
            not callable(getattr(adapter, name, None)) for name in methods
        ):
            raise UnsupportedCore(
                "caller must provide execution, budgeted transport and independent acceptance adapters"
            )
        adapter.validate(request)

    if request.implementation_proposal is not None:
        from infinicoder.runtime.implementation_proposal import ImplementationProposal

        if not isinstance(request.implementation_proposal, ImplementationProposal):
            raise UnsupportedCore("explicit implementation proposal policy required")
        request.implementation_proposal.validate(request)
    if request.candidate_search is not None:
        from infinicoder.search.native_candidate_search import CandidateSearchPolicy

        if not isinstance(request.candidate_search, CandidateSearchPolicy):
            raise UnsupportedCore("explicit candidate search policy required")
        request.candidate_search.validate(request)
    if request.repair_adapter is not None:
        from infinicoder.runtime.extensions.native_repair import NativeRepairContinuation

        if not isinstance(request.repair_adapter, NativeRepairContinuation):
            raise UnsupportedCore("explicit native failed-candidate continuation required")
        request.repair_adapter.validate(request)
    if request.planning_adapter is not None:
        from infinicoder.planning.native_source_planning import NativeSourcePlanning
        from infinicoder.planning.native_artifact_architecture import NativeArtifactArchitecture

        if not isinstance(
            request.planning_adapter, (NativeSourcePlanning, NativeArtifactArchitecture)
        ):
            raise UnsupportedCore("explicit source planning adapter required")
        request.planning_adapter.validate(request)
    if request.context_adapter is not None:
        adapter = request.context_adapter
        if not getattr(adapter, "identity", None) or any(
            not callable(getattr(adapter, name, None)) for name in ("validate", "bind")
        ):
            raise UnsupportedCore("context adapter must supply identity, validate and bind")
        adapter.validate(request)


def create_runtime_request(runtime, spec, *, resume=False, pause_required=False):
    """Describe the existing Runtime;do not replace its client or executor."""
    return CoreRequest(
        spec=spec,
        workspace=runtime.workspace,
        state_dir=runtime.state_dir,
        model_profile=runtime.assembler.profile,
        budget=CoreBudget(runtime.max_model_calls, runtime.time_limit, runtime.request_timeout),
        resume=resume,
        pause_required=pause_required,
    )


def validate_runtime_request(request, runtime):
    """Reject declarations that the caller's existing Runtime cannot preserve."""
    if not isinstance(request, CoreRequest):
        raise TypeError("CoreRequest required")
    if runtime is None:
        raise ValueError("runtime backend requires the caller existing Runtime instance")
    if not isinstance(request.spec, TaskSpec):
        raise TypeError("CoreRequest.spec must be TaskSpec")
    validate_task_spec(request.spec)
    if type(request.resume) is not bool or type(request.pause_required) is not bool:
        raise ValueError("resume and pause_required must be boolean")
    native = (
        "execution_adapter",
        "transport_adapter",
        "acceptance_adapter",
        "context_adapter",
        "repair_adapter",
        "planning_adapter",
        "candidate_search",
        "implementation_proposal",
    )
    if any(getattr(request, name) is not None for name in native):
        raise UnsupportedCore(
            "Runtime cannot silently ignore Native adapters;use the caller Runtime client,executor and verifier"
        )
    for name in ("workspace", "state_dir"):
        if Path(getattr(runtime, name)).resolve() != Path(getattr(request, name)).resolve():
            raise ValueError("Runtime " + name + " differs from request")
    if not isinstance(request.budget, CoreBudget) or not isinstance(
        request.model_profile, ModelProfile
    ):
        raise ValueError("explicit CoreBudget and ModelProfile required")
    for declared, actual in (
        ("max_model_calls", "max_model_calls"),
        ("wall_seconds", "time_limit"),
        ("request_timeout", "request_timeout"),
    ):
        if getattr(request.budget, declared) != getattr(runtime, actual):
            raise ValueError("Runtime " + actual + " differs from CoreRequest budget")
    if request.model_profile != runtime.assembler.profile:
        raise ValueError("Runtime model_profile differs from request")


def execute_agent(request, *, backend="runtime", runtime=None):
    """Choose one core once; unsupported native contracts never silently fall back."""
    if backend == "runtime":
        validate_runtime_request(request, runtime)
        return runtime.execute(request.spec, resume=request.resume)
    if backend != NATIVE_CORE:
        raise UnsupportedCore(f"unknown execution backend: {backend}")
    from infinicoder.runtime.native_run import NativeRunCore

    return NativeRunCore().run(request)
