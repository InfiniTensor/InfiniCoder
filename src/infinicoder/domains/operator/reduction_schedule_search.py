"""Exactly two sum layouts using the existing public best-correct executor.

The shared executor has no convolution arithmetic. This explicit producer owns
its own family contract, identity, bundle, selection type and audit filename.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.verification.candidate_public_check import contract_hash
from infinicoder.domains.operator.convolution_provider import digest
from infinicoder.domains.operator.convolution_schedule_search import (
    ConvolutionScheduleSearch,
    ConvolutionScheduleSelection,
)
from infinicoder.domains.operator.reduction import generate
from infinicoder.optimization.native_measured_optimization import PublicPerformancePolicy
from infinicoder.domains.operator.reduction_provider import (
    ReductionProviderContract,
    assess,
    identity,
    _source_and_lowerer_task,
)


@dataclass(frozen=True)
class ReductionSchedule:
    name: str

    def __post_init__(self):
        if self.name not in ("scalar", "coalesced"):
            raise ValueError("only the fixed scalar and coalesced sum schedules are supported")


@dataclass(frozen=True)
class ReductionScheduleVariant:
    schedule: ReductionSchedule
    source: str

    @property
    def sha256(self):
        return digest(self.source)


def producer_identity():
    return dict(
        identity(),
        **{"domains/operator/reduction_schedule_search.py": digest(Path(__file__).read_bytes())},
    )


@dataclass(frozen=True)
class ReductionScheduleBundle:
    task_sha256: str
    contract_sha256: str
    reference_sha256: str
    producer_pins: tuple
    variants: tuple

    def current_for(self, spec, contract):
        return (
            type(contract) is ReductionProviderContract
            and digest(spec.task) == self.task_sha256
            and contract_hash(asdict(contract)) == self.contract_sha256
            and contract.public_reference_sha256 == self.reference_sha256
            and tuple(sorted(producer_identity().items())) == self.producer_pins
        )


def propose_reduction_schedules(spec, workspace, contract):
    admission = assess(spec, workspace, contract)
    if admission["status"] != "Applicable":
        return None
    source, _ = _source_and_lowerer_task(spec, contract)
    variants = tuple(
        ReductionScheduleVariant(ReductionSchedule(name), generate(source, schedule=name).source)
        for name in ("scalar", "coalesced")
    )
    if (
        variants[0].source != admission["candidate_source"]
        or len({v.sha256 for v in variants}) != 2
    ):
        raise ValueError("baseline plus exactly one distinct sum schedule required")
    return ReductionScheduleBundle(
        digest(spec.task),
        contract_hash(asdict(contract)),
        contract.public_reference_sha256,
        tuple(sorted(producer_identity().items())),
        variants,
    )


@dataclass(frozen=True)
class ReductionScheduleSelection(ConvolutionScheduleSelection):
    """Source-bound selection fields shared with the existing executor."""


class ReductionScheduleSearch(ConvolutionScheduleSearch):
    state_filename = "native-reduction-schedules.json"
    selection_type = ReductionScheduleSelection

    def _producer_identity(self):
        return producer_identity()

    def _propose(self, owner):
        return propose_reduction_schedules(owner.request.spec, owner._workspace(), owner.contract)


def with_reduction_schedule_search(
    request, *, performance_decoder, decoder_identity, exclusive_workspace
):
    from infinicoder.domains.operator.native_implementation_provider import (
        NativeImplementationProvider,
    )

    owner = request.context_adapter
    if (
        type(owner) is not NativeImplementationProvider
        or type(owner.contract) is not ReductionProviderContract
        or owner.bound_once
        or getattr(owner, "schedule_search", None) is not None
        or not getattr(owner.handoff_policy, "enabled", False)
    ):
        raise UnsupportedCore(
            "sum schedules require one unused sum provider with explicit handoff policy"
        )
    target = owner.contract.target
    if owner.paths != (target,) or owner.public_check.operation.observed_paths != (target,):
        raise UnsupportedCore("sum selection requires a single observed target")
    policy = PublicPerformancePolicy(
        owner.public_check.operation,
        (target,),
        exclusive_workspace,
        1,
        max(60, owner.contract.final_reserve_seconds),
        1.02,
    )
    owner.schedule_search = ReductionScheduleSearch(
        owner.public_check, policy, performance_decoder, decoder_identity
    )
    return request
