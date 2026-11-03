"""Two fusion schedules selected by the existing external measurement loop."""

from dataclasses import asdict, dataclass
from pathlib import Path

from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.verification.candidate_public_check import contract_hash
from infinicoder.domains.operator.convolution_provider import digest
from infinicoder.domains.operator.convolution_schedule_search import (
    ConvolutionScheduleSearch,
    ConvolutionScheduleSelection,
)
from infinicoder.domains.operator.linear_gelu import generate, SCHEDULES
from infinicoder.domains.operator.fusion_provider import FusionProviderContract, identity
from infinicoder.optimization.native_measured_optimization import PublicPerformancePolicy


def producer_identity():
    return dict(
        identity(),
        **{"domains/operator/fusion_schedule_search.py": digest(Path(__file__).read_bytes())},
    )


@dataclass(frozen=True)
class FusionSchedule:
    name: str


@dataclass(frozen=True)
class FusionVariant:
    schedule: FusionSchedule
    source: str

    @property
    def sha256(self):
        return digest(self.source)


@dataclass(frozen=True)
class FusionScheduleBundle:
    task_sha256: str
    contract_sha256: str
    producer_pins: tuple
    variants: tuple

    def current_for(self, spec, contract):
        return (
            type(contract) is FusionProviderContract
            and digest(spec.task) == self.task_sha256
            and contract_hash(asdict(contract)) == self.contract_sha256
            and tuple(sorted(producer_identity().items())) == self.producer_pins
        )


class FusionScheduleSearch(ConvolutionScheduleSearch):
    state_filename = "native-fusion-schedules.json"
    selection_type = ConvolutionScheduleSelection

    def _producer_identity(self):
        return producer_identity()

    def _propose(self, owner):
        # The original provider already bound the reference to public source.
        # Its first independently checked artifact must still be the live file.
        variants = tuple(
            FusionVariant(
                FusionSchedule(name),
                generate(owner.contract.reference_source, schedule=name).source,
            )
            for name in SCHEDULES
        )
        if variants[0].sha256 != owner.report.get("generated_sha256"):
            return None
        return FusionScheduleBundle(
            digest(owner.request.spec.task),
            contract_hash(asdict(owner.contract)),
            tuple(sorted(producer_identity().items())),
            variants,
        )


def with_fusion_schedule_search(
    request, *, performance_decoder, decoder_identity, exclusive_workspace
):
    from infinicoder.domains.operator.native_implementation_provider import (
        NativeImplementationProvider,
    )

    owner = request.context_adapter
    if (
        type(owner) is not NativeImplementationProvider
        or type(owner.contract) is not FusionProviderContract
        or owner.bound_once
        or owner.schedule_search is not None
        or not getattr(owner.handoff_policy, "enabled", False)
    ):
        raise UnsupportedCore(
            "fusion schedules require an unused fusion expert with explicit handoff"
        )
    target = owner.contract.target
    if owner.paths != (target,) or owner.public_check.operation.observed_paths != (target,):
        raise UnsupportedCore("fusion selection requires one original observed target")
    policy = PublicPerformancePolicy(
        owner.public_check.operation,
        (target,),
        exclusive_workspace,
        1,
        max(60, owner.contract.final_reserve_seconds),
        1.02,
    )
    owner.schedule_search = FusionScheduleSearch(
        owner.public_check, policy, performance_decoder, decoder_identity
    )
    return request
