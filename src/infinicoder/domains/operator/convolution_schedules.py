"""Fixed launch variants of the unchanged convolution lowerer.

No reference execution,task-ID routing,arithmetic rewriting or timing here.
These are statically constrained proposals,not compiled/correct/fast kernels.
"""

import ast
from dataclasses import asdict, dataclass
from pathlib import Path

from infinicoder.verification.candidate_public_check import contract_hash
from infinicoder.domains.operator.convolution_provider import assess, digest, identity


@dataclass(frozen=True)
class ConvolutionSchedule:
    name: str
    bm: int
    bn: int
    bk: int
    warps: int
    depthwise_block: int

    @property
    def abstract_tile_elements(self):
        return self.bm * self.bn + (self.bm + self.bn) * self.bk

    def __post_init__(self):
        for n in (self.bm, self.bn, self.bk, self.warps, self.depthwise_block):
            if type(n) is not int or n <= 0 or n & (n - 1):
                raise ValueError("power-of-two static schedule required")
        if (
            self.bm not in (16, 32)
            or self.bn not in (32, 64)
            or self.bk != 32
            or self.warps != 4
            or self.depthwise_block not in (128, 256, 512)
            or self.abstract_tile_elements > 4096
        ):
            raise ValueError("schedule outside the fixed static resource envelope")


SCHEDULES = (
    ConvolutionSchedule("baseline", 16, 32, 32, 4, 256),
    ConvolutionSchedule("channel_tile", 16, 64, 32, 4, 512),
    ConvolutionSchedule("spatial_tile", 32, 32, 32, 4, 128),
)


def producer_identity():
    return dict(
        identity(),
        **{"domains/operator/convolution_schedules.py": digest(Path(__file__).read_bytes())},
    )


def _launches(schedule):
    s = schedule
    return (
        f"if cg==1:_ic_depthwise[(triton.cdiv(result.numel(),{s.depthwise_block}),)](*common,og,*metadata,BLOCK={s.depthwise_block},num_warps={s.warps})",
        f"else:_ic_conv[(triton.cdiv(x.shape[0]*math.prod(out),{s.bm}),triton.cdiv(og,{s.bn}),module.groups)](*common,cg,og,*metadata,BM={s.bm},BN={s.bn},BK={s.bk},num_warps={s.warps})",
    )


def _render(baseline, schedule):
    if schedule not in SCHEDULES:
        raise ValueError("only the three frozen schedules are available")
    original = _launches(SCHEDULES[0])
    replacement = _launches(schedule)
    if any(baseline.count(line) != 1 for line in original):
        raise ValueError("original lowerer launch sites changed;do not guess a new template")
    source = baseline
    for old, new in zip(original, replacement):
        source = source.replace(old, new)
    ast.parse(source)
    return source


@dataclass(frozen=True)
class ConvolutionScheduleVariant:
    schedule: ConvolutionSchedule
    source: str

    @property
    def sha256(self):
        return digest(self.source)


@dataclass(frozen=True)
class ConvolutionScheduleBundle:
    task_sha256: str
    contract_sha256: str
    reference_sha256: str
    producer_pins: tuple
    variants: tuple

    def current_for(self, spec, contract):
        return (
            digest(spec.task) == self.task_sha256
            and contract_hash(asdict(contract)) == self.contract_sha256
            and contract.public_reference_sha256 == self.reference_sha256
            and tuple(sorted(producer_identity().items())) == self.producer_pins
        )


def propose_convolution_schedules(spec, workspace, contract):
    """Reuse the existing strict lease;never widen its language or write scope."""
    admission = assess(spec, workspace, contract)
    if admission["status"] != "Applicable":
        return None
    baseline = admission["candidate_source"]
    variants = tuple(
        ConvolutionScheduleVariant(schedule, _render(baseline, schedule)) for schedule in SCHEDULES
    )
    if variants[0].source != baseline or len({v.sha256 for v in variants}) != 3:
        raise ValueError("baseline plus exactly two genuinely different source versions required")
    return ConvolutionScheduleBundle(
        digest(spec.task),
        contract_hash(asdict(contract)),
        contract.public_reference_sha256,
        tuple(sorted(producer_identity().items())),
        variants,
    )
