"""Explicit, versioned public contract for a single-axis sum producer.

Source extraction shares the existing byte-lease validator. Neither reference
fact discovery nor candidate generation imports or executes the reference.
"""

from infinicoder._resources import package_root

import ast
from dataclasses import asdict, dataclass
import math
from pathlib import PurePosixPath
import re
import time

from infinicoder.domains.operator.candidate_provider import ProviderLeaseError, UnknownPrerequisite
from infinicoder.domains.operator.convolution_provider import _source_and_lowerer_task, digest
from infinicoder.domains.operator.reduction import generate


@dataclass(frozen=True)
class ReductionProviderContract:
    target: str
    public_reference_sha256: str
    scope_quote: str
    complete_module_authorized: bool
    final_reserve_seconds: float
    provider: str = "single_axis_sum_v1"
    mode: str = "enabled"
    entry: str = "ModelNew"
    backend: str = "triton"
    dtype: str = "float32"
    device: str = "cuda"
    public_reference_span: tuple | None = None

    def __post_init__(self):
        if not isinstance(self.target, str):
            raise ValueError("one normalized relative Python target required")
        p = PurePosixPath(self.target)
        if (
            not self.target.endswith(".py")
            or p.is_absolute()
            or str(p) != self.target
            or ".." in p.parts
            or any(c in self.target for c in "*?[\\:\0")
        ):
            raise ValueError("one normalized relative Python target required")
        if not isinstance(self.public_reference_sha256, str) or not re.fullmatch(
            "[0-9a-f]{64}", self.public_reference_sha256
        ):
            raise ValueError("exact public source SHA required")
        if (
            not isinstance(self.scope_quote, str)
            or self.target not in self.scope_quote
            or len(self.scope_quote) > 2048
        ):
            raise ValueError("an exact public target scope quote is required")
        if self.complete_module_authorized is not True:
            raise ValueError("caller must explicitly authorize complete-module replacement")
        if (
            isinstance(self.final_reserve_seconds, bool)
            or not isinstance(self.final_reserve_seconds, (int, float))
            or not math.isfinite(self.final_reserve_seconds)
            or self.final_reserve_seconds <= 0
        ):
            raise ValueError("caller must reserve a finite positive final-check interval")
        if self.provider != "single_axis_sum_v1" or self.mode not in ("enabled", "disabled"):
            raise ValueError("unsupported single-axis sum provider or mode")
        if (self.entry, self.backend, self.dtype, self.device) != (
            "ModelNew",
            "triton",
            "float32",
            "cuda",
        ):
            raise ValueError("only declared FP32 CUDA Triton forward modules are supported")
        span = self.public_reference_span
        if span is not None:
            if (
                not isinstance(span, (tuple, list))
                or len(span) != 2
                or any(type(v) is not int for v in span)
                or not 0 <= span[0] < span[1]
            ):
                raise ValueError("explicit UTF-8 public reference byte interval required")
            object.__setattr__(self, "public_reference_span", tuple(span))


def identity():
    root = package_root()
    return {
        name: digest((root / name).read_bytes())
        for name in (
            "domains/operator/reduction_provider.py",
            "domains/operator/reduction.py",
            "domains/operator/convolution_provider.py",
            "domains/operator/candidate_provider.py",
        )
    }


def assess(spec, workspace, contract):
    result = dict(provider=contract.provider, status="Unknown", acceptance_proof=False)
    try:
        if type(contract) is not ReductionProviderContract:
            raise ValueError("explicit single-axis sum contract required")
        if contract.mode != "enabled":
            raise UnknownPrerequisite("provider disabled")
        if contract.scope_quote not in spec.task.split("REFERENCE:\n", 1)[0]:
            raise ValueError("complete-module scope quote not in original public task")
        source, _ = _source_and_lowerer_task(spec, contract)
        path = workspace.path(contract.target, write=True)
        if path.exists() and (not path.is_file() or path.stat().st_size > 1_000_000):
            raise UnknownPrerequisite("bounded regular target required")
        original = path.read_bytes() if path.exists() else None
        if original is not None and not original:
            raise UnknownPrerequisite("empty existing target")
        start = time.monotonic()
        candidate = generate(source, schedule="scalar")
        if time.monotonic() - start > 2:
            raise UnknownPrerequisite("sum generation CPU bound exceeded")
        if len(candidate.source.encode()) > 1_000_000:
            raise ValueError("generated source size bound")
        ast.parse(candidate.source)
        from infinicoder.memory.repair_memory import signature

        result.update(
            status="Applicable",
            requirements=signature(spec),
            contract=asdict(contract),
            implementation=identity(),
            root=str(workspace.root),
            target_before=original,
            target_before_sha256=digest(original) if original is not None else None,
            candidate_source=candidate.source,
            candidate_sha256=digest(candidate.source),
            facts=dict(
                semantics="single_axis_sum",
                descriptor=asdict(candidate.descriptor),
                ranks=(2, 3),
                max_reduction_extent=8192,
                max_input_elements=67108864,
                layout="runtime actual strides and data pointer including storage offset",
                hidden_input_domain="unknown; independent final required",
            ),
            preconditions=list(candidate.preconditions),
            provenance="deterministic public-AST single-axis sum lowering;not model reasoning",
        )
    except (
        ValueError,
        TypeError,
        SyntaxError,
        RecursionError,
        OverflowError,
        UnicodeError,
        OSError,
    ) as error:
        result.update(
            status="Unknown" if isinstance(error, UnknownPrerequisite) else "Unsupported",
            reason=type(error).__name__ + ": " + str(error),
        )
    return result


def validate_lease(spec, workspace, contract, admission):
    from infinicoder.memory.repair_memory import signature

    if (
        type(contract) is not ReductionProviderContract
        or admission.get("status") != "Applicable"
        or admission["requirements"] != signature(spec)
        or admission["contract"] != asdict(contract)
        or admission["implementation"] != identity()
        or admission["root"] != str(workspace.root)
        or digest(admission["candidate_source"]) != admission["candidate_sha256"]
    ):
        raise ProviderLeaseError("sum implementation/source lease changed")
    source, _ = _source_and_lowerer_task(spec, contract)
    if contract.scope_quote not in spec.task.split("REFERENCE:\n", 1)[0]:
        raise ProviderLeaseError("sum caller scope changed")
    if generate(source, schedule="scalar").source != admission["candidate_source"]:
        raise ProviderLeaseError("sum candidate no longer matches the public producer")
    path = workspace.path(contract.target, write=True)
    actual = path.read_bytes() if path.exists() else None
    if actual != admission["target_before"]:
        raise ProviderLeaseError("target changed after admission")


def apply(spec, workspace, contract, admission):
    validate_lease(spec, workspace, contract, admission)
    args = dict(path=contract.target, expected_hash=admission["target_before_sha256"])
    if admission["target_before"] is None:
        args["content"] = admission["candidate_source"]
    else:
        args.update(
            old=admission["target_before"].decode("utf8"), new=admission["candidate_source"]
        )
    return workspace.edit(**args)
