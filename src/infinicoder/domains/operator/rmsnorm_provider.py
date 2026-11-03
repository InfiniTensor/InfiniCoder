"""Source-bound RMSNorm production through existing atomic edit/public-check APIs."""

from infinicoder._resources import package_root

import ast
from dataclasses import asdict, dataclass, fields
import re
import time

from infinicoder.domains.operator.candidate_provider import ProviderLeaseError, UnknownPrerequisite
from infinicoder.domains.operator.convolution_provider import (
    ConvolutionProviderContract,
    _source_and_lowerer_task,
    digest,
)
from infinicoder.domains.operator.rmsnorm import generate


@dataclass(frozen=True)
class RMSNormProviderContract(ConvolutionProviderContract):
    provider: str = "rmsnorm_inference_v1"

    def __post_init__(self):
        if self.provider != "rmsnorm_inference_v1":
            raise ValueError("unsupported RMSNorm provider")
        args = {f.name: getattr(self, f.name) for f in fields(ConvolutionProviderContract)}
        args["provider"] = "convolution_v1"
        checked = ConvolutionProviderContract(**args)
        object.__setattr__(self, "public_reference_span", checked.public_reference_span)


def identity():
    root = package_root()
    return {
        name: digest((root / name).read_bytes())
        for name in (
            "domains/operator/rmsnorm_provider.py",
            "domains/operator/rmsnorm.py",
            "domains/operator/convolution_provider.py",
            "domains/operator/candidate_provider.py",
        )
    }


def assess(spec, workspace, contract):
    result = dict(
        provider=getattr(contract, "provider", "rmsnorm_inference_v1"),
        status="Unknown",
        acceptance_proof=False,
    )
    try:
        if type(contract) is not RMSNormProviderContract:
            raise ValueError("explicit RMSNorm contract required")
        if contract.mode != "enabled":
            raise UnknownPrerequisite("provider disabled")
        prefix = spec.task.split("REFERENCE:\n", 1)[0]
        if contract.scope_quote not in prefix:
            raise ValueError("original caller complete-module scope required")
        if not re.search(r"\binference\b", prefix, re.I):
            raise UnknownPrerequisite("explicit inference-only declaration required")
        source, _ = _source_and_lowerer_task(spec, contract)
        path = workspace.path(contract.target, write=True)
        if path.exists() and (not path.is_file() or path.stat().st_size > 1_000_000):
            raise UnknownPrerequisite("bounded regular target required")
        before = path.read_bytes() if path.exists() else None
        started = time.monotonic()
        candidate = generate(source)
        if time.monotonic() - started > 2:
            raise UnknownPrerequisite("RMSNorm generation CPU bound exceeded")
        ast.parse(candidate.source)
        from infinicoder.memory.repair_memory import signature

        result.update(
            status="Applicable",
            requirements=signature(spec),
            contract=asdict(contract),
            implementation=identity(),
            root=str(workspace.root),
            target_before=before,
            target_before_sha256=digest(before) if before is not None else None,
            candidate_source=candidate.source,
            candidate_sha256=digest(candidate.source),
            facts=dict(
                semantics="affine_RMSNorm_last_axis",
                descriptor=asdict(candidate.descriptor),
                rank=2,
                max_features=8192,
                parameter_state="original RMSNorm field and state_dict preserved",
                hidden_input_domain="unknown;original independent final required",
            ),
            preconditions=list(candidate.preconditions),
            provenance="Deterministic public-AST RMSNorm lowering;not model reasoning",
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
        type(contract) is not RMSNormProviderContract
        or admission.get("status") != "Applicable"
        or admission["requirements"] != signature(spec)
        or admission["contract"] != asdict(contract)
        or admission["implementation"] != identity()
        or admission["root"] != str(workspace.root)
        or digest(admission["candidate_source"]) != admission["candidate_sha256"]
    ):
        raise ProviderLeaseError("RMSNorm implementation/source lease changed")
    source, _ = _source_and_lowerer_task(spec, contract)
    if (
        contract.scope_quote not in spec.task.split("REFERENCE:\n", 1)[0]
        or generate(source).source != admission["candidate_source"]
    ):
        raise ProviderLeaseError("RMSNorm public source or caller scope changed")
    path = workspace.path(contract.target, write=True)
    if (path.read_bytes() if path.exists() else None) != admission["target_before"]:
        raise ProviderLeaseError("RMSNorm target changed after admission")


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
