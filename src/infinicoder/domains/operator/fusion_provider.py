"""A source-bound fusion expert for existing linear/GELU compositions."""

from infinicoder._resources import package_root

from dataclasses import asdict, dataclass
import math
from pathlib import PurePosixPath
import re

from infinicoder.domains.operator.candidate_provider import ProviderLeaseError, UnknownPrerequisite
from infinicoder.domains.operator.convolution_provider import digest
from infinicoder.domains.operator.linear_gelu import generate


@dataclass(frozen=True)
class FusionProviderContract:
    target: str
    reference_source: str
    public_reference_sha256: str
    scope_quote: str
    complete_module_authorized: bool
    final_reserve_seconds: float
    mode: str = "enabled"
    provider: str = "linear_gelu_fusion_v1"
    entry: str = "linear_gelu"
    backend: str = "triton"
    dtype: str = "float16"
    device: str = "cuda"

    def __post_init__(self):
        if not isinstance(self.target, str):
            raise ValueError("relative Python target required")
        path = PurePosixPath(self.target)
        if (
            not self.target.endswith(".py")
            or path.is_absolute()
            or str(path) != self.target
            or ".." in path.parts
            or any(c in self.target for c in "*?[\\:\0")
            or any(p in (".git", ".infinicoder") for p in path.parts)
        ):
            raise ValueError("normalized public Python target required")
        if (
            not isinstance(self.reference_source, str)
            or not self.reference_source
            or len(self.reference_source.encode()) > 65536
            or digest(self.reference_source) != self.public_reference_sha256
        ):
            raise ValueError("bounded exact public reference and SHA256 required")
        if (
            not isinstance(self.scope_quote, str)
            or not self.scope_quote
            or len(self.scope_quote) > 2048
        ):
            raise ValueError("literal caller scope quote required")
        if self.complete_module_authorized is not True:
            raise ValueError("caller must authorize complete-module replacement")
        if (
            isinstance(self.final_reserve_seconds, bool)
            or not isinstance(self.final_reserve_seconds, (int, float))
            or not math.isfinite(self.final_reserve_seconds)
            or self.final_reserve_seconds <= 0
        ):
            raise ValueError("positive finite final reserve required")
        if (
            self.mode not in ("enabled", "disabled")
            or self.provider != "linear_gelu_fusion_v1"
            or (self.backend, self.dtype, self.device) != ("triton", "float16", "cuda")
            or not isinstance(self.entry, str)
            or not re.fullmatch(r"[A-Za-z]\w*", self.entry)
        ):
            raise ValueError("unsupported fusion contract")


def identity():
    root = package_root()
    return {
        name: digest((root / name).read_bytes())
        for name in (
            "domains/operator/fusion_provider.py",
            "domains/operator/linear_gelu.py",
        )
    }


def _source_and_lowerer_task(spec, contract):
    if (
        type(contract) is not FusionProviderContract
        or spec.amendments
        or contract.scope_quote not in spec.task
        or digest(contract.reference_source) != contract.public_reference_sha256
    ):
        raise ValueError("original task, scope and reference contract required")
    return contract.reference_source, spec.task


def assess(spec, workspace, contract):
    result = dict(provider="linear_gelu_fusion_v1", status="Unknown", acceptance_proof=False)
    try:
        source, _ = _source_and_lowerer_task(spec, contract)
        if contract.mode != "enabled":
            raise UnknownPrerequisite("fusion expert disabled")
        target = workspace.path(contract.target, write=True)
        if target.exists() and (not target.is_file() or target.stat().st_size > 1_000_000):
            raise UnknownPrerequisite("bounded regular target required")
        before = target.read_bytes() if target.exists() else None
        visible_source = before.decode("utf-8").replace("\r\n", "\n") if before else None
        if source not in spec.task and visible_source != source.replace("\r\n", "\n"):
            raise UnknownPrerequisite("reference must be visible in the task or original target")
        candidate = generate(source)
        if candidate.entry != contract.entry:
            raise UnknownPrerequisite("generated public entry differs from caller contract")
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
                semantics="tanh_GELU_after_linear_with_half_intermediate",
                entry=candidate.entry,
                schedule=candidate.schedule,
                reference_sha256=candidate.reference_sha256,
                unsupported_domains="original Torch composition fallback",
            ),
            preconditions=list(candidate.preconditions),
            provenance="Public-source AST fusion expert; correctness and speed remain externally judged",
        )
    except (
        ValueError,
        TypeError,
        SyntaxError,
        OSError,
        RecursionError,
        UnknownPrerequisite,
    ) as error:
        result["reason"] = str(error)
    return result


def validate_lease(spec, workspace, contract, admission):
    from infinicoder.memory.repair_memory import signature

    source, _ = _source_and_lowerer_task(spec, contract)
    if (
        admission.get("status") != "Applicable"
        or admission["requirements"] != signature(spec)
        or admission["contract"] != asdict(contract)
        or admission["implementation"] != identity()
        or admission["root"] != str(workspace.root)
        or generate(source).source != admission["candidate_source"]
        or digest(admission["candidate_source"]) != admission["candidate_sha256"]
    ):
        raise ProviderLeaseError("fusion task, reference or producer changed")
    target = workspace.path(contract.target, write=True)
    if (target.read_bytes() if target.exists() else None) != admission["target_before"]:
        raise ProviderLeaseError("fusion target changed after admission")


def apply(spec, workspace, contract, admission):
    validate_lease(spec, workspace, contract, admission)
    args = dict(path=contract.target, expected_hash=admission["target_before_sha256"])
    if admission["target_before"] is None:
        args["content"] = admission["candidate_source"]
    else:
        args.update(old=admission["target_before"].decode(), new=admission["candidate_source"])
    return workspace.edit(**args)
