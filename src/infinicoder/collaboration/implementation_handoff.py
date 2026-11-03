"""Explicit caller objectives can hand a real specialist artifact to Core.

Readiness covers caller-declared public requirements, never independent final
correctness. No assistant response or Submit command is synthesized.
"""

from dataclasses import asdict, dataclass
import json
import re

from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.verification.candidate_public_check import contract_hash
from infinicoder.domains.operator.convolution_provider import digest


@dataclass(frozen=True)
class ImplementationHandoffPolicy:
    task_sha256: str
    public_check_sha256: str
    hard_requirements: tuple
    public_pass_covers: tuple
    hard_performance_required: bool
    soft_aspiration_quotes: tuple = ()
    enabled: bool = False

    def __post_init__(self):
        for value in (self.task_sha256, self.public_check_sha256):
            if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
                raise ValueError("exact task and original public-check hashes required")
        for name in ("hard_requirements", "public_pass_covers", "soft_aspiration_quotes"):
            values = getattr(self, name)
            if (
                not isinstance(values, (tuple, list))
                or len(values) > 32
                or any(
                    not isinstance(x, str) or not x.strip() or len(x.encode()) > 2048
                    for x in values
                )
                or len(set(values)) != len(values)
            ):
                raise ValueError("bounded distinct caller requirement/quote list required")
            object.__setattr__(self, name, tuple(values))
        if not self.hard_requirements or not set(self.public_pass_covers) <= set(
            self.hard_requirements
        ):
            raise ValueError("public coverage must name declared hard requirements")
        if type(self.enabled) is not bool or type(self.hard_performance_required) is not bool:
            raise ValueError("explicit boolean handoff and hard-performance policy required")


@dataclass(frozen=True)
class SpecialistHandoff:
    task_sha256: str
    public_check_sha256: str
    provider_contract_sha256: str
    policy_sha256: str
    namespace_sha256: str
    source_versions: tuple
    generated_sha256: str
    hard_requirements_covered: tuple
    soft_aspirations_not_established: tuple
    public_report_ref: str
    schedule_selection_sha256: str | None = None
    authority: str = "Public readiness only;original independent final acceptance still required"


def with_specialist_handoff(request, policy):
    """Attach to the same unused owner;never clone its acceptance observer."""
    from infinicoder.domains.operator.native_implementation_provider import (
        NativeImplementationProvider,
    )

    if isinstance(policy, dict):
        policy = ImplementationHandoffPolicy(**policy)
    provider = request.context_adapter
    if (
        type(provider) is not NativeImplementationProvider
        or provider.bound_once
        or getattr(provider, "handoff_policy", None) is not None
    ):
        raise UnsupportedCore("handoff requires one unused ordinary implementation provider")
    if not isinstance(policy, ImplementationHandoffPolicy):
        raise ValueError("explicit caller handoff policy required")
    if any(
        getattr(request, name) is not None
        for name in (
            "planning_adapter",
            "repair_adapter",
            "candidate_search",
            "implementation_proposal",
        )
    ):
        raise UnsupportedCore("handoff cannot be combined with another controller")
    provider.handoff_policy = policy
    return request


def _record(provider, status, reason, receipt=None):
    value = dict(
        status=status,
        reason=reason,
        neural_calls=0,
        final_used_for_selection=False,
        soft_goals_hidden=False,
    )
    if receipt is not None:
        value["receipt"] = asdict(receipt)
    path = provider.state / "native-specialist-handoff.json"
    temporary = path.with_suffix(".next")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf8")
        temporary.replace(path)
        return True
    except OSError as error:
        provider._save_event(
            "specialist_handoff_record_unavailable", dict(error=type(error).__name__)
        )
        return False


def _precondition(provider):
    policy = provider.handoff_policy
    if digest(provider.request.spec.task) != policy.task_sha256:
        return "task_changed"
    if contract_hash(provider.public_check.manifest()) != policy.public_check_sha256:
        return "public_check_changed"
    if policy.hard_performance_required:
        # This first conservative profile has no trusted hard-performance
        # decoder. It must keep ordinary optimization, not demote the objective.
        return "hard_performance_not_supported"
    if set(policy.hard_requirements) != set(policy.public_pass_covers):
        return "hard_coverage_incomplete"
    if set(provider.request.spec.required_artifacts) != {provider.contract.target}:
        return "artifact_scope_incomplete"
    if provider.public_check.operation.observed_paths != (provider.contract.target,) or tuple(
        provider.paths
    ) != (provider.contract.target,):
        # The ordinary provider proves only its target's execution-time version.
        # A dependency's post-check hash must not be mislabeled as tested.
        return "single_source_handoff_only"
    if not set(provider.public_check.operation.observed_paths) <= set(provider.paths):
        return "observed_dependency_coverage_incomplete"
    source = provider.public_reference()
    raw = provider.request.spec.task.encode("utf8")
    span = getattr(provider.contract, "public_reference_span", None)
    if span is None:
        instructions = provider.request.spec.task.split("REFERENCE:\n", 1)[0]
    else:
        start, end = span
        instructions = (raw[:start] + raw[end:]).decode("utf8")
    if (
        digest(source) != provider.contract.public_reference_sha256
        or provider.contract.scope_quote not in instructions
        or any(quote not in instructions for quote in policy.soft_aspiration_quotes)
    ):
        return "public_objective_quote_missing"
    return None


def _receipt(provider):
    report = provider.report
    operation = provider.public_operation
    if (
        report is None
        or report.get("status") != "public_checked"
        or report.get("public_status") != "public_pass"
        or not operation
        or not operation.active
        or operation.operation is not provider.public_check.operation
        or operation._identity() != operation.contract_sha256
        or report.get("implementation") != provider.producer_identity()
    ):
        return None
    current = provider._current()
    selection = None
    search = getattr(provider, "schedule_search", None)
    if search is not None:
        if not _supported_schedule_search(provider, search):
            return None
        selection = search.selection_for(provider)
    if selection is None and (
        current != report.get("tested_sha256") or current != report.get("generated_sha256")
    ):
        return None
    snapshot = provider._snapshot()
    if not snapshot or any(v.get("kind") != "file" for v in snapshot.values()):
        return None
    return SpecialistHandoff(
        digest(provider.request.spec.task),
        contract_hash(provider.public_check.manifest()),
        contract_hash(asdict(provider.contract)),
        contract_hash(asdict(provider.handoff_policy)),
        contract_hash(provider.binding),
        tuple(sorted((p, v["sha256"]) for p, v in snapshot.items())),
        current,
        provider.handoff_policy.public_pass_covers,
        provider.handoff_policy.soft_aspiration_quotes,
        selection.public_report_ref
        if selection
        else str(provider.state / "native-implementation-provider.json"),
        schedule_selection_sha256=contract_hash(asdict(selection)) if selection else None,
    )


def prepare_specialist_handoff(provider, messages):
    """Run the original producer/public operation once before any SDK query."""
    policy = getattr(provider, "handoff_policy", None)
    if policy is None or not policy.enabled:
        return None
    try:
        reason = _precondition(provider)
        if reason:
            _record(provider, "ordinary_fallback", reason)
            return None
        provider._sync(messages)
        provider.before_query(messages)
        search = getattr(provider, "schedule_search", None)
        if search is not None:
            if not _supported_schedule_search(provider, search):
                raise ValueError("explicit matching schedule component required")
            search.run(provider)
        receipt = _receipt(provider)
        recorded = _record(
            provider,
            "ready" if receipt else "ordinary_fallback",
            "public_requirements_covered" if receipt else "public_readiness_unavailable",
            receipt,
        )
        return receipt if recorded else None
    except (ValueError, TypeError, OSError, UnsupportedCore, TimeoutError):
        # No public evidence, source, or policy failure can fabricate readiness.
        _record(provider, "ordinary_fallback", "readiness_unknown")
        return None


def _supported_schedule_search(provider, search):
    from infinicoder.domains.operator.fusion_provider import FusionProviderContract
    from infinicoder.domains.operator.fusion_schedule_search import FusionScheduleSearch

    """Closed family/type dispatch;never accept an arbitrary plugin as proof."""
    from infinicoder.domains.operator.convolution_provider import ConvolutionProviderContract
    from infinicoder.domains.operator.convolution_schedule_search import ConvolutionScheduleSearch
    from infinicoder.domains.operator.reduction_provider import ReductionProviderContract
    from infinicoder.domains.operator.reduction_schedule_search import ReductionScheduleSearch

    return (
        (type(provider.contract) is FusionProviderContract and type(search) is FusionScheduleSearch)
        or (
            type(provider.contract) is ConvolutionProviderContract
            and type(search) is ConvolutionScheduleSearch
        )
        or (
            type(provider.contract) is ReductionProviderContract
            and type(search) is ReductionScheduleSearch
        )
    )


def consume_specialist_handoff(provider, receipt):
    """Recheck every bound version immediately at Core's handoff boundary."""
    if type(receipt) is not SpecialistHandoff:
        return False
    try:
        if _precondition(provider) is not None or _receipt(provider) != receipt:
            _record(provider, "ordinary_fallback", "readiness_changed")
            return False
        return _record(provider, "handed_to_core_final", "public_requirements_covered", receipt)
    except (ValueError, TypeError, OSError, UnsupportedCore, TimeoutError):
        _record(provider, "ordinary_fallback", "readiness_changed_or_unknown")
        return False
