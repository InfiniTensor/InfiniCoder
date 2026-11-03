"""One public entry point for conditionally selected native capabilities."""

from dataclasses import replace
from pathlib import Path

from infinicoder.runtime.agent_execution import CoreRequest, UnsupportedCore
from infinicoder.memory.source.evidence_source import ReadOnlyTaskSources
from infinicoder.runtime.extensions.native_capability_pipeline import (
    CapabilityPipelinePolicy,
    NativeCapabilityPipeline,
    NativeRouteObserver,
    capability_selection,
)
from infinicoder.domains.operator.native_implementation_provider import NativeImplementationProvider
from infinicoder.optimization.native_optimizing_provider import optimize_provider_request
from infinicoder.memory.source.native_reversible_working_set import SourceAttachment
from infinicoder.verification.public_performance_report import PublicPerformanceReport


def _policy(value):
    if value is None:
        return CapabilityPipelinePolicy()
    return CapabilityPipelinePolicy(**value) if isinstance(value, dict) else value


def _performance(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return PublicPerformanceReport(
            **dict(value, rollback_paths=tuple(value.get("rollback_paths", ())))
        )
    if not isinstance(value, PublicPerformanceReport):
        raise ValueError("explicit external public performance report contract required")
    return value


def configure_development_request(
    request,
    *,
    source_capability,
    policy=None,
    attachment=None,
    artifact_write_paths=(),
    reproduction=None,
    performance_policy=None,
    public_performance=None,
    exclusive_workspace=False,
    convolution_provider=None,
    reduction_provider=None,
    rmsnorm_provider=None,
    fusion_provider=None,
    namespace_binding=None,
):
    """Select one owner on an unused request;preserve every CoreRequest field."""
    if not isinstance(request, CoreRequest):
        raise TypeError("original CoreRequest required")
    if any(
        getattr(request, k) is not None
        for k in (
            "planning_adapter",
            "repair_adapter",
            "candidate_search",
            "implementation_proposal",
        )
    ):
        raise UnsupportedCore("configure one ordinary request before attaching other owners")
    old = request.context_adapter
    if (
        old is None
        or getattr(old, "bound_once", False)
        or not callable(getattr(old, "count_messages", None))
    ):
        raise ValueError("unused ordinary context with the actual tokenizer required")
    from infinicoder.runtime.native_plain import NativePlainObserver
    from infinicoder.verification.native_public_reproduction import (
        NativePublicReproductionBootstrap,
    )

    if type(old) not in (
        NativePlainObserver,
        NativePublicReproductionBootstrap,
        NativeImplementationProvider,
    ):
        raise UnsupportedCore("existing experimental context is not silently replaced")
    policy = _policy(policy)
    if not isinstance(policy, CapabilityPipelinePolicy):
        raise ValueError("explicit CapabilityPipelinePolicy required")
    public_performance = _performance(public_performance)
    if performance_policy is not None:
        from infinicoder.optimization.native_measured_optimization import PublicPerformancePolicy

        if not isinstance(performance_policy, PublicPerformancePolicy):
            raise ValueError("caller-issued typed performance policy required")
    if public_performance and performance_policy:
        raise ValueError("choose one identified performance operation")
    if public_performance and exclusive_workspace is not True:
        raise ValueError("measured rollback requires exclusive workspace ownership")
    if isinstance(attachment, dict):
        attachment = SourceAttachment(**attachment)
    if attachment is not None:
        attachment.validate(request.spec.task)
    source_capability.validate(request.workspace, source_capability.paths)
    reproduction = reproduction or getattr(old, "reproduction", None)
    if reproduction is not None:
        frozen = set(reproduction.diagnostic.spec.public_input_hashes)
        rollback = (
            performance_policy.rollback_paths
            if performance_policy
            else public_performance.rollback_paths
            if public_performance
            else ()
        )

        def overlaps(a, b):
            return a == b or a.startswith(b.rstrip("/") + "/") or b.startswith(a.rstrip("/") + "/")

        if any(overlaps(p, key) for p in (*artifact_write_paths, *rollback) for key in frozen):
            raise ValueError(
                "readonly public inputs cannot be writable artifacts or rollback targets"
            )
        for cap in (reproduction.ordinary, reproduction.diagnostic):
            for key, backing in (cap.readonly_sources or {}).items():
                if (
                    key in source_capability.paths
                    and (Path(request.workspace) / key).resolve() != Path(backing).resolve()
                ):
                    raise ValueError(
                        "readonly public identity differs from the workspace-backed source reader"
                    )
    check = getattr(request.acceptance_adapter, "public_check", None)
    if public_performance and check is None:
        raise ValueError(
            "standard performance reporting requires an explicit original public checker"
        )
    if public_performance:
        public_performance.policy(check.operation, exclusive_workspace)
        public_performance.decoder(check)
    if (
        sum(
            v is not None
            for v in (convolution_provider, reduction_provider, rmsnorm_provider, fusion_provider)
        )
        > 1
    ):
        raise ValueError("choose one explicitly typed implementation provider")
    if isinstance(convolution_provider, dict):
        from infinicoder.domains.operator.convolution_provider import ConvolutionProviderContract

        convolution_provider = ConvolutionProviderContract(**convolution_provider)
    if isinstance(reduction_provider, dict):
        from infinicoder.domains.operator.reduction_provider import ReductionProviderContract

        reduction_provider = ReductionProviderContract(**reduction_provider)
    if isinstance(rmsnorm_provider, dict):
        from infinicoder.domains.operator.rmsnorm_provider import RMSNormProviderContract

        rmsnorm_provider = RMSNormProviderContract(**rmsnorm_provider)
    if isinstance(fusion_provider, dict):
        from infinicoder.domains.operator.fusion_provider import FusionProviderContract

        fusion_provider = FusionProviderContract(**fusion_provider)
    supplied = next(
        (
            v
            for v in (convolution_provider, reduction_provider, rmsnorm_provider, fusion_provider)
            if v is not None
        ),
        None,
    )
    contract = supplied if supplied is not None else getattr(old, "contract", None)
    if contract is not None:
        from infinicoder.domains.operator.convolution_provider import ConvolutionProviderContract
        from infinicoder.domains.operator.reduction_provider import ReductionProviderContract
        from infinicoder.domains.operator.rmsnorm_provider import RMSNormProviderContract
        from infinicoder.domains.operator.fusion_provider import FusionProviderContract

        if type(contract) not in (
            ConvolutionProviderContract,
            ReductionProviderContract,
            RMSNormProviderContract,
            FusionProviderContract,
        ):
            raise ValueError("explicit supported provider contract required")
        if type(old) is NativeImplementationProvider and old.contract != contract:
            raise ValueError("supplied provider contract differs from the existing owner")
    available_provider = contract is not None and getattr(contract, "mode", None) == "enabled"
    selection = capability_selection(
        request,
        source_capability.paths,
        policy,
        attachment=attachment,
        artifact_write_paths=artifact_write_paths,
        reproduction=reproduction,
        performance=performance_policy is not None or public_performance is not None,
        provider=available_provider,
    )
    if selection["modules"]["implementation_provider"]["active"]:
        if type(old) is not NativeImplementationProvider:
            if check is None:
                raise ValueError("implementation generation requires the original public checker")
            old = NativeImplementationProvider(
                old.count_messages,
                tokenizer_identity=old.tokenizer_identity,
                implementation_paths=check.operation.observed_paths,
                contract=contract,
                public_check=check,
            )
            request = replace(request, context_adapter=old)
        if performance_policy:
            raise ValueError(
                "operator bundle needs the exact original public report;use its report decoder or direct optimize_provider_request"
            )
        if public_performance:
            decode, identity = public_performance.decoder(check)
            request = optimize_provider_request(
                request,
                performance_decoder=decode,
                decoder_identity=identity,
                exclusive_workspace=exclusive_workspace,
                rollback_paths=public_performance.rollback_paths,
                max_candidates=public_performance.max_candidates,
                final_reserve_seconds=public_performance.final_reserve_seconds,
                min_speedup=public_performance.min_speedup,
            )
        request.context_adapter.capability_selection = selection
        return request
    active = any(v["active"] for v in selection["modules"].values())
    if not active:
        context = NativeRouteObserver(
            old.count_messages,
            tokenizer_identity=old.tokenizer_identity,
            implementation_paths=old.paths,
            selection=selection,
        )
        return replace(request, context_adapter=context)
    performance_paths = (
        performance_policy.operation.observed_paths
        if performance_policy
        else check.operation.observed_paths
        if public_performance
        else ()
    )
    if not set(performance_paths).issubset(source_capability.paths):
        raise ValueError("measurement source coverage exceeds the declared read capability")
    watched = tuple(
        dict.fromkeys(
            (
                *getattr(old, "paths", ()),
                *artifact_write_paths,
                *performance_paths,
                *(reproduction.diagnostic.spec.source_paths if reproduction else ()),
            )
        )
    )
    if namespace_binding is None and reproduction:
        namespace_binding = reproduction.diagnostic.namespace_binding
    context = NativeCapabilityPipeline(
        old.count_messages,
        tokenizer_identity=old.tokenizer_identity,
        implementation_paths=watched,
        source_capability=source_capability,
        attachment=attachment,
        artifact_write_paths=artifact_write_paths,
        selection=selection,
        reproduction=reproduction,
        namespace_binding=namespace_binding,
        performance_policy=performance_policy,
        performance_report=public_performance,
        exclusive_workspace=exclusive_workspace,
        public_check=check,
    )
    return replace(request, context_adapter=context, planning_adapter=context)


def create_development_request(
    spec, workspace, state_dir, *, pipeline_policy=None, public_performance=None, **options
):
    """CPU CLI/API factory. Custom GPU/SWE callers use configure above."""
    from infinicoder.app.native_api import create_cpu_request

    policy = _policy(pipeline_policy)
    if not isinstance(policy, CapabilityPipelinePolicy):
        raise ValueError("explicit pipeline policy required")
    unsupported = (
        "body_write_scopes",
        "public_targets",
        "phase_history",
        "hypothesis_branch",
        "transition_contract",
        "transition_reference_script",
        "transition_operation",
        "proposal_body_scope",
        "proposal_read_paths",
        "source_question_completion",
        "source_question_actions",
        "delivery_policy",
        "observed_interface_map",
        "observed_public_sources",
        "observed_scope_quotes",
        "contract_review",
        "contract_review_policy",
        "behavior_plan",
        "behavior_plan_policy",
        "task_state_consolidation",
        "revision_task_checkpoint",
        "action_effect_memory",
        "public_replay",
    )
    if any(options.get(k) for k in unsupported):
        raise UnsupportedCore(
            "choose the existing explicit profile for additional legacy mechanisms;cpu-auto never clears their contracts"
        )
    readable = tuple(
        dict.fromkeys(
            (*options.get("implementation_paths", ()), *options.pop("source_read_paths", ()))
        )
    )
    attachment = options.pop("source_attachment", None)
    writes = tuple(options.pop("architecture_write_paths", ()))
    performance = options.pop("performance_policy", None)
    exclusive = options.pop("exclusive_workspace", False)
    public_performance = _performance(public_performance)
    if public_performance and (not options.get("public_check_script") or exclusive is not True):
        raise ValueError(
            "standard performance reporting requires a separate public script and exclusive workspace"
        )
    provider = options.pop("convolution_provider", None)
    reduction = options.pop("reduction_provider", None)
    rmsnorm = options.pop("rmsnorm_provider", None)
    fusion = options.pop("fusion_provider", None)
    reproduction_spec = options.pop("public_reproduction", None)
    reproduction_sources = options.pop("public_reproduction_sources", None)
    diagnostics = options.pop("public_diagnostics", True)
    if type(diagnostics) is not bool:
        raise ValueError("public_diagnostics must be boolean")
    from infinicoder.verification.public_reproduction import validate_public_input_sources

    validate_public_input_sources(reproduction_sources, options.get("check_script"))
    if reproduction_spec is None and (not diagnostics or reproduction_sources):
        raise ValueError("public reproduction options need their explicit contract")
    policy = replace(policy, diagnostics=policy.diagnostics and diagnostics)
    # The neutral builder does not run a model. These explicit capabilities
    # are installed below on the returned request,not erased from TaskSpec.
    request = create_cpu_request(spec, workspace, state_dir, profile="cpu-plain", **options)
    reproduction = None
    if reproduction_spec is not None:
        from infinicoder.verification.public_reproduction import cpu_public_reproduction

        reproduction = cpu_public_reproduction(
            reproduction_spec,
            workspace,
            state_dir,
            request.execution_adapter,
            public_sources=reproduction_sources,
        )
    check = getattr(request.acceptance_adapter, "public_check", None)
    if public_performance and check is None:
        raise ValueError("performance contract requires an explicit public checker")
    extra = (
        performance.operation.observed_paths
        if performance
        else check.operation.observed_paths
        if public_performance
        else ()
    )
    readable = tuple(dict.fromkeys((*readable, *extra, *writes)))
    return configure_development_request(
        request,
        source_capability=ReadOnlyTaskSources(workspace, readable),
        policy=policy,
        attachment=attachment,
        artifact_write_paths=writes,
        performance_policy=performance,
        public_performance=public_performance,
        exclusive_workspace=exclusive,
        convolution_provider=provider,
        reduction_provider=reduction,
        rmsnorm_provider=rmsnorm,
        fusion_provider=fusion,
        reproduction=reproduction,
    )
