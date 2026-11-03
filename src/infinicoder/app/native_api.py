"""Public one-shot Native API with explicit execution and verification contracts."""

from infinicoder._resources import package_file

from pathlib import Path

from infinicoder.runtime.agent_execution import CoreRequest, NATIVE_CORE, execute_agent


def run_native(request):
    """Run a caller-owned CoreRequest without changing or dropping its contract.

    Execution, transport and acceptance adapters continue through execute_agent
    and the Native action loop without gaining additional permissions.
    Unsupported scopes,stages or resume requests fail before generation.
    """
    if not isinstance(request, CoreRequest):
        raise TypeError("run_native requires CoreRequest")
    return execute_agent(request, backend=NATIVE_CORE)


def create_cpu_request(
    spec,
    workspace,
    state_dir,
    *,
    model_config,
    tokenizer_path,
    check_script,
    python_environment=None,
    model_profile=None,
    budget=None,
    profile="cpu",
    task_working_memory=False,
    implementation_paths=(),
    public_targets=(),
    body_write_scopes=(),
    source_read_paths=(),
    final_timeout=180,
    planning_repository_task=True,
    public_check_script=None,
    public_check_coverage=None,
    public_check_paths=(),
    public_check_timeout=60,
    exclusive_workspace=False,
    phase_history=False,
    hypothesis_branch=False,
    task_environment="command-private",
    public_check_feedback="direct",
    transition_contract=None,
    transition_reference_script=None,
    transition_operation=None,
    proposal_body_scope=None,
    proposal_read_paths=(),
    source_question_completion=False,
    source_question_actions=False,
    delivery_policy=None,
    observed_interface_map=False,
    observed_public_sources=None,
    observed_scope_quotes=(),
    contract_review=False,
    contract_review_policy=None,
    behavior_plan=False,
    behavior_plan_policy=None,
    task_state_consolidation=False,
    revision_task_checkpoint=False,
    action_effect_memory=False,
    convolution_provider=None,
    reduction_provider=None,
    fusion_provider=None,
    public_replay=None,
    public_reproduction=None,
    public_diagnostics=True,
    public_reproduction_sources=None,
    source_attachment=None,
    architecture_write_paths=(),
    architecture_context="reversible",
    performance_policy=None,
    pipeline_policy=None,
    public_performance=None,
):
    """Construct the packaged Linux CPU profile;no model request is made here.

    The caller supplies a standalone trusted final-check script outside the
    mutable workspace. The CLI uses this same builder. Custom adapters can
    instead be supplied directly to run_native(CoreRequest(...)).
    """
    if profile == "cpu-auto":
        if task_working_memory:
            raise ValueError("task working memory requires cpu-plain or cpu-scoped")
        arguments = locals().copy()
        arguments.pop("profile")
        arguments.pop("task_working_memory")
        from infinicoder.app.development_api import create_development_request

        return create_development_request(**arguments)
    scoped = profile == "cpu-scoped"
    if scoped:
        profile = "cpu-plain"
    if type(task_working_memory) is not bool or task_working_memory and profile != "cpu-plain":
        raise ValueError("task working memory requires cpu-plain or cpu-scoped")
    if (
        task_working_memory
        and public_check_script is not None
        and public_check_feedback != "reports"
    ):
        raise ValueError("task working memory with public checks requires reports feedback")
    if pipeline_policy is not None or public_performance is not None:
        raise ValueError("capability policy/performance report requires cpu-auto")
    from infinicoder.runtime.agent_execution import CoreBudget
    from infinicoder.models.budget import ModelProfile
    from infinicoder.execution.native_cpu import CpuExecution, ScriptAcceptance
    from infinicoder.models.native_transport import LocalChatCounter, NativeCompatibleTransport
    from infinicoder.tools.native_shell_workbench import NativeShellCodeWorkbench
    from infinicoder.search.native_candidate_trial import NativeCandidateTrialWorkset
    from infinicoder.memory.source.native_anchored_source import NativeAnchoredSourceWorkset
    from infinicoder.memory.source.native_observation_reselection import (
        NativeObservationReselection,
    )
    from infinicoder.runtime.native_plain import NativePlainObserver
    from infinicoder.memory.native_evidence_route import NativeEvidenceRoute
    from infinicoder.memory.source.evidence_source import ReadOnlyTaskSources
    from infinicoder.planning.native_source_planning import (
        NativeSourcePlanning,
        workspace_source_reader,
    )
    from infinicoder.memory.native_task_evidence_context import NativeTaskEvidenceContext
    from infinicoder.memory.native_action_reassessment import NativeActionReassessment

    if profile not in (
        "cpu",
        "cpu-trial",
        "cpu-anchored",
        "cpu-dependencies",
        "cpu-reselection",
        "cpu-plain",
        "cpu-workset",
        "cpu-architecture",
        "cpu-measured",
        "public-transition-probe",
        "implementation-proposal",
        "cpu-evidence",
        "cpu-planning",
        "cpu-task-context",
        "cpu-task-context-off",
        "cpu-reassessment",
        "cpu-reassessment-off",
        "cpu-candidate-search",
        "cpu-single-history",
    ):
        raise ValueError("unsupported CPU profile")
    if (
        architecture_context not in ("reversible", "plain")
        or architecture_context != "reversible"
        and profile != "cpu-architecture"
    ):
        raise ValueError("architecture_context plain requires explicit cpu-architecture")
    if performance_policy is not None and profile != "cpu-measured":
        raise ValueError("performance policy requires cpu-measured")
    if profile == "cpu-measured":
        from infinicoder.optimization.native_measured_optimization import PublicPerformancePolicy

        if not isinstance(performance_policy, PublicPerformancePolicy):
            raise ValueError("caller-issued PublicPerformancePolicy required")
        if Path(performance_policy.operation.workspace).resolve() != Path(workspace).resolve():
            raise ValueError("performance workspace differs")
    if source_attachment is not None and profile not in (
        "cpu-workset",
        "cpu-architecture",
        "cpu-measured",
    ):
        raise ValueError("source attachment requires cpu-workset")
    if architecture_write_paths and profile != "cpu-architecture":
        raise ValueError("artifact write authorization requires cpu-architecture")
    if profile in ("cpu-workset", "cpu-architecture", "cpu-measured"):
        from infinicoder.memory.source.native_reversible_working_set import SourceAttachment

        if isinstance(source_attachment, dict):
            source_attachment = SourceAttachment(**source_attachment)
        if source_attachment is not None:
            if not isinstance(source_attachment, SourceAttachment):
                raise ValueError("explicit SourceAttachment required")
            source_attachment.validate(spec.task)
        if (
            public_check_script is not None
            or delivery_policy is not None
            or transition_contract is not None
        ):
            raise ValueError(
                "working-set ablation requires ordinary transport and a separate final"
            )
    if (
        type(source_question_actions) is not bool
        or source_question_actions
        and (profile != "cpu-plain" or transition_contract is not None)
    ):
        raise ValueError(
            "source_question_actions is an explicit cpu-plain-only option;other mechanisms remain separate"
        )
    if (
        type(observed_interface_map) is not bool
        or observed_interface_map
        and (profile != "cpu-plain" or source_question_actions or transition_contract is not None)
    ):
        raise ValueError("observed_interface_map is an explicit,plain-only separate capability")
    if (
        type(contract_review) is not bool
        or contract_review
        and (
            profile != "cpu-plain"
            or source_question_actions
            or observed_interface_map
            or transition_contract is not None
            or delivery_policy is not None
        )
    ):
        raise ValueError(
            "contract_review requires cpu-plain with the common legacy transport and no other mechanism"
        )
    if not contract_review and contract_review_policy is not None:
        raise ValueError("review policy requires contract_review")
    if (
        type(behavior_plan) is not bool
        or behavior_plan
        and (
            profile != "cpu-plain"
            or contract_review
            or source_question_actions
            or observed_interface_map
            or transition_contract is not None
            or delivery_policy is not None
        )
    ):
        raise ValueError(
            "behavior_plan requires cpu-plain and the common legacy transport with other mechanisms off"
        )
    if not behavior_plan and behavior_plan_policy is not None:
        raise ValueError("behavior policy requires behavior_plan")
    if (
        type(task_state_consolidation) is not bool
        or task_state_consolidation
        and (
            profile != "cpu-plain"
            or behavior_plan
            or contract_review
            or source_question_actions
            or observed_interface_map
            or transition_contract is not None
            or delivery_policy is not None
        )
    ):
        raise ValueError(
            "task_state_consolidation requires cpu-plain and legacy transport with other mechanisms off"
        )
    if (
        type(revision_task_checkpoint) is not bool
        or revision_task_checkpoint
        and (
            profile != "cpu-plain"
            or task_state_consolidation
            or behavior_plan
            or contract_review
            or source_question_actions
            or observed_interface_map
            or transition_contract is not None
            or delivery_policy is not None
        )
    ):
        raise ValueError(
            "revision_task_checkpoint requires cpu-plain,legacy transport and other mechanisms off"
        )
    if revision_task_checkpoint and not (*implementation_paths, *source_read_paths):
        raise ValueError("revision checkpoint requires explicit caller-readable source paths")
    if (
        type(action_effect_memory) is not bool
        or action_effect_memory
        and (
            profile != "cpu-plain"
            or revision_task_checkpoint
            or task_state_consolidation
            or behavior_plan
            or contract_review
            or source_question_actions
            or observed_interface_map
            or transition_contract is not None
            or delivery_policy is not None
        )
    ):
        raise ValueError(
            "action_effect_memory requires cpu-plain,legacy transport and other mechanisms off"
        )
    if sum(v is not None for v in (convolution_provider, reduction_provider, fusion_provider)) > 1:
        raise ValueError("choose one explicitly typed implementation provider")
    if convolution_provider is not None:
        from infinicoder.domains.operator.convolution_provider import ConvolutionProviderContract

        if isinstance(convolution_provider, dict):
            convolution_provider = ConvolutionProviderContract(**convolution_provider)
        if type(convolution_provider) is not ConvolutionProviderContract:
            raise ValueError("explicit convolution provider contract required")
    if reduction_provider is not None:
        from infinicoder.domains.operator.reduction_provider import ReductionProviderContract

        if isinstance(reduction_provider, dict):
            reduction_provider = ReductionProviderContract(**reduction_provider)
        if type(reduction_provider) is not ReductionProviderContract:
            raise ValueError("explicit sum provider contract required")
    if fusion_provider is not None:
        from infinicoder.domains.operator.fusion_provider import FusionProviderContract

        if isinstance(fusion_provider, dict):
            fusion_provider = FusionProviderContract(**fusion_provider)
        if type(fusion_provider) is not FusionProviderContract:
            raise ValueError("explicit fusion expert contract required")
    implementation_contract = next(
        (v for v in (convolution_provider, reduction_provider, fusion_provider) if v is not None),
        None,
    )
    if implementation_contract is not None:
        if (
            profile != "cpu-plain"
            or action_effect_memory
            or revision_task_checkpoint
            or task_state_consolidation
            or behavior_plan
            or contract_review
            or source_question_actions
            or observed_interface_map
            or transition_contract is not None
            or delivery_policy is not None
            or public_check_script is None
        ):
            raise ValueError(
                "implementation provider requires cpu-plain,original public script,legacy transport and other mechanisms off"
            )
    if public_replay is not None:
        from infinicoder.verification.public_replay import PublicReplaySpec

        if isinstance(public_replay, dict):
            public_replay = PublicReplaySpec(**public_replay)
        if not isinstance(public_replay, PublicReplaySpec):
            raise ValueError("explicit caller-public replay spec required")
        if (
            profile != "cpu-plain"
            or implementation_contract is not None
            or action_effect_memory
            or revision_task_checkpoint
            or task_state_consolidation
            or behavior_plan
            or contract_review
            or source_question_actions
            or observed_interface_map
            or transition_contract is not None
            or delivery_policy is not None
            or public_check_script is not None
            or task_environment != "command-private"
        ):
            raise ValueError(
                "public replay requires a separate command-private cpu-plain profile with other mechanisms off"
            )
    if type(public_diagnostics) is not bool:
        raise ValueError("public_diagnostics must be boolean")
    if public_reproduction is not None:
        from infinicoder.verification.public_replay import PublicReplaySpec

        if isinstance(public_reproduction, dict):
            public_reproduction = PublicReplaySpec(**public_reproduction)
        if not isinstance(public_reproduction, PublicReplaySpec):
            raise ValueError("explicit public development reproduction required")
        if (
            profile != "cpu-plain"
            or public_replay is not None
            or implementation_contract is not None
            or action_effect_memory
            or revision_task_checkpoint
            or task_state_consolidation
            or behavior_plan
            or contract_review
            or source_question_actions
            or observed_interface_map
            or transition_contract is not None
            or delivery_policy is not None
            or public_check_script is not None
        ):
            raise ValueError(
                "common public reproduction is a separate cpu-plain profile with other mechanisms off"
            )
    elif not public_diagnostics or public_reproduction_sources:
        raise ValueError("public reproduction options require a common public reproduction")
    from infinicoder.verification.public_reproduction import validate_public_input_sources

    validate_public_input_sources(public_reproduction_sources, check_script)
    if not observed_interface_map and (
        observed_public_sources is not None or observed_scope_quotes
    ):
        raise ValueError("observed source declarations require observed_interface_map")
    if source_question_actions and not (*implementation_paths, *source_read_paths):
        raise ValueError("source_question_actions requires explicit readable source paths")
    if profile == "implementation-proposal":
        if not proposal_body_scope:
            raise ValueError("One explicit proposal_body_scope required")
    elif proposal_body_scope is not None or proposal_read_paths or source_question_completion:
        raise ValueError(
            "Proposal scope and readable paths require implementation-proposal profile"
        )
    if transition_contract is not None:
        from infinicoder.verification.public_transition_probe import PublicTransitionContract

        if isinstance(transition_contract, dict):
            transition_contract = PublicTransitionContract(**transition_contract)
        if (
            not isinstance(transition_contract, PublicTransitionContract)
            or transition_reference_script is None
        ):
            raise ValueError(
                "explicit transition contract and separate caller public reference script required"
            )
        if profile not in ("cpu-plain", "public-transition-probe"):
            raise ValueError(
                "transition materials use only neutral cpu-plain or public-transition-probe"
            )
        implementation_paths = tuple(
            dict.fromkeys((*implementation_paths, *transition_contract.observed_paths))
        )
    elif transition_reference_script is not None or transition_operation is not None:
        raise ValueError("public transition reference requires its explicit contract")
    if task_environment not in ("command-private", "task-private"):
        raise ValueError("explicit command-private/task-private environment required")
    if public_check_feedback not in ("direct", "reports"):
        raise ValueError("explicit direct/reports public feedback required")
    if public_check_feedback == "reports" and (
        public_check_script is None
        or profile not in ("cpu-plain", "public-transition-probe", "implementation-proposal")
    ):
        raise ValueError(
            "reported public feedback requires explicit public_check_script and single-run cpu-plain"
        )
    if task_environment == "task-private":
        if profile not in (
            "cpu-plain",
            "cpu-workset",
            "cpu-architecture",
            "cpu-measured",
            "public-transition-probe",
            "implementation-proposal",
        ):
            raise ValueError(
                "task-private development environment requires a neutral single-run profile"
            )
        if public_check_script is not None and public_check_feedback != "reports":
            raise ValueError(
                "task-private requires a separately served public operation;direct public scripts are not yet supported with persistent temporary files"
            )
        from infinicoder.execution.native_sandbox import SANDBOX_NOTICE, PERSISTENT_SANDBOX_NOTICE
        from dataclasses import replace

        if SANDBOX_NOTICE in spec.task:
            raise ValueError(
                "public task contains the conflicting command-private notice;select the intended environment explicitly"
            )
        if PERSISTENT_SANDBOX_NOTICE not in spec.task:
            spec = replace(spec, task=spec.task + "\n" + PERSISTENT_SANDBOX_NOTICE)
    if type(phase_history) is not bool or phase_history and profile != "cpu-candidate-search":
        raise ValueError("phase_history is an explicit candidate-search-only boolean")
    if (
        type(hypothesis_branch) is not bool
        or hypothesis_branch
        and profile != "cpu-candidate-search"
    ):
        raise ValueError("hypothesis_branch requires cpu-candidate-search")
    if body_write_scopes and profile not in (
        "cpu-anchored",
        "cpu-dependencies",
        "cpu-reselection",
        "cpu-evidence",
    ):
        raise ValueError("explicit body diagnostic scopes require an anchored profile")
    if (
        source_read_paths
        and profile not in ("cpu-evidence", "cpu-workset", "cpu-architecture", "cpu-measured")
        and not source_question_actions
        and not revision_task_checkpoint
        and not action_effect_memory
    ):
        raise ValueError(
            "source_read_paths are read-only evidence capability inputs;select cpu-evidence"
        )
    workspace, state = Path(workspace).resolve(), Path(state_dir).resolve()
    if public_check_script is not None and profile not in (
        "cpu-plain",
        "public-transition-probe",
        "implementation-proposal",
        "cpu-candidate-search",
        "cpu-single-history",
    ):
        raise ValueError(
            "explicit public CPU checker requires cpu-plain,cpu-single-history or cpu-candidate-search"
        )
    if public_check_script is None and (public_check_coverage is not None or public_check_paths):
        raise ValueError("public metadata requires an explicit public check script")
    if exclusive_workspace and profile != "cpu-candidate-search":
        raise ValueError("exclusive candidate workspace declaration requires cpu-candidate-search")
    counter = LocalChatCounter(tokenizer_path)
    execution = CpuExecution(
        state.parent / ("." + state.name + "-scratch"),
        trials=profile == "cpu-trial",
        task_environment=task_environment,
        python_environment=python_environment,
    )
    if transition_contract is not None:
        if python_environment is not None:
            raise ValueError("transition profiles bind their own explicit runtime environment")
        from infinicoder.verification.transition_public_cpu import TransitionCpuExecution

        execution = TransitionCpuExecution(
            state.parent / ("." + state.name + "-scratch"),
            state,
            task_environment=task_environment,
            runtime_library_dirs=transition_contract.public_input_schema.get(
                "runtime_library_dirs", ()
            ),
        )
    acceptance = None
    search = None
    if public_check_script is not None:
        from infinicoder.verification.candidate_public_cpu import (
            CandidateCpuExecution,
            CandidateScriptAcceptance,
        )

        if transition_contract is None:
            execution = CandidateCpuExecution(
                state.parent / ("." + state.name + "-scratch"),
                state,
                task_environment=task_environment,
                python_environment=python_environment,
            )
        acceptance = CandidateScriptAcceptance(
            check_script,
            execution,
            public_script=public_check_script,
            workspace=workspace,
            state=state,
            coverage=public_check_coverage,
            observed_paths=public_check_paths,
            public_timeout=public_check_timeout,
            timeout=final_timeout,
            feedback=public_check_feedback,
        )
    transition = None
    if transition_contract is not None:
        from infinicoder.verification.transition_public_cpu import (
            CpuTransitionOperation,
            CallerTransitionOperation,
        )

        transition = (
            CpuTransitionOperation(
                transition_contract, transition_reference_script, workspace, state, execution
            )
            if transition_operation is None
            else CallerTransitionOperation(
                transition_contract,
                transition_reference_script,
                workspace,
                state,
                execution,
                transition_operation,
            )
        )
    if profile == "cpu-candidate-search":
        from infinicoder.search.native_candidate_search import CandidateSearchPolicy

        if implementation_paths or public_targets:
            raise ValueError("candidate search uses plain SDK without automatic source packages")
        context = None
        search = CandidateSearchPolicy(
            acceptance.public_check if acceptance is not None else None,
            exclusive_workspace,
            hypothesis_branch=hypothesis_branch,
        )
        if phase_history:
            from dataclasses import replace
            from infinicoder.memory.native_phase_history import NativePhaseHistoryContext

            search = replace(search, phase_history_factory=NativePhaseHistoryContext)
    elif profile == "public-transition-probe":
        if public_targets:
            raise ValueError("transition probes do not install source prefixes")
        from infinicoder.verification.native_transition_probe import NativeTransitionProbe

        context = NativeTransitionProbe(
            counter,
            tokenizer_identity=counter.identity,
            implementation_paths=implementation_paths,
            capability=transition.capability if transition is not None else None,
        )
    elif profile == "cpu-single-history":
        if public_targets:
            raise ValueError("single-run history does not install source packages")
        from infinicoder.memory.native_single_history import NativeSinglePhaseHistory

        context = NativeSinglePhaseHistory(
            counter, tokenizer_identity=counter.identity, implementation_paths=implementation_paths
        )
    elif profile in (
        "cpu-plain",
        "cpu-workset",
        "cpu-architecture",
        "cpu-measured",
        "implementation-proposal",
        "cpu-evidence",
        "cpu-planning",
        "cpu-task-context",
        "cpu-task-context-off",
        "cpu-reassessment",
        "cpu-reassessment-off",
    ):
        if public_targets:
            raise ValueError(
                "plain/evidence profiles do not install automatic public-target source packages"
            )
        extra = (
            dict(
                source_capability=ReadOnlyTaskSources(
                    workspace, tuple(dict.fromkeys((*implementation_paths, *source_read_paths)))
                ),
                body_write_scopes=body_write_scopes,
            )
            if profile == "cpu-evidence"
            else {}
        )
        if profile in (
            "cpu-planning",
            "cpu-task-context",
            "cpu-task-context-off",
            "cpu-reassessment",
            "cpu-reassessment-off",
        ):
            extra = dict(
                reader_factory=workspace_source_reader, repository_task=planning_repository_task
            )
        if profile.startswith("cpu-task-context"):
            extra["working_set"] = profile == "cpu-task-context"
        if profile.startswith("cpu-reassessment"):
            extra.update(working_set=True, reassessment=profile == "cpu-reassessment")
        cls = (
            NativeActionReassessment
            if profile.startswith("cpu-reassessment")
            else NativeTaskEvidenceContext
            if profile.startswith("cpu-task-context")
            else NativeSourcePlanning
            if profile == "cpu-planning"
            else NativeEvidenceRoute
            if profile == "cpu-evidence"
            else NativePlainObserver
        )
        context = cls(
            counter,
            tokenizer_identity=counter.identity,
            implementation_paths=implementation_paths,
            **extra,
        )
        if profile in ("cpu-workset", "cpu-architecture", "cpu-measured"):
            from infinicoder.memory.source.native_reversible_working_set import NativeWorkingSet

            performance_paths = (
                performance_policy.operation.observed_paths if performance_policy else ()
            )
            readable = tuple(
                dict.fromkeys(
                    (
                        *implementation_paths,
                        *source_read_paths,
                        *architecture_write_paths,
                        *performance_paths,
                        *(source_attachment.files if source_attachment else ()),
                    )
                )
            )
            if profile == "cpu-architecture":
                from infinicoder.planning.native_artifact_architecture import (
                    NativeArtifactArchitecture,
                )

                context_class = NativeArtifactArchitecture
            elif profile == "cpu-measured":
                from infinicoder.optimization.native_measured_optimization import (
                    NativeMeasuredOptimization,
                )

                context_class = NativeMeasuredOptimization
            else:
                context_class = NativeWorkingSet
            observed = tuple(dict.fromkeys((*implementation_paths, *performance_paths)))
            context = context_class(
                counter,
                tokenizer_identity=counter.identity,
                implementation_paths=observed,
                source_capability=ReadOnlyTaskSources(workspace, readable),
                attachment=source_attachment,
                **(
                    dict(
                        repository_task=planning_repository_task,
                        artifact_write_paths=architecture_write_paths,
                        context_policy=architecture_context,
                    )
                    if profile == "cpu-architecture"
                    else dict(performance_policy=performance_policy)
                    if profile == "cpu-measured"
                    else {}
                ),
            )
        if observed_interface_map:
            from infinicoder.memory.native_observed_interface_map import NativeObservedInterfaceMap

            context = NativeObservedInterfaceMap(
                counter,
                tokenizer_identity=counter.identity,
                implementation_paths=implementation_paths,
                reader_factory=workspace_source_reader,
                public_sources=observed_public_sources,
                scope_quotes=observed_scope_quotes,
            )
        if contract_review:
            from infinicoder.planning.native_contract_review import NativeContractReview
            from infinicoder.verification.readonly_diagnostic import ReadonlyCpuDiagnostic

            context = NativeContractReview(
                counter,
                tokenizer_identity=counter.identity,
                implementation_paths=implementation_paths,
                policy=contract_review_policy,
                reader_factory=workspace_source_reader,
                diagnostic_factory=lambda bound, deadline: ReadonlyCpuDiagnostic(
                    execution, bound, deadline
                ),
            )
        if behavior_plan:
            from infinicoder.planning.native_behavior_plan import NativeBehaviorPlan
            from infinicoder.verification.readonly_diagnostic import ReadonlyCpuDiagnostic

            context = NativeBehaviorPlan(
                counter,
                tokenizer_identity=counter.identity,
                implementation_paths=implementation_paths,
                policy=behavior_plan_policy,
                diagnostic_factory=lambda bound, deadline: ReadonlyCpuDiagnostic(
                    execution,
                    bound,
                    deadline,
                    directory=Path(bound.state_dir) / "behavior-plan" / "diagnostic",
                ),
            )
        if task_state_consolidation:
            from infinicoder.memory.native_task_state_consolidation import (
                NativeTaskStateConsolidation,
            )

            context = NativeTaskStateConsolidation(
                counter,
                tokenizer_identity=counter.identity,
                implementation_paths=implementation_paths,
            )
        if revision_task_checkpoint:
            from infinicoder.memory.native_revision_task_checkpoint import (
                NativeRevisionTaskCheckpoint,
                cpu_checkpoint_reader,
            )

            readable = tuple(dict.fromkeys((*implementation_paths, *source_read_paths)))

            def checkpoint_reader(bound, index, environment, **kwargs):
                return cpu_checkpoint_reader(bound, index, environment, paths=readable, **kwargs)

            context = NativeRevisionTaskCheckpoint(
                counter,
                tokenizer_identity=counter.identity,
                implementation_paths=implementation_paths,
                reader_factory=checkpoint_reader,
            )
        if action_effect_memory:
            from infinicoder.memory.native_action_effect_memory import NativeActionEffectMemory
            from infinicoder.memory.native_revision_task_checkpoint import cpu_checkpoint_reader

            readable = tuple(dict.fromkeys((*implementation_paths, *source_read_paths)))

            def effect_reader(bound, index, environment, **kwargs):
                return cpu_checkpoint_reader(bound, index, environment, paths=readable, **kwargs)

            context = NativeActionEffectMemory(
                counter,
                tokenizer_identity=counter.identity,
                implementation_paths=implementation_paths,
                reader_factory=effect_reader if readable else None,
            )
        if implementation_contract is not None and implementation_contract.mode == "enabled":
            from infinicoder.domains.operator.native_implementation_provider import (
                NativeImplementationProvider,
            )

            context = NativeImplementationProvider(
                counter,
                tokenizer_identity=counter.identity,
                implementation_paths=implementation_paths,
                contract=implementation_contract,
                public_check=acceptance.public_check,
            )
        if public_replay is not None:
            from infinicoder.verification.public_replay_cpu import cpu_public_replay
            from infinicoder.verification.native_public_failure_slice import (
                NativePublicFailureSlice,
            )

            capability = cpu_public_replay(public_replay, workspace, state, execution)
            context = NativePublicFailureSlice(
                counter, tokenizer_identity=counter.identity, capability=capability
            )
        if public_reproduction is not None:
            from infinicoder.verification.public_reproduction import cpu_public_reproduction
            from infinicoder.verification.native_public_reproduction import (
                NativePublicReproductionBootstrap,
            )

            reproduction = cpu_public_reproduction(
                public_reproduction,
                workspace,
                state,
                execution,
                public_sources=public_reproduction_sources,
            )
            context = NativePublicReproductionBootstrap(
                counter,
                tokenizer_identity=counter.identity,
                reproduction=reproduction,
                diagnostics=public_diagnostics,
            )
        if source_question_actions:
            import hashlib
            from infinicoder.memory.source.native_source_question import NativeSourceQuestionActions

            mapping_files = [
                package_file(name)
                for name in (
                    "execution/native_cpu.py",
                    "verification/candidate_public_cpu.py",
                    "execution/native_sandbox.py",
                )
            ]
            mapping_identity = hashlib.sha256(
                b"".join(path.read_bytes() for path in mapping_files)
            ).hexdigest()
            sources = ReadOnlyTaskSources(
                workspace, tuple(dict.fromkeys((*implementation_paths, *source_read_paths)))
            )
            context = NativeSourceQuestionActions(
                counter,
                tokenizer_identity=counter.identity,
                implementation_paths=implementation_paths,
                source_capability=sources,
                source_namespace=dict(
                    environment="packaged_CPU_exact_live_workspace_bind",
                    root=str(workspace),
                    model_root="/workspace",
                    binding_identity=mapping_identity,
                    source_coverage="caller_readable_paths;queried_files_only",
                ),
            )
    else:
        cls = (
            NativeObservationReselection
            if profile == "cpu-reselection"
            else NativeAnchoredSourceWorkset
            if profile in ("cpu-anchored", "cpu-dependencies")
            else NativeCandidateTrialWorkset
            if profile == "cpu-trial"
            else NativeShellCodeWorkbench
        )
        extra = (
            dict(
                source_dependency_policy="anchored_static_v1"
                if profile in ("cpu-dependencies", "cpu-reselection")
                else None,
                workspace_aliases=("/workspace",),
                body_write_scopes=body_write_scopes,
            )
            if issubclass(cls, NativeAnchoredSourceWorkset)
            else {}
        )
        if profile == "cpu-reselection":
            extra["observation_policy"] = "observation_reselection_v1"
        context = cls(
            counter,
            tokenizer_identity=counter.identity,
            implementation_paths=implementation_paths,
            public_targets=public_targets,
            scratch=execution.scratch,
            source_tokens=1600,
            **extra,
        )
    acceptance = acceptance or ScriptAcceptance(check_script, execution, timeout=final_timeout)
    if transition is not None:
        from infinicoder.verification.transition_public_cpu import TransitionAcceptance

        acceptance = TransitionAcceptance(acceptance, transition, execution, context)
    proposal = None
    if profile == "implementation-proposal":
        from infinicoder.runtime.implementation_proposal import ImplementationProposal

        target_path = proposal_body_scope.split("::")[0]
        sources = ReadOnlyTaskSources(
            workspace, tuple(dict.fromkeys((target_path, *proposal_read_paths)))
        )
        proposal = ImplementationProposal(
            proposal_body_scope,
            sources,
            counter,
            source_namespace="caller-host-workspace",
            source_question_completion=source_question_completion,
        )
    request = CoreRequest(
        spec,
        workspace,
        state,
        model_profile or ModelProfile(16384, 4096),
        budget or CoreBudget(),
        execution,
        NativeCompatibleTransport(model_config, counter, delivery_policy=delivery_policy),
        acceptance,
        context_adapter=context,
        candidate_search=search,
        implementation_proposal=proposal,
        planning_adapter=context
        if profile
        in (
            "cpu-architecture",
            "cpu-planning",
            "cpu-task-context",
            "cpu-task-context-off",
            "cpu-reassessment",
            "cpu-reassessment-off",
        )
        else None,
    )
    if task_working_memory:
        if type(request.context_adapter) is not NativePlainObserver or not implementation_paths:
            raise ValueError(
                "task working memory requires plain execution and observed source paths"
            )
        from dataclasses import replace

        request = replace(
            request,
            context_adapter=NativeEvidenceRoute(
                counter,
                tokenizer_identity=counter.identity,
                implementation_paths=implementation_paths,
                policy=None,
                task_working_memory=True,
            ),
        )
    if scoped:
        from infinicoder.execution.native_scoped_execution import with_scoped_native_execution

        request = with_scoped_native_execution(request)
    return request
