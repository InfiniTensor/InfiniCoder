"""Declarative capability contracts and composition receipts, not an executor.

The host Runtime retains every permission, total budget and acceptance decision.
Unknown module cost and coding benefit remain unknown, never silently zero.
"""

CONTRACTS = {
    "pointwise_implementation": dict(
        role="implementation_provider",
        inputs=["explicit_caller_ProviderContract", "public_reference_AST", "current_target_lease"],
        outputs=["complete_local_candidate", "original_acceptance_observation"],
        effects=["guarded_source_edit", "original_verification"],
        activation={"domain": ["operator"]},
        extra_model_calls=0,
        call_cost="At most one admitted local pointwise implementation attempt;original edits,checks,time,steps and rollback remain authoritative. Tags alone cannot establish applicability. Explicit caller contract required;not automatically selected.",
    ),
    "inspection_recovery": dict(
        role="decision_workflow",
        inputs=[
            "three_identical_validated_reads",
            "current_source_versions",
            "current_caller_contract",
        ],
        outputs=["one_source_grounded_workflow_transition"],
        effects=["session_reroute"],
        activation={},
        extra_model_calls=0,
        call_cost="At most one bounded proposal after three unchanged validated reads;all calls,time,source guards and original independent acceptance remain shared.",
    ),
    "source_working_set": dict(
        role="memory_organization",
        inputs=["validated_source_reads"],
        outputs=["current_working_source_fragments"],
        effects=["context_only"],
        activation={},
        extra_model_calls=0,
        call_cost="Existing source_sync policy,not a new analysis capability.",
    ),
    "source_archive": dict(
        role="memory_organization",
        inputs=["validated_reads", "exact_literal_read_output"],
        outputs=["versioned_archive_references", "bounded_partial_recall"],
        effects=["session_local_archive", "context_only"],
        activation={"structure": ["cross_module"]},
        extra_model_calls=0,
        call_cost="At most64 bounded snapshots;recall uses original task calls/time and actual tokenizer output budget. Source observations are not behavior proof.",
    ),
    "design_delta": dict(
        role="decision_workflow",
        inputs=["fully_displayed_unverified_hypothesis", "current_reference_offer"],
        outputs=["explicit_reference_amendment", "materialized_full_design"],
        effects=["session_design_revision"],
        activation={"language": ["cpp", "cuda"]},
        extra_model_calls=0,
        call_cost="Explicit reuse of complete displayed semantics with current references;all original reference/capacity/check budgets remain. Semantic changes require a full new design.",
    ),
    "design_revision_memory": dict(
        role="memory",
        inputs=["reference_bound_design", "capacity_rejection", "current_source_versions"],
        outputs=["bounded_unverified_semantic_claims"],
        effects=["session_hypothesis_memory"],
        activation={"language": ["cpp", "cuda"]},
        extra_model_calls=0,
        call_cost="Retain whole prior model claims within512 actual tokens at a capacity revision;unknown semantics remain hypotheses. No additional model or acceptance.",
    ),
    "binding_candidates": dict(
        role="verified_proposal_workflow",
        inputs=[
            "current_failed_source",
            "caller_examples",
            "unique_binding_hypothesis",
            "full_final_checks",
        ],
        outputs=["versioned_candidate", "independent_trial", "accepted_source_or_guarded_rollback"],
        effects=["guarded_source_edit", "original_verification"],
        activation={"language": ["python"]},
        extra_model_calls=0,
        call_cost="At most2 task-owned source-derived candidates;all full original verification consumes the existing deadline. No new model credits or candidate self-certification.",
    ),
    "example_diagnosis": dict(
        role="source_failure_diagnosis",
        inputs=[
            "caller_literal_numeric_assertions",
            "current_pure_arithmetic_source",
            "independent_failure",
        ],
        outputs=["versioned_example_contradictions", "argument_order_hypotheses"],
        effects=["read_only_static_evaluation"],
        activation={"language": ["python"]},
        extra_model_calls=0,
        call_cost="Bounded AST interpretation of supplied numeric examples after worker failure;no model/process/hidden judge/source edit. Unsupported syntax remains unknown.",
    ),
    "design_handoff": dict(
        role="workflow_preflight",
        inputs=[
            "bound_design_hypothesis",
            "complete_task",
            "current_source_records",
            "actual_tokenizer_budget",
        ],
        outputs=["capacity_admitted_design", "bounded_design_revision_feedback"],
        effects=["read_only_capacity_preflight"],
        activation={"language": ["cpp", "cuda", "c"]},
        call_cost="Zero model calls for preflight;one capacity revision shares original Runtime budget. Capacity is not semantic proof.",
    ),
    "source_table": dict(
        role="context_representation",
        inputs=["visible_source_records", "current_source_versions"],
        outputs=["lossless_file_version_table", "stable_file_aliases"],
        effects=["context_only"],
        activation={"language": ["cpp", "cuda", "c"]},
        extra_model_calls=0,
        call_cost="Lossless source provenance factoring within original context/deadline;no source read,summary,write or acceptance authority",
    ),
    "state_contract": dict(
        role="state_memory_and_preflight",
        inputs=[
            "current_owner_fields",
            "constructor_initializers",
            "model_state_effect_claims",
            "proposed_body",
        ],
        outputs=["versioned_initialization_view", "state_order_conflict_or_unknown"],
        effects=["read_only_analysis", "pre_edit_consistency_check"],
        activation={"structure": ["state_lifecycle"], "language": ["cpp", "cuda", "c"]},
        extra_model_calls=0,
        call_cost="Checks use zero model calls;up to2 conflict repairs and1 design revision consume original shared Runtime budget;unknown is not proof",
    ),
    "native_analysis": dict(
        role="memory_organization",
        inputs=["source_catalog", "compile_configuration"],
        outputs=["coarse_catalog", "requested_semantic_facts"],
        effects=["bounded_read_only_parser_process"],
        activation={"language": ["cpp", "cuda", "c"]},
        extra_model_calls=0,
        call_cost="Coarse catalog then one requested translation unit,at most five parser seconds or remaining shared Runtime time;unexpanded relationships stay unknown",
    ),
    "native_plan": dict(
        role="design_implementation_workflow",
        inputs=[
            "current_clang_scope",
            "source_owner_fields",
            "dependency_records",
            "caller_contract",
        ],
        outputs=[
            "source_bound_design_hypothesis",
            "guarded_body_candidate",
            "independent_acceptance",
        ],
        effects=["context_and_proposal_strategy"],
        activation={"domain": ["framework", "operator"], "language": ["cpp", "cuda", "c"]},
        call_cost="Two initial proposals share Runtime calls,context,deadline;failure retains all spent budget and original broader scope",
    ),
    "source_focus": dict(
        role="implementation_workflow",
        inputs=[
            "current_static_scope_candidate",
            "complete_caller_contract",
            "bounded_final_acceptance",
        ],
        outputs=[
            "provisional_function_work_packet",
            "guarded_body_candidate",
            "independent_acceptance",
        ],
        effects=["context_and_proposal_strategy"],
        activation={"domain": ["general"], "intent": ["debug"], "language": ["python"]},
        call_cost="one initial source proposal charges original Runtime budget;fallback retains all spent calls and original deadline",
    ),
    "source_boundaries": dict(
        role="context_view",
        inputs=["current_python_read", "current_ast"],
        outputs=["versioned_syntax_spans", "conditional_body_read_hint"],
        effects=["read_only_analysis"],
        activation={"language": ["python"]},
        extra_model_calls=0,
        call_cost="metadata shares original context budget;read coverage still only visible original text",
    ),
    "python_syntax_edits": dict(
        role="edit_preflight",
        inputs=["proposed_typed_python_write"],
        outputs=["compile_only_rejection_or_permission_to_write"],
        effects=["typed_write_preflight"],
        activation={"language": ["python"]},
        extra_model_calls=0,
        call_cost="bounded compile-only CPU work shares original deadline;no external acceptance proof",
    ),
    "source_direct": dict(
        role="executor",
        inputs=["task_contract"],
        outputs=["candidate_source"],
        effects=["workspace_edit"],
        activation={},
        call_cost="within_shared_runtime_budget",
    ),
    "action_loop": dict(
        role="executor",
        inputs=["task_contract", "command_permission"],
        outputs=["action_observation_history"],
        effects=["workspace_edit", "caller_permitted_commands"],
        activation={},
        call_cost="within_shared_runtime_budget",
    ),
    "text_driver": dict(
        role="decision_frontend",
        inputs=["pinned_sdk_source", "bounded_runtime_context"],
        outputs=["canonical_runtime_action", "sdk_formatted_observation"],
        effects=["context_and_action_protocol"],
        activation={},
        conflicts=["xml_tools", "xml_palette", "protocol_compat"],
        extra_model_calls=0,
        call_cost="one SDK query uses the existing Runtime reservation;no independent budget or retry",
    ),
    "task_first_context": dict(
        role="context_view",
        inputs=["mandatory_runtime_contract"],
        outputs=["lossless_task_first_packet", "projection_receipt"],
        effects=["context_only"],
        activation={},
        extra_model_calls=0,
        call_cost="existing context/deadline budgets;no summarizer call",
    ),
    "working_code": dict(
        role="memory",
        inputs=["task_contract", "current_code_index"],
        outputs=["question_working_set", "versioned_source_facts"],
        effects=["read_only_analysis"],
        activation={"structure": ["cross_module", "state_lifecycle"]},
        call_cost="explicit_memory_actions_share_runtime_budget",
    ),
    "domain_guidance": dict(
        role="knowledge_template",
        inputs=["task_contract", "domain_profile"],
        outputs=["development_workflow"],
        effects=["context_only"],
        activation={"domain": ["operator", "framework", "system"]},
        extra_model_calls=0,
    ),
    "grounded_plan": dict(
        role="decision",
        inputs=["task_contract", "current_source_observations"],
        outputs=["validated_work_graph"],
        effects=["control_flow"],
        activation={"structure": ["cross_module", "state_lifecycle"]},
        call_cost="planning_calls_share_runtime_budget",
    ),
    "domain_design": dict(
        role="design_checks",
        inputs=["task_contract", "current_source_observations"],
        outputs=["conditional_design_probes"],
        effects=["bounded_private_probes"],
        activation={"domain": ["operator", "framework"]},
        call_cost="design_calls_share_runtime_budget",
    ),
    "hypothesis_diagnosis": dict(
        role="diagnosis",
        inputs=["versioned_independent_failure"],
        outputs=["discriminating_probe"],
        effects=["bounded_private_probes"],
        activation={"intent": ["debug"]},
        call_cost="diagnostic_calls_share_runtime_budget",
    ),
    "failure_diagnosis": dict(
        role="diagnosis",
        inputs=["versioned_repeated_failure"],
        outputs=["failure_routing_feedback"],
        effects=["control_flow"],
        activation={"intent": ["debug"]},
        call_cost="diagnostic_calls_share_runtime_budget",
    ),
    "protocol_compat": dict(
        role="tool_protocol",
        inputs=["actual_action_transport"],
        outputs=["canonical_runtime_action"],
        effects=["action_normalization"],
        activation={},
        conflicts=["xml_tools", "xml_palette"],
        extra_model_calls=0,
    ),
    "symbol_edit": dict(
        role="tool",
        inputs=["current_python_ast", "source_lease"],
        outputs=[
            "guarded_symbol_edit",
            "versioned_function_body_view",
            "interface_preserving_body_edit",
        ],
        effects=["workspace_edit"],
        activation={"language": ["python"]},
        call_cost="tool_calls_share_runtime_budget",
    ),
    "xml_palette": dict(
        role="context_view",
        inputs=["actual_action_transport"],
        outputs=["on_demand_tool_schema"],
        effects=["context_only"],
        activation={},
        conflicts=["protocol_compat"],
        extra_model_calls=0,
    ),
    "xml_tools": dict(
        role="tool_protocol",
        inputs=["actual_action_transport"],
        outputs=["canonical_runtime_action"],
        effects=["action_normalization"],
        activation={},
        conflicts=["protocol_compat"],
        extra_model_calls=0,
    ),
    "problem_workflow": dict(
        role="decision_workflow",
        inputs=["public_task", "current_source", "validated_reads"],
        outputs=["problem_work_packet", "versioned_prediction_observation"],
        effects=["context_organization", "bounded_caller_probe"],
        activation={},
        call_cost="existing model/tool/time budgets;max2 probes/session and1/question",
    ),
    "read_batch": dict(
        role="tool_protocol",
        inputs=["complete_schema_validated_xml_calls"],
        outputs=["per_child_read_observations"],
        effects=["read_only_workspace_inspection"],
        activation={},
        extra_model_calls=0,
        call_cost="each attempted child charges shared tool-step budget;one paid model reply",
    ),
    "tensor_dataflow": dict(
        role="analysis_memory",
        inputs=["current_python_source_observation", "caller_tensor_declarations"],
        outputs=["conditional_shape_flow"],
        effects=["read_only_analysis"],
        activation={
            "domain": ["operator", "framework"],
            "language": ["python"],
            "inputs": ["declared_tensor_shapes"],
        },
        extra_model_calls=0,
        max_context_tokens=420,
    ),
    "operator_protocol": dict(
        role="knowledge_workflow",
        inputs=["task_contract", "domain_profile"],
        outputs=["primitive_method_questions"],
        effects=["context_only"],
        activation={"domain": ["operator"], "primitive": ["scan", "reduction", "contraction"]},
        extra_model_calls=0,
        max_context_tokens=420,
    ),
    "literal_reads": dict(
        role="context_view",
        inputs=["actual_source_read"],
        outputs=["literal_source_view"],
        effects=["context_only"],
        activation={},
        extra_model_calls=0,
    ),
    "checked_edits": dict(
        role="validation_policy",
        inputs=["bounded_independent_nonperformance_acceptance"],
        outputs=["original_acceptance_feedback"],
        effects=["independent_check_execution"],
        activation={"intent": ["debug", "implement"]},
        call_cost="original_checks_charge_shared_time_budget",
    ),
}


def applicability(name, axes):
    """AND across axes;OR within an axis. Empty predicates are caller-only."""
    contract = CONTRACTS[name]
    predicate = contract["activation"]
    matched = {
        axis: sorted(set(choices) & set(axes.get(axis, []))) for axis, choices in predicate.items()
    }
    missing = [axis for axis, values in matched.items() if not values]
    return dict(
        candidate=bool(predicate) and not missing,
        matched=matched,
        missing_axes=missing,
        interpretation="rule applicability only;not calibrated probability or demonstrated coding gain",
    )


def compose(modules, catalog, axes):
    if set(catalog) != set(CONTRACTS):
        raise ValueError("Every capability requires a routing contract")
    conflicts = [
        (name, other)
        for name in modules
        for other in CONTRACTS[name].get("conflicts", [])
        if other in modules
    ]
    if conflicts:
        raise ValueError(
            "Conflicting capability modules: "
            + ",".join(sorted({a + " / " + b for a, b in conflicts}))
        )
    rows = []
    for name, module in catalog.items():
        contract = CONTRACTS[name]
        rows.append(
            dict(
                module=name,
                selected=name in modules,
                dependencies=list(module.requires),
                contract=contract,
                applicability=applicability(name, axes),
                coding_evidence="not_calibrated_for_this_model_and_task_stratum",
                state="selected_pending_runtime_inputs" if name in modules else "inactive",
                freshness="Source artifacts must be revalidated at use;selection is not a current-source fact",
            )
        )
    return dict(
        version=1,
        modules=rows,
        executor=[name for name in modules if CONTRACTS[name]["role"] == "executor"],
        known_context_ceiling=sum(CONTRACTS[name].get("max_context_tokens", 0) for name in modules),
        unbounded_context_cost_modules=[
            name for name in modules if "max_context_tokens" not in CONTRACTS[name]
        ],
        budget_interpretation="Per-module declared ceilings are not reservations. Actual assembly and every model/tool/check share the existing Runtime budgets.",
        learned_gate=False,
        model_calls=0,
        subagents_spawned=0,
        artifact_bus_status="declarative_contracts;existing Runtime stores own artifacts;no generic cross-agent bus claimed",
    )
