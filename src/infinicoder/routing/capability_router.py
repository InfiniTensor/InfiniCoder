"""Task-conditioned composition of existing, independently switchable abilities.

No task IDs, benchmark outcomes, executable repository configuration or model
calls are routing features. Initial rules are hypotheses, not calibrated gates.
"""

from infinicoder._resources import package_root

from dataclasses import dataclass, asdict
import hashlib
import json
import os
from itertools import islice
from pathlib import Path
import re
from infinicoder.routing.profiles import repository_route_facts


@dataclass(frozen=True)
class Module:
    name: str
    kind: str
    tags: tuple
    requires: tuple = ()
    options: tuple = ()
    auto_candidate: bool = False
    status: str = "experimental_unvalidated"
    input_artifacts: tuple = ()
    output_artifacts: tuple = ()
    effects: tuple = ()
    cost_contract: tuple = ()


MODULES = (
    Module(
        "pointwise_implementation",
        "implementation_provider",
        ("operator", "pointwise", "triton", "float32"),
        requires=("action_loop", "checked_edits"),
        input_artifacts=(
            "explicit_caller_ProviderContract",
            "public_reference_AST",
            "current_target_lease",
        ),
        output_artifacts=("complete_local_candidate", "original_acceptance_observation"),
        effects=("guarded_source_edit", "original_verification"),
        cost_contract=(
            ("maximum_session_attempts", 1),
            ("shared_runtime_budget", True),
            ("model_calls", 0),
        ),
    ),
    Module(
        "inspection_recovery",
        "decision_workflow",
        ("repeated_read", "current_source", "bounded_general_debug"),
        requires=("action_loop", "checked_edits"),
        options=(("inspection_recovery", True),),
        input_artifacts=(
            "three_identical_validated_reads",
            "current_source_versions",
            "current_caller_contract",
        ),
        output_artifacts=("one_source_grounded_workflow_transition",),
        effects=("session_reroute",),
        cost_contract=(("maximum_transitions", 1), ("shared_runtime_budget", True)),
    ),
    Module(
        "source_working_set",
        "memory_organization",
        ("current_source", "versioned_observation"),
        requires=("working_code",),
        options=(("source_sync", True),),
        input_artifacts=("validated_source_reads",),
        output_artifacts=("current_working_source_fragments",),
        effects=("context_only",),
        cost_contract=(("extra_model_calls", 0),),
    ),
    Module(
        "source_archive",
        "memory_organization",
        ("cross_module", "partial_expansion", "source_observation"),
        requires=("source_working_set",),
        options=(("source_archive", True),),
        input_artifacts=("validated_reads", "exact_literal_read_output"),
        output_artifacts=("versioned_archive_references", "bounded_partial_recall"),
        effects=("session_local_archive", "context_only"),
        cost_contract=(
            ("max_snapshots", 64),
            ("snapshot_bytes", 32768),
            ("recall_model_calls", "existing task budget"),
        ),
    ),
    Module(
        "problem_workflow",
        "decision_workflow",
        ("task_question", "source_relations", "discriminating_experiment"),
        requires=("working_code", "xml_tools"),
        options=(("problem_workflow", True),),
        input_artifacts=("public_task", "current_source", "validated_reads"),
        output_artifacts=("problem_work_packet", "versioned_prediction_observation"),
        effects=("context_organization", "bounded_caller_probe"),
        cost_contract=(
            ("maximum_session_experiments", 2),
            ("maximum_question_experiments", 1),
            ("shared_runtime_budget", True),
        ),
    ),
    Module(
        "read_batch",
        "tool_protocol",
        ("multi_file_inspection", "bounded_read_only"),
        requires=("xml_tools",),
        options=(("native_read_batch", True),),
        input_artifacts=("complete_schema_validated_xml_calls",),
        output_artifacts=("per_child_read_observations",),
        effects=("read_only_workspace_inspection",),
        cost_contract=(
            ("maximum_children", 8),
            ("each_attempt_costs_tool_step", True),
            ("shared_runtime_budget", True),
        ),
    ),
    Module(
        "design_delta",
        "decision_workflow",
        ("native_source", "design_revision", "sparse_artifact"),
        requires=("design_revision_memory",),
        options=(("design_delta", True),),
        input_artifacts=("fully_displayed_unverified_hypothesis", "current_reference_offer"),
        output_artifacts=("explicit_reference_amendment", "materialized_full_design"),
        effects=("session_design_revision",),
        cost_contract=(("extra_model_calls", 0), ("shared_runtime_budget", True)),
    ),
    Module(
        "design_revision_memory",
        "memory",
        ("native_source", "design_revision", "hypothesis_retention"),
        requires=("design_handoff",),
        options=(("design_revision_memory", True),),
        input_artifacts=("reference_bound_design", "capacity_rejection", "current_source_versions"),
        output_artifacts=("bounded_unverified_semantic_claims",),
        effects=("session_hypothesis_memory",),
        cost_contract=(
            ("extra_model_calls", 0),
            ("context_tokens_maximum", 512),
            ("shared_runtime_budget", True),
        ),
    ),
    Module(
        "binding_candidates",
        "verified_proposal_workflow",
        ("python", "argument_binding", "independent_failure", "bounded_search"),
        requires=("example_diagnosis",),
        options=(("binding_candidates", True),),
        input_artifacts=(
            "current_failed_source",
            "caller_examples",
            "unique_binding_hypothesis",
            "full_final_checks",
        ),
        output_artifacts=(
            "versioned_candidate",
            "independent_trial",
            "accepted_source_or_guarded_rollback",
        ),
        effects=("guarded_source_edit", "original_verification"),
        cost_contract=(
            ("extra_model_calls", 0),
            ("maximum_candidates_per_task", 2),
            ("shared_runtime_budget", True),
        ),
    ),
    Module(
        "example_diagnosis",
        "source_failure_diagnosis",
        ("python", "provided_examples", "argument_binding", "semantic_failure"),
        options=(("example_diagnosis", True),),
        input_artifacts=(
            "caller_literal_numeric_assertions",
            "current_pure_arithmetic_source",
            "independent_failure",
        ),
        output_artifacts=("versioned_example_contradictions", "argument_order_hypotheses"),
        effects=("read_only_static_evaluation",),
        cost_contract=(
            ("extra_model_calls", 0),
            ("extra_commands", 0),
            ("max_functions", 1),
            ("max_examples", 6),
            ("max_arity", 4),
        ),
    ),
    Module(
        "design_handoff",
        "workflow_preflight",
        ("short_context", "stage_transition", "design_capacity"),
        requires=("native_plan",),
        options=(("native_handoff_check", True),),
        input_artifacts=(
            "bound_design_hypothesis",
            "complete_task",
            "current_source_records",
            "actual_tokenizer_budget",
        ),
        output_artifacts=("capacity_admitted_design", "bounded_design_revision_feedback"),
        effects=("read_only_capacity_preflight",),
        cost_contract=(
            ("shared_runtime_budget", True),
            ("maximum_capacity_revisions", 1),
            ("semantic_proof", False),
        ),
    ),
    Module(
        "source_table",
        "context_representation",
        ("short_context", "source_provenance", "repeated_metadata"),
        requires=("native_plan",),
        options=(("native_source_table", True),),
        input_artifacts=("visible_source_records", "current_source_versions"),
        output_artifacts=("lossless_file_version_table", "stable_file_aliases"),
        effects=("context_only",),
        cost_contract=(("extra_model_calls", 0), ("source_association_roundtrip", True)),
    ),
    Module(
        "native_analysis",
        "memory_organization",
        ("native_source", "bounded_analysis"),
        requires=("action_loop",),
        options=(("native_index_policy", "on_demand"),),
        input_artifacts=("source_catalog", "compile_configuration"),
        output_artifacts=("coarse_catalog", "requested_semantic_facts"),
        effects=("bounded_read_only_parser_process",),
        cost_contract=(("extra_model_calls", 0), ("max_seconds_per_expansion", 5)),
    ),
    Module(
        "state_contract",
        "state_memory_and_preflight",
        ("native_source", "state_lifecycle", "state_update_order"),
        requires=("native_plan", "native_analysis"),
        options=(("native_state_contract", True),),
        input_artifacts=(
            "current_owner_fields",
            "constructor_initializers",
            "model_state_effect_claims",
            "proposed_body",
        ),
        output_artifacts=("versioned_initialization_view", "state_order_conflict_or_unknown"),
        effects=("read_only_analysis", "pre_edit_consistency_check"),
        cost_contract=(
            ("extra_model_calls", 0),
            ("shared_repair_budget", True),
            ("maximum_conflict_repairs", 2),
        ),
    ),
    Module(
        "native_plan",
        "design_implementation_workflow",
        ("native_source", "framework", "state_lifecycle"),
        requires=("action_loop",),
        options=(("native_plan", True),),
        input_artifacts=(
            "current_clang_scope",
            "source_owner_fields",
            "dependency_records",
            "caller_contract",
        ),
        output_artifacts=(
            "source_bound_design_hypothesis",
            "guarded_body_candidate",
            "independent_acceptance",
        ),
        effects=("context_and_proposal_strategy", "optional_project_configuration"),
        cost_contract=(
            ("shared_runtime_budget", True),
            ("initial_design_calls", 1),
            ("initial_code_calls", 1),
            ("maximum_metadata_configures", 1),
        ),
    ),
    Module(
        "source_focus",
        "implementation_workflow",
        ("general", "debug", "python", "current_source"),
        requires=("action_loop", "checked_edits"),
        options=(("source_focus", True),),
        input_artifacts=(
            "current_static_scope_candidate",
            "complete_caller_contract",
            "bounded_final_acceptance",
        ),
        output_artifacts=(
            "provisional_function_work_packet",
            "guarded_body_candidate",
            "independent_acceptance",
        ),
        effects=("context_and_proposal_strategy",),
        cost_contract=(("initial_proposal_calls", 1), ("shared_runtime_budget", True)),
    ),
    Module(
        "source_boundaries",
        "source_view",
        ("python", "long_docstring", "range_edit"),
        requires=("action_loop",),
        options=(("source_boundaries", True),),
        input_artifacts=("current_python_read", "current_ast"),
        output_artifacts=("versioned_syntax_spans", "conditional_body_read_hint"),
        effects=("read_only_analysis",),
        cost_contract=(("extra_model_calls", 0), ("max_metadata_ranges", 3)),
    ),
    Module(
        "python_syntax_edits",
        "edit_preflight",
        ("python", "syntax_error"),
        requires=("action_loop",),
        options=(("syntax_edits", True),),
        input_artifacts=("proposed_typed_python_write",),
        output_artifacts=("compile_only_rejection_or_permission_to_write",),
        effects=("typed_write_preflight",),
        cost_contract=(("extra_model_calls", 0),),
    ),
    Module(
        "source_direct",
        "workflow",
        ("local", "implement"),
        options=(("direct_first", True), ("contract_first", True)),
        auto_candidate=True,
    ),
    Module(
        "action_loop",
        "workflow",
        ("debug", "implement", "optimize"),
        options=(
            ("shell_loop", True),
            ("observed_edits", True),
            ("auto_verify_commands", True),
            ("defer_permission_plan", True),
        ),
        auto_candidate=True,
    ),
    Module(
        "text_driver",
        "decision_frontend",
        ("upstream_text_actions", "general"),
        requires=("action_loop",),
        options=(("decision_driver", "text"),),
        input_artifacts=("pinned_sdk_source", "bounded_runtime_context"),
        output_artifacts=("canonical_runtime_action", "sdk_formatted_observation"),
        effects=("context_and_action_protocol",),
        cost_contract=(("extra_model_calls", 0), ("query_slot", "existing Runtime reservation")),
    ),
    Module(
        "task_first_context",
        "context_view",
        ("caller_contract", "short_context"),
        requires=("text_driver",),
        options=(("decision_driver", "text_task_first"),),
        input_artifacts=("mandatory_runtime_contract",),
        output_artifacts=("lossless_task_first_packet", "projection_receipt"),
        effects=("context_only",),
        cost_contract=(("extra_model_calls", 0), ("context_cost", "charged by existing assembler")),
    ),
    Module(
        "working_code",
        "memory",
        ("cross_module", "short_context"),
        requires=("action_loop",),
        options=(
            ("code_memory", True),
            ("task_memory", True),
            ("question_memory", True),
            ("flow_memory", False),
        ),
        auto_candidate=True,
    ),
    Module(
        "domain_guidance",
        "knowledge_template",
        ("operator", "framework", "system"),
        options=(("development_workflows", True),),
        auto_candidate=True,
    ),
    Module(
        "grounded_plan",
        "decision",
        ("cross_module", "implement"),
        requires=("action_loop",),
        options=(("grounded_planning", True), ("adaptive_replanning", True)),
    ),
    Module(
        "domain_design",
        "design_checks",
        ("operator", "framework"),
        requires=("action_loop", "domain_guidance"),
        options=(("design_contracts", True), ("design_probes", True)),
    ),
    Module(
        "hypothesis_diagnosis",
        "diagnosis",
        ("debug",),
        requires=("action_loop",),
        options=(("repair_memory", True), ("diagnostic_planning", True)),
    ),
    Module(
        "failure_diagnosis",
        "diagnosis",
        ("repeated_independent_failure",),
        requires=("action_loop",),
        options=(("failure_routing", True),),
    ),
    Module(
        "protocol_compat",
        "tool_protocol",
        ("complete_xml_in_shell_loop",),
        requires=("action_loop",),
        options=(("compatible_tools", True),),
    ),
    Module(
        "symbol_edit",
        "editing_tool",
        ("python", "function_scope", "old_text_mismatch"),
        requires=("action_loop",),
        options=(("symbol_tools", True),),
    ),
    Module(
        "xml_palette",
        "tool_context",
        ("short_context", "on_demand_tools"),
        requires=("action_loop",),
        options=(("native_tools", True), ("compact_tools", True)),
    ),
    Module(
        "xml_tools",
        "tool_protocol",
        ("verified_xml_transport",),
        requires=("action_loop",),
        options=(("native_tools", True),),
    ),
    Module(
        "tensor_dataflow",
        "analysis_memory",
        ("tensor", "shape_error", "python", "operator", "framework"),
        requires=("action_loop",),
        options=(("tensor_dataflow", True),),
        input_artifacts=("current_source_observation", "caller_tensor_declarations"),
        output_artifacts=("conditional_shape_flow", "versioned_analysis_receipt"),
        effects=("read_only_analysis",),
        cost_contract=(("extra_model_calls", 0), ("max_context_tokens", 420)),
    ),
    Module(
        "operator_protocol",
        "knowledge_workflow",
        ("operator", "scan", "reduction", "contraction"),
        requires=("action_loop",),
        options=(("operator_protocol", True),),
        input_artifacts=("task_contract", "domain_profile", "actual_context_budget"),
        output_artifacts=("budgeted_method_questions", "selection_receipt"),
        effects=("context_only",),
        cost_contract=(("extra_model_calls", 0), ("max_context_tokens", 420)),
    ),
    Module(
        "literal_reads",
        "context_view",
        ("escape_sensitive_source", "literal_inspection"),
        requires=("action_loop",),
        options=(("literal_reads", True),),
    ),
    Module(
        "checked_edits",
        "workflow",
        ("bounded_independent_acceptance", "debug", "implement"),
        requires=("action_loop",),
        options=(("auto_verify_edits", True),),
    ),
)


def registry():
    return [asdict(module) for module in MODULES]


def identity():
    root = package_root()
    return hashlib.sha256(
        b"".join(
            (root / name).read_bytes()
            for name in (
                "routing/capability_router.py",
                "routing/task_tags.py",
                "routing/routing_contracts.py",
                "routing/profiles.py",
                "analysis/tensor_dataflow.py",
                "memory/source/task_source_evidence.py",
                "routing/execution_profiles.py",
                "memory/source/source_boundaries.py",
                "analysis/syntax_edits.py",
                "memory/source/source_focus.py",
                "planning/native_plan.py",
                "analysis/clang_resources.py",
                "execution/native_scope.py",
                "planning/native_prepare.py",
                "analysis/native_analysis.py",
                "analysis/code_context.py",
                "tools/native_questions.py",
                "analysis/native_macros.py",
                "runtime/json_proposals.py",
                "memory/repository_memory.py",
                "memory/source/native_working_set.py",
                "runtime/worker.py",
                "memory/source/source_questions.py",
                "planning/native_design.py",
                "memory/state_memory.py",
                "planning/state_contract.py",
                "memory/source/source_table.py",
                "runtime/shell_loop.py",
            )
        )
    ).hexdigest()


def repository_inventory(workspace, limit=256):
    """Bounded filename metadata,not a duplicate source index or a write plan."""
    pending = [(Path(workspace), 0)]
    visited = files = 0
    truncated = False
    suffixes = {}
    ignored = {".git", ".venv", "venv", "node_modules", "__pycache__", "build", ".cache"}
    code = {".py", ".c", ".cc", ".cpp", ".cu", ".h", ".hpp", ".cuh", ".rs", ".go", ".js", ".ts"}
    while pending and visited < limit:
        folder, depth = pending.pop(0)
        try:
            with os.scandir(folder) as stream:
                entries = list(islice(stream, limit - visited))
            for item in sorted(entries, key=lambda e: e.name):
                visited += 1
                if item.is_symlink():
                    continue
                if item.is_dir(follow_symlinks=False) and item.name not in ignored:
                    if depth < 4:
                        pending.append((Path(item.path), depth + 1))
                    else:
                        truncated = True
                elif item.is_file(follow_symlinks=False) and Path(item.name).suffix.lower() in code:
                    suffix = Path(item.name).suffix.lower()
                    suffixes[suffix] = suffixes.get(suffix, 0) + 1
                    files += 1
        except OSError:
            truncated = True
    return dict(
        code_files_observed=files,
        entries_observed=visited,
        scan_truncated=truncated or bool(pending) or visited >= limit,
        suffix_counts=suffixes,
        evidence="bounded filename metadata;file count does not establish semantic dependencies or modification scope",
    )


def classify(spec, workspace, *, model="", context_limit=16384):
    if type(context_limit) is not int or context_limit < 2048:
        raise ValueError("Explicit valid context limit required")
    text = spec.task[:24000].lower()
    reasons = []
    unknowns = []
    # Documentation requests do not activate CUDA/framework machinery merely
    # because they mention those systems. No repository text is an instruction.
    docs = bool(
        re.search(r"\b(readme|documentation|docstring|typo)\b|文档|注释|错别字", text)
    ) and not re.search(r"\b(implement|bug|crash|incorrect)\b|实现|崩溃|错误行为", text)
    optimize = bool(
        re.search(
            r"\b(optimize|optimise|speedup|throughput|latency|bottleneck)\b|加速|性能优化|降低延迟|提升吞吐",
            text,
        )
    )
    debug = bool(
        re.search(
            r"\b(fix|debug|bug|incorrect|crash|leak|failing|failure|regression)\b|修复|调试|泄漏|报错|回退",
            text,
        )
    )
    intent = (
        "documentation" if docs else "debug" if debug else "optimize" if optimize else "implement"
    )
    framework_facts, native_facts, scoped_native = repository_route_facts(
        workspace, spec.allowed_paths
    )
    inventory = repository_inventory(workspace)
    if docs:
        domain = "general"
        reasons.append("documentation request: domain terms alone do not activate code workflows")
    elif "operator" in spec.capabilities or re.search(
        r"\b(triton|cuda kernel|gemm|rmsnorm|layernorm|softmax)\b|算子|归约核", text
    ):
        domain = "operator"
        reasons.append("declared operator capability or computational primitive")
    elif framework_facts or re.search(
        r"\b(vllm|sglang|megatron|pytorch|infinitrain|infinilm|infinitensor|autograd|kv cache)\b|训练框架|推理框架|梯度累积",
        text,
    ):
        domain = "framework"
        reasons.extend(framework_facts or ("framework/interface requirement",))
    elif re.search(
        r"\b(sqlite|concurrency|deadlock|transaction|thread pool|state machine)\b|并发|事务|死锁|状态机|资源回收",
        text,
    ):
        domain = "system"
        reasons.append("state/concurrency/system requirement")
    else:
        domain = "general"
        reasons.append("no sufficient specialty evidence")
    explicit = [
        p for p in spec.allowed_paths if not any(c in p for c in "*?[") and not p.endswith("/")
    ]
    local = False
    if len(explicit) == len(spec.allowed_paths) == 1:
        try:
            path = (Path(workspace) / explicit[0]).resolve()
            local = (
                path.is_relative_to(Path(workspace).resolve())
                and path.is_file()
                and path.stat().st_size <= 24000
            )
        except OSError:
            pass
    cross = len(explicit) > 1 or bool(
        re.search(
            r"\b(cross.file|cross.module|multiple modules|call graph|data flow)\b|跨文件|跨模块|调用链|数据流",
            text,
        )
    )
    scope = "cross_module" if cross else "local" if local else "unknown"
    if scope == "unknown":
        unknowns.append("broad write permission does not establish architectural complexity")
    tags = [domain, intent, scope, "short_context" if context_limit <= 8192 else "long_context"]
    if scoped_native or native_facts:
        tags.append("native_source")
    # Model ID is retained as a conditioning feature; no unsupported parameter
    # count or skill is inferred from a name. Per-model evidence comes later.
    from infinicoder.routing.task_tags import extract

    multi = extract(spec, inventory, framework_facts, context_limit=context_limit)
    return dict(
        domain=domain,
        intent=intent,
        scope=scope,
        tags=tags,
        model=model,
        context_limit=context_limit,
        reasons=reasons,
        unknowns=unknowns,
        repository=inventory,
        multi_labels=multi,
        confidence="uncalibrated_rule_hypothesis",
        model_calls=0,
    )


def decide(
    spec,
    workspace,
    *,
    model="",
    context_limit=16384,
    allow_commands=False,
    enabled=(),
    disabled=(),
    policy="standard",
    performance_search=False,
):
    if policy not in (
        "standard",
        "native_candidate",
        "conditional_candidate",
        "evidence_candidate",
        "focus_candidate",
        "native_plan_candidate",
    ):
        raise ValueError("Unknown routing policy")
    if type(performance_search) is not bool:
        raise ValueError("performance_search must be bool")
    features = classify(spec, workspace, model=model, context_limit=context_limit)
    features["performance_search"] = performance_search
    axes = features["multi_labels"]["axes"]
    axes["commands"] = ["permitted" if allow_commands else "unavailable"]
    if policy in ("evidence_candidate", "focus_candidate", "native_plan_candidate"):
        from infinicoder.memory.source.task_source_evidence import collect

        source = collect(spec, workspace)
        features["source_evidence"] = source
        if "static_cross_module" in source["labels"]:
            axes["structure"] = sorted(set(axes["structure"]) | {"cross_module"})
            features["multi_labels"]["evidence"].append(
                dict(
                    axis="structure",
                    label="cross_module",
                    kind="task_referenced_static_import_dependencies",
                    source=source,
                )
            )
    if policy in (
        "conditional_candidate",
        "evidence_candidate",
        "focus_candidate",
        "native_plan_candidate",
    ):
        features["domain"] = next(
            (d for d in ("operator", "framework", "system") if d in axes["domain"]), "general"
        )
        features["reasons"].append(
            "conditional candidate uses independent domain/intent/structure labels;unknowns remain explicit"
        )
    catalog = {module.name: module for module in MODULES}
    enabled = tuple(enabled)
    disabled = set(disabled)
    if (set(enabled) | disabled) - set(catalog):
        raise ValueError("Unknown capability module")
    if set(enabled) & disabled:
        raise ValueError("Module cannot be enabled and disabled together")
    selected = []
    rejected = []
    reasons = {}

    def add(name, reason):
        if name in selected:
            return True
        if name in disabled:
            rejected.append(dict(module=name, reason="caller-disabled"))
            return False
        if (
            name == "action_loop" or "action_loop" in catalog[name].requires
        ) and not allow_commands:
            rejected.append(dict(module=name, reason="explicit command permission unavailable"))
            return False
        if name == "checked_edits":
            # Explicit opt-in only. A syntax-only repository preflight has no
            # declared Check and cannot stand in for complete task acceptance.
            # Large/multi-stage/performance tasks need a different cadence.
            bounded = (
                bool(spec.checks)
                and all(
                    type(c.timeout) in (int, float) and 0 < c.timeout <= 60 for c in spec.checks
                )
                and sum(c.timeout for c in spec.checks) <= 60
            )
            performance = features["intent"] == "optimize" or bool(
                re.search(
                    r"\b(optimize|optimise|speedup|throughput|latency|bottleneck)\b|\u52a0\u901f|\u6027\u80fd\u4f18\u5316|\u964d\u4f4e\u5ef6\u8fdf|\u63d0\u5347\u541e\u5410",
                    spec.task[:24000].lower(),
                )
            )
            if not bounded or spec.milestones or performance or performance_search:
                rejected.append(
                    dict(
                        module=name,
                        reason="requires explicit bounded caller checks,one final phase,and non-optimization task",
                    )
                )
                return False
        for dependency in catalog[name].requires:
            if not add(dependency, "dependency of " + name):
                return False
        selected.append(name)
        reasons[name] = reason
        return True

    closed = (
        features["scope"] == "local"
        and features["domain"] == "general"
        and features["intent"] in ("implement", "documentation")
    )
    if (
        policy
        in (
            "conditional_candidate",
            "evidence_candidate",
            "focus_candidate",
            "native_plan_candidate",
        )
        and axes["intent"] != ["documentation"]
        and (axes["structure"] or set(axes["intent"]) - {"implement", "documentation"})
    ):
        closed = False
    if closed or not allow_commands:
        add("source_direct", "small scoped task or no command permission")
    else:
        add(
            "action_loop",
            "interactive read/edit/verify loop without mandatory architecture planning",
        )
    memory_needed = features["scope"] == "cross_module" or (
        features["scope"] == "unknown"
        and context_limit <= 8192
        and features["repository"]["code_files_observed"] >= 16
    )
    if policy in (
        "conditional_candidate",
        "evidence_candidate",
        "focus_candidate",
        "native_plan_candidate",
    ):
        memory_needed = bool(set(axes["structure"]) & {"cross_module", "state_lifecycle"})
    if "action_loop" in selected and features["intent"] != "documentation" and memory_needed:
        add(
            "working_code",
            "cross-module question: expandable current source rather than full repository prompt",
        )
    if features["domain"] in ("operator", "framework", "system"):
        add("domain_guidance", "domain-specific invariants and development sequence")
    if policy == "native_candidate" and "action_loop" in selected:
        add(
            "xml_palette",
            "experimental policy: validated native action presentation for interactive workflow",
        )
        add("checked_edits", "experimental policy: bounded final caller checks after source edits")
    if (
        policy
        in (
            "conditional_candidate",
            "evidence_candidate",
            "focus_candidate",
            "native_plan_candidate",
        )
        and "action_loop" in selected
        and axes["intent"] != ["documentation"]
    ):
        from infinicoder.routing.routing_contracts import applicability

        for name in ("working_code", "operator_protocol", "tensor_dataflow"):
            match = applicability(name, axes)
            if match["candidate"]:
                add(
                    name,
                    "conditional experimental module;matched axes: "
                    + json.dumps(match["matched"], sort_keys=True),
                )
            elif name in ("operator_protocol", "tensor_dataflow"):
                rejected.append(
                    dict(
                        module=name,
                        reason="conditional applicability unknown or unmatched",
                        missing_axes=match["missing_axes"],
                    )
                )
    execution_profile = None
    if (
        policy in ("evidence_candidate", "focus_candidate", "native_plan_candidate")
        and "action_loop" in selected
    ):
        from infinicoder.routing.execution_profiles import choose

        execution_profile = choose(features, spec, enabled=enabled)
        for name in execution_profile["modules"]:
            add(name, "evidence-conditioned execution profile: " + execution_profile["name"])
        execution_profile["applied_modules"] = [
            n for n in execution_profile["modules"] if n in selected
        ]
        execution_profile["rejected_modules"] = [
            n for n in execution_profile["modules"] if n not in selected
        ]
    if (
        policy in ("focus_candidate", "native_plan_candidate")
        and "action_loop" in selected
        and "source_focus" not in disabled
        and features["domain"] == "general"
        and axes["intent"] == ["debug"]
    ):
        from infinicoder.memory.source.source_focus import select as focus_select
        from infinicoder.execution.workspace_tools import Workspace

        protected = [
            str(Path(p).resolve() if Path(p).is_absolute() else (Path(workspace) / p).resolve())
            for check in spec.checks
            for p in check.protected_paths
        ]
        candidate, admission = focus_select(
            spec,
            Workspace(workspace, spec.allowed_paths),
            protected_paths=protected,
            profile=features["domain"],
            auto_verify=True,
            optimization=performance_search,
        )
        features["source_focus_admission"] = admission
        if candidate:
            add(
                "source_focus",
                "current source supports one provisional body proposal;original fallback and budget retained",
            )
    if (
        policy == "native_plan_candidate"
        and "action_loop" in selected
        and "native_source" in features["tags"]
    ):
        add(
            "native_analysis",
            "Defer native semantics until a source path is requested;bounded parser work shares the original time budget",
        )
    if (
        policy == "native_plan_candidate"
        and "action_loop" in selected
        and features["domain"] in ("framework", "operator")
        and "native_source" in features["tags"]
        and not performance_search
        and not set(axes["intent"]) & {"optimize", "refactor", "documentation"}
    ):
        add(
            "native_plan",
            "Provisional source-grounded native design/code/check workflow;actual Clang admission happens in Runtime",
        )
    for name in enabled:
        if not add(name, "caller-selected experimental module"):
            raise ValueError("Explicit module prerequisites unavailable: " + name)
    if "source_direct" in selected and "action_loop" in selected:
        selected.remove("source_direct")
        reasons.pop("source_direct", None)
    if not {"source_direct", "action_loop"} & set(selected):
        if not add("source_direct", "fallback after optional workflow rejection"):
            raise ValueError("No permitted execution workflow remains")
    from infinicoder.routing.routing_contracts import compose

    composition = compose(selected, catalog, axes)
    options = dict(
        code_memory=True,
        development_workflows=False,
        grounded_planning=False,
        task_memory=False,
        adaptive_replanning=False,
        repair_memory=False,
        diagnostic_planning=False,
        design_contracts=False,
        design_probes=False,
        shell_loop=False,
        auto_verify_commands=False,
        observed_edits=False,
        question_memory=False,
        native_tools=False,
        flow_memory=False,
        structured_tools=False,
        compact_tools=False,
        source_sync=False,
        source_archive=False,
        native_read_batch=False,
        problem_workflow=False,
        format_repair=False,
        failure_routing=False,
        compatible_tools=False,
        symbol_tools=False,
        literal_reads=False,
        operator_protocol=False,
        tensor_dataflow=False,
        decision_driver="runtime",
        source_boundaries=False,
        syntax_edits=False,
        source_focus=False,
        inspection_recovery=False,
        native_plan=False,
        native_state_contract=False,
        native_source_table=False,
        native_handoff_check=False,
        design_revision_memory=False,
        design_delta=False,
        example_diagnosis=False,
        binding_candidates=False,
        native_index_policy="eager",
    )
    for name in selected:
        for key, value in catalog[name].options:
            options[key] = value
    domain = features["domain"]
    options["profile"] = "framework" if domain == "system" else domain
    return dict(
        version=1,
        router_identity=identity(),
        policy=policy,
        features=features,
        modules=selected,
        module_reasons=reasons,
        rejected=rejected,
        runtime_options=options,
        composition=composition,
        execution_profile=execution_profile,
        evidence_status="experimental_no_coding_gain_established",
        permission_policy="Caller permissions,write scope,acceptance and budgets are never enlarged",
        fallback="Explicit modules can override routing; new contracts require a new route; failed routes remain recorded",
    )


def contract_identity(spec):
    return hashlib.sha256(
        json.dumps(asdict(spec), sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


route_capabilities = decide
