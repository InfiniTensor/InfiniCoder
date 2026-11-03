"""Finite collaboration strategies over existing grounded focus and actions.

The caller supplies the namespace and public checks. This profile changes
strategy configuration, not TaskSpec, the work graph, permissions or budgets.
"""

from infinicoder.collaboration.collaboration import BranchContext
from infinicoder.collaboration.collaboration_namespace import NamespaceCollaborationRuntime


def create_focus_collaboration_runtime(
    context,
    *,
    namespace_builder,
    request_timeout,
    allow_commands=False,
    execution_profile=None,
    physical_context_budget=True,
):
    """A/B use grounded focus; C pairs it with ordinary read/edit/actions.

    A model must inspect real current source before selecting focus_edit. The
    initial source packet is not registered as an inspection certificate.
    Missing/ambiguous Clang evidence declines focus through existing rules.
    """
    if not isinstance(context, BranchContext):
        raise TypeError("caller-created BranchContext required")
    if type(physical_context_budget) is not bool:
        raise TypeError("physical_context_budget must be boolean")
    runtime = NamespaceCollaborationRuntime(
        context.workspace,
        context.client,
        context.state_dir,
        namespace_builder=namespace_builder,
        model_profile=context.model_profile,
        max_steps=context.call_quota,
        max_model_calls=context.call_quota,
        time_limit=max(0.001, context.owner.remaining()),
        request_timeout=request_timeout,
        allow_commands=allow_commands,
        execution_profile=execution_profile,
        automatic_workers=context.automatic_workers,
        source_fallback=context.source_fallback,
        grounded_planning=context.automatic_workers,
        auto_verify_edits=True,
        native_worker=False,
        qualified_worker=False,
        native_index_policy="on_demand",
        profile="general",
        domain_compiler=False,
        worker_format="json_source",
    )
    runtime.collaboration_strategy = (
        "grounded_focus_v1" if context.automatic_workers else "ordinary_actions_v1"
    )
    runtime.physical_context_budget = physical_context_budget
    runtime.assembler.physical_alignment = physical_context_budget
    return runtime
