"""Experimental bundles of heterogeneous capabilities, not a learned MoE gate."""

PROFILES = {
    "python_dependency_repair": dict(
        modules=["working_code", "xml_palette", "checked_edits"],
        basis="Task-referenced Python import dependencies and nonperformance debug;restore native inspect/edit workflow.",
    ),
    "interactive_sdk": dict(
        modules=["text_driver"],
        basis="General interactive fallback using pinned mini-SWE query/actions;domain modules compose independently.",
    ),
}


def choose(features, spec, *, enabled=()):
    axes = features["multi_labels"]["axes"]
    source = features["source_evidence"]
    explicit_protocol = bool(
        set(enabled)
        & {"text_driver", "task_first_context", "xml_tools", "xml_palette", "protocol_compat"}
    )
    localized = (
        "static_cross_module" in source["labels"]
        and axes["domain"] == ["general"]
        and "debug" in axes["intent"]
        and not set(axes["intent"]) & {"implement", "optimize", "refactor"}
        and not spec.milestones
        and not features["performance_search"]
    )
    name = "python_dependency_repair" if localized else "interactive_sdk"
    profile = PROFILES[name]
    return dict(
        name=name,
        modules=[] if explicit_protocol else list(profile["modules"]),
        reason="Caller-selected protocol takes precedence;source memory remains independently composable."
        if explicit_protocol
        else profile["basis"],
        evidence="development_hypothesis;historical workflow observations are not same-source causal evidence",
        learned_gate=False,
        model_specific_calibration=False,
        extra_model_calls=0,
    )
