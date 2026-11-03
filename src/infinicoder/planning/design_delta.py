"""Explicit reference-only amendment of a fully displayed model hypothesis."""

from infinicoder._resources import package_root

import copy, hashlib, json
from infinicoder.memory.revision_memory import current, fingerprint

GUIDANCE = "When retaining exactly ALL displayed earlier semantic claims,return only reuse_design,read_ids,state_targets. Set reuse_design to retained_design_hypotheses.id and select current visible references/fields. This changes context selection only;all invariant,steps and state effects are reused unchanged as unverified hypotheses. To reconsider ANY semantics,return the ordinary complete design instead. Neither response certifies code."


def identity():
    root = package_root()
    return hashlib.sha256(
        b"".join(
            (root / n).read_bytes()
            for n in (
                "planning/design_delta.py",
                "planning/native_design.py",
                "memory/revision_memory.py",
                "planning/native_plan.py",
                "runtime/engine.py",
            )
        )
    ).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def available(plan, spec, state, shown):
    memory = current(plan, spec, state)
    return bool(
        plan.get("design_delta_enabled")
        and not state.get("native_plan_design")
        and memory
        and shown
        and shown.get("id") == memory["id"]
        and not shown.get("omitted_claims")
        and shown.get("claims") == memory["claims"]
    )


def offer(plan, spec, state, shown, reference):
    if not available(plan, spec, state, shown):
        state.pop("native_plan_delta_offer", None)
        return
    state["native_plan_delta_offer"] = dict(
        id=shown["id"],
        source_fingerprint=fingerprint(plan),
        reference_sha256=digest(reference),
        claims_sha256=digest(shown["claims"]),
        provider=identity(),
    )


def materialize(value, plan, spec, state):
    """No source effect;the original reference/admission checks run afterwards."""
    memory = current(plan, spec, state)
    shown = state.get("native_plan_delta_offer")
    if (
        not plan.get("design_delta_enabled")
        or state.get("native_plan_design")
        or not memory
        or not shown
        or set(value) != {"reuse_design", "read_ids", "state_targets"}
        or value["reuse_design"] != memory["id"]
        or shown
        != dict(
            id=memory["id"],
            source_fingerprint=fingerprint(plan),
            reference_sha256=digest(state["native_plan_reference_contract"]),
            claims_sha256=digest(memory["claims"]),
            provider=identity(),
        )
    ):
        raise ValueError(
            "Design delta requires a current,fully displayed,unchanged hypothesis and reference offer"
        )
    claims = memory["claims"]
    invariants = [c["value"] for c in claims if c["kind"] == "invariant"]
    steps = [c["value"] for c in claims if c["kind"] == "step"]
    effects = [c["value"] for c in claims if c["kind"] == "state_effects"]
    if len(invariants) != 1 or not steps or len(effects) > 1:
        raise ValueError("Incomplete design hypothesis artifact")
    result = dict(
        read_ids=copy.deepcopy(value["read_ids"]),
        state_targets=copy.deepcopy(value["state_targets"]),
        invariant=invariants[0],
        steps=copy.deepcopy(steps),
    )
    if plan.get("state_contract_enabled"):
        result["state_effects"] = copy.deepcopy(effects[0] if effects else [])
    receipt = dict(
        hypothesis_id=memory["id"],
        claims_sha256=digest(claims),
        source_fingerprint=fingerprint(plan),
        call=state.get("model_calls"),
        read_ids=copy.deepcopy(value["read_ids"]),
        state_targets=copy.deepcopy(value["state_targets"]),
        semantics_reused_exactly=True,
        semantic_proof=False,
        status="awaiting_original_reference_and_capacity_checks",
        provider=identity(),
    )
    return result, receipt
