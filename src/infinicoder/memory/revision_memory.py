"""Versioned model hypotheses across representation-only design rejection.

Neither reference binding nor context admission proves a semantic claim.
Keep complete claims in the session;expose only whole claims within the
caller's token estimator budget. No model,source edit or acceptance occurs.
"""

from infinicoder._resources import package_root

import copy
import hashlib
import json
from infinicoder.memory.repair_memory import signature

LIMIT = 512
GUIDANCE = "Capacity rejection concerns implementation context size,not a semantic verdict. These earlier model claims remain unverified. Preserve applicable semantics when narrowing references,or reconsider them against current source;the original caller contract and final checks remain authoritative."


def identity():
    root = package_root()
    return hashlib.sha256(
        b"".join(
            (root / n).read_bytes()
            for n in ("memory/revision_memory.py", "planning/native_plan.py", "runtime/engine.py")
        )
    ).hexdigest()


def fingerprint(plan):
    scope = plan["scope"]
    value = dict(
        requirements=plan["requirements"],
        target=plan["path"],
        symbol=scope["qualname"],
        sources=dict(scope["evidence_source_versions"], **plan["lookup_versions"]),
        compile_database=[scope["compile_database_path"], scope["compile_database_sha256"]],
        external_compile_versions=scope["external_compile_versions"],
        resources=scope.get("resource_configuration"),
    )
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def capture(plan, spec, state, design):
    """Only caller supplies an already reference-bound,rejected hypothesis."""
    if not plan.get("revision_memory_enabled"):
        return
    claims = [dict(id="H0", kind="invariant", value=design["invariant"])]
    claims += [
        dict(id="H" + str(i + 1), kind="step", value=s) for i, s in enumerate(design["steps"])
    ]
    if design.get("state_effects"):
        claims.append(
            dict(
                id="H" + str(len(claims)),
                kind="state_effects",
                value=copy.deepcopy(design["state_effects"]),
            )
        )
    value = dict(
        kind="unverified_model_hypothesis",
        requirements=signature(spec),
        source_fingerprint=fingerprint(plan),
        provider=identity(),
        call=state.get("model_calls"),
        claims=claims,
        rejection_dimension="implementation_context_capacity",
        semantic_verdict="unknown",
        acceptance_proof=False,
    )
    value["id"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    state["design_revision_memory"] = value


def current(plan, spec, state):
    memory = state.get("design_revision_memory")
    if not memory or not plan.get("revision_memory_enabled"):
        return None
    if (
        memory.get("requirements") != signature(spec)
        or memory.get("source_fingerprint") != fingerprint(plan)
        or memory.get("provider") != identity()
    ):
        return None
    original = dict(memory)
    original.pop("id", None)
    if hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest() != memory.get(
        "id"
    ):
        return None
    return memory


def view(plan, spec, state, assembler):
    memory = current(plan, spec, state)
    if not memory:
        return None

    # Runtime independently checks live prerequisites before native context.
    # IDs only identify claims in this session;they are not source evidence IDs.
    def packet(rows):
        return dict(
            id=memory["id"],
            kind=memory["kind"],
            call=memory["call"],
            rejection_dimension=memory["rejection_dimension"],
            semantic_verdict="unknown",
            instruction=GUIDANCE,
            claims=rows,
            omitted_claims=[c["id"] for c in memory["claims"] if c not in rows],
            acceptance_proof=False,
        )

    def tokens(value):
        from infinicoder.memory.source.source_table import render

        return assembler.estimator.estimate(
            render(value, compact=bool(plan.get("source_table_enabled")))
        )

    rows = []
    for claim in memory["claims"]:
        if tokens(packet([*rows, claim])) <= LIMIT:
            rows.append(copy.deepcopy(claim))
    value = packet(rows)
    # Even pathological claim inventories never enlarge the declared budget.
    if tokens(value) > LIMIT:
        return None
    return value
