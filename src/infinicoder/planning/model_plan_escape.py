"""A model's implementation plan is revocable; the caller contract is not.

One transition per caller version, after a real ordinary action has settled.
No rejected edit, model request, check, budget or permission is replayed here.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from infinicoder.analysis.task_compiler import compile_task
from infinicoder.planning.task_contract import TaskSpec
from infinicoder.execution.workspace_tools import Workspace

POLICY = "model_plan_escape_v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def identity():
    return dict(policy=POLICY, sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def bind(runtime, state, caller):
    value = asdict(caller)
    contract = digest(value)
    old = state.get("model_plan_escape")
    if old and old["contract_sha256"] == contract:
        return
    if old:
        state.setdefault("model_plan_escape_history", []).append(old)
    state["model_plan_escape"] = dict(
        contract_sha256=contract,
        caller=value,
        applicable=not caller.milestones
        and not runtime._explicit_worker_path
        and not runtime._configured_worker_symbol,
        consumed=False,
        origin=None,
    )


def active(state):
    return bool(state.get("model_plan_escape", {}).get("consumed"))


def eligible(state):
    lease = state.get("model_plan_escape", {})
    return (
        lease.get("applicable")
        and not lease.get("consumed")
        and lease.get("origin") == "model"
        and state.get("plan_milestones")
        and state.get("status") == "running"
    )


def scope_refusal(runtime, state, workspace, unit, args, protected):
    """Called only inside the real phase Workspace.path rejection handler."""
    if not eligible(state):
        return
    try:
        caller = TaskSpec.from_dict(state["model_plan_escape"]["caller"])
        path = Workspace(workspace.root, caller.allowed_paths).path(args["path"], write=True)
        if str(path.resolve()) in protected or not path.is_file():
            return
        current = hashlib.sha256(path.read_bytes()).hexdigest()
        if args.get("expected_hash") != current:
            return
        # Recheck the specific phase restriction; other path errors do not qualify.
        if not unit.write_set:
            return
        try:
            Workspace(workspace.root, unit.write_set).path(args["path"], write=True)
        except ValueError:
            pass
        else:
            return
        state["model_plan_escape_pending"] = dict(
            kind="phase_scope_refusal",
            path=args["path"],
            source_sha256=current,
            revision=workspace.revision(),
            phase=state["milestone"],
            edit_applied=False,
        )
    except (ValueError, OSError, KeyError, TypeError):
        return


def public_rejection(
    runtime, state, workspace, check_ids, require_artifacts, evidence, live_before
):
    if not eligible(state) or not require_artifacts:
        return
    if state["milestone"] != len(runtime._graph.units) - 1:
        return
    caller = TaskSpec.from_dict(state["model_plan_escape"]["caller"])
    ids = {c.id for c in caller.checks}
    if (
        not ids
        or set(check_ids) != ids
        or live_before is None
        or observed_revision(workspace) != live_before
    ):
        return
    executed = [e for e in evidence if e.get("executed_argv") and type(e.get("exit_code")) is int]
    if {e.get("check_id") for e in executed} != ids:
        return
    outcomes = getattr(runtime._acceptance, "last_process_outcomes", {})
    if set(outcomes) != ids or not all(
        o.get("completed_normally") is True for o in outcomes.values()
    ):
        return
    if any(
        e.get("error") or e.get("failure_origin") or e.get("check_id") not in ids for e in evidence
    ):
        return
    if not all(
        e.get("revision")
        and e["revision"] == e.get("snapshot_after")
        and e.get("checker_files") == state.get("checker_files")
        for e in executed
    ):
        return
    state["model_plan_escape_pending"] = dict(
        kind="last_stage_public_rejection",
        revision=live_before,
        phase=state["milestone"],
        check_ids=sorted(ids),
        evidence_sha256=digest(evidence),
    )


def pending(state):
    return eligible(state) and bool(state.get("model_plan_escape_pending"))


def observed_revision(workspace):
    try:
        return workspace.revision()
    except (OSError, ValueError):
        return None


def settle(runtime, state, workspace, remaining, reserve):
    proof = state.pop("model_plan_escape_pending", None)
    if not proof or not eligible(state):
        return None
    lease = state["model_plan_escape"]
    try:
        caller = TaskSpec.from_dict(lease["caller"])
        original = compile_task(caller)
        historical_graph = asdict(runtime._graph)
        historical_hash = digest(historical_graph)
        original_hash = digest(asdict(original))
        current = observed_revision(workspace)
        controlled = (runtime.store.path / "control.json").exists()
    except (OSError, ValueError):
        return None
    # Observations and graph construction can be expensive. Recheck control and
    # budget after them, immediately before retiring the current implementation.
    if (
        not eligible(state)
        or controlled
        or current is None
        or current != proof["revision"]
        or remaining() <= reserve + 1
        or state["model_calls"] >= runtime.max_model_calls
        or state["steps"] >= runtime.max_steps
    ):
        return None
    lease.update(
        consumed=True,
        trigger=proof,
        call=state["model_calls"],
        step=state["steps"],
        prior_graph=historical_graph,
        prior_plan=state["plan_milestones"],
        prior_phase=state["milestone"],
    )
    for key in (
        "plan_milestones",
        "source_plan_certificate",
        "phase_execution",
        "pending_replan",
        "phase_context",
        "edit_focus",
        "fallback_path",
        "source_scope",
        "source_inspections",
        "planning_feedback",
        "planning_inspection_required",
        "planning_inspection_gate",
    ):
        state.pop(key, None)
    state.update(milestone=0, planning_required=False, evidence=[])
    runtime._graph = original
    state["progress_hint"] = (
        "The model-generated phase plan has been retired after an observed scope refusal "
        "or final public rejection. Use ordinary source inspection, edits and verification "
        "under the complete original caller contract. The rejected edit was not applied. "
        "All previous failures, costs and original final checks remain binding. "
        "Do not submit another mandatory plan; repair the implementation directly."
    )
    runtime.store.event(
        "model_plan_escaped",
        dict(
            contract_sha256=lease["contract_sha256"],
            trigger=proof,
            call=state["model_calls"],
            step=state["steps"],
            historical_graph_sha256=historical_hash,
            original_graph_sha256=original_hash,
            budgets_reset=False,
            rejected_action_replayed=False,
            acceptance_proof=False,
        ),
    )
    return caller
