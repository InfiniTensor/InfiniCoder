"""Route a real phase refusal to the existing model replanner, never to an edit."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from infinicoder.execution.workspace_tools import Workspace

POLICY = "authorized_owner_v1"


def identity():
    return dict(
        policy=POLICY, implementation=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    )


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _control_pending(runtime):
    # Leave consumption and terminal handling to the original Runtime loop.
    return (runtime.store.path / "control.json").exists()


class PhaseScopeRejection(ValueError):
    def __init__(self, record):
        super().__init__(
            "Current model phase excludes this caller-authorized file; a validated replacement plan is required before editing"
        )
        self.record = record


def prepare(runtime, spec, state, workspace, unit, arguments, protected_paths=()):
    """Called only after the original phase Workspace.path(write=True) rejects."""
    if (
        not runtime.phase_replan_policy
        or runtime._phase_caller_milestones
        or runtime._explicit_worker_path
        or not state.get("plan_milestones")
        or state.get("planning_required")
        or state.get("replans", 0) >= 3
    ):
        return None
    if spec.milestones != state["plan_milestones"] or unit.kind != "implement":
        return None
    if not isinstance(arguments, dict) or not isinstance(arguments.get("path"), str):
        return None
    try:
        path = workspace.path(arguments["path"], write=True)
        if not path.is_file() or str(path.resolve()) in protected_paths:
            return None
        relative = path.relative_to(workspace.root).as_posix()
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if arguments.get("expected_hash") != source_hash:
            return None
        # Confirm this is the real current graph's rejected phase, not a stale
        # unit or a path-format error. Unknown/ambiguous ownership stays refused.
        phase = state["milestone"]
        graph = runtime._graph
        if graph.unit(graph.order[phase]) != unit:
            return None
        try:
            Workspace(workspace.root, unit.write_set).path(relative, write=True)
        except ValueError:
            pass
        else:
            return None
        owners = []
        for i, name in enumerate(graph.order):
            other = graph.unit(name)
            if other.kind != "implement" or not other.write_set:
                continue
            try:
                Workspace(workspace.root, other.write_set).path(relative, write=True)
            except ValueError:
                continue
            owners.append(dict(phase=i, unit_id=name))
        if len(owners) != 1 or owners[0]["phase"] == phase:
            return None
        from infinicoder.memory.repair_memory import signature

        record = dict(
            kind="caller_authorized_model_phase_refusal",
            path=relative,
            source_sha256=source_hash,
            workspace_revision=workspace.revision(),
            requirements=signature(spec),
            plan_sha256=_digest(state["plan_milestones"]),
            graph_sha256=_digest(asdict(graph)),
            rejected_phase=phase,
            rejected_unit=unit.id,
            owner=owners[0],
            caller_authorized=True,
            edit_applied=False,
        )
        record["id"] = _digest(record)
        if any(row["id"] == record["id"] for row in state.get("phase_scope_replans", [])):
            return None
        return record
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None


def request(runtime, spec, state, workspace, record, remaining, reserve):
    """Recheck the refusal lease, then reuse the existing replan invalidation."""
    from infinicoder.memory.repair_memory import signature

    if (
        runtime._phase_caller_milestones
        or runtime._explicit_worker_path
        or state.get("planning_required")
        or state.get("replans", 0) >= 3
        or remaining() <= reserve
        or _control_pending(runtime)
    ):
        return None
    try:
        path = workspace.path(record["path"], write=True)
        valid = (
            path.is_file()
            and hashlib.sha256(path.read_bytes()).hexdigest() == record["source_sha256"]
            and workspace.revision() == record["workspace_revision"]
            and signature(spec) == record["requirements"]
            and _digest(state.get("plan_milestones")) == record["plan_sha256"]
            and _digest(asdict(runtime._graph)) == record["graph_sha256"]
            and state["milestone"] == record["rejected_phase"]
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not valid or remaining() <= reserve or _control_pending(runtime):
        return None
    # Original begin_replan invalidates phase evidence and routes to the model.
    # Preserve existing stagnation/protocol counters; no new time or call budget.
    counters = {
        key: state.get(key) for key in ("failures", "no_progress", "plan_protocol_failures")
    }
    event = runtime._request_replan(
        spec,
        state,
        workspace,
        "A current-source edit was refused by the model phase, although the original task authorizes it and another model unit owns it",
        [record["path"]],
    )
    for key, value in counters.items():
        if value is not None:
            state[key] = value
    state["phase_scope_replans"] = [*state.get("phase_scope_replans", []), record][-12:]
    state["phase_scope_pending_id"] = record["id"]
    state["progress_hint"] += (
        " The refused edit was not executed. Reinspect the owning module and propose a replacement plan; "
        "the rejected source proposal is not replayed, and caller permissions/checks remain unchanged."
    )
    runtime.store.event(
        "phase_scope_replan_requested",
        dict(
            refusal=record,
            transition=event,
            edit_applied=False,
            model_selects_replacement_plan=True,
            original_budget_retained=True,
        ),
    )
    return event


def view(spec, state, workspace):
    """Only the original refusal, as historical plan evidence, not a write lease."""
    from infinicoder.memory.repair_memory import signature

    pending = state.get("phase_scope_pending_id")
    record = next(
        (r for r in reversed(state.get("phase_scope_replans", [])) if r["id"] == pending), None
    )
    if not record or signature(spec) != record["requirements"]:
        return None
    try:
        current = hashlib.sha256(workspace.path(record["path"]).read_bytes()).hexdigest()
    except (OSError, ValueError):
        current = None
    return dict(
        origin="original_controller_phase_refusal",
        path=record["path"],
        refused_phase=record["rejected_phase"],
        previous_plan_owner=record["owner"],
        observed_source_sha256=record["source_sha256"],
        current_source_sha256=current,
        edit_applied=False,
        caller_authorized_when_refused=True,
        rule="Inspect current source and submit a validated replacement plan. This old model-plan ownership is not a semantic diagnosis or permission to replay the rejected edit.",
    )
