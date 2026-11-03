"""One bounded live-patch episode after an actual whole-source output limit.

This module has no model, editor, planner, or checker. Only the existing worker
may select a file; only the original editor may apply its complete proposal.
"""

from infinicoder._resources import package_root

import ast
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from infinicoder.verification.check_identity import checker_files
from infinicoder.memory.repair_memory import signature
from infinicoder.execution.workspace_tools import Workspace

POLICY = "bounded_patch_after_length_v1"
MAX_ATTEMPTS = 3


class SourceOutputClosed(Exception):
    """No request in this turn; return to the original worker selector."""


class SourceOutputHalt(Exception):
    """Existing user control or deadline stopped dispatch/edit, not model error."""


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def identity():
    root = package_root()
    return hashlib.sha256(
        b"".join(
            (root / name).read_bytes()
            for name in (
                "memory/source/source_output.py",
                "runtime/engine.py",
                "runtime/worker.py",
                "runtime/work_packet.py",
                "execution/workspace_tools.py",
            )
        )
    ).hexdigest()


def _unit(runtime, state):
    return runtime._graph.unit(runtime._graph.order[state["milestone"]])


def _source(runtime, spec, state, workspace, path, unit):
    if unit.kind != "implement":
        raise ValueError("bounded live patch requires an implementation phase")
    if Path(path).suffix != ".py" or runtime._worker_scope:
        raise ValueError("bounded live patch requires a whole existing Python worker")
    target = workspace.path(path, write=True)
    Workspace(workspace.root, unit.write_set or spec.allowed_paths).path(path, write=True)
    if not target.is_file() or str(target.resolve()) in state.get("checker_files", {}):
        raise ValueError("bounded live patch target is missing or protected")
    data = target.read_bytes()
    if not data or len(data) > 2_000_000:
        raise ValueError("bounded live patch source is not a bounded nonempty file")
    ast.parse(data.decode("utf-8"), filename=path)
    current_checkers = checker_files(spec, workspace)
    if current_checkers != state.get("checker_files", {}):
        raise ValueError("caller checker identity changed")
    return data, digest(current_checkers)


def stop(runtime, state, reason):
    episode = state.get("source_output_episode")
    if not episode or episode.get("status") != "active":
        return
    episode.update(status="closed", reason=reason)
    # Automatic source dispatch may return to existing actions in the same
    # phase. An explicit caller worker retains its original dispatch contract.
    if not episode["caller_worker"]:
        blocked = state.setdefault("source_worker_blocked_paths", [])
        if episode["path"] not in blocked:
            blocked.append(episode["path"])
        state.pop("fallback_path", None)
    episode["fallback"] = (
        "original_caller_source_contract"
        if episode["caller_worker"]
        else "original_ordinary_actions"
    )
    runtime.store.event("source_output_closed", deepcopy(episode))


def capture(
    runtime,
    spec,
    state,
    workspace,
    path,
    *,
    finish_reason,
    response_format,
    source_before,
    origin="model",
    staged=False,
):
    """Observe an existing unapplied output; never request one to trigger this."""
    if state.get("source_output_episode") is not None or origin != "model" or staged:
        return
    if finish_reason not in ("length", "max_tokens") or response_format not in (
        "source",
        "json_source",
        "json_lines",
    ):
        return
    try:
        unit = _unit(runtime, state)
        data, checkers = _source(runtime, spec, state, workspace, path, unit)
        revision = workspace.revision()
        source_hash = hashlib.sha256(data).hexdigest()
        if revision != source_before or source_hash != runtime._worker_hash:
            return
    except (ValueError, OSError, UnicodeError, SyntaxError, IndexError):
        return
    episode = dict(
        policy=POLICY,
        status="active",
        path=path,
        trigger_call=state.get("model_calls", 0),
        trigger_finish_reason=finish_reason,
        original_format=response_format,
        requirements=signature(spec),
        phase=state["milestone"],
        unit_identity=digest(asdict(unit)),
        checker_identity=checkers,
        source_sha256=source_hash,
        workspace_revision=revision,
        other_sources_revision=workspace.revision([path]),
        attempts=[],
        applied=0,
        caller_worker=bool(
            runtime._explicit_worker_path or unit.worker_path and not state.get("plan_milestones")
        ),
        acceptance_proof=False,
        original_budget_retained=True,
    )
    episode["episode_id"] = digest(episode)
    state["source_output_episode"] = episode
    runtime.store.event("source_output_episode", deepcopy(episode))


def _current(runtime, spec, state, workspace, episode, path):
    unit = _unit(runtime, state)
    data, checkers = _source(runtime, spec, state, workspace, path, unit)
    return bool(
        path == episode["path"]
        and signature(spec) == episode["requirements"]
        and state["milestone"] == episode["phase"]
        and digest(asdict(unit)) == episode["unit_identity"]
        and checkers == episode["checker_identity"]
        and hashlib.sha256(data).hexdigest() == episode["source_sha256"]
        and workspace.revision() == episode["workspace_revision"]
        and workspace.revision([path]) == episode["other_sources_revision"]
    )


def select(runtime, spec, state, workspace, path, *, candidate_format=None):
    """Consumes the original worker selection, never chooses a different file."""
    episode = state.get("source_output_episode")
    if not episode or episode.get("status") != "active":
        return None
    if len(episode["attempts"]) >= MAX_ATTEMPTS:
        stop(runtime, state, "patch_attempt_limit")
        raise SourceOutputClosed("Patch episode closed;reselect through original dispatch")
    try:
        valid = candidate_format is None and _current(
            runtime, spec, state, workspace, episode, path
        )
    except (ValueError, OSError, UnicodeError, SyntaxError, IndexError):
        valid = False
    if not valid:
        stop(runtime, state, "worker_source_phase_or_contract_changed")
        raise SourceOutputClosed("Patch lease closed;reselect through original dispatch")
    return dict(
        episode_id=episode["episode_id"],
        path=path,
        source_sha256=episode["source_sha256"],
        workspace_revision=episode["workspace_revision"],
        unit_identity=episode["unit_identity"],
        attempts_used=len(episode["attempts"]),
        maximum_attempts=MAX_ATTEMPTS,
        reason="actual_complete_source_output_truncated",
        acceptance_proof=False,
        payload_fit="unknown_until_response;original_output_cap_retained",
    )


def validate(runtime, spec, state, workspace, remaining, verification_reserve):
    episode = state.get("source_output_episode")
    selection = runtime._source_output_selection
    try:
        valid = bool(
            episode
            and episode.get("status") == "active"
            and selection
            and selection["episode_id"] == episode["episode_id"]
            and selection["source_sha256"] == episode["source_sha256"]
            and selection["workspace_revision"] == episode["workspace_revision"]
            and _current(runtime, spec, state, workspace, episode, selection["path"])
        )
    except (ValueError, OSError, UnicodeError, SyntaxError, KeyError, IndexError):
        valid = False
    # Control/deadline take precedence even if the same scan found stale code.
    boundary(runtime, state, remaining, verification_reserve, "after_source_output_validation")
    if not valid:
        stop(runtime, state, "source_or_contract_lease_invalidated")
        raise ValueError(
            "Bounded patch source/phase/checker lease changed;proposal was not applied"
        )


def boundary(runtime, state, remaining, verification_reserve, where):
    """Recheck the original clock/control after expensive hashes, before work."""
    control = runtime.store.take_control()
    if control:
        rejected(runtime, state, "not_applied_user_" + control)
        stop(runtime, state, "user_" + control)
        state.update(
            status="paused" if control == "pause" else "cancelled", reason="user_" + control
        )
        runtime.store.event("control", dict(action=control, boundary=where))
        raise SourceOutputHalt("user_" + control)
    available = remaining() - verification_reserve
    if available <= 0:
        rejected(runtime, state, "deadline_" + where)
        stop(runtime, state, "deadline_" + where)
        # Stop coding while retaining the original single reserved acceptance.
        # The original finalizer decides success or exhausted budget; user
        # control above is terminal and must never enter that finalizer.
        state["source_output_coding_stopped"] = where
        raise SourceOutputHalt("source_output_deadline")
    return available


def claim(runtime, spec, state, workspace, remaining, verification_reserve):
    validate(runtime, spec, state, workspace, remaining, verification_reserve)
    episode = state["source_output_episode"]
    if len(episode["attempts"]) >= MAX_ATTEMPTS:
        raise ValueError("Bounded patch episode used all three request attempts")
    attempt = dict(
        call=state.get("model_calls", 0) + 1,
        source_sha256=episode["source_sha256"],
        workspace_revision=episode["workspace_revision"],
        outcome="requested",
    )
    episode["attempts"].append(attempt)
    runtime._source_output_active = True
    runtime.store.event(
        "source_output_patch_request", dict(episode_id=episode["episode_id"], **attempt)
    )
    runtime.store.save(state)


def applied(runtime, spec, state, workspace, result):
    episode = state["source_output_episode"]
    attempt = episode["attempts"][-1]
    attempt.update(outcome="applied", before=result["before"], after=result["after"])
    episode["applied"] += 1
    runtime._source_output_committed = True
    runtime.store.event(
        "source_output_patch_applied", dict(episode_id=episode["episode_id"], **attempt)
    )
    if (
        result["before"] != episode["source_sha256"]
        or workspace.revision([episode["path"]]) != episode["other_sources_revision"]
        or hashlib.sha256(workspace.path(episode["path"]).read_bytes()).hexdigest()
        != result["after"]
    ):
        # The original atomic edit did occur. Do not label it unapplied or
        # overwrite another process's edits while invalidating its lease.
        stop(runtime, state, "concurrent_dependency_change_after_committed_edit")
        return
    episode["source_sha256"] = result["after"]
    episode["workspace_revision"] = workspace.revision()
    if len(episode["attempts"]) >= MAX_ATTEMPTS:
        stop(runtime, state, "patch_attempt_limit")


def rejected(runtime, state, outcome="rejected_without_write"):
    episode = state.get("source_output_episode")
    if (
        not runtime._source_output_active
        or not episode
        or not episode["attempts"]
        or runtime._source_output_committed
    ):
        return
    episode["attempts"][-1]["outcome"] = outcome
    runtime.store.event(
        "source_output_patch_rejected",
        dict(episode_id=episode["episode_id"], **episode["attempts"][-1]),
    )
    if len(episode["attempts"]) >= MAX_ATTEMPTS:
        stop(runtime, state, "patch_attempt_limit")


def settle_non_edit(runtime, state, action):
    episode = state.get("source_output_episode")
    if not episode or not runtime._source_output_active or runtime._source_output_committed:
        return
    attempt = episode["attempts"][-1]
    attempt.update(outcome="non_edit_action", action=action)
    runtime.store.event(
        "source_output_patch_non_edit", dict(episode_id=episode["episode_id"], **attempt)
    )
    if len(episode["attempts"]) >= MAX_ATTEMPTS:
        stop(runtime, state, "patch_attempt_limit")
