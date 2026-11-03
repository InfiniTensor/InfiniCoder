"""Once-only caller verification after an observed source-reading stall.

This produces controller-origin diagnostic evidence, never an assistant action,
candidate, hypothesis, phase completion or reusable acceptance certificate.
"""

from infinicoder._resources import package_root

import hashlib
import time
from infinicoder.runtime.problem_workflow import contract, digest
from infinicoder.memory.revision_feedback import revision


def identity():
    root = package_root()
    return hashlib.sha256(
        b"".join(
            (root / name).read_bytes()
            for name in (
                "runtime/stalled_feedback.py",
                "runtime/engine.py",
                "runtime/shell_loop.py",
                "runtime/problem_decision.py",
                "memory/question_memory.py",
                "memory/inspection_ledger.py",
                "runtime/feedback.py",
            )
        )
    ).hexdigest()


def lease(state, spec, workspace, check_ids):
    return dict(
        contract_ref=contract(state, spec),
        phase=state["milestone"],
        source_revision=revision(spec, workspace),
        check_ids=list(check_ids),
    )


def note_check(runtime, state, spec, workspace, check_ids):
    """Observe original verify dispatch, including an attempt that later raises."""
    state["stalled_feedback_last_check"] = dict(
        lease(state, spec, workspace, check_ids),
        status="dispatched",
        call=state.get("model_calls", 0),
        origin="original_verify",
    )
    runtime.store.save(state)


def _same(record, current):
    return all(record.get(key) == value for key, value in current.items())


def _selected_read_target(index, state, spec, unit, workspace):
    from infinicoder.memory.question_memory import select
    from infinicoder.memory.inspection_ledger import InspectionLedger

    choice = select(index, spec, state, unit, workspace)
    records = {row["path"]: row for row in InspectionLedger(index).current()}
    for anchor in choice["anchors"]:
        symbol = index.symbol(anchor["symbol"])
        row = records.get(anchor["path"])
        if (
            symbol["kind"] != "function"
            or not row
            or row["source_sha256"] != anchor["source_sha256"]
            or not any(
                start <= symbol["line"] and end >= symbol["end"] for start, end in row["read_spans"]
            )
        ):
            continue
        try:
            workspace.path(anchor["path"], write=True)
        except ValueError:
            continue
        return dict(
            symbol=anchor["symbol"],
            path=anchor["path"],
            source_sha256=anchor["source_sha256"],
            start=symbol["line"],
            end=symbol["end"],
            origin="question_selected_actual_read",
            selection_is_hypothesis=True,
        )
    return None


def schedule(runtime, state, spec, workspace, index, action, result, ordinary):
    """Reuse the existing repeat-round detector; do not add another read log."""
    if (
        not ordinary
        or state["status"] != "running"
        or state.get("stalled_feedback_attempt")
        or runtime.optimization
        or state.get("changed_files")
        or any(q.get("candidates") for q in state.get("problem_questions", {}).values())
        or state.get("problem_decision", {}).get("repeat_rounds", 0) < 2
    ):
        return False
    from infinicoder.tools.read_batch import observations

    children = list(observations(action, result))
    if (
        not children
        or result.get("error")
        or any(
            a.get("tool") != "read"
            or r.get("error")
            or not r.get("inspection_progress", {}).get("repeated_unchanged_range")
            for a, r in children
        )
    ):
        return False
    unit = runtime._graph.unit(runtime._graph.order[state["milestone"]])
    if not unit.check_ids:
        return False
    current = lease(state, spec, workspace, unit.check_ids)
    if _same(state.get("stalled_feedback_last_check", {}), current):
        return False
    try:
        target = _selected_read_target(index, state, spec, unit, workspace)
    except (OSError, ValueError, KeyError):
        return False
    if target is None:
        return False
    state["stalled_feedback_attempt"] = dict(
        current,
        id=digest([current, state.get("model_calls", 0)]),
        status="pending",
        origin="controller_stalled_read_check",
        target=target,
        trigger_call=state.get("model_calls", 0),
        acceptance_proof=False,
        require_artifacts=state["milestone"] >= len(runtime._graph.units) - 1,
        final_verification_still_required=True,
    )
    runtime.store.event("stalled_feedback_pending", state["stalled_feedback_attempt"])
    runtime.store.save(state)
    return True


def dispatch(runtime, state, spec, workspace, remaining, reserve, ordinary):
    """Persist dispatch before execution; resume never replays an uncertain check."""
    record = state.get("stalled_feedback_attempt")
    if not record or record["status"] != "pending":
        return False
    current = lease(state, spec, workspace, record["check_ids"])
    if not ordinary or state["status"] != "running":
        return False
    if (
        not _same(record, current)
        or state.get("changed_files")
        or runtime.optimization
        or any(q.get("candidates") for q in state.get("problem_questions", {}).values())
        or _same(state.get("stalled_feedback_last_check", {}), current)
    ):
        record.update(status="skipped_stale_or_already_checked", acceptance_proof=False)
        runtime.store.event("stalled_feedback_skipped", record)
        runtime.store.save(state)
        return False
    # Reserve an ordinary model request in addition to the original final gate.
    request_reserve = float(runtime.request_timeout)
    available = lambda: max(0.0, remaining() - reserve - request_reserve)
    if (
        available() <= 0
        or state.get("model_calls", 0) >= runtime.max_model_calls
        or state["steps"] + 1 >= runtime.max_steps
        or state.get("output_bytes", 0) >= runtime.output_budget_bytes
    ):
        record.update(
            status="skipped_budget",
            reserved_final_seconds=reserve,
            reserved_model_seconds=request_reserve,
            acceptance_proof=False,
        )
        runtime.store.event("stalled_feedback_skipped", record)
        runtime.store.save(state)
        return False
    control = runtime.store.take_control()
    if control:
        state.update(
            status="paused" if control == "pause" else "cancelled", reason="user_" + control
        )
        runtime.store.event("control", dict(action=control, boundary="before_stalled_check"))
        runtime.store.save(state)
        return False
    state["steps"] += 1
    record.update(
        status="dispatched",
        controller_step=state["steps"],
        dispatched_epoch=time.time(),
        reserved_final_seconds=reserve,
        reserved_model_seconds=request_reserve,
    )
    runtime.store.event("stalled_feedback_dispatched", record)
    runtime.store.save(state)
    started = time.monotonic()
    try:
        ok, evidence = runtime._verify(
            spec,
            workspace,
            record["check_ids"],
            available,
            require_artifacts=record["require_artifacts"],
        )
    except BaseException as error:
        record.update(
            status="check_error" if isinstance(error, Exception) else "interrupted",
            error=type(error).__name__ + ": " + str(error)[:1000],
            wall_s=time.monotonic() - started,
            acceptance_proof=False,
        )
        runtime.store.event("stalled_feedback_result", record)
        runtime.store.save(state)
        if not isinstance(error, Exception):
            raise
        return True
    current = _same(record, lease(state, spec, workspace, record["check_ids"]))
    infrastructure = any(
        row.get("error") or row.get("rejected_because") or row.get("exit_code") == 124
        for row in evidence
    )
    record.update(
        status="check_error" if infrastructure else "returned",
        accepted=bool(ok),
        current=current,
        evidence=evidence,
        evidence_sha256=digest(evidence),
        wall_s=time.monotonic() - started,
        acceptance_proof=False,
    )
    if infrastructure:
        record["error"] = str(
            next(
                (
                    row.get("error")
                    or row.get("rejected_because")
                    or "Independent check exited124: " + str(row.get("stdout_tail", ""))
                    for row in evidence
                    if row.get("error")
                    or row.get("rejected_because")
                    or row.get("exit_code") == 124
                ),
                "Unknown check execution error",
            )
        )[:1000]
    if current and not ok and not infrastructure:
        state["last_failure"] = dict(evidence=evidence, controller_observation_id=record["id"])
    runtime.store.event("stalled_feedback_result", record)
    runtime.store.save(state)
    return True


def context(state, spec, workspace):
    """A small mandatory origin/version label; failure uses existing feedback."""
    record = state.get("stalled_feedback_attempt")
    if not record or record["status"] in (
        "pending",
        "skipped_budget",
        "skipped_stale_or_already_checked",
    ):
        return None
    current = _same(record, lease(state, spec, workspace, record["check_ids"]))
    if (
        not current
        and isinstance(state.get("last_failure"), dict)
        and state["last_failure"].get("controller_observation_id") == record["id"]
    ):
        state["last_failure"] = None
    return dict(
        origin=record["origin"],
        status=record["status"],
        id=record["id"],
        source_revision=record["source_revision"],
        phase=record["phase"],
        contract_ref=record["contract_ref"],
        check_ids=record["check_ids"],
        current=current,
        historical_not_current_fact=not current,
        accepted=record.get("accepted"),
        error=record.get("error"),
        failure_feedback="last_failure contains original check feedback"
        if current and record["status"] == "returned" and record.get("accepted") is False
        else None,
        acceptance_proof=False,
        final_verification_still_required=True,
    )
