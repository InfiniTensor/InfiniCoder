"""Optional independent feedback after an observed shell revision transition.

This schedules the caller's existing checks. It neither interprets shell output
as success nor advances a work unit. Explicit verification and final acceptance
always execute afresh; automatic observations are not reusable certificates.
"""

from infinicoder._resources import package_root

import hashlib
import time


GUIDANCE = (
    "Runtime automatically runs the current phase checks after a shell "
    "command changes the workspace revision, including commands that exit with "
    "an error after writing. Unchanged writes and read-only commands do not "
    "trigger another automatic check. Inspect the independent failure, repair "
    "its cause, and submit normally. Automatic checks are diagnostic only, "
    "share the session time/output budgets, and do not complete a phase."
)


def identity():
    root = package_root()
    return dict(
        version=1,
        sha256=hashlib.sha256(
            b"".join(
                (root / name).read_bytes()
                for name in ("memory/revision_feedback.py", "runtime/engine.py")
            )
        ).hexdigest(),
    )


def revision(spec, workspace):
    # Checker-owned reports are not implementation changes. All other regular
    # files, deletions and links participate in Workspace's complete revision.
    outputs = list(dict.fromkeys(path for check in spec.checks for path in check.output_paths))
    return workspace.revision(outputs)


def after_command(runtime, spec, state, workspace, before, result, remaining, reserve):
    after = revision(spec, workspace)
    if after == before:
        return result
    unit = runtime._graph.unit(runtime._graph.order[state["milestone"]])
    last_unit = state["milestone"] >= len(runtime._graph.units) - 1
    record = dict(
        before_revision=before,
        observed_revision=after,
        phase=state["milestone"],
        check_ids=list(unit.check_ids),
        require_artifacts=last_unit,
        model_call=state.get("model_calls", 0),
        command_exit_code=result.get("exit_code"),
        diagnostic_only=True,
        final_verification_still_required=True,
        reserved_final_seconds=reserve,
    )
    if not unit.check_ids and runtime.verifier is None:
        record.update(status="skipped_no_independent_phase_check", acceptance_proof=False)
        runtime.store.event("command_revision_feedback", record)
        return dict(result, automatic_revision_feedback=record)
    control = runtime.store.take_control()
    if control:
        state.update(
            status="paused" if control == "pause" else "cancelled", reason="user_" + control
        )
        runtime.store.event(
            "control", dict(action=control, boundary="before_automatic_command_check")
        )
        record.update(status="skipped_user_control", acceptance_proof=False)
        runtime.store.event("command_revision_feedback", record)
        return dict(result, automatic_revision_feedback=record)
    automatic_remaining = lambda: max(0, remaining() - reserve)
    if automatic_remaining() <= 0:
        record.update(status="skipped_final_budget_reserved", acceptance_proof=False)
        runtime.store.event("command_revision_feedback", record)
        return dict(result, automatic_revision_feedback=record)
    started = time.monotonic()
    try:
        ok, evidence = runtime._verify(
            spec, workspace, unit.check_ids, automatic_remaining, require_artifacts=last_unit
        )
    except Exception as error:
        # Preserve the executed command's output if infrastructure checking
        # fails. The next action must not repeat a write as if it never ran.
        record.update(
            status="check_error",
            error=type(error).__name__ + ": " + str(error)[:1000],
            failure_origin="independent_check_execution",
            acceptance_proof=False,
        )
        state["automatic_check_error"] = record
        combined = dict(result, automatic_revision_feedback=record)
    else:
        record.update(
            status="checked",
            accepted=bool(ok),
            retained_revision=revision(spec, workspace),
            evidence_revisions=list(
                dict.fromkeys(row["revision"] for row in evidence if row.get("revision"))
            ),
        )
        state.pop("automatic_check_error", None)
        state["last_failure"] = None if ok else evidence
        combined = dict(
            result,
            diagnostic=True,
            accepted=bool(ok),
            evidence=evidence,
            automatic_revision_feedback=record,
        )
    record["wall_s"] = time.monotonic() - started
    state["automatic_command_check_runs"] = state.get("automatic_command_check_runs", 0) + 1
    runtime.store.event(
        "command_revision_feedback", dict(record, evidence=combined.get("evidence", []))
    )
    return combined
