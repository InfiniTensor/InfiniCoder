"""Optional model view of runtime-proven pre-dispatch format rejections.

Raw proposals/events are never changed. A failed command, checker failure, model
claim or legacy invalid-action marker cannot certify that nothing executed.
"""

from infinicoder._resources import package_root

from copy import deepcopy
import hashlib
import json


MARKER = "proposal_rejected_before_dispatch"


def identity():
    root = package_root()
    return hashlib.sha256(
        b"".join(
            (root / name).read_bytes()
            for name in ("runtime/format_repair.py", "runtime/shell_loop.py", "runtime/engine.py")
        )
    ).hexdigest()


def _notice(rejected, structured, language):
    if type(structured) is not bool or language not in ("bash", "powershell"):
        raise ValueError("A runtime-selected action grammar is required")
    h = hashlib.sha256()
    for row in rejected:
        h.update(json.dumps(row, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode())
        h.update(b"\n")
    grammar = (
        "Return exactly ONE JSON action with tool and arguments, conforming "
        "to the current advertised schema. No markdown or extra objects."
        if structured
        else "Return exactly ONE complete fenced " + language + " block. Use an "
        "ordinary shell script or one ic-tool JSON action in the fence."
    )
    return dict(
        runtime_observation="action_format_rejections",
        rejected_count=len(rejected),
        last_rejected_call=rejected[-1]["call"],
        raw_records_sha256=h.hexdigest(),
        raw_records_retained=True,
        execution="The runtime rejected these proposals before dispatch;they did not execute.",
        next=grammar + " Keep the next action small enough to complete. Inspect current "
        "source when needed;do not assume rejected text changed the workspace.",
        acceptance_proof=False,
    )


def project(records, *, structured=False, language="bash"):
    """Remove only trusted format rejections; retain one chronological notice.

    Call numbering, current task/budgets, executed observations and source memory
    are owned elsewhere and are not reset or changed by this presentation.
    """
    rejected = [r for r in records if r.get(MARKER) is True]
    report = dict(
        total_records=len(records),
        rejected_records=len(rejected),
        collapsed_records=max(0, len(rejected) - 1),
        raw_records_retained=True,
    )
    if not rejected:
        return records, report
    last = rejected[-1]
    notice = deepcopy(last)
    notice["proposal"] = None
    notice["observation"] = _notice(rejected, structured, language)
    notice["interpretation"] = (
        "Runtime format diagnostic;not an action example or acceptance proof."
    )
    # Identity, rather than equal content, preserves distinct identical rejected
    # proposals at different calls without removing a successful observation.
    result = []
    for row in records:
        if row is last:
            result.append(notice)
        elif row.get(MARKER) is not True:
            result.append(row)
    report.update(
        last_rejected_call=last["call"], notice_sha256=notice["observation"]["raw_records_sha256"]
    )
    return result, report


def recent_view(state, *, structured=False, language="bash"):
    """Project stage-builder pairs only after exact alignment with raw history.

    Old/migrated/mismatched history is retained. No marker is inferred from an
    error string, invalid-action tool name, or model-provided observation.
    """
    recent = state.get("recent", [])
    history = state.get("shell_loop_history", [])
    if not recent or len(recent) > len(history):
        return recent, dict(aligned=False, raw_records_retained=True)
    tail = history[-len(recent) :]
    for pair, row in zip(recent, tail):
        observation = json.dumps(row["observation"], ensure_ascii=False)
        if len(observation) > 6000:
            observation = (
                observation[:5500]
                + "\n[TRUNCATED: narrow the query/read range; full result in session event log]"
            )
        expected = [
            dict(role="assistant", content=json.dumps(row["action"], ensure_ascii=False)),
            dict(role="user", content=observation),
        ]
        if pair != expected:
            return recent, dict(aligned=False, raw_records_retained=True)
    rejected = [r for r in tail if r.get(MARKER) is True]
    if not rejected:
        return recent, dict(aligned=True, rejected_records=0, raw_records_retained=True)
    last = rejected[-1]
    notice = [
        dict(
            role="user",
            content=json.dumps(_notice(rejected, structured, language), ensure_ascii=False),
        )
    ]
    result = [
        notice if row is last else pair
        for pair, row in zip(recent, tail)
        if row is last or row.get(MARKER) is not True
    ]
    return result, dict(
        aligned=True,
        rejected_records=len(rejected),
        collapsed_records=len(rejected) - 1,
        raw_records_retained=True,
    )
