"""One source-grounded workflow change after three identical validated reads."""

from infinicoder._resources import package_root

from dataclasses import asdict
import hashlib, json, time
from pathlib import Path


def identity():
    root = package_root()
    return hashlib.sha256(
        b"".join(
            (root / n).read_bytes()
            for n in (
                "memory/inspection_recovery.py",
                "memory/source/source_focus.py",
                "runtime/engine.py",
            )
        )
    ).hexdigest()


def contract(spec, state):
    return hashlib.sha256(
        json.dumps(
            dict(spec=asdict(spec), clarifications=state.get("clarifications", [])),
            sort_keys=True,
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def _reset_reads(state):
    state.pop("inspection_recovery_reads", None)
    return False


def validated_read(workspace, result, protected_paths=()):
    """Return a current complete-line observation without inferring understanding."""
    file = workspace.path(result["path"])
    data = file.read_bytes()
    if len(data) > 2_000_000 or b"\0" in data:
        raise ValueError("unsupported source")
    for name in protected_paths:
        item = Path(name)
        target = item.resolve() if item.is_absolute() else workspace.path(name).resolve()
        if file.resolve() == target or file.resolve().is_relative_to(target):
            raise ValueError("protected source")
    digest = hashlib.sha256(data).hexdigest()
    start = result["start"]
    text = result["text"]
    count = result.get("covered_line_count", len(text.splitlines()))
    lines = data.decode("utf-8-sig").splitlines()
    if (
        digest != result["sha256"]
        or type(start) is not int
        or start < 1
        or type(count) is not int
        or count < 1
        or start + count - 1 > len(lines)
        or text != "\n".join(lines[start - 1 : start - 1 + count])
    ):
        raise ValueError("stale or incomplete-line observation")
    return dict(
        path=result["path"],
        sha256=digest,
        start=start,
        count=count,
        total_lines=len(lines),
        text=text,
    )


def observe(runtime, spec, state, workspace, action, result):
    if state.get("inspection_recovery", {}).get("status") in ("pending", "selected", "declined"):
        return False
    if action.get("tool") not in ("read", "read_symbol", "read_function_body") or result.get(
        "error"
    ):
        return _reset_reads(state)
    try:
        value = validated_read(
            workspace, result, [p for check in spec.checks for p in check.protected_paths]
        )
        sha = value["sha256"]
        start = value["start"]
        text = value["text"]
        revision = workspace.revision()
        requirement = contract(spec, state)
        key = hashlib.sha256(
            json.dumps(
                [result["path"], sha, start, text, revision, requirement], ensure_ascii=False
            ).encode()
        ).hexdigest()
    except (ValueError, OSError, UnicodeError, KeyError, TypeError):
        return _reset_reads(state)
    previous = state.get("inspection_recovery_reads", {})
    repeat = previous.get("count", 0) + 1 if previous.get("key") == key else 1
    state["inspection_recovery_reads"] = dict(key=key, count=repeat)
    if (
        repeat < 3
        or state.get("model_calls", 0) >= runtime.max_model_calls
        or time.time() >= state.get("session_deadline_epoch", 0)
    ):
        return False
    state["inspection_recovery"] = dict(
        status="pending",
        trigger="three_identical_current_source_reads",
        trigger_call=state.get("model_calls", 0),
        source_revision=revision,
        requirement_sha256=requirement,
        path=result["path"],
        source_sha256=sha,
        acceptance_proof=False,
    )
    return True


def pending(spec, state, workspace):
    record = state.get("inspection_recovery", {})
    if record.get("status") != "pending":
        return False
    try:
        current = record["source_revision"] == workspace.revision() and record[
            "requirement_sha256"
        ] == contract(spec, state)
    except (OSError, ValueError, TypeError, KeyError):
        current = False
    if not current:
        record.update(
            status="declined",
            reason="Source or caller requirements changed before workflow selection",
        )
        return False
    return True


def session_context(state, remaining_calls):
    from infinicoder.runtime.feedback import failure_packet

    return dict(
        reason="Runtime observed repeated unchanged source reads;one bounded implementation proposal may replace the prior action loop.",
        trigger=state.get("inspection_recovery"),
        clarifications=state.get("clarifications", []),
        historical_failure=failure_packet(state.get("last_failure")),
        failure_evidence_type="Historical runtime observation,not current behavioral proof",
        remaining_model_calls=remaining_calls,
        raw_history_retained=True,
        prior_protocol="Historical only;follow the current one-function proposal protocol. No budget or permission reset.",
        acceptance_proof=False,
    )
