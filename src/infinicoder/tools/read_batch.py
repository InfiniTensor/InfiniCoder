"""Bounded read-only execution over existing tools,receipts and total budgets.

This is an opt-in native transport extension,not a second execution engine.
No command,write,verification or phase transition can enter a batch.
"""

from infinicoder._resources import package_root

import hashlib
from pathlib import Path

TOOLS = ("read", "files", "directories")
MAX_ACTIONS = 8


def identity():
    root = package_root()
    return hashlib.sha256(
        b"".join(
            (root / name).read_bytes()
            for name in (
                "tools/read_batch.py",
                "tools/native_tools.py",
                "runtime/engine.py",
                "runtime/shell_loop.py",
                "memory/source/source_sync.py",
                "memory/source/source_archive.py",
            )
        )
    ).hexdigest()


def observations(action, result):
    """Traverse executed children in order;unexecuted entries are not evidence."""
    if action.get("tool") != "read_batch":
        yield action, result
        return
    actions = action.get("arguments", {}).get("actions", [])
    for item in result.get("items", []):
        ordinal = item.get("ordinal")
        if (
            item.get("executed") is True
            and type(ordinal) is int
            and 1 <= ordinal <= len(actions)
            and item.get("action") == actions[ordinal - 1]
            and isinstance(item.get("observation"), dict)
        ):
            yield item["action"], item["observation"]


def execute(
    runtime, spec, state, workspace, index, actions, action_schema, remaining, reserve, finish_read
):
    """Preflight the whole request,then serially attempt children without refunds."""
    from infinicoder.tools.native_tools import dependency

    packet = dict(
        kind="read_only_batch",
        requested=len(actions) if isinstance(actions, list) else 0,
        executed=0,
        completed=False,
        items=[],
        acceptance_proof=False,
    )
    try:
        if not isinstance(actions, list) or not 2 <= len(actions) <= MAX_ACTIONS:
            raise ValueError("A read batch needs 2–" + str(MAX_ACTIONS) + " actions")
        validator = dependency()(action_schema)
        for child in actions:
            validator.validate(child)
            if child["tool"] not in TOOLS:
                raise ValueError("Only read,files,directories may be batched")
            if child["tool"] == "read":
                path = workspace.path(child["arguments"]["path"]).resolve()
                for check in spec.checks:
                    for protected in check.protected_paths:
                        target = Path(protected)
                        target = (
                            target.resolve()
                            if target.is_absolute()
                            else workspace.path(protected).resolve()
                        )
                        if path == target or path.is_relative_to(target):
                            raise ValueError("Acceptance checker source cannot enter a read batch")
    except Exception as error:
        packet.update(
            error="Read batch rejected before dispatch: "
            + type(error).__name__
            + ": "
            + str(error),
            preflight_rejected=True,
        )
        if isinstance(actions, list):
            packet["items"] = [
                dict(
                    ordinal=i,
                    action=child,
                    executed=False,
                    observation=dict(error="Whole batch failed preflight;not executed"),
                )
                for i, child in enumerate(actions, 1)
            ]
        return packet
    exhausted = None
    for ordinal, child in enumerate(actions, 1):
        if exhausted is None:
            if remaining() <= reserve:
                exhausted = "Task deadline exhausted before child dispatch"
            elif ordinal > 1 and state["steps"] >= runtime.max_steps:
                exhausted = "Task tool-step budget exhausted before child dispatch"
        if exhausted:
            packet["items"].append(
                dict(
                    ordinal=ordinal, action=child, executed=False, observation=dict(error=exhausted)
                )
            )
            continue
        # The ordinary Runtime loop already charged the first attempt.
        if ordinal > 1:
            state["steps"] += 1
        row = dict(ordinal=ordinal, action=child, executed=True, step=state["steps"])
        runtime.store.event("read_batch_child_started", dict(row, call=state.get("model_calls", 0)))
        try:
            args = child["arguments"]
            if child["tool"] == "read":
                result = finish_read(args, workspace.read(**args))
            else:
                result = getattr(index, child["tool"])(**args)
            row["observation"] = result if isinstance(result, dict) else dict(data=result)
        except Exception as error:
            row["observation"] = dict(error=type(error).__name__ + ": " + str(error))
        packet["items"].append(row)
        packet["executed"] += 1
        runtime.store.event("read_batch_child_result", dict(row, call=state.get("model_calls", 0)))
        state["native_read_batch_progress"] = dict(
            call=state.get("model_calls", 0),
            requested=len(actions),
            executed=packet["executed"],
            step=state["steps"],
        )
        runtime.store.save(state)
    packet["completed"] = packet["executed"] == len(actions)
    if any(item["observation"].get("error") for item in packet["items"]):
        packet["error"] = (
            "One or more read-only children failed or were not executed;inspect individual results"
        )
    return packet
