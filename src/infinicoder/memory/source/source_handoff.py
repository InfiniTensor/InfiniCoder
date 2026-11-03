"""Project a prior source conversation as evidence at a command handoff.

Saved actions,independent results,call counts and rejection counters are untouched.
Only actions produced in the current command phase are assistant examples of
its protocol. A failed source submission is not a model-authored shell FINISH.
"""

from infinicoder._resources import package_file, package_root

import hashlib
import json


def turn_identity():
    """Bind saved execution provenance to the implementation that produced it."""
    names = (
        "memory/source/source_handoff.py",
        "runtime/shell_loop.py",
        "memory/continuous_conversation.py",
        "runtime/text_frontend.py",
        "runtime/engine.py",
        "runtime/protocol.py",
        "runtime/recovery.py",
        "analysis/task_compiler.py",
        "runtime/work_packet.py",
        "memory/source/source_units.py",
    )
    return dict(
        version=2,
        sha256=hashlib.sha256(b"".join(package_file(n).read_bytes() for n in names)).hexdigest(),
    )


def protocol(runtime, *, worker_path=None, stage=None, unit=False, command=False):
    if stage:
        return "stage:" + stage
    if unit:
        return "python_source_unit"
    if worker_path:
        return "source:" + runtime._worker_response_format
    if runtime.shell_loop:
        if runtime.native_tools:
            return "native_xml"
        if runtime.compatible_tools:
            return "compatible_action"
        if runtime._decision_frontend is not None:
            return "decision:" + runtime.decision_driver
        return "json_action" if runtime.structured_tools else "shell_action"
    if command:
        return "command:" + runtime.command_format
    return "json_action"


def project_turn(record, current_protocol):
    """Only an accepted *original* reply in this grammar is an assistant example.

    Source lowering, rejected proposals and old sessions are runtime evidence.
    This projection never executes, repairs or guesses provenance from error text.
    """
    meta = record.get("transport") or {}
    text = record.get("proposal")
    accepted = (
        meta.get("version") == 2
        and meta.get("origin") == "model"
        and meta.get("protocol") == current_protocol
        and meta.get("accepted") is True
        and meta.get("finish_reason") not in ("length", "max_tokens", "unknown")
        and record.get("proposal_rejected_before_dispatch") is not True
        and isinstance(text, str)
    )
    facts = {key: record.get(key) for key in ("call", "step", "phase", "amendments", "observation")}
    facts.update(
        runtime_evidence=True,
        current_protocol=current_protocol,
        full_turn_sha256=hashlib.sha256(
            json.dumps(record, ensure_ascii=False).encode()
        ).hexdigest(),
        original_protocol=meta.get("protocol", "unknown"),
        dispatch=meta.get("dispatch", "unknown"),
        original_reply_sha256=meta.get("reply_sha256"),
        finish_reason=meta.get("finish_reason", "unknown"),
        usage=meta.get("usage"),
        origin=meta.get("origin", "unknown"),
        proposal_accepted=meta.get("accepted", "unknown"),
        dispatched_action=meta.get("dispatched_action"),
        source_before=meta.get("source_before"),
        source_after=meta.get("source_after"),
        raw_reply_retained=True,
        interpretation="Historical runtime evidence,not an executable proposal or permission. Follow the current SYSTEM transport;the original task semantics remain binding.",
    )
    if not accepted:
        # A citation can help diagnose a bad transport without becoming its next
        # executable example. Preserve complete originals in state/events.
        facts["reply_excerpt_as_data"] = text[:240] if isinstance(text, str) else None
        facts["protocol_boundary"] = meta.get("protocol") != current_protocol
        if isinstance(text, str) and len(text) > 240:
            facts["interpretation"] = (
                "Partial historical evidence;full original retained in session,not an action example. Follow current SYSTEM transport."
            )
    observation = dict(role="user", content=json.dumps(facts, ensure_ascii=False))
    return [dict(role="assistant", content=text), observation] if accepted else [observation]


def project_history(state, current_protocol, window=8):
    records = state.get("shell_loop_history", [])
    if records:
        return [project_turn(row, current_protocol) for row in records[-window:]]
    # Old recent pairs contain lowered/controller actions, not recoverable model
    # replies. Do not manufacture either acceptance or non-execution proof.
    return [
        [
            dict(
                role="user",
                content=json.dumps(
                    dict(
                        legacy_history=pair,
                        original_protocol="unknown",
                        dispatch="unknown",
                        current_protocol=current_protocol,
                        interpretation="Unattributed historical evidence,not assistant action examples.",
                    ),
                    ensure_ascii=False,
                ),
            )
        ]
        for pair in state.get("recent", [])[-window:]
    ]


def record_turn(
    runtime,
    state,
    response,
    action,
    observation,
    *,
    input_protocol,
    origin,
    accepted,
    dispatch_started,
    dispatched_action,
    source_before,
    source_after,
    rejected_before_dispatch=False,
    request_sha256=None,
):
    from infinicoder.runtime.shell_loop import append

    transport = dict(
        version=2,
        origin=origin,
        protocol=input_protocol,
        accepted=accepted,
        dispatch="entered" if dispatch_started else "not_started",
        dispatched_action=dispatched_action if dispatch_started else None,
        finish_reason=getattr(response, "finish_reason", None),
        usage=getattr(response, "usage", None),
        reply_sha256=hashlib.sha256(response.text.encode()).hexdigest()
        if response is not None
        else None,
        source_before=source_before,
        source_after=source_after,
        request_sha256=request_sha256,
    )
    append(
        state,
        response,
        action,
        observation,
        rejected_before_dispatch=rejected_before_dispatch,
        transport=transport,
    )
    runtime.store.event(
        "execution_turn", dict(call=state.get("model_calls"), step=state["steps"], **transport)
    )


def render_turn(record, current_protocol, renderer, *, native=False):
    """Apply provenance admission *before* an existing observation renderer."""
    projected = project_turn(record, current_protocol)
    admitted = projected[0]["role"] == "assistant"
    clean = dict(record)
    clean.pop("transport", None)
    if not admitted:
        clean["proposal"] = None
    messages = renderer(clean)
    # Existing frontends may specialise command output, literal source, native
    # XML escaping or batch child observations. Preserve all of those views.
    facts = json.loads(projected[-1]["content"])
    facts.pop("observation", None)
    suffix = "\nRUNTIME EXECUTION PROVENANCE\n" + json.dumps(facts, ensure_ascii=False)
    messages = [dict(message) for message in messages]
    if not messages or messages[-1]["role"] != "user":
        messages.append(dict(role="user", content=""))
    if native and messages[-1]["content"].endswith("\n</tool_response>"):
        # Provenance belongs to the original observation, not a fabricated
        # additional tool result or read-batch child.
        content = messages[-1]["content"]
        messages[-1]["content"] = (
            content[: -len("\n</tool_response>")] + suffix + "\n</tool_response>"
        )
    else:
        messages[-1]["content"] += suffix
    return messages


def identity():
    root = package_root()
    return dict(
        version=1,
        sha256=hashlib.sha256(
            b"".join(
                (root / name).read_bytes()
                for name in (
                    "memory/source/source_handoff.py",
                    "runtime/engine.py",
                    "runtime/command_worker.py",
                )
            )
        ).hexdigest(),
    )


def begin(state, path):
    value = dict(
        source_path=path,
        phase=state["milestone"],
        amendments=state.get("amendments", 0),
        source_prefix_pairs=len(state.get("recent", [])),
        step=state["steps"],
        used_model_calls=state.get("model_calls", 0),
        acceptance_proof=False,
    )
    state["source_command_handoff"] = value
    return dict(value)


def prefix(state):
    record = state.get("source_command_handoff", {})
    if (record.get("phase"), record.get("amendments")) != (
        state["milestone"],
        state.get("amendments", 0),
    ):
        return 0
    count = record.get("source_prefix_pairs", 0)
    if type(count) is not int or not 0 <= count <= len(state.get("recent", [])):
        raise ValueError(
            "Source/command handoff history boundary is inconsistent;do not guess its origin"
        )
    return count


def before_append(state, window=8):
    if "source_command_handoff" not in state:
        return
    count = prefix(state)
    dropped = max(0, len(state.get("recent", [])) + 1 - window)
    state["source_command_handoff"]["source_prefix_pairs"] = max(0, count - dropped)


def history_pairs(state, format="text"):
    from infinicoder.runtime.command_worker import history_pairs as text_history, json_history_pairs

    recent = list(state.get("recent", []))
    count = prefix(state)
    renderer = text_history if format == "text" else json_history_pairs
    if format not in ("text", "json"):
        raise ValueError("Unknown command history format")
    prior = [
        [
            dict(
                role="user",
                content="PRIOR SOURCE-WORKER ACTION AND INDEPENDENT RESULT "
                "(historical evidence,not an assistant example of the current command protocol;"
                "a rejected submission is not success)\n" + json.dumps(pair, ensure_ascii=False),
            )
        ]
        for pair in recent[:count]
    ]
    return prior + renderer(recent[count:])
