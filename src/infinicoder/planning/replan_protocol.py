"""Keep proven unexecuted response failures out of architecture replanning."""

from infinicoder._resources import package_root

import hashlib
import json

POLICY = "protocol_failure_replan_guard_v1"
REASONS = frozenset(("incomplete_response", "action_envelope", "action_schema", "source_envelope"))
CONTROL_ACTIONS = frozenset(
    ("plan", "replan", "domain_design", "diagnose", "focus_edit", "release_edit_focus", "ask_user")
)


def identity():
    root = package_root()
    return dict(
        policy=POLICY,
        implementation=hashlib.sha256(
            b"".join(
                (root / name).read_bytes()
                for name in (
                    "planning/replan_protocol.py",
                    "runtime/engine.py",
                    "runtime/worker.py",
                    "runtime/protocol.py",
                )
            )
        ).hexdigest(),
    )


def refusal(
    reason,
    response,
    *,
    origin,
    staged,
    dispatch_started,
    proposal_accepted,
    action,
    request_sha256,
    input_protocol,
    source_before,
    source_after,
    call,
    phase,
):
    """Only the controller's exact completeness/decoder boundaries supply reason."""
    if (
        reason not in REASONS
        or response is None
        or origin != "model"
        or staged
        or dispatch_started
        or proposal_accepted
    ):
        return None
    if isinstance(action, dict) and action.get("tool") in CONTROL_ACTIONS:
        return None
    if input_protocol not in (
        "json_action",
        "source:json_source",
        "source:json_lines",
        "source:source",
        "source:auto",
    ):
        return None
    if any(
        not isinstance(v, str) or len(v) != 64
        for v in (request_sha256, source_before, source_after)
    ):
        return None
    finish = getattr(response, "finish_reason", None)
    if reason == "incomplete_response" and finish not in ("length", "max_tokens", "unknown"):
        return None
    if (
        not isinstance(getattr(response, "text", None), str)
        or type(call) is not int
        or type(phase) is not int
    ):
        return None
    result = dict(
        kind="controller_confirmed_unexecuted_response_rejection",
        reason=reason,
        origin="model",
        call=call,
        phase=phase,
        request_sha256=request_sha256,
        response_sha256=hashlib.sha256(response.text.encode()).hexdigest(),
        finish_reason=finish if finish in ("stop", "length", "max_tokens", "unknown") else "other",
        input_protocol=input_protocol,
        source_before=source_before,
        source_after=source_after,
        dispatch="not_started",
        candidate_executed=False,
        architecture_failure_proven=False,
    )
    result["id"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def blocked(runtime, state, record, branch):
    """Internal audit only: do not change feedback, counters, budgets or prompts."""
    row = dict(
        branch=branch,
        refusal=record,
        original_replans=state.get("replans", 0),
        original_failures_retained=True,
        original_budget_retained=True,
    )
    state["protocol_replan_blocks"] = [*state.get("protocol_replan_blocks", []), row][-12:]
    runtime.store.event("automatic_protocol_replan_blocked", row)
