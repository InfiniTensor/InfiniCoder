"""Project a real prior controller event into the initial task context slot.

Canonical SDK history is unchanged. The immutable initial event is separate from
the current source observation;neither is a new instruction after each result.
"""

from copy import deepcopy
import hashlib
from infinicoder.runtime.implementation_proposal import dump


MARKER = "HISTORICAL CONTROLLER IMPLEMENTATION EVENT\n"


def historical_event(preparation, proposal, assembly):
    response_sha = hashlib.sha256((proposal.get("text") or "").encode()).hexdigest()
    event = dict(
        origin="controller implementation proposal;not SDK assistant action",
        produced_phase="before_ordinary_agent",
        response_text_sha256=response_sha,
        status=assembly["status"],
        applied=assembly.get("applied", False),
        reason=assembly.get("reason"),
        source_path=preparation.plan["target"]["path"],
        initial_source_sha256=preparation.plan["target"]["sha256"],
        proposed_source_sha256=assembly.get("patch", {}).get("after"),
        diff=assembly.get("patch", {}).get("diff", "")[:1000],
        diff_truncated=assembly.get("patch", {}).get("diff_truncated", False)
        or len(assembly.get("patch", {}).get("diff", "")) > 1000,
        acceptance_proof=False,
    )
    return dict(event, event_id=hashlib.sha256(dump(event).encode()).hexdigest())


def project_history(messages, event, current_hash):
    result = deepcopy(messages)
    slot = next((i for i, m in enumerate(result) if m.get("role") == "user"), None)
    if slot is None or not isinstance(result[slot].get("content"), str):
        raise ValueError(
            "A textual initial user task slot is required for historical event projection"
        )
    expected = (
        event.get("proposed_source_sha256") if event["applied"] else event["initial_source_sha256"]
    )
    currency = (
        "unknown" if current_hash is None else "current" if current_hash == expected else "stale"
    )
    observation = dict(
        current_source_sha256=current_hash,
        source_currency=currency,
        meaning="Observation at input preparation;the preceding initial event is immutable",
    )
    result[slot]["content"] += (
        "\n\n" + MARKER + dump(dict(initial_event=event, current_source_observation=observation))
    )
    return result, currency, slot
