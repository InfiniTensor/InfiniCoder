"""Lossless task-first projection of Runtime's mandatory contract packet.

No summarization, source inference, model call, permission or budget change.
Code facts, observations and model notes retain their existing separate stores.
"""

import hashlib
import json
from pathlib import Path


def identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def render(contract):
    if not isinstance(contract, str):
        raise ValueError("Runtime contract text required")
    prefix = len(contract) - len(contract.lstrip())
    text = contract[prefix:]

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate Runtime packet field")
            result[key] = value
        return result

    core, end = json.JSONDecoder(object_pairs_hook=unique).raw_decode(text)
    if not isinstance(core, dict) or not isinstance(core.get("task"), str):
        raise ValueError("Mandatory Runtime object with literal task required")
    suffix = text[end:]
    remaining = dict(core)
    projected = {}
    parts = []
    spans = []
    length = 0

    def add(label, fields, literal=False):
        nonlocal length
        values = {key: remaining.pop(key) for key in fields if key in remaining}
        if not values:
            return
        value = next(iter(values.values())) if literal else values
        if literal and not isinstance(value, str):
            raise ValueError("Literal task must be text")
        body = value if literal else json.dumps(value, ensure_ascii=False)
        header = label + "\n"
        start = length + len(header)
        parts.append(header + body + "\n\n")
        length += len(parts[-1])
        projected.update(values)
        spans.append(
            dict(
                label=label,
                fields=list(values),
                start=start,
                end=start + len(body),
                literal=literal,
            )
        )

    add("CALLER TASK (literal text;no summary)", ("task",), True)
    add("REMAINING MODEL CALLS (runtime;including the next request)", ("remaining_model_calls",))
    add("CURRENT CALLER REQUIREMENT UPDATES", ("amendments", "clarifications"))
    add(
        "CURRENT WORK UNIT (runtime plan;not edit authorization)",
        ("current_work_unit", "completed_milestones", "phase_versions"),
    )
    add(
        "CALLER CONSTRAINTS AND ACCEPTANCE (declarations;all still binding)",
        ("allowed_paths", "required_artifacts", "target_facts", "tensor_layouts", "acceptance"),
    )
    add("LATEST RECORDED FAILURE (observation;check its source version)", ("last_failure",))
    add("RUNTIME PROGRESS GUIDANCE", ("progress_hint",))
    add("OTHER RUNTIME DATA (original values;no new facts)", tuple(remaining))
    if remaining or digest(projected) != digest(core):
        raise ValueError("Task-first view failed to preserve complete Runtime data")
    body = "".join(parts) + suffix
    receipt = dict(
        version=1,
        kind="mandatory_contract_projection",
        analysis_identity=identity(),
        source_packet_sha256=hashlib.sha256(contract.encode()).hexdigest(),
        core_sha256=digest(core),
        projected_core_sha256=digest(projected),
        task_sha256=hashlib.sha256(core["task"].encode()).hexdigest(),
        fields_retained=sorted(core),
        dropped_fields=[],
        sections=spans,
        projection_sha256=hashlib.sha256(body.encode()).hexdigest(),
        suffix_sha256=hashlib.sha256(suffix.encode()).hexdigest(),
        model_calls=0,
        acceptance_proof=False,
        source_analysis_performed=False,
        scope="Layout of existing mandatory task/control data only. Existing code-memory freshness and final checks remain authoritative.",
    )
    return body, receipt
