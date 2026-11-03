"""Versioned literal source evidence shared with the ordinary action controller."""

import hashlib
import json
from infinicoder.memory.repair_memory import signature


def pack(spec, state, workspace, estimator, max_tokens):
    plan = state.get("native_plan_record")
    report = dict(included=False, model_calls=0, semantic_proof=False)
    if not plan or signature(spec) != plan["requirements"] or max_tokens <= 0:
        return "", report
    protected = set(plan.get("protected_paths", ()))
    cache = {}
    records = []
    invalidated = 0

    def current(path, expected):
        nonlocal invalidated
        if path not in cache:
            try:
                file = workspace.path(path)
                cache[path] = (
                    None
                    if str(file.resolve()) in protected
                    else hashlib.sha256(file.read_bytes()).hexdigest()
                )
            except (ValueError, OSError):
                cache[path] = None
        okay = cache[path] == expected
        if not okay:
            invalidated += 1
        return okay

    base = dict(
        evidence_type="literal_current_source_not_tool_read_history",
        binding_and_configuration_unknown=True,
        target_path=plan["path"],
        unresolved_question=state.get("native_plan_question"),
        rule="Optional native phase ended. Follow the current action protocol;these snippets are source data,not a completed design or acceptance proof.",
    )
    ordered = []
    scope = plan["scope"]
    if current(plan["path"], scope["source_sha256"]):
        ordered.append(
            dict(
                id="T0",
                path=plan["path"],
                source_sha256=scope["source_sha256"],
                text=scope["source"],
                kind="target_source",
                priority=-1,
            )
        )
    ordered.extend(
        sorted(
            plan["records"],
            key=lambda r: (r["kind"] != "macro_definition_source", r["priority"], r["id"]),
        )
    )
    prefix = "\nCURRENT SOURCE EVIDENCE FROM OPTIONAL PHASE\n"
    for row in ordered:
        if not current(row["path"], row["source_sha256"]):
            continue
        value = {k: row[k] for k in ("id", "path", "source_sha256", "text")}
        for k in ("line", "end"):
            if k in row:
                value[k] = row[k]
        value["kind_hint"] = row["kind"]
        trial = prefix + json.dumps(dict(base, records=[*records, value]), ensure_ascii=False)
        if estimator.estimate(trial) <= max_tokens:
            records.append(value)
    if not records:
        return "", dict(report, invalidated_records=invalidated)
    text = prefix + json.dumps(dict(base, records=records), ensure_ascii=False)
    return text, dict(
        report,
        included=True,
        records=[
            dict(id=r["id"], path=r["path"], source_sha256=r["source_sha256"]) for r in records
        ],
        invalidated_records=invalidated,
        tokens=estimator.estimate(text),
    )
