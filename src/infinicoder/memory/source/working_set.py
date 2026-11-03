"""Put versioned implementation excerpts before optional memory metadata.

The existing TaskMemory selects/validates sources and relationships. Its bounded
local retrieval buffer is never sent to the model as-is. This renderer admits
whole JSON records against the caller tokenizer, retaining source locations and
uncertainty. It creates no read receipts, edits, commands or semantic proofs.
"""

import json


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _windows(row, anchor=None):
    """Smaller literal source windows always retain an in-range anchor line."""
    lines = row["excerpt"].splitlines()
    first = row["excerpt_line"]
    if not lines:
        return
    target = anchor if type(anchor) is int and first <= anchor < first + len(lines) else first
    offset = target - first
    for count in dict.fromkeys(
        (len(lines), min(12, len(lines)), min(6, len(lines)), min(3, len(lines)), 1)
    ):
        start = max(0, min(offset - count // 3, len(lines) - count))
        end = start + count
        # A partial window is an excerpt, never a rewritten/complete function.
        yield dict(
            symbol=row.get("symbol"),
            path=row["path"],
            source_sha256=row["source_sha256"],
            excerpt="\n".join(lines[start:end]),
            excerpt_line=first + start,
            excerpt_end=first + end - 1,
            implementation_excerpt_complete=bool(row.get("implementation_excerpt_complete"))
            and start == 0
            and end == len(lines),
            **({"unparsed_source": True} if row.get("unparsed_source") else {}),
        )


def pack(memory, choice, query, focus, max_tokens, estimator, exclude_paths=()):
    header = dict(
        working_code_question=dict(
            view=choice["view"],
            direction=choice["direction"],
            basis=choice["selection_basis"],
            selection_is_hypothesis=True,
            policy="Source syntax only;alias/ownership/dynamic dispatch unknown. Excerpts grant no read/write authority.",
        )
    )
    prefix = _json(header)
    records = [prefix]
    selected = []
    omitted = 0
    fallback_count = 0
    report = dict(
        tokens=0,
        selected=[],
        fallback_sources_included=0,
        source_body_included=False,
        included=False,
        encoding="source_first_v1",
        max_tokens=max_tokens,
    )
    if estimator.estimate(prefix) > max_tokens:
        return "", dict(report, omitted_for_budget=True)

    def offer(row):
        text = "\n".join([*records, _json(row)])
        if estimator.estimate(text) > max_tokens:
            return False
        records.append(_json(row))
        return True

    anchors = [
        dict(symbol=row["symbol"], line=row["line"], reason=row["basis"])
        for row in choice["anchors"]
    ]
    locations = {row["symbol"]: row["line"] for row in choice["anchors"]}
    # Bounded local search is separate from the actual model context budget.
    # No additional neural request or new index/answer source is introduced.
    scratch_budget = max(2400, min(8000, max_tokens * 4))
    body, retrieval = memory.pack(
        query,
        scratch_budget,
        estimator,
        focus_paths=focus,
        exclude_paths=exclude_paths,
        view=choice["view"],
        anchors=anchors,
        direction=choice["direction"],
    )
    candidates = [json.loads(line) for line in body.splitlines()]
    source_rows = [row for row in candidates if "excerpt" in row]
    for fallback in choice["fallback_sources"]:
        source_rows.insert(
            0,
            dict(
                symbol=None,
                path=fallback["path"],
                source_sha256=fallback["source_sha256"],
                excerpt=fallback["source"],
                excerpt_line=fallback["start"],
                excerpt_end=fallback["end"],
                implementation_excerpt_complete=False,
                unparsed_source=True,
                anchor=fallback["line"],
            ),
        )
    for row in source_rows:
        admitted = False
        for window in _windows(row, row.get("anchor", locations.get(row.get("symbol")))):
            visible = window
            if row.get("unparsed_source"):
                visible = dict(
                    current_unparsed_source=dict(
                        path=window["path"],
                        source_sha256=window["source_sha256"],
                        start=window["excerpt_line"],
                        end=window["excerpt_end"],
                        source=window["excerpt"],
                        relations="unknown",
                    )
                )
            if offer(visible):
                selected.append({key: value for key, value in window.items() if key != "excerpt"})
                fallback_count += bool(row.get("unparsed_source"))
                admitted = True
                break
        if not admitted:
            omitted += 1
    source_count = len(selected)
    # Attach only whole semantic observations, with their original uncertainty
    # and source version. Do not shorten expressions or turn candidates into
    # definite alias, value, dispatch or ownership statements.
    facts = 0
    for row in candidates:
        if (
            "excerpt" in row
            or "task_memory" in row
            or row.get("source_omitted")
            or "memory_omissions" in row
        ):
            continue
        if "function_summary" in row or "use_def_of" in row:
            if offer(row):
                facts += 1
            else:
                omitted += 1
    if not source_count and not facts:
        return "", dict(
            report,
            omitted_for_budget=True,
            retrieval_report=retrieval,
            reason="No current source observation fits;routing metadata alone is not code memory.",
        )
    more = omitted + retrieval.get("omitted", 0) + retrieval.get("frontier", 0)
    if more:
        offer(
            dict(
                more_source_available=True,
                expand="read/function_summary/use_def;omitted is not irrelevant",
            )
        )
    text = "\n".join(records)
    return text, dict(
        report,
        tokens=estimator.estimate(text),
        selected=selected,
        fallback_sources_included=fallback_count,
        source_body_included=bool(source_count),
        semantic_records=facts,
        included=True,
        omitted=more,
        retrieval_report=retrieval,
        local_retrieval_budget=scratch_budget,
        source_records=source_count,
    )
