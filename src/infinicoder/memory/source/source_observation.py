"""Literal views of actual historical reads;no source facts or authority minted."""

import hashlib, json, re
from pathlib import Path


def identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def observed(record):
    action = record.get("action") or {}
    source = record.get("observation")
    if (
        not isinstance(action, dict)
        or action.get("tool") not in ("read", "read_symbol", "read_function_body")
        or record.get("proposal_rejected_before_dispatch")
        or not isinstance(source, dict)
    ):
        return None
    if not isinstance(source.get("path"), str) or not isinstance(source.get("text"), str):
        return None
    if not isinstance(source.get("sha256"), str) or not re.fullmatch(
        "[0-9a-f]{64}", source["sha256"]
    ):
        return None
    if (
        type(source.get("start")) is not int
        or source["start"] < 1
        or type(source.get("total_lines")) is not int
        or source["total_lines"] < 0
    ):
        return None
    return source


def fence(text):
    runs = re.findall(chr(96) + "+", text)
    marker = chr(96) * max(3, 1 + max(map(len, runs), default=0))
    return marker + "\n" + text + "\n" + marker


def full(record):
    """Keep every original read field;only text serialization changes."""
    source = observed(record)
    if source is None:
        return None
    packet = {k: record[k] for k in ("call", "step", "phase", "amendments", "interpretation")}
    packet["observation"] = {k: v for k, v in source.items() if k != "text"}
    packet["literal_read_view"] = dict(
        version=1,
        historical=True,
        source_text="separate literal read output below;backslashes,quotes and indentation are not JSON escapes",
    )
    return (
        json.dumps(packet, ensure_ascii=False)
        + "\nLITERAL HISTORICAL READ OUTPUT\n"
        + fence(source["text"])
    )


def bounded(record, budget, estimate):
    """Fit whole observed line windows;never invent a partial assistant action."""
    source = observed(record)
    if source is None:
        return None
    text = source["text"]
    lines = text.split("\n") if text or source["start"] <= source["total_lines"] else []
    start = source["start"]
    count = len(lines)
    raw = json.dumps(record, ensure_ascii=False)

    def view(width):
        spans = []
        if width and count:
            if width * 2 >= count:
                spans = [(0, count)]
            else:
                spans = [(0, width), (count - width, count)]
        packet = dict(
            call=record["call"],
            phase=record["phase"],
            amendments=record["amendments"],
            omitted=True,
            full_turn_sha256=hashlib.sha256(raw.encode()).hexdigest(),
            path=source["path"],
            source_sha256=source["sha256"],
            displayed_spans=[dict(start=start + a, end=start + b - 1) for a, b in spans],
            partial_line=source.get("partial_line"),
            historical=True,
            interpretation="Subset of historical read output;full turn retained in session. No new read/edit authority or acceptance.",
        )
        if source.get("source_reference"):
            packet["source_reference"] = source["source_reference"]
        content = json.dumps(packet, ensure_ascii=False)
        for a, b in spans:
            content += (
                "\nLITERAL OBSERVED LINES "
                + str(start + a)
                + "-"
                + str(start + b - 1)
                + "\n"
                + fence("\n".join(lines[a:b]))
            )
        return [dict(role="user", content=content)]

    low, high, best = 0, (count + 1) // 2, None
    while low <= high:
        width = (low + high) // 2
        pair = view(width)
        if sum(estimate(m["content"]) + 6 for m in pair) <= budget:
            best = pair
            low = width + 1
        else:
            high = width - 1
    return best
