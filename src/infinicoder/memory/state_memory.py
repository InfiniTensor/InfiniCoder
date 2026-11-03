"""Compact constructor/default initializer evidence;not live object values."""

import hashlib
import re
from infinicoder.runtime.worker import _native_mask


def close(mask, start):
    pairs = {"(": ")", "{": "}", "[": "]"}
    stack = []
    for n in range(start, len(mask)):
        c = mask[n]
        if c in pairs:
            stack.append(pairs[c])
        elif c in ")}]":
            if not stack or stack.pop() != c:
                raise ValueError("unbalanced source initializer")
            if not stack:
                return n
    raise ValueError("incomplete source initializer")


def clauses(text):
    """Yield exact initializer-list clauses from a known Clang constructor."""
    mask = _native_mask(text)
    opening = mask.find("(")
    if opening < 0:
        return []
    at = close(mask, opening) + 1
    while at < len(mask) and mask[at].isspace():
        at += 1
    if at >= len(mask) or mask[at] != ":":
        return []
    at += 1
    rows = []
    while at < len(mask) and len(rows) < 32:
        while at < len(mask) and (mask[at].isspace() or mask[at] == ","):
            at += 1
        match = re.match(r"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*\s*", mask[at:])
        if not match:
            break
        name = match.group().strip()
        start = at
        at += match.end()
        if at >= len(mask) or mask[at] not in "({":
            break
        end = close(mask, at)
        if end - at <= 240:
            rows.append(
                dict(
                    name=name,
                    expression=text[at + 1 : end],
                    text=text[start : end + 1],
                    offset=start,
                )
            )
        at = end + 1
    return rows


def collect(plan, workspace):
    owner = plan["scope"]["qualname"].rsplit("::", 1)[0]
    fields = [r for r in plan["records"] if r["kind"] == "field_decl"]
    by_name = {}
    for r in fields:
        if r["name"].rsplit("::", 1)[0] == owner:
            by_name.setdefault(r["name"].split("::")[-1], []).append(r)
    groups = {}
    versions = {}
    omitted = 0
    for source in plan["records"]:
        if source["kind"] != "constructor" or source["name"].rsplit("::", 1)[0] != owner:
            continue
        try:
            values = clauses(source["text"])
        except ValueError:
            omitted += 1
            continue
        for value in values:
            matches = by_name.get(value["name"], [])
            if (
                len(matches) != 1
                or not value["expression"].strip()
                or len(value["expression"]) > 160
            ):
                continue
            field = matches[0]
            key = (field["id"], value["expression"])
            line = source["line"] + source["text"].count("\n", 0, value["offset"])
            if key not in groups:
                if len(groups) >= 8:
                    omitted += 1
                    continue
                groups[key] = dict(
                    path=source["path"],
                    source_sha256=source["source_sha256"],
                    line=line,
                    end=line + value["text"].count("\n"),
                    name=field["name"],
                    field_id=field["id"],
                    text=value["text"],
                    expression=value["expression"],
                    constructor_refs=[],
                    kind="state_initializer_source",
                    view="constructor_initializer_clause",
                    origin="current_source_state_memory",
                    priority=0,
                    runtime_value_known=False,
                    precondition="Selected constructor ran and no subsequent overwrite;constructor body,load paths and callees may change state.",
                )
            groups[key]["constructor_refs"].append(source["id"])
            versions[source["path"]] = source["source_sha256"]
    # Declaration defaults remain separate from constructor path evidence.
    for field in fields:
        name = field["name"].split("::")[-1]
        text = field["text"]
        mask = _native_mask(text)
        match = re.search(r"\b" + re.escape(name) + r"\s*=(?!=)", mask)
        if not match or len(text) > 240:
            continue
        expression = text[match.end() :].strip().rstrip(";")
        if not expression or len(groups) >= 8:
            continue
        key = (field["id"], expression)
        groups.setdefault(
            key,
            dict(
                path=field["path"],
                source_sha256=field["source_sha256"],
                line=field["line"],
                end=field["end"],
                name=field["name"],
                field_id=field["id"],
                text=text,
                expression=expression,
                constructor_refs=[],
                kind="state_initializer_source",
                view="declaration_default_initializer",
                origin="current_source_state_memory",
                priority=0,
                runtime_value_known=False,
                precondition="Default used only if the selected constructor does not override it;later writes remain possible.",
            ),
        )
        versions[field["path"]] = field["source_sha256"]
    try:
        current = all(
            hashlib.sha256(workspace.path(p).read_bytes()).hexdigest() == h
            for p, h in versions.items()
        )
    except (ValueError, OSError):
        current = False
    if not current:
        return [], {}, dict(status="source_changed", runtime_values_known=False)
    return (
        list(groups.values()),
        versions,
        dict(
            status="literal_initializers",
            records=len(groups),
            omitted=omitted,
            runtime_values_known=False,
            limitation="Expressions are conditional initializers,not live values. Defaults may be overridden;selected constructor,bodies,delegation,loads,aliases and later writes are not proven.",
        ),
    )
