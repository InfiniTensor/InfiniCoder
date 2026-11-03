"""Controller-selected working questions and versioned source anchors.

Selection is a routing hypothesis, not a source fact or edit authorization.
Runtime output may suggest a location, but only current workspace bytes provide
the returned code. No model call, command, permission change or read receipt is
created by assembling this view.
"""

from infinicoder._resources import package_root

import hashlib
import re
from infinicoder.runtime.command_evidence import locations
from infinicoder.memory.task_memory import TaskMemory


def identity():
    root = package_root()
    return hashlib.sha256(
        b"".join(
            (root / name).read_bytes()
            for name in (
                "memory/question_memory.py",
                "memory/source/working_set.py",
                "memory/task_memory.py",
                "runtime/command_evidence.py",
                "runtime/shell_loop.py",
            )
        )
    ).hexdigest()


def _strings(value, limit=12000):
    """Bound output inspection; strings remain untrusted routing observations."""
    parts = []
    remaining = limit
    pending = [value]
    visited = 0
    while pending and remaining > 0 and visited < 128:
        item = pending.pop()
        visited += 1
        if isinstance(item, str):
            text = item[-remaining:]
            parts.append(text)
            remaining -= len(text)
        elif isinstance(item, dict):
            pending.extend(reversed(list(item.values())[:32]))
        elif isinstance(item, (list, tuple)):
            pending.extend(reversed(item[:32]))
    return "\n".join(parts)


def select(index, spec, state, unit, workspace, mounts=()):
    """Use actual versioned reads/edits and output locations to choose a view."""
    memory = TaskMemory(index)
    anchors = []
    fallback = []
    unknown = []
    seen = set()
    query = state.get("memory_focus", {}).get("question") or unit.goal or spec.task
    query = str(query)[:1200]
    explicit = state.get("memory_focus", {}).get("view", "auto")
    if explicit not in ("auto", "values", "state", "resources", "interfaces"):
        explicit = "auto"

    def offer(path, line, basis, expected_hash=None):
        if len(anchors) + len(fallback) >= 4:
            return
        try:
            safe = workspace.path(path)
            file = memory._current(path)
            if file is None or safe.resolve() != index.root.joinpath(path).resolve():
                unknown.append(
                    dict(path=path, reason="source stale,missing or outside indexed workspace")
                )
                return
            if expected_hash is not None and file["hash"] != expected_hash:
                unknown.append(
                    dict(path=path, reason="observation source version no longer current")
                )
                return
            lines = file["source"].splitlines()
            if type(line) is not int or not 1 <= line <= len(lines):
                unknown.append(dict(path=path, reason="no complete current line"))
                return
            symbols = [
                dict(v)
                for v in index.db.execute(
                    "SELECT id,path,qualname,line,end,kind FROM symbols WHERE path=? AND line<=? AND end>=? ORDER BY (end-line),line,id",
                    (path, line, line),
                )
            ]
            if symbols:
                symbol = symbols[0]
                if symbol["id"] in seen:
                    return
                seen.add(symbol["id"])
                anchors.append(
                    dict(
                        symbol=symbol["id"],
                        path=path,
                        line=line,
                        source_sha256=file["hash"],
                        basis=basis,
                    )
                )
            else:
                if path in seen:
                    return
                seen.add(path)
                start = max(1, line - 4)
                end = min(len(lines), line + 7)
                fallback.append(
                    dict(
                        path=path,
                        source_sha256=file["hash"],
                        line=line,
                        start=start,
                        end=end,
                        source="\n".join(lines[start - 1 : end]),
                        basis=basis,
                        limitations=[
                            "No containing parsed symbol;source excerpt only,relations unknown."
                        ],
                    )
                )
        except (OSError, ValueError, KeyError):
            unknown.append(dict(path=str(path)[:160], reason="source location unavailable"))

    output = _strings(state.get("last_failure"))
    observed = []
    if output:
        for item in locations(output, workspace, mounts):
            if item.get("path"):
                observed.append(
                    dict(
                        path=item["path"],
                        line=item["line"],
                        mapping=item["mapping"],
                        trust="reported location is untrusted;current source is checked separately",
                    )
                )
                offer(
                    item["path"],
                    item["line"],
                    "execution_location_routing_hypothesis",
                    item["sha256"],
                )
            else:
                unknown.append(
                    dict(
                        reported_path=item.get("reported_path"), reason="unmapped reported location"
                    )
                )
    failure_anchor = bool(anchors or fallback)
    # A successful edit provides a current-file anchor, but never authorizes
    # another edit. Prefer the prior observed span over the first helper in a file.
    coverage = state.get("read_coverage", {})
    changed = list(dict.fromkeys(state.get("changed_files", [])))[-3:]
    history = state.get("shell_loop_history", [])
    last_source_tool = next(
        (
            row.get("action", {}).get("tool")
            for row in reversed(history)
            if isinstance(row.get("action"), dict)
            and row["action"].get("tool")
            in (
                "read",
                "read_symbol",
                "read_function_body",
                "workpack",
                "edit",
                "edit_symbol",
                "edit_function_body",
            )
            and not row.get("observation", {}).get("error")
        ),
        None,
    )
    edited = (
        changed
        if last_source_tool in ("edit", "edit_symbol", "edit_function_body") or not history
        else []
    )
    for path in reversed(edited):
        file = memory._current(path)
        if file is None:
            continue
        previous = coverage.get(path, {})
        spans = previous.get("spans", []) if previous.get("sha256") == file["hash"] else []
        line = spans[-1][0] if spans else None
        if line is None or not 1 <= line <= len(file["source"].splitlines()):
            relevant = [row for row in index.search(query, limit=32) if row["path"] == path]
            line = relevant[0]["line"] if relevant else 1
        offer(path, line, "current_edited_source;read_dependencies_not_write_scope")
    for path, value in reversed(list(coverage.items())):
        spans = value.get("spans", [])
        if not isinstance(value.get("sha256"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", value["sha256"]
        ):
            unknown.append(
                dict(path=path, reason="unversioned read is not a current source observation")
            )
            continue
        if spans:
            offer(path, spans[-1][0], "actual_read_span", value.get("sha256"))
        if len(anchors) + len(fallback) >= 4:
            break
    if explicit != "auto":
        view = explicit
        basis = "model_question_view_not_fact"
    elif re.search(
        r"releas|lifetim|owner|leak|alloc|cancel|close|释放|资源|生命周期|泄漏", query, re.I
    ):
        view = "resources"
        basis = "task_resource_question"
    elif failure_anchor:
        view = "values"
        basis = "execution_location_question"
    elif edited:
        view = "interfaces"
        basis = "modified_interface_consumers"
    elif anchors:
        view = "values"
        basis = "follow_inputs_outputs_from_current_read"
    else:
        view = "interfaces"
        basis = "locate_modules_and_interfaces"
    direction = (
        "incoming"
        if view == "interfaces" and edited and not failure_anchor
        else "outgoing"
        if view == "values" and not failure_anchor
        else "both"
    )
    return dict(
        question=query,
        view=view,
        direction=direction,
        selection_basis=basis,
        selection_is_hypothesis=True,
        anchors=anchors,
        fallback_sources=fallback,
        runtime_locations=observed,
        unknowns=unknown[:4],
        source_revision=index.revision(),
        analysis_identity=identity(),
        policy="Current source facts are versioned;reported locations and model questions are not facts. No edit scope or read-before-write authority is granted.",
    )


def pack(index, spec, state, unit, workspace, max_tokens, estimator, exclude_paths=(), mounts=()):
    if type(max_tokens) is not int or max_tokens < 0:
        raise ValueError("Nonnegative token budget required")
    choice = select(index, spec, state, unit, workspace, mounts)
    base = dict(
        revision=choice["source_revision"],
        source_revision=choice["source_revision"],
        analysis_identity=choice["analysis_identity"],
        selection_basis=choice["selection_basis"],
        view=choice["view"],
        direction=choice["direction"],
        anchors=choice["anchors"],
        runtime_locations=choice["runtime_locations"],
        unknowns=choice["unknowns"],
        max_tokens=max_tokens,
        selected=[],
        fallback_sources_included=0,
        included=False,
    )
    focus = list(dict.fromkeys([row["path"] for row in choice["anchors"]] + list(unit.write_set)))
    query = (
        choice["question"]
        + " "
        + " ".join(row["symbol"].split("::")[-1] for row in choice["anchors"])
    )
    from infinicoder.memory.source.working_set import pack as working_set

    text, report = working_set(
        TaskMemory(index), choice, query, focus, max_tokens, estimator, exclude_paths
    )
    return text, dict(base, **report)
