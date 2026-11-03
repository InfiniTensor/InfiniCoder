"""Small source-linked static relations;group alternatives,never choose a branch."""

from copy import deepcopy
import hashlib


def relation(flow, source_ref, groups=3, calls=1):
    if flow.get("status") != "static_question_flow":
        return None
    grouped = {}
    for edge in flow.get("use_def", []):
        key = (edge["scope"], edge["name"], edge["use_line"])
        row = grouped.setdefault(
            key,
            dict(scope=key[0], name=key[1], use_line=key[2], definition_lines=[], confidence=[]),
        )
        row["definition_lines"].append(edge["definition_line"])
        row["confidence"].append(edge.get("confidence", "unknown"))
    rows = list(grouped.values())
    for row in rows:
        row["definition_lines"] = sorted(set(row["definition_lines"]))
        row["confidence"] = sorted(set(row["confidence"]))
    links = flow.get("input_output_links", [])
    updates = flow.get("heap_updates", [])
    if not rows and not links and not updates:
        return None
    result = dict(
        kind="source_syntax_candidates",
        status=flow["status"],
        symbol=flow["symbol"],
        source_ref=source_ref,
        view=flow["view"],
        use_def=rows[:groups],
        input_output_links=deepcopy(links[:calls]),
        omitted=dict(
            use_groups=max(0, len(rows) - groups),
            calls=max(0, len(links) - calls),
            fragments=len(flow.get("fragments", [])),
            original=deepcopy(flow.get("omissions", {})),
        ),
        uncertainty="Scope-local static candidates;branches,exceptions,aliases,callee effects and ownership remain unresolved.",
        expand=dict(tool="use_def", arguments=dict(path=flow["path"])),
    )
    if "heap_updates" in flow:
        result.update(
            heap_updates=deepcopy(updates[:groups]),
            heap_update_analysis=deepcopy(flow["heap_update_analysis"]),
        )
        result["omitted"]["heap_updates"] = max(0, len(updates) - groups)
        result["expand"] = dict(tool="function_summary", arguments=dict(symbol_id=flow["symbol"]))
    return result


def dependency_versions(packet, workspace):
    """All shown source/callee facts keep their dependency leases,even unexpanded."""
    versions = {}
    pending = [packet]
    visited = 0
    while pending:
        node = pending.pop()
        visited += 1
        if visited > 10000:
            raise ValueError("Problem dependency metadata exceeds bound")
        if isinstance(node, dict):
            if (
                "path" not in node
                and isinstance(node.get("function_summary"), str)
                and "source_sha256" in node
            ):
                pending.append(
                    dict(
                        path=node["function_summary"].split("::", 1)[0],
                        source_sha256=node["source_sha256"],
                    )
                )
            for path_key, sha_key in (
                ("path", "source_sha256"),
                ("callee_path", "callee_source_sha256"),
            ):
                if path_key not in node or sha_key not in node:
                    continue
                path, sha = node[path_key], node[sha_key]
                if not isinstance(path, str) or not isinstance(sha, str):
                    raise ValueError("Invalid source dependency")
                current = hashlib.sha256(workspace.path(path).read_bytes()).hexdigest()
                if current != sha or path in versions and versions[path] != sha:
                    raise ValueError("Source dependency changed while assembling problem packet")
                versions[path] = sha
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)
    return versions


def first_pair(candidates, choice, index, packet, offer):
    """Reserve a complete source record and one bounded relation before other files."""
    from infinicoder.memory.flow_memory import slice_function
    from infinicoder.memory.source.working_set import _windows

    anchors = {row["symbol"]: row["line"] for row in choice["anchors"]}
    ordered = sorted(
        candidates,
        key=lambda row: (
            row.get("symbol") not in anchors,
            not row.get("implementation_excerpt_complete", False),
            row.get("symbol") is None,
        ),
    )
    for candidate in ordered:
        symbol = candidate.get("symbol")
        if not symbol:
            continue
        try:
            flow = slice_function(index, symbol, view=choice["view"], max_lines=8)
        except (ValueError, KeyError, SyntaxError):
            continue
        for source in _windows(candidate, anchors.get(symbol)):
            source_ref = len(packet["sources"])
            if not offer("sources", source):
                continue
            for groups, calls in ((3, 1), (2, 1), (1, 0)):
                small = relation(flow, source_ref, groups, calls)
                if small is not None and offer("relations", small):
                    return source
            packet["sources"].pop()
    return None
