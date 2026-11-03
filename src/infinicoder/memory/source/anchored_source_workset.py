"""Common public-root repair and independently optional dependency exposure."""

import ast
import re

from infinicoder.memory.native_memory import _json, _sha
from infinicoder.memory.source.source_dependency_view import CurrentSources, expand, location
from infinicoder.memory.source.task_source_roots import resolve_public_roots


def _python_nodes(source, task, hint):
    tree = ast.parse(source)
    definitions = []

    def visit(body, prefix=""):
        for node in body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            name = prefix + node.name
            definitions.append((name, node))
            visit(node.body, name + ".")

    visit(tree.body)
    if hint:
        exact = [(q, n) for q, n in definitions if q == hint]
        if exact:
            return exact
    dotted = set(re.findall(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b", task))
    exact = [(q, n) for q, n in definitions if q in dotted]
    if exact:
        return exact
    names = set(re.findall(r"\b[A-Za-z_]\w*\b", task))
    functions = [
        (q, n) for q, n in definitions if n.name in names and not isinstance(n, ast.ClassDef)
    ]
    return functions or [(q, n) for q, n in definitions if n.name in names]


def build_anchored_workset(
    index,
    public_task,
    public_targets,
    token_budget,
    count_text,
    *,
    dependency_policy=None,
    workspace_aliases=(),
    remaining=None,
):
    if dependency_policy not in (None, "anchored_static_v1"):
        raise ValueError("unknown source dependency policy")
    if type(token_budget) is not int or token_budget < 0:
        raise ValueError("nonnegative source budget required")
    revision = index.revision()
    sources = CurrentSources(index, remaining)
    records = []
    targets = []
    report = dict(
        revision=revision,
        public_task_sha256=_sha(public_task),
        budget=token_budget,
        source_policy="public_roots_v1",
        dependency_policy=dependency_policy,
        source_files={},
        selected=[],
        omitted=[],
    )

    def offer(record):
        text = "\n".join([*records, _json(record)])
        if count_text(text) <= token_budget:
            records.append(_json(record))
            report["selected"].append(
                {k: record[k] for k in ("kind", "path", "symbol", "status") if k in record}
            )
            return True
        report["omitted"].append({k: record[k] for k in ("kind", "path", "symbol") if k in record})
        return False

    roots = resolve_public_roots(
        index, public_task, public_targets, workspace_aliases=workspace_aliases
    )
    report["roots"] = roots
    try:
        # This entire target phase is identical in both arms. Do not reduce the
        # control's target record to manufacture dependency-policy headroom.
        for root in roots["roots"]:
            file = sources.read(root["path"])
            if not file:
                report["omitted"].append(
                    dict(path=root["path"], reason="source_unavailable_or_changed")
                )
                continue
            lines = file["source"].splitlines(keepends=True)
            start, end = 1, len(lines)
            symbol = root.get("symbol")
            nodes = []
            hint = root.get("symbol_hint")
            if root["path"].endswith(".py"):
                try:
                    matches = _python_nodes(file["source"], public_task, hint)
                    nodes = [n for _, n in matches]
                    if len(matches) == 1:
                        sid = root["path"] + "::" + matches[0][0]
                        if index.db.execute("SELECT id FROM symbols WHERE id=?", (sid,)).fetchone():
                            symbol = sid
                except SyntaxError:
                    pass
            if symbol:
                row = index.db.execute(
                    "SELECT line,end FROM symbols WHERE id=?", (symbol,)
                ).fetchone()
                if row and root["path"].endswith(".py"):
                    start, end = row["line"], row["end"]
            elif len(nodes) == 1:
                node = nodes[0]
                start = min([node.lineno, *[d.lineno for d in node.decorator_list]])
                end = node.end_lineno
            if root.get("line"):
                start = max(1, min(len(lines), root["line"]))
                end = max(start, min(len(lines), root["end_line"]))
            source = "".join(lines[start - 1 : end])
            status = "complete_python_definition" if nodes and len(nodes) == 1 else "complete_file"
            if start != 1 or end != len(lines):
                status = "source_extent;semantic_binding_unknown"
            record = location(
                file,
                start,
                end,
                "target_source",
                symbol=symbol,
                source=source,
                selection=root["selection"],
                status=status,
            )
            if count_text(_json(record)) > min(token_budget, 1000):
                # Native methods may not be in the coarse catalog. A bounded
                # excerpt at the literal public identifier remains useful and
                # explicitly does not claim a complete function extent.
                hints = [hint] if hint else root.get("identifier_hints", [])
                positions = [
                    i
                    for i, line in enumerate(lines, 1)
                    if any(re.search(r"(?<!\w)" + re.escape(h) + r"(?!\w)", line) for h in hints)
                ]
                center = root.get("line") or (positions[0] if positions else start)
                width = 48
                while width >= 2:
                    a = max(1, center - min(8, width // 4))
                    b = min(len(lines), a + width - 1)
                    record = location(
                        file,
                        a,
                        b,
                        "target_source",
                        symbol=symbol,
                        source="".join(lines[a - 1 : b]),
                        selection=root["selection"],
                        status="bounded_excerpt;extent_and_omitted_code_unknown",
                    )
                    if count_text(_json(record)) <= min(token_budget, 1000):
                        break
                    width //= 2
            if offer(record):
                targets.append(record)
            else:
                stub = {k: v for k, v in record.items() if k != "source"}
                stub["status"] = "body_omitted;read_original_location"
                if offer(stub):
                    targets.append(stub)
        for root in roots["roots"]:
            chain = root.get("resolution_chain", [])
            if chain and all(
                (row := sources.read(h["path"])) and row["hash"] == h["source_sha256"]
                for h in chain
            ):
                offer(
                    dict(
                        kind="root_resolution",
                        path=root["path"],
                        evidence_chain=chain,
                        status="static_candidate;runtime_import_binding_unknown",
                    )
                )
        if not roots["roots"]:
            offer(
                dict(
                    kind="source_roots_unknown",
                    status="unresolved_public_anchor;no_unrelated_fallback",
                    unknown=roots["unknown"],
                )
            )
        if dependency_policy:
            # Only the remaining budget is available; target records are immutable.
            report["dependencies"] = expand(sources, roots["roots"], targets, offer)
    except TimeoutError:
        report["analysis_unknown"] = "bounded_analysis_timeout;available_target_records_retained"
    # Every record, including a route through a header, inherits supporting
    # versions. Negative results also depend on the full index revision.
    if index.revision() != revision or not sources.stable():
        report.update(status="omitted_source_race", source_files=sources.observed)
        return "", report
    report.update(
        status="ready",
        source_files=sources.observed,
        tokens=count_text("\n".join(records)),
        target_records=targets,
    )
    return "\n".join(records), report
