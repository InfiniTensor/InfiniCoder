"""Whole current definitions, then conservative semantic one-hop evidence.

No imports, model summaries, commands, gold materials or new relation inference.
The existing qualified resolver follows pure forwarders before semantic hops.
"""

import ast
import hashlib
from pathlib import PurePosixPath
import re

from infinicoder.memory.native_memory import _json, _sha
from infinicoder.analysis.qualified_symbols import anchors, module_names


def public_target_present(task, target):
    if "::" in target:
        path, name = target.split("::", 1)
        return path in task and name in task
    return target in task


def build_source_workset(index, public_task, explicit_public_targets, token_budget, count_text):
    """Index must be freshly refreshed. The returned report is audit-only.

    Text budgeting uses the caller's actual tokenizer. No selected record is
    cut. Incomplete static analysis is described, never treated as no effects.
    """
    if type(token_budget) is not int or token_budget < 0:
        raise ValueError("nonnegative source budget required")
    revision = index.revision()
    report = dict(
        revision=revision,
        public_task_sha256=_sha(public_task),
        budget=token_budget,
        selected=[],
        omitted=[],
        resolutions=[],
        semantic_depth=1,
        analysis="existing conservative static index;unknown is not absent",
        source_files={},
    )
    records, observed, targets = [], {}, []

    def current(path):
        row = index.db.execute(
            "SELECT hash,source,error FROM files WHERE path=?", (path,)
        ).fetchone()
        p = index.root / path
        try:
            if row is None or p.is_symlink() or any(parent.is_symlink() for parent in p.parents):
                return None
            if (
                not p.resolve().is_relative_to(index.root)
                or hashlib.sha256(p.read_bytes()).hexdigest() != row["hash"]
            ):
                return None
        except OSError:
            return None
        observed[path] = row["hash"]
        return row

    def offer(record):
        text = "\n".join([*records, _json(record)])
        if count_text(text) <= token_budget:
            records.append(_json(record))
            report["selected"].append(
                {k: record[k] for k in ("kind", "symbol", "path", "status") if k in record}
            )
            return True
        report["omitted"].append({k: record[k] for k in ("kind", "symbol", "path") if k in record})
        return False

    resolutions = anchors(index, public_task)
    for name in explicit_public_targets:
        if not isinstance(name, str) or not public_target_present(public_task, name):
            raise ValueError("source targets must occur in the original public task")
        if "::" in name:
            path, qualname = name.split("::", 1)
            p = PurePosixPath(path)
            if p.is_absolute() or ".." in p.parts or "\\" in path or ":" in path:
                raise ValueError("target must be workspace-relative")
            row = index.db.execute("SELECT id FROM symbols WHERE id=?", (name,)).fetchone()
            if row:
                paths = {r["path"] for r in index.files()}
                modules = sorted(module_names(path, paths), key=lambda n: (-len(n), n))
                resolved = (
                    index.resolve_qualified(modules[0] + "." + qualname, follow_forwarders=True)
                    if modules
                    else None
                )
                if resolved:
                    resolutions.insert(0, resolved)
                else:
                    targets.append((name, "explicit_public_target"))
            else:
                report["omitted"].append(dict(symbol=name, reason="public target not indexed"))
        elif re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+", name):
            resolutions.insert(0, index.resolve_qualified(name, follow_forwarders=True))
        else:
            raise ValueError("target must be a path::qualified_symbol or dotted public interface")
    chains = []
    for resolution in resolutions:
        if resolution in report["resolutions"]:
            continue
        report["resolutions"].append(resolution)
        if resolution["status"] == "unique":
            answer = resolution["resolutions"][0]
            targets.append((answer["symbol"], "public_interface:" + resolution["query"]))
            chains.append(
                dict(
                    kind="resolution_chain",
                    query=resolution["query"],
                    target=answer["symbol"],
                    status=resolution["status"],
                    chain=answer["chain"],
                    limitations=resolution["limitations"],
                )
            )
    # Task text only: never use hidden-check error identifiers as retrieval terms.
    # Multiple lexical candidates are explicitly candidates, not a unique route.
    if not targets:
        ranked = index.search(public_task, limit=6)
        functions = [r for r in ranked if r["kind"] == "function"]
        for row in (functions or ranked)[:2]:
            targets.append((row["id"], "lexical_candidate;ambiguity_not_resolved"))
    targets = list(dict.fromkeys(targets))[:2]
    edges, neighbors = [], []
    for sid, reason in targets:
        try:
            symbol = index.symbol(sid)
        except KeyError:
            report["omitted"].append(dict(symbol=sid, reason="target disappeared"))
            continue
        file = current(symbol["path"])
        if file is None:
            report["omitted"].append(dict(symbol=sid, reason="source race or unavailable"))
            continue
        start, end = symbol["line"], symbol["end"]
        if symbol["path"].endswith(".py"):
            try:
                nodes = [
                    n
                    for n in ast.walk(ast.parse(file["source"]))
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    and n.lineno == start
                ]
                if len(nodes) == 1:
                    start = min([start, *[d.lineno for d in nodes[0].decorator_list]])
            except SyntaxError:
                pass
        source = "".join(file["source"].splitlines(keepends=True)[start - 1 : end])
        record = dict(
            kind="target_definition",
            symbol=sid,
            path=symbol["path"],
            source_sha256=file["hash"],
            start=start,
            end=end,
            signature=symbol["signature"],
            selection=reason,
            status=("complete_definition" if not file["error"] else "parse_incomplete")
            if symbol["path"].endswith(".py")
            else "heuristic_declaration;extent_unknown",
            source=source,
        )
        if not offer(record):
            stub = {k: v for k, v in record.items() if k != "source"}
            stub["status"] = "body_omitted;read_original_location"
            offer(stub)
        for edge in index.relationships(sid, limit=50):
            # Only these original target edges expand. Never expand neighbors.
            other = edge.get("dst") if edge.get("src") == sid else edge.get("src")
            if edge["kind"] == "calls" and other and other != sid:
                neighbors.append((other, sid, edge))
            row = current(edge["path"])
            if row:
                edge = dict(
                    edge,
                    source_sha256=row["hash"],
                    source_line=row["source"].splitlines()[edge["line"] - 1],
                )
                edges.append(
                    dict(
                        kind="direct_relation",
                        of=sid,
                        relation=edge,
                        status="static_candidate"
                        if edge.get("dst")
                        else "unresolved_or_state_access",
                    )
                )
    # Target bodies precede route metadata. Chain evidence must retain every
    # file version, not consume the separate semantic depth allowance.
    for chain in chains:
        valid = all(
            (row := current(h["path"])) is not None and row["hash"] == h["source_sha256"]
            for h in chain["chain"]
        )
        if valid:
            if not offer(chain):
                offer(
                    dict(
                        kind="resolution_chain",
                        query=chain["query"],
                        target=chain["target"],
                        status="chain_omitted_for_budget",
                    )
                )
        else:
            report["omitted"].append(
                dict(kind="resolution_chain", query=chain["query"], reason="source race")
            )
    for edge in edges:
        offer(edge)
    seen = {sid for sid, _ in targets}
    for sid, origin, edge in neighbors:
        if sid in seen:
            continue
        seen.add(sid)
        try:
            row = index.symbol(sid)
        except KeyError:
            continue
        file = current(row["path"])
        if file:
            offer(
                dict(
                    kind="one_hop_interface",
                    symbol=sid,
                    path=row["path"],
                    source_sha256=file["hash"],
                    start=row["line"],
                    end=row["end"],
                    signature=row["signature"],
                    of=origin,
                    relation_direction="callee" if edge["src"] == origin else "caller",
                    status="static_candidate",
                )
            )
    for sid, _ in targets:
        try:
            row = index.symbol(sid)
            flow = index.use_def(
                row["path"], scope=row["qualname"], limit=24, expand_semantics=False
            )
        except (KeyError, ValueError, SyntaxError):
            continue
        for edge in flow["edges"]:
            offer(
                dict(
                    kind="use_def",
                    of=sid,
                    path=row["path"],
                    source_sha256=flow["source_sha256"],
                    edge=edge,
                    status="static_candidate;alias_and_runtime_effects_unknown",
                )
            )
    # Recheck all supporting versions after collecting relations and before
    # returning. Do not return cached facts after a concurrent edit or rename.
    if index.revision() != revision or any(
        (row := current(p)) is None or row["hash"] != sha for p, sha in list(observed.items())
    ):
        report.update(status="omitted_source_race", source_files=observed)
        return "", report
    report.update(
        status="ready",
        source_files=observed,
        tokens=count_text("\n".join(records)),
        selected_targets=[sid for sid, _ in targets],
    )
    return "\n".join(records), report
