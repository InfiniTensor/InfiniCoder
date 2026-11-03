"""Source-derived, persistent module memory with budgeted graph expansion.

No model summaries are stored as facts. Every view belongs to an index revision;
source excerpts and dataflow retain file identities and analysis limitations.
"""

from infinicoder._resources import package_root

from collections import deque
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath

from infinicoder.analysis.code_context import words


def identity():
    root = package_root()
    from infinicoder.analysis.native_effects import identity as native_effects_identity

    return hashlib.sha256(
        Path(__file__).read_bytes()
        + (root / "analysis/code_context.py").read_bytes()
        + (root / "analysis/symbol_tools.py").read_bytes()
        + (root / "languages/clang_adapter.py").read_bytes()
        + (root / "languages/native_adapter.py").read_bytes()
        + native_effects_identity().encode()
    ).hexdigest()


class RepositoryMemory:
    def __init__(self, index):
        self.index = index

    def modules(self):
        index = self.index
        revision = index.revision()
        index.db.execute(
            "CREATE TABLE IF NOT EXISTS memory_modules(module TEXT PRIMARY KEY, revision TEXT, analysis TEXT, facts TEXT)"
        )
        rows = index.db.execute("SELECT revision,analysis FROM memory_modules LIMIT 1").fetchone()
        if rows and tuple(rows) == (revision, identity()):
            return [
                json.loads(r[0])
                for r in index.db.execute("SELECT facts FROM memory_modules ORDER BY module")
            ]
        groups = {}
        for file in index.files():
            parent = PurePosixPath(file["path"]).parent
            ancestors = [parent, *parent.parents]
            for module in ancestors:
                key = str(module)
                item = groups.setdefault(
                    key,
                    dict(
                        module=key,
                        files=0,
                        direct_files=0,
                        languages={},
                        parse_errors=0,
                        interfaces=[],
                    ),
                )
                item["files"] += 1
                item["direct_files"] += module == parent
                item["languages"][file["language"]] = item["languages"].get(file["language"], 0) + 1
                item["parse_errors"] += bool(file["error"])
            # Deterministic interface hints, not an architectural interpretation.
            for symbol in index.db.execute(
                'SELECT id,signature,line,confidence FROM symbols WHERE path=? AND kind IN ("class","type","function") ORDER BY line LIMIT 4',
                (file["path"],),
            ):
                item = groups[str(parent)]
                if len(item["interfaces"]) < 6:
                    item["interfaces"].append(
                        dict(symbol, path=file["path"], source_sha256=file["hash"])
                    )
        with index.db:
            index.db.execute("DELETE FROM memory_modules")
            index.db.executemany(
                "INSERT INTO memory_modules VALUES(?,?,?,?)",
                [
                    (key, revision, identity(), json.dumps(value, ensure_ascii=False))
                    for key, value in sorted(groups.items())
                ],
            )
        return list(groups.values())

    def route(self, query, limit=8):
        if type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError("limit must be 1..50")
        groups = self.modules()
        scores = {}
        for rank, hit in enumerate(self.index.search(query, limit=50)):
            parent = PurePosixPath(hit["path"]).parent
            for depth, module in enumerate([parent, *parent.parents]):
                scores[str(module)] = scores.get(str(module), 0) + 1 / ((rank + 1) * (depth + 1))
        terms = set(words(query))
        for group in groups:
            group["score"] = round(
                scores.get(group["module"], 0) + len(terms & set(words(group["module"]))) * 0.2, 6
            )
        groups.sort(key=lambda v: (-v["score"], v["module"]))
        result = dict(
            revision=self.index.revision(),
            analysis_identity=identity(),
            level="modules",
            groups=groups[:limit],
            omitted=max(0, len(groups) - limit),
            next="memory_expand module;symbol/relationships/structure/use_def;then read exact source. Unknown edges are not absent dependencies.",
        )
        # Tool observations must also be bounded as whole JSON. Detailed
        # interface records remain in SQLite and are available on expansion.
        while len(json.dumps(result, ensure_ascii=False)) > 3800:
            candidates = [g for g in reversed(result["groups"]) if g["interfaces"]]
            if candidates:
                candidates[0]["interfaces"].pop()
            elif result["groups"]:
                result["groups"].pop()
                result["omitted"] += 1
            else:
                break
        return result

    def pack(self, query, max_tokens, estimator, focus_paths=(), exclude_paths=()):
        if type(max_tokens) is not int or max_tokens < 0:
            raise ValueError("max_tokens must be a nonnegative int")
        index = self.index
        revision = index.revision()
        header = dict(
            code_memory=dict(
                revision=revision,
                analysis_identity=identity(),
                policy="Source facts only;static edges need verification. Expand only relevant modules/symbols;use read for omitted source.",
                expansion_tools=[
                    "memory_route",
                    "memory_expand",
                    "relationships",
                    "structure",
                    "use_def",
                    "read",
                ],
            )
        )
        records = []

        def offer(row):
            text = "\n".join([*records, json.dumps(row, ensure_ascii=False)])
            if estimator.estimate(text) <= max_tokens:
                records.append(json.dumps(row, ensure_ascii=False))
                return True
            return False

        if not offer(header):
            return "", dict(
                revision=revision, tokens=0, selected=[], omitted=1, max_tokens=max_tokens
            )
        # Resolve explicit public interfaces before broad lexical matches. The
        # existing resolver refuses conditional/rebound aliases and callbacks.
        from infinicoder.analysis.qualified_symbols import anchors

        resolutions = anchors(index, query)
        seeds = []
        anchor_records = []
        for anchor in resolutions:
            if anchor["status"] == "unique":
                answer = anchor["resolutions"][0]
                seeds.append((answer["symbol"], 0, "public interface forwarding"))
                chain = answer["chain"]
                anchor_records.append(
                    dict(
                        public_interface=anchor["query"],
                        target=answer["symbol"],
                        confidence="static_candidate",
                        revision=revision,
                        entry=chain[0],
                        terminal=chain[-1],
                        hops=len(chain),
                        expansion="resolve_qualified(name,follow_forwarders=True)",
                    )
                )
        focused = []
        for path in focus_paths:
            if any(char in path for char in "*?["):
                continue
            if path.endswith("/"):
                # Avoid LIKE metacharacters in a caller-supplied path.
                rows = index.db.execute("SELECT id,path,line FROM symbols ORDER BY path,line")
                focused.extend(row["id"] for row in rows if row["path"].startswith(path))
            else:
                focused.extend(
                    r[0]
                    for r in index.db.execute(
                        "SELECT id FROM symbols WHERE path=? ORDER BY line LIMIT 8", (path,)
                    )
                )
        ranked = index.search(query, limit=16)
        seeds += [(sid, 0, "current scope") for sid in focused[:8]]
        seeds += [(row["id"], 0, "task retrieval") for row in ranked[:6]]
        queue = deque(seeds)
        seen = set()
        definition_ranges = {}
        selected = []
        omitted = 0
        # Expand dependencies breadth-first. Enqueue neighbors immediately so
        # exact interface chains are not buried under all lexical candidates.
        while queue and len(seen) < 36:
            sid, depth, reason = queue.popleft()
            if sid in seen:
                continue
            seen.add(sid)
            try:
                symbol = index.symbol(sid)
            except KeyError:
                continue
            file = index.db.execute(
                "SELECT hash,source,error FROM files WHERE path=?", (symbol["path"],)
            ).fetchone()
            # Never serve a previously indexed fact after concurrent source edits.
            try:
                source_path = (index.root / symbol["path"]).resolve()
                current = (
                    file is not None
                    and source_path.is_relative_to(index.root)
                    and hashlib.sha256(source_path.read_bytes()).hexdigest() == file["hash"]
                )
            except OSError:
                current = False
            if not current:
                omitted += 1
                continue
            edges = index.relationships(sid, limit=12)
            neighbors = []
            for edge in edges:
                other = edge.get("dst") if edge.get("src") == sid else edge.get("src")
                if other and other != sid and depth < 3 and other not in seen:
                    neighbors.append((other, depth + 1, "call/reference dependency of " + sid))
            queue.extendleft(reversed(neighbors[:6]))
            source_already_present = symbol["path"] in exclude_paths
            start = index._implementation_start(
                symbol["path"], file["hash"], file["source"], symbol
            )
            end = min(symbol["end"], start + 31)
            lines = file["source"].splitlines()
            # Long individual source lines are omitted whole, never clipped into
            # an apparently complete statement.
            excerpt = []
            for line in lines[start - 1 : end]:
                if sum(map(len, excerpt)) + len(line) > 5000:
                    break
                excerpt.append(line)
            row = dict(
                symbol=sid,
                path=symbol["path"],
                source_sha256=file["hash"],
                line=symbol["line"],
                signature=symbol["signature"],
                confidence=symbol["confidence"],
                reason=reason,
                depth=depth,
                excerpt="\n".join(excerpt),
                excerpt_line=start,
                excerpt_end=start + len(excerpt) - 1,
                implementation_excerpt_complete=start + len(excerpt) - 1 == symbol["end"],
                source_unit_complete=False,
            )
            # Prefer a complete declaration, including Python decorators and
            # docstrings, over an arbitrary 32-line prefix when it fits. A
            # complete lexical unit still does not close over its dependencies.
            unit_start = symbol["line"]
            unit_supported = symbol["path"].endswith(".py") and not file["error"]
            if symbol["path"].endswith(".py") and symbol["kind"] in ("function", "class"):
                key = (symbol["path"], file["hash"])
                if key not in definition_ranges:
                    from infinicoder.analysis.symbol_tools import _definitions

                    try:
                        definitions = _definitions(ast.parse(file["source"]))
                        ranges = {}
                        for qualified, node in definitions:
                            ranges.setdefault(qualified, []).append(
                                (
                                    node.lineno,
                                    min([node.lineno, *[d.lineno for d in node.decorator_list]]),
                                    node.end_lineno,
                                )
                            )
                        definition_ranges[key] = ranges
                    except (SyntaxError, ValueError, RecursionError):
                        definition_ranges[key] = {}
                matches = definition_ranges[key].get(symbol["qualname"], [])
                unit_supported = (
                    unit_supported
                    and len(matches) == 1
                    and matches[0][0] == symbol["line"]
                    and matches[0][2] == symbol["end"]
                )
                if unit_supported:
                    unit_start = matches[0][1]
            row["source_retrieval"] = dict(
                tool="read",
                arguments=dict(
                    path=symbol["path"],
                    start=unit_start,
                    count=min(300, max(1, symbol["end"] - unit_start + 1)),
                ),
                source_sha256=file["hash"],
                unit_end=symbol["end"],
            )
            complete = dict(
                row,
                excerpt="\n".join(lines[unit_start - 1 : symbol["end"]]),
                excerpt_line=unit_start,
                excerpt_end=symbol["end"],
                implementation_excerpt_complete=True,
                source_unit_complete=True,
                dependency_closure="unknown;follow versioned relationships and exact source",
            )
            selection = dict(
                symbol=sid, path=symbol["path"], source_sha256=file["hash"], depth=depth
            )
            if source_already_present:
                # Do not duplicate mandatory source, but still expand its
                # calls,fields and use-def: those are distinct analysis views.
                selected.append(
                    dict(
                        symbol=sid,
                        path=symbol["path"],
                        source_sha256=file["hash"],
                        source_already_present=True,
                    )
                )
            elif unit_supported and offer(complete):
                selected.append(dict(selection, source_unit_complete=True))
            elif offer(row):
                selected.append(dict(selection, source_unit_complete=False))
            else:
                # Small-context models still see the interface and exact read
                # address when a complete source record cannot fit.
                stub = {
                    k: v
                    for k, v in row.items()
                    if k
                    not in (
                        "excerpt",
                        "excerpt_line",
                        "excerpt_end",
                        "implementation_excerpt_complete",
                    )
                }
                stub["source_omitted"] = True
                if not offer(stub):
                    omitted += 1
                    continue
                selected.append(
                    dict(
                        symbol=sid,
                        path=symbol["path"],
                        source_sha256=file["hash"],
                        source_omitted=True,
                    )
                )
            if edges:
                omitted += not offer(
                    dict(relationships_of=sid, source_sha256=file["hash"], edges=edges)
                )
            if symbol.get("fields"):
                omitted += not offer(
                    dict(structure_of=sid, source_sha256=file["hash"], fields=symbol["fields"])
                )
            if depth == 0 and symbol["kind"] == "function":
                try:
                    python = symbol["path"].endswith(".py")
                    flow = index.use_def(
                        symbol["path"],
                        limit=500,
                        **({"scope": symbol["qualname"]} if python else {}),
                        **(
                            {"expand_semantics": False}
                            if getattr(index, "native_index_policy", "eager") == "on_demand"
                            else {}
                        ),
                    )
                    edges = (
                        flow.get("edges", [])
                        if python
                        else [
                            edge
                            for edge in flow.get("edges", [])
                            if edge.get("scope") == symbol["qualname"]
                            or symbol["line"] <= edge.get("use_line", -1) <= symbol["end"]
                        ]
                    )
                    if edges:
                        omitted += not offer(
                            dict(
                                use_def_of=sid,
                                source_sha256=file["hash"],
                                analysis_identity=flow.get("analysis_identity"),
                                edges=edges[:12],
                                omitted=max(0, len(edges) - 12) + flow.get("omitted", 0),
                                limitations=flow.get("limitations", []),
                            )
                        )
                except (SyntaxError, ValueError):
                    pass
        # Graph anchors and the module overview use remaining space, rather
        # than displacing the implementation that an explicit API reaches.
        for anchor in anchor_records:
            omitted += not offer(anchor)
        route = index.memory_route(query, limit=4)
        if route.get("groups"):
            compact = [
                dict(module=g.get("module"), files=g.get("files"), score=g.get("score"))
                for g in route["groups"]
            ]
            omitted += not offer(
                dict(
                    route_map=dict(
                        revision=revision, groups=compact, omitted_groups=route.get("omitted", 0)
                    )
                )
            )
        omitted += len(queue)
        offer(
            dict(
                memory_omissions=omitted,
                expansion="Use versioned inspection tools for omitted facts;omission is not proof of irrelevance.",
            )
        )
        text = "\n".join(records)
        return text, dict(
            revision=revision,
            analysis_identity=identity(),
            max_tokens=max_tokens,
            tokens=estimator.estimate(text),
            selected=selected,
            omitted=omitted,
        )
