"""Question-focused, source-versioned function facts and dependency expansion.

These are syntax observations and candidate bindings, not runtime value, alias,
ownership or dispatch proofs. Model-authored questions never enter the fact cache.
"""

from infinicoder._resources import package_root

import ast
import hashlib
import heapq
import json
from pathlib import Path


def identity():
    from infinicoder.memory.repository_memory import identity as code_memory_identity

    return hashlib.sha256(
        Path(__file__).read_bytes()
        + (package_root() / "memory/inspection_ledger.py").read_bytes()
        + code_memory_identity().encode()
    ).hexdigest()


def _text(node):
    return ast.unparse(node) if node is not None else None


def python_function_facts(source):
    """Extract bounded whole facts; nested bodies have their own lexical scope."""
    tree = ast.parse(source)
    functions = {}

    class Functions(ast.NodeVisitor):
        def __init__(self):
            self.scope = []

        def visit_ClassDef(self, node):
            self.scope.append(node.name)
            for child in node.body:
                self.visit(child)
            self.scope.pop()

        def visit_FunctionDef(self, node):
            qual = ".".join([*self.scope, node.name])
            parameters = []
            positional = [*node.args.posonlyargs, *node.args.args]
            defaults = [None] * (len(positional) - len(node.args.defaults)) + list(
                node.args.defaults
            )
            for arg, default in zip(positional, defaults):
                parameters.append(
                    dict(
                        name=arg.arg,
                        kind="posonly" if arg in node.args.posonlyargs else "positional",
                        default=_text(default),
                        annotation=_text(arg.annotation),
                    )
                )
            for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
                parameters.append(
                    dict(
                        name=arg.arg,
                        kind="keyword_only",
                        default=_text(default),
                        annotation=_text(arg.annotation),
                    )
                )
            for kind, arg in [("varargs", node.args.vararg), ("kwargs", node.args.kwarg)]:
                if arg:
                    parameters.append(
                        dict(name=arg.arg, kind=kind, annotation=_text(arg.annotation))
                    )
            facts = dict(
                parameters=parameters,
                returns=[],
                transformations=[],
                calls=[],
                writes=[],
                resources=[],
                omitted={},
                limitations=[
                    "Syntax only;branches/exceptions may prevent execution. Names may be rebound.",
                    "Heap aliases, dynamic dispatch, ownership transfer and mutation by callees are unknown.",
                ],
            )

            def add(key, value):
                # Omit oversized expressions whole, never present a truncated
                # expression as a complete source fact.
                if len(facts[key]) >= 64 or len(json.dumps(value, ensure_ascii=False)) > 2400:
                    facts["omitted"][key] = facts["omitted"].get(key, 0) + 1
                else:
                    facts[key].append(value)

            class Body(ast.NodeVisitor):
                def visit_FunctionDef(self, child):
                    pass

                visit_AsyncFunctionDef = visit_FunctionDef
                visit_ClassDef = visit_FunctionDef
                visit_Lambda = visit_FunctionDef

                def visit_Return(self, child):
                    add("returns", dict(line=child.lineno, expression=_text(child.value)))
                    self.generic_visit(child)

                def visit_Assign(self, child):
                    add(
                        "transformations",
                        dict(
                            line=child.lineno,
                            targets=[_text(t) for t in child.targets],
                            expression=_text(child.value),
                        ),
                    )
                    self.generic_visit(child)

                def visit_AnnAssign(self, child):
                    if child.value is not None:
                        add(
                            "transformations",
                            dict(
                                line=child.lineno,
                                targets=[_text(child.target)],
                                expression=_text(child.value),
                            ),
                        )
                    self.generic_visit(child)

                def visit_AugAssign(self, child):
                    add(
                        "transformations",
                        dict(
                            line=child.lineno,
                            targets=[_text(child.target)],
                            expression=_text(child),
                            reads_previous_value=True,
                        ),
                    )
                    self.generic_visit(child)

                def visit_Attribute(self, child):
                    if isinstance(child.ctx, (ast.Store, ast.Del)):
                        add(
                            "writes",
                            dict(
                                line=child.lineno,
                                target=_text(child),
                                operation=type(child.ctx).__name__,
                                alias="unknown",
                            ),
                        )
                    self.generic_visit(child)

                visit_Subscript = visit_Attribute

                def visit_Call(self, child):
                    add(
                        "calls",
                        dict(
                            line=child.lineno,
                            column=child.col_offset,
                            target=_text(child.func),
                            arguments=[_text(a) for a in child.args],
                            keywords=[
                                dict(name=k.arg, expression=_text(k.value)) for k in child.keywords
                            ],
                            unpacked=any(isinstance(a, ast.Starred) for a in child.args)
                            or any(k.arg is None for k in child.keywords),
                        ),
                    )
                    if isinstance(child.func, ast.Attribute) and child.func.attr in (
                        "close",
                        "release",
                        "free",
                        "shutdown",
                        "dispose",
                    ):
                        add(
                            "resources",
                            dict(
                                line=child.lineno,
                                kind="explicit_release_call",
                                expression=_text(child),
                                guaranteed=False,
                            ),
                        )
                    self.generic_visit(child)

                def visit_With(self, child):
                    for item in child.items:
                        add(
                            "resources",
                            dict(
                                line=child.lineno,
                                end=child.end_lineno,
                                kind="async_context"
                                if isinstance(child, ast.AsyncWith)
                                else "context",
                                manager=_text(item.context_expr),
                                binding=_text(item.optional_vars),
                                ownership="unknown",
                            ),
                        )
                    self.generic_visit(child)

                visit_AsyncWith = visit_With

                def visit_Try(self, child):
                    if child.finalbody:
                        add(
                            "resources",
                            dict(
                                line=child.finalbody[0].lineno,
                                end=child.finalbody[-1].end_lineno,
                                kind="finally_region",
                                ownership="unknown",
                            ),
                        )
                    self.generic_visit(child)

            visitor = Body()
            for child in node.body:
                visitor.visit(child)
            functions[qual] = facts
            self.scope.append(node.name)
            for child in node.body:
                self.visit(child)
            self.scope.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

    Functions().visit(tree)
    return functions


def _binding(call, callee):
    """Candidate argument-to-parameter links for simple module functions only."""
    if call["unpacked"] or "." in callee["qualname"]:
        return dict(
            status="unknown",
            reason="unpacking, closure or method receiver requires semantic resolution",
        )
    parameters = callee["parameters"]
    if any(p["kind"] in ("varargs", "kwargs") for p in parameters):
        return dict(status="unknown", reason="variadic signature")
    positional = [p for p in parameters if p["kind"] in ("posonly", "positional")]
    if len(call["arguments"]) > len(positional):
        return dict(status="unknown", reason="too many positional arguments")
    values = {
        p["name"]: dict(expression=value, origin="argument")
        for p, value in zip(positional, call["arguments"])
    }
    by_name = {p["name"]: p for p in parameters}
    for keyword in call["keywords"]:
        name = keyword["name"]
        if name not in by_name or by_name[name]["kind"] == "posonly" or name in values:
            return dict(status="unknown", reason="incompatible or duplicate keyword")
        values[name] = dict(expression=keyword["expression"], origin="keyword")
    for parameter in parameters:
        if parameter["name"] not in values:
            if parameter.get("default") is None:
                return dict(status="unknown", reason="required argument missing")
            values[parameter["name"]] = dict(
                expression=parameter["default"], origin="default_syntax_not_evaluated_value"
            )
    return dict(
        status="static_candidate",
        inputs=values,
        limitation="Expressions belong to caller scope;callee locals, aliases, decorators and dynamic rebinding may change behavior.",
    )


def bounded_summary(result, max_chars=4600):
    """Bound tool observations by dropping whole facts, never serialized bytes."""
    result = dict(result)
    result["observation_omissions"] = {}
    keys = (
        "input_output_links",
        "transformations",
        "calls",
        "resources",
        "writes",
        "returns",
        "parameters",
    )
    for key in keys:
        if key in result:
            result[key] = list(result[key])
    while len(json.dumps(result, ensure_ascii=False)) > max_chars:
        candidates = [key for key in keys if result.get(key)]
        if candidates:
            key = max(candidates, key=lambda k: len(json.dumps(result[k], ensure_ascii=False)))
            result[key].pop()
            result["observation_omissions"][key] = result["observation_omissions"].get(key, 0) + 1
        elif result.get("signature"):
            result.pop("signature")
            result["signature_omitted"] = True
        else:
            return dict(
                symbol=result["symbol"],
                path=result.get("path"),
                source_sha256=result.get("source_sha256"),
                status="summary_omitted",
                reason="Metadata exceeds observation budget;inspect focused source.",
            )
    return result


class TaskMemory:
    def __init__(self, index):
        self.index = index
        self._native_fingerprint_cache = {}
        index.db.execute(
            "CREATE TABLE IF NOT EXISTS memory_function_facts(path TEXT PRIMARY KEY, source_sha256 TEXT, analysis_identity TEXT, facts TEXT)"
        )

    def _current(self, path):
        row = self.index.db.execute(
            "SELECT hash,source,error,language,build FROM files WHERE path=?", (path,)
        ).fetchone()
        if row is None:
            return None
        try:
            file = (self.index.root / path).resolve()
            if (
                not file.is_relative_to(self.index.root)
                or hashlib.sha256(file.read_bytes()).hexdigest() != row["hash"]
            ):
                return None
        except OSError:
            return None
        return row

    def summary(self, symbol_id, include_links=True):
        index = self.index
        symbol = index.symbol(symbol_id)
        file = self._current(symbol["path"])
        if file is None:
            return dict(
                symbol=symbol_id,
                status="stale",
                reason="Source changed or disappeared;refresh and inspect again.",
            )
        result = dict(
            symbol=symbol_id,
            path=symbol["path"],
            source_sha256=file["hash"],
            line=symbol["line"],
            end=symbol["end"],
            qualname=symbol["qualname"],
            signature=symbol["signature"],
            analysis_identity=identity(),
            status="syntax_observations",
        )
        if file["language"] != "python":
            from infinicoder.analysis.native_effects import FUNCTIONS, current_native_facts, links

            if symbol["kind"].upper() not in FUNCTIONS:
                return dict(
                    result,
                    status="unknown",
                    limitations=["Native function effects unavailable for this symbol."],
                )
            facts, problem = current_native_facts(
                index, symbol, file, self._native_fingerprint_cache
            )
            if facts is None:
                return dict(result, status=problem["status"], limitations=[problem["reason"]])
            result.update(facts)
            if include_links:
                result["input_output_links"] = links(self, symbol, facts)
            return result
        if file["language"] != "python" or symbol["kind"] != "function" or file["error"]:
            return dict(
                result,
                status="unknown",
                limitations=[
                    "Function effects unavailable for this language/symbol;use existing Clang/use_def and exact source."
                ],
            )
        cached = index.db.execute(
            "SELECT * FROM memory_function_facts WHERE path=?", (symbol["path"],)
        ).fetchone()
        if (
            cached
            and cached["source_sha256"] == file["hash"]
            and cached["analysis_identity"] == identity()
        ):
            functions = json.loads(cached["facts"])
        else:
            functions = python_function_facts(file["source"])
            with index.db:
                index.db.execute(
                    "DELETE FROM memory_function_facts WHERE path NOT IN (SELECT path FROM files)"
                )
                index.db.execute(
                    "INSERT OR REPLACE INTO memory_function_facts VALUES(?,?,?,?)",
                    (
                        symbol["path"],
                        file["hash"],
                        identity(),
                        json.dumps(functions, ensure_ascii=False),
                    ),
                )
        if symbol["qualname"] not in functions:
            return dict(
                result,
                status="unknown",
                limitations=["Function scope not found in current syntax."],
            )
        result.update(functions[symbol["qualname"]])
        if not include_links:
            return result
        result["input_output_links"] = []
        edges = index.relationships(symbol_id, direction="outgoing", limit=200)
        for call in result["calls"]:
            matches = [
                e
                for e in edges
                if e["kind"] == "calls"
                and e["line"] == call["line"]
                and e["target"] == call["target"]
            ]
            targets = {e["dst"] for e in matches if e["dst"]}
            link = dict(
                call_line=call["line"], call_column=call["column"], target_expression=call["target"]
            )
            if len(targets) != 1 or any(e.get("shadowed") for e in matches):
                link.update(status="unknown", reason="unresolved, shadowed or ambiguous call")
            else:
                target = next(iter(targets))
                callee = self.summary(target, include_links=False)
                if callee["status"] != "syntax_observations":
                    link.update(
                        status="unknown",
                        callee=target,
                        reason="callee source stale or effects unavailable",
                    )
                else:
                    link.update(
                        _binding(call, callee),
                        callee=target,
                        callee_path=callee["path"],
                        callee_source_sha256=callee["source_sha256"],
                        callee_line=callee["line"],
                        return_expressions=callee["returns"],
                        may_write=callee["writes"],
                        resource_regions=callee["resources"],
                    )
            result["input_output_links"].append(link)
        return result

    def pack(
        self,
        query,
        max_tokens,
        estimator,
        focus_paths=(),
        exclude_paths=(),
        view="auto",
        *,
        anchors=(),
        direction="both",
    ):
        if type(max_tokens) is not int or max_tokens < 0:
            raise ValueError("max_tokens must be a nonnegative int")
        if view not in ("auto", "values", "state", "resources", "interfaces"):
            raise ValueError("invalid task memory view")
        if direction not in ("both", "incoming", "outgoing"):
            raise ValueError("invalid memory edge direction")
        if not isinstance(anchors, (list, tuple)) or len(anchors) > 8:
            raise ValueError("at most8 source anchors required")
        anchor_locations = {}
        for anchor in anchors:
            if not isinstance(anchor, dict) or set(anchor) != {"symbol", "line", "reason"}:
                raise ValueError("source anchor needs symbol,line,reason")
            if (
                not isinstance(anchor["symbol"], str)
                or not isinstance(anchor["reason"], str)
                or len(anchor["reason"]) > 160
            ):
                raise ValueError("bounded source anchor required")
            try:
                symbol = self.index.symbol(anchor["symbol"])
            except KeyError:
                raise ValueError("source anchor symbol unavailable;refresh") from None
            if (
                type(anchor["line"]) is not int
                or not symbol["line"] <= anchor["line"] <= symbol["end"]
            ):
                raise ValueError("source anchor outside symbol")
            if anchor["symbol"] in anchor_locations:
                raise ValueError("duplicate source anchor")
            anchor_locations[anchor["symbol"]] = anchor
        if view == "auto":
            import re

            view = (
                "resources"
                if re.search(
                    r"releas|lifetim|owner|leak|alloc|cancel|close|释放|资源|生命周期|泄漏",
                    query,
                    re.I,
                )
                else "state"
                if re.search(r"state|cache|lock|race|状态|缓存|同步", query, re.I)
                else "values"
            )
        index = self.index
        revision = index.revision()
        records, selected, seen = [], [], set()
        omitted = 0

        def offer(row):
            text = "\n".join([*records, json.dumps(row, ensure_ascii=False, separators=(",", ":"))])
            if estimator.estimate(text) <= max_tokens:
                records.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                return True
            return False

        header = dict(
            task_memory=dict(
                revision=revision,
                analysis_identity=identity(),
                question=query,
                view=view,
                policy="Source syntax and static candidates only. Model questions/hypotheses and runtime observations remain separate. No alias/ownership proof.",
                expand="function_summary(symbol_id);use_def(path);read(path,start,count)",
            )
        )
        if not offer(header):
            return "", dict(
                revision=revision, tokens=0, max_tokens=max_tokens, selected=[], omitted=1
            )
        # Module routing precedes symbol expansion. Detailed interfaces remain
        # expandable rather than filling every small-context prompt.
        route = index.memory_route(query, limit=3)
        offer(
            dict(
                modules=[
                    dict(module=g["module"], files=g["files"]) for g in route.get("groups", [])
                ]
            )
        )
        from infinicoder.analysis.qualified_symbols import anchors

        queue = []

        def enqueue(sid, priority, reason, depth=0):
            heapq.heappush(queue, (priority, sid, reason, depth))

        for rank, anchor in enumerate(anchor_locations.values()):
            enqueue(anchor["symbol"], -2 + rank * 0.1, anchor["reason"])
        for anchor in anchors(index, query):
            if anchor["status"] == "unique":
                enqueue(anchor["resolutions"][0]["symbol"], 0, "qualified interface terminal")
        ranked = index.search(query, limit=20)
        for path in focus_paths:
            # A broad editable glob is permission, not evidence of relevance.
            if any(c in path for c in "*?["):
                continue
            relevant = [
                hit
                for hit in ranked
                if hit["path"] == path or (path.endswith("/") and hit["path"].startswith(path))
            ]
            for rank, hit in enumerate(relevant[:4]):
                enqueue(hit["id"], 1 + rank * 0.1, "question within current focus")
            if not relevant and not path.endswith("/"):
                row = index.db.execute(
                    "SELECT id FROM symbols WHERE path=? ORDER BY line LIMIT 1", (path,)
                ).fetchone()
                if row:
                    enqueue(row["id"], 3, "current file interface")
        for rank, hit in enumerate(ranked):
            enqueue(hit["id"], 2 + rank * 0.3, "question retrieval")
        # No fixed three-hop cutoff: traverse relevant graph until bounded work
        # or context is consumed. The cap and omitted frontier are explicit.
        while queue and len(seen) < 128:
            priority, sid, reason, depth = heapq.heappop(queue)
            if sid in seen:
                continue
            seen.add(sid)
            try:
                symbol = index.symbol(sid)
            except KeyError:
                continue
            file = self._current(symbol["path"])
            if file is None:
                omitted += 1
                continue
            native_inputs_current = True
            if file["language"] != "python":
                from infinicoder.analysis.native_effects import current_native_inputs

                _, problem = current_native_inputs(
                    index, symbol, file, self._native_fingerprint_cache
                )
                native_inputs_current = problem is None
            edges = (
                index.relationships(sid, direction=direction, limit=50)
                if native_inputs_current
                else []
            )
            for edge in edges:
                other = edge.get("dst") if edge["src"] == sid else edge["src"]
                if edge["kind"] == "calls" and other and other != sid and other not in seen:
                    enqueue(other, priority + 0.15, "call dependency of " + sid, depth + 1)
            # Source records outrank auxiliary views; neither a selected path
            # nor a cached summary certifies full-body inspection.
            start = index._implementation_start(
                symbol["path"], file["hash"], file["source"], symbol
            )
            implementation_start = start
            if sid in anchor_locations:
                start = max(start, anchor_locations[sid]["line"] - 6)
            lines = file["source"].splitlines()
            body = lines[start - 1 : min(symbol["end"], start + 23)]
            row = dict(
                symbol=sid,
                path=symbol["path"],
                source_sha256=file["hash"],
                line=symbol["line"],
                signature=symbol["signature"],
                reason=reason,
                depth=depth,
                excerpt="\n".join(body),
                excerpt_line=start,
                excerpt_end=start + len(body) - 1,
                implementation_excerpt_complete=start == implementation_start
                and start + len(body) - 1 == symbol["end"],
            )
            if symbol["path"] in exclude_paths:
                selected.append(
                    dict(
                        symbol=sid,
                        path=symbol["path"],
                        source_sha256=file["hash"],
                        source_already_present=True,
                    )
                )
            elif offer(row):
                selected.append(
                    dict(
                        symbol=sid,
                        path=symbol["path"],
                        source_sha256=file["hash"],
                        depth=depth,
                        excerpt_line=start,
                        excerpt_end=row["excerpt_end"],
                        implementation_excerpt_complete=row["implementation_excerpt_complete"],
                    )
                )
            else:
                stub = {
                    k: v
                    for k, v in row.items()
                    if not k.startswith("excerpt") and k != "implementation_excerpt_complete"
                }
                stub["source_omitted"] = True
                if offer(stub):
                    selected.append(
                        dict(
                            symbol=sid,
                            path=symbol["path"],
                            source_sha256=file["hash"],
                            source_omitted=True,
                        )
                    )
                omitted += 1
            summary = self.summary(sid)
            keys = (
                ("parameters",)
                if view == "interfaces"
                else ("parameters", "resources", "writes")
                if view == "resources"
                else ("parameters", "writes", "transformations")
                if view == "state"
                else ("parameters", "transformations", "returns")
            )
            for key in (*keys, "input_output_links", "limitations"):
                values = summary.get(key, [])
                if values:
                    if key == "limitations":
                        omitted += not offer(
                            dict(
                                function_summary=sid,
                                source_sha256=file["hash"],
                                view=key,
                                facts=values,
                            )
                        )
                    else:
                        for value in values:
                            omitted += not offer(
                                dict(
                                    function_summary=sid,
                                    source_sha256=file["hash"],
                                    view=key,
                                    fact=value,
                                )
                            )
            omitted += sum(summary.get("omitted", {}).values())
            if (
                view in ("values", "state")
                and depth == 0
                and native_inputs_current
                and symbol["kind"] in ("function", "function_decl", "cxx_method")
            ):
                flow = index.use_def(symbol["path"], limit=500)
                scope = symbol["qualname"] if file["language"] == "python" else sid
                matching = [e for e in flow.get("edges", []) if e.get("scope") == scope]
                if matching:
                    omitted += not offer(
                        dict(
                            use_def_of=sid,
                            source_sha256=file["hash"],
                            analysis_identity=flow.get("analysis_identity"),
                            edges=matching[:6],
                            omitted=max(0, len(matching) - 6) + flow.get("omitted", 0),
                            limitations=flow.get("limitations", []),
                        )
                    )
            if estimator.estimate("\n".join(records)) >= max_tokens - 64:
                break
        omitted += len(queue)
        offer(
            dict(
                memory_omissions=omitted,
                expansion="function_summary or source inspection;omitted is not irrelevant",
            )
        )
        text = "\n".join(records)
        return text, dict(
            revision=revision,
            analysis_identity=identity(),
            view=view,
            max_tokens=max_tokens,
            tokens=estimator.estimate(text),
            selected=selected,
            omitted=omitted,
            frontier=len(queue),
            expansion_cap=128,
        )
