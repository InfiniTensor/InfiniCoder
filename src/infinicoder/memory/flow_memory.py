"""Question-focused static value slices with shared source/version references.

The graph is conservative navigation evidence, not a value, alias, ownership or
path proof. Caller constraints and observations remain in the Runtime packet.
"""

from infinicoder._resources import package_root

import ast
from collections import deque
import hashlib
import heapq
import io
import json
import re
import tokenize
from types import SimpleNamespace


def identity():
    root = package_root()
    return hashlib.sha256(
        b"".join(
            (root / name).read_bytes()
            for name in (
                "memory/flow_memory.py",
                "memory/task_memory.py",
                "analysis/dataflow.py",
                "analysis/code_context.py",
                "analysis/heap_flow.py",
            )
        )
    ).hexdigest()


def slice_function(index, symbol_id, names=(), view="values", max_lines=16):
    from infinicoder.memory.task_memory import TaskMemory, bounded_summary

    if (
        not isinstance(names, (list, tuple))
        or len(names) > 8
        or any(not isinstance(n, str) or not n.isidentifier() for n in names)
    ):
        raise ValueError("names must contain at most8 identifiers")
    if (
        view not in ("values", "state", "resources", "interfaces")
        or type(max_lines) is not int
        or not 1 <= max_lines <= 40
    ):
        raise ValueError("invalid view or max_lines (1..40)")
    memory = TaskMemory(index)
    symbol = index.symbol(symbol_id)
    file = memory._current(symbol["path"])
    if file is None:
        return dict(
            symbol=symbol_id,
            status="stale",
            reason="Refresh changed/deleted source before inspection",
        )
    facts = memory.summary(symbol_id)
    if file["language"] != "python" or facts["status"] != "syntax_observations":
        return dict(
            bounded_summary(facts),
            flow_fallback="Existing native/unknown summary;no Python flow claims",
        )
    functions = [
        n
        for n in ast.walk(ast.parse(file["source"]))
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.lineno == symbol["line"]
    ]
    if len(functions) != 1:
        return dict(bounded_summary(facts), flow_fallback="Ambiguous function location")
    source = file["source"].splitlines()
    nodes = []
    guards = {}

    def collect(node, parents=()):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return
        if isinstance(node, ast.stmt):
            nodes.append(node)
            guards[node.lineno] = parents
            if isinstance(
                node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try)
            ):
                parents = (*parents, node.lineno)
        for child in ast.iter_child_nodes(node):
            collect(child, parents)

    for child in functions[0].body:
        collect(child)
    # finally/else are not AST statement nodes. Preserve their actual tokens;
    # a bare close() or assignment without its branch header is misleading.
    keywords = [
        t
        for t in tokenize.generate_tokens(io.StringIO(file["source"]).readline)
        if t.type == tokenize.NAME
    ]
    ordinary = list(nodes)
    for node in ordinary:
        branches = []
        for attribute, word in (("finalbody", "finally"), ("orelse", "else")):
            body = getattr(node, attribute, [])
            if body:
                candidates = [
                    t.start[0]
                    for t in keywords
                    if t.string == word
                    and t.start[1] == node.col_offset
                    and node.lineno < t.start[0] < body[0].lineno
                ]
                if candidates:
                    branches.append((max(candidates), body))
        for handler in getattr(node, "handlers", []):
            branches.append((handler.lineno, handler.body))
        for header, body in branches:
            nodes.append(
                SimpleNamespace(lineno=header, end_lineno=body[-1].end_lineno, control_header=True)
            )
            guards[header] = guards.get(node.lineno, ())
            for child in ordinary:
                if body[0].lineno <= child.lineno <= body[-1].end_lineno:
                    guards[child.lineno] = (*guards.get(child.lineno, ()), header)
    by_line = {n.lineno: n for n in nodes}

    def owner(line):
        candidates = [n for n in nodes if n.lineno <= line <= n.end_lineno]
        return min(candidates, key=lambda n: n.end_lineno - n.lineno).lineno if candidates else line

    # Filter the lexical scope BEFORE the file-wide limit. Other functions may
    # otherwise consume all500 edges even when this function has only a few.
    flow = index.use_def(symbol["path"], scope=symbol["qualname"], limit=500)
    edges = flow["edges"]
    seeds = []
    if names:
        seeds = [owner(e["use_line"]) for e in edges if e["name"] in names]
    if not seeds:
        groups = (
            ("parameters",)
            if view == "interfaces"
            else ("writes", "resources")
            if view in ("state", "resources")
            else ("returns",)
        )
        seeds = [owner(f["line"]) for key in groups for f in facts.get(key, []) if f.get("line")]
    if not seeds:
        seeds = (
            [n.lineno for n in nodes if isinstance(n, ast.Return)] or [nodes[-1].lineno]
            if nodes
            else []
        )
    updates = []
    update_report = None
    if getattr(index, "flow_update_policy", None) == "same_root_v1":
        from infinicoder.analysis.heap_flow import may_updates

        updates, update_report = may_updates(
            nodes,
            guards,
            edges,
            facts.get("writes", []),
            owner,
            calls=facts.get("calls", []),
            priority_lines=seeds,
        )
        update_report["omitted_write_facts"] = facts.get("omitted", {}).get("writes", 0)
    pending = deque(dict.fromkeys(seeds))
    selected = []
    seen = set()
    while pending and len(selected) < max_lines:
        line = pending.popleft()
        if line in seen or line <= symbol["line"] or line not in by_line:
            continue
        seen.add(line)
        selected.append(line)
        # A returned container can depend on item/field writes without any new
        # local binding. Visit those writes before the root's initialization.
        pending.extend(
            owner(link["write_line"])
            for link in updates
            if owner(link["use_line"]) == line and owner(link["write_line"]) not in seen
        )
        pending.extend(p for p in guards.get(line, ()) if p not in seen)
        for edge in edges:
            use, definition = (
                owner(edge["use_line"]),
                owner(edge["definition_line"]) if edge["definition_line"] else 0,
            )
            if use == line and definition > symbol["line"] and definition not in seen:
                pending.append(definition)
            if names and definition == line and use not in seen:
                pending.append(use)
    fragments = []
    for line in selected:
        node = by_line[line]
        end = min(node.end_lineno, line + 3)
        if getattr(node, "control_header", False):
            end = line
        if isinstance(
            node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try)
        ):
            first = next(iter(getattr(node, "body", [])), None)
            end = min(end, first.lineno - 1 if first else line)
        end = max(line, end)
        fragments.append(
            dict(
                start=line,
                end=end,
                statement_end=node.end_lineno,
                complete_statement=end == node.end_lineno
                and not getattr(node, "control_header", False),
                text="\n".join(source[line - 1 : end]),
            )
        )
    selected_edges = [e for e in edges if owner(e["use_line"]) in seen]
    links = [
        link for link in facts.get("input_output_links", []) if owner(link["call_line"]) in seen
    ]
    result = dict(
        symbol=symbol_id,
        path=symbol["path"],
        source_sha256=file["hash"],
        line=symbol["line"],
        end=symbol["end"],
        signature=symbol["signature"],
        status="static_question_flow",
        analysis_identity=identity(),
        view=view,
        parameters=facts.get("parameters", []),
        fragments=fragments,
        use_def=selected_edges,
        input_output_links=links,
        seed_lines=list(dict.fromkeys(seeds)),
        omissions=dict(frontier=len(set(pending) - seen), file_scope_edges=flow.get("omitted", 0)),
        limitations=[
            "Conservative static candidates;branches,early exits and exceptions may change execution.",
            "Aliases,ownership,dynamic dispatch and runtime values remain unknown;fragments are not full inspection.",
            *flow.get("limitations", []),
        ],
    )
    if update_report is not None:
        chosen = [link for link in updates if owner(link["use_line"]) in seen]
        result.update(
            heap_updates=chosen,
            heap_update_analysis=dict(update_report, unselected_links=len(updates) - len(chosen)),
        )
    return result


def bounded_slice(result, max_chars=4800):
    result = dict(result)
    result["observation_omissions"] = {}
    for key in ("fragments", "use_def", "heap_updates", "input_output_links", "parameters"):
        if key in result:
            result[key] = list(result[key])
    while len(json.dumps(result, ensure_ascii=False)) > max_chars:
        candidates = [
            key
            for key in ("fragments", "input_output_links", "use_def", "heap_updates", "parameters")
            if result.get(key)
        ]
        if not candidates:
            return dict(
                symbol=result["symbol"],
                source_sha256=result.get("source_sha256"),
                status="omitted",
                reason="Use narrower source inspection;metadata exceeds bound",
            )
        key = max(candidates, key=lambda k: len(json.dumps(result[k], ensure_ascii=False)))
        result[key].pop()
        result["observation_omissions"][key] = result["observation_omissions"].get(key, 0) + 1
    return result


def pack(index, query, max_tokens, estimator, focus_paths=(), exclude_paths=(), view="auto"):
    from infinicoder.memory.task_memory import TaskMemory
    from infinicoder.analysis.qualified_symbols import anchors

    if type(max_tokens) is not int or max_tokens < 0:
        raise ValueError("max_tokens must be nonnegative int")
    if view not in ("auto", "values", "state", "resources", "interfaces"):
        raise ValueError("invalid memory view")
    if view == "auto":
        view = (
            "resources"
            if re.search(r"owner|releas|lifetime|cancel|close|资源|释放", query, re.I)
            else "state"
            if re.search(r"state|cache|lock|状态|缓存", query, re.I)
            else "values"
        )
    records = []
    selected = []
    seen = set()
    files = {}
    queue = []
    deferred = []
    omitted = 0
    source_counts = {}
    revision = index.revision()

    def cost(rows):
        return estimator.estimate("\n".join(rows))

    def offer(value, ceiling=None):
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if cost([*records, encoded]) <= min(
            max_tokens, ceiling if ceiling is not None else max_tokens
        ):
            records.append(encoded)
            if "source_fragment" in value:
                ref = value["symbol_ref"]
                source_counts[ref] = source_counts.get(ref, 0) + 1
            return True
        return False

    header = dict(
        flow_memory=dict(
            revision=revision,
            analysis_identity=identity(),
            view=view,
            question_sha256=hashlib.sha256(query.encode()).hexdigest(),
            question_ref="caller task or memory_focus;not duplicated here",
            policy="Versioned source candidates only;facts refer to file/symbol records. No runtime,alias or ownership proof.",
            edge_fields=["name", "use_line", "definition_line;0=unresolved"],
            expand="function_summary(symbol_id),use_def(path,scope),read(path,start,count)",
        )
    )
    if not offer(header):
        return "", dict(
            revision=revision,
            tokens=0,
            max_tokens=max_tokens,
            selected=[],
            omitted=1,
            flow_memory=True,
        )
    route = index.memory_route(query, limit=3)
    offer(dict(modules=[g["module"] for g in route.get("groups", [])]))

    def enqueue(sid, priority, reason, depth=0):
        heapq.heappush(queue, (priority, sid, reason, depth))

    for anchor in anchors(index, query):
        if anchor["status"] == "unique":
            # A forwarding declaration still owns its own defaults/types.
            # Never replace that interface with only the terminal signature.
            enqueue(
                anchor.get("entry_symbol", anchor["resolutions"][0]["symbol"]),
                0,
                "qualified public interface",
            )
            enqueue(anchor["resolutions"][0]["symbol"], 0.2, "qualified terminal implementation")
    ranked = index.search(query, limit=20)
    for path in focus_paths:
        if any(c in path for c in "*?["):
            continue
        hits = [
            h
            for h in ranked
            if h["path"] == path or path.endswith("/") and h["path"].startswith(path)
        ]
        if not hits and not path.endswith("/"):
            first = index.db.execute(
                "SELECT id FROM symbols WHERE path=? ORDER BY line LIMIT 1", (path,)
            ).fetchone()
            if first:
                enqueue(first["id"], 3, "current file interface")
        for rank, hit in enumerate(hits[:4]):
            enqueue(hit["id"], 1 + rank * 0.1, "question within current focus")
    for rank, hit in enumerate(ranked):
        enqueue(hit["id"], 2 + rank * 0.3, "question retrieval")
    query_names = re.findall(r"`([^`]+)`", query)
    query_names = [n for n in query_names if n.isidentifier()][:8]
    native_fallback = False
    while queue and len(seen) < 128:
        priority, sid, reason, depth = heapq.heappop(queue)
        if sid in seen:
            continue
        seen.add(sid)
        try:
            symbol = index.symbol(sid)
        except KeyError:
            continue
        file = TaskMemory(index)._current(symbol["path"])
        if file is None:
            omitted += 1
            continue
        if file["language"] != "python":
            native_fallback = True
            omitted += 1
            continue
        if symbol["kind"] != "function":
            children = index.db.execute(
                'SELECT id FROM symbols WHERE path=? AND kind="function" AND qualname LIKE ? ORDER BY line LIMIT 6',
                (symbol["path"], symbol["qualname"] + ".%"),
            ).fetchall()
            for child in children:
                enqueue(child["id"], priority + 0.1, "class interface member", depth + 1)
            continue
        result = slice_function(index, sid, names=query_names, view=view)
        if result["status"] != "static_question_flow":
            omitted += 1
            continue
        fid = files.get(symbol["path"], "f" + str(len(files)))
        sref = "s" + str(len(selected))
        declaration = dict(
            symbol_ref=sref,
            id=sid,
            file_ref=fid,
            line=symbol["line"],
            end=symbol["end"],
            signature=symbol["signature"],
            reason=reason,
            depth=depth,
        )
        file_record = dict(file_ref=fid, path=symbol["path"], source_sha256=file["hash"])
        candidate = [
            *records,
            *(
                [json.dumps(file_record, separators=(",", ":"), ensure_ascii=False)]
                if symbol["path"] not in files
                else []
            ),
            json.dumps(declaration, separators=(",", ":"), ensure_ascii=False),
        ]
        if cost(candidate) > max_tokens:
            omitted += 1
            continue
        records[:] = candidate
        files[symbol["path"]] = fid
        selected.append(
            dict(
                symbol=sid,
                symbol_ref=sref,
                path=symbol["path"],
                source_sha256=file["hash"],
                depth=depth,
                source_slice=True,
                implementation_excerpt_complete=False,
            )
        )
        ceiling = cost(records) + max(200, (max_tokens - cost(records)) // 2)
        values = []
        if "heap_updates" in result:
            values.append(
                dict(
                    symbol_ref=sref,
                    heap_updates=result["heap_updates"],
                    heap_update_analysis=result["heap_update_analysis"],
                )
            )
        values.extend(
            [
                dict(symbol_ref=sref, parameters=result["parameters"]),
                dict(
                    symbol_ref=sref,
                    use_def=[
                        [e["name"], e["use_line"], e["definition_line"]] for e in result["use_def"]
                    ],
                    flow_omissions=result["omissions"],
                ),
            ]
        )
        if symbol["path"] not in exclude_paths:
            values.extend(
                dict(symbol_ref=sref, source_fragment=fragment) for fragment in result["fragments"]
            )
        for link in result["input_output_links"]:
            values.append(dict(symbol_ref=sref, call_binding=link))
            if link.get("callee"):
                enqueue(link["callee"], priority + 0.15, "selected value-flow call", depth + 1)
        for value in values:
            if not offer(value, ceiling):
                deferred.append(value)
        if cost(records) >= max_tokens - 40:
            break
    # Whole atoms only; shared references were emitted before their dependents.
    for value in deferred:
        if not offer(value):
            omitted += 1
    omitted += len(queue)
    if not selected and native_fallback:
        text, report = TaskMemory(index).pack(
            query, max_tokens, estimator, focus_paths, exclude_paths, view=view
        )
        report.update(flow_memory=True, flow_fallback="native:original task-memory views")
        return text, report
    offer(
        dict(
            memory_omissions=omitted,
            expand="function_summary or focused read;omitted is not irrelevant",
        )
    )
    for entry in selected:
        entry["source_fragments"] = source_counts.get(entry["symbol_ref"], 0)
        if entry["path"] in exclude_paths:
            entry["source_already_present"] = True
        elif not entry["source_fragments"]:
            entry["source_omitted"] = True
    text = "\n".join(records)
    return text, dict(
        revision=revision,
        analysis_identity=identity(),
        flow_memory=True,
        view=view,
        max_tokens=max_tokens,
        tokens=estimator.estimate(text),
        selected=selected,
        omitted=omitted,
        frontier=len(queue),
        expansion_cap=128,
    )
