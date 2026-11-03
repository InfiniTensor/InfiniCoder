"""Bounded, non-executing source evidence for initial workflow selection.

Import-call edges are static candidates, never runtime binding or edit scope.
Only task-mentioned qualified symbols seed this analysis. Nothing in a repository
is treated as a command, instruction, acceptance result or permission.
"""

import ast
import hashlib
from pathlib import Path
import re


VERSION = 1
REFERENCE = re.compile(r"(?<![\w./])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)(?![\w/])")


def collect(
    spec, workspace, *, max_files=24, max_bytes=524288, max_file_bytes=131072, max_depth=12
):
    root = Path(workspace).resolve()
    cache = {}
    unknowns = []
    edges = []
    aliases = []
    symbols = {}
    entries = []
    expanded = set()
    bytes_read = 0
    roots = [root]
    if (root / "src").is_dir():
        roots.append(root / "src")

    def unknown(reason):
        if reason not in unknowns:
            unknowns.append(reason)

    def read(path):
        nonlocal bytes_read
        if path in cache:
            return cache[path]
        if len(cache) >= max_files:
            unknown("source file budget exhausted")
            return None
        try:
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                unknown("source path is a symlink or outside workspace")
                return None
            # Also refuse symlinked parent directories, including symlinks into
            # the workspace: their mutable resolution is not this snapshot.
            if any(
                p.is_symlink()
                for p in (path, *path.parents)
                if p != root and p.is_relative_to(root)
            ):
                unknown("source path has a symlinked component")
                return None
            with path.open("rb") as stream:
                data = stream.read(max_file_bytes + 1)
            if len(data) > max_file_bytes or bytes_read + len(data) > max_bytes:
                unknown("source byte budget exhausted")
                return None
            bytes_read += len(data)
            text = data.decode("utf8")
            tree = ast.parse(text)
            if sum(1 for _ in ast.walk(tree)) > 24000:
                unknown("source AST budget exhausted")
                return None
        except (OSError, UnicodeError, SyntaxError, ValueError, RecursionError):
            unknown("source unavailable or not parseable Python")
            return None
        value = dict(
            path=path.relative_to(root).as_posix(),
            sha256=hashlib.sha256(data).hexdigest(),
            tree=tree,
            bindings={},
            ambiguous=set(),
        )
        # Count every module-level store, including conditional imports and
        # assignments. A later rebinding must not look like a stable import.
        for node in tree.body:
            names = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names = [node.name]
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [
                    a.asname or (a.name.split(".")[0] if isinstance(node, ast.Import) else a.name)
                    for a in node.names
                ]
            else:
                for child in ast.walk(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        names.append(child.name)
                    elif isinstance(child, ast.Name) and isinstance(
                        child.ctx, (ast.Store, ast.Del)
                    ):
                        names.append(child.id)
                    elif isinstance(child, (ast.Import, ast.ImportFrom)):
                        names.extend(a.asname or a.name.split(".")[0] for a in child.names)
            for name in names:
                if name in value["bindings"]:
                    value["ambiguous"].add(name)
                value["bindings"][name] = node
            if not isinstance(
                node,
                (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ):
                value["ambiguous"].update(names)
        if any(
            isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names) for n in tree.body
        ):
            value["ambiguous"].update(value["bindings"])
            unknown("wildcard import may replace module bindings")
        cache[path] = value
        return value

    def module(name):
        if not name or any(not p.isidentifier() for p in name.split(".")):
            return None
        candidates = []
        for base in roots:
            part = base.joinpath(*name.split("."))
            for path in (part.with_suffix(".py"), part / "__init__.py"):
                if path.is_file():
                    candidates.append(path)
        if len(candidates) != 1:
            if candidates:
                unknown("ambiguous Python module: " + name)
            return None
        path = candidates[0]
        value = read(path)
        return (name, value, path.name == "__init__.py") if value else None

    def imported(record, binding, suffix=""):
        name, value, package = record
        node = value["bindings"].get(binding)
        if binding in value["ambiguous"]:
            return None
        if isinstance(node, ast.ImportFrom):
            parts = name.split(".") if package else name.split(".")[:-1]
            if node.level:
                if node.level > len(parts):
                    return None
                base = parts[: len(parts) - node.level + 1]
            else:
                base = []
            origin = ".".join(base + ([node.module] if node.module else []))
            for alias in node.names:
                if (alias.asname or alias.name) == binding and alias.name != "*":
                    return ".".join(x for x in (origin, alias.name, suffix) if x)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if (alias.asname or alias.name.split(".")[0]) == binding:
                    prefix = alias.name if alias.asname else alias.name.split(".")[0]
                    return ".".join(x for x in (prefix, suffix) if x)
        return None

    def resolve(qualified, seen=()):
        if qualified in seen or len(seen) >= max_depth:
            unknown("symbol alias cycle or depth limit")
            return None
        parts = qualified.split(".")
        if len(parts) < 2:
            return None
        record = module(".".join(parts[:-1]))
        if not record:
            return None
        value = record[1]
        name = parts[-1]
        if name in value["ambiguous"]:
            unknown("ambiguous or conditional binding: " + qualified)
            return None
        node = value["bindings"].get(name)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            key = value["path"] + "::" + node.name
            if key not in symbols and len(symbols) >= 64:
                unknown("symbol budget exhausted")
                return None
            symbols[key] = dict(
                symbol=key,
                path=value["path"],
                source_sha256=value["sha256"],
                start=node.lineno,
                end=node.end_lineno,
                analysis_version=VERSION,
            )
            return record, node, key
        target = imported(record, name)
        result = resolve(target, (*seen, qualified)) if target else None
        if result:
            fact = dict(
                path=value["path"],
                source_sha256=value["sha256"],
                line=node.lineno,
                binding=name,
                target=target,
                resolved_symbol=result[2],
                relation="static_import_alias_candidate",
            )
            if fact not in aliases and len(aliases) < 64:
                aliases.append(fact)
        return result

    def attribute(node):
        if isinstance(node, ast.Name):
            return [node.id]
        if isinstance(node, ast.Attribute):
            parent = attribute(node.value)
            return parent + [node.attr] if parent else None
        return None

    def calls(node):
        # Nested functions/classes have their own lexical scope. Do not infer
        # their execution just because a containing function was referenced.
        pending = list(node.body)
        while pending:
            child = pending.pop()
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            if isinstance(child, ast.Call):
                yield child
            pending.extend(ast.iter_child_nodes(child))

    def expand(found, depth=0, seen=()):
        record, node, key = found
        if key in seen:
            unknown("static call cycle")
            return
        if key in expanded:
            return
        if depth >= max_depth:
            unknown("static call depth limit")
            return
        expanded.add(key)
        bound = {a.arg for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)}
        if node.args.vararg:
            bound.add(node.args.vararg.arg)
        if node.args.kwarg:
            bound.add(node.args.kwarg.arg)
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
                bound.add(child.id)
            if child is not node and isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                bound.add(child.name)
            if isinstance(child, ast.ExceptHandler) and child.name:
                bound.add(child.name)
            if isinstance(child, (ast.MatchAs, ast.MatchStar)) and child.name:
                bound.add(child.name)
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                bound.update(a.asname or a.name.split(".")[0] for a in child.names)
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        for call in calls(node):
            parts = attribute(call.func)
            if not parts or parts[0] in bound:
                continue
            target = imported(record, parts[0], ".".join(parts[1:]))
            if not target:
                continue
            resolved = resolve(target)
            if not resolved:
                continue
            to = resolved[2]
            if to == key:
                unknown("static call cycle")
                continue
            forward = len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value is call
            edge = dict(
                caller=key,
                callee=to,
                line=call.lineno,
                relation="static_import_call_candidate",
                pure_return_call=forward,
                caller_sha256=record[1]["sha256"],
                callee_sha256=resolved[0][1]["sha256"],
            )
            if len(edges) >= 96:
                unknown("edge budget exhausted")
                return
            if edge not in edges:
                edges.append(edge)
            expand(resolved, depth + 1, (*seen, key))

    mentions = list(REFERENCE.finditer(spec.task[:24000]))[:12]
    for match in mentions:
        found = resolve(match.group(1))
        if found:
            entries.append(
                dict(task_span=list(match.span(1)), text=match.group(1), symbol=found[2])
            )
            expand(found)
    consistent = True
    for path, value in cache.items():
        try:
            with path.open("rb") as stream:
                data = stream.read(max_file_bytes + 1)
            if hashlib.sha256(data).hexdigest() != value["sha256"]:
                consistent = False
        except OSError:
            consistent = False
    if not consistent:
        unknown("source changed during analysis;all dependency labels withheld")
    cross = consistent and (
        any(e["caller"].split("::")[0] != e["callee"].split("::")[0] for e in edges)
        or any(a["path"] != a["resolved_symbol"].split("::")[0] for a in aliases)
    )
    if not entries:
        unknown("no task-mentioned qualified function resolved")
    return dict(
        version=VERSION,
        task_sha256=hashlib.sha256(spec.task.encode()).hexdigest(),
        entries=entries,
        symbols=list(symbols.values()),
        edges=edges,
        aliases=aliases,
        source_snapshot_consistent=consistent,
        labels=["static_cross_module"] if cross else [],
        unknowns=unknowns,
        files_read=len(cache),
        bytes_read=bytes_read,
        model_calls=0,
        interpretation="Bounded static import candidates;not runtime execution,semantic correctness,ownership or a modification plan.",
        read_dependencies=sorted(
            {s["path"] for s in symbols.values()} | {a["path"] for a in aliases}
        ),
        write_set=None,
    )
