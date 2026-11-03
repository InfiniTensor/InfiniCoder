"""Optional localized repair target from source-proven public Python names.

This selects a source work packet, not a semantic repair or acceptance result.
Ambiguous roots, dynamic forwarding and multiple target functions are declined.
"""

import ast
import builtins
import hashlib
from infinicoder.analysis.qualified_symbols import anchors
from infinicoder.memory.source.source_scope import packet


def select(index, task, workspace, max_chars=24000):
    evidence = anchors(index, task)
    if any(item["status"] == "ambiguous" for item in evidence):
        return None
    resolved = [item for item in evidence if item["status"] == "unique"]
    targets = {item["resolutions"][0]["symbol"] for item in resolved}
    if len(targets) != 1:
        return None
    target = next(iter(targets))
    path, symbol = target.split("::", 1)
    try:
        actual = workspace.path(path, write=True)
        data = actual.read_bytes()
        if len(data) > max_chars * 4:
            return None
        # Every hop is checked against current bytes, including the final file.
        chains = [item["resolutions"][0]["chain"] for item in resolved]
        for chain in chains:
            for hop in chain:
                source = workspace.path(hop["path"]).read_bytes()
                if hashlib.sha256(source).hexdigest() != hop["source_sha256"]:
                    return None
        scope = packet(data.decode("utf-8"), symbol)
        if len(scope["source"]) > max_chars:
            return None
        function = ast.parse(scope["source"]).body[0]
        if function.decorator_list:
            return None
        statements = [
            n
            for n in function.body
            if not (
                isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Constant)
                and isinstance(n.value.value, str)
            )
        ]
        if (
            len(statements) == 1
            and isinstance(statements[0], ast.Return)
            and isinstance(statements[0].value, ast.Call)
        ):
            # Do not mistake a stopped, non-identity delegate for its callee.
            # A plain unshadowed builtin expression (e.g. sorted) is a leaf.
            callee = statements[0].value.func
            module = ast.parse(data.decode("utf-8"))
            bound = {
                n.id
                for n in ast.walk(module)
                if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))
            }
            bound.update(
                n.name
                for n in ast.walk(module)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            )
            bound.update(
                a.asname or a.name.split(".")[0]
                for n in ast.walk(module)
                if isinstance(n, (ast.Import, ast.ImportFrom))
                for a in n.names
            )
            bound.update(
                a.arg
                for n in ast.walk(module)
                if isinstance(n, ast.arguments)
                for a in [
                    *n.posonlyargs,
                    *n.args,
                    *n.kwonlyargs,
                    *([n.vararg] if n.vararg else []),
                    *([n.kwarg] if n.kwarg else []),
                ]
            )
            if (
                not isinstance(callee, ast.Name)
                or not hasattr(builtins, callee.id)
                or callee.id in bound
            ):
                return None
    except (ValueError, OSError, UnicodeError, SyntaxError):
        return None
    return dict(
        path=path,
        symbol=symbol,
        public_names=[item["query"] for item in resolved],
        index_revision=index.revision(),
        source_sha256=hashlib.sha256(data).hexdigest(),
        resolutions=resolved,
        confidence="static_candidate",
        limitation="Identity forwarding only;function-body scope preserves other source. Independent final checks remain mandatory;broader repairs may require action mode.",
    )
