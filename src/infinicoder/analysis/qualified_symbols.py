"""Conservative resolution of public Python names through import re-exports.

No imports execute. Conditional, wildcard and rebound aliases are not guessed;
multiple source roots remain ambiguous. Every hop carries a source location.
"""

import ast
import hashlib
import re


def module_names(path, paths):
    """Static source-root candidates, without changing sys.path or importing."""
    if not path.endswith(".py"):
        return set()
    bits = path[:-3].split("/")
    names = {".".join(bits)}
    if bits[0] in ("src", "python") and len(bits) > 1:
        names.add(".".join(bits[1:]))
    # A nonstandard outer source directory can contain a regular package.
    # Follow its contiguous __init__.py ancestors to the top package marker.
    start = len(bits) - 1
    while start > 0 and "/".join(bits[:start]) + "/__init__.py" in paths:
        start -= 1
    if start < len(bits) - 1:
        names.add(".".join(bits[start:]))
    return {name[:-9] if name.endswith(".__init__") else name for name in names}


def _absolute(path, name):
    if not name.startswith("."):
        return name
    level = len(name) - len(name.lstrip("."))
    parent = path.removesuffix(".py").replace("/", ".").split(".")[:-1]
    if level > len(parent):
        return None
    return ".".join(parent[: len(parent) - level + 1] + [name.lstrip(".")])


def _bindings(source, path):
    tree = ast.parse(source)
    aliases = {}
    uncertain = set()

    class Bindings(ast.NodeVisitor):
        def visit_Name(self, node):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                uncertain.add(node.id)

        def visit_FunctionDef(self, node):
            uncertain.add(node.name)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node):
            uncertain.add(node.name)

        def visit_Import(self, node):
            uncertain.update(a.asname or a.name.split(".")[0] for a in node.names)

        def visit_ImportFrom(self, node):
            uncertain.update(a.asname or a.name for a in node.names)

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
                return {}
            for item in node.names:
                if isinstance(node, ast.Import):
                    bound = item.asname or item.name.split(".")[0]
                    target = item.name if item.asname else bound
                else:
                    bound = item.asname or item.name
                    module = "." * node.level + (node.module or "")
                    target = _absolute(path, module + ("." if node.module else "") + item.name)
                if bound in aliases:
                    uncertain.add(bound)
                aliases[bound] = dict(target=target, line=node.lineno, end=node.end_lineno)
        else:
            Bindings().visit(node)
    return {
        name: fact for name, fact in aliases.items() if name not in uncertain and fact["target"]
    }


def _resolve_exports(index, name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+", name):
        raise ValueError("expected a dotted Python public name")
    revision = index.revision()
    cache = getattr(index, "_qualified_modules", None)
    if cache is None or cache[0] != revision:
        modules = {}
        paths = {
            row["path"] for row in index.db.execute("SELECT path FROM files WHERE path LIKE '%.py'")
        }
        for path in sorted(paths):
            for key in module_names(path, paths):
                modules.setdefault(key, []).append(path)
        cache = (revision, modules)
        index._qualified_modules = cache
    modules = cache[1]
    answers = []
    unknown = []

    def walk(qualified, chain, seen):
        if qualified in seen or len(chain) >= 16:
            unknown.append("cycle or depth bound")
            return
        seen = seen | {qualified}
        bits = qualified.split(".")
        # Use the longest actual module prefix; do not guess missing packages.
        for cut in range(len(bits) - 1, 0, -1):
            paths = modules.get(".".join(bits[:cut]), [])
            if not paths:
                continue
            if len(paths) > 8:
                unknown.append("module alternatives truncated")
            attribute = ".".join(bits[cut:])
            first, *rest = attribute.split(".")
            for path in paths[:8]:
                file = index.db.execute(
                    "SELECT hash,source FROM files WHERE path=?", (path,)
                ).fetchone()
                actual = index.root / path
                if (
                    not actual.is_file()
                    or hashlib.sha256(actual.read_bytes()).hexdigest() != file["hash"]
                ):
                    unknown.append("source changed;refresh index")
                    continue
                symbol = path + "::" + attribute
                row = index.db.execute(
                    "SELECT id,line,end FROM symbols WHERE id=?", (symbol,)
                ).fetchone()
                if row:
                    answers.append(
                        dict(
                            symbol=symbol,
                            chain=chain
                            + [
                                dict(
                                    path=path,
                                    line=row["line"],
                                    end=row["end"],
                                    source_sha256=file["hash"],
                                    kind="declaration",
                                )
                            ],
                            confidence="static_candidate",
                        )
                    )
                    continue
                try:
                    alias = _bindings(file["source"], path).get(first)
                except SyntaxError:
                    alias = None
                if not alias:
                    unknown.append("unresolved,conditional or rebound export")
                    continue
                target = alias["target"] + ("." + ".".join(rest) if rest else "")
                fact = dict(
                    path=path,
                    line=alias["line"],
                    end=alias["end"],
                    source_sha256=file["hash"],
                    kind="import_alias",
                    binding=first,
                    target=target,
                )
                walk(target, chain + [fact], seen)
            return
        unknown.append("module not indexed")

    walk(name, [], set())
    unique = {a["symbol"]: a for a in answers}
    # An incomplete alternative import root can also hide a different target.
    status = (
        "unique"
        if len(unique) == 1 and not unknown
        else "ambiguous"
        if len(unique) > 1 or unique and unknown
        else "unresolved"
    )
    return dict(
        query=name,
        revision=revision,
        status=status,
        resolutions=list(unique.values()),
        unknown=sorted(set(unknown)),
        limitations=[
            "static candidates only;runtime import hooks,star/conditional imports and dynamic attribute access are not resolved"
        ],
    )


def _stable_function(tree, qualname):
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == qualname]
    if len(selected) != 1:
        return None
    function = selected[0]

    # Reassignments/conditional definitions of the function's global binding
    # invalidate the otherwise simple-looking forwarding declaration.
    class Rebindings(ast.NodeVisitor):
        rebound = False

        def visit_Name(self, node):
            if node.id == qualname and isinstance(node.ctx, (ast.Store, ast.Del)):
                self.rebound = True

        def visit_FunctionDef(self, node):
            if node.name == qualname:
                self.rebound = True

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef

        def visit_Import(self, node):
            if any((a.asname or a.name.split(".")[0]) == qualname for a in node.names):
                self.rebound = True

        def visit_ImportFrom(self, node):
            if any((a.asname or a.name) == qualname or a.name == "*" for a in node.names):
                self.rebound = True

    check = Rebindings()
    for node in tree.body:
        if node is not function:
            check.visit(node)
    if check.rebound:
        return None
    return function


def _forward_target(source, path, qualname):
    """Recognize an exact, unconditional top-level argument-forwarding call."""
    tree = ast.parse(source)
    function = _stable_function(tree, qualname)
    if function is None:
        return None
    args = function.args
    if function.decorator_list or args.vararg or args.kwarg:
        return None
    body = [
        n
        for n in function.body
        if not (
            isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Constant)
            and isinstance(n.value.value, str)
        )
    ]
    if (
        len(body) != 1
        or not isinstance(body[0], ast.Return)
        or not isinstance(body[0].value, ast.Call)
    ):
        return None
    call = body[0].value
    positional = [a.arg for a in [*args.posonlyargs, *args.args]]
    keyword = [a.arg for a in args.kwonlyargs]
    if [n.id if isinstance(n, ast.Name) else None for n in call.args] != positional:
        return None
    if [(n.arg, n.value.id if isinstance(n.value, ast.Name) else None) for n in call.keywords] != [
        (a, a) for a in keyword
    ]:
        return None
    callee = ast.unparse(call.func)
    if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", callee):
        return None
    first, *rest = callee.split(".")
    if first in positional + keyword:
        return None  # runtime callback parameter
    alias = _bindings(source, path).get(first)
    if alias:
        target = alias["target"] + ("." + ".".join(rest) if rest else "")
    elif not rest and any(isinstance(n, ast.FunctionDef) and n.name == first for n in tree.body):
        # Use the exact indexed source-root path,not an inferred sys.path.
        target = path.removesuffix(".py").replace("/", ".") + "." + first
    else:
        return None
    return dict(
        target=target,
        line=body[0].lineno,
        end=body[0].end_lineno,
        kind="identity_argument_forwarder",
        binding=callee,
    )


def resolve(index, name, follow_forwarders=False):
    if type(follow_forwarders) is not bool:
        raise ValueError("follow_forwarders must be bool")
    result = _resolve_exports(index, name)
    if not follow_forwarders or result["status"] != "unique":
        return result
    resolution = result["resolutions"][0]
    result["entry_symbol"] = resolution["symbol"]
    seen = set()
    for depth in range(32):
        symbol = resolution["symbol"]
        if symbol in seen:
            result["status"] = "unresolved"
            result["unknown"].append("forwarding cycle")
            break
        seen.add(symbol)
        path, qualname = symbol.split("::", 1)
        file = index.db.execute("SELECT hash,source FROM files WHERE path=?", (path,)).fetchone()
        actual = index.root / path
        if (
            not file
            or not actual.is_file()
            or hashlib.sha256(actual.read_bytes()).hexdigest() != file["hash"]
        ):
            result["status"] = "unresolved"
            result["unknown"].append("forwarding source changed;refresh index")
            break
        try:
            tree = ast.parse(file["source"])
            if (
                "." not in qualname
                and any(
                    isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == qualname
                    for n in ast.walk(tree)
                )
                and _stable_function(tree, qualname) is None
            ):
                result["status"] = "unresolved"
                result["unknown"].append("conditional or rebound function declaration")
                break
            forward = _forward_target(file["source"], path, qualname)
        except SyntaxError:
            forward = None
        if not forward:
            result["terminal_reason"] = "not a proven identity argument forwarder"
            break
        target = _resolve_exports(index, forward["target"])
        if target["status"] != "unique":
            result["status"] = "unresolved"
            result["unknown"].extend(target["unknown"] or ["ambiguous forwarded target"])
            break
        leaf = target["resolutions"][0]
        fact = dict(path=path, source_sha256=file["hash"], **forward)
        resolution = dict(leaf, chain=[*resolution["chain"], fact, *leaf["chain"]])
        result["resolutions"] = [resolution]
    else:
        result["status"] = "unresolved"
        result["unknown"].append("forwarding depth bound32")
    result["forwarding_steps"] = sum(
        hop["kind"] == "identity_argument_forwarder" for hop in resolution["chain"]
    )
    result["limitations"].append(
        "only exact top-level argument forwarding is followed;decorators,callbacks,transformed arguments and extra effects are not expanded"
    )
    return result


def anchors(index, query):
    names = list(dict.fromkeys(re.findall(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+", query)))[:12]
    return [resolve(index, name, follow_forwarders=True) for name in names]
