"""Optional code-compilation proposal after an independently correct baseline.

Caches executable code,never tensor outputs. Runtime's external correctness,
timing,shared deadline and best-source rollback still decide whether to keep it.
No candidate module is imported or executed during this AST transformation.
"""

import ast, copy
from infinicoder.domains.operator.pointwise import Proposal


def infer_entry(source):
    """Unique top-level function not called by another declared function."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    called = {
        n.func.id
        for node in functions.values()
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in functions
    }
    roots = set(functions) - called
    return next(iter(roots)) if len(roots) == 1 else None


def propose(source, entry, scoped=False):
    if not isinstance(entry, str) or not entry.isidentifier() or len(source) > 65536:
        return None
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    if sum(1 for _ in ast.walk(tree)) > 4000:
        return None
    if not any(
        isinstance(n, ast.Import)
        and any(a.name == "torch" and a.asname in (None, "torch") for a in n.names)
        for n in tree.body
    ):
        return None
    if any(
        not isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))
        and not (
            isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Constant)
            and isinstance(n.value.value, str)
        )
        for n in tree.body
    ):
        return None
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == entry]
    if len(functions) != 1:
        return None
    target = functions[0]
    args = target.args
    if (
        target.decorator_list
        or args.defaults
        or any(v is not None for v in args.kw_defaults)
        or args.vararg
        or args.kwarg
        or not (args.args or args.posonlyargs)
    ):
        return None
    prefix = "_infinicoder_compiled_code"
    if prefix in source:
        return None
    forbidden = {
        "open",
        "exec",
        "eval",
        "compile",
        "print",
        "input",
        "globals",
        "locals",
        "getattr",
        "setattr",
        "delattr",
        "set",
        "random",
    }
    for n in ast.walk(tree):
        if isinstance(
            n,
            (
                ast.ClassDef,
                ast.AsyncFunctionDef,
                ast.Await,
                ast.Yield,
                ast.YieldFrom,
                ast.Global,
                ast.Nonlocal,
                ast.Delete,
                ast.Raise,
            ),
        ):
            return None
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            modules = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""]
            if any(m not in ("torch", "math") for m in modules):
                return None
            if any(
                (a.asname or a.name.split(".")[0]) == "torch"
                and not (isinstance(n, ast.Import) and a.name == "torch")
                for a in n.names
            ):
                return None
        if isinstance(n, ast.Name) and (
            n.id in forbidden or isinstance(n.ctx, ast.Store) and n.id in {"torch", entry}
        ):
            return None
        if isinstance(n, ast.arg) and n.arg in {"torch", entry}:
            return None
        if isinstance(n, ast.FunctionDef) and n.name == "torch":
            return None
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == entry:
            return None
        if isinstance(n, (ast.Attribute, ast.Subscript)) and isinstance(
            n.ctx, (ast.Store, ast.Del)
        ):
            return None
        if isinstance(n, ast.Attribute) and (
            n.attr.startswith("__")
            or n.attr.endswith("_")
            or n.attr
            in {
                "append",
                "extend",
                "clear",
                "pop",
                "remove",
                "update",
                "setdefault",
                "backward",
                "grad",
                "compile",
                "manual_seed",
                "seed",
                "rand",
                "randn",
                "randint",
                "randperm",
                "rand_like",
                "randn_like",
                "normal",
                "bernoulli",
                "multinomial",
                "dropout",
                "set_grad_enabled",
                "set_default_dtype",
                "set_default_device",
            }
        ):
            return None
        if isinstance(n, ast.keyword) and n.arg == "out":
            return None
    call_args = [ast.Name(id=a.arg, ctx=ast.Load()) for a in [*args.posonlyargs, *args.args]]
    call_keywords = [
        ast.keyword(arg=a.arg, value=ast.Name(id=a.arg, ctx=ast.Load())) for a in args.kwonlyargs
    ]
    inner = copy.deepcopy(target)
    inner.name = prefix + "_implementation"
    inner.returns = None
    inner.type_comment = None
    for a in [*inner.args.posonlyargs, *inner.args.args, *inner.args.kwonlyargs]:
        a.annotation = None
    local = prefix + "_callable"
    setup = ast.parse(
        "try:\n " + local + "=" + entry + "." + prefix + "\nexcept AttributeError:\n pass\n"
    ).body[0]
    setup.handlers[0].body = [
        inner,
        *ast.parse(
            local
            + "=torch.compile("
            + inner.name
            + ",dynamic=True)\n"
            + entry
            + "."
            + prefix
            + "="
            + local
            + "\n"
        ).body,
    ]
    original_doc = (
        target.body[:1]
        if target.body
        and isinstance(target.body[0], ast.Expr)
        and isinstance(target.body[0].value, ast.Constant)
        and isinstance(target.body[0].value.value, str)
        else []
    )
    target.body = [
        *original_doc,
        setup,
        ast.Return(
            value=ast.Call(
                func=ast.Name(id=local, ctx=ast.Load()), args=call_args, keywords=call_keywords
            )
        ),
    ]
    ast.fix_missing_locations(tree)
    result = ast.unparse(target if scoped else tree) + "\n"
    return Proposal(
        result,
        "torch.compile(dynamic=True) code specialization",
        (
            "caller supplies independently correct baseline and trusted performance objective",
            "fixed Python function interface;unshadowed torch;no detected mutation or recursion",
            "fresh external changed-input correctness and repeated timing required",
            "compilation and failed proposals consume the original task deadline",
        ),
        "generic_torch_compile_code_seed_v1",
    )
