"""Bounded syntactic host/kernel edges. Never import or execute candidate code."""

import ast
import hashlib

VERSION = "direct_launch_v1"
ASSUMPTIONS = [
    "Imported library names denote their usual Torch/Triton APIs;runtime rebinding is unproved."
]
UNKNOWNS = [
    "Reachability,aliasing,external side effects,mask coverage and complete output initialization are unproved."
]


def analyze(source, *, path, source_sha256, entry_symbol=None, max_source=32768, max_nodes=2000):
    report = dict(
        status="not_applicable",
        source_ref=dict(path=path, sha256=source_sha256),
        relations=[],
        assumptions=ASSUMPTIONS[:],
        unknowns=UNKNOWNS[:],
        coverage=VERSION,
        acceptance_proof=False,
    )

    def unknown(reason):
        report["unknowns"].append(reason)

    if not isinstance(source, str) or len(source.encode("utf8")) > max_source:
        return dict(report, status="unknown", unknowns=["source bound"])
    if hashlib.sha256(source.encode("utf8")).hexdigest() != source_sha256:
        return dict(report, status="unknown", unknowns=["source identity mismatch"])
    try:
        tree = ast.parse(source)
        if sum(1 for _ in ast.walk(tree)) > max_nodes:
            return dict(report, status="unknown", unknowns=["AST bound"])
    except (SyntaxError, RecursionError, ValueError):
        return dict(report, status="unknown", unknowns=["source not completely parseable"])

    def names_bound(node):
        bound = {
            n.id
            for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))
        }
        for value in ast.walk(node):
            if isinstance(value, ast.Attribute) and isinstance(value.ctx, (ast.Store, ast.Del)):
                base = value.value
                while isinstance(base, ast.Attribute):
                    base = base.value
                if isinstance(base, ast.Name):
                    bound.add(base.id)
        bound.update(n.arg for n in ast.walk(node) if isinstance(n, ast.arg))
        bound.update(
            n.name
            for n in ast.walk(node)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        )
        bound.update(
            a.asname or a.name.split(".")[0]
            for n in ast.walk(node)
            if isinstance(n, (ast.Import, ast.ImportFrom))
            for a in n.names
        )
        return bound

    aliases = {}
    module_bindings = {}
    for node in tree.body:
        for name in (
            names_bound(node)
            if not isinstance(node, (ast.FunctionDef, ast.ClassDef))
            else [node.name]
        ):
            module_bindings[name] = module_bindings.get(name, 0) + 1
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name in ("torch", "triton", "triton.language"):
                    aliases[a.asname or a.name.split(".")[0]] = (
                        a.name if a.asname else a.name.split(".")[0]
                    )
        if isinstance(node, ast.ImportFrom) and node.module in (
            "torch",
            "triton",
            "triton.language",
        ):
            for a in node.names:
                if a.name != "*":
                    aliases[a.asname or a.name] = node.module + "." + a.name
    aliases = {k: v for k, v in aliases.items() if module_bindings.get(k) == 1}

    def resolved(node, bindings):
        if isinstance(node, ast.Name):
            return bindings.get(node.id)
        if isinstance(node, ast.Attribute):
            base = resolved(node.value, bindings)
            return base + "." + node.attr if base else None
        return None

    def local_aliases(fn):
        return {k: v for k, v in aliases.items() if k not in names_bound(fn)}

    kernels = {}
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef) or module_bindings.get(fn.name) != 1:
            continue
        if len(fn.decorator_list) == 1 and resolved(fn.decorator_list[0], aliases) == "triton.jit":
            kernels[fn.name] = fn
    hosts = [
        (n.name, n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name not in kernels
    ]
    hosts += [
        (c.name + "." + n.name, n)
        for c in tree.body
        if isinstance(c, ast.ClassDef)
        for n in c.body
        if isinstance(n, ast.FunctionDef)
    ]
    for host_name, host in hosts[:16]:
        if entry_symbol is not None and host_name != entry_symbol:
            continue
        imports = local_aliases(host)
        host_bound = names_bound(host)
        assignments = {}
        for row in host.body:
            if (
                isinstance(row, ast.Assign)
                and len(row.targets) == 1
                and isinstance(row.targets[0], ast.Name)
            ):
                assignments.setdefault(row.targets[0].id, []).append(row)
        # Only direct top-level launches/returns. Branches,helpers and aliases
        # never silently establish a complete write/ownership analysis.
        returns = [
            row
            for row in host.body
            if isinstance(row, ast.Return) and isinstance(row.value, ast.Name)
        ]
        launches = [
            row.value
            for row in host.body
            if isinstance(row, ast.Expr)
            and isinstance(row.value, ast.Call)
            and isinstance(row.value.func, ast.Subscript)
        ]
        if any(
            isinstance(n, (ast.If, ast.For, ast.While, ast.Try, ast.With)) for n in ast.walk(host)
        ):
            unknown(host_name + ":control flow not modeled")
        for launch in launches:
            callee = launch.func.value
            if (
                not isinstance(callee, ast.Name)
                or callee.id not in kernels
                or callee.id in host_bound
            ):
                unknown(host_name + ":unresolved or shadowed launch")
                continue
            kernel = kernels[callee.id]
            params = kernel.args.posonlyargs + kernel.args.args
            formal_names = [a.arg for a in params] + [a.arg for a in kernel.args.kwonlyargs]
            if (
                kernel.args.vararg
                or kernel.args.kwarg
                or any(isinstance(a, ast.Starred) for a in launch.args)
                or any(k.arg is None for k in launch.keywords)
            ):
                unknown(host_name + ":dynamic argument binding")
                continue
            if len(launch.args) > len(params):
                unknown(host_name + ":invalid positional binding")
                continue
            binding = dict(zip([a.arg for a in params], launch.args))
            valid = True
            for kw in launch.keywords:
                if kw.arg in binding:
                    valid = False
                elif kw.arg in formal_names:
                    binding[kw.arg] = kw.value
                elif kw.arg not in ("num_warps", "num_stages", "num_ctas", "enable_fp_fusion"):
                    valid = False
            if not valid:
                unknown(host_name + ":ambiguous argument binding")
                continue
            rebound = {
                n.id
                for n in ast.walk(kernel)
                if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))
            }
            kernel_imports = local_aliases(kernel)
            for store in ast.walk(kernel):
                if not isinstance(store, ast.Call):
                    continue
                if resolved(store.func, kernel_imports) != "triton.language.store":
                    if isinstance(store.func, ast.Attribute) and store.func.attr == "store":
                        unknown(host_name + ":store binding not established")
                    continue
                pointer = (
                    store.args[0]
                    if store.args
                    else next((k.value for k in store.keywords if k.arg == "pointer"), None)
                )
                base = pointer
                index_terms = []
                while isinstance(base, ast.BinOp) and isinstance(base.op, ast.Add):
                    index_terms.append(base.right)
                    base = base.left
                if not isinstance(base, ast.Name) or base.id not in binding or base.id in rebound:
                    unknown(host_name + ":unknown store pointer root or parameter reassignment")
                    continue
                if any(
                    n.id in formal_names
                    for term in index_terms
                    for n in ast.walk(term)
                    if isinstance(n, ast.Name)
                ):
                    unknown(
                        host_name
                        + ":index contains another formal;pointer/scalar distinction unknown"
                    )
                    continue
                actual = binding[base.id]
                for ret in returns[:2]:
                    name = ret.value.id
                    writes = [
                        n
                        for n in ast.walk(host)
                        if isinstance(n, ast.Name)
                        and n.id == name
                        and isinstance(n.ctx, (ast.Store, ast.Del))
                    ]
                    allocation = assignments.get(name, [])
                    allocation = allocation[0] if len(allocation) == len(writes) == 1 else None
                    allocated = (
                        allocation is not None
                        and allocation.lineno < launch.lineno
                        and isinstance(allocation.value, ast.Call)
                        and resolved(allocation.value.func, imports)
                        in (
                            "torch.empty",
                            "torch.empty_like",
                            "torch.zeros",
                            "torch.zeros_like",
                            "torch.ones",
                            "torch.ones_like",
                        )
                    )
                    actuals = list(binding.values())
                    report["relations"].append(
                        dict(
                            host=host_name,
                            kernel=kernel.name,
                            store_parameter=base.id,
                            launch_argument=ast.unparse(actual)[:160],
                            returned_name=name,
                            return_origin="direct_allocation_call" if allocated else "unknown",
                            allocation_line=allocation.lineno if allocated else None,
                            returned_name_directly_passed=any(
                                isinstance(a, ast.Name) and a.id == name for a in actuals
                            ),
                            source_lines=dict(
                                store=store.lineno, launch=launch.lineno, returned=ret.lineno
                            ),
                            interpretation="Syntactic parameter/argument/return connection only;absence of a direct argument does not prove absent writes through aliases or other calls.",
                        )
                    )
                    if len(report["relations"]) == 4:
                        break
                if len(report["relations"]) == 4:
                    break
            if len(report["relations"]) == 4:
                break
        if len(report["relations"]) == 4:
            break
    report["unknowns"] = list(dict.fromkeys(report["unknowns"]))[:8]
    report["status"] = (
        "source_relations"
        if report["relations"]
        else "unknown"
        if len(report["unknowns"]) > len(UNKNOWNS)
        else "not_applicable"
    )
    return report
