"""Python reaching-definition candidates. Branches merge; loops use a fixed point.

Scope-local names only: aliasing, heap writes and dynamic dispatch are unknown.
Results are evidence for navigation, never a proof of whole-program semantics.
"""

import ast


def python_use_def(source):
    tree = ast.parse(source)
    edges = set()
    limitations = set()

    def loads(node, env, scope):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return
        if isinstance(node, ast.NamedExpr):
            # The RHS is evaluated before the assignment becomes visible to
            # following operands,conditions or statements.
            loads(node.value, env, scope)
            bind(node.target, env, node.lineno)
            return
        if isinstance(node, ast.BoolOp):
            loads(node.values[0], env, scope)
            for operand in node.values[1:]:
                optional = {k: set(v) for k, v in env.items()}
                loads(operand, optional, scope)
                updated = merge(env, optional)
                env.clear()
                env.update(updated)
            return
        if isinstance(node, ast.IfExp):
            loads(node.test, env, scope)
            left = {k: set(v) for k, v in env.items()}
            right = {k: set(v) for k, v in env.items()}
            loads(node.body, left, scope)
            loads(node.orelse, right, scope)
            updated = merge(left, right)
            env.clear()
            env.update(updated)
            return
        if isinstance(node, ast.Compare):
            loads(node.left, env, scope)
            for i, operand in enumerate(node.comparators):
                if i == 0:
                    loads(operand, env, scope)
                else:
                    optional = {k: set(v) for k, v in env.items()}
                    loads(operand, optional, scope)
                    updated = merge(env, optional)
                    env.clear()
                    env.update(updated)
            return
        if isinstance(node, ast.Dict):
            # AST field order groups all keys before all values;Python pairs
            # each key with its value instead. Assignment expressions expose
            # this distinction even when both keys are syntactically valid.
            for key, value in zip(node.keys, node.values):
                if key is not None:
                    loads(key, env, scope)
                loads(value, env, scope)
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            # The first iterable is eager even for a generator expression.
            # Iteration targets belong to the comprehension,whereas walrus
            # bindings can affect the enclosing scope only if execution occurs.
            limitations.add(
                "comprehension iteration counts and generator resumption are unresolved"
            )
            loads(node.generators[0].iter, env, scope)
            optional = {k: set(v) for k, v in env.items()}
            local_scope = (
                scope + ".<comprehension:" + str(node.lineno) + ":" + str(node.col_offset) + ">"
            )
            local_names = {
                child.id
                for generator in node.generators
                for child in ast.walk(generator.target)
                if isinstance(child, ast.Name)
            }
            for i, generator in enumerate(node.generators):
                if i:
                    loads(generator.iter, optional, local_scope)
                bind(generator.target, optional, generator.target.lineno)
                for condition in generator.ifs:
                    loads(condition, optional, local_scope)
            if isinstance(node, ast.DictComp):
                loads(node.key, optional, local_scope)
                loads(node.value, optional, local_scope)
            else:
                loads(node.elt, optional, local_scope)
            if not isinstance(node, ast.GeneratorExp):
                outer = {k: set(v) for k, v in env.items()}
                for name, value in optional.items():
                    if name not in local_names:
                        outer[name] = value
                updated = merge(env, outer)
                env.clear()
                env.update(updated)
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            definitions = env.get(node.id, {0})
            for line in definitions:
                edges.add((scope, node.id, node.lineno, line))
        for child in ast.iter_child_nodes(node):
            loads(child, env, scope)

    def merge(a, b):
        return {k: a.get(k, {0}) | b.get(k, {0}) for k in a.keys() | b.keys()}

    def bind(target, env, line):
        if isinstance(target, ast.Name):
            env[target.id] = {line}
        elif isinstance(target, ast.Starred):
            bind(target.value, env, line)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for child in target.elts:
                bind(child, env, line)
        else:
            limitations.add("heap/alias writes are not resolved")

    def unbind(target, env):
        if isinstance(target, ast.Name):
            env[target.id] = {0}
        elif isinstance(target, (ast.Tuple, ast.List)):
            for child in target.elts:
                unbind(child, env)
        else:
            # Deleting a field/item does not delete its base local name.
            limitations.add("heap/alias writes are not resolved")

    def block(body, env, scope):
        env = {k: set(v) for k, v in env.items()}
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for item in (
                    node.decorator_list
                    + node.args.defaults
                    + [x for x in node.args.kw_defaults if x]
                ):
                    loads(item, env, scope)
                args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
                if node.args.vararg:
                    args += [node.args.vararg]
                if node.args.kwarg:
                    args += [node.args.kwarg]
                nested = scope + "." + node.name
                block(node.body, {a.arg: {a.lineno} for a in args}, nested)
                env[node.name] = {node.lineno}
            elif isinstance(node, ast.ClassDef):
                for item in node.bases + node.decorator_list:
                    loads(item, env, scope)
                block(node.body, {}, scope + "." + node.name)
                env[node.name] = {node.lineno}
            elif isinstance(node, ast.If):
                loads(node.test, env, scope)
                env = merge(block(node.body, env, scope), block(node.orelse, env, scope))
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
                entry = env
                if isinstance(node, ast.While):
                    loads(node.test, env, scope)
                else:
                    loads(node.iter, env, scope)
                # Finite domain: each iteration adds only definition line numbers.
                for _ in range(64):
                    loop_env = {k: set(v) for k, v in env.items()}
                    if not isinstance(node, ast.While):
                        bind(node.target, loop_env, node.lineno)
                    updated = merge(entry, block(node.body, loop_env, scope))
                    if updated == env:
                        break
                    env = updated
                else:
                    limitations.add("loop analysis reached iteration bound")
                if isinstance(node, ast.While):
                    loads(node.test, env, scope)
                env = merge(env, block(node.orelse, env, scope))
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                if node.value:
                    loads(node.value, env, scope)
                for target in getattr(node, "targets", [getattr(node, "target", None)]):
                    if target is not None:
                        # Store targets can still read address/index operands.
                        loads(target, env, scope)
                        bind(target, env, node.lineno)
            elif isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name):
                    for line in env.get(node.target.id, {0}):
                        edges.add((scope, node.target.id, node.lineno, line))
                loads(node.target, env, scope)
                loads(node.value, env, scope)
                bind(node.target, env, node.lineno)
            elif isinstance(node, ast.Delete):
                for target in node.targets:
                    loads(target, env, scope)
                    unbind(target, env)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    env[alias.asname or alias.name.split(".")[0]] = {node.lineno}
            elif isinstance(node, (ast.Try, getattr(ast, "TryStar", ast.Try))):
                limitations.add("exception paths conservatively merged")
                branches = [block(node.body, env, scope)] + [
                    block(h.body, env, scope) for h in node.handlers
                ]
                for branch in branches:
                    env = merge(env, branch)
                env = block(node.finalbody, merge(env, block(node.orelse, env, scope)), scope)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    loads(item.context_expr, env, scope)
                    if item.optional_vars:
                        bind(item.optional_vars, env, node.lineno)
                env = block(node.body, env, scope)
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                limitations.add("global/nonlocal rebinding is unresolved")
            else:
                loads(node, env, scope)
                if isinstance(node, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                    limitations.add("early exits conservatively approximated")
        return env

    block(tree.body, {}, "<module>")
    return {
        "edges": [
            dict(
                scope=s,
                name=n,
                use_line=u,
                definition_line=d,
                confidence="scope-local-static-candidate" if d else "unresolved",
            )
            for s, n, u, d in sorted(edges)
        ],
        "limitations": sorted(
            limitations
            | {
                "closures, heap aliasing, comprehensions and dynamic calls require source inspection"
            }
        ),
    }
