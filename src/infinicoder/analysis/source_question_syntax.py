"""Source-only field/guard extraction reused from method_interface_view.
Origin SHA256: c3d3096fabb1e769b7ad82baadf7d10aec3a5d81ea60a4ca51992434d8eff4e9
No receiver dispatch,ownership or runtime-value inference.
"""

import ast
from infinicoder.memory.source.source_dependency_view import location

FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _walk(node):
    """Stay within this function's scope; closures are separate receivers."""
    yield node
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (*FUNCTIONS, ast.ClassDef, ast.Lambda)):
            continue
        yield from _walk(child)


def _parents(tree):
    return {
        child: (node, field)
        for node in ast.walk(tree)
        for field, value in ast.iter_fields(node)
        for child in (value if isinstance(value, list) else [value])
        if isinstance(child, ast.AST)
    }


def _source(file, node):
    return "".join(file["source"].splitlines(keepends=True)[node.lineno - 1 : node.end_lineno])


def _conditions(file, node, parents, function):
    result = []
    while node in parents and node is not function:
        parent, branch = parents[node]
        if isinstance(
            parent,
            (
                ast.If,
                ast.For,
                ast.AsyncFor,
                ast.While,
                ast.With,
                ast.AsyncWith,
                ast.Try,
                ast.ExceptHandler,
                ast.Match,
            ),
        ):
            # The original header plus branch name is retained. In particular,
            # an orelse location must not be mistaken for the positive branch.
            body = getattr(parent, "body", [])
            end = max(parent.lineno, body[0].lineno - 1) if body else parent.lineno
            result.append(
                location(
                    file,
                    parent.lineno,
                    end,
                    "guard",
                    source="".join(
                        file["source"].splitlines(keepends=True)[parent.lineno - 1 : end]
                    ),
                    branch=branch,
                    certainty="enclosing_syntax;execution_and_early_exit_unknown",
                )
            )
        if isinstance(parent, ast.match_case):
            result.append(
                location(
                    file,
                    parent.pattern.lineno,
                    parent.pattern.end_lineno,
                    "guard",
                    source=_source(file, parent.pattern),
                    branch=branch,
                    guard=ast.unparse(parent.guard) if parent.guard else None,
                    certainty="case_syntax;execution_unknown",
                )
            )
        node = parent
    return list(reversed(result))


def _receiver(function):
    args = [*function.args.posonlyargs, *function.args.args]
    return args[0].arg if args else None


def _fields(node, receiver, context=None):
    return {
        n.attr
        for n in _walk(node)
        if isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Name)
        and n.value.id == receiver
        and (context is None or isinstance(n.ctx, context))
    }


def _statement(node, parents, function):
    while not isinstance(node, ast.stmt) and node in parents:
        node = parents[node][0]
    return node if isinstance(node, ast.stmt) and node is not function else None


def _piece(file, node, parents, function):
    return location(
        file,
        node.lineno,
        node.end_lineno,
        "statement",
        source=_source(file, node),
        conditions=_conditions(file, node, parents, function),
    )
