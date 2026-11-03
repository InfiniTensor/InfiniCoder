"""Bounded evaluation of caller examples against a pure arithmetic source view.

No eval/exec/import of candidate code,process,model,hidden test or source edit.
Unsupported syntax stays unknown. Argument permutations are hypotheses only.
"""

import ast
import hashlib
import itertools
import math
import operator
from pathlib import Path
import re

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def number(value):
    if (
        type(value) not in (int, float)
        or abs(value) > 1e100
        or type(value) is float
        and not math.isfinite(value)
    ):
        raise ValueError("outside bounded scalar arithmetic")
    return value


def expression(node, env):
    if isinstance(node, ast.Constant):
        return number(node.value)
    if isinstance(node, ast.Name) and node.id in env:
        return number(env[node.id])
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = expression(node.operand, env)
        return number(value if isinstance(node.op, ast.UAdd) else -value)
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        a, b = expression(node.left, env), expression(node.right, env)
        if isinstance(node.op, ast.Pow) and abs(b) > 64:
            raise ValueError("power outside bound")
        return number(OPS[type(node.op)](a, b))
    raise ValueError("unsupported expression;not an execution proof")


def functions(source):
    tree = ast.parse(source)
    if len(source) > 32768 or len(list(ast.walk(tree))) > 512:
        return {}
    # Module statements could otherwise replace a function binding or add
    # import/decorator/default/annotation effects that this view cannot model.
    if any(
        not isinstance(n, ast.FunctionDef)
        and not (
            isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Constant)
            and isinstance(n.value.value, str)
        )
        for n in tree.body
    ):
        return {}
    result = {}
    duplicates = set()
    for fn in (n for n in tree.body if isinstance(n, ast.FunctionDef)):
        if fn.name in result:
            duplicates.add(fn.name)
            continue
        args = [*fn.args.posonlyargs, *fn.args.args]
        body = fn.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        if (
            not 1 <= len(args) <= 4
            or fn.decorator_list
            or fn.args.defaults
            or fn.args.kw_defaults
            or fn.args.vararg
            or fn.args.kwarg
            or fn.args.kwonlyargs
            or fn.returns
            or any(a.annotation for a in args)
            or len(body) != 1
            or not isinstance(body[0], ast.Return)
            or body[0].value is None
        ):
            continue
        expr = body[0].value
        if len(list(ast.walk(expr))) > 48:
            continue
        result[fn.name] = ([a.arg for a in args], expr)
    return {k: v for k, v in result.items() if k not in duplicates}


def analyze(task, source, path):
    if not path.endswith(".py") or len(task) > 65536 or len(source) > 32768:
        return None
    try:
        known = functions(source)
    except (SyntaxError, ValueError, RecursionError):
        return None
    cases = {}
    for match in itertools.islice(re.finditer(r"(?m)^[ \t]*assert[ \t]+[^\r\n]{1,1000}", task), 32):
        try:
            nodes = ast.parse(match.group().strip()).body
            if len(nodes) != 1 or not isinstance(nodes[0], ast.Assert) or nodes[0].msg:
                continue
            test = nodes[0].test
            if not (
                isinstance(test, ast.Compare)
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)
                and isinstance(test.left, ast.Call)
                and isinstance(test.left.func, ast.Name)
            ):
                continue
            call = test.left
            name = call.func.id
            if name not in known or call.keywords or len(call.args) != len(known[name][0]):
                continue
            args = [number(ast.literal_eval(n)) for n in call.args]
            expected = number(ast.literal_eval(test.comparators[0]))
            rows = cases.setdefault(name, [])
            if len(rows) < 6:
                rows.append(
                    dict(arguments=args, expected=expected, task_span=[match.start(), match.end()])
                )
        except (SyntaxError, ValueError, TypeError, OverflowError, RecursionError):
            continue
    output = []
    for name, rows in cases.items():
        names, expr = known[name]

        def run(args):
            try:
                return dict(value=expression(expr, dict(zip(names, args))))
            except ZeroDivisionError:
                return dict(exception="ZeroDivisionError")
            except (ValueError, OverflowError, TypeError, RecursionError) as exc:
                return dict(unknown=type(exc).__name__)

        try:
            observed = [run(r["arguments"]) for r in rows]
            if any("unknown" in o for o in observed):
                continue
            if all(o.get("value") == r["expected"] for r, o in zip(rows, observed)):
                continue
            fits = []
            for permutation in itertools.permutations(range(len(names))):
                if permutation == tuple(range(len(names))):
                    continue
                if all(
                    run([r["arguments"][i] for i in permutation]).get("value") == r["expected"]
                    for r in rows
                ):
                    fits.append(list(permutation))
            output.append(
                dict(
                    function=name,
                    formal_parameters=names,
                    examples=[
                        dict(r, predicted=o, contradiction=o.get("value") != r["expected"])
                        for r, o in zip(rows, observed)
                    ][:3],
                    argument_order_hypotheses=fits[:3],
                    hypotheses_omitted=max(0, len(fits) - 3),
                )
            )
        except (ValueError, OverflowError, TypeError, RecursionError):
            continue
        if len(output) >= 1:
            break
    if not output:
        return None
    return dict(
        kind="bounded_static_example_diagnosis",
        path=path,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        task_sha256=hashlib.sha256(task.encode()).hexdigest(),
        analyzer_identity=identity(),
        results=output,
        hypothesis_not_contract=True,
        acceptance_proof=False,
        limitations="Only caller-supplied literal numeric assertions and a pure straight-line arithmetic return are modeled. Permutations rebind example arguments to formal parameters;they are diagnostic hypotheses,not proposed API changes. Derive the intended computation from the full caller contract and verify independently. Dynamic effects,unsupported syntax and hidden cases remain unknown.",
    )
