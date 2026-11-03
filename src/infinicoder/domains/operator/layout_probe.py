"""Bounded integer index-map checks;never executes model expressions as Python.

This checks a proposed schedule on one declared layout,not an implementation,
mathematical operator semantics,compiler integer widths or cross-buffer aliasing.
"""

import ast
import itertools
import math
import re
from infinicoder.domains.operator.tensor_layout import TensorLayout

MAX_POINTS = 4096
MAX_INTEGER = (1 << 63) - 1


def integer(value):
    if type(value) is not int or abs(value) > MAX_INTEGER:
        raise ValueError("bounded signed64 integer required")
    return value


class Expression:
    def __init__(self, text):
        if type(text) is int:
            text = str(integer(text))
        if not isinstance(text, str) or not 1 <= len(text) <= 512:
            raise ValueError("expression must be at most512 characters")
        try:
            self.tree = ast.parse(text, mode="eval").body
        except (SyntaxError, RecursionError) as exc:
            raise ValueError("invalid integer expression") from exc
        nodes = list(ast.walk(self.tree))
        allowed = (
            ast.Constant,
            ast.Name,
            ast.Load,
            ast.BinOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.FloorDiv,
            ast.Mod,
            ast.UnaryOp,
            ast.USub,
            ast.UAdd,
            ast.Not,
            ast.BoolOp,
            ast.And,
            ast.Or,
            ast.Compare,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            ast.Eq,
            ast.NotEq,
            ast.Call,
            ast.IfExp,
        )
        if len(nodes) > 96 or any(not isinstance(n, allowed) for n in nodes):
            raise ValueError("unsupported or excessive expression syntax")
        for node in nodes:
            if isinstance(node, ast.Constant) and type(node.value) not in (int, bool):
                raise ValueError("only integer/boolean constants allowed")
            if isinstance(node, ast.Name) and (
                not re.fullmatch("[A-Za-z][A-Za-z0-9_]{0,23}", node.id) or "__" in node.id
            ):
                raise ValueError("invalid expression identifier")
            if isinstance(node, ast.Call) and (
                not isinstance(node.func, ast.Name)
                or node.func.id not in ("ceildiv", "min", "max")
                or node.keywords
                or not 2 <= len(node.args) <= 4
            ):
                raise ValueError("only bounded ceildiv,min,max calls allowed")
            if isinstance(node, ast.Call) and node.func.id == "ceildiv" and len(node.args) != 2:
                raise ValueError("ceildiv needs2 arguments")
        self.text = text

    def value(self, environment):
        def boolean(v):
            if type(v) is not bool:
                raise ValueError("boolean guard required")
            return v

        def walk(node):
            if isinstance(node, ast.Constant):
                return node.value if type(node.value) is bool else integer(node.value)
            if isinstance(node, ast.Name):
                if node.id not in environment:
                    raise ValueError("unknown expression variable:" + node.id)
                return integer(environment[node.id])
            if isinstance(node, ast.BinOp):
                a, b = integer(walk(node.left)), integer(walk(node.right))
                if isinstance(node.op, ast.Add):
                    v = a + b
                elif isinstance(node.op, ast.Sub):
                    v = a - b
                elif isinstance(node.op, ast.Mult):
                    v = a * b
                elif isinstance(node.op, ast.FloorDiv):
                    v = a // b
                else:
                    v = a % b
                return integer(v)
            if isinstance(node, ast.UnaryOp):
                v = walk(node.operand)
                return (
                    not boolean(v)
                    if isinstance(node.op, ast.Not)
                    else integer(-integer(v) if isinstance(node.op, ast.USub) else integer(v))
                )
            if isinstance(node, ast.BoolOp):
                for item in node.values:
                    v = boolean(walk(item))
                    if isinstance(node.op, ast.And) and not v:
                        return False
                    if isinstance(node.op, ast.Or) and v:
                        return True
                return isinstance(node.op, ast.And)
            if isinstance(node, ast.Compare):
                left = integer(walk(node.left))
                for operation, right_node in zip(node.ops, node.comparators):
                    right = integer(walk(right_node))
                    matched = (
                        left < right
                        if isinstance(operation, ast.Lt)
                        else left <= right
                        if isinstance(operation, ast.LtE)
                        else left > right
                        if isinstance(operation, ast.Gt)
                        else left >= right
                        if isinstance(operation, ast.GtE)
                        else left == right
                        if isinstance(operation, ast.Eq)
                        else left != right
                    )
                    if not matched:
                        return False
                    left = right
                return True
            if isinstance(node, ast.IfExp):
                return walk(node.body if boolean(walk(node.test)) else node.orelse)
            if isinstance(node, ast.Call):
                values = [integer(walk(n)) for n in node.args]
                if node.func.id == "ceildiv":
                    a, b = values
                    if a < 0 or b <= 0:
                        raise ValueError(
                            "ceildiv requires nonnegative dividend and positive divisor"
                        )
                    return integer((a + b - 1) // b)
                return min(values) if node.func.id == "min" else max(values)
            raise ValueError("unsupported expression")

        return walk(self.tree)


def check(arguments):
    required = {"shape", "element_strides", "axes", "indices", "element_offset", "mode"}
    if (
        not isinstance(arguments, dict)
        or not required <= set(arguments)
        or set(arguments) - required - {"mask"}
    ):
        raise ValueError(
            "layout mapping requires shape,element_strides,axes,indices,element_offset,mode and optional mask"
        )
    shape, strides = arguments["shape"], arguments["element_strides"]
    mode = arguments["mode"]
    if (
        not isinstance(shape, list)
        or not isinstance(strides, list)
        or len(shape) > 4
        or any(type(v) is not int for v in shape + strides)
    ):
        raise ValueError("concrete rank<=4 integer layout required")
    layout = TensorLayout(shape, strides)
    if any(v > MAX_INTEGER for v in shape + strides):
        raise ValueError("layout values exceed signed64 bounds")
    if mode not in ("read", "write"):
        raise ValueError("mode must be read or write;atomic schedules unsupported")
    logical_size = math.prod(shape)
    axes = arguments["axes"]
    indices = arguments["indices"]
    if (
        not isinstance(axes, list)
        or len(axes) > 4
        or not isinstance(indices, list)
        or len(indices) != len(shape)
    ):
        raise ValueError("at most4 axes and one index per tensor dimension required")
    environment = {
        **{"d" + str(i): v for i, v in enumerate(shape)},
        **{"s" + str(i): v for i, v in enumerate(strides)},
    }
    names = []
    extents = []
    for axis in axes:
        if not isinstance(axis, dict) or set(axis) != {"name", "extent"}:
            raise ValueError("axis needs name and extent")
        name = axis["name"]
        if (
            not isinstance(name, str)
            or not re.fullmatch("[A-Za-z][A-Za-z0-9_]{0,23}", name)
            or "__" in name
            or name in {*environment, *names, "ceildiv", "min", "max"}
        ):
            raise ValueError("distinct nonreserved iteration variable required")
        extent = integer(Expression(axis["extent"]).value(environment))
        if extent < 0:
            raise ValueError("nonnegative iteration extent required")
        names.append(name)
        extents.append(extent)
    total = math.prod(extents)
    result = dict(
        status="bounded_design_checked",
        scope="one declared integer index map;not implementation correctness",
        mode=mode,
        thread_points=total,
        logical_elements=logical_size,
        minimum_span_elements=layout.minimum_span(),
        limitations=[
            "Coordinates/offsets relative to this tensor logical origin. No compiler-width,numerical,synchronization,atomic or cross-buffer alias proof.",
            "Read mode checks visited accesses only;does not prove all semantically required inputs are consumed.",
        ],
    )
    if total > MAX_POINTS or mode == "write" and logical_size > MAX_POINTS:
        return dict(
            result,
            status="inconclusive",
            reason="exhaustive domain exceeds4096;no sampled pass substituted",
        )
    index_expr = [Expression(v) for v in indices]
    offset_expr = Expression(arguments["element_offset"])
    mask = Expression(arguments.get("mask", "True"))
    writes = {}
    logical_seen = set()
    active = 0

    def fail(kind, thread, **detail):
        return dict(
            result,
            status="counterexample",
            counterexample=dict(kind=kind, iteration=thread, **detail),
            active_accesses=active,
        )

    for point in itertools.product(*(range(v) for v in extents)):
        thread = dict(zip(names, point))
        values = dict(environment, **thread)
        try:
            enabled = mask.value(values)
            if type(enabled) is not bool:
                raise ValueError("mask must produce boolean")
            if not enabled:
                continue
            coordinates = tuple(integer(expr.value(values)) for expr in index_expr)
            offset = integer(offset_expr.value(values))
        except (ValueError, ArithmeticError) as exc:
            return fail("expression_error", thread, detail=str(exc))
        active += 1
        if any(not 0 <= i < n for i, n in zip(coordinates, shape)):
            return fail("logical_out_of_bounds", thread, indices=coordinates, shape=shape)
        expected = sum(i * s for i, s in zip(coordinates, strides))
        if offset != expected:
            return fail(
                "independent_stride_or_offset_mismatch",
                thread,
                indices=coordinates,
                actual_offset=offset,
                expected_offset=expected,
            )
        if not 0 <= offset < layout.minimum_span():
            return fail("physical_out_of_bounds", thread, offset=offset)
        if mode == "write" and offset in writes:
            return fail(
                "multiple_writers",
                thread,
                indices=coordinates,
                offset=offset,
                previous_iteration=writes[offset],
            )
        writes[offset] = thread
        logical_seen.add(coordinates)
    if mode == "write":
        expected = set(itertools.product(*(range(v) for v in shape)))
        missing = expected - logical_seen
        if missing:
            return fail(
                "unwritten_output",
                {},
                first_missing_indices=min(missing),
                missing_elements=len(missing),
            )
    return dict(
        result,
        active_accesses=active,
        unique_logical_indices=len(logical_seen),
        full_write_coverage=mode == "write",
        unique_writers=mode == "write",
    )
