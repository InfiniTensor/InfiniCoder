"""Compile a restricted public single-axis sum AST to a stride-aware kernel.

This module never imports/executes the reference or inspects benchmark IDs.
Caller FP32 CUDA forward/rank/resource declarations remain prerequisites.
The emitted program allocates with torch and computes only with Triton.
"""

import ast
from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class SumDescriptor:
    constructor_signature: str
    constructor_field: str | None
    constructor_parameter: str | None
    dimension_expression: str
    keepdim: bool
    input_name: str
    reference_sha256: str
    constructor_values: tuple


@dataclass(frozen=True)
class ReductionCandidate:
    source: str
    descriptor: SumDescriptor
    schedule: str
    preconditions: tuple


def _value(node, env):
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ValueError("unknown static constructor binding")
        return env[node.id]
    return ast.literal_eval(node)


def _body(function):
    return [
        n
        for n in function.body
        if not (
            isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Constant)
            and isinstance(n.value.value, str)
        )
    ]


def analyze(source):
    if not isinstance(source, str) or len(source.encode()) > 65536:
        raise ValueError("bounded public reference required")
    tree = ast.parse(source)
    if sum(1 for _ in ast.walk(tree)) > 4000:
        raise ValueError("public reference AST bound exceeded")
    aliases = {}
    constants = {}
    model = None
    init_inputs = None
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in ("torch", "torch.nn", "torch.nn.functional"):
                    raise ValueError("unknown reference import")
                aliases[a.asname or a.name.split(".")[0]] = a.name if a.asname else "torch"
        elif isinstance(node, ast.ImportFrom):
            if (
                node.module != "torch"
                or node.level
                or any(a.name not in ("nn", "sum") for a in node.names)
            ):
                raise ValueError("unknown reference binding")
            for a in node.names:
                aliases[a.asname or a.name] = "torch." + a.name
        elif isinstance(node, ast.ClassDef):
            if model is not None or node.name != "Model" or node.decorator_list or node.keywords:
                raise ValueError("one undecorated Model entry required")
            model = node
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if (
                type(node) is not ast.FunctionDef
                or node.name not in ("get_inputs", "get_init_inputs")
                or node.decorator_list
            ):
                raise ValueError("unknown public helper")
            if node.name == "get_init_inputs":
                if init_inputs is not None:
                    raise ValueError("duplicate constructor helper")
                init_inputs = node
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            if node.targets[0].id in aliases or node.targets[0].id in ("Model", "get_init_inputs"):
                raise ValueError("reference rebinds a semantic name")
            try:
                constants[node.targets[0].id] = _value(node.value, constants)
            except (ValueError, TypeError, KeyError):
                raise ValueError("global metadata must be a static literal") from None
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            pass
        else:
            raise ValueError("unsupported reference module statement")
    if model is None:
        raise ValueError("public Model not found")

    def qualified(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return qualified(node.value) + "." + node.attr
        return ""

    if len(model.bases) != 1 or qualified(model.bases[0]) != "torch.nn.Module":
        raise ValueError("Model must have the ordinary torch.nn.Module base")
    for n in ast.walk(tree):
        if isinstance(n, (ast.Global, ast.Nonlocal)):
            raise ValueError("mutable global binding is unsupported")
        if isinstance(n, ast.arg) and n.arg in aliases:
            raise ValueError("shadowed tensor-library alias")
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id
            in (
                "exec",
                "eval",
                "compile",
                "globals",
                "locals",
                "getattr",
                "setattr",
                "delattr",
                "__import__",
            )
        ):
            raise ValueError("dynamic semantic binding is unsupported")
        if isinstance(n, (ast.Attribute, ast.Subscript)) and isinstance(n.ctx, ast.Store):
            root = n
            while isinstance(root, (ast.Attribute, ast.Subscript)):
                root = root.value
            if isinstance(root, ast.Name) and root.id in aliases:
                raise ValueError("library mutation is unsupported")
    methods = {}
    for n in model.body:
        if (
            isinstance(n, ast.FunctionDef)
            and n.name in ("__init__", "forward")
            and n.name not in methods
            and not n.decorator_list
        ):
            methods[n.name] = n
        elif (
            isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Constant)
            and isinstance(n.value.value, str)
        ):
            pass
        else:
            raise ValueError("unknown Model state or methods")
    forward = methods.get("forward")
    if forward is None:
        raise ValueError("forward method required")
    a = forward.args
    if (
        a.posonlyargs
        or a.vararg
        or a.kwarg
        or a.kwonlyargs
        or a.defaults
        or len(a.args) != 2
        or a.args[0].arg != "self"
    ):
        raise ValueError("single tensor forward interface required")
    input_name = a.args[1].arg
    statements = _body(forward)
    if (
        len(statements) != 1
        or not isinstance(statements[0], ast.Return)
        or not isinstance(statements[0].value, ast.Call)
    ):
        raise ValueError("forward must return one direct reduction")
    call = statements[0].value
    if qualified(call.func) == "torch.sum":
        if not call.args or not isinstance(call.args[0], ast.Name) or call.args[0].id != input_name:
            raise ValueError("sum must consume the forward tensor directly")
        positional = call.args[1:]
    elif (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "sum"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == input_name
    ):
        positional = call.args
    else:
        raise ValueError("not a bound single-axis torch sum")
    if len(positional) > 2 or any(k.arg not in ("dim", "keepdim") for k in call.keywords):
        raise ValueError("dtype/out or non-axis sum options are unsupported")
    options = dict(zip(("dim", "keepdim"), positional))
    for k in call.keywords:
        if k.arg in options:
            raise ValueError("duplicate sum argument")
        options[k.arg] = k.value
    if "dim" not in options:
        raise ValueError("an explicit single axis is required")
    keepdim = _value(options["keepdim"], constants) if "keepdim" in options else False
    if type(keepdim) is not bool:
        raise ValueError("keepdim must be a literal/static bool")
    init = methods.get("__init__")
    field = parameter = None
    signature = "self"
    defaults = {}
    if init is not None:
        a = init.args
        if (
            a.posonlyargs
            or a.vararg
            or a.kwarg
            or a.kwonlyargs
            or not 1 <= len(a.args) <= 2
            or a.args[0].arg != "self"
        ):
            raise ValueError("unsupported constructor interface")
        for arg in a.args:
            if arg.annotation is not None and (
                not isinstance(arg.annotation, ast.Name) or arg.annotation.id != "int"
            ):
                raise ValueError("only a plain integer constructor annotation is supported")
        signature = ast.unparse(a)
        for arg, value in zip(a.args[-len(a.defaults) :], a.defaults):
            defaults[arg.arg] = _value(value, constants)
        for n in _body(init):
            if (
                isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Call)
                and ast.unparse(n.value.func) == "super().__init__"
                and not n.value.args
                and not n.value.keywords
            ):
                continue
            if (
                isinstance(n, ast.Assign)
                and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Attribute)
                and isinstance(n.targets[0].value, ast.Name)
                and n.targets[0].value.id == "self"
                and isinstance(n.value, ast.Name)
                and len(a.args) == 2
                and n.value.id == a.args[1].arg
                and field is None
            ):
                field = n.targets[0].attr
                parameter = n.value.id
            else:
                raise ValueError("constructor may only bind its scalar axis field")
    dim = options["dim"]
    if (
        isinstance(dim, ast.Attribute)
        and isinstance(dim.value, ast.Name)
        and dim.value.id == "self"
        and dim.attr == field
        and field is not None
    ):
        expression = "self." + field
    else:
        axis = _value(dim, constants)
        if type(axis) is not int:
            raise ValueError("multiple/dynamic axes are unsupported")
        expression = repr(axis)
    values = []
    if init_inputs is not None:
        if (
            init_inputs.args.args
            or init_inputs.args.posonlyargs
            or init_inputs.args.kwonlyargs
            or init_inputs.args.vararg
            or init_inputs.args.kwarg
        ):
            raise ValueError("constructor helper must have no arguments")
        env = dict(constants)
        returned = False
        for n in _body(init_inputs):
            if (
                isinstance(n, ast.Assign)
                and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
            ):
                env[n.targets[0].id] = _value(n.value, env)
            elif isinstance(n, ast.Return) and not returned:
                if isinstance(n.value, (ast.List, ast.Tuple)):
                    values = [_value(v, env) for v in n.value.elts]
                else:
                    values = _value(n.value, env)
                returned = True
            else:
                raise ValueError("unknown constructor helper side effect")
        if not returned or not isinstance(values, (list, tuple)):
            raise ValueError("static constructor argument sequence required")
    if init is None and values:
        raise ValueError("constructor values do not match the entry")
    if init is not None:
        names = [arg.arg for arg in init.args.args[1:]]
        if len(values) > len(names):
            raise ValueError("constructor argument count mismatch")
        bound = dict(defaults)
        bound.update(zip(names, values))
        if any(name not in bound for name in names) or any(
            type(bound[name]) is not int for name in names
        ):
            raise ValueError("constructor axis binding must be a known plain integer")
    return SumDescriptor(
        signature,
        field,
        parameter,
        expression,
        keepdim,
        input_name,
        hashlib.sha256(source.encode()).hexdigest(),
        tuple(values),
    )


KERNELS = """import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _ic_sum_one(X, Y, R:tl.constexpr, N1:tl.constexpr, S0:tl.constexpr, S1:tl.constexpr, SR:tl.constexpr, BR:tl.constexpr):
    out = tl.program_id(0).to(tl.int64)
    r = tl.arange(0, BR).to(tl.int64)
    base = (out // N1) * S0 + (out % N1) * S1
    values = tl.load(X + base + r * SR, r < R, other=0.0)
    total = tl.sum(values.to(tl.float32), axis=0)
    tl.store(Y + out, total)

@triton.jit
def _ic_sum_columns(X, Y, M:tl.constexpr, R:tl.constexpr, N1:tl.constexpr, S0:tl.constexpr, S1:tl.constexpr, SR:tl.constexpr, BR:tl.constexpr, BO:tl.constexpr):
    out = (tl.program_id(0) * BO + tl.arange(0, BO)).to(tl.int64)
    r = tl.arange(0, BR).to(tl.int64)
    base = (out // N1) * S0 + (out % N1) * S1
    values = tl.load(X + r[:, None] * SR + base[None, :], (r[:, None] < R) & (out[None, :] < M), other=0.0)
    total = tl.sum(values.to(tl.float32), axis=0)
    tl.store(Y + out, total, out < M)
"""


def generate(source, schedule="scalar"):
    if schedule not in ("scalar", "coalesced"):
        raise ValueError("unknown sum schedule")
    d = analyze(source)
    return _generate_descriptor(d, schedule)


def _generate_descriptor(d, schedule="scalar"):
    """Internal lowering shared by source-bound callers;the old output is unchanged."""
    if type(d) is not SumDescriptor:
        raise TypeError("validated SumDescriptor required")
    if schedule not in ("scalar", "coalesced"):
        raise ValueError("unknown sum schedule")
    init = "    def __init__(" + d.constructor_signature + "):\n        super().__init__()\n"
    if d.constructor_field is not None:
        init += "        self." + d.constructor_field + " = " + d.constructor_parameter + "\n"
    body = f"""
class ModelNew(nn.Module):
{init}
    def forward(self, {d.input_name}):
        x = {d.input_name}
        if type(x) is not torch.Tensor or x.dtype != torch.float32 or x.device.type != 'cuda' or x.requires_grad:
            raise ValueError('single-axis provider requires an ordinary FP32 CUDA forward tensor')
        axis = {d.dimension_expression}
        if type(axis) is not int or x.ndim not in (2,3) or not -x.ndim <= axis < x.ndim:
            raise ValueError('single integer axis and rank2/rank3 required')
        if axis < 0:
            axis += x.ndim
        shape = list(x.shape)
        if any(n <= 0 for n in shape) or shape[axis] > 8192 or x.numel() > 67108864:
            raise ValueError('input exceeds the declared nonempty reduction resource domain')
        remaining = [i for i in range(x.ndim) if i != axis]
        lengths = [shape[i] for i in remaining]
        strides = [x.stride(i) for i in remaining]
        n1 = lengths[1] if len(lengths) == 2 else 1
        s0 = strides[0]
        s1 = strides[1] if len(strides) == 2 else 0
        sr = x.stride(axis)
        if any(s < 0 for s in (*strides,sr)):
            raise ValueError('negative physical strides unsupported')
        count = x.numel() // shape[axis]
        output_shape = [1 if i == axis else n for i,n in enumerate(shape)] if {d.keepdim!r} else lengths
        y = torch.empty(output_shape, device=x.device, dtype=torch.float32)
        block = triton.next_power_of_2(shape[axis])
        adjacent = s1 if len(lengths) == 2 and n1 > 1 else s0
        if {schedule == "coalesced"!r} and adjacent < sr:
            columns = min(32, max(1, 8192 // block))
            _ic_sum_columns[(triton.cdiv(count,columns),)](x,y,count,shape[axis],n1,s0,s1,sr,block,columns,num_warps=4)
        else:
            _ic_sum_one[(count,)](x,y,shape[axis],n1,s0,s1,sr,block,num_warps=4)
        return y
"""
    program = KERNELS + body
    ast.parse(program)
    return ReductionCandidate(
        program,
        d,
        schedule,
        (
            "caller-declared FP32 CUDA forward only",
            "one statically bound sum axis",
            "rank2/rank3 nonempty tensors",
            "actual nonnegative strides and data-pointer offset",
            "reduction extent <=8192;numel <=67108864",
            "fresh output with original keepdim and dtype;independent correctness still required",
        ),
    )
