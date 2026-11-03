"""Public-AST RMSNorm lowering;never execute a reference or use task IDs."""

import ast
import copy
from dataclasses import dataclass
import hashlib
import math


@dataclass(frozen=True)
class RMSNormDescriptor:
    constructor_signature: str
    norm_field: str
    input_name: str
    reference_sha256: str


@dataclass(frozen=True)
class RMSNormCandidate:
    source: str
    descriptor: RMSNormDescriptor
    preconditions: tuple


def _body(node):
    return [
        n
        for n in node.body
        if not (
            isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Constant)
            and isinstance(n.value.value, str)
        )
    ]


def _analyze(source):
    if not isinstance(source, str) or len(source.encode()) > 65536:
        raise ValueError("bounded public reference required")
    tree = ast.parse(source)
    if sum(1 for _ in ast.walk(tree)) > 4000:
        raise ValueError("public AST bound exceeded")
    aliases = {}
    model = None
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ("torch", "torch.nn"):
                    raise ValueError("unsupported import")
                name = alias.asname or alias.name.split(".")[0]
                value = alias.name if alias.asname else "torch"
                if name in aliases and aliases[name] != value:
                    raise ValueError("rebound alias")
                aliases[name] = value
        elif isinstance(node, ast.ImportFrom) and node.module == "torch" and not node.level:
            for alias in node.names:
                if alias.name != "nn":
                    raise ValueError("unsupported import binding")
                name = alias.asname or alias.name
                if name in aliases:
                    raise ValueError("duplicate alias")
                aliases[name] = "torch.nn"
        elif isinstance(node, ast.ClassDef) and node.name == "Model" and model is None:
            model = node
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            pass
        else:
            raise ValueError("unsupported module statement")
    if any(
        name in ("triton", "tl", "math", "ModelNew") or name.startswith("_ic_") for name in aliases
    ):
        raise ValueError("reserved generation binding")

    def qualified(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return qualified(node.value) + "." + node.attr
        return ""

    if (
        model is None
        or model.decorator_list
        or model.keywords
        or len(model.bases) != 1
        or qualified(model.bases[0]) != "torch.nn.Module"
    ):
        raise ValueError("one ordinary nn.Module Model required")
    methods = {}
    for node in _body(model):
        if (
            not isinstance(node, ast.FunctionDef)
            or node.name not in ("__init__", "forward")
            or node.name in methods
            or node.decorator_list
        ):
            raise ValueError("unsupported Model method or state")
        methods[node.name] = node
    if set(methods) != {"__init__", "forward"}:
        raise ValueError("constructor and forward required")
    for method in methods.values():
        a = method.args
        if a.posonlyargs or a.kwonlyargs or a.vararg or a.kwarg or a.kw_defaults:
            raise ValueError("ordinary positional interface required")
        for arg in a.args:
            if (
                arg.arg in aliases
                or arg.arg in ("triton", "tl", "math")
                or arg.arg.startswith("_ic_")
            ):
                raise ValueError("shadowed generation binding")
            if arg.annotation is not None and ast.unparse(arg.annotation) not in (
                "int",
                "float",
                "torch.Tensor",
            ):
                raise ValueError("unsupported annotation")
        if method.returns is not None and ast.unparse(method.returns) not in (
            "None",
            "torch.Tensor",
        ):
            raise ValueError("unsupported return annotation")
    init = methods["__init__"]
    a = init.args
    if len(a.args) not in (2, 3) or a.args[0].arg != "self":
        raise ValueError("shape and optional epsilon constructor required")
    for default in a.defaults:
        value = ast.literal_eval(default)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("finite positive constructor defaults required")
    statements = _body(init)
    if (
        len(statements) != 2
        or not isinstance(statements[0], ast.Expr)
        or not isinstance(statements[0].value, ast.Call)
        or ast.unparse(statements[0].value) != "super().__init__()"
    ):
        raise ValueError("constructor may only initialize the Module and RMSNorm")
    assignment = statements[1]
    if (
        not isinstance(assignment, ast.Assign)
        or len(assignment.targets) != 1
        or not isinstance(assignment.targets[0], ast.Attribute)
        or not isinstance(assignment.targets[0].value, ast.Name)
        or assignment.targets[0].value.id != "self"
        or assignment.targets[0].attr.startswith("_")
        or not isinstance(assignment.value, ast.Call)
    ):
        raise ValueError("one ordinary RMSNorm field required")
    field = assignment.targets[0].attr
    call = assignment.value
    if qualified(call.func) != "torch.nn.RMSNorm" or len(call.args) > 2:
        raise ValueError("constructor must bind nn.RMSNorm")
    options = dict(zip(("normalized_shape", "eps"), call.args))
    for kw in call.keywords:
        if kw.arg not in ("normalized_shape", "eps", "elementwise_affine") or kw.arg in options:
            raise ValueError("unsupported or duplicate RMSNorm option")
        options[kw.arg] = kw.value
    if "elementwise_affine" in options and not (
        isinstance(options["elementwise_affine"], ast.Constant)
        and options["elementwise_affine"].value is True
    ):
        raise ValueError("affine RMSNorm required")
    shape = options.get("normalized_shape")
    epsilon = options.get("eps")
    if not isinstance(shape, ast.Name) or shape.id != a.args[1].arg:
        raise ValueError("RMSNorm must use the original shape parameter")
    if len(a.args) == 3:
        if not isinstance(epsilon, ast.Name) or epsilon.id != a.args[2].arg:
            raise ValueError("RMSNorm must use the original epsilon parameter")
    else:
        value = ast.literal_eval(epsilon) if epsilon is not None else None
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("explicit positive epsilon required")
    forward = methods["forward"]
    f = forward.args
    body = _body(forward)
    if (
        len(f.args) != 2
        or f.args[0].arg != "self"
        or f.defaults
        or len(body) != 1
        or not isinstance(body[0], ast.Return)
    ):
        raise ValueError("one direct single-tensor forward required")
    invocation = body[0].value
    if (
        not isinstance(invocation, ast.Call)
        or ast.unparse(invocation.func) != "self." + field
        or len(invocation.args) != 1
        or invocation.keywords
        or not isinstance(invocation.args[0], ast.Name)
        or invocation.args[0].id != f.args[1].arg
    ):
        raise ValueError("forward must directly call the original RMSNorm field")
    return (
        tree,
        model,
        RMSNormDescriptor(
            ast.unparse(a), field, f.args[1].arg, hashlib.sha256(source.encode()).hexdigest()
        ),
    )


_BACKEND = """import torch
import math
import triton
import triton.language as tl

@triton.jit
def _ic_rmsnorm_kernel(X, W, Y, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + col, col < N, other=0).to(tl.float32)
    w = tl.load(W + col, col < N, other=0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, axis=0) / N + EPS)
    tl.store(Y + row * N + col, x * inv * w, col < N)

def _ic_rmsnorm_forward(x, weight, epsilon, normalized_shape, training):
    if training or not torch.is_inference_mode_enabled() or torch.is_grad_enabled():
        raise ValueError('RMSNorm producer requires eval and inference_mode')
    if (type(x) is not torch.Tensor or type(weight) is not torch.nn.Parameter
            or x.layout != torch.strided or weight.layout != torch.strided
            or x.device.type != 'cuda' or weight.device != x.device
            or x.dtype != torch.float32 or weight.dtype != torch.float32
            or x.ndim != 2 or x.shape[0] <= 0 or not 1 <= x.shape[1] <= 8192
            or not x.is_contiguous() or not weight.is_contiguous()
            or tuple(normalized_shape) != (x.shape[1],) or tuple(weight.shape) != (x.shape[1],)
            or any(name is not None for name in x.names)):
        raise ValueError('unsupported declared FP32 CUDA rank2 RMSNorm domain')
    if type(epsilon) not in (int,float) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('explicit finite positive epsilon required')
    output = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _ic_rmsnorm_kernel[(x.shape[0],)](x, weight, output, x.shape[1], epsilon,
            triton.next_power_of_2(x.shape[1]), num_warps=4 if x.shape[1] <= 2048 else 8)
    return output
"""


def generate(source):
    tree, model, descriptor = _analyze(source)
    model = copy.deepcopy(model)
    model.name = "ModelNew"
    forward = next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    field = descriptor.norm_field
    name = descriptor.input_name
    forward.body = ast.parse(
        "return _ic_rmsnorm_forward("
        + name
        + ",self."
        + field
        + ".weight,self."
        + field
        + ".eps,self."
        + field
        + ".normalized_shape,self.training)"
    ).body
    imports = [copy.deepcopy(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    candidate = (
        _BACKEND
        + "\n"
        + ast.unparse(
            ast.fix_missing_locations(ast.Module(body=[*imports, model], type_ignores=[]))
        )
        + "\n"
    )
    ast.parse(candidate)
    return RMSNormCandidate(
        candidate,
        descriptor,
        (
            "Caller-authorized FP32 CUDA rank2 contiguous affine RMSNorm inference only",
            "Ordinary tensors and dispatch,eval,inference_mode;no gradient or alternate execution semantics",
            "Original shape,epsilon,parameter field/state_dict and weights remain authoritative",
            "Independent correctness and complete-forward performance validation still required",
        ),
    )
