"""Source-derived linear/GELU fusion with explicit fallback and tile candidates."""

import ast
import copy
from dataclasses import dataclass
import hashlib
import textwrap


@dataclass(frozen=True)
class FusionCandidate:
    source: str
    entry: str
    schedule: str
    reference_sha256: str
    preconditions: tuple


SCHEDULES = {"compact": (32, 64, 32, 4), "wide": (64, 128, 32, 4)}


def _analyze(source):
    if not isinstance(source, str) or len(source.encode()) > 65536:
        raise ValueError("bounded public Python reference required")
    tree = ast.parse(source)
    aliases = {}
    functions = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name not in ("torch", "torch.nn.functional"):
                    raise ValueError("unsupported import in fusion reference")
                name = item.asname or item.name.split(".")[0]
                value = item.name if item.asname else item.name.split(".")[0]
                if name in aliases:
                    raise ValueError("duplicate import binding")
                aliases[name] = value
        elif isinstance(node, ast.ImportFrom) and node.module == "torch.nn" and not node.level:
            for item in node.names:
                if item.name != "functional":
                    raise ValueError("unsupported imported reference binding")
                name = item.asname or item.name
                if name in aliases:
                    raise ValueError("duplicate import binding")
                aliases[name] = "torch.nn.functional"
        elif isinstance(node, ast.FunctionDef):
            functions.append(node)
        elif not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            raise ValueError("fusion reference must contain imports and one function")
    if len(functions) != 1:
        raise ValueError("one public linear/GELU function required")
    function = functions[0]
    args = function.args
    if (
        function.decorator_list
        or args.posonlyargs
        or args.kwonlyargs
        or args.vararg
        or args.kwarg
        or args.defaults
        or len(args.args) != 3
    ):
        raise ValueError("three ordinary positional inputs without defaults required")
    names = [v.arg for v in args.args]
    if (
        len(set(names)) != 3
        or any(n in aliases for n in names)
        or function.name in aliases
        or function.name.startswith("_ic_")
        or any(n.startswith("_ic_") for n in [*aliases, *names])
    ):
        raise ValueError("shadowed or reserved fusion binding")
    body = [
        v
        for v in function.body
        if not (
            isinstance(v, ast.Expr)
            and isinstance(v.value, ast.Constant)
            and isinstance(v.value.value, str)
        )
    ]
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        raise ValueError("one direct composition return required")

    def qualified(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, "")
        if isinstance(node, ast.Attribute):
            return qualified(node.value) + "." + node.attr
        return ""

    gelu = body[0].value
    if (
        not isinstance(gelu, ast.Call)
        or qualified(gelu.func) != "torch.nn.functional.gelu"
        or len(gelu.args) != 1
        or len(gelu.keywords) != 1
        or gelu.keywords[0].arg != "approximate"
        or not isinstance(gelu.keywords[0].value, ast.Constant)
        or gelu.keywords[0].value.value != "tanh"
    ):
        raise ValueError("explicit tanh GELU composition required")
    linear = gelu.args[0]
    if (
        not isinstance(linear, ast.Call)
        or qualified(linear.func) != "torch.nn.functional.linear"
        or linear.keywords
        or len(linear.args) != 3
        or any(not isinstance(v, ast.Name) for v in linear.args)
        or [v.id for v in linear.args] != names
    ):
        raise ValueError("linear inputs must preserve the declared input/weight/bias order")
    # Annotations must not introduce arbitrary import-time execution.
    allowed = {"torch.Tensor", "int", "float", "None"}
    if any(
        v.annotation is not None and ast.unparse(v.annotation) not in allowed for v in args.args
    ):
        raise ValueError("unsupported reference annotation")
    if function.returns is not None and ast.unparse(function.returns) not in allowed:
        raise ValueError("unsupported reference return annotation")
    return tree, function, names


_KERNEL = """@_ic_triton.jit
def _ic_linear_gelu_kernel(X,W,B,Y,M:_ic_tl.constexpr,N:_ic_tl.constexpr,K:_ic_tl.constexpr,
                           BM:_ic_tl.constexpr,BN:_ic_tl.constexpr,BK:_ic_tl.constexpr):
    rows=_ic_tl.program_id(0)*BM+_ic_tl.arange(0,BM)
    cols=_ic_tl.program_id(1)*BN+_ic_tl.arange(0,BN)
    kk=_ic_tl.arange(0,BK)
    acc=_ic_tl.zeros((BM,BN),_ic_tl.float32)
    for base in range(_ic_tl.cdiv(K,BK)):
        k=base*BK+kk
        a=_ic_tl.load(X+rows[:,None]*K+k[None,:],(rows[:,None]<M)&(k[None,:]<K),other=0)
        b=_ic_tl.load(W+k[:,None]+cols[None,:]*K,(k[:,None]<K)&(cols[None,:]<N),other=0)
        acc=_ic_tl.dot(a,b,acc)
    bias=_ic_tl.load(B+cols,cols<N,other=0).to(_ic_tl.float32)
    # F.linear has a half-precision output before the GELU consumes it.
    z=(acc+bias[None,:]).to(_ic_tl.float16).to(_ic_tl.float32)
    value=0.5*z*(1.0+_ic_math.tanh(0.7978845608028654*(z+0.044715*z*z*z)))
    _ic_tl.store(Y+rows[:,None]*N+cols[None,:],value,(rows[:,None]<M)&(cols[None,:]<N))
"""

_WRAPPER = """def _ic_linear_gelu(x,weight,bias,fallback):
    tensors=(x,weight,bias)
    if (_ic_triton is None or _ic_torch.is_autocast_enabled()
            or any(type(v) not in (_ic_torch.Tensor,_ic_torch.nn.Parameter) for v in tensors)
            or any(v.layout!=_ic_torch.strided for v in tensors)
            or x.device.type!='cuda' or any(v.device!=x.device for v in tensors)
            or any(v.dtype!=_ic_torch.float16 or not v.is_contiguous() for v in tensors)
            or x.ndim!=2 or weight.ndim!=2 or bias.ndim!=1
            or x.shape[1]!=weight.shape[1] or weight.shape[0]!=bias.shape[0]
            or min(*x.shape,*weight.shape)<=0
            or any(any(name is not None for name in v.names) for v in tensors)
            or any(v.is_neg() or _ic_torch.autograd.forward_ad.unpack_dual(v).tangent is not None for v in tensors)
            or _ic_torch.is_grad_enabled() and any(v.requires_grad for v in tensors)):
        return fallback(x,weight,bias)
    m,k=x.shape;n=weight.shape[0]
    output=_ic_torch.empty((m,n),device=x.device,dtype=x.dtype)
    with _ic_torch.cuda.device(x.device):
        _ic_linear_gelu_kernel[(_ic_triton.cdiv(m,__BM__),_ic_triton.cdiv(n,__BN__))](
            x,weight,bias,output,m,n,k,__BM__,__BN__,__BK__,num_warps=__WARPS__)
    return output
"""


def generate(source, *, schedule="compact"):
    if schedule not in SCHEDULES:
        raise ValueError("unknown fusion schedule")
    tree, function, names = _analyze(source)
    fallback = copy.deepcopy(function)
    fallback.name = "_ic_linear_gelu_fallback"
    entry = copy.deepcopy(function)
    entry.body = ast.parse(
        "return _ic_linear_gelu(" + ",".join(names) + ",_ic_linear_gelu_fallback)"
    ).body
    declarations = [v for v in tree.body if isinstance(v, (ast.Import, ast.ImportFrom))]
    module = ast.fix_missing_locations(
        ast.Module(body=[*declarations, fallback, entry], type_ignores=[])
    )
    backend = (
        """import torch as _ic_torch
try:
    import triton as _ic_triton
    import triton.language as _ic_tl
    from triton.language.extra.cuda import libdevice as _ic_math
except ImportError:
    _ic_triton = None

if _ic_triton is not None:
"""
        + textwrap.indent(_KERNEL, "    ")
        + "\n"
        + _WRAPPER
    )
    for key, value in zip(("BM", "BN", "BK", "WARPS"), SCHEDULES[schedule]):
        backend = backend.replace("__" + key + "__", str(value))
    candidate = backend + "\n" + ast.unparse(module) + "\n"
    ast.parse(candidate)
    return FusionCandidate(
        candidate,
        function.name,
        schedule,
        hashlib.sha256(source.encode()).hexdigest(),
        (
            "Fast path: ordinary contiguous FP16 CUDA rank2 input/weight and rank1 bias",
            "Original Torch composition handles CPU, other layouts/dtypes, gradients and unsupported domains",
            "Independent correctness and full-call timing against eager and compiled reference required",
        ),
    )
