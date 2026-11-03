"""Conservative pointwise AST lowering to a Triton candidate, never acceptance.

Only a declared FP32 single-tensor reference is eligible. Unsupported
syntax returns None. No reference code is executed during analysis.
"""

import ast
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Proposal:
    source: str
    expression: str
    preconditions: tuple
    provenance: str = "static_pointwise_lowering_v1"


def propose(task):
    if "REFERENCE:\n" not in task or not ("FP32" in task or "float32" in task):
        return None
    source = task.split("REFERENCE:\n", 1)[1]
    if len(source) > 65536:
        return None
    try:
        tree = ast.parse(source)
        if sum(1 for _ in ast.walk(tree)) > 2000:
            return None
        # Names used by lowering must retain their declared library meaning.
        if not any(
            isinstance(n, ast.Import)
            and any(a.name == "torch" and a.asname in (None, "torch") for a in n.names)
            for n in tree.body
        ):
            return None
        reserved = {"torch", "math", "F"}
        for n in ast.walk(tree):
            if (
                isinstance(n, ast.Name)
                and isinstance(n.ctx, (ast.Store, ast.Del))
                and n.id in reserved
            ):
                return None
            if (
                isinstance(n, ast.Attribute)
                and isinstance(n.ctx, (ast.Store, ast.Del))
                and ast.unparse(n).split(".")[0] in reserved
            ):
                return None
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                for alias in n.names:
                    binding = alias.asname or (
                        alias.name.split(".")[0] if isinstance(n, ast.Import) else alias.name
                    )
                    if binding not in reserved:
                        continue
                    valid = (
                        isinstance(n, ast.Import)
                        and (
                            alias.name in ("torch", "math")
                            and binding == alias.name
                            or alias.name == "torch.nn.functional"
                            and binding in ("torch", "F")
                        )
                        or isinstance(n, ast.ImportFrom)
                        and n.level == 0
                        and n.module == "torch.nn"
                        and alias.name == "functional"
                        and binding == "F"
                    )
                    if not valid:
                        return None
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Model")
        if cls.decorator_list:
            return None
        forward = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"
        )
        inputs = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_inputs"
        )
        # Establish contiguity from the generator, not from a guessed task name.
        if not isinstance(inputs.body[-1], ast.Return) or any(
            isinstance(n, ast.Return) for n in inputs.body[:-1]
        ):
            return None
        local = {}
        for statement in inputs.body[:-1]:
            if (
                not isinstance(statement, ast.Assign)
                or len(statement.targets) != 1
                or not isinstance(statement.targets[0], ast.Name)
            ):
                return None
            local[statement.targets[0].id] = statement.value
        generated = inputs.body[-1].value
        if not isinstance(generated, (ast.List, ast.Tuple)) or len(generated.elts) != 1:
            return None
        allocation = generated.elts[0]
        for _ in range(8):
            if not isinstance(allocation, ast.Name) or allocation.id not in local:
                break
            allocation = local[allocation.id]
        if not isinstance(allocation, ast.Call) or ast.unparse(allocation.func) not in (
            "torch.randn",
            "torch.rand",
            "torch.zeros",
            "torch.ones",
        ):
            return None
        if any(
            k.arg not in (None, "device", "dtype", "requires_grad") for k in allocation.keywords
        ):
            return None
        for k in allocation.keywords:
            if k.arg == "dtype" and ast.unparse(k.value) not in ("torch.float", "torch.float32"):
                return None
        args = forward.args
        if (
            len(args.args) != 2
            or args.posonlyargs
            or args.vararg
            or args.kwarg
            or args.kwonlyargs
            or args.defaults
            or forward.decorator_list
        ):
            return None
        if (
            not isinstance(forward.body[-1], ast.Return)
            or sum(isinstance(n, ast.Return) for n in forward.body) != 1
        ):
            return None
        x = args.args[1].arg
        if x in reserved:
            return None
        env = {x: "v"}
        lowering_calls = 0

        def lower(node):
            nonlocal lowering_calls
            lowering_calls += 1
            if lowering_calls > 4096:
                raise ValueError("lowering budget exceeded")
            value = lower_node(node)
            if len(value) > 8192:
                raise ValueError("expanded expression budget exceeded")
            return value

        def lower_node(node):
            if isinstance(node, ast.Name) and node.id in env:
                return env[node.id]
            if (
                isinstance(node, ast.Constant)
                and type(node.value) in (float, int)
                and math.isfinite(node.value)
            ):
                return repr(node.value)
            if isinstance(node, ast.Attribute) and ast.unparse(node) == "math.pi":
                return repr(math.pi)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
                return (
                    "("
                    + ("-" if isinstance(node.op, ast.USub) else "+")
                    + lower(node.operand)
                    + ")"
                )
            if isinstance(node, ast.BinOp):
                if isinstance(node.op, ast.Pow):
                    return power(node.left, node.right)
                op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}[type(node.op)]
                return "(" + lower(node.left) + op + lower(node.right) + ")"
            if isinstance(node, ast.Call) and not node.keywords:
                name = ast.unparse(node.func)
                if name in ("torch.pow", "torch.square"):
                    return power(
                        node.args[0], node.args[1] if name == "torch.pow" else ast.Constant(2)
                    )
                mapping = {
                    "torch.exp": "tl.exp",
                    "torch.log": "tl.log",
                    "torch.sqrt": "tl.sqrt",
                    "math.sqrt": "tl.sqrt",
                    "torch.tanh": "libdevice.tanh",
                    "torch.abs": "tl.abs",
                    "torch.sigmoid": "sigmoid",
                    "torch.relu": "relu",
                    "F.relu": "relu",
                    "torch.nn.functional.relu": "relu",
                }
                if name not in mapping or len(node.args) != 1:
                    raise ValueError("unsupported call")
                value = lower(node.args[0])
                operation = mapping[name]
                if operation == "relu":
                    return "tl.where((" + value + ") <= 0.0, 0.0, " + value + ")"
                if operation == "sigmoid":
                    return "(1.0/(1.0+tl.exp(-(" + value + "))))"
                return operation + "(" + value + ")"
            raise ValueError("unsupported expression")

        def power(base, exponent):
            if not isinstance(exponent, ast.Constant) or type(exponent.value) not in (int, float):
                raise ValueError("dynamic power")
            n = exponent.value
            if not float(n).is_integer() or not 0 <= n <= 8:
                raise ValueError("unsupported power")
            value = lower(base)
            return "(" + "*".join([value] * int(n)) + ")" if n else "1.0"

        expression = None
        for node in forward.body:
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                env[node.targets[0].id] = lower(node.value)
            elif isinstance(node, ast.Return):
                expression = lower(node.value)
            else:
                return None
        if expression is None or expression == "v":
            return None
        init = next(
            (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None
        )
        if init:
            # This first seed supports only state-free constructors.
            for node in init.body:
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                    continue
                if (
                    isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Call)
                    and not node.value.args
                    and not node.value.keywords
                    and ast.unparse(node.value.func)
                    in ("super().__init__", "super(Model, self).__init__")
                ):
                    continue
                return None
            if init.decorator_list:
                return None
            init_text = "def __init__(" + ast.unparse(init.args) + "):\n    super().__init__()"
        else:
            init_text = "def __init__(self):\n    super().__init__()"
        code = (
            """import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
@triton.jit
def _pointwise(X,Y,N:tl.constexpr,S:tl.constexpr,XS:tl.constexpr,YS:tl.constexpr,C:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0).to(tl.int64)*B+tl.arange(0,B)
 if C:
  xi=i
  yi=i
 else:
  remainder=i
  xi=tl.full((B,),0,tl.int64)
  yi=tl.full((B,),0,tl.int64)
  for axis in tl.static_range(len(S)-1,-1,-1):
   coordinate=remainder % S.value[axis]
   remainder=remainder // S.value[axis]
   xi+=coordinate*XS.value[axis]
   yi+=coordinate*YS.value[axis]
 v=tl.load(X+xi,i<N,other=0)
 result=EXPRESSION
 tl.store(Y+yi,result,i<N)
class ModelNew(torch.nn.Module):
INIT
 def forward(self,INPUT):
  x=INPUT
  if x.dtype!=torch.float32 or not x.is_cuda:
   raise ValueError('pointwise seed requires a CUDA float32 tensor')
  out=torch.empty_like(x)
  if x.numel():_pointwise[(triton.cdiv(x.numel(),256),)](x,out,x.numel(),tuple(x.shape),tuple(x.stride()),tuple(out.stride()),x.is_contiguous() and out.is_contiguous(),256,enable_fp_fusion=False)
  return out
""".replace("EXPRESSION", expression)
            .replace("INIT", "\n".join(" " + line for line in init_text.splitlines()))
            .replace("INPUT", x)
        )
        ast.parse(code)
        return Proposal(
            code,
            expression,
            (
                "FP32",
                "CUDA",
                "one tensor input",
                "state-free Model constructor",
                "pointwise-only forward;independent input/output strides",
            ),
        )
    except (
        SyntaxError,
        ValueError,
        KeyError,
        StopIteration,
        TypeError,
        AttributeError,
        RecursionError,
        OverflowError,
    ):
        return None
