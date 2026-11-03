"""Stateless 2D FP32 matmul lowering with explicit runtime preconditions.

This proposes code;independent correctness/performance checks remain mandatory.
No task ID or expected output is used. Unsupported ASTs return None.
"""

import ast
import copy
from infinicoder.domains.operator.pointwise import Proposal


def propose(task):
    if "REFERENCE:\n" not in task or not ("FP32" in task or "float32" in task):
        return None
    source = task.split("REFERENCE:\n", 1)[1]
    if len(source) > 65536:
        return None
    try:
        tree = ast.parse(source)
        if sum(1 for _ in ast.walk(tree)) > 3000:
            return None
        if not any(
            isinstance(n, ast.Import)
            and any(a.name == "torch" and a.asname in (None, "torch") for a in n.names)
            for n in tree.body
        ):
            return None
        reserved = {"torch", "nn", "super"}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, (ast.Store, ast.Del))
                and node.id in reserved
            ):
                return None
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in reserved:
                return None
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.ctx, (ast.Store, ast.Del))
                and ast.unparse(node).split(".")[0] in reserved
            ):
                return None
            if isinstance(node, ast.Import):
                for item in node.names:
                    bound = item.asname or item.name.split(".")[0]
                    if (
                        bound == "torch"
                        and item.name not in ("torch", "torch.nn")
                        or bound == "nn"
                        and item.name != "torch.nn"
                        or bound == "super"
                    ):
                        return None
            if isinstance(node, ast.ImportFrom):
                for item in node.names:
                    bound = item.asname or item.name
                    if bound in reserved and not (
                        node.module == "torch" and item.name == "nn" and bound == "nn"
                    ):
                        return None
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Model")
        if (
            cls.decorator_list
            or cls.keywords
            or len(cls.bases) != 1
            or ast.unparse(cls.bases[0]) not in ("nn.Module", "torch.nn.Module")
        ):
            return None
        statements = [
            n
            for n in cls.body
            if not isinstance(n, ast.Pass)
            and not (
                isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Constant)
                and isinstance(n.value.value, str)
            )
        ]
        if any(
            not isinstance(n, ast.FunctionDef)
            or n.name not in ("forward", "__init__")
            or n.decorator_list
            for n in statements
        ):
            return None
        methods = {n.name: n for n in statements}
        if len(methods) != len(statements):
            return None
        forward = methods["forward"]
        args = forward.args
        if (
            len(args.args) != 3
            or args.posonlyargs
            or args.kwonlyargs
            or args.vararg
            or args.kwarg
            or args.defaults
        ):
            return None
        if any(a.arg in reserved for a in args.args):
            return None
        if "__init__" in methods:
            init = methods["__init__"]
            a = init.args
            if (
                len(a.args) != 1
                or a.posonlyargs
                or a.kwonlyargs
                or a.vararg
                or a.kwarg
                or a.defaults
            ):
                return None
            body = [
                n
                for n in init.body
                if not (
                    isinstance(n, ast.Expr)
                    and isinstance(n.value, ast.Constant)
                    and isinstance(n.value.value, str)
                )
            ]
            if (
                len(body) != 1
                or not isinstance(body[0], ast.Expr)
                or not isinstance(body[0].value, ast.Call)
            ):
                return None
            call = body[0].value
            if (
                ast.unparse(call.func) not in ("super().__init__", "super(Model, self).__init__")
                or call.args
                or call.keywords
            ):
                return None
        body = [
            n
            for n in forward.body
            if not (
                isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Constant)
                and isinstance(n.value.value, str)
            )
        ]
        if len(body) != 1 or not isinstance(body[0], ast.Return):
            return None
        expression = body[0].value
        if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.MatMult):
            left, right = expression.left, expression.right
        elif (
            isinstance(expression, ast.Call)
            and ast.unparse(expression.func) in ("torch.matmul", "torch.mm")
            and len(expression.args) == 2
            and not expression.keywords
        ):
            left, right = expression.args
        else:
            return None
        inputs = {a.arg for a in args.args[1:]}

        def operand(node):
            if isinstance(node, ast.Name) and node.id in inputs:
                return node.id, False
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "T"
                and isinstance(node.value, ast.Name)
                and node.value.id in inputs
            ):
                return node.value.id, True
            if (
                isinstance(node, ast.Call)
                and not node.keywords
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in inputs
            ):
                if node.func.attr == "t" and not node.args:
                    return node.func.value.id, True
                if node.func.attr == "transpose" and len(node.args) == 2:
                    axes = tuple(ast.literal_eval(v) for v in node.args)
                    if axes in [(0, 1), (1, 0), (-1, -2), (-2, -1)]:
                        return node.func.value.id, True
            raise ValueError("unsupported operand")

        left, ta = operand(left)
        right, tb = operand(right)
        candidate = copy.deepcopy(cls)
        candidate.name = "ModelNew"

        class Names(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id == "Model":
                    node.id = "ModelNew"
                return node

        candidate = Names().visit(candidate)
        f = next(
            n for n in candidate.body if isinstance(n, ast.FunctionDef) and n.name == "forward"
        )
        f.body = ast.parse(
            "return _ic_matmul(" + left + ", " + right + ", " + repr(ta) + ", " + repr(tb) + ")"
        ).body
        code = _RUNTIME + "\n" + ast.unparse(ast.fix_missing_locations(candidate)) + "\n"
        return Proposal(
            code,
            "FP32 2D matmul;transpose_left=" + str(ta) + ";transpose_right=" + str(tb),
            (
                "stateless two-input matrix product",
                "CUDA FP32;2D;compatible inner dimensions",
                "each operand has independent strides/storage offset",
                "forward only;no input mutation;no autograd",
                "IEEE dot;no TF32 precision relaxation",
            ),
            provenance="static_matrix_product_lowering_v1",
        )
    except (SyntaxError, ValueError, TypeError, KeyError, StopIteration, AttributeError):
        return None


_RUNTIME = """from __future__ import annotations
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _ic_product(A,B,C,M,N,K,SAM,SAK,SBK,SBN,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    rows=tl.program_id(0).to(tl.int64)*BM+tl.arange(0,BM).to(tl.int64)
    cols=tl.program_id(1).to(tl.int64)*BN+tl.arange(0,BN).to(tl.int64)
    lanes=tl.arange(0,BK).to(tl.int64)
    total=tl.full((BM,BN),0,tl.float32)
    for block in range(tl.cdiv(K,BK)):
        inner=block.to(tl.int64)*BK+lanes
        lhs=tl.load(A+rows[:,None]*SAM+inner[None,:]*SAK,(rows[:,None]<M)&(inner[None,:]<K),0)
        rhs=tl.load(B+inner[:,None]*SBK+cols[None,:]*SBN,(inner[:,None]<K)&(cols[None,:]<N),0)
        total=tl.dot(lhs,rhs,total,input_precision='ieee')
    tl.store(C+rows[:,None]*N+cols[None,:],total,(rows[:,None]<M)&(cols[None,:]<N))

def _ic_matmul(left,right,transpose_left,transpose_right):
    if not isinstance(left,torch.Tensor) or not isinstance(right,torch.Tensor):raise NotImplementedError('tensor inputs required')
    if left.ndim!=2 or right.ndim!=2 or left.dtype!=torch.float32 or right.dtype!=torch.float32:raise NotImplementedError('FP32 2D only')
    if not left.is_cuda or not right.is_cuda or left.device!=right.device:raise NotImplementedError('same CUDA device required')
    if left.requires_grad or right.requires_grad:raise NotImplementedError('autograd unsupported')
    m,k=left.shape[::-1] if transpose_left else left.shape
    inner,n=right.shape[::-1] if transpose_right else right.shape
    if k!=inner:raise ValueError('inner dimensions disagree')
    am,ak=left.stride()[::-1] if transpose_left else left.stride()
    bk,bn=right.stride()[::-1] if transpose_right else right.stride()
    result=torch.empty((m,n),device=left.device,dtype=torch.float32)
    if m and n:
        with torch.cuda.device(left.device):
            _ic_product[(triton.cdiv(m,16),triton.cdiv(n,32))](left,right,result,m,n,k,am,ak,bk,bn,16,32,32,num_warps=4)
    return result
"""
