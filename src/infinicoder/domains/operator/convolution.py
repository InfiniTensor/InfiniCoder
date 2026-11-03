"""Conservative single-convolution reference lowering;fresh verification required.

Forward FP32 only,on Ampere-or-newer CUDA. The generated module initializes
ordinary Parameters with PyTorch's initialization equations;all convolution
arithmetic runs in Triton. No reference code is executed during analysis.
"""

import ast, copy
from infinicoder.domains.operator.pointwise import Proposal


def propose(task):
    if "REFERENCE:\n" not in task or not ("FP32" in task or "float32" in task):
        return None
    instructions, source = task.split("REFERENCE:\n", 1)
    if any(
        word in instructions.lower()
        for word in ["backward", "autograd", "gradient", "training", "训练"]
    ):
        return None
    if len(source) > 65536:
        return None
    try:
        tree = ast.parse(source)
        if sum(1 for _ in ast.walk(tree)) > 3000:
            return None
        # Resolve only explicit official torch imports;reject shadowed aliases.
        modules = {}
        for n in tree.body:
            if isinstance(n, ast.Import):
                for a in n.names:
                    modules[a.asname or a.name] = a.name
            elif isinstance(n, ast.ImportFrom) and n.module == "torch":
                for a in n.names:
                    if a.name == "nn":
                        modules[a.asname or "nn"] = "torch.nn"

        def qualified(node):
            if isinstance(node, ast.Name):
                return modules.get(node.id, node.id)
            if isinstance(node, ast.Attribute):
                return qualified(node.value) + "." + node.attr
            return ""

        if not any(value == "torch" for value in modules.values()):
            return None
        if any(
            key in modules and modules[key] != value
            for key, value in [("torch", "torch"), ("nn", "torch.nn"), ("math", "math")]
        ):
            return None
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and n.id.startswith("_ic_"):
                return None
            if (
                isinstance(n, ast.Name)
                and isinstance(n.ctx, (ast.Store, ast.Del))
                and n.id in modules
            ):
                return None
            if (
                isinstance(n, ast.Attribute)
                and isinstance(n.ctx, (ast.Store, ast.Del))
                and qualified(n.value).split(".")[0] == "torch"
            ):
                return None
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Model")
        if (
            cls.decorator_list
            or len(cls.bases) != 1
            or qualified(cls.bases[0]) != "torch.nn.Module"
        ):
            return None
        methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
        if set(methods) != {"__init__", "forward"}:
            return None
        forward = methods["forward"]
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
        call = body[0].value
        if (
            not isinstance(call, ast.Call)
            or call.keywords
            or len(call.args) != 1
            or not isinstance(call.args[0], ast.Name)
            or call.args[0].id != args.args[1].arg
        ):
            return None
        if (
            not isinstance(call.func, ast.Attribute)
            or not isinstance(call.func.value, ast.Name)
            or call.func.value.id != args.args[0].arg
        ):
            return None
        field = call.func.attr
        constructors = []
        for n in methods["__init__"].body:
            if (
                isinstance(n, ast.Assign)
                and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Attribute)
                and ast.unparse(n.targets[0]) == "self." + field
                and isinstance(n.value, ast.Call)
            ):
                constructors.append(n.value)
        if len(constructors) != 1:
            return None
        constructor = constructors[0]
        name = qualified(constructor.func).removeprefix("torch.nn.")
        if (
            name
            not in [
                "Conv1d",
                "Conv2d",
                "Conv3d",
                "ConvTranspose1d",
                "ConvTranspose2d",
                "ConvTranspose3d",
            ]
            or qualified(constructor.func) != "torch.nn." + name
        ):
            return None
        # The initializer must only initialize its superclass and this one layer.
        statements = [
            n
            for n in methods["__init__"].body
            if not (
                isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Constant)
                and isinstance(n.value.value, str)
            )
        ]
        for n in statements:
            if isinstance(n, ast.Assign) and n.value is constructor:
                continue
            if (
                isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Call)
                and ast.unparse(n.value.func) in ["super().__init__", "super(Model, self).__init__"]
                and not n.value.args
                and not n.value.keywords
            ):
                continue
            return None
        for keyword in constructor.keywords:
            if keyword.arg not in [
                "in_channels",
                "out_channels",
                "kernel_size",
                "stride",
                "padding",
                "dilation",
                "groups",
                "bias",
                "output_padding",
                "padding_mode",
                "device",
                "dtype",
            ]:
                return None
            if (
                keyword.arg == "padding"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ):
                return None
        rank = int(name[-2])
        transposed = "Transpose" in name
        new = copy.deepcopy(cls)

        class Rewrite(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id == "Model":
                    node.id = "ModelNew"
                return node

            def visit_Call(self, node):
                node = self.generic_visit(node)
                if qualified(node.func) == "torch.nn." + name:
                    node.func = ast.Name("_ic_" + name, ast.Load())
                return node

            def visit_Expr(self, node):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    return None
                return self.generic_visit(node)

        new = Rewrite().visit(new)
        new.name = "ModelNew"
        f = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        f.body = ast.parse(
            "return _ic_forward(" + args.args[1].arg + ", " + args.args[0].arg + "." + field + ")"
        ).body
        imports = "\n".join(
            ast.unparse(n)
            for n in tree.body
            if (
                isinstance(n, ast.Import)
                and all(a.name in ["torch", "torch.nn", "math", "typing"] for a in n.names)
            )
            or (isinstance(n, ast.ImportFrom) and n.module in ["torch", "typing", "math"])
        )
        imports += "\nimport torch"
        if "torch.nn as nn" not in imports:
            imports += "\nimport torch.nn as nn"
        factory = _factory(name, rank, transposed)
        result = (
            imports
            + "\n"
            + _RUNTIME
            + "\n"
            + factory
            + "\n"
            + ast.unparse(ast.fix_missing_locations(new))
            + "\n"
        )
        return Proposal(
            result,
            "single " + name,
            (
                "single stateful convolution;exact parameter initialization",
                "FP32 forward only;no autograd",
                "CUDA compute capability>=8",
                "numeric nonnegative padding;positive spatial dimensions",
                "independent input/weight strides;groups,dilation,transpose output padding",
            ),
            provenance="static_single_convolution_lowering_v1",
        )
    except (
        SyntaxError,
        ValueError,
        TypeError,
        KeyError,
        StopIteration,
        AttributeError,
        IndexError,
    ):
        return None


def _factory(name, rank, transposed):
    if transposed:
        signature = 'in_channels,out_channels,kernel_size,stride=1,padding=0,output_padding=0,groups=1,bias=True,dilation=1,padding_mode="zeros",device=None,dtype=None'
    else:
        signature = 'in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True,padding_mode="zeros",device=None,dtype=None'
    return (
        "def _ic_"
        + name
        + "("
        + signature
        + "):\n    return _ic_parameters(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode,device,dtype,"
        + str(rank)
        + ","
        + str(transposed)
        + ","
        + ("output_padding" if transposed else "0")
        + ")\n"
    )


_RUNTIME = """
import math
import triton
import triton.language as tl

def _ic_tuple(value,rank):
    values=(value,)*rank if isinstance(value,int) else tuple(value)
    if len(values)!=rank or any(type(v) is not int for v in values):raise ValueError('numeric convolution metadata required')
    return values

def _ic_parameters(ci,co,kernel,stride,padding,dilation,groups,bias,padding_mode,device,dtype,rank,transposed,output_padding):
    if padding_mode!='zeros' or groups<=0 or ci<=0 or co<=0 or ci%groups or co%groups:raise ValueError('unsupported convolution contract')
    k=_ic_tuple(kernel,rank);s=_ic_tuple(stride,rank);p=_ic_tuple(padding,rank);d=_ic_tuple(dilation,rank);op=_ic_tuple(output_padding,rank)
    if any(v<=0 for v in k+s+d) or any(v<0 for v in p+op):raise ValueError('invalid convolution metadata')
    if transposed and any(o>=st and o>=di for o,st,di in zip(op,s,d)):raise ValueError('invalid output padding')
    module=nn.Module();module.in_channels=ci;module.out_channels=co;module.groups=groups;module.kernel_size=k;module.stride=s;module.padding=p;module.dilation=d;module.output_padding=op;module.transposed=transposed
    shape=(ci,co//groups,*k) if transposed else (co,ci//groups,*k)
    module.weight=nn.Parameter(torch.empty(shape,device=device,dtype=dtype))
    nn.init.kaiming_uniform_(module.weight,a=math.sqrt(5))
    if bias:
        module.bias=nn.Parameter(torch.empty(co,device=device,dtype=dtype))
        bound=1/math.sqrt(shape[1]*math.prod(k));nn.init.uniform_(module.bias,-bound,bound)
    else:module.register_parameter('bias',None)
    return module

@triton.jit
def _ic_conv(X,W,Bias,Y,B:tl.constexpr,CI:tl.constexpr,CO:tl.constexpr,CG:tl.constexpr,OG:tl.constexpr,
    ID:tl.constexpr,IH:tl.constexpr,IW:tl.constexpr,OD:tl.constexpr,OH:tl.constexpr,OW:tl.constexpr,
    KD:tl.constexpr,KH:tl.constexpr,KW:tl.constexpr,SD:tl.constexpr,SH:tl.constexpr,SW:tl.constexpr,
    PD:tl.constexpr,PH:tl.constexpr,PW:tl.constexpr,DD:tl.constexpr,DH:tl.constexpr,DW:tl.constexpr,
    X0:tl.constexpr,X1:tl.constexpr,X2:tl.constexpr,X3:tl.constexpr,X4:tl.constexpr,
    W0:tl.constexpr,W1:tl.constexpr,W2:tl.constexpr,W3:tl.constexpr,W4:tl.constexpr,
    B0:tl.constexpr,HAS_BIAS:tl.constexpr,TRANSPOSED:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    m=(tl.program_id(0)*BM+tl.arange(0,BM)).to(tl.int64);n=tl.program_id(1)*BN+tl.arange(0,BN);group=tl.program_id(2)
    volume=OD*OH*OW;batch=m//volume;spatial=m%volume;oz=spatial//(OH*OW);oy=spatial//OW%OH;ox=spatial%OW
    acc=tl.full((BM,BN),0,tl.float32)
    for part in range(tl.cdiv(CG*KD*KH*KW,BK)):
        r=part*BK+tl.arange(0,BK);ic=r//(KD*KH*KW);z=r//(KH*KW)%KD;y=r//KW%KH;x=r%KW
        if TRANSPOSED:
            iz=oz[:,None]+PD-z[None,:]*DD;iy=oy[:,None]+PH-y[None,:]*DH;ix=ox[:,None]+PW-x[None,:]*DW
            aligned=(iz%SD==0)&(iy%SH==0)&(ix%SW==0);iz=iz//SD;iy=iy//SH;ix=ix//SW
        else:
            iz=oz[:,None]*SD-PD+z[None,:]*DD;iy=oy[:,None]*SH-PH+y[None,:]*DH;ix=ox[:,None]*SW-PW+x[None,:]*DW;aligned=tl.full((BM,BK),True,tl.int1)
        xo=batch[:,None]*X0+(group*CG+ic[None,:])*X1+iz*X2+iy*X3+ix*X4
        a=tl.load(X+xo,(batch[:,None]<B)&(ic[None,:]<CG)&(iz>=0)&(iz<ID)&(iy>=0)&(iy<IH)&(ix>=0)&(ix<IW)&aligned,0)
        if TRANSPOSED:wo=(group*CG+ic[:,None])*W0+n[None,:]*W1+z[:,None]*W2+y[:,None]*W3+x[:,None]*W4
        else:wo=(group*OG+n[None,:])*W0+ic[:,None]*W1+z[:,None]*W2+y[:,None]*W3+x[:,None]*W4
        w=tl.load(W+wo,(ic[:,None]<CG)&(n[None,:]<OG),0)
        acc+=tl.dot(a,w,input_precision='tf32x3')
    if HAS_BIAS:acc+=tl.load(Bias+(group*OG+n)*B0,n<OG,0)[None,:]
    out=batch[:,None]*CO*volume+(group*OG+n[None,:])*volume+spatial[:,None]
    tl.store(Y+out,acc,(batch[:,None]<B)&(n[None,:]<OG))

@triton.jit
def _ic_depthwise(X,W,Bias,Y,B:tl.constexpr,CI:tl.constexpr,CO:tl.constexpr,OG:tl.constexpr,
    ID:tl.constexpr,IH:tl.constexpr,IW:tl.constexpr,OD:tl.constexpr,OH:tl.constexpr,OW:tl.constexpr,
    KD:tl.constexpr,KH:tl.constexpr,KW:tl.constexpr,SD:tl.constexpr,SH:tl.constexpr,SW:tl.constexpr,
    PD:tl.constexpr,PH:tl.constexpr,PW:tl.constexpr,DD:tl.constexpr,DH:tl.constexpr,DW:tl.constexpr,
    X0:tl.constexpr,X1:tl.constexpr,X2:tl.constexpr,X3:tl.constexpr,X4:tl.constexpr,
    W0:tl.constexpr,W1:tl.constexpr,W2:tl.constexpr,W3:tl.constexpr,W4:tl.constexpr,
    B0:tl.constexpr,HAS_BIAS:tl.constexpr,TRANSPOSED:tl.constexpr,BLOCK:tl.constexpr):
    i=(tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)).to(tl.int64);volume=OD*OH*OW;batch=i//(CO*volume);co=i//volume%CO;group=co//OG
    spatial=i%volume;oz=spatial//(OH*OW);oy=spatial//OW%OH;ox=spatial%OW;acc=tl.full((BLOCK,),0,tl.float32)
    for k in range(KD*KH*KW):
        z=k//(KH*KW);y=k//KW%KH;x=k%KW
        if TRANSPOSED:
            iz=oz+PD-z*DD;iy=oy+PH-y*DH;ix=ox+PW-x*DW;aligned=(iz%SD==0)&(iy%SH==0)&(ix%SW==0);iz=iz//SD;iy=iy//SH;ix=ix//SW
        else:iz=oz*SD-PD+z*DD;iy=oy*SH-PH+y*DH;ix=ox*SW-PW+x*DW;aligned=tl.full((BLOCK,),True,tl.int1)
        a=tl.load(X+batch*X0+group*X1+iz*X2+iy*X3+ix*X4,(batch<B)&(iz>=0)&(iz<ID)&(iy>=0)&(iy<IH)&(ix>=0)&(ix<IW)&aligned,0)
        if TRANSPOSED:wo=group*W0+(co%OG)*W1+z*W2+y*W3+x*W4
        else:wo=co*W0+z*W2+y*W3+x*W4
        w=tl.load(W+wo,batch<B,0);acc=acc+a*w
    if HAS_BIAS:acc+=tl.load(Bias+co*B0,batch<B,0)
    tl.store(Y+i,acc,i<B*CO*volume)

def _ic_forward(x,module):
    rank=len(module.kernel_size)
    if x.ndim!=rank+2 or x.shape[1]!=module.in_channels or x.dtype!=torch.float32 or module.weight.dtype!=torch.float32 or not x.is_cuda:raise ValueError('FP32 CUDA batched convolution required')
    if x.device!=module.weight.device or (module.bias is not None and module.bias.device!=x.device):raise ValueError('convolution parameters/device mismatch')
    if module.bias is not None and module.bias.dtype!=torch.float32:raise ValueError('FP32 bias required')
    if torch.is_grad_enabled() and x.requires_grad:raise ValueError('this seed implements forward only;autograd integration required')
    if torch.cuda.get_device_capability(x.device)[0]<8:raise ValueError('Ampere-or-newer CUDA required')
    dims=(1,)*(3-rank)+tuple(x.shape[2:]);kernel=(1,)*(3-rank)+module.kernel_size;stride=(1,)*(3-rank)+module.stride;padding=(0,)*(3-rank)+module.padding;dilation=(1,)*(3-rank)+module.dilation;op=(0,)*(3-rank)+module.output_padding
    if any(v<=0 for v in dims):raise ValueError('positive input spatial dimensions required')
    if module.transposed:out=tuple((v-1)*s-2*p+d*(k-1)+o+1 for v,s,p,d,k,o in zip(dims,stride,padding,dilation,kernel,op))
    else:out=tuple((v+2*p-d*(k-1)-1)//s+1 for v,s,p,d,k in zip(dims,stride,padding,dilation,kernel))
    if any(v<=0 for v in out):raise ValueError('invalid convolution output shape')
    result=torch.empty((x.shape[0],module.out_channels,*out[-rank:]),device=x.device,dtype=torch.float32)
    if result.numel()==0:return result
    xs=(x.stride(0),x.stride(1))+(0,)*(3-rank)+tuple(x.stride()[2:]);w=module.weight;ws=(w.stride(0),w.stride(1))+(0,)*(3-rank)+tuple(w.stride()[2:]);cg=module.in_channels//module.groups;og=module.out_channels//module.groups
    common=(x,w,module.bias if module.bias is not None else w,result,x.shape[0],module.in_channels,module.out_channels)
    metadata=(*dims,*out,*kernel,*stride,*padding,*dilation,*xs,*ws,module.bias.stride(0) if module.bias is not None else 0,module.bias is not None,module.transposed)
    with torch.cuda.device(x.device):
        if cg==1:_ic_depthwise[(triton.cdiv(result.numel(),256),)](*common,og,*metadata,BLOCK=256,num_warps=4)
        else:_ic_conv[(triton.cdiv(x.shape[0]*math.prod(out),16),triton.cdiv(og,32),module.groups)](*common,cg,og,*metadata,BM=16,BN=32,BK=32,num_warps=4)
    return result
"""
