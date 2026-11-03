"""Explicit public-source contract for the existing convolution lowerer.

This module never executes reference or candidate code. Applicability describes
the declared forward domain;only the caller's independent verifier proves it.
"""

from infinicoder._resources import package_root

import ast
from dataclasses import asdict, dataclass
import hashlib
from pathlib import PurePosixPath
import re
import time

from infinicoder.domains.operator.candidate_provider import (
    ProviderLeaseError,
    UnknownPrerequisite,
    _public_source,
)


def digest(raw):
    return hashlib.sha256(raw if isinstance(raw, bytes) else raw.encode("utf8")).hexdigest()


@dataclass(frozen=True)
class ConvolutionProviderContract:
    target: str
    public_reference_sha256: str
    scope_quote: str
    complete_module_authorized: bool
    final_reserve_seconds: float
    provider: str = "convolution_v1"
    mode: str = "enabled"
    entry: str = "ModelNew"
    backend: str = "triton"
    dtype: str = "float32"
    device: str = "cuda"
    public_reference_span: tuple | None = None

    def __post_init__(self):
        import math

        p = PurePosixPath(self.target)
        if (
            not isinstance(self.target, str)
            or not self.target.endswith(".py")
            or p.is_absolute()
            or str(p) != self.target
            or ".." in p.parts
            or any(c in self.target for c in "*?[\\:\0")
        ):
            raise ValueError("one normalized relative Python target required")
        if not re.fullmatch("[0-9a-f]{64}", self.public_reference_sha256):
            raise ValueError("exact public source SHA required")
        if (
            not isinstance(self.scope_quote, str)
            or self.target not in self.scope_quote
            or len(self.scope_quote) > 2048
        ):
            raise ValueError("an exact public target scope quote is required")
        if self.complete_module_authorized is not True:
            raise ValueError("caller must explicitly authorize complete-module replacement")
        if (
            isinstance(self.final_reserve_seconds, bool)
            or not isinstance(self.final_reserve_seconds, (int, float))
            or not math.isfinite(self.final_reserve_seconds)
            or self.final_reserve_seconds <= 0
        ):
            raise ValueError("caller must reserve a finite positive final-check interval")
        if self.provider != "convolution_v1" or self.mode not in ("enabled", "disabled"):
            raise ValueError("unsupported convolution provider or mode")
        if (self.entry, self.backend, self.dtype, self.device) != (
            "ModelNew",
            "triton",
            "float32",
            "cuda",
        ):
            raise ValueError("only declared FP32 CUDA Triton forward modules are supported")
        span = self.public_reference_span
        if span is not None:
            if (
                not isinstance(span, (tuple, list))
                or len(span) != 2
                or any(type(v) is not int for v in span)
                or not 0 <= span[0] < span[1]
            ):
                raise ValueError("explicit UTF-8 public reference byte interval required")
            object.__setattr__(self, "public_reference_span", tuple(span))


def identity():
    root = package_root()
    return {
        name: digest((root / name).read_bytes())
        for name in (
            "domains/operator/convolution_provider.py",
            "domains/operator/convolution.py",
            "domains/operator/pointwise.py",
        )
    }


def _source_and_lowerer_task(spec, contract):
    """A caller byte lease avoids guessing where a multi-section packet ends.

    Only the compiler's source carrier is assembled. The actual TaskSpec,SDK
    input,scope and signature are never rewritten or stripped of their tail.
    """
    if contract.public_reference_span is None:
        return _public_source(spec, contract), spec.task
    marker = "REFERENCE:\n"
    raw = spec.task.encode("utf8")
    start, end = contract.public_reference_span
    if spec.task.count(marker) != 1 or end > len(raw):
        raise ValueError("public reference interval is outside the exact task")
    prefix = spec.task.split(marker, 1)[0].encode("utf8")
    if start != len(prefix) + len(marker.encode()):
        raise ValueError(
            "public reference interval must start immediately after the original marker"
        )
    source = raw[start:end].decode("utf8")
    instructions = (prefix + raw[end:]).decode("utf8")
    if len(raw[start:end]) > 65536 or digest(source) != contract.public_reference_sha256:
        raise ValueError("public reference interval hash/size mismatch")
    if not re.search(r"\btriton\b", instructions, re.I) or not re.search(
        r"\b(?:FP32|float32)\b", instructions
    ):
        raise UnknownPrerequisite("explicit Triton and FP32 requirements missing")
    if re.search(r"\b(backward|autograd|gradient|training)\b|反向|梯度|训练", instructions, re.I):
        raise ValueError("gradient/training semantics unsupported")
    if spec.amendments:
        raise ValueError("amendments require a new explicit reference contract")
    return source, instructions + "\n" + marker + source


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


def _literal(node, env):
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        values = [_literal(n, env) for n in node.elts]
        return tuple(values) if isinstance(node, ast.Tuple) else values
    if isinstance(node, ast.Constant) and type(node.value) in (int, float, bool, str, type(None)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _literal(node.operand, env)
        if type(value) in (int, float):
            return -value if isinstance(node.op, ast.USub) else value
    raise UnknownPrerequisite("nonliteral convolution/input metadata")


def _annotation(node, aliases):
    if node is None:
        return
    name = ast.unparse(node)
    if name in ("int", "float", "bool", "tuple", "None"):
        return
    if any(name == key + ".Tensor" and value == "torch" for key, value in aliases.items()):
        return
    raise UnknownPrerequisite("annotation may execute unknown code")


def reference_facts(source):
    """Resolve literal public factories without importing them or inferring hidden cases."""
    tree = ast.parse(source)
    if sum(1 for _ in ast.walk(tree)) > 3000:
        raise ValueError("public AST bound exceeded")
    aliases = {}
    globals = {}
    functions = {}
    classes = []
    declared = set()

    def claim(name):
        if name in declared:
            raise ValueError("duplicate/shadowed public binding")
        declared.add(name)

    for n in tree.body:
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name not in ("torch", "torch.nn"):
                    raise UnknownPrerequisite("unsupported reference import")
                name = a.asname or a.name.split(".")[0]
                # A normal import torch;import torch.nn is the same root binding.
                value = a.name if a.asname else a.name.split(".")[0]
                if name in aliases and aliases[name] == value:
                    continue
                claim(name)
                aliases[name] = value
        elif isinstance(n, ast.ImportFrom) and n.module == "torch" and n.level == 0:
            for a in n.names:
                if a.name != "nn":
                    raise UnknownPrerequisite("unsupported from import")
                name = a.asname or a.name
                claim(name)
                aliases[name] = "torch.nn"
        elif (
            isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
        ):
            name = n.targets[0].id
            claim(name)
            globals[name] = _literal(n.value, globals)
        elif isinstance(n, ast.FunctionDef):
            claim(n.name)
            if (
                n.name not in ("get_inputs", "get_init_inputs")
                or n.decorator_list
                or n.args.args
                or n.args.posonlyargs
                or n.args.kwonlyargs
                or n.args.vararg
                or n.args.kwarg
            ):
                raise UnknownPrerequisite("unknown public input factory")
            _annotation(n.returns, aliases)
            functions[n.name] = n
        elif isinstance(n, ast.ClassDef):
            claim(n.name)
            classes.append(n)
        elif (
            isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Constant)
            and isinstance(n.value.value, str)
        ):
            pass
        else:
            raise UnknownPrerequisite("unknown module effect")
    if (
        set(functions) != {"get_inputs", "get_init_inputs"}
        or len(classes) != 1
        or classes[0].name != "Model"
    ):
        raise UnknownPrerequisite("one Model and the two public factories are required")
    cls = classes[0]
    if cls.keywords or cls.decorator_list:
        raise UnknownPrerequisite("unknown class construction")
    methods = _body(cls)
    if (
        len(methods) != 2
        or any(not isinstance(n, ast.FunctionDef) for n in methods)
        or {n.name for n in methods} != {"__init__", "forward"}
    ):
        raise UnknownPrerequisite("unknown model state or descriptors")
    methods = {n.name: n for n in methods}
    init = methods["__init__"]
    args = init.args
    for method in methods.values():
        if method.decorator_list:
            raise UnknownPrerequisite("decorated method")
        _annotation(method.returns, aliases)
        for a in (*method.args.posonlyargs, *method.args.args, *method.args.kwonlyargs):
            _annotation(a.annotation, aliases)
    if args.posonlyargs or args.vararg or args.kwarg or args.kwonlyargs:
        raise UnknownPrerequisite("variadic constructor")
    names = [a.arg for a in args.args]
    if (
        not names
        or names[0] != "self"
        or len(set(names)) != len(names)
        or set(names) & set(aliases)
    ):
        raise ValueError("shadowed constructor binding")
    returns = _body(functions["get_init_inputs"])
    if len(returns) != 1 or not isinstance(returns[0], ast.Return):
        raise UnknownPrerequisite("unknown init arguments")
    values = _literal(returns[0].value, globals)
    if not isinstance(values, (list, tuple)) or len(values) > len(names) - 1:
        raise ValueError("invalid init argument list")
    env = (
        {
            name: _literal(default, {})
            for name, default in zip(names[-len(args.defaults) :], args.defaults)
        }
        if args.defaults
        else {}
    )
    env.update(zip(names[1:], values))
    if set(env) != set(names[1:]):
        raise UnknownPrerequisite("unbound constructor input")
    statements = _body(init)
    assignments = [n for n in statements if isinstance(n, ast.Assign)]
    if len(statements) != 2 or len(assignments) != 1:
        raise UnknownPrerequisite("initializer has additional effects")
    assignment = assignments[0]
    call = assignment.value
    if (
        len(assignment.targets) != 1
        or not isinstance(assignment.targets[0], ast.Attribute)
        or ast.unparse(assignment.targets[0].value) != "self"
        or not isinstance(call, ast.Call)
    ):
        raise UnknownPrerequisite("one direct layer field required")
    qualified = ast.unparse(call.func)
    prefix, _, kind = qualified.rpartition(".")
    resolved = aliases.get(prefix, prefix)
    if resolved != "torch.nn" or kind not in (
        "Conv1d",
        "Conv2d",
        "Conv3d",
        "ConvTranspose1d",
        "ConvTranspose2d",
        "ConvTranspose3d",
    ):
        raise UnknownPrerequisite("not an existing supported convolution structure")
    rank = int(kind[-2])
    transposed = "Transpose" in kind
    parameters = (
        ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
        + (
            ["output_padding", "groups", "bias", "dilation"]
            if transposed
            else ["dilation", "groups", "bias"]
        )
        + ["padding_mode", "device", "dtype"]
    )
    config = dict(
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        output_padding=0,
    )
    if len(call.args) > len(parameters):
        raise ValueError("too many convolution arguments")
    config.update((key, _literal(value, env)) for key, value in zip(parameters, call.args))
    seen = set(parameters[: len(call.args)])
    for kw in call.keywords:
        if kw.arg not in parameters or kw.arg in seen:
            raise ValueError("unknown/duplicate convolution argument")
        seen.add(kw.arg)
        config[kw.arg] = _literal(kw.value, env)
    ci, co, groups = (config[key] for key in ("in_channels", "out_channels", "groups"))
    if any(type(v) is not int or v <= 0 for v in (ci, co, groups)) or ci % groups or co % groups:
        raise ValueError("invalid channel/group contract")
    if (
        type(config["bias"]) is not bool
        or config["padding_mode"] != "zeros"
        or config["device"] is not None
        or config["dtype"] is not None
    ):
        raise UnknownPrerequisite("unproven device/dtype/padding initializer")
    for key in ("kernel_size", "stride", "padding", "dilation", "output_padding"):
        value = config[key]
        value = (value,) * rank if type(value) is int else value
        if (
            not isinstance(value, tuple)
            or len(value) != rank
            or any(
                type(v) is not int or v < (0 if key in ("padding", "output_padding") else 1)
                for v in value
            )
        ):
            raise ValueError("invalid numeric convolution metadata")
        config[key] = value
    if transposed and any(
        o >= s and o >= d
        for o, s, d in zip(config["output_padding"], config["stride"], config["dilation"])
    ):
        raise ValueError("invalid output padding")
    input_body = _body(functions["get_inputs"])
    if (
        len(input_body) == 2
        and isinstance(input_body[0], ast.Assign)
        and len(input_body[0].targets) == 1
        and isinstance(input_body[0].targets[0], ast.Name)
    ):
        create, ret = input_body
        if create.targets[0].id in declared:
            raise UnknownPrerequisite("input local shadows a public binding")
        if (
            not isinstance(ret, ast.Return)
            or not isinstance(ret.value, ast.List)
            or len(ret.value.elts) != 1
            or ast.unparse(ret.value.elts[0]) != create.targets[0].id
        ):
            raise UnknownPrerequisite("unproven input return")
        create = create.value
    elif (
        len(input_body) == 1
        and isinstance(input_body[0], ast.Return)
        and isinstance(input_body[0].value, ast.List)
        and len(input_body[0].value.elts) == 1
    ):
        create = input_body[0].value.elts[0]
    else:
        raise UnknownPrerequisite("unproven input factory effects")
    if (
        not isinstance(create, ast.Call)
        or create.keywords
        or ast.unparse(create.func)
        not in {
            key + "." + op
            for key, value in aliases.items()
            if value == "torch"
            for op in ("rand", "randn", "empty")
        }
    ):
        raise UnknownPrerequisite("unproven public tensor factory")
    shape = [_literal(n, globals) for n in create.args]
    if len(shape) != rank + 2 or any(type(v) is not int or v <= 0 for v in shape) or shape[1] != ci:
        raise ValueError("invalid public tensor shape")
    if transposed:
        out = [
            (v - 1) * s - 2 * p + d * (k - 1) + o + 1
            for v, k, s, p, d, o in zip(
                shape[2:],
                config["kernel_size"],
                config["stride"],
                config["padding"],
                config["dilation"],
                config["output_padding"],
            )
        ]
    else:
        out = [
            (v + 2 * p - d * (k - 1) - 1) // s + 1
            for v, k, s, p, d in zip(
                shape[2:],
                config["kernel_size"],
                config["stride"],
                config["padding"],
                config["dilation"],
            )
        ]
    if any(v <= 0 for v in out):
        raise ValueError("invalid public output shape")
    return dict(
        kind=kind,
        layer=ast.unparse(assignment.targets[0]),
        configuration=config,
        public_input_shape=shape,
        public_output_shape=[shape[0], co, *out],
        public_factory_only=True,
        hidden_input_domain="not inferred;original independent final required",
    )


def assess(spec, workspace, contract):
    result = dict(provider=contract.provider, status="Unknown", acceptance_proof=False)
    try:
        if contract.mode != "enabled":
            raise UnknownPrerequisite("provider disabled")
        instructions = spec.task.split("REFERENCE:\n", 1)[0]
        if contract.scope_quote not in instructions:
            raise ValueError("complete-module scope quote not in original public task")
        source, lowerer_task = _source_and_lowerer_task(spec, contract)
        facts = reference_facts(source)
        path = workspace.path(contract.target, write=True)
        if path.exists() and (not path.is_file() or path.stat().st_size > 1_000_000):
            raise UnknownPrerequisite("bounded regular target required")
        original = path.read_bytes() if path.exists() else None
        if original is not None and (not original or len(original) > 1_000_000):
            raise UnknownPrerequisite("empty/oversized existing target")
        from infinicoder.domains.operator.convolution import propose

        start = time.monotonic()
        candidate = propose(lowerer_task)
        if time.monotonic() - start > 2:
            raise UnknownPrerequisite("convolution generation CPU bound exceeded")
        if candidate is None:
            raise UnknownPrerequisite(
                "unchanged convolution lowerer does not support this reference"
            )
        if len(candidate.source.encode()) > 1_000_000:
            raise ValueError("generated source size bound")
        ast.parse(candidate.source)
        from infinicoder.memory.repair_memory import signature

        result.update(
            status="Applicable",
            requirements=signature(spec),
            contract=asdict(contract),
            implementation=identity(),
            root=str(workspace.root),
            target_before=original,
            target_before_sha256=digest(original) if original is not None else None,
            candidate_source=candidate.source,
            candidate_sha256=digest(candidate.source),
            facts=facts,
            preconditions=list(candidate.preconditions),
            provenance=candidate.provenance,
        )
    except (
        ValueError,
        TypeError,
        SyntaxError,
        RecursionError,
        OverflowError,
        UnicodeError,
        OSError,
    ) as error:
        result.update(
            status="Unknown" if isinstance(error, UnknownPrerequisite) else "Unsupported",
            reason=type(error).__name__ + ": " + str(error),
        )
    return result


def validate_lease(spec, workspace, contract, admission):
    from infinicoder.memory.repair_memory import signature

    if (
        admission.get("status") != "Applicable"
        or admission["requirements"] != signature(spec)
        or admission["contract"] != asdict(contract)
        or admission["implementation"] != identity()
        or admission["root"] != str(workspace.root)
        or digest(admission["candidate_source"]) != admission["candidate_sha256"]
    ):
        raise ProviderLeaseError("convolution implementation/source lease changed")
    _source_and_lowerer_task(spec, contract)
    path = workspace.path(contract.target, write=True)
    actual = path.read_bytes() if path.exists() else None
    if actual != admission["target_before"]:
        raise ProviderLeaseError("target changed after admission")


def apply(spec, workspace, contract, admission):
    validate_lease(spec, workspace, contract, admission)
    args = dict(path=contract.target, expected_hash=admission["target_before_sha256"])
    if admission["target_before"] is None:
        args["content"] = admission["candidate_source"]
    else:
        args.update(
            old=admission["target_before"].decode("utf8"), new=admission["candidate_source"]
        )
    return workspace.edit(**args)
