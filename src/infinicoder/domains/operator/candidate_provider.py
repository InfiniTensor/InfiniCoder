"""Conditional executable implementation module, not model/judge authority.

Only public AST and caller declarations are used. No reference import/execution,
benchmark IDs, hidden tests, stored patches, generated IR or new process access.
"""

from infinicoder._resources import package_root

import ast
from dataclasses import asdict, dataclass
import hashlib
import re


def digest(text):
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def identity():
    names = (
        "domains/operator/candidate_provider.py",
        "domains/operator/pointwise.py",
        "runtime/engine.py",
        "memory/continuous_conversation.py",
        "runtime/shell_loop.py",
        "memory/source/source_handoff.py",
        "memory/source/source_units.py",
        "execution/workspace_tools.py",
        "optimization/optimization.py",
    )
    root = package_root()
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


@dataclass(frozen=True)
class ProviderContract:
    target: str
    public_reference_sha256: str
    provider: str = "pointwise_v1"
    mode: str = "enabled"
    entry: str = "ModelNew"
    backend: str = "triton"
    dtype: str = "float32"
    device: str = "cuda"
    replacement: str = "missing_implementation"

    def __post_init__(self):
        if self.provider != "pointwise_v1" or self.mode not in ("enabled", "disabled"):
            raise ValueError("Unknown implementation provider/mode")
        if self.entry != "ModelNew" or (self.backend, self.dtype, self.device) != (
            "triton",
            "float32",
            "cuda",
        ):
            raise ValueError("Unsupported caller implementation requirements")
        if self.replacement not in ("missing_implementation", "caller_authorized_complete"):
            raise ValueError("Explicit replacement scope required")
        if (
            not isinstance(self.target, str)
            or not self.target.endswith(".py")
            or any(c in self.target for c in "*?[")
        ):
            raise ValueError("Provider requires one explicit Python target")
        if not isinstance(self.public_reference_sha256, str) or not re.fullmatch(
            "[0-9a-f]{64}", self.public_reference_sha256
        ):
            raise ValueError("Exact public reference hash required")


class ProviderLeaseError(ValueError):
    pass


class UnknownPrerequisite(ValueError):
    pass


def _public_source(spec, contract):
    if spec.task.count("REFERENCE:\n") != 1:
        raise ValueError("One caller public reference required")
    instructions, source = spec.task.split("REFERENCE:\n", 1)
    if len(source) > 65536 or digest(source) != contract.public_reference_sha256:
        raise ValueError("Public reference hash/size mismatch")
    if not re.search(r"\btriton\b", instructions, re.I) or not re.search(
        r"\b(?:FP32|float32)\b", instructions
    ):
        raise UnknownPrerequisite("Explicit Triton and FP32 requirements missing")
    if re.search(r"\b(backward|autograd|gradient|training)\b|反向|梯度|训练", instructions, re.I):
        raise ValueError("Gradient/training semantics unsupported")
    if spec.amendments:
        raise ValueError(
            "Initial provider does not infer how amendments change reference semantics"
        )
    return source


def _reference_shape(source):
    tree = ast.parse(source)
    if sum(1 for _ in ast.walk(tree)) > 2000:
        raise ValueError("Public AST bound")
    aliases = {}
    classes = []
    functions = []
    globals = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in ("torch", "torch.nn", "torch.nn.functional", "math"):
                    raise ValueError("Unsupported reference import")
                aliases[a.asname or a.name.split(".")[0]] = a.name
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module == "torch"
            and node.level == 0
            and all(a.name == "nn" for a in node.names)
        ):
            for a in node.names:
                aliases[a.asname or a.name] = "torch.nn"
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            # Only static literal shape metadata; never evaluate a function call.
            value = ast.literal_eval(node.value)
            if type(value) not in (int, float, tuple, list):
                raise ValueError("Unsupported global metadata")
            globals.add(node.targets[0].id)
        elif isinstance(node, ast.ClassDef):
            classes.append(node)
        elif isinstance(node, ast.FunctionDef):
            functions.append(node)
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            pass
        else:
            raise ValueError("Reference has unsupported module effects")
    if globals & set(aliases):
        raise ValueError("Shadowed library binding")
    if len(classes) != 1 or classes[0].name != "Model":
        raise ValueError("One public Model required")
    cls = classes[0]
    bases = [ast.unparse(b) for b in cls.bases]
    valid = ["torch.nn.Module"] + [
        "%s.Module" % name for name, value in aliases.items() if value == "torch.nn"
    ]
    if cls.keywords or cls.decorator_list or len(bases) != 1 or bases[0] not in valid:
        raise ValueError("Unsupported class construction")
    methods = [
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
        not isinstance(n, ast.FunctionDef) or n.name not in ("__init__", "forward") for n in methods
    ):
        raise ValueError("Unknown model state/descriptors")
    if len({n.name for n in methods}) != len(methods):
        raise ValueError("Duplicate model methods")
    if sorted(n.name for n in functions) != ["get_init_inputs", "get_inputs"]:
        raise ValueError("Only public input factories supported")
    init = next(n for n in functions if n.name == "get_init_inputs")
    if init.decorator_list or len(init.body) != 1 or not isinstance(init.body[0], ast.Return):
        raise ValueError("Unknown constructor arguments")
    ast.literal_eval(init.body[0].value)  # Facts only; provider lowerer decides support.
    forward = next(n for n in methods if n.name == "forward")
    _tensor_expression_types(forward, aliases)
    ctor = next((n for n in methods if n.name == "__init__"), None)
    if ctor and (
        len(ctor.args.args) != 1
        or ctor.args.defaults
        or ctor.args.vararg
        or ctor.args.kwarg
        or ctor.args.kwonlyargs
    ):
        raise ValueError("Only zero-argument state-free constructor supported")
    if ast.literal_eval(init.body[0].value) not in ([], ()):
        raise ValueError("Only empty constructor arguments supported")
    # Reject mutations/callbacks anywhere, including library-name parameters.
    for n in ast.walk(tree):
        if isinstance(n, ast.arg) and n.arg in aliases:
            raise ValueError("Shadowed parameter")
        if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            if any(isinstance(t, (ast.Attribute, ast.Subscript)) for t in targets):
                raise ValueError("Reference mutation unsupported")
    return dict(
        public_class="Model",
        public_entry="Model.forward",
        source_hash=digest(source),
        observed_input_factory_only=True,
        hidden_input_domain="unknown;original judge remains required",
    )


def _tensor_expression_types(forward, aliases):
    """Conservative scalar/tensor distinction missing from the old lowerer.

    Do not turn an invalid math.sqrt(tensor) or scalar return into a purported
    equivalent tensor kernel. This only narrows applicability, not the grammar.
    """
    if len(forward.args.args) != 2:
        raise ValueError("One tensor argument required")
    env = {forward.args.args[1].arg: "tensor"}

    def kind(n):
        if isinstance(n, ast.Name) and n.id in env:
            return env[n.id]
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            return "scalar"
        if (
            isinstance(n, ast.Attribute)
            and ast.unparse(n) == "math.pi"
            and aliases.get("math") == "math"
        ):
            return "scalar"
        if isinstance(n, ast.UnaryOp):
            return kind(n.operand)
        if isinstance(n, ast.BinOp):
            return "tensor" if "tensor" in (kind(n.left), kind(n.right)) else "scalar"
        if isinstance(n, ast.Call):
            name = ast.unparse(n.func)
            if n.keywords:
                raise ValueError("Unknown pointwise call keywords")
            if name == "math.sqrt":
                if (
                    aliases.get("math") != "math"
                    or len(n.args) != 1
                    or not isinstance(n.args[0], ast.Constant)
                    or type(n.args[0].value) not in (int, float)
                    or n.args[0].value < 0
                ):
                    raise ValueError("Only nonnegative literal math.sqrt is known scalar")
                return "scalar"
            arity = 2 if name == "torch.pow" else 1
            if len(n.args) != arity or kind(n.args[0]) != "tensor":
                raise ValueError("Expected tensor intrinsic input and exact arity")
            if arity == 2 and kind(n.args[1]) != "scalar":
                raise ValueError("Expected scalar exponent")
            return "tensor"
        raise ValueError("Unknown expression type")

    output = None
    for n in forward.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            env[n.targets[0].id] = kind(n.value)
        elif isinstance(n, ast.Return):
            output = kind(n.value)
        elif (
            isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Constant)
            and isinstance(n.value.value, str)
        ):
            pass
        else:
            raise ValueError("Unsupported pointwise statements")
    if output != "tensor":
        raise ValueError("Public result is not established as a tensor")


def assess(spec, workspace, contract):
    """Return explicit applicability facts; unsupported conditions are not repaired."""
    result = dict(
        provider=contract.provider,
        status="Unknown",
        acceptance_proof=False,
        implementation=identity(),
    )
    try:
        source = _public_source(spec, contract)
        facts = _reference_shape(source)
        target = workspace.path(contract.target, write=True)
        if target.exists():
            original = target.read_bytes().decode("utf8")
            if original == "":
                raise ValueError(
                    "Existing empty target has no original unique edit anchor;use the ordinary path"
                )
            if contract.replacement == "missing_implementation" and ast.parse(original).body:
                raise ValueError("Existing implementation is not an empty caller target")
        else:
            original = None
        from infinicoder.domains.operator.pointwise import propose

        candidate = propose(spec.task)
        if candidate is None:
            raise ValueError("Original pointwise compiler does not support this reference")
        from infinicoder.memory.repair_memory import signature

        result.update(
            status="Applicable",
            requirements=signature(spec),
            workspace_revision=workspace.revision(),
            target=contract.target,
            target_before_sha256=digest(original) if original is not None else None,
            target_before=original,
            public_reference_sha256=digest(source),
            facts=facts,
            supported_conditions=list(candidate.preconditions),
            expression=candidate.expression,
            contract=asdict(contract),
            candidate_sha256=digest(candidate.source),
        )
    except (
        ValueError,
        TypeError,
        SyntaxError,
        RecursionError,
        OverflowError,
        IndexError,
        StopIteration,
        UnicodeError,
        OSError,
    ) as error:
        result.update(
            status="Unknown" if isinstance(error, UnknownPrerequisite) else "Unsupported",
            reason=type(error).__name__ + ": " + str(error),
        )
    return result


def propose(spec, workspace, contract, admission):
    validate(spec, workspace, contract, admission)
    from infinicoder.domains.operator.pointwise import propose as lower

    candidate = lower(spec.task)
    if candidate is None or digest(candidate.source) != admission["candidate_sha256"]:
        raise ProviderLeaseError("Implementation changed after admission")
    args = dict(path=contract.target, expected_hash=admission["target_before_sha256"])
    if admission["target_before"] is None:
        args["content"] = candidate.source
    else:
        args.update(old=admission["target_before"], new=candidate.source)
    return dict(tool="edit", arguments=args)


def validate(spec, workspace, contract, admission):
    from infinicoder.memory.repair_memory import signature

    if admission.get("status") != "Applicable":
        raise ProviderLeaseError("No applicable provider admission")
    source = _public_source(spec, contract)
    if (
        admission["requirements"] != signature(spec)
        or admission["implementation"] != identity()
        or admission["workspace_revision"] != workspace.revision()
        or admission["contract"] != asdict(contract)
        or admission["public_reference_sha256"] != digest(source)
    ):
        raise ProviderLeaseError("Implementation provider lease changed;candidate not applied")
