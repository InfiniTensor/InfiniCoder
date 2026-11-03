"""Source-bound local integration of the existing sum kernel with exact fallback.

No target imports,model calls,benchmark identities or evaluator access are used
to prepare a bundle. A caller must authorize the function and both helper files,
and explicitly allow numerical reassociation for the optimized subset.
"""

from infinicoder._resources import package_root

import ast
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

from infinicoder.search.candidate_checkpoint import (
    CandidateCheckpointStore,
    CheckpointConflict,
    revision,
)
from infinicoder.domains.operator.candidate_provider import ProviderLeaseError, UnknownPrerequisite
from infinicoder.domains.operator.reduction import SumDescriptor, _generate_descriptor
from infinicoder.domains.operator.guarded_sum_runtime import TEMPLATE
from infinicoder.execution.workspace_tools import Workspace


def digest(data):
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def _path(value):
    if not isinstance(value, str):
        raise ValueError("explicit normalized Python path required")
    p = PurePosixPath(value)
    if (
        p.is_absolute()
        or ".." in p.parts
        or str(p) != value
        or p.suffix != ".py"
        or any(part in (".git", ".infinicoder") for part in p.parts)
        or any(c in value for c in "\\:*?[]\0")
    ):
        raise ValueError("explicit normalized Python path required")
    return p


@dataclass(frozen=True)
class GuardedSumContract:
    target: str
    function: str
    source_sha256: str
    expression_span: tuple
    dispatcher_path: str
    backend_path: str
    scope_quote: str
    numerical_quote: str
    rtol: float
    atol: float
    enabled: bool = False
    allow_reassociation: bool = False

    def __post_init__(self):
        paths = [_path(p) for p in (self.target, self.dispatcher_path, self.backend_path)]
        if len(set(paths)) != 3 or any(p.parent != paths[0].parent for p in paths):
            raise ValueError("one target and two distinct sibling helpers required")
        if any(not p.stem.isidentifier() or not p.stem.startswith("_ic_") for p in paths[1:]):
            raise ValueError("explicit private helper module names required")
        if (
            not isinstance(self.function, str)
            or not self.function.isidentifier()
            or not isinstance(self.source_sha256, str)
            or not re.fullmatch("[a-f0-9]{64}", self.source_sha256)
        ):
            raise ValueError("exact function name and original source hash required")
        span = self.expression_span
        if (
            not isinstance(span, (tuple, list))
            or len(span) != 4
            or any(type(n) is not int or n < 0 for n in span)
            or span[0] < 1
            or (span[0], span[1]) >= (span[2], span[3])
        ):
            raise ValueError("exact AST call position required")
        object.__setattr__(self, "expression_span", tuple(span))
        if type(self.enabled) is not bool or type(self.allow_reassociation) is not bool:
            raise ValueError("explicit boolean enablement and numerical permission required")
        if (
            not isinstance(self.scope_quote, str)
            or not 1 <= len(self.scope_quote) <= 4096
            or any(
                p not in self.scope_quote
                for p in (self.target, self.dispatcher_path, self.backend_path, self.function)
            )
        ):
            raise ValueError("caller quote must authorize the function and both exact helper paths")
        if not isinstance(self.numerical_quote, str) or not 1 <= len(self.numerical_quote) <= 4096:
            raise ValueError("exact public numerical agreement required")
        for value in (self.rtol, self.atol):
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("finite nonnegative numerical tolerances required")
        if not self.rtol and not self.atol:
            raise ValueError("bitwise equivalence is not supported by this optimization")
        try:
            agreement = json.loads(self.numerical_quote)
        except (TypeError, ValueError):
            raise ValueError("numerical quote must be the explicit JSON agreement")
        if agreement != dict(
            allow_fp32_sum_reassociation=True,
            rtol=self.rtol,
            atol=self.atol,
            equal_nan=True,
            signed_zero="not_preserved",
        ):
            raise ValueError(
                "numerical quote must state the actual tolerance and exceptional-value policy"
            )


@dataclass(frozen=True)
class GuardedSumBundle:
    contract_json: str
    task_sha256: str
    producer_pins: tuple
    target_before: bytes
    artifacts: tuple
    schedule: str


def identity():
    root = package_root()
    return {
        name: digest((root / name).read_bytes())
        for name in (
            "domains/operator/guarded_sum_callsite.py",
            "domains/operator/guarded_sum_runtime.py",
            "domains/operator/reduction.py",
        )
    }


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _binding(source, contract):
    if len(source) > 524288:
        raise ValueError("bounded public module required")
    tree = ast.parse(source.decode("utf-8"))
    if sum(1 for _ in ast.walk(tree)) > 60000:
        raise ValueError("bounded public AST required")
    aliases = {
        item.asname or "torch"
        for node in tree.body
        if isinstance(node, ast.Import)
        for item in node.names
        if item.name == "torch"
    }
    definitions = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == contract.function
    ]
    if len(definitions) != 1:
        raise UnknownPrerequisite("one top-level ordinary function required")
    function = definitions[0]
    statements = [
        n
        for n in function.body
        if not (
            isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Constant)
            and isinstance(n.value.value, str)
        )
    ]
    if function.decorator_list or len(statements) != 1 or not isinstance(statements[0], ast.Return):
        raise UnknownPrerequisite(
            "only a direct,undecorated reduction function is currently supported"
        )
    statement = statements[0]
    call = statement.value
    if (
        not isinstance(call, ast.Call)
        or (call.lineno, call.col_offset, call.end_lineno, call.end_col_offset)
        != contract.expression_span
    ):
        raise ProviderLeaseError("exact call position changed")
    if (
        not isinstance(call.func, ast.Attribute)
        or call.func.attr != "sum"
        or not isinstance(call.func.value, ast.Name)
        or call.func.value.id not in aliases
        or not call.args
        or not isinstance(call.args[0], ast.Name)
    ):
        raise UnknownPrerequisite("ordinary bound torch.sum on one plain argument required")
    alias = call.func.value.id
    input_name = call.args[0].id
    args = function.args
    if (
        args.posonlyargs
        or args.kwonlyargs
        or args.vararg
        or args.kwarg
        or args.defaults
        or len(args.args) != 1
        or args.args[0].arg != input_name
        or input_name == alias
    ):
        raise UnknownPrerequisite("single ordinary tensor parameter required")
    # Reject namespace rebinding;runtime also checks the actual callable before
    # choosing the kernel,so legal torch.sum monkeypatches retain their behavior.
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(n, ast.Name) and n.id == alias for t in targets for n in ast.walk(t)):
                raise UnknownPrerequisite("tensor-library binding is reassigned")
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == alias:
            raise UnknownPrerequisite("tensor-library binding is shadowed")
    if len(call.args) > 3 or any(k.arg not in ("dim", "keepdim") for k in call.keywords):
        raise UnknownPrerequisite("only a static single axis and keepdim are supported")
    options = dict(zip(("dim", "keepdim"), call.args[1:]))
    for keyword in call.keywords:
        if keyword.arg in options:
            raise ValueError("duplicate reduction argument")
        options[keyword.arg] = keyword.value
    try:
        axis = ast.literal_eval(options["dim"])
        keepdim = ast.literal_eval(options["keepdim"]) if "keepdim" in options else False
    except (KeyError, ValueError, TypeError):
        raise UnknownPrerequisite("literal axis and keepdim required")
    if type(axis) is not int or type(keepdim) is not bool:
        raise UnknownPrerequisite("one literal integer axis and bool keepdim required")
    return function, statement, call, input_name, alias, axis, keepdim


def prepare_guarded_sum(spec, workspace, contract, *, schedule="scalar"):
    if type(contract) is not GuardedSumContract:
        raise TypeError("GuardedSumContract required")
    if not contract.enabled or not contract.allow_reassociation:
        raise UnknownPrerequisite(
            "optimization and approximate numerical relation must be explicitly enabled"
        )
    if (
        contract.scope_quote not in spec.task
        or contract.numerical_quote not in spec.task
        or spec.amendments
    ):
        raise ProviderLeaseError("current caller permissions or numerical agreement do not match")
    if spec.milestones:
        raise UnknownPrerequisite(
            "staged calls require an explicit active-unit binding;not inferred here"
        )
    paths = (contract.target, contract.dispatcher_path, contract.backend_path)
    for path in paths:
        workspace.path(path, write=True)
    if spec.allowed_paths:
        declared = Workspace(workspace.root, spec.allowed_paths)
        for path in paths:
            declared.path(path, write=True)
    source = workspace.path(contract.target).read_bytes()
    if digest(source) != contract.source_sha256:
        raise ProviderLeaseError("original module changed")
    if any(workspace.path(path).exists() for path in paths[1:]):
        raise ProviderLeaseError("helpers must be fresh caller-authorized files")
    function, statement, call, name, alias, axis, keepdim = _binding(source, contract)
    descriptor = SumDescriptor(
        "self", None, None, repr(axis), keepdim, "x", contract.source_sha256, ()
    )
    backend = _generate_descriptor(descriptor, schedule).source
    original = ast.parse(ast.unparse(call), mode="eval").body
    original.func = ast.Name(id="original_sum", ctx=ast.Load())
    original.args[0] = ast.Name(id="x", ctx=ast.Load())
    numerical = _json(
        dict(
            reassociation_explicitly_allowed=True,
            rtol=contract.rtol,
            atol=contract.atol,
            guarantee="caller agreement;metadata guard is not a numerical proof",
            source_sha256=contract.source_sha256,
        )
    )
    # repr emits valid Python,including boolean literals;no caller prose is executed.
    numerical = repr(json.loads(numerical))
    dispatcher = (
        TEMPLATE.replace("__NUMERICAL_CONTRACT__", numerical)
        .replace("__AXIS__", repr(axis))
        .replace("__BACKEND_MODULE__", PurePosixPath(contract.backend_path).stem)
        .replace("__ORIGINAL_CALL__", ast.unparse(original))
    )
    local = "_ic_sum_" + contract.source_sha256[:12]
    if any(
        isinstance(n, (ast.Name, ast.arg)) and (n.id if isinstance(n, ast.Name) else n.arg) == local
        for n in ast.walk(function)
    ):
        raise ProviderLeaseError("generated local name collides with the original function")
    lines = source.splitlines(keepends=True)
    start = sum(map(len, lines[: statement.lineno - 1])) + statement.col_offset
    end = sum(map(len, lines[: statement.end_lineno - 1])) + statement.end_col_offset
    indent = " " * statement.col_offset
    replacement = (
        f"from .{PurePosixPath(contract.dispatcher_path).stem} import run as {local}\n"
        f"{indent}return {local}({name}, {alias}.sum)"
    ).encode("utf-8")
    target = source[:start] + replacement + source[end:]
    for raw in (target, dispatcher.encode(), backend.encode()):
        ast.parse(raw)
    return GuardedSumBundle(
        _json(asdict(contract)),
        digest(spec.task),
        tuple(sorted(identity().items())),
        source,
        ((paths[0], target), (paths[1], dispatcher.encode()), (paths[2], backend.encode())),
        schedule,
    )


def apply_guarded_sum(spec, workspace, contract, bundle, *, checkpoint_store):
    if (
        type(bundle) is not GuardedSumBundle
        or type(checkpoint_store) is not CandidateCheckpointStore
        or checkpoint_store.workspace != workspace.root
    ):
        raise ValueError("explicit exclusive task checkpoint store for this workspace required")
    current = prepare_guarded_sum(spec, workspace, contract, schedule=bundle.schedule)
    if current != bundle:
        raise ProviderLeaseError("caller,source,helper,producer or artifact lease changed")
    baseline = checkpoint_store.capture("baseline")
    before = checkpoint_store.load(baseline)
    allowed = {p for p, _ in bundle.artifacts}
    expected = {p: digest(raw) for p, raw in bundle.artifacts}
    try:
        # The workspace is exclusive. Helpers are written before publishing the
        # single local import,then the existing checkpoint provides rollback.
        for path, raw in (*bundle.artifacts[1:], bundle.artifacts[0]):
            if path == contract.target:
                workspace.edit(
                    path,
                    contract.source_sha256,
                    old=bundle.target_before.decode(),
                    new=raw.decode(),
                )
            else:
                workspace.edit(path, None, content=raw.decode())
        after = checkpoint_store.scan()
        if any(
            before["entries"].get(p) != v for p, v in after["entries"].items() if p not in allowed
        ) or any(
            v != after["entries"].get(p) for p, v in before["entries"].items() if p not in allowed
        ):
            raise CheckpointConflict("unrelated source changed during local installation")
        return dict(
            status="installed_not_verified",
            revision=revision(after),
            schedule=bundle.schedule,
            artifacts=expected,
            source_before_sha256=contract.source_sha256,
            numerical_proof=False,
            optimized_subset="explicit inference-mode ordinary FP32 CUDA rank2/3 tensors only",
            default_enabled=False,
            independent_correctness_and_performance_required=True,
        )
    except BaseException as error:
        try:
            observed = checkpoint_store.scan(guard=False)
            # Never overwrite another writer's bytes to manufacture an atomic result.
            if (
                all(
                    before["entries"].get(p) == v
                    for p, v in observed["entries"].items()
                    if p not in allowed
                )
                and all(
                    before["entries"].get(p) == observed["entries"].get(p)
                    for p in before["entries"]
                    if p not in allowed
                )
                and all(
                    p not in observed["entries"]
                    or observed["entries"][p].get("sha256")
                    in (expected[p], before["entries"].get(p, {}).get("sha256"))
                    for p in allowed
                )
            ):
                checkpoint_store.restore(baseline, expected_revision=revision(observed))
        except BaseException as rollback_error:
            if hasattr(error, "add_note"):
                error.add_note("Checkpoint rollback incomplete: " + type(rollback_error).__name__)
        raise


def main():
    """Explicit CLI capability usable through the ordinary command executor."""
    import argparse, time
    from infinicoder.planning.task_contract import TaskSpec

    parser = argparse.ArgumentParser(
        description="Prepare or install an explicit guarded sum callsite;no model or GPU runs"
    )
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--task-spec", required=True, help="Caller-owned full TaskSpec JSON")
    parser.add_argument("--contract", required=True, help="Caller-owned GuardedSumContract JSON")
    parser.add_argument("--schedule", choices=("scalar", "coalesced"), default="scalar")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--exclusive-workspace", action="store_true")
    parser.add_argument("--state-dir")
    parser.add_argument("--seconds", type=float, default=30)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 5 <= args.seconds <= 120:
        parser.error("seconds must be between5 and120")
    spec = TaskSpec(**json.loads(Path(args.task_spec).read_bytes()))
    contract = GuardedSumContract(**json.loads(Path(args.contract).read_bytes()))
    workspace = Workspace(args.workspace, spec.allowed_paths, validate_python=True)
    bundle = prepare_guarded_sum(spec, workspace, contract, schedule=args.schedule)
    if args.apply:
        if not args.exclusive_workspace or not args.state_dir:
            parser.error("apply requires exclusive workspace and separate fresh state-dir")
        deadline = time.monotonic() + args.seconds
        store = CandidateCheckpointStore(
            workspace.root,
            Path(args.state_dir),
            exclusive_workspace=True,
            remaining=lambda: deadline - time.monotonic(),
        )
        result = apply_guarded_sum(spec, workspace, contract, bundle, checkpoint_store=store)
    else:
        result = dict(
            status="prepared_not_installed",
            schedule=args.schedule,
            artifacts={path: digest(raw) for path, raw in bundle.artifacts},
            numerical_proof=False,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
