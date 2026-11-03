"""Public behavior scenarios are executable hypotheses,never acceptance oracles."""

import ast
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat

SCRIPT_NAME = "behavior-plan.py"
MODEL_SCRIPT = "/tmp/task-scratch/" + SCRIPT_NAME
MODEL_EVIDENCE = "/tmp/ic-behavior-plan-evidence"


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def requirement_text(public_task):
    """Mask recognized public source JSON for nomination;retain original offsets.

    The full Core task remains unchanged for generation and literal provenance.
    Code mentioning checkpoint/CUDA is not itself a requested behavior.
    """
    masked = list(public_task)
    offset = 0
    marker = "\n\nFILES:\n"
    decoder = json.JSONDecoder()
    while True:
        index = public_task.find(marker, offset)
        if index < 0:
            break
        start = index + len(marker)
        while start < len(public_task) and public_task[start].isspace():
            start += 1
        try:
            value, length = decoder.raw_decode(public_task[start:])
        except ValueError:
            offset = start
            continue
        end = start + length
        offset = end
        if isinstance(value, dict) and all(
            isinstance(k, str) and isinstance(v, str) for k, v in value.items()
        ):
            masked[start:end] = ["\n" if c == "\n" else " " for c in public_task[start:end]]
    return "".join(masked)


@dataclass(frozen=True)
class BehaviorPlanPolicy:
    max_runs: int = 3
    max_script_versions: int = 2
    run_seconds: float = 20
    total_seconds: float = 60
    context_tokens: int = 768

    def __post_init__(self):
        import math

        for name, maximum in [("max_runs", 3), ("max_script_versions", 2), ("context_tokens", 768)]:
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError("bounded " + name + " required")
        for name, maximum in [("run_seconds", 20), ("total_seconds", 60)]:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 < value <= maximum
            ):
                raise ValueError("bounded " + name + " required")


def route(public_task, *, observed_paths, cpu_capability):
    """Literal evidence nominates a workflow;it does not infer task semantics."""
    result = dict(
        applicable=False,
        public_quotes=[],
        basis="explicit public text and caller capabilities;not starter size",
        semantic_classification="heuristic nomination;not proof of hidden behavior",
    )
    if not cpu_capability:
        return dict(result, reason="no caller CPU scenario capability")
    if not observed_paths:
        return dict(result, reason="no caller observed implementation version set")
    requirements = requirement_text(public_task)
    if re.search(r"\b(?:CUDA|Triton)\b|same CUDA device|GPU.only", requirements, re.I):
        return dict(
            result, reason="CUDA/Triton requirements;CPU scenario does not validate this domain"
        )
    # Strong task terms avoid matching the generic sandbox notice about tmp
    # persistence. No dataset names,failure strings or task IDs participate.
    patterns = [
        (
            "persistent_state",
            r"\b(?:checkpoint|state_dict|resume_existing|restor\w*|serializ\w*|deserializ\w*)\b|持久化|状态机|断点恢复",
        ),
        (
            "cross_call",
            r"\b(?:across (?:multiple )?calls|subsequent calls|later calls|next call|cancel\w*|resource lifecycle)\b|跨调用|资源生命周期",
        ),
        (
            "integration",
            r"\b(?:cross.module|multiple components|multi.module|component integration|modular)\b|跨模块|组件集成",
        ),
    ]
    hits = []
    for kind, pattern in patterns:
        match = re.search(pattern, requirements, re.I)
        if match:
            hits.append(dict(kind=kind, quote=match[0], offset=match.start()))
    if not hits:
        return dict(
            result, reason="no explicit multistep,state or integration requirement nominated"
        )
    return dict(
        result,
        applicable=True,
        public_quotes=hits,
        reason="public multistep behavior nominated;actual execution remains required",
    )


def read_script(scratch):
    """Read only the exact bounded regular file;never import it in the host."""
    root = Path(scratch).absolute()
    path = root / SCRIPT_NAME
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("scenario scratch cannot have symlink parents")
    if not path.exists() and not path.is_symlink():
        return None
    preliminary = path.lstat()
    if not stat.S_ISREG(preliminary.st_mode) or preliminary.st_size > 16384:
        raise ValueError("scenario must be a regular file of at most16KiB")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    )
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > 16384:
            raise ValueError("scenario must be a regular file of at most16KiB")
        body = stream.read(16385)
        after = os.fstat(stream.fileno())
    # Compare ctime across the same open handle. Windows path/handle APIs can
    # expose different creation-time metadata;inode/size/mtime still identify
    # the path,as in the shared NativeEpisodeMemory source snapshot reader.
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if (
        identity(before) != identity(after)
        or before.st_ctime_ns != after.st_ctime_ns
        or identity(after) != identity(path.lstat())
    ):
        raise ValueError("scenario changed while reading")
    text = body.decode("utf8")
    return dict(text=text, sha256=digest(text), bytes=len(body))


def parse_script(value, public_task):
    tree = ast.parse(value["text"])
    basis = []
    functions = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "PUBLIC_BASIS" for t in node.targets
        ):
            basis.append(ast.literal_eval(node.value))
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "PUBLIC_BASIS"
        ):
            basis.append(ast.literal_eval(node.value))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "exercise":
            functions.append(node)
    if (
        len(basis) != 1
        or not isinstance(basis[0], str)
        or not 16 <= len(basis[0]) <= 512
        or basis[0] not in public_task
    ):
        raise ValueError(
            "PUBLIC_BASIS must be one exact16..512 character quote from the original public task"
        )
    if len(functions) != 1 or not isinstance(functions[0], ast.FunctionDef):
        raise ValueError("one synchronous exercise(workspace,scratch) required")
    args = functions[0].args
    if (
        args.posonlyargs
        or [a.arg for a in args.args] != ["workspace", "scratch"]
        or args.defaults
        or args.vararg
        or args.kwarg
        or args.kwonlyargs
        or functions[0].decorator_list
    ):
        raise ValueError(
            "exercise requires only workspace,scratch positional arguments and no decorator"
        )
    return dict(
        value,
        public_basis=basis[0],
        authority="model_scenario_and_expected_values_unverified",
        basis_validation="literal provenance only;not proof of scenario applicability",
    )


def executable_script(plan):
    # Compile/execute only inside ReadonlyCpuDiagnostic's namespace. No module
    # is imported in the controller and no candidate solution is synthesized.
    return (
        "from pathlib import Path as _ic_Path\n"
        "_ic_namespace={'__name__':'infinicoder_behavior_scenario'}\n"
        "exec(compile(" + repr(plan["text"]) + ",'behavior-plan.py','exec'),_ic_namespace)\n"
        "_ic_result=_ic_namespace['exercise'](_ic_Path('/workspace'),_ic_Path('/tmp/task-scratch'))\n"
        "if _ic_result is not None: print(repr(_ic_result))\n"
    )


GUIDANCE = """EXECUTABLE BEHAVIOR PLAN (optional ordinary coding workflow)
Turn one unresolved public multistep behavior into a repeatable Python scenario at
/tmp/task-scratch/behavior-plan.py. Use ordinary bash to read,write and debug.
Define PUBLIC_BASIS as an exact16..512 character quote from the original task,
and def exercise(workspace,scratch): with pathlib.Path arguments. Observe one
public behavior;print or return observations. Assertions/expected values are
your hypotheses,not an acceptance oracle. Preserve the public input forms.
The controller runs this scenario with candidate source READ ONLY and a separate
writable diagnostic scratch. Write checkpoint/temporary outputs only in scratch.
It repeats the SAME scenario after observed implementation changes,at most3 runs;
you may revise the scenario once. Use the actual observations to guide edits.
Ordinary read/write/testing and original submission remain available. Final
independent acceptance is still required. No scenario is created on your behalf."""
