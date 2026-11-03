"""Optional public-reference experiments; never an acceptance oracle.

The caller supplies immutable public files, their replacement targets and the
one callable interface. The model supplies inputs and a short driver. Runtime
captures actual callable arguments/results, rather than trusting driver prints.
File snapshots reuse the original executor; they add no OS sandbox authority.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import time

from infinicoder.memory.repair_memory import signature
from infinicoder.verification.verification import Snapshot, environment_id


def encoded(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def safe_path(value):
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 240
        or "\\" in value
        or ":" in value
        or "\x00" in value
        or any(c in value for c in "*?[")
    ):
        raise ValueError("reference paths must be exact bounded relative files")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(p in (".", "..", ".git", ".infinicoder") for p in path.parts)
    ):
        raise ValueError("unsafe reference path")
    return value


@dataclass(frozen=True)
class ReferenceFile:
    target: str
    public_path: str
    sha256: str

    def __post_init__(self):
        safe_path(self.target)
        safe_path(self.public_path)
        if not isinstance(self.sha256, str) or not re.fullmatch("[0-9a-f]{64}", self.sha256):
            raise ValueError("reference file requires a caller-bound SHA256")


@dataclass(frozen=True)
class ReferenceContract:
    files: tuple
    entry_path: str
    entrypoint: str
    requirements: str
    phases: tuple
    public: bool
    deterministic: bool
    seed: int = 42
    atol: float = 0.0
    rtol: float = 0.0

    def __post_init__(self):
        object.__setattr__(
            self,
            "files",
            tuple(ReferenceFile(**f) if isinstance(f, dict) else f for f in self.files),
        )
        object.__setattr__(self, "phases", tuple(self.phases))
        if not 1 <= len(self.files) <= 8 or any(
            not isinstance(f, ReferenceFile) for f in self.files
        ):
            raise ValueError("reference contract requires 1..8 public file mappings")
        targets = [f.target for f in self.files]
        sources = [f.public_path for f in self.files]
        if (
            len(set(targets)) != len(targets)
            or len(set(sources)) != len(sources)
            or set(targets) & set(sources)
        ):
            raise ValueError("reference targets and public paths must be distinct")
        if (
            self.entry_path not in targets
            or not isinstance(self.entrypoint, str)
            or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", self.entrypoint)
        ):
            raise ValueError("reference entrypoint must be a caller-selected mapped callable")
        if not isinstance(self.requirements, str) or not re.fullmatch(
            "[0-9a-f]{64}", self.requirements
        ):
            raise ValueError("reference requires the exact task requirements identity")
        if (
            not self.phases
            or any(type(p) is not int or p < 0 for p in self.phases)
            or len(set(self.phases)) != len(self.phases)
        ):
            raise ValueError("reference requires explicit distinct implementation phases")
        if self.public is not True or self.deterministic is not True:
            raise ValueError("only explicitly public deterministic references are supported")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("reference seed must be a uint32")
        for value in (self.atol, self.rtol):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("reference tolerances must be finite nonnegative caller values")

    @property
    def identity(self):
        return digest(asdict(self))


def identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def bind(contract, spec, workspace, phase=None):
    """No task-name heuristics, baseline inference or private file lookup."""
    if (
        signature(spec) != contract.requirements
        or phase is not None
        and phase not in contract.phases
    ):
        raise ValueError("public reference is unavailable for these requirements/phase")
    protected = {
        str((Path(p) if Path(p).is_absolute() else workspace.path(p)).resolve())
        for check in spec.checks
        for p in check.protected_paths
    }
    values = {}
    for item in contract.files:
        target = workspace.path(item.target, write=True)
        source = workspace.path(item.public_path)
        if not source.is_file() or source.is_symlink() or target.is_dir() or target.is_symlink():
            raise ValueError(
                "public reference must be a regular file;implementation cannot be a directory/link"
            )
        if phase is not None and not target.is_file():
            raise ValueError(
                "reference experiment requires an existing implementation;continue normal coding first"
            )
        # Require both the declared protection and a non-writable source scope.
        try:
            workspace.path(item.public_path, write=True)
        except ValueError:
            pass
        else:
            raise ValueError("public reference must be outside the editable file scope")
        if str(source.resolve()) not in protected:
            raise ValueError("caller must protect the public reference file")
        if hashlib.sha256(source.read_bytes()).hexdigest() != item.sha256:
            raise ValueError("public reference changed from its caller-bound hash")
        values[item.target] = (
            hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else None
        )
    return values


def available(contract, state):
    return contract is not None and (state or {}).get("milestone") in contract.phases


def view(contract, phase=None):
    return dict(
        origin="caller_public_reference_contract",
        identity=contract.identity,
        files=[asdict(f) for f in contract.files],
        entry_path=contract.entry_path,
        entrypoint=contract.entrypoint,
        phases=list(contract.phases),
        seed=contract.seed,
        atol=contract.atol,
        rtol=contract.rtol,
        available_in_current_phase=phase in contract.phases,
        invoke='ic-tool {"tool":"reference_compare","arguments":{"driver":"def experiment(call, inputs):\\n    for x in inputs: call(x)","inputs":[0,1]}} inside the current shell fence',
        limits="Optional Python JSON-callable experiment only. Same driver/inputs and Python random seed in both snapshots;no other randomness or external state. At most8 calls,2 experiments/session,5 seconds/pair within the original budget. Driver output/prints are ignored;Runtime captures actual calls. Observations are not final acceptance.",
    )


def schema():
    return dict(
        type="object",
        properties=dict(
            driver=dict(type="string", minLength=1, maxLength=4096),
            inputs=dict(type="array", minItems=1, maxItems=8),
        ),
        required=["driver", "inputs"],
        additionalProperties=False,
    )


def prepare(contract, spec, state, workspace, args):
    revision = workspace.revision()
    targets = bind(contract, spec, workspace, state["milestone"])
    if workspace.revision() != revision:
        raise ValueError("source changed while binding public reference")
    if not isinstance(args, dict) or set(args) != {"driver", "inputs"}:
        raise ValueError(
            "reference_compare requires driver and inputs only;caller owns all tolerances"
        )
    driver = args["driver"]
    inputs = args["inputs"]
    if (
        not isinstance(driver, str)
        or not driver.strip()
        or len(driver.encode()) > 4096
        or "\x00" in driver
    ):
        raise ValueError("reference experiment driver must be nonempty and at most4096 bytes")
    compile(driver, "<reference-driver>", "exec")
    if (
        not isinstance(inputs, list)
        or not 1 <= len(inputs) <= 8
        or len(encoded(inputs).encode()) > 2048
    ):
        raise ValueError("provide 1..8 finite JSON inputs in at most2048 bytes")
    key = digest([contract.identity, targets, driver, inputs])
    prior = state.get("reference_experiment_attempts", [])
    if len(prior) >= 2:
        raise ValueError("reference experiment session limit reached")
    if any(r["key"] == key for r in prior):
        raise ValueError("identical reference experiment already attempted on this candidate")
    return dict(
        key=key,
        contract=contract.identity,
        requirements=signature(spec),
        phase=state["milestone"],
        revision=revision,
        targets=targets,
        driver=driver,
        inputs=inputs,
        driver_sha256=hashlib.sha256(driver.encode()).hexdigest(),
        inputs_sha256=digest(inputs),
    )


# Only Runtime-owned call records are compared. Returning/printing PASSED,
# fabricated expected values, or never invoking call produces no observation.
HARNESS = r"""
import contextlib, importlib.util, json, os, random, sys
from pathlib import Path
artifacts=Path(__file__).resolve().parent;root=Path.cwd()
payload=json.loads((artifacts/"payload.json").read_text())
rows=[];driver_error=None;setup_error=None;limit_error=None
def pack(value):
    text=json.dumps(value,allow_nan=False,ensure_ascii=False,separators=(",",":"))
    if len(text.encode())>1024:raise ValueError("observation exceeds1024 bytes")
    return json.loads(text)
def error(exc):return dict(type=type(exc).__name__,message=str(exc)[:240])
random.seed(payload["seed"])
with open(os.devnull,"w") as sink,contextlib.redirect_stdout(sink),contextlib.redirect_stderr(sink):
    try:
        spec=importlib.util.spec_from_file_location("_ic_compared_implementation",root/payload["entry_path"])
        module=importlib.util.module_from_spec(spec);sys.path.insert(0,str(root));spec.loader.exec_module(module)
        target=module
        for part in payload["entrypoint"].split("."):target=getattr(target,part)
        if not callable(target):raise TypeError("caller entrypoint is not callable")
    except Exception as exc:setup_error=error(exc)
    if setup_error is None:
        def call(*args,**kwargs):
            global limit_error
            if len(rows)>=8:
                limit_error="call limit exceeded";raise ValueError(limit_error)
            try:arguments=pack(dict(args=args,kwargs=kwargs))
            except Exception as exc:
                limit_error="unrepresentable call arguments";raise ValueError(limit_error) from exc
            row=dict(arguments=arguments);rows.append(row)
            try:value=target(*args,**kwargs)
            except Exception as exc:
                row["exception"]=error(exc);raise
            try:row["value"]=pack(value)
            except Exception as exc:
                limit_error="unrepresentable call result";raise ValueError(limit_error) from exc
            return value
        try:
            namespace={};exec(compile(payload["driver"],"<reference-driver>","exec"),namespace)
            namespace["experiment"](call,payload["inputs"])
        except Exception as exc:driver_error=error(exc)
report=dict(calls=rows,setup_error=setup_error,driver_error=driver_error,limit_error=limit_error)
blob=json.dumps(report,allow_nan=False,ensure_ascii=False,separators=(",",":"))
if len(blob.encode())>12000:blob=json.dumps(dict(calls=[],limit_error="report exceeds12000 bytes"))
print(blob)
"""


def difference(a, b, atol, rtol, path="$"):
    """Preserve sequence order, duplicates, types and mapping keys."""
    if type(a) in (int, float) and type(b) in (int, float):
        if abs(a - b) <= atol + rtol * abs(a):
            return None
    elif type(a) is type(b):
        if isinstance(a, dict) and a.keys() == b.keys():
            for key in sorted(a):
                found = difference(a[key], b[key], atol, rtol, path + "." + key)
                if found:
                    return found
            return None
        if isinstance(a, list) and len(a) == len(b):
            for i, (left, right) in enumerate(zip(a, b)):
                found = difference(left, right, atol, rtol, path + "[" + str(i) + "]")
                if found:
                    return found
            return None
        if not isinstance(a, (list, dict)) and a == b:
            return None
    return dict(path=path, reference=a, current=b)


def compare_reports(reference, current, contract):
    errors = lambda report: {
        k: report.get(k) for k in ("setup_error", "driver_error", "limit_error") if report.get(k)
    }
    if (
        reference.get("setup_error")
        or current.get("setup_error")
        or reference.get("limit_error")
        or current.get("limit_error")
    ):
        return "inconclusive", dict(
            reason="setup_or_observation_limit",
            reference=errors(reference),
            current=errors(current),
        )
    left = reference.get("calls")
    right = current.get("calls")
    if not isinstance(left, list) or not isinstance(right, list) or not left or not right:
        return "inconclusive", dict(reason="no_actual_callable_observations")
    if reference.get("driver_error") or any("exception" in row for row in left):
        return "inconclusive", dict(
            reason="reference_or_reference_driver_failed", reference=errors(reference)
        )
    for i, (a, b) in enumerate(zip(left, right)):
        if encoded(a["arguments"]) != encoded(b["arguments"]):
            return "inconclusive", dict(reason="driver_called_different_inputs", call=i)
        if "exception" in b:
            return "counterexample", dict(
                call=i,
                arguments=a["arguments"],
                reference=a["value"],
                current_exception=b["exception"],
            )
        found = difference(a["value"], b["value"], contract.atol, contract.rtol)
        if found:
            return "counterexample", dict(call=i, arguments=a["arguments"], difference=found)
    if len(left) != len(right) or current.get("driver_error"):
        return "inconclusive", dict(
            reason="incomplete_or_divergent_driver",
            reference=errors(reference),
            current=errors(current),
        )
    return "observed_values_match", dict(
        calls=len(left),
        meaning="Only these actual calls matched;not general equivalence or acceptance.",
    )


def probe(contract, declaration, workspace, snapshot_dir, process, remaining, clamp_timeout):
    """Both processes share <=5s including preparation and original reserves."""
    started = time.monotonic()
    expires = started + min(5.0, remaining())
    left = lambda: max(0.0, min(expires - time.monotonic(), remaining()))
    snapshots = []
    executions = []
    outcome = "inconclusive"
    detail = {}
    unchanged = False
    folder = Path(snapshot_dir).resolve()
    if folder.is_relative_to(workspace.root) or folder.exists():
        raise ValueError("reference experiment requires a new external artifact directory")
    folder.mkdir(parents=True)
    try:
        if left() <= 0:
            raise TimeoutError("no shared reference experiment budget remains")
        if workspace.revision() != declaration["revision"]:
            raise ValueError("candidate changed before reference experiment")
        for arm in ("reference", "current"):
            snap = Snapshot(workspace.root, folder / arm)
            snapshots.append(snap)
            if (
                snap.revision() != declaration["revision"]
                or workspace.revision() != declaration["revision"]
            ):
                raise ValueError("source changed while copying reference experiment")
            if arm == "reference":
                for item in contract.files:
                    source = snap.dir / item.public_path
                    if hashlib.sha256(source.read_bytes()).hexdigest() != item.sha256:
                        raise ValueError("public reference copy changed")
                    (snap.dir / item.target).write_bytes(source.read_bytes())
            if left() <= 0:
                raise TimeoutError(
                    "reference experiment budget expired during snapshot preparation"
                )
        reports = []
        for snap in snapshots:
            temporary = snap.dir / "_ic_reference_experiment"
            if temporary.exists():
                raise ValueError("reserved experiment artifact path already exists")
            temporary.mkdir()
            payload = {k: declaration[k] for k in ("driver", "inputs")}
            payload.update(
                entry_path=contract.entry_path, entrypoint=contract.entrypoint, seed=contract.seed
            )
            (temporary / "payload.json").write_text(encoded(payload), encoding="utf8")
            script = temporary / "driver.py"
            script.write_text(HARNESS, encoding="utf8")
            before = snap.revision()
            if left() <= 0:
                raise TimeoutError("shared reference experiment deadline exhausted")
            # Do not call clamp_timeout on the final remaining slice: its
            # minimum must never extend this pair's deadline.
            result = asdict(
                process(
                    [sys.executable, "-B", str(script)], snap.dir, min(clamp_timeout(5.0), left())
                )
            )
            executions.append(
                dict(arm=snap.dir.name, process=result, before=before, after=snap.revision())
            )
            if (
                result.get("timed_out")
                or result.get("truncated")
                or result["exit_code"] != 0
                or executions[-1]["after"] != before
                or workspace.revision() != declaration["revision"]
                or left() <= 0
            ):
                raise ValueError("incomplete process or changed experiment/source files")
            reports.append(
                json.loads(
                    result["output"],
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError("nonfinite observation")
                    ),
                )
            )
        outcome, detail = compare_reports(*reports, contract)
        unchanged = workspace.revision() == declaration["revision"]
    except Exception as error:
        detail = dict(reason=type(error).__name__, message=str(error)[:400])
    finally:
        for snap in snapshots:
            snap.cleanup()
        shutil.rmtree(folder, ignore_errors=True)
    unchanged = workspace.revision() == declaration["revision"]
    if not unchanged:
        outcome = "inconclusive"
        detail = dict(reason="live_source_changed")
    record = {
        k: declaration[k]
        for k in (
            "key",
            "contract",
            "requirements",
            "phase",
            "revision",
            "targets",
            "driver_sha256",
            "inputs_sha256",
        )
    }
    record.update(
        origin="caller_executed_public_reference_pair",
        outcome=outcome,
        detail=detail,
        source_unchanged=unchanged,
        process_count=len(executions),
        wall_s=time.monotonic() - started,
        process_output_bytes=sum(r["process"].get("spool_bytes", 0) for r in executions),
        environment=environment_id(),
        acceptance_proof=False,
        limitation="Model-designed finite experiment;file snapshots add no OS isolation,performance evidence or final acceptance.",
    )
    return record, executions
