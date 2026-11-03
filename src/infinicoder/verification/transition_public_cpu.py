"""Packaged CPU public transition capability;GPU callers supply their own.

Two fresh role processes use the same declared public runtime and bounded
source snapshot. The ordinary shell receives only a readonly client/reference;
it receives no private final script or additional GPU permissions.
"""

from infinicoder._resources import package_file, package_root

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import stat
import sys
import time
import uuid

from infinicoder.runtime.agent_execution import (
    CallerPublicOperation,
    NativeTaskBinding,
    PublicTaskContext,
    UnsupportedCore,
)
from infinicoder.verification.candidate_public_cpu import CandidateCpuExecution
from infinicoder.execution.native_cpu import CpuExecution
from infinicoder.verification.public_check_reports import observed_versions
from infinicoder.verification.public_transition_probe import (
    PublicTransitionContract,
    compare_transition_reports,
    encoded,
    digest,
)
from infinicoder.verification.transition_reports import PublicTransitionCapability
from infinicoder.execution.workspace_tools import run_process


CLIENT = "/tmp/ic-transition-probe.py"
SOCKET = "/tmp/ic-transition-probe.sock"
REFERENCE = "/tmp/ic-transition-reference.py"
PAYLOAD = "/tmp/ic-transition-contract.json"


def _public_reference_bytes(path, workspace, state, mounts, data, previous):
    if (
        not path.is_file()
        or any(p.is_symlink() for p in (path, *path.parents))
        or path.resolve().is_relative_to(workspace)
        or path.resolve().is_relative_to(state)
        or any(path.resolve().is_relative_to(p.resolve()) for p in mounts)
        or path.stat().st_size > 65536
    ):
        raise UnsupportedCore(
            "caller public reference must be a bounded standalone file outside workspace,state and exposed runtime"
        )
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != data["public_reference_identity"]["artifact_sha256"]:
        raise ValueError("public reference artifact SHA differs")
    if previous is not None and previous != raw:
        raise ValueError("public reference changed")
    return raw


class CallerTransitionOperation:
    """Explicit application-owned public runner;no new execution permission.

    The normal CPU shell and run/read disclosure stay the same. The trusted
    application owns isolation,device binding and provenance of its separate
    diagnostic operation. Neither the CLI nor a model can create this callback.
    """

    def __init__(self, contract, reference_script, workspace, state, execution, operation):
        if not isinstance(contract, PublicTransitionContract) or not isinstance(
            execution, TransitionCpuExecution
        ):
            raise ValueError(
                "explicit public contract and ordinary transition CPU execution required"
            )
        self.contract = contract
        self.reference = Path(reference_script).absolute()
        self.workspace = Path(workspace).resolve()
        self.state = Path(state).resolve()
        self.execution = execution
        self.data = contract.manifest()
        self.raw = None
        self.capability = PublicTransitionCapability(contract, operation)
        self._validate_reference()

    def _validate_reference(self):
        self.raw = _public_reference_bytes(
            self.reference, self.workspace, self.state, self.execution.mounts(), self.data, self.raw
        )

    def validate(self, request):
        self._validate_reference()
        self.capability.validate(request)


class TransitionCpuExecution(CandidateCpuExecution):
    def __init__(
        self, scratch, state, *, task_environment="command-private", runtime_library_dirs=()
    ):
        if any(
            not isinstance(p, str)
            or not Path(p).is_absolute()
            or ".." in Path(p).parts
            or str(Path(p)) != p
            for p in runtime_library_dirs
        ):
            raise ValueError("public library directories must be normalized absolute paths")
        directories = tuple(Path(p).absolute() for p in runtime_library_dirs)
        if len(directories) > 8 or len(set(directories)) != len(directories):
            raise ValueError("at most8 distinct public library directories")
        for path in directories:
            if (
                not path.is_absolute()
                or not path.is_dir()
                or path.name not in ("lib", "lib64")
                or len(path.parts) < 4
                or path.parts[1] not in ("opt", "usr")
                or path.resolve().parts[1] not in ("opt", "usr")
            ):
                raise UnsupportedCore(
                    "explicit existing public /opt or /usr lib directory required"
                )
        self.public_runtime_library_dirs = directories
        super().__init__(scratch, state, task_environment=task_environment)
        self.transition_mounts = None
        self.identity = (
            CpuExecution.identity
            + ";public_transition_libraries:"
            + digest([str(p) for p in directories])
        )

    def mounts(self):
        return super().mounts() + [
            p
            for p in self.public_runtime_library_dirs
            if not any(p.is_relative_to(q) for q in super().mounts())
        ]

    def runtime_flags(self):
        return (
            [
                "--setenv",
                "LD_LIBRARY_PATH",
                ":".join(str(p) for p in self.public_runtime_library_dirs),
            ]
            if self.public_runtime_library_dirs
            else []
        )

    def validate(self, request):
        super().validate(request)
        for path in self.public_runtime_library_dirs:
            if Path(request.workspace).resolve().is_relative_to(
                path.resolve()
            ) or path.resolve().is_relative_to(Path(request.workspace).resolve()):
                raise UnsupportedCore(
                    "public library mounts cannot overlap the candidate workspace"
                )

    def builder(self, workspace, command):
        argv = super().builder(workspace, command)
        argv[-len(command) : -len(command)] = self.runtime_flags()
        if self.transition_mounts is not None:
            client, socket, reference = self.transition_mounts
            argv[-len(command) : -len(command)] = [
                "--ro-bind",
                str(client),
                CLIENT,
                "--ro-bind",
                str(socket),
                SOCKET,
                "--ro-bind",
                str(reference),
                REFERENCE,
            ]
        return argv

    @contextmanager
    def bind_transition_service(self, client, socket, reference):
        client, socket, reference = map(lambda p: Path(p).absolute(), (client, socket, reference))
        if (
            self.transition_mounts is not None
            or client.parent != self.state
            or reference.parent != self.state
            or not client.is_file()
            or not reference.is_file()
            or not stat.S_ISSOCK(socket.lstat().st_mode)
            or any(
                p.is_symlink()
                for path in (client, socket, reference)
                for p in (path, *path.parents)
            )
        ):
            raise UnsupportedCore(
                "exact task-owned readonly transition client,reference and socket required"
            )
        self.transition_mounts = client, socket, reference
        try:
            yield
        finally:
            self.transition_mounts = None


class CpuTransitionOperation:
    """Trusted callable adapter;actual candidate execution stays in bwrap."""

    def __init__(self, contract, reference_script, workspace, state, execution):
        if not isinstance(contract, PublicTransitionContract) or not isinstance(
            execution, TransitionCpuExecution
        ):
            raise ValueError("explicit transition contract and CPU execution required")
        self.contract = contract
        self.reference = Path(reference_script).absolute()
        self.workspace = Path(workspace).resolve()
        self.state = Path(state).resolve()
        self.execution = execution
        self.package = package_root().resolve()
        self.data = contract.manifest()
        self.raw = None
        if self.data["public_input_schema"].get("device", "cpu") != "cpu":
            raise UnsupportedCore(
                "packaged operation is CPU-only;GPU requires a caller-provided public operation"
            )
        if tuple(self.data["public_input_schema"].get("runtime_library_dirs", ())) != tuple(
            str(p) for p in execution.public_runtime_library_dirs
        ):
            raise UnsupportedCore("declared public runtime library contract differs from execution")
        self.module_hashes = {
            name: hashlib.sha256((self.package / name).read_bytes()).hexdigest()
            for name in (
                "verification/public_transition_probe.py",
                "verification/torch_transition_adapters.py",
                "verification/transition_runner.py",
                "verification/reference_compare.py",
            )
        }
        self._validate_reference()
        operation = CallerPublicOperation(
            self.data["execution_operation"]["id"],
            self.workspace,
            dict(
                transition_contract_sha256=contract.identity,
                runtime_modules=self.module_hashes,
                public_reference_sha256=hashlib.sha256(self.raw).hexdigest(),
            ),
            "finite public tensor/state differential diagnostics",
            self.data["coverage"],
            30,
            dict(
                execution=execution.identity,
                device="cpu",
                runtime=str(execution.runtime),
                python=sys.version,
                source_scope="declared files copied;dependency closure unknown",
                role_processes="fresh reference and candidate",
            ),
            tuple(self.data["observed_paths"]),
            ("task-local bounded source snapshots and diagnostic artifacts",),
            600,
            self.execute,
        )
        self.capability = PublicTransitionCapability(contract, operation)

    def _validate_reference(self):
        self.raw = _public_reference_bytes(
            self.reference, self.workspace, self.state, self.execution.mounts(), self.data, self.raw
        )

    def validate(self, request):
        self._validate_reference()
        self.capability.validate(request)
        for name, expected_digest in self.module_hashes.items():
            if hashlib.sha256((self.package / name).read_bytes()).hexdigest() != expected_digest:
                raise ValueError("transition runtime changed")

    def _copy_sources(self, directory, versions, deadline):
        directory.mkdir()
        total = 0
        for name in self.data["observed_paths"]:
            if time.monotonic() >= deadline:
                raise TimeoutError("source copy deadline")
            source = self.workspace / name
            target = directory / name
            if any(p.is_symlink() for p in (source, *source.parents)) or not source.is_file():
                raise ValueError("source snapshot cannot follow symlinks")
            size = source.stat().st_size
            total += size
            if size > 1024**2 or total > 8 * 1024**2:
                raise ValueError("public snapshot source bound;ordinary workflow preserved")
            raw = source.read_bytes()
            if hashlib.sha256(raw).hexdigest() != versions[name].get("sha256"):
                raise ValueError("source changed while copying")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)

    def execute(self, deadline):
        self._validate_reference()
        versions = observed_versions(self.workspace, tuple(self.data["observed_paths"]), deadline)
        if any(v["kind"] != "file" for v in versions.values()):
            raise ValueError("public candidate source version unknown")
        root = self.state / "transition-runs"
        root.mkdir(exist_ok=True)
        folder = root / uuid.uuid4().hex
        folder.mkdir()
        workspace = folder / "workspace"
        self._copy_sources(workspace, versions, deadline)
        frozen = self.state / "transition-reference.py"
        if frozen.read_bytes() != self.raw:
            raise ValueError("frozen public reference changed")
        payload = dict(
            contract=self.data,
            source_versions=versions,
            workspace="/workspace",
            public_reference_path=REFERENCE,
        )
        pp = folder / "payload.json"
        pp.write_bytes((encoded(payload) + "\n").encode())
        pp.chmod(0o400)
        python = self.data["public_input_schema"].get("python_executable", "python")
        if python != "python":
            path = Path(python)
            if (
                not path.is_absolute()
                or not path.is_file()
                or not any(
                    path.resolve().is_relative_to(p.resolve()) for p in self.execution.mounts()
                )
            ):
                raise UnsupportedCore(
                    "public Python must already be in the declared readonly runtime mounts"
                )
        roles = {}
        processes = []
        for role in ("reference", "candidate"):
            if time.monotonic() >= deadline:
                raise TimeoutError("shared role-process deadline exhausted")
            bootstrap = "import runpy,sys;sys.path.insert(0,'/tmp/ic-transition-runtime');runpy.run_module('infinicoder.verification.transition_runner',run_name='__main__')"
            command = [python, "-I", "-B", "-c", bootstrap, "--payload", PAYLOAD, "--role", role]
            # Bypass ordinary persistent tmp and both public sockets. Only the
            # frozen contract,public reference and public runtime are mounted.
            argv = CpuExecution.builder(self.execution, workspace, command)
            argv[-len(command) : -len(command)] = [
                "--ro-bind",
                str(pp),
                PAYLOAD,
                "--ro-bind",
                str(frozen),
                REFERENCE,
                "--dir",
                "/tmp/ic-transition-runtime",
                "--ro-bind",
                str(self.package),
                "/tmp/ic-transition-runtime/infinicoder",
                *self.execution.runtime_flags(),
            ]
            process = run_process(
                argv,
                workspace,
                timeout=max(0, deadline - time.monotonic()),
                max_output=524288,
                spool_cap=1024**2,
            )
            (folder / (role + "-process.json")).write_bytes(
                (encoded(process.__dict__) + "\n").encode()
            )
            processes.append(process)
            lines = [
                line[len("IC_TRANSITION_REPORT:") :]
                for line in process.output.splitlines()
                if line.startswith("IC_TRANSITION_REPORT:")
            ]
            if process.exit_code != 0 or process.timed_out or process.truncated or len(lines) != 1:
                roles[role] = dict(
                    contract_sha256=self.contract.identity,
                    role=role,
                    calls=[],
                    environment_error="isolated role runner failed;read process artifact",
                )
            else:
                roles[role] = json.loads(lines[0])
            (folder / (role + "-report.json")).write_bytes((encoded(roles[role]) + "\n").encode())
        result = compare_transition_reports(
            self.contract, roles["reference"], roles["candidate"], versions
        )
        if (
            observed_versions(self.workspace, tuple(self.data["observed_paths"]), deadline)
            != versions
        ):
            result.update(status="unknown", reason="live_sources_changed_during_public_probe")
        (folder / "comparison.json").write_bytes((encoded(result) + "\n").encode())
        summary = dict(
            passed=int(result["status"] == "observed_values_match"),
            failed=int(result["status"] in ("counterexample", "candidate_error")),
            error=int(result["status"] in ("unknown", "env_error")),
            transition_status=result["status"],
        )
        return dict(
            transition_report=result,
            summary=summary,
            process=dict(
                exit_code=0 if summary["passed"] else 1,
                timed_out=any(p.timed_out for p in processes),
                truncated=any(p.truncated for p in processes),
            ),
            role_reports=roles,
            artifacts=str(folder),
            independent_final_used=False,
        )


class TransitionAcceptance:
    """Preserve the existing final/public check;add the same public tool to both arms."""

    def __init__(self, base, transition, execution, context):
        self.base, self.transition, self.execution, self.context = (
            base,
            transition,
            execution,
            context,
        )
        self.identity = base.identity + ";caller_public_transition_material_v1"

    def validate(self, request):
        self.base.validate(request)
        self.transition.validate(request)
        if (
            getattr(self.base, "script", None) is not None
            and Path(self.base.script).resolve() == self.transition.reference.resolve()
        ):
            raise UnsupportedCore("private final script cannot be the public transition reference")

    @contextmanager
    def public_command(self, request, deadline):
        self.validate(request)
        with self.base.public_command(request, deadline) as original:
            state = Path(request.state_dir)
            reference = state / "transition-reference.py"
            client = state / "transition-client.py"
            with reference.open("xb") as stream:
                stream.write(self.transition.raw)
            reference.chmod(0o400)
            raw = (
                package_file("verification/public_check_client.py")
                .read_bytes()
                .replace(b"/tmp/ic-public-check.sock", SOCKET.encode())
            )
            with client.open("xb") as stream:
                stream.write(raw)
            client.chmod(0o400)
            with self.transition.capability.bind_reports(
                request, deadline, state / "transition-reports"
            ) as store:
                with (
                    store.serve() as socket,
                    self.execution.bind_transition_service(client, socket, reference),
                ):
                    old_operation = (
                        original.operation if isinstance(original, PublicTaskContext) else None
                    )
                    binding = (
                        original.binding if isinstance(original, PublicTaskContext) else original
                    )
                    old_command = "" if isinstance(binding, NativeTaskBinding) else binding + "\n"
                    disclosure = (
                        "Public transition diagnostic (development feedback,not final acceptance): python -B "
                        + CLIENT
                        + " run . "
                        "It compares actual candidate calls with the caller's frozen public reference on the declared small recipe;at most two source revisions. "
                        "Read without rerunning: python -B "
                        + CLIENT
                        + " read REPORT_ID --offset 0 --limit 2000 . "
                        "The public reference is readable at "
                        + REFERENCE
                        + ". Public contract: "
                        + encoded(self.transition.data)
                        + "\n"
                    )
                    yield PublicTaskContext(old_command + disclosure, old_operation)

    def verify(self, request):
        return self.base.verify(request)
