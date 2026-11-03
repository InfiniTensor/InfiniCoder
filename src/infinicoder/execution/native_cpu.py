"""Packaged CPU namespace and caller-owned independent final-check adapter."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

from infinicoder.runtime.agent_execution import UnsupportedCore, NativeTaskBinding, FinalAcceptance
from infinicoder.execution.native_sandbox import NativeSandboxExecution
from infinicoder.verification.verification import Snapshot, parse_test_summary, outcome_acceptable
from infinicoder.execution.workspace_tools import run_process


class CpuExecution(NativeSandboxExecution):
    """Only system/runtime files and /workspace are exposed;no host-root bind."""

    identity = "packaged_linux_CPU_bwrap_v1;private_network_pid;explicit_runtime_mounts"

    def __init__(
        self, scratch, *, trials=False, task_environment="command-private", python_environment=None
    ):
        self.program = shutil.which("bwrap")
        self.runtime = Path(python_environment or sys.prefix).resolve()
        self.explicit_runtime = python_environment is not None
        if python_environment is not None and (
            not (self.runtime / "pyvenv.cfg").is_file()
            or not (self.runtime / "bin/python").is_file()
        ):
            raise ValueError("python_environment must be an existing Linux virtual environment")
        self.checked = None
        super().__init__(
            self.builder, scratch, snapshot_trials=trials, task_environment=task_environment
        )

    def mounts(self):
        values = [Path(p) for p in ("/usr", "/bin", "/sbin", "/lib", "/lib64") if Path(p).exists()]
        if not any(self.runtime.is_relative_to(p) for p in values):
            values.append(self.runtime)
        return values

    def builder(self, workspace, command):
        if self.program is None:
            raise UnsupportedCore("install bubblewrap;no host-shell fallback")
        argv = [
            self.program,
            "--unshare-user",
            "--unshare-pid",
            "--unshare-net",
            "--unshare-ipc",
            "--unshare-uts",
            "--die-with-parent",
            "--new-session",
            "--clearenv",
            "--tmpfs",
            "/tmp",
            "--tmpfs",
            "/home",
            "--tmpfs",
            "/proc",
            "--dev",
            "/dev",
            "--dir",
            "/etc",
        ]
        for path in self.mounts():
            argv += ["--ro-bind", str(path), str(path)]
        for path in ("/etc/ld.so.cache", "/etc/alternatives"):
            if Path(path).exists():
                argv += ["--ro-bind", path, path]
        argv += [
            "--bind",
            str(workspace),
            "/workspace",
            "--chdir",
            "/workspace",
            "--dir",
            "/tmp/home",
            "--setenv",
            "HOME",
            "/tmp/home",
            "--setenv",
            "USER",
            "worker",
            "--setenv",
            "PATH",
            str(self.runtime / "bin") + ":/usr/bin:/bin",
            "--setenv",
            "LANG",
            "C.UTF-8",
            "--setenv",
            "PYTHONDONTWRITEBYTECODE",
            "1",
            "--setenv",
            "CUDA_VISIBLE_DEVICES",
            "",
        ]
        return argv + list(command)

    def validate(self, request):
        if sys.platform != "linux":
            raise UnsupportedCore("packaged CPU profile requires Linux and bubblewrap")
        super().validate(request)
        workspace = Path(request.workspace).resolve()
        if self.explicit_runtime and (
            self.runtime.is_relative_to(workspace) or workspace.is_relative_to(self.runtime)
        ):
            raise UnsupportedCore(
                "project Python environment must be separate from the editable workspace"
            )
        if self.program is None:
            raise UnsupportedCore("install bubblewrap;no host-shell fallback")
        if not Path(request.state_dir).parent.is_dir():
            raise UnsupportedCore(
                "state parent must exist;choose a fresh run directory outside workspace"
            )
        for private in (Path(request.state_dir).resolve(), self.scratch.resolve()):
            if any(private.is_relative_to(path.resolve()) for path in self.mounts()):
                raise UnsupportedCore(
                    "state/scratch must be outside the read-only system/runtime mounts"
                )
        if self.checked != Path(request.workspace).resolve():
            probe = run_process(
                self.builder(request.workspace, ["/bin/true"]),
                request.workspace,
                timeout=5,
                max_output=2000,
                spool_cap=10000,
            )
            if probe.exit_code != 0 or probe.timed_out:
                raise UnsupportedCore(
                    "CPU namespace unavailable;no host-shell fallback: " + probe.output[-300:]
                )
            self.checked = Path(request.workspace).resolve()


class ScriptAcceptance:
    """Freeze a trusted caller script,then check a separate final snapshot."""

    identity = "caller_frozen_python_check_v1;external_snapshot;structured_test_result"

    def __init__(self, script, execution, *, timeout=180):
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not 0 < timeout <= 3600
        ):
            raise ValueError("final_timeout must be positive and at most3600 seconds")
        self.script = Path(script).absolute()
        self.execution = execution
        self.timeout = timeout
        self.raw = None

    def validate(self, request):
        if request.spec.checks:
            raise UnsupportedCore(
                "CPU profile uses explicit check_script;use a custom acceptance adapter for TaskSpec.checks"
            )
        workspace = Path(request.workspace).resolve()
        state = Path(request.state_dir).resolve()
        if (
            any(p.is_symlink() for p in (self.script, *self.script.parents))
            or not self.script.is_file()
            or self.script.resolve().is_relative_to(workspace)
            or self.script.resolve().is_relative_to(state)
        ):
            raise UnsupportedCore(
                "trusted regular check_script must be outside workspace and fresh state"
            )
        if any(
            self.script.resolve().is_relative_to(path.resolve()) for path in self.execution.mounts()
        ):
            raise UnsupportedCore(
                "trusted check_script must be outside exposed system/runtime mounts"
            )
        if self.script.stat().st_size > 1024**2:
            raise UnsupportedCore("standalone check_script must be at most1MiB")
        if self.raw is None:
            self.raw = self.script.read_bytes()

    @contextmanager
    def public_command(self, request, deadline):
        self.validate(request)
        frozen = Path(request.state_dir) / "caller-final-check.py"
        with frozen.open("xb") as stream:
            stream.write(self.raw)
        frozen.chmod(0o400)
        yield NativeTaskBinding(
            request.spec.task, "verbatim_task;caller_final_script_is_not_public_test"
        )

    def verify(self, request, *, deadline=None):
        started = time.monotonic()
        state = Path(request.state_dir)
        frozen = state / "caller-final-check.py"
        if frozen.read_bytes() != self.raw:
            raise ValueError("frozen caller check changed")
        folder = state / "independent-final"
        folder.mkdir(exist_ok=False)
        snapshot = Snapshot(request.workspace, folder / "workspace")
        before = snapshot.revision()
        command = [
            "python",
            "-B",
            "-c",
            "import runpy,sys;sys.path.insert(0,'/workspace');runpy.run_path('/tmp/ic-final-check.py',run_name='__main__')",
        ]
        argv = self.execution.builder(snapshot.dir, command)
        argv[-len(command) : -len(command)] = ["--ro-bind", str(frozen), "/tmp/ic-final-check.py"]
        remaining = self.timeout - (time.monotonic() - started)
        if deadline is not None:
            remaining = min(remaining, deadline - time.monotonic())
        result = run_process(
            argv, snapshot.dir, timeout=max(0, remaining), max_output=16000, spool_cap=4 * 1024**2
        )
        summary = parse_test_summary(result.output)
        acceptable, reason = outcome_acceptable(summary)
        if summary.get("format") == "exit-code-only":
            acceptable = False
            reason = "trusted check must report nonzero structured test counts"
        missing = [p for p in request.spec.required_artifacts if not (snapshot.dir / p).is_file()]
        passed = (
            result.exit_code == 0
            and not result.timed_out
            and not result.truncated
            and acceptable
            and not missing
        )
        reference = folder / "result.json"
        reference.write_text(
            json.dumps(
                dict(
                    passed=passed,
                    check_sha256=hashlib.sha256(self.raw).hexdigest(),
                    source_revision_before=before,
                    source_revision_after=snapshot.revision(),
                    identity_scope="snapshot content;external runtime dependencies separately identified by CPU profile",
                    summary=summary,
                    reason=reason,
                    missing_required_artifacts=missing,
                    process=result.__dict__,
                    wall_s=time.monotonic() - started,
                    independent_of_agent_submission=True,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf8",
        )
        return FinalAcceptance(bool(passed), str(reference))
