"""Explicit CPU diagnosis in the original namespace,with read-only candidate."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.execution.candidate_sandbox import SnapshotRunner
from infinicoder.execution.workspace_tools import run_process


class ReadonlyCpuDiagnostic:
    identity = "original_bwrap_readonly_CPU_diagnostic_v1;private_pid_net;no_candidate_selection"

    def __init__(self, execution, request, deadline, *, directory=None):
        if sys.platform != "linux":
            raise UnsupportedCore("Linux namespace required for readonly CPU diagnostics")
        self.execution, self.workspace, self.deadline = (
            execution,
            Path(request.workspace).resolve(),
            deadline,
        )
        state = Path(request.state_dir).resolve()
        self.directory = (
            Path(directory) if directory is not None else state / "contract-review" / "diagnostic"
        )
        if not self.directory.is_absolute() or not self.directory.resolve().is_relative_to(state):
            raise UnsupportedCore("diagnostic records must stay inside the caller state")
        # Reuse the existing namespace admission without enabling snapshot trials.
        self._argv(["/bin/true"])

    def _argv(self, command):
        argv = SnapshotRunner._argv(self, self.workspace, command)
        # These are explicit mounts of the packaged public-operation adapter.
        # A diagnostic is not authorized to reach that live Unix capability.
        for name, target in [
            ("public_file", "/tmp/ic-public-check.py"),
            ("public_socket", "/tmp/ic-public-check.sock"),
        ]:
            path = getattr(self.execution, name, None)
            if path is None:
                continue
            positions = [
                i
                for i in range(len(argv) - 2)
                if argv[i : i + 3] == ["--ro-bind", str(path), target]
            ]
            if len(positions) != 1:
                raise UnsupportedCore("declared public client/socket namespace changed")
            del argv[positions[0] : positions[0] + 3]
        positions = [
            i
            for i in range(len(argv) - 2)
            if argv[i : i + 3] == ["--bind", str(self.workspace), "/workspace"]
        ]
        if len(positions) != 1:
            raise UnsupportedCore("one exact live workspace binding required")
        argv[positions[0]] = "--ro-bind"
        return argv

    def run(self, script, *, absolute_deadline):
        cutoff = min(self.deadline, absolute_deadline)
        left = cutoff - time.monotonic()
        if left <= 0:
            raise TimeoutError("CPU diagnostic deadline expired")
        folder = self.directory / uuid.uuid4().hex
        folder.mkdir(mode=0o700, parents=True, exist_ok=False)
        source = folder / "probe.py"
        source.write_bytes(script.encode())
        source.chmod(0o400)
        scratch = folder / "scratch"
        scratch.mkdir(mode=0o700)
        runtime = getattr(self.execution, "runtime", None)
        if runtime is None:
            # Frozen custom callers bind their Python explicitly;the caller's
            # existing clean PATH resolves it inside the original namespace.
            command = [
                "/bin/bash",
                "-c",
                "exec python -B -c \"import runpy,sys;sys.path.insert(0,'/workspace');runpy.run_path('/tmp/ic-contract-probe.py',run_name='__main__')\"",
            ]
        else:
            command = [
                str(Path(runtime) / "bin/python"),
                "-B",
                "-c",
                "import runpy,sys;sys.path.insert(0,'/workspace');runpy.run_path('/tmp/ic-contract-probe.py',run_name='__main__')",
            ]
        argv = self._argv(command)
        argv[-len(command) : -len(command)] = [
            "--ro-bind",
            str(source),
            "/tmp/ic-contract-probe.py",
            "--bind",
            str(scratch),
            "/tmp/task-scratch",
            "--setenv",
            "PYTHONDONTWRITEBYTECODE",
            "1",
            "--setenv",
            "TMPDIR",
            "/tmp/task-scratch",
        ]
        left = cutoff - time.monotonic()
        if left <= 0:
            raise TimeoutError("CPU diagnostic preparation exhausted its deadline")
        environment = {k: v for k, v in self.execution.env.items() if not k.startswith("IC_CODE_")}
        environment["CUDA_VISIBLE_DEVICES"] = ""
        result = run_process(
            argv, scratch, timeout=left, max_output=6000, spool_cap=4 * 1024**2, env=environment
        )
        value = dict(
            authority="ordinary_readonly_execution_observation;not_final_acceptance",
            environment_identity=self.identity,
            script_sha256=hashlib.sha256(script.encode()).hexdigest(),
            candidate_source_readonly=True,
            input_applicability="unverified_probe",
            failure_implicates_candidate=False,
            process=asdict(result),
            classification="execution_timeout"
            if result.timed_out
            else "output_limit"
            if result.truncated
            else "script_finished;semantics_unverified",
        )
        (folder / "result.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf8"
        )
        return value
