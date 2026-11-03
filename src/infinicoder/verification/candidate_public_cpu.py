"""Explicit public CPU checker, separate from the private final-check script."""

from infinicoder._resources import package_file

from contextlib import contextmanager
from dataclasses import replace
import hashlib
from pathlib import Path
import stat
import sys
import time

from infinicoder.runtime.agent_execution import (
    CallerPublicOperation,
    PublicTaskContext,
    UnsupportedCore,
)
from infinicoder.verification.candidate_public_check import (
    CandidatePublicCheck,
    PublicCandidateVerdict,
)
from infinicoder.execution.native_cpu import CpuExecution, ScriptAcceptance
from infinicoder.verification.verification import parse_test_summary, outcome_acceptable
from infinicoder.execution.workspace_tools import run_process

PUBLIC_SCRIPT_PATH = "/tmp/ic-public-check.py"
PUBLIC_TESTS_PATH = "/tmp/ic-public-tests.py"
PUBLIC_SOCKET_PATH = "/tmp/ic-public-check.sock"


class CandidateCpuExecution(CpuExecution):
    def __init__(
        self, scratch, state, *, task_environment="command-private", python_environment=None
    ):
        super().__init__(
            scratch, task_environment=task_environment, python_environment=python_environment
        )
        self.state = Path(state).resolve()
        self.public_file = None
        self.public_socket = None

    def builder(self, workspace, command):
        argv = super().builder(workspace, command)
        if self.public_file is not None:
            argv[-len(command) : -len(command)] = [
                "--ro-bind",
                str(self.public_file),
                PUBLIC_SCRIPT_PATH,
            ]
        if self.public_socket is not None:
            argv[-len(command) : -len(command)] = [
                "--ro-bind",
                str(self.public_socket),
                PUBLIC_SOCKET_PATH,
            ]
        return argv

    @contextmanager
    def bind_public_script(self, path):
        path = Path(path).absolute()
        if (
            self.public_file is not None
            or path != self.state / "caller-public-check.py"
            or not path.is_file()
            or any(p.is_symlink() for p in (path, *path.parents))
        ):
            raise UnsupportedCore("exact caller-frozen public script required")
        self.public_file = path
        try:
            yield
        finally:
            self.public_file = None

    @contextmanager
    def bind_public_service(self, client, socket):
        client, socket = Path(client).absolute(), Path(socket).absolute()
        if (
            self.public_file is not None
            or self.public_socket is not None
            or client != self.state / "public-check-client.py"
            or not client.is_file()
            or not stat.S_ISSOCK(socket.lstat().st_mode)
            or any(p.is_symlink() for v in (client, socket) for p in (v, *v.parents))
        ):
            raise UnsupportedCore("exact task-owned frozen public client and socket required")
        self.public_file, self.public_socket = client, socket
        try:
            yield
        finally:
            self.public_file = self.public_socket = None


def decode_public_cpu_result(raw):
    """Narrow decoder of this trusted checker process,not a candidate claim."""
    result, summary = raw["process"], raw["summary"]
    if result["timed_out"] or result["truncated"]:
        return PublicCandidateVerdict("unknown")
    if (
        summary.get("format") == "exit-code-only"
        or summary.get("collection_error")
        or summary.get("zero_tests")
        or summary.get("error", 0)
    ):
        return PublicCandidateVerdict("env_error")
    acceptable, _ = outcome_acceptable(summary)
    if result["exit_code"] == 0 and acceptable:
        return PublicCandidateVerdict("public_pass")
    if result["exit_code"] == 1 and summary.get("failed", 0) > 0:
        return PublicCandidateVerdict(
            "public_fail", result["output"].encode()[:8000].decode(errors="ignore")
        )
    return PublicCandidateVerdict("env_error" if result["exit_code"] else "unknown")


class CandidateScriptAcceptance(ScriptAcceptance):
    """Both public API arms can expose the same caller-selected public script."""

    def __init__(
        self,
        script,
        execution,
        *,
        public_script,
        workspace,
        state,
        coverage,
        observed_paths,
        public_timeout=60,
        timeout=180,
        feedback="direct",
    ):
        super().__init__(script, execution, timeout=timeout)
        if feedback not in ("direct", "reports"):
            raise ValueError("explicit direct/reports public feedback required")
        self.feedback = feedback
        if execution.task_environment == "task-private" and feedback == "direct":
            raise UnsupportedCore(
                "direct public scripts share ordinary task tmp;use a separately served public operation for task-private development"
            )
        self.public_script = Path(public_script).absolute()
        self.public_raw = None
        self.state = Path(state).resolve()
        self.workspace = Path(workspace).resolve()
        self.coverage = coverage
        self.observed_paths = tuple(observed_paths)
        self.public_timeout = public_timeout
        # Regular-file checks are repeated by validate before the caller opens.
        self._validate_public()
        self.public_check = CandidatePublicCheck(
            CallerPublicOperation(
                "caller_frozen_CPU_public_script_v1",
                self.workspace,
                dict(
                    script_sha256=hashlib.sha256(self.public_raw).hexdigest(),
                    decoder="structured_counts_exit_v1",
                ),
                "explicit caller public CPU correctness tests",
                coverage,
                public_timeout,
                dict(
                    execution=execution.identity,
                    runtime=str(execution.runtime),
                    python=sys.version,
                    feedback=feedback,
                    checker_temporary_environment="fresh original CPU builder"
                    if feedback == "reports"
                    else "ordinary command-private script",
                ),
                self.observed_paths,
                ("task-local test artifacts;any workspace change invalidates selection",),
                2048,
                self._execute,
            ),
            decode_public_cpu_result,
            "trusted_CPU_counts_and_process_exit_v1",
        )

    def _validate_public(self):
        path = self.public_script
        if (
            not path.is_file()
            or any(p.is_symlink() for p in (path, *path.parents))
            or path.resolve().is_relative_to(self.workspace)
            or path.resolve().is_relative_to(self.state)
            or path.resolve() == self.script.resolve()
            or any(path.resolve().is_relative_to(p.resolve()) for p in self.execution.mounts())
            or path.stat().st_size > 1024**2
        ):
            raise UnsupportedCore(
                "distinct standalone caller public script outside workspace,state and runtime mounts required"
            )
        raw = path.read_bytes()
        if self.public_raw is None:
            self.public_raw = raw
        elif raw != self.public_raw:
            raise UnsupportedCore("caller public script changed before execution")

    def validate(self, request):
        super().validate(request)
        self._validate_public()

    @contextmanager
    def public_command(self, request, deadline):
        with super().public_command(request, deadline):
            frozen = self.state / "caller-public-check.py"
            with frozen.open("xb") as stream:
                stream.write(self.public_raw)
            frozen.chmod(0o400)
            if self.feedback == "reports":
                from infinicoder.verification.public_check_reports import PublicCheckReports
                from infinicoder.memory.native_evidence_route import NativeEvidenceRoute

                observer = request.context_adapter
                operation = self.public_check.operation
                post_edit = (
                    type(observer) is NativeEvidenceRoute and observer.post_edit_budget is not None
                )
                memory = type(observer) is NativeEvidenceRoute and observer.task_working_memory
                if post_edit or memory:
                    original = operation

                    def notify(cutoff):
                        token = observer.begin_check(
                            judge_identity=original.implementation_identity,
                            scope=original.public_scope,
                            coverage=original.public_coverage,
                        )
                        result = None
                        error = None
                        try:
                            result = original.executor(cutoff)
                            return result
                        except BaseException as caught:
                            error = caught
                            raise
                        finally:
                            observer.end_check(token, report=result, error=error)

                    operation = replace(operation, executor=notify)
                store = PublicCheckReports(
                    operation,
                    self.state / "public-reports",
                    request,
                    deadline,
                    on_result=observer.observe_public_report if memory else None,
                )
                exposed = (
                    replace(operation, executor=lambda cutoff: store.run(absolute_deadline=cutoff))
                    if post_edit
                    else operation
                )
                client = self.state / "public-check-client.py"
                with client.open("xb") as stream:
                    stream.write(package_file("verification/public_check_client.py").read_bytes())
                client.chmod(0o400)
                with store.serve() as socket, self.execution.bind_public_service(client, socket):
                    command = (
                        "python -B " + PUBLIC_SCRIPT_PATH + " run\n"
                        "This runs caller public development tests in a separate fresh environment and returns report_id;it is not final acceptance. "
                        "Read that result without running tests again: python -B "
                        + PUBLIC_SCRIPT_PATH
                        + " read REPORT_ID --offset 0 --limit 2000 . Continue with next_offset. "
                        "After source edits,the previous result may be stale;run explicitly to test the new candidate."
                    )
                    yield PublicTaskContext(command, exposed)
                return
            with self.execution.bind_public_script(frozen):
                command = (
                    "python -B -c \"import runpy,sys;sys.path.insert(0,'/workspace');runpy.run_path('"
                    + PUBLIC_SCRIPT_PATH
                    + "',run_name='__main__')\""
                )
                yield PublicTaskContext(command, self.public_check.operation)

    def _execute(self, deadline):
        frozen = self.state / "caller-public-check.py"
        if frozen.read_bytes() != self.public_raw:
            raise ValueError("frozen public script changed")
        script_path = PUBLIC_TESTS_PATH if self.feedback == "reports" else PUBLIC_SCRIPT_PATH
        command = [
            "python",
            "-B",
            "-c",
            "import runpy,sys;sys.path.insert(0,'/workspace');runpy.run_path('"
            + script_path
            + "',run_name='__main__')",
        ]
        if self.feedback == "reports":
            # Use the original builder,not ordinary execute: no persistent
            # tmp/HOME/cache,public client or socket enters this checker.
            argv = CpuExecution.builder(self.execution, self.workspace, command)
            argv[-len(command) : -len(command)] = ["--ro-bind", str(frozen), PUBLIC_TESTS_PATH]
        else:
            argv = self.execution.builder(self.workspace, command)
        result = run_process(
            argv,
            self.workspace,
            timeout=max(0, deadline - time.monotonic()),
            max_output=4 * 1024**2 if self.feedback == "reports" else 16000,
            spool_cap=4 * 1024**2,
        )
        return dict(
            process=result.__dict__,
            summary=parse_test_summary(result.output),
            public_script_sha256=hashlib.sha256(self.public_raw).hexdigest(),
            independent_final_used=False,
        )
