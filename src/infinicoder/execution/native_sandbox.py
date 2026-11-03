"""Shared caller-owned sandbox contract for native agents and references.

The caller supplies its frozen namespace builder. This adapter adds only a
fresh, task-private persistent scratch mount and a shared command time limit.
It does not mount a verifier, change networking or export scratch as source.
"""

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import time

from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.execution.workspace_tools import run_process


SCRATCH_PATH = "/tmp/task-scratch"
CANDIDATE_SOURCE_PATH = "/tmp/ic-candidate-A"
CANDIDATE_EVIDENCE_PATH = "/tmp/ic-candidate-evidence"

SANDBOX_NOTICE = (
    "Working directory: /workspace. Each command has a new private /tmp and HOME; "
    "files there do not persist between commands, except /tmp/task-scratch. "
    "/tmp/task-scratch is private to this task and persists between commands. "
    "Use /tmp/task-scratch for local experiments and temporary "
    "downloads. Only the original candidate workspace is exported and independently "
    "graded; scratch files are not candidate source. Each ordinary command is limited "
    "to the smaller of 1800 seconds and the remaining task time. Original public "
    "checks may have a shorter limit. Task scope and public interfaces still apply."
)

PERSISTENT_SANDBOX_NOTICE = (
    "Working directory: /workspace. Each command uses a fresh shell and isolated namespace. "
    "/tmp, HOME=/tmp/home, and HOME/.cache are private to this task and persist across ordinary commands. "
    "Use them for temporary tests, local experiments and build caches. Shell exports and cd do not persist. "
    "/tmp/task-scratch also remains task-private. Task temporary files are cleaned when the run ends; "
    "only the original candidate workspace is exported and independently graded. "
    "Independent final and caller checks launched outside ordinary commands use separate fresh temporary environments. "
    "A public script invoked inside an ordinary command shares this task temporary environment; "
    "its output is development feedback,not independent acceptance. "
    "Ordinary commands remain CPU-only and limited to the smaller of1800 seconds and remaining task time. "
    "Original task scope, interfaces and shorter public-check deadlines still apply."
)


class NativeSandboxExecution:
    identity = "native_sandbox_v1_scratch_1800_remaining"
    command_limit = 1800
    candidate_branch_capability = "fresh_scratch_readonly_historical_A_v1"

    def __init__(
        self,
        sandbox_argv,
        scratch,
        *,
        env=None,
        snapshot_trials=False,
        trial_generated_paths=(),
        task_environment="command-private",
    ):
        self.sandbox_argv = sandbox_argv
        self.scratch = Path(scratch)
        self.env = dict(env or {})
        if type(snapshot_trials) is not bool:
            raise ValueError("snapshot_trials must be explicitly boolean")
        self.snapshot_trials = snapshot_trials
        self.trial_generated_paths = tuple(trial_generated_paths)
        if task_environment not in ("command-private", "task-private"):
            raise ValueError("explicit command-private/task-private environment required")
        self.task_environment = task_environment
        if task_environment == "task-private":
            self.identity += ";task_private_tmp_home_cache_v1"

    def snapshot_runner(self, request, deadline):
        from infinicoder.execution.candidate_sandbox import SnapshotRunner

        return SnapshotRunner(
            self,
            workspace=request.workspace,
            directory=Path(request.state_dir) / "shell-code-tools" / "trials",
            deadline=deadline,
        )

    def validate(self, request):
        if not callable(self.sandbox_argv) or not self.scratch.is_absolute():
            raise UnsupportedCore("explicit frozen sandbox builder and absolute scratch required")
        workspace = Path(request.workspace).resolve()
        scratch = self.scratch.resolve()
        if (
            scratch.is_relative_to(workspace)
            or workspace.is_relative_to(scratch)
            or self.scratch.is_symlink()
            or any(p.is_symlink() for p in self.scratch.parents)
        ):
            raise UnsupportedCore(
                "scratch must be separate from candidate source without symlink parents"
            )
        if self.scratch.exists() or not self.scratch.parent.is_dir():
            raise UnsupportedCore("scratch must be a fresh task-owned path with an existing parent")
        if self.task_environment == "task-private":
            from infinicoder.execution.task_private_files import TaskPrivateFiles

            command = ["/bin/true"]
            TaskPrivateFiles.validate_builder(list(self.sandbox_argv(workspace, command)), command)

    @contextmanager
    def open(self, request, deadline):
        from infinicoder.runtime.native_environment import NativeEnvironment

        self.validate(request)
        self.scratch.mkdir(mode=0o700)
        scratch = self.scratch.resolve()
        workspace = Path(request.workspace).resolve()
        builder = self.sandbox_argv
        log = Path(request.state_dir) / "native-execution.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        task_files = {}

        class Environment(NativeEnvironment):
            code_client = None
            history_archive = None
            candidate_branch = None
            behavior_plan_evidence = None

            def implementation_workspace_binding(inner):
                """Describe the same live source mount used by ordinary commands.

                This is controller metadata,not an agent tool or write grant.
                Unknown builders cannot authorize a provider's host-side write.
                """
                probe = ["/bin/true"]
                argv = list(builder(workspace, probe))
                if not argv or Path(argv[0]).name != "bwrap" or argv[-1:] != probe:
                    raise UnsupportedCore(
                        "implementation provider needs a known live workspace namespace"
                    )
                bindings = [
                    argv[i : i + 3]
                    for i, part in enumerate(argv[:-2])
                    if part in ("--bind", "--ro-bind", "--bind-try", "--ro-bind-try")
                    and argv[i + 2] == "/workspace"
                ]
                if bindings != [["--bind", str(workspace), "/workspace"]]:
                    raise UnsupportedCore("implementation provider workspace mount is ambiguous")
                if not any(
                    argv[i : i + 2] == ["--chdir", "/workspace"] for i in range(len(argv) - 1)
                ):
                    raise UnsupportedCore(
                        "implementation provider model working directory is unknown"
                    )
                return dict(
                    root=str(workspace),
                    model_root="/workspace",
                    execution_identity=self.identity,
                    namespace_sha256=hashlib.sha256(
                        json.dumps(argv, separators=(",", ":")).encode()
                    ).hexdigest(),
                )

            @contextmanager
            def bind_behavior_plan_evidence(inner, directory):
                from infinicoder.planning.behavior_plan import MODEL_EVIDENCE

                original = Path(directory).absolute()
                expected = (Path(request.state_dir) / "behavior-plan" / "public-evidence").resolve()
                if (
                    inner.behavior_plan_evidence is not None
                    or original.resolve() != expected
                    or not original.is_dir()
                    or any(p.is_symlink() for p in (original, *original.parents))
                ):
                    raise UnsupportedCore("exact task-owned behavior evidence directory required")
                inner.behavior_plan_evidence = expected
                try:
                    yield dict(
                        path=MODEL_EVIDENCE, identity="readonly_task_behavior_observations_v1"
                    )
                finally:
                    inner.behavior_plan_evidence = None

            @contextmanager
            def bind_candidate_branch(inner, source, evidence):
                if self.task_environment == "task-private":
                    raise UnsupportedCore(
                        "task-private development profile is one original run;candidate search remains separate"
                    )
                state = Path(request.state_dir).resolve()
                source, evidence = Path(source).absolute(), Path(evidence).absolute()
                if (
                    inner.candidate_branch is not None
                    or inner.code_client is not None
                    or inner.history_archive is not None
                    or source != state / "candidate-checkpoints" / "A" / "tree"
                    or evidence != state / "candidate-evidence"
                    or any(
                        not p.is_dir() or any(q.is_symlink() for q in (p, *p.parents))
                        for p in (source, evidence)
                    )
                ):
                    raise UnsupportedCore(
                        "exact readonly historical A and public evidence required"
                    )
                fresh = state / "candidate-B-scratch"
                fresh.mkdir(mode=0o700, exist_ok=False)
                inner.candidate_branch = (source, evidence, fresh)
                try:
                    yield dict(
                        source_path=CANDIDATE_SOURCE_PATH,
                        evidence_path=CANDIDATE_EVIDENCE_PATH,
                        scratch=str(fresh),
                        identity=NativeSandboxExecution.candidate_branch_capability,
                    )
                finally:
                    inner.candidate_branch = None

            @contextmanager
            def bind_history_archive(inner, directory, *, phase=None, namespace=None):
                from infinicoder.memory.task_history_archive import MODEL_ROOT

                original_directory = Path(directory).absolute()
                if any(p.is_symlink() for p in (original_directory, *original_directory.parents)):
                    raise UnsupportedCore("history mapping cannot use symlinks")
                directory = original_directory.resolve()
                state = Path(request.state_dir).resolve()
                if phase is None:
                    if namespace is not None:
                        raise UnsupportedCore("phase required with history namespace")
                    expected = state / "history-export"
                else:
                    if (
                        phase not in ("plain", "A", "B")
                        or not isinstance(namespace, str)
                        or len(namespace) != 64
                        or any(c not in "0123456789abcdef" for c in namespace)
                        or (phase == "B") != (inner.candidate_branch is not None)
                    ):
                        raise UnsupportedCore(
                            "identified phase history must match the active candidate branch"
                        )
                    expected = state / phase / "history-export"
                if (
                    inner.history_archive is not None
                    or directory != expected
                    or directory.is_relative_to(workspace)
                    or any(p.is_symlink() for p in (directory, *directory.parents))
                ):
                    raise UnsupportedCore("exact separate task-owned history export required")
                inner.history_archive = directory
                try:
                    yield dict(
                        directory=str(directory),
                        model_root=MODEL_ROOT,
                        identity="actual_CPU_readonly_task_history_mount_v1",
                        **({} if phase is None else dict(phase=phase, namespace=namespace)),
                    )
                finally:
                    inner.history_archive = None

            @contextmanager
            def bind_code_client(inner, directory, action_environment, *, optional=False):
                if type(optional) is not bool:
                    raise UnsupportedCore("explicit optional client binding required")
                directory = Path(directory).resolve()
                expected = (Path(request.state_dir) / "shell-code-client").resolve()
                if (
                    inner.code_client is not None
                    or directory != expected
                    or not directory.is_dir()
                    or not callable(action_environment)
                    or any(p.is_symlink() for p in (directory, *directory.parents))
                    or {p.name for p in directory.iterdir()} != {"ic-code"}
                    or not (directory / "ic-code").is_file()
                    or (directory / "ic-code").is_symlink()
                ):
                    raise UnsupportedCore("exact task-owned read-only code client required")
                inner.code_client = (directory, action_environment, optional)
                try:
                    yield
                finally:
                    inner.code_client = None

            def execute(inner, action, cwd="", timeout=None):
                remaining = deadline - time.monotonic()
                limit = min(NativeSandboxExecution.command_limit, remaining)
                if timeout is not None:
                    if (
                        isinstance(timeout, bool)
                        or not isinstance(timeout, (int, float))
                        or not math.isfinite(timeout)
                        or timeout <= 0
                    ):
                        raise ValueError("finite positive command timeout required")
                    limit = min(limit, timeout)
                if remaining <= 0:
                    return dict(
                        output="", returncode=-1, exception_info="shared task deadline exhausted"
                    )
                if cwd not in ("", str(workspace)):
                    raise UnsupportedCore("native sandbox cwd is the bound candidate workspace")
                command = ["/bin/bash", "-c", action.get("command", "")]
                argv = list(builder(workspace, command))
                if len(argv) <= len(command) or argv[-len(command) :] != command:
                    raise UnsupportedCore("sandbox builder must append the exact command argv")
                if self.task_environment == "task-private":
                    from infinicoder.execution.task_private_files import TaskPrivateFiles

                    if scratch not in task_files:
                        task_files[scratch] = TaskPrivateFiles(scratch)
                    argv = task_files[scratch].prepare(argv, command)
                # /tmp is already a private tmpfs in the frozen builder. Its
                # nested bind target can be created without modifying the
                # read-only host root to add a previously absent /scratch.
                active_scratch = (
                    scratch if inner.candidate_branch is None else inner.candidate_branch[2]
                )
                bindings = ["--bind", str(active_scratch), SCRATCH_PATH]
                if inner.candidate_branch is not None:
                    historical, evidence, _ = inner.candidate_branch
                    bindings += [
                        "--ro-bind",
                        str(historical),
                        CANDIDATE_SOURCE_PATH,
                        "--ro-bind",
                        str(evidence),
                        CANDIDATE_EVIDENCE_PATH,
                    ]
                if inner.history_archive is not None:
                    from infinicoder.memory.task_history_archive import MODEL_ROOT

                    if not inner.history_archive.is_dir():
                        raise UnsupportedCore("history export unavailable")
                    bindings += ["--ro-bind", str(inner.history_archive), MODEL_ROOT]
                if inner.behavior_plan_evidence is not None:
                    from infinicoder.planning.behavior_plan import MODEL_EVIDENCE

                    if not inner.behavior_plan_evidence.is_dir():
                        raise UnsupportedCore("behavior evidence unavailable")
                    bindings += ["--ro-bind", str(inner.behavior_plan_evidence), MODEL_EVIDENCE]
                lease = None
                if inner.code_client is not None:
                    directory, provider, optional = inner.code_client
                    lease = provider()
                    if lease is None and optional:
                        lease = None
                    elif not isinstance(lease, dict) or set(lease) != {
                        "IC_CODE_LEASE",
                        "IC_CODE_DEADLINE",
                    }:
                        raise UnsupportedCore("explicit current action lease required")
                    if lease is not None:
                        lease_deadline = float(lease["IC_CODE_DEADLINE"])
                        if not math.isfinite(lease_deadline):
                            raise UnsupportedCore("finite original action deadline required")
                        limit = min(limit, lease_deadline - time.monotonic())
                        if limit <= 0:
                            return dict(
                                output="",
                                returncode=-1,
                                exception_info="original code-tool action deadline exhausted",
                            )
                        bindings += ["--ro-bind", str(directory), "/tmp/ic-code-bin"]
                        for key, value in lease.items():
                            bindings += ["--setenv", key, value]
                argv[-len(command) : -len(command)] = bindings
                if self.task_environment == "task-private":
                    task_files[scratch].check_mount_targets(argv)
                started = time.monotonic()
                record = dict(
                    command_sha256=hashlib.sha256(command[-1].encode()).hexdigest(),
                    command_limit=NativeSandboxExecution.command_limit,
                    effective_timeout=limit,
                    remaining_at_start=remaining,
                    scratch=str(active_scratch),
                )
                if self.task_environment == "task-private":
                    record["task_environment"] = "task-private;temporary files persist;fresh shell"
                if inner.candidate_branch is not None:
                    record["candidate_branch"] = "B;historical_A_readonly;fresh_scratch"
                if lease is not None:
                    record.update(
                        code_client_bound=True,
                        PATH_changed=False,
                        action_lease_sha256=hashlib.sha256(
                            lease["IC_CODE_LEASE"].encode()
                        ).hexdigest(),
                    )
                try:
                    # Launch only the validated sandbox argv, without a host shell.
                    process = run_process(
                        argv,
                        cwd=str(workspace),
                        timeout=limit,
                        env=inner.config.env,
                        max_output=64 * 1024 * 1024,
                    )
                    result = dict(
                        output=process.output,
                        returncode=process.exit_code,
                        exception_info="command deadline exceeded" if process.timed_out else "",
                    )
                    inner._check_finished(result)
                    record.update(
                        returncode=result.get("returncode"),
                        exception_info=result.get("exception_info"),
                    )
                    return result
                except BaseException as exc:
                    record["exception"] = type(exc).__name__
                    raise
                finally:
                    record["wall_seconds"] = time.monotonic() - started
                    with log.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, sort_keys=True) + "\n")

        environment = Environment(cwd=str(workspace), timeout=self.command_limit, env=self.env)
        environment.deadline = deadline
        try:
            yield environment
        finally:
            if self.task_environment == "task-private":
                records = []
                for runtime in task_files.values():
                    runtime.close()
                    records.append(
                        dict(
                            root=str(runtime.root),
                            commands=runtime.commands,
                            cleaned=runtime.closed,
                        )
                    )
                (Path(request.state_dir) / "native-task-files.json").write_text(
                    json.dumps(
                        dict(
                            identity="task_private_tmp_home_cache_v1",
                            runs=records,
                            ordinary_commands_only=True,
                            trusted_builder_unchanged=True,
                            persistent_shell=False,
                        ),
                        indent=2,
                    ),
                    encoding="utf8",
                )
