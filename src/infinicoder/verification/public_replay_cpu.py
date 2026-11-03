"""One caller-declared public replay in the existing readonly CPU namespace."""

from infinicoder._resources import package_file

from dataclasses import asdict
import json
from pathlib import Path
import time
from types import SimpleNamespace

from infinicoder.runtime.agent_execution import CallerPublicOperation, UnsupportedCore
from infinicoder.verification.public_replay import PublicReplaySpec, PublicReplayCapability, digest
from infinicoder.verification.readonly_diagnostic import ReadonlyCpuDiagnostic
from infinicoder.execution.workspace_tools import run_process


def cpu_public_replay(
    spec,
    workspace,
    state_dir,
    execution,
    *,
    trace_enabled=True,
    independent_reproduction=False,
    readonly_sources=None,
):
    if isinstance(spec, dict):
        spec = PublicReplaySpec(**spec)
    if not isinstance(spec, PublicReplaySpec):
        raise ValueError("explicit public replay spec required")
    if type(trace_enabled) is not bool or type(independent_reproduction) is not bool:
        raise ValueError("explicit public reproduction execution policy required")
    if execution.task_environment != "command-private" and not independent_reproduction:
        raise UnsupportedCore(
            "first public replay profile requires command-private execution;stateful replay unproven"
        )
    workspace, state = Path(workspace).resolve(), Path(state_dir).resolve()
    files = [
        package_file(name)
        for name in ("verification/public_replay.py", "verification/public_replay_worker.py")
    ]

    def invoke(deadline):
        started = time.monotonic()
        folder = state / ("public-replay" if trace_enabled else "public-bootstrap")
        folder.mkdir(mode=0o700, exist_ok=False)
        output = folder / "evidence"
        output.mkdir(mode=0o700)
        config = folder / "config.json"
        config.write_text(
            json.dumps(
                dict(
                    spec=asdict(spec),
                    output="/tmp/ic-replay-records/trace.json",
                    trace_enabled=trace_enabled,
                    deadline=deadline,
                )
            ),
            encoding="utf8",
        )
        bound = SimpleNamespace(workspace=workspace, state_dir=state)
        diagnostic = ReadonlyCpuDiagnostic(execution, bound, deadline, directory=folder)
        command = [
            spec.argv[0],
            "-B",
            "/tmp/ic-replay-code/public_replay_worker.py",
            "/tmp/ic-replay-config.json",
        ]
        argv = diagnostic._argv(command)
        extra = [
            "--dir",
            "/tmp/ic-replay-code",
            "--ro-bind",
            str(config),
            "/tmp/ic-replay-config.json",
            "--bind",
            str(output),
            "/tmp/ic-replay-records",
        ]
        for path in files:
            extra += ["--ro-bind", str(path), "/tmp/ic-replay-code/" + path.name]
        argv[-len(command) : -len(command)] = extra
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("public replay preparation exhausted original deadline")
        environment = {k: v for k, v in execution.env.items() if not k.startswith("IC_CODE_")}
        environment["CUDA_VISIBLE_DEVICES"] = ""
        process = run_process(
            argv, folder, timeout=remaining, max_output=4000, spool_cap=256 * 1024, env=environment
        )
        path = output / "trace.json"
        trace = None
        if path.is_file() and not path.is_symlink() and path.stat().st_size <= 256 * 1024:
            trace = json.loads(path.read_bytes())
        result = dict(
            trace=trace,
            process=asdict(process),
            wall_seconds=time.monotonic() - started,
            environment="original CPU namespace;readonly workspace;fresh diagnostic scratch",
            authority="public execution evidence only;not acceptance or performance measurement",
        )
        (folder / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8"
        )
        return result

    operation = CallerPublicOperation(
        "public-python-replay" if trace_enabled else "public-python-bootstrap",
        workspace,
        dict(
            adapter="readonly_public_python_replay_v1",
            files={p.name: digest(p.read_bytes()) for p in files},
        ),
        spec.public_origin,
        dict(command=spec.command, argv=spec.argv, public_input_hashes=spec.public_input_hashes),
        spec.timeout_seconds,
        dict(
            execution=execution.identity,
            temporary_state="fresh;not ordinary persistent state",
            instrumentation="bounded_python_frames" if trace_enabled else "none",
        ),
        spec.source_paths,
        ("public subprocess only;readonly candidate source;private diagnostic output",),
        spec.context_tokens,
        invoke,
    )
    return PublicReplayCapability(spec, operation, readonly_sources=readonly_sources)
