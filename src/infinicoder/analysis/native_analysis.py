"""Bounded,read-only semantic expansion for one requested translation unit."""

from infinicoder._resources import package_root

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

_PARENT_GUARD = None


def install_parent_guard(expected_parent, deadline):
    """Linux parser lifetime,including death before prctl's startup handshake."""
    import ctypes
    import signal

    if sys.platform != "linux":
        raise ValueError("native parent-death guard requires Linux")
    if os.getppid() != expected_parent:
        os._exit(124)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "cannot bind native parser lifetime")
    if os.getppid() != expected_parent:
        os._exit(124)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        os._exit(124)
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    global _PARENT_GUARD
    _PARENT_GUARD = dict(
        owner_pid=expected_parent,
        parser_pid=os.getpid(),
        absolute_deadline=deadline,
        parent_death_signal="SIGKILL",
    )


class IndexBudgetExceeded(ValueError):
    pass


def policy_identity(resources):
    from infinicoder.analysis.clang_resources import arguments

    try:
        configuration = arguments(resources, [])[1]
    except (ValueError, OSError, ImportError) as error:
        configuration = dict(
            directory=str(resources), invalid=type(error).__name__ + ": " + str(error)[:300]
        )
    return hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()


def expand(path, source, profile, *, resources=None, timeout=5, guard_parent=False):
    """Only the installed parser worker executes;repository commands are data.

    A process boundary bounds libclang work without sharing a SQLite connection.
    It is resource control,not an OS sandbox. No model or project command runs.
    """
    from infinicoder.languages.native_adapter import native_context
    from infinicoder.execution.workspace_tools import run_process, clean_environment
    from infinicoder.execution.workspace_tools import Workspace

    Workspace(profile.root).path(path)
    fallback = native_context(path, source, profile, semantic=False)
    build = fallback[4]
    build.update(analysis_policy="on_demand", semantic_analysis_deferred=True)
    if timeout <= 0:
        raise IndexBudgetExceeded("No remaining time for native semantic expansion")
    package = package_root().parent
    original = (profile.root / path).read_bytes()
    if original.decode("utf-8-sig", errors="replace") != source:
        raise ValueError("Source changed before semantic expansion")
    original_sha = hashlib.sha256(original).hexdigest()
    payload = dict(path=path, source=source, root=str(profile.root.resolve()), resources=resources)
    # A private payload avoids shell interpretation and command-line size limits.
    with tempfile.TemporaryDirectory(prefix="ic-native-analysis-") as temporary:
        request = Path(temporary) / "request.json"
        request.write_text(json.dumps(payload), encoding="utf8")
        env = clean_environment(
            dict(
                PYTHONPATH=str(package) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                PYTHONDONTWRITEBYTECODE="1",
                CUDA_VISIBLE_DEVICES="",
            )
        )
        if guard_parent:
            bootstrap = (
                "import sys;sys.path.insert(0,sys.argv.pop(1));"
                "from infinicoder.analysis.native_analysis import install_parent_guard,main;"
                "install_parent_guard(int(sys.argv.pop(1)),float(sys.argv.pop(1)));main()"
            )
            guard_args = [str(os.getpid()), str(time.monotonic() + timeout)]
        else:
            bootstrap = "import sys;sys.path.insert(0,sys.argv.pop(1));from infinicoder.analysis.native_analysis import main;main()"
            guard_args = []
        process = run_process(
            [sys.executable, "-I", "-B", "-c", bootstrap, str(package), *guard_args, str(request)],
            profile.root,
            timeout,
            max_output=16 * 1024 * 1024,
            spool_cap=16 * 1024 * 1024,
            env=env,
        )
    receipt = dict(
        timeout_s=timeout,
        duration_s=process.duration,
        timed_out=process.timed_out,
        exit_code=process.exit_code,
        model_calls=0,
        project_commands=0,
        parent_guard_requested=guard_parent,
    )
    try:
        if process.exit_code or process.timed_out or process.truncated:
            raise ValueError("Parser worker failed,timed out or exceeded output budget")
        result = json.loads(process.output)
        if guard_parent:
            if (
                not isinstance(result.get("parent_guard"), dict)
                or result["parent_guard"].get("owner_pid") != os.getpid()
            ):
                raise ValueError("Parser did not confirm its original parent guard")
            receipt["parent_guard"] = result["parent_guard"]
        if result["compile_database_sha256"] != profile.compile_database_sha256:
            raise ValueError("Compile metadata changed during expansion")
        facts = result["facts"]
        assert isinstance(facts, list) and len(facts) == 5
        if hashlib.sha256((profile.root / path).read_bytes()).hexdigest() != original_sha:
            raise ValueError("Source changed during expansion")
        from infinicoder.languages.clang_adapter import header_fingerprint

        cache = {}
        for filename, digest in facts[4].get("_native_inputs", {}).get("files", {}).items():
            if (header_fingerprint(filename, cache) or (None,))[0] != digest:
                raise ValueError("Header changed during expansion")
        facts[4].update(
            analysis_policy="on_demand",
            semantic_analysis_deferred=not facts[4].get("semantic", False),
            analysis_receipt=receipt,
        )
        return facts
    except (ValueError, KeyError, AssertionError, OSError, TypeError) as error:
        build.update(
            analysis_receipt=dict(receipt, reason=str(error)[:400]),
            limitation="Semantic expansion unavailable;coarse source catalog only,no native dataflow or binding claims",
        )
        return fallback


def main():
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf8"))
    from infinicoder.languages.native_adapter import BuildProfile, native_context
    from infinicoder.execution.workspace_tools import Workspace

    Workspace(request["root"]).path(request["path"])
    profile = BuildProfile(request["root"])
    facts = native_context(
        request["path"], request["source"], profile, resource_dir=request["resources"]
    )
    print(
        json.dumps(
            dict(
                compile_database_sha256=profile.compile_database_sha256,
                facts=facts,
                parent_guard=_PARENT_GUARD,
            )
        )
    )


if __name__ == "__main__":
    main()
