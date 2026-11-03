"""Parent-owned, CPU-only execution for bounded candidate snapshots.

The worker passes references through a private channel; scripts never run on
the host. No original live-check URL or tool socket is mounted into a trial.
"""

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.execution.workspace_tools import clean_environment


class SnapshotRunner:
    identity = "caller_bwrap_candidate_snapshot_v1;CPU;private_network_and_pid"

    def __init__(self, execution, *, workspace, directory, deadline):
        if sys.platform != "linux" or not execution.snapshot_trials:
            raise UnsupportedCore("explicit Linux CPU snapshot-run capability required")
        self.execution = execution
        self.workspace = Path(workspace).resolve()
        self.directory = Path(directory).resolve()
        self.deadline = deadline
        if self.directory.is_relative_to(self.workspace) or self.workspace.is_relative_to(
            self.directory
        ):
            raise UnsupportedCore("snapshot state must be outside the live workspace")
        # Structural admission before the model;actual namespace availability is
        # still an execution observation,never a host-shell fallback.
        self._argv(self.workspace, ["/bin/true"])

    def _argv(self, snapshot, command):
        argv = list(self.execution.sandbox_argv(snapshot, command))
        if (
            not argv
            or Path(argv[0]).name not in ("bwrap", "bubblewrap")
            or argv[-len(command) :] != command
            or "--die-with-parent" not in argv
        ):
            raise UnsupportedCore(
                "snapshot trials require the explicit original bubblewrap builder"
            )
        arguments = argv[1 : -len(command)]
        writable = []
        for i, value in enumerate(arguments):
            if value in (
                "--dev-bind",
                "--dev-bind-try",
                "--bind-try",
                "--bind-fd",
                "--overlay",
                "--tmp-overlay",
            ):
                raise UnsupportedCore("unsupported mutable host binding in snapshot builder")
            if value == "--bind":
                if i + 2 >= len(arguments):
                    raise UnsupportedCore("incomplete workspace binding")
                writable.append((arguments[i + 1], arguments[i + 2]))
        private_dev = any(
            arguments[i : i + 2] == ["--dev", "/dev"] for i in range(len(arguments) - 1)
        )
        device_rebind = any(
            arguments[i] in ("--ro-bind", "--ro-bind-try")
            and i + 2 < len(arguments)
            and arguments[i + 2].startswith("/dev/")
            for i in range(len(arguments))
        )
        if writable != [(str(snapshot), "/workspace")] or not private_dev or device_rebind:
            raise UnsupportedCore("only the snapshot workspace may have a writable host bind")
        # The ordinary caller environment stays unchanged. Candidate trials add
        # private network/PID namespaces and retain the caller's /proc policy.
        if "--share-net" in arguments:
            raise UnsupportedCore(
                "snapshot builder must not explicitly share the live checker network"
            )
        argv[-len(command) : -len(command)] = [
            "--unshare-net",
            "--unshare-pid",
            "--unsetenv",
            "IC_CODE_LEASE",
            "--unsetenv",
            "IC_CODE_DEADLINE",
            "--setenv",
            "CUDA_VISIBLE_DEVICES",
            "",
        ]
        return argv

    def run(self, packet, lease_guard):
        lease_guard()
        snapshot = Path(packet["snapshot"])
        script = Path(packet["script"])
        phase = packet["phase"]
        if phase not in ("edit", "base_probe", "candidate_probe"):
            raise ValueError("unknown candidate experiment phase")
        if (
            not snapshot.is_absolute()
            or not script.is_absolute()
            or any(path.is_symlink() for path in (snapshot, *snapshot.parents, script))
        ):
            raise ValueError("exact regular task-owned snapshot/script paths required")
        parent = snapshot.parent
        identity = parent.name
        if (
            parent.parent != self.directory
            or len(identity) != 32
            or any(c not in "0123456789abcdef" for c in identity)
            or snapshot.name != ("base" if phase == "base_probe" else "candidate")
            or script.parent != parent
            or script.name != ("edit.sh" if phase == "edit" else "probe.sh")
            or not snapshot.is_dir()
            or not script.is_file()
        ):
            raise ValueError("snapshot request does not match the task-owned trial layout")
        script_bytes = script.read_bytes()
        if (
            len(script_bytes) > 1_000_000
            or hashlib.sha256(script_bytes).hexdigest() != packet["script_sha256"]
        ):
            raise ValueError("complete frozen script required")
        cutoff = min(float(packet["cutoff"]), self.deadline)
        if not time.monotonic() < cutoff <= self.deadline:
            raise TimeoutError("candidate experiment deadline expired")
        scratch = parent / (phase + "-scratch")
        scratch.mkdir(mode=0o700, exist_ok=False)
        command = ["/bin/bash", "/tmp/ic-trial-script.sh"]
        argv = self._argv(snapshot, command)
        argv[-len(command) : -len(command)] = [
            "--ro-bind",
            str(script),
            "/tmp/ic-trial-script.sh",
            "--bind",
            str(scratch),
            "/tmp/task-scratch",
        ]
        output = parent / (phase + ".output.bin")
        started = time.monotonic()
        stored = 0
        observed = 0
        cap = 4 * 1024**2
        hit = threading.Event()
        drained = threading.Event()
        read_error = []
        environment = clean_environment(self.execution.env)
        environment = {
            key: value for key, value in environment.items() if not key.startswith("IC_CODE_")
        }
        environment["CUDA_VISIBLE_DEVICES"] = ""
        with output.open("xb") as stream:
            process = subprocess.Popen(
                argv,
                cwd=scratch,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

            def read():
                nonlocal stored, observed
                try:
                    while True:
                        block = process.stdout.read(65536)
                        if not block:
                            break
                        observed += len(block)
                        allowed = min(len(block), cap - stored)
                        if allowed:
                            stream.write(block[:allowed])
                            stored += allowed
                        if allowed < len(block):
                            hit.set()
                            break
                except (OSError, ValueError) as error:
                    read_error.append(type(error).__name__)
                finally:
                    drained.set()

            thread = threading.Thread(target=read, daemon=True, name="infinicoder-trial-output")
            thread.start()
            timed_out = False
            revoked = False

            def terminate():
                # Do not target a recycled process-group ID after poll/wait
                # reaped the leader. PID-namespace exit cleans its descendants.
                if process.returncode is not None:
                    return
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

            try:
                while True:
                    try:
                        lease_guard()
                    except (ValueError, TimeoutError):
                        revoked = True
                        terminate()
                        break
                    if time.monotonic() >= cutoff:
                        timed_out = True
                        terminate()
                        break
                    if hit.is_set():
                        terminate()
                        break
                    if process.poll() is not None and drained.is_set():
                        break
                    time.sleep(min(0.02, max(0.001, cutoff - time.monotonic())))
                process.wait(timeout=1)
            except BaseException:
                terminate()
                process.wait(timeout=1)
                raise
            finally:
                # The private PID namespace also owns detached descendants;
                # successful namespace exit cannot leave a writer in the copy.
                terminate()
                thread.join(timeout=1)
                if thread.is_alive():
                    read_error.append("output_reader_not_closed")
                else:
                    process.stdout.close()
            stream.flush()
        raw = output.read_bytes()
        value = dict(
            authority="sandbox_execution_observation",
            phase=phase,
            environment_identity=self.identity,
            script_sha256=hashlib.sha256(script_bytes).hexdigest(),
            snapshot=str(snapshot),
            returncode=124
            if timed_out or revoked
            else 125
            if hit.is_set() or read_error
            else process.returncode,
            timed_out=timed_out,
            revoked=revoked,
            truncated=hit.is_set() or bool(read_error),
            wall_s=time.monotonic() - started,
            output_ref=str(output),
            output_bytes=len(raw),
            observed_output_bytes=observed,
            output_sha256=hashlib.sha256(raw).hexdigest(),
            output_preview=raw[:600].decode("utf8", errors="replace"),
            read_errors=read_error,
            private_acceptance=False,
            live_checker_reachable=False,
        )
        (parent / (phase + ".execution.json")).write_text(
            json.dumps(value, sort_keys=True), encoding="utf8"
        )
        return value
