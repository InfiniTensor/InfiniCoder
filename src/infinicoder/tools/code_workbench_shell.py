"""Own one source-tool process, with explicit action leases and token counting.

No SDK loop, model generation, repository execution, or acceptance calls live
here. Only the worker owns its SQLite connection. The counter broker never does.
"""

from infinicoder._resources import package_root

import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import stat
import subprocess
import sys
import threading
import time

from infinicoder.tools import code_workbench_client
from infinicoder.tools.code_workbench_worker import atomic_json, MAX_COUNT_BYTES

CLIENT_PATH = "/tmp/ic-code-bin/ic-code"


class ShellWorkbench:
    def __init__(
        self,
        *,
        workspace,
        database,
        state,
        scratch,
        allowed_paths,
        output_limit,
        deadline,
        count_output,
        payload_mapping=True,
        trial_runner=None,
        trial_generated_paths=(),
        trial_output_limit=None,
        readonly_source_namespace=None,
        source_read_paths=(),
        source_excluded_paths=(),
    ):
        if sys.platform != "linux":
            raise ValueError("shell code tools require the admitted Linux Unix-socket sandbox")
        self.deadline, self.count_output = deadline, count_output
        self.folder = Path(state) / "shell-code-tools"
        self.folder.mkdir(exist_ok=False)
        self.scratch = Path(scratch).resolve()
        self.client_directory = Path(state) / "shell-code-client"
        self.client_directory.mkdir(exist_ok=False)
        source = Path(code_workbench_client.__file__).read_text(encoding="utf8")
        client = self.client_directory / "ic-code"
        # This stdlib client runs in the task namespace, which may use a
        # different Python installation from the controller process.
        client.write_text("#!/usr/bin/env python3\n" + source, encoding="utf8", newline="\n")
        client.chmod(0o555)
        self.control = self.folder / "lease.json"
        self.status = self.folder / "status.json"
        self.audit = self.folder / "worker-events.jsonl"
        self.counter_log = self.folder / "tokenizer.jsonl"
        self.lifecycle_log = self.folder / "lifecycle.jsonl"
        self.lock = threading.RLock()
        self.ready = threading.Event()
        self.available = False
        self.lease = None
        self.grants = {}
        self.offset = 0
        self.pending_count = None
        self.socket_identity = None
        self.closed = False
        self.trial_runner = trial_runner
        self.trial_channel = self.trial_thread = None
        atomic_json(self.control, dict(active=False, closed=False, deadline=deadline))
        config = dict(
            workspace=str(Path(workspace).resolve()),
            database=str(Path(database).resolve()),
            scratch=str(self.scratch),
            allowed_paths=list(allowed_paths),
            output_limit=output_limit,
            deadline=deadline,
            parent_pid=os.getpid(),
            control=str(self.control.resolve()),
            status=str(self.status.resolve()),
            audit=str(self.audit.resolve()),
            payload_mapping=payload_mapping,
        )
        if readonly_source_namespace is not None:
            if (
                not isinstance(readonly_source_namespace, dict)
                or not readonly_source_namespace
                or trial_runner is not None
                or payload_mapping
                or not source_read_paths
            ):
                raise ValueError(
                    "readonly source actions require explicit readable paths;no payload edits or trials"
                )
            config.update(
                readonly_source_namespace=readonly_source_namespace,
                source_read_paths=list(source_read_paths),
                source_excluded_paths=list(source_excluded_paths),
            )
        elif source_read_paths or source_excluded_paths:
            raise ValueError("source readable paths require an explicit readonly namespace")
        child_channel = None
        if trial_runner is not None:
            if not payload_mapping:
                raise ValueError("candidate trials require the explicit scratch payload mapping")
            self.trial_channel, child_channel = socket.socketpair()
            config.update(
                trial_channel_fd=child_channel.fileno(),
                trial_directory=str(trial_runner.directory),
                trial_generated_paths=list(trial_generated_paths),
                trial_output_limit=trial_output_limit or output_limit,
            )
        config_file = self.folder / "worker.json"
        config_file.write_text(json.dumps(config), encoding="utf8")
        env = dict(os.environ)
        root = str(package_root().parent)
        env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        self.stderr = (self.folder / "worker.stderr").open("wb")
        try:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "infinicoder.tools.code_workbench_worker",
                    str(config_file.resolve()),
                ],
                cwd=self.scratch,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.stderr,
                pass_fds=(child_channel.fileno(),) if child_channel is not None else (),
            )
        except BaseException:
            if self.trial_channel is not None:
                self.trial_channel.close()
            self.stderr.close()
            raise
        finally:
            if child_channel is not None:
                child_channel.close()
        self.broker = threading.Thread(
            target=self._broker, daemon=True, name="infinicoder-tool-tokenizer"
        )
        self.broker.start()
        if self.trial_channel is not None:
            self.trial_thread = threading.Thread(
                target=self._serve_trials, daemon=True, name="infinicoder-snapshot-runner"
            )
            self.trial_thread.start()
        try:
            if not self.ready.wait(max(0, min(10, deadline - time.monotonic()))):
                raise TimeoutError("task source-tool process did not become ready")
            if self.process.poll() is not None:
                raise RuntimeError("task source-tool process exited during initialization")
            self.available = True
            socket_info = (self.scratch / ".ic-code.sock").lstat()
            self.socket_identity = (socket_info.st_dev, socket_info.st_ino)
            self._log(dict(event="worker_ready", pid=self.process.pid))
        except BaseException:
            self.close()
            raise

    def _log(self, record, path=None):
        record = dict(record, monotonic=time.monotonic(), epoch=time.time())
        with self.lock:
            with (path or self.lifecycle_log).open("a", encoding="utf8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")

    def _active(self, digest):
        with self.lock:
            lease = dict(self.lease or {})
            if (
                self.closed
                or not lease.get("active")
                or lease.get("lease_sha256") != digest
                or time.monotonic() >= lease["deadline"]
            ):
                raise TimeoutError("original action lease is not active")
            return lease

    def _serve_trials(self):
        """Private inherited FD, separate from tokenizer IPC and the SDK loop."""
        try:
            with self.trial_channel.makefile("rb") as stream:
                while not self.closed:
                    raw = stream.readline(16385)
                    if not raw:
                        return
                    if len(raw) > 16384 or not raw.endswith(b"\n"):
                        return
                    packet = json.loads(raw)
                    reply = dict(id=packet.get("id"))
                    try:
                        lease = self._active(packet["lease_sha256"])
                        packet["cutoff"] = min(float(packet["cutoff"]), lease["deadline"])
                        self._log(
                            dict(
                                event="snapshot_start",
                                execution_id=lease["execution_id"],
                                phase=packet.get("phase"),
                                request_id=packet.get("id"),
                            )
                        )
                        reply["observation"] = self.trial_runner.run(
                            packet, lambda: self._active(packet["lease_sha256"])
                        )
                    except Exception as error:
                        reply["error"] = type(error).__name__ + ": " + str(error)[:500]
                    try:
                        self.trial_channel.sendall(json.dumps(reply).encode() + b"\n")
                    except OSError:
                        return
        except (OSError, ValueError, KeyError) as error:
            self._log(dict(event="snapshot_channel_closed", error=type(error).__name__))

    def _broker(self):
        """Only prepare/count bounded text; never access SQLite/history/query."""
        try:
            while True:
                raw = self.process.stdout.readline(8 * MAX_COUNT_BYTES + 65537)
                if not raw:
                    return
                if len(raw) > 8 * MAX_COUNT_BYTES + 65536 or not raw.endswith(b"\n"):
                    return
                packet = json.loads(raw)
                if packet.get("kind") == "ready":
                    if packet.get("pid") == self.process.pid:
                        self.ready.set()
                    continue
                if packet.get("kind") != "count":
                    return
                identity = packet.get("id")
                record = dict(
                    id=identity,
                    request_id=packet.get("request_id"),
                    lease_sha256=packet.get("lease_sha256"),
                    event="count",
                    new_generation_requests=0,
                )
                response = dict(id=identity)
                started = time.monotonic()
                try:
                    lease = self._active(packet["lease_sha256"])
                    text = packet["text"]
                    if (
                        not isinstance(identity, str)
                        or len(identity) > 64
                        or not isinstance(text, str)
                        or len(text.encode("utf8")) > MAX_COUNT_BYTES
                    ):
                        raise ValueError("bounded tokenizer request required")
                    record.update(
                        execution_id=lease["execution_id"],
                        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    )
                    self.pending_count = dict(record)
                    count = self.count_output(text, lease["deadline"])
                    if type(count) is not int or count < 0:
                        raise ValueError("original tokenizer returned an invalid count")
                    self._active(packet["lease_sha256"])
                    response["count"] = count
                    record.update(status="complete", count=count)
                except Exception as error:
                    response["error"] = type(error).__name__ + ": " + str(error)[:400]
                    record.update(
                        status="rejected_or_late;no_success_returned", error=response["error"]
                    )
                    from infinicoder.memory.source.source_question_action import (
                        SourceAnswerCapacityError,
                    )

                    if isinstance(error, SourceAnswerCapacityError):
                        capacity = dict(
                            standalone_tokens=error.standalone_tokens,
                            next_input_tokens=error.next_input_tokens,
                            input_limit=error.input_limit,
                        )
                        response["source_capacity_limited"] = capacity
                        record.update(status="actual_next_input_capacity_limited", **capacity)
                finally:
                    record["wall_s"] = time.monotonic() - started
                    self.pending_count = None
                    self._log(record, self.counter_log)
                try:
                    self.process.stdin.write(json.dumps(response).encode() + b"\n")
                    self.process.stdin.flush()
                except (BrokenPipeError, OSError, ValueError):
                    return
        finally:
            self.ready.set()
            for stream in (self.process.stdin, self.process.stdout):
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass

    def begin_action(self, execution_id, deadline):
        with self.lock:
            if self.closed:
                raise ValueError("one-shot source-tool binding already closed")
            nonce = secrets.token_hex(32)
            digest = hashlib.sha256(nonce.encode()).hexdigest()
            self.lease = dict(
                active=True,
                closed=False,
                lease_sha256=digest,
                execution_id=execution_id,
                deadline=min(self.deadline, deadline),
            )
            self.grants[digest] = dict(self.lease)
            atomic_json(self.control, self.lease)
            self._log(dict(event="begin_action", **self.lease))
            return dict(IC_CODE_LEASE=nonce, IC_CODE_DEADLINE=str(self.lease["deadline"]))

    def _stop_process(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process.wait(timeout=1)
        self.available = False

    def _idle(self):
        if self.process.poll() is not None:
            return True
        try:
            return json.loads(self.status.read_bytes())["state"] == "idle"
        except (OSError, ValueError, KeyError):
            return False

    def end_action(self):
        with self.lock:
            lease = dict(self.lease or {})
            if not lease:
                return []
            self.lease["active"] = False
            atomic_json(self.control, self.lease)
        # Foreground CLI has already finished. Only detached/in-flight requests
        # need this bounded drain. No task deadline is extended for tool work.
        until = min(self.deadline, lease["deadline"], time.monotonic() + 0.2)
        while not self._idle() and time.monotonic() < until:
            time.sleep(min(0.01, max(0, until - time.monotonic())))
        interrupted = not self._idle()
        if interrupted:
            self._stop_process()
        records = self.drain_events()
        self._log(
            dict(
                event="end_action",
                execution_id=lease.get("execution_id"),
                worker_interrupted=interrupted,
                in_flight_outcome="unknown" if interrupted else "drained",
                pending_token_count=self.pending_count is not None,
                worker_available=self.available and self.process.poll() is None,
            )
        )
        return records

    def drain_events(self):
        if not self.audit.is_file():
            return []
        with self.audit.open("rb") as stream:
            stream.seek(self.offset)
            raw = stream.read()
        stop = raw.rfind(b"\n") + 1
        records = [json.loads(line) for line in raw[:stop].splitlines()]
        self.offset += stop
        for record in records:
            grant = self.grants.get(record.get("lease_sha256"))
            if (
                grant is None
                or record.get("execution_id") != grant["execution_id"]
                or record.get("worker_pid") != self.process.pid
            ):
                raise ValueError("unbound source-tool audit record")
        return records

    def close(self):
        if self.closed:
            return
        with self.lock:
            self.closed = True
            self.available = False
            if self.lease is not None:
                self.lease["active"] = False
            atomic_json(
                self.control,
                dict(self.lease or {}, active=False, closed=True, deadline=self.deadline),
            )
        try:
            self.process.wait(timeout=max(0, min(0.2, self.deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            self._stop_process()
        self._log(
            dict(
                event="closed",
                worker_returncode=self.process.poll(),
                pending_token_count=self.pending_count is not None,
            )
        )
        # A late counter callback only owns immutable prepared input and its
        # count log. The source worker is reaped before any final verification.
        self.broker.join(timeout=0.05)
        if self.trial_channel is not None:
            try:
                self.trial_channel.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.trial_channel.close()
            self.trial_thread.join(timeout=2)
            self._log(dict(event="snapshot_runner_closed", pending=self.trial_thread.is_alive()))
        for stream in (self.process.stdin, self.process.stdout, self.stderr):
            if not self.broker.is_alive() or stream is self.stderr:
                stream.close()
        address = self.scratch / ".ic-code.sock"
        try:
            info = address.lstat()
            if stat.S_ISSOCK(info.st_mode) and (info.st_dev, info.st_ino) == self.socket_identity:
                address.unlink()
        except FileNotFoundError:
            pass
