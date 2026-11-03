"""Task-owned serial source-tool process, with its own SQLite connection.

The parent alone grants action leases. Stdout/stdin are a private token-count
channel; the Unix socket accepts only bounded workbench arguments from shell.
"""

import hashlib
import json
import os
from pathlib import Path
import select
import socket
import sys
import time

from infinicoder.analysis.code_context import CodeIndex
from infinicoder.tools.code_workbench import CodeWorkbench
from infinicoder.tools.code_workbench_command import run_workbench_command
from infinicoder.execution.workspace_tools import Workspace

MAX_COUNT_BYTES = 16 * 1024 * 1024


def atomic_json(path, value):
    temporary = path.with_suffix(".next")
    temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf8")
    temporary.replace(path)


class LeaseWorkspace(Workspace):
    def __init__(self, *args, remaining, **kwargs):
        super().__init__(*args, **kwargs)
        self.remaining = remaining

    def edit(self, *args, **kwargs):
        self.remaining()  # Check revocation again at the actual mutation entry.
        return super().edit(*args, **kwargs)


class Worker:
    def __init__(self, config):
        self.config = config
        self.control = Path(config["control"])
        self.status = Path(config["status"])
        self.audit = Path(config["audit"])
        self.lease = None
        self.ordinal = self.count_ordinal = 0
        self.trial_ordinal = 0
        self.trial_channel = None
        self.trial_buffer = bytearray()
        self.trial_pending = {}
        if config.get("trial_channel_fd") is not None:
            self.trial_channel = socket.socket(fileno=config["trial_channel_fd"])
            self.trial_channel.set_inheritable(False)

    def grant(self):
        return json.loads(self.control.read_bytes())

    def remaining(self):
        if os.getppid() != self.config["parent_pid"]:
            raise TimeoutError("original caller process no longer owns this worker")
        grant = self.grant()
        if (
            grant.get("closed")
            or self.lease is None
            or not grant.get("active")
            or grant.get("lease_sha256") != self.lease["lease_sha256"]
            or grant.get("execution_id") != self.lease["execution_id"]
        ):
            raise TimeoutError("original action lease revoked")
        remaining = min(grant["deadline"], self.config["deadline"]) - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("original action deadline expired")
        return remaining

    def record(self, kind, value):
        record = dict(kind=kind, **value, worker_pid=os.getpid(), epoch=time.time())
        if self.lease is not None:
            record.update(
                execution_id=self.lease["execution_id"],
                lease_sha256=self.lease["lease_sha256"],
                request_id=self.request_id,
            )
        with self.audit.open("a", encoding="utf8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def count(self, text):
        remaining = self.remaining()
        if len(text.encode("utf8")) > MAX_COUNT_BYTES:
            raise ValueError("tokenizer text exceeds16MiB;use a narrower source view")
        self.count_ordinal += 1
        identity = str(self.count_ordinal)
        packet = dict(
            kind="count",
            id=identity,
            text=text,
            lease_sha256=self.lease["lease_sha256"],
            request_id=self.request_id,
        )
        print(json.dumps(packet, ensure_ascii=False), flush=True)
        if not select.select([sys.stdin], [], [], remaining)[0]:
            raise TimeoutError("original tokenizer deadline expired")
        response = json.loads(sys.stdin.buffer.readline(65537))
        self.remaining()
        if response.get("id") == identity and isinstance(
            response.get("source_capacity_limited"), dict
        ):
            from infinicoder.memory.source.source_question_action import SourceAnswerCapacityError

            raise SourceAnswerCapacityError(**response["source_capacity_limited"])
        if (
            response.get("id") != identity
            or type(response.get("count")) is not int
            or response["count"] < 0
        ):
            raise ValueError(
                "original tokenizer failed: " + str(response.get("error", "invalid response"))
            )
        return response["count"]

    def run_snapshot(self, snapshot, script, cutoff, phase):
        if self.trial_channel is None:
            raise ValueError("no admitted snapshot execution channel")
        self.remaining()
        self.trial_ordinal += 1
        packet = dict(
            id=self.trial_ordinal,
            lease_sha256=self.lease["lease_sha256"],
            snapshot=str(snapshot),
            script=str(script),
            script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
            cutoff=cutoff,
            phase=phase,
        )
        raw = json.dumps(packet).encode() + b"\n"
        if len(raw) > 16384:
            raise ValueError("snapshot request exceeds its bounded metadata channel")
        self.trial_pending[packet["id"]] = dict(
            original_execution_id=self.lease["execution_id"],
            original_request_id=self.request_id,
            original_lease_sha256=self.lease["lease_sha256"],
            phase=phase,
        )
        self.trial_channel.sendall(raw)
        while True:
            self.remaining()
            limit = min(self.remaining(), max(0, cutoff - time.monotonic()))
            if limit <= 0:
                raise TimeoutError("original candidate experiment deadline expired")
            split = self.trial_buffer.find(b"\n")
            if split < 0:
                if len(self.trial_buffer) >= 16384:
                    raise ValueError("snapshot execution outcome exceeds its metadata bound")
                if not select.select([self.trial_channel], [], [], limit)[0]:
                    raise TimeoutError("original candidate experiment deadline expired")
                block = self.trial_channel.recv(16384 - len(self.trial_buffer))
                if not block:
                    raise ValueError("snapshot execution channel closed;outcome unknown")
                self.trial_buffer.extend(block)
                continue
            raw = bytes(self.trial_buffer[: split + 1])
            del self.trial_buffer[: split + 1]
            response = json.loads(raw)
            self.remaining()
            identity = response.get("id")
            if (
                type(identity) is not int
                or identity not in self.trial_pending
                or identity > packet["id"]
            ):
                raise ValueError("snapshot execution returned an unbound reply")
            original = self.trial_pending.pop(identity)
            if identity != packet["id"]:
                # A timed-out phase may report after its action returned. Keep
                # its provenance, but never use it as a new trial observation.
                self.record(
                    "workbench_trial_late_reply",
                    dict(
                        original,
                        phase_request_id=identity,
                        reply_sha256=hashlib.sha256(raw).hexdigest(),
                        accepted_as_current=False,
                        operation_not_retried=True,
                    ),
                )
                continue
            if "observation" not in response:
                raise ValueError(
                    "snapshot execution rejected: " + str(response.get("error", "invalid reply"))
                )
            return response["observation"]

    def serve(self):
        config = self.config
        assert Path.cwd().resolve() == Path(config["scratch"]).resolve()
        # Bind a relative name so a long original host scratch path is valid.
        address = ".ic-code.sock"
        if Path(address).exists() or Path(address).is_symlink():
            raise ValueError("fresh task socket path required")
        with CodeIndex(
            config["workspace"],
            config["database"],
            native_index_policy="on_demand",
            remaining=self.remaining,
            native_parent_guard=True,
        ) as index:
            workspace = LeaseWorkspace(
                config["workspace"], config["allowed_paths"], remaining=self.remaining
            )
            workbench = CodeWorkbench(
                index,
                workspace,
                count_output=self.count,
                remaining=self.remaining,
                output_limit=config["output_limit"],
                scratch=config["scratch"] if config["payload_mapping"] else None,
                record_event=lambda kind, value: self.record("workbench_" + kind, value),
            )
            source_reader = None
            if config.get("readonly_source_namespace") is not None:
                if config["payload_mapping"] or self.trial_channel is not None:
                    raise ValueError("readonly source actions cannot edit or run trials")
                from types import SimpleNamespace
                from infinicoder.memory.source.evidence_source import ReadOnlyTaskSources
                from infinicoder.planning.task_source_plan import WorkspaceSourcePlanReader

                sources = ReadOnlyTaskSources(
                    config["workspace"],
                    tuple(config["source_read_paths"]),
                    excluded_paths=tuple(config["source_excluded_paths"]),
                )
                sources.remaining = self.remaining
                source_reader = WorkspaceSourcePlanReader(
                    SimpleNamespace(workspace=config["workspace"]),
                    index,
                    count_output=self.count,
                    remaining=self.remaining,
                )
                source_reader.namespace = config["readonly_source_namespace"]
                source_reader.workbench.output_limit = min(2048, config["output_limit"])
                source_reader.workbench._read = sources.read
                source_reader.session._raw_read = sources.read
                source_reader.session._paths = sources.paths
            if self.trial_channel is not None:
                from infinicoder.search.candidate_trial import CandidateTrials

                workbench.trials = CandidateTrials(
                    workspace,
                    config["trial_directory"],
                    read_payload=workbench._payload,
                    runner=self.run_snapshot,
                    remaining=self.remaining,
                    task_deadline=config["deadline"],
                    generated_paths=config["trial_generated_paths"],
                    record=lambda kind, value: self.record("workbench_" + kind, value),
                )
                workbench.trial_output_limit = config["trial_output_limit"]
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(address)
                os.chmod(address, 0o600)
                server.listen(4)
                server.settimeout(0.1)
                atomic_json(self.status, dict(state="idle", pid=os.getpid()))
                print(json.dumps(dict(kind="ready", pid=os.getpid())), flush=True)
                while True:
                    if (
                        os.getppid() != config["parent_pid"]
                        or time.monotonic() >= config["deadline"]
                        or self.grant().get("closed")
                    ):
                        break
                    try:
                        connection, _ = server.accept()
                    except socket.timeout:
                        continue
                    with connection:
                        self.lease = None
                        # Mark busy before checking the grant. The parent must
                        # never observe idle while an authorized request starts.
                        atomic_json(self.status, dict(state="running", request_id=None))
                        try:
                            grant = self.grant()
                            connection.settimeout(
                                max(0.001, min(2, grant["deadline"] - time.monotonic()))
                            )
                            with connection.makefile("rb") as stream:
                                raw = stream.readline(16385)
                            if len(raw) > 16384 or not raw.endswith(b"\n"):
                                raise ValueError("bounded complete argument packet required")
                            packet = json.loads(raw)
                            nonce = packet.get("lease")
                            if not isinstance(nonce, str) or len(nonce) != 64:
                                raise ValueError("invalid action lease")
                            digest = hashlib.sha256(nonce.encode()).hexdigest()
                            if not grant.get("active") or digest != grant.get("lease_sha256"):
                                raise ValueError("expired or unrecognized action lease")
                            self.lease = grant
                            self.ordinal += 1
                            self.request_id = self.ordinal
                            self.remaining()
                            argv = packet.get("argv")
                            if (
                                not isinstance(argv, list)
                                or not 1 <= len(argv) <= 32
                                or any(not isinstance(arg, str) or len(arg) > 16000 for arg in argv)
                            ):
                                raise ValueError("bounded string arguments required")
                            atomic_json(
                                self.status,
                                dict(
                                    state="running",
                                    request_id=self.request_id,
                                    execution_id=grant["execution_id"],
                                ),
                            )
                            self.record("request_start", dict(argv=argv))
                            if source_reader is None:
                                output = run_workbench_command(workbench, argv)
                            else:
                                from infinicoder.memory.source.source_question_command import (
                                    run_source_question_command,
                                )

                                view = run_source_question_command(
                                    source_reader,
                                    argv,
                                    scratch=config["scratch"],
                                    deadline=min(config["deadline"], grant["deadline"]),
                                )
                                ref = view["read_event_ref"]
                                row = index.db.execute(
                                    "SELECT sha256 FROM source_plan_views WHERE ref=?", (ref,)
                                ).fetchone()
                                self.record(
                                    "source_view_committed",
                                    dict(
                                        read_event_ref=ref,
                                        sha256=row["sha256"],
                                        namespace=source_reader.namespace,
                                        source_files=view["source_files"],
                                    ),
                                )
                                output = json.dumps(
                                    view, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                                )
                            response = dict(output=output, returncode=0)
                        except Exception as error:
                            response = dict(
                                output=json.dumps(
                                    dict(error=type(error).__name__ + ": " + str(error)[:500])
                                ),
                                returncode=124 if isinstance(error, TimeoutError) else 2,
                            )
                        if self.lease is not None:
                            self.record(
                                "request_end",
                                dict(
                                    returncode=response["returncode"],
                                    output_sha256=hashlib.sha256(
                                        response["output"].encode()
                                    ).hexdigest(),
                                ),
                            )
                        atomic_json(self.status, dict(state="idle", pid=os.getpid()))
                        try:
                            connection.sendall(
                                json.dumps(response, ensure_ascii=False).encode() + b"\n"
                            )
                        except OSError:
                            if self.lease is not None:
                                self.record("delivery_unknown", dict(operation_not_retried=True))
                        finally:
                            self.lease = None


def main():
    config = json.loads(Path(sys.argv[1]).read_bytes())
    Worker(config).serve()


if __name__ == "__main__":
    main()
