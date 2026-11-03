"""Explicit public runs and immutable, paged reads; never final acceptance.

The caller supplies the existing public operation. Reads do not execute it,
even when source changed. Currency covers only the declared observed paths.
"""

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import socketserver
import stat
import threading
import tempfile
import time
import uuid

from infinicoder.runtime.agent_execution import CallerPublicOperation


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def observed_versions(workspace, paths, deadline):
    """Hash bounded declared files via no-follow directory descriptors."""
    root = Path(workspace)
    result = {}
    total = 0
    for name in paths:
        parent = None
        try:
            if time.monotonic() >= deadline:
                raise TimeoutError("source observation deadline")
            parent = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            parts = Path(name).parts
            for part in parts[:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                os.close(parent)
                parent = child
            fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            with os.fdopen(fd, "rb") as stream:
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode) or before.st_size > 16 * 1024**2:
                    raise ValueError("source observation bound")
                total += before.st_size
                if total > 128 * 1024**2:
                    raise ValueError("source total observation bound")
                digest = hashlib.sha256()
                seen = 0
                while True:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("source observation deadline")
                    chunk = stream.read(min(1024**2, 16 * 1024**2 - seen + 1))
                    if not chunk:
                        break
                    seen += len(chunk)
                    if seen > 16 * 1024**2:
                        raise ValueError("source grew beyond observation bound")
                    digest.update(chunk)
                after = os.fstat(stream.fileno())
                current = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
                identity = lambda v: (v.st_dev, v.st_ino, v.st_size, v.st_mtime_ns, v.st_ctime_ns)
                if (
                    identity(before) != identity(after)
                    or identity(after) != identity(current)
                    or seen != after.st_size
                ):
                    raise ValueError("source changed while hashing")
                result[name] = dict(kind="file", sha256=digest.hexdigest())
        except FileNotFoundError:
            result[name] = dict(kind="missing")
        except (OSError, ValueError, TimeoutError) as error:
            result[name] = dict(kind="unknown", reason=type(error).__name__)
        finally:
            if parent is not None:
                os.close(parent)
    return result


class PublicCheckReports:
    """One task-local store. Only run() can invoke the public executor."""

    def __init__(self, operation, directory, request, deadline, *, on_result=None):
        if on_result is not None and not callable(on_result):
            raise ValueError("public report observer must be callable")
        self.on_result = on_result
        if not isinstance(operation, CallerPublicOperation) or operation.source_mutation_allowed:
            raise ValueError("identified source-preserving caller public operation required")
        if not operation.observed_paths or len(operation.observed_paths) > 4096:
            raise ValueError("one to4096 explicitly observed public source paths required")
        if (
            type(deadline) not in (int, float)
            or not math.isfinite(deadline)
            or deadline <= time.monotonic()
            or Path(operation.workspace).resolve() != Path(request.workspace).resolve()
        ):
            raise ValueError("matching task workspace and live shared deadline required")
        self.operation = operation
        self.request = request
        self.deadline = deadline
        self.directory = Path(directory).absolute()
        if (
            self.directory.exists()
            or not self.directory.parent.is_dir()
            or self.directory.is_relative_to(Path(request.workspace).resolve())
            or any(p.is_symlink() for p in self.directory.parents)
        ):
            raise ValueError("fresh task state directory outside candidate source required")
        self.directory.mkdir(mode=0o700)
        self.records = {}
        self.gate = threading.Lock()
        self.active = True
        self.runs = 0
        self.reads = 0
        self.contract = _sha(_json(operation.manifest()).encode())
        self.stored_bytes = 0

    def _guard(self):
        if not self.active:
            raise ValueError("public report context closed")
        if time.monotonic() >= self.deadline:
            raise TimeoutError("shared task deadline exhausted")
        if _sha(_json(self.operation.manifest()).encode()) != self.contract:
            raise ValueError("public operation changed")

    def _versions(self, deadline=None):
        return observed_versions(
            self.request.workspace,
            self.operation.observed_paths,
            self.deadline if deadline is None else min(self.deadline, deadline),
        )

    @staticmethod
    def _known(versions):
        return bool(versions) and all(v["kind"] in ("file", "missing") for v in versions.values())

    def _load(self, report_id):
        if not isinstance(report_id, str) or not re.fullmatch("[0-9a-f]{32}", report_id):
            raise ValueError("invalid report_id")
        item = self.records.get(report_id)
        if item is None:
            raise ValueError("unknown task-local report_id")
        raw = (self.directory / (report_id + ".json")).read_bytes()
        if _sha(raw) != item["sha256"]:
            raise ValueError("stored public report changed")
        return json.loads(raw), raw.decode()

    def _currency(self, record, current):
        before, after = record["source_before"], record["source_after"]
        if (
            not self._known(before)
            or not self._known(after)
            or not self._known(current)
            or before != after
        ):
            return "unknown"
        return "current_on_declared_paths" if after == current else "stale"

    def run(self, *, absolute_deadline=None):
        if absolute_deadline is not None and (
            type(absolute_deadline) not in (int, float) or not math.isfinite(absolute_deadline)
        ):
            raise ValueError("finite absolute public-run deadline required")
        cutoff = (
            self.deadline if absolute_deadline is None else min(self.deadline, absolute_deadline)
        )
        self._guard()
        if not self.gate.acquire(timeout=max(0, cutoff - time.monotonic())):
            raise TimeoutError("public check busy")
        try:
            self._guard()
            if time.monotonic() >= cutoff:
                raise TimeoutError("public run deadline exhausted before dispatch")
            if len(self.records) >= 128 or self.stored_bytes > 56 * 1024**2:
                raise ValueError("public report storage budget exhausted")
            started = time.time_ns()
            before = self._versions(cutoff)
            error = None
            result = None
            try:
                with self.operation.bind(self.request, cutoff) as lease:
                    result = lease.invoke(cutoff)
            except Exception as exc:
                error = dict(type=type(exc).__name__, message=str(exc)[:2000])
            self.runs += 1
            after = self._versions(cutoff)
            report_id = uuid.uuid4().hex
            record = dict(
                report_id=report_id,
                operation=self.operation.manifest(),
                started_unix_ns=started,
                ended_unix_ns=time.time_ns(),
                source_before=before,
                source_after=after,
                result=result,
                error=error,
                authority="caller public development feedback only;not final acceptance",
                source_coverage="declared paths only;no whole-repository or semantic equivalence claim",
            )
            raw = _json(record).encode()
            if len(raw) > 8 * 1024**2:
                raise ValueError("caller public result exceeds8MiB record bound")
            path = self.directory / (report_id + ".json")
            with path.open("xb") as stream:
                stream.write(raw)
            path.chmod(0o400)
            self.records[report_id] = dict(sha256=_sha(raw), characters=len(raw.decode()))
            self.stored_bytes += len(raw)
            process = result.get("process", {}) if isinstance(result, dict) else {}
            summary = result.get("summary", {}) if isinstance(result, dict) else {}
            reply = dict(
                report_id=report_id,
                record_sha256=_sha(raw),
                characters=len(raw.decode()),
                error=error,
                source_currency=self._currency(record, after),
                summary=summary,
                process={
                    k: process[k]
                    for k in ("exit_code", "timed_out", "truncated", "duration", "spool_bytes")
                    if k in process
                },
                read="read REPORT_ID --offset 0 --limit 2000",
                authority=record["authority"],
            )
            # A caller may have a richer report format: retain it exactly,
            # but do not return arbitrary large summaries in the run reply.
            if len(_json(reply).encode()) > 12000:
                reply["summary"] = dict(omitted="read stored report for complete public result")
            if self.on_result is not None:
                self.on_result("run", reply)
            return reply
        finally:
            self.gate.release()

    def read(self, report_id, offset=0, limit=2000):
        self._guard()
        if (
            type(offset) is not int
            or offset < 0
            or type(limit) is not int
            or not 1 <= limit <= 4096
        ):
            raise ValueError("nonnegative character offset and limit1..4096 required")
        record, text = self._load(report_id)
        if offset > len(text):
            raise ValueError("offset beyond report")
        end = min(len(text), offset + limit)
        self.reads += 1
        value = dict(
            report_id=report_id,
            record_sha256=self.records[report_id]["sha256"],
            offset=offset,
            next_offset=end if end < len(text) else None,
            characters=len(text),
            text=text[offset:end],
            source_currency=self._currency(record, self._versions()),
            authority="immutable historical public feedback;read never invokes checks",
        )
        if self.on_result is not None:
            self.on_result("read", value)
        return value

    @contextmanager
    def serve(self):
        """Filesystem Unix socket works inside the CPU network namespace."""
        if not hasattr(socket, "AF_UNIX") or os.name != "posix":
            raise ValueError("POSIX Unix-domain socket required")
        # Linux Unix socket paths are limited to about108 bytes. A fresh
        # private short host directory is mounted only into this task; long
        # repository/state paths never become socket addresses.
        temporary = tempfile.TemporaryDirectory(prefix="icpub-")
        socket_path = Path(temporary.name) / "check.sock"
        owner = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                self.connection.settimeout(2)
                raw = self.rfile.readline(4097)
                try:
                    if len(raw) > 4096 or not raw.endswith(b"\n"):
                        raise ValueError("bounded JSON request required")
                    req = json.loads(raw)
                    if req == dict(action="run"):
                        value = owner.run()
                    elif (
                        isinstance(req, dict)
                        and req.get("action") == "read"
                        and set(req) == {"action", "report_id", "offset", "limit"}
                    ):
                        value = owner.read(req["report_id"], req["offset"], req["limit"])
                    else:
                        raise ValueError("explicit run or read request required")
                    response = dict(ok=True, result=value)
                except Exception as error:
                    response = dict(ok=False, error=type(error).__name__ + ": " + str(error)[:2000])
                try:
                    self.wfile.write((_json(response) + "\n").encode())
                except (BrokenPipeError, ConnectionResetError):
                    pass

        class Server(socketserver.ThreadingUnixStreamServer):
            daemon_threads = False
            block_on_close = True

        server = Server(str(socket_path), Handler)
        socket_path.chmod(0o600)
        thread = threading.Thread(
            target=lambda: server.serve_forever(poll_interval=0.05), daemon=True
        )
        thread.start()
        try:
            yield socket_path
        finally:
            # Never replace task state until owned public checks are joined.
            # Each executor is already clamped to the shared task deadline.
            server.shutdown()
            server.server_close()
            thread.join()
            self.active = False
            socket_path.unlink(missing_ok=True)
            temporary.cleanup()
            (self.directory / "lifecycle.json").write_text(
                _json(
                    dict(
                        runs=self.runs,
                        reads=self.reads,
                        reports=self.records,
                        stored_bytes=self.stored_bytes,
                        closed=True,
                        final_acceptance_invoked=False,
                    )
                )
            )
