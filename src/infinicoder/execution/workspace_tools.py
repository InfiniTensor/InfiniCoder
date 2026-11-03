"""Bounded local tools. Path checks are not an OS sandbox.

Shell execution is opt-in in the runtime; use a container/VM for untrusted code.
"""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def clean_environment(extra=None):
    env = {
        k: v
        for k, v in os.environ.items()
        if not re.search(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH", k, re.I)
        and not k.startswith(("INFINICODER_", "OPENAI_", "ANTHROPIC_"))
    }
    # Explicit caller-approved build variables only; never model-controlled.
    env.update(extra or {})
    return env


@dataclass
class ProcessResult:
    exit_code: int
    output: str
    timed_out: bool
    truncated: bool
    duration: float
    spool_bytes: int = 0


class ProcessCancelled(RuntimeError):
    """The owning task cancelled a process;not a candidate failure."""


def run_process(
    argv, cwd, timeout=120, max_output=12000, env=None, spool_cap=64 * 1024 * 1024, cancelled=None
):
    """Drain a pipe into a bounded spool; kill owned process tree on limits.

    Resource control is not filesystem/network isolation. Use an external
    sandbox for candidate code that must not access the host.
    """
    import threading

    if cancelled is not None and not callable(cancelled):
        raise TypeError("cancelled must be a caller callback")
    if cancelled is not None and cancelled():
        raise ProcessCancelled("task cancelled before process dispatch")
    if timeout <= 0:
        return ProcessResult(124, "deadline exhausted", True, False, 0)
    if isinstance(spool_cap, bool) or not isinstance(spool_cap, int) or spool_cap < 1000:
        raise ValueError("spool_cap must be an int >= 1000")
    start = time.monotonic()
    with tempfile.TemporaryFile() as output:
        options = (
            {"start_new_session": True}
            if os.name != "nt"
            else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        )
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=clean_environment(env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            **options,
        )
        cap_hit = threading.Event()
        drained = threading.Event()
        observed = [0]
        errors = []

        def reader():
            try:
                stored = 0
                while True:
                    chunk = proc.stdout.read(min(65536, spool_cap - stored + 1))
                    if not chunk:
                        break
                    if not isinstance(chunk, bytes):
                        raise ValueError("stdout reader returned non-byte data")
                    observed[0] += len(chunk)
                    allowed = min(len(chunk), spool_cap - stored)
                    output.write(chunk[:allowed])
                    stored += allowed
                    if allowed < len(chunk):
                        cap_hit.set()
                        break
            except (OSError, ValueError) as error:
                errors.append(type(error).__name__)
            finally:
                drained.set()

        def terminate():
            if os.name == "nt":
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=10,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    pass
                if proc.poll() is None:
                    proc.kill()
            else:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        timed_out = False
        try:
            deadline = start + timeout
            while True:
                if cancelled is not None and cancelled():
                    terminate()
                    proc.wait(timeout=10)
                    raise ProcessCancelled("owning task cancelled the running process")
                if cap_hit.is_set():
                    terminate()
                    proc.wait(timeout=10)
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    terminate()
                    proc.wait(timeout=10)
                    break
                try:
                    proc.wait(timeout=min(0.05, max(0.001, deadline - time.monotonic())))
                    # Finish draining stdout after a fast process exits.
                    if drained.wait(timeout=min(0.05, max(0, deadline - time.monotonic()))):
                        break
                except subprocess.TimeoutExpired:
                    continue
        except BaseException:
            terminate()
            proc.wait(timeout=10)
            raise
        finally:
            thread.join(timeout=1)
            # On normal completion a pipe held by descendants also times out;
            # POSIX process-group cleanup handles inherited stdout writers.
            if thread.is_alive():
                terminate()
                thread.join(timeout=1)
            if not thread.is_alive():
                proc.stdout.close()
        output.flush()
        size = output.seek(0, 2)
        output.seek(0)
        if size > max_output:
            head = output.read(max_output // 2)
            output.seek(-max_output // 2, 2)
            data = head + b"\n[output truncated]\n" + output.read(max_output // 2)
        else:
            data = output.read()
    disk_hit = cap_hit.is_set()
    code = 124 if timed_out else 125 if disk_hit else proc.returncode
    if errors and code == 0:
        code = 125
    note = "\n[process killed: output exceeded spool cap]" if disk_hit else ""
    return ProcessResult(
        code,
        data.decode("utf-8", errors="replace") + note,
        timed_out,
        size > max_output or disk_hit,
        time.monotonic() - start,
        observed[0],
    )


class Workspace:
    def __init__(self, root, allowed_paths=(), validate_python=False):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError("Workspace must exist")
        self.allowed_paths = tuple(allowed_paths)
        if type(validate_python) is not bool:
            raise ValueError("validate_python must be bool")
        self.validate_python = validate_python

    def path(self, relative, write=False):
        raw = Path(relative)
        if raw.is_absolute() or ".." in raw.parts or not raw.parts:
            raise ValueError("A workspace-relative path without .. is required")
        p = self.root.joinpath(raw)
        if not p.resolve().is_relative_to(self.root):
            raise ValueError("Path leaves workspace")
        current = self.root
        for part in raw.parts:
            current /= part
            if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
                raise ValueError("Links are not permitted by file tools")
        if any(part in (".git", ".infinicoder") for part in raw.parts):
            raise ValueError("Protected workspace metadata")
        if (
            write
            and self.allowed_paths
            and not any(
                raw.as_posix() == a.rstrip("/")
                or (a.endswith("/") and raw.as_posix().startswith(a))
                for a in self.allowed_paths
            )
        ):
            raise ValueError("Path is outside the task edit scope")
        return p

    def read(self, path, start=1, count=100):
        if start < 1 or not 1 <= count <= 300:
            raise ValueError("Invalid read range")
        p = self.path(path)
        if p.stat().st_size > 2_000_000:
            raise ValueError("File too large for text tool")
        data = p.read_bytes()
        if b"\x00" in data:
            raise ValueError("Binary file")
        lines = data.decode("utf-8-sig").splitlines()
        return {
            "path": path,
            "sha256": sha256(data),
            "total_lines": len(lines),
            "start": start,
            "text": "\n".join(lines[start - 1 : start - 1 + count]),
        }

    def edit(self, path, expected_hash, content=None, old=None, new=None, start=None, end=None):
        p = self.path(path, write=True)
        exists = p.exists()
        data = p.read_bytes() if exists else b""
        actual = sha256(data) if exists else None
        if expected_hash != actual:
            raise ValueError("File changed; read it again before editing")
        if start is not None or end is not None:
            # Line-range replace with the SAME hash guard: safe for large
            # block edits/insertions where exact-string matching is brittle.
            if content is not None or old is not None:
                raise ValueError("start/end cannot combine with old/content")
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or isinstance(start, bool)
                or isinstance(end, bool)
                or start < 1
                or end < start
            ):
                raise ValueError("start/end must be 1-based lines with end >= start")
            source = data.decode("utf-8")
            lines = source.splitlines(True)
            if end > len(lines):
                raise ValueError(f"end {end} beyond file length {len(lines)}")
            removed = "".join(lines[start - 1 : end])
            if new is not None:
                if not isinstance(new, str):
                    raise ValueError("new must be a string")
                replacement = new if (new.endswith("\n") or not new) else new + "\n"
            else:
                replacement = ""
            result = "".join(lines[: start - 1]) + replacement + "".join(lines[end:])
            encoded = result.encode("utf-8")
            if len(encoded) > 1_000_000:
                raise ValueError("Edit too large")
            self._atomic_replace(p, encoded, expected_hash, actual)
            return {
                "path": path,
                "before": actual,
                "after": sha256(encoded),
                "removed_lines": end - start + 1,
                "removed_preview": removed[:120],
            }
        if content is not None:
            if exists:
                raise ValueError("Existing files require a unique old/new replacement")
            result = content
        else:
            if not exists or not isinstance(old, str) or not old or not isinstance(new, str):
                raise ValueError("old/new strings and an existing file are required")
            source = data.decode("utf-8")
            if source.count(old) != 1:
                # give the model the material to fix its
                # own mismatch in ONE step instead of blind retries.
                n = source.count(old)
                probe = old.strip().splitlines()[0][:40] if old.strip() else old[:40]
                near = [repr(ln[:100]) for ln in source.splitlines() if probe and probe[:20] in ln][
                    :2
                ]
                raise ValueError(
                    f"old text matches {n} times (need exactly 1)."
                    + (
                        f" Lines containing {probe[:20]!r}: {near}"
                        if near
                        else " Re-read the file and copy text exactly."
                    )
                )
            result = source.replace(old, new, 1)
        encoded = result.encode("utf-8")
        if len(encoded) > 1_000_000:
            raise ValueError("Edit too large")
        return self._atomic_replace(p, encoded, expected_hash, actual)

    def _atomic_replace(self, p, encoded, expected_hash, actual):
        if self.validate_python:
            from infinicoder.analysis.syntax_edits import preflight

            preflight(p, encoded)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix=".ic-edit-", dir=p.parent)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(encoded)
                f.flush()
                os.fsync(f.fileno())
            # Detect ordinary concurrent edits before replace; use OS isolation for hostile races.
            if (sha256(p.read_bytes()) if p.exists() else None) != expected_hash:
                raise ValueError("Concurrent edit conflict")
            os.replace(temp, p)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
        return {"path": str(p.relative_to(self.root)), "before": actual, "after": sha256(encoded)}

    def revision(self, excluded_paths=()):
        ignored = {".git", ".infinicoder", "__pycache__", ".pytest_cache"}
        excluded = tuple(p.rstrip("/") for p in excluded_paths)
        hashes = []
        for folder, dirs, files in os.walk(self.root, followlinks=False):
            links = []
            regular = []
            for name in dirs:
                if name in ignored:
                    continue
                path = Path(folder, name)
                rel = path.relative_to(self.root).as_posix()
                if any(rel == p or rel.startswith(p + "/") for p in excluded):
                    continue
                if path.is_symlink() or hasattr(path, "is_junction") and path.is_junction():
                    links.append(name)
                else:
                    regular.append(name)
            dirs[:] = sorted(regular)
            for name in sorted(links):
                path = Path(folder, name)
                hashes.append(path.relative_to(self.root).as_posix() + ":link:" + os.readlink(path))
            for name in sorted(files):
                path = Path(folder, name)
                rel = path.relative_to(self.root).as_posix()
                if (
                    name.startswith(".ic-edit-")
                    or path.suffix == ".pyc"
                    or any(rel == p or rel.startswith(p + "/") for p in excluded)
                ):
                    continue
                if path.is_symlink():
                    hashes.append(rel + ":link:" + os.readlink(path))
                    continue
                # Stream all files (including non-source) without a file count cutoff.
                h = hashlib.sha256()
                with path.open("rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
                hashes.append(path.relative_to(self.root).as_posix() + ":" + h.hexdigest())
        return sha256("\n".join(hashes).encode())
