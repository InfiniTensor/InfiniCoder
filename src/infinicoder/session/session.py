"""Canonical session boundary."""

import json
import os
from pathlib import Path
import tempfile
import time


class SessionStore:
    SCHEMA_VERSION = 3

    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path / "writer.lock"
        self._locked = False
        self._seq = self._tail_seq()
        self.event_handler = None
        self.listener_errors = []

    # -- single-writer lock ------------------------------------
    def acquire_lock(self, stale_seconds=3600):
        if self._locked:
            return
        import socket

        payload = json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "at": time.time()})
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            self._locked = True
            self._seq = self._tail_seq()
            return
        except FileExistsError:
            pass
        # stale detection: holder pid dead or lock too old -> steal
        try:
            info = json.loads(self.lock_path.read_text(encoding="utf-8"))
        except Exception:
            info = {}
        pid_alive = False
        try:
            if os.name == "nt":
                import ctypes

                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(0x1000, False, int(info.get("pid", 0)))
                if handle:
                    kernel32.CloseHandle(handle)
                    pid_alive = True
            else:
                os.kill(int(info.get("pid", 0)), 0)
                pid_alive = True
        except Exception:
            pid_alive = False
        lock_age = time.time() - float(info.get("at", 0))
        # Age never permits stealing a live writer, including long GPU jobs.
        if info.get("host") not in (None, socket.gethostname()) or pid_alive:
            raise RuntimeError(
                f"session locked by pid={info.get('pid')} on "
                f"{info.get('host')} (age {lock_age:.0f}s); single-writer rule"
            )
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass
        return self.acquire_lock()

    def release_lock(self):
        if self._locked:
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass
            self._locked = False

    # -- event sequence numbers ---------------------------------
    def _tail_seq(self):
        p = self.path / "events.jsonl"
        if not p.exists():
            return 0
        try:
            last = p.read_text(encoding="utf-8").splitlines()[-1]
            return int(json.loads(last).get("seq", 0))
        except Exception:
            return 0

    def save(self, state):
        target = self.path / "state.json"
        fd, tmp = tempfile.mkstemp(dir=self.path, prefix="state-")
        try:
            state = dict(state)
            state.setdefault("schema_version", self.SCHEMA_VERSION)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def load(self):
        p = self.path / "state.json"
        if not p.exists():
            return None
        state = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            raise ValueError("Session state must be a JSON object")
        if state.get("schema_version", 0) != self.SCHEMA_VERSION:
            raise RuntimeError(
                f"session schema {state.get('schema_version')} != "
                f"{self.SCHEMA_VERSION}; migrate or start a new session"
            )
        return state

    def event(self, kind, data):
        self._seq += 1
        event = {"seq": self._seq, "time": time.time(), "kind": kind, "data": data}
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
            f.flush()
        if self.event_handler:
            try:
                self.event_handler(event)
            except Exception as e:
                self.listener_errors = (self.listener_errors + [type(e).__name__ + ": " + str(e)])[
                    -8:
                ]

    def request_control(self, action):
        if action not in ("pause", "cancel"):
            raise ValueError("control must be pause or cancel")
        target = self.path / "control.json"
        fd, temporary = tempfile.mkstemp(dir=self.path, prefix="control-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"action": action}, f)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def take_control(self):
        target = self.path / "control.json"
        if not target.exists():
            return None
        value = json.loads(target.read_text(encoding="utf-8"))
        target.unlink()
        if value.get("action") not in ("pause", "cancel"):
            raise ValueError("invalid control request")
        return value["action"]

    def events(self, after=0):
        target = self.path / "events.jsonl"
        if not target.exists():
            return []
        result = []
        for line in target.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item.get("seq", 0) > after:
                result.append(item)
        return result

    def capture_before(self, workspace, path):
        import hashlib

        source = workspace.path(path)
        key = hashlib.sha256(path.encode()).hexdigest()
        folder = self.path / "before"
        folder.mkdir(exist_ok=True)
        target = folder / (key + ".json")
        if not target.exists():
            content = source.read_text(encoding="utf-8") if source.exists() else ""
            target.write_text(
                json.dumps({"path": path, "existed": source.exists(), "source": content}),
                encoding="utf-8",
            )

    def diff(self):
        import difflib
        from infinicoder.execution.workspace_tools import Workspace

        state = self.load()
        if not state:
            return ""
        ws = Workspace(state["workspace"])
        patches = []
        for path in sorted((self.path / "before").glob("*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            current = ws.path(value["path"])
            content = current.read_text(encoding="utf-8") if current.exists() else ""
            patches.extend(
                difflib.unified_diff(
                    value["source"].splitlines(True),
                    content.splitlines(True),
                    fromfile="before/" + value["path"],
                    tofile="after/" + value["path"],
                )
            )
        return "".join(patches)

    def migrate(self):
        """Explicit additive v1/v2 migration; reject unknown or incomplete states."""
        self.acquire_lock()
        try:
            target = self.path / "state.json"
            state = json.loads(target.read_text(encoding="utf-8"))
            old = state.get("schema_version")
            if old == self.SCHEMA_VERSION:
                return {"changed": False, "schema_version": old}
            required = {
                "contract",
                "workspace",
                "steps",
                "milestone",
                "notes",
                "recent",
                "changed_files",
                "evidence",
                "failures",
                "usage",
                "status",
                "session_deadline_epoch",
            }
            if old not in (1, 2) or required - set(state):
                raise ValueError(
                    "unknown/incomplete historical session; inspect and start a new session"
                )
            backup = self.path / ("state.pre-v3-" + str(time.time_ns()) + ".json")
            backup.write_bytes(target.read_bytes())
            state["schema_version"] = self.SCHEMA_VERSION
            state.setdefault("amendments", 0)
            self.save(state)
            self.event("migration", {"from": old, "to": self.SCHEMA_VERSION, "backup": str(backup)})
            return {"changed": True, "schema_version": self.SCHEMA_VERSION, "backup": str(backup)}
        finally:
            self.release_lock()
