"""Explicit public Python replay contracts and bounded, non-invasive traces.

This records execution observations, never expected answers or acceptance.
Tracing belongs in a caller-isolated subprocess, not in the agent process.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shlex
import sys
import time
from itertools import islice


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class PublicReplaySpec:
    argv: tuple
    source_paths: tuple
    public_input_hashes: dict
    public_origin: str
    timeout_seconds: float = 30
    final_reserve_seconds: float = 60
    max_events: int = 96
    context_tokens: int = 1024
    capture_mode: str = "prefix"
    public_module_paths: tuple = ()

    def __post_init__(self):
        for name in ("argv", "source_paths", "public_module_paths"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.argv or any(type(x) is not str or not x or "\0" in x for x in self.argv):
            raise ValueError("explicit public Python argv required")
        python_entry(self.argv)
        if (
            not self.source_paths
            or len(self.source_paths) > 32
            or len(set(self.source_paths)) != len(self.source_paths)
        ):
            raise ValueError("one to32 distinct caller-readable Python sources required")
        for value in self.source_paths:
            p = PurePosixPath(value)
            if (
                type(value) is not str
                or p.is_absolute()
                or ".." in p.parts
                or str(p) != value
                or "\\" in value
                or ":" in value
                or p.suffix != ".py"
            ):
                raise ValueError("normalized relative Python source paths required")
        if (
            type(self.public_input_hashes) is not dict
            or not self.public_input_hashes
            or not set(self.public_input_hashes) <= set(self.source_paths)
        ):
            raise ValueError(
                "freeze caller-public reproduction inputs separately from mutable implementation"
            )
        for pin in self.public_input_hashes.values():
            if (
                type(pin) is not str
                or len(pin) != 64
                or any(c not in "0123456789abcdef" for c in pin)
            ):
                raise ValueError("public input SHA256 required")
        mode, entry, _ = python_entry(self.argv)
        if mode == "script" and entry not in self.public_input_hashes:
            raise ValueError("public reproduction script must be hash-bound")
        if (
            type(self.public_origin) is not str
            or not self.public_origin.strip()
            or len(self.public_origin.encode()) > 4096
        ):
            raise ValueError("caller-public provenance required")
        for name, upper in (("timeout_seconds", 30), ("final_reserve_seconds", 3600)):
            v = getattr(self, name)
            if type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= upper:
                raise ValueError("finite bounded replay timeout and final reserve required")
        if type(self.max_events) is not int or not 8 <= self.max_events <= 256:
            raise ValueError("bounded trace events required")
        if type(self.context_tokens) is not int or not 128 <= self.context_tokens <= 1024:
            raise ValueError("bounded diagnostic context required")
        if self.capture_mode not in ("prefix", "failure_tail"):
            raise ValueError("explicit prefix or failure_tail capture mode required")
        if (
            len(set(self.public_module_paths)) != len(self.public_module_paths)
            or not set(self.public_module_paths) <= set(self.public_input_hashes)
            or self.public_module_paths
            and self.capture_mode != "failure_tail"
        ):
            raise ValueError(
                "module frames require failure_tail and explicitly frozen public inputs"
            )

    @property
    def command(self):
        # Exact matching avoids reconstructing compound shell operations.
        return shlex.join(self.argv)


def python_entry(argv):
    values = list(argv)
    if len(values) < 2 or not Path(values.pop(0)).name.startswith("python"):
        raise ValueError("public replay supports explicit Python entrypoints only")
    if values[0] == "-B":
        values.pop(0)
    if values[:2] == ["-m", "pytest"]:
        return "module", "pytest", values[2:]
    if not values or values[0].startswith("-"):
        raise ValueError("public replay supports a public script or python -m pytest")
    p = PurePosixPath(values[0])
    if p.is_absolute() or ".." in p.parts or str(p) != values[0] or p.suffix != ".py":
        raise ValueError("public script must be workspace-relative")
    return "script", values[0], values[1:]


def describe_public_value(value, depth=0, *, identity=None):
    """Exact builtins only: never call user repr, getters or container overrides."""
    cls = type(value)
    name = type.__getattribute__(cls, "__name__")[:100]
    if cls in (type(None), bool):
        return dict(type=name, value=value)
    if cls is int:
        return (
            dict(type=name, value=value)
            if value.bit_length() <= 128
            else dict(type=name, bits=value.bit_length())
        )
    if cls is float:
        return (
            dict(type=name, value=value)
            if math.isfinite(value)
            else dict(type=name, nonfinite=True)
        )
    if cls is str:
        return dict(type=name, value=value[:160], length=len(value), truncated=len(value) > 160)
    if cls in (list, tuple, dict):
        result = dict(type=name, length=len(value))
        if depth < 2:
            if cls is dict:
                result["items"] = [
                    [
                        describe_public_value(k, 2, identity=identity),
                        describe_public_value(v, depth + 1, identity=identity),
                    ]
                    for k, v in islice(value.items(), 8)
                ]
            else:
                result["items"] = [
                    describe_public_value(v, depth + 1, identity=identity) for v in value[:8]
                ]
            result["truncated"] = len(value) > 8
        return result
    token = identity(value) if identity is not None else None
    return dict(
        type=name,
        observed_object_id=token,
        identity_scope="trace-local observation token;address reuse and object lifetime unknown;not ownership proof",
    )


class PublicTraceRecorder:
    """Trace bounded Python frames without wrapping functions or changing errors."""

    def __init__(
        self,
        root,
        source_paths,
        *,
        max_events=96,
        capture_mode="prefix",
        public_input_hashes=None,
        public_module_paths=(),
    ):
        if type(max_events) is not int or not 8 <= max_events <= 256:
            raise ValueError("bounded trace event count required")
        self.root = Path(root).resolve()
        self.sources = {}
        for relative in source_paths:
            p = self.root / relative
            resolved = p.resolve()
            if (
                p.is_symlink()
                or any(x.is_symlink() for x in p.parents)
                or self.root not in resolved.parents
            ):
                raise ValueError("exact source files without linked parents required")
            raw = p.read_bytes()
            if len(raw) > 1024**2:
                raise ValueError("trace source exceeds1MiB")
            self.sources[str(p.resolve())] = dict(path=relative, sha256=digest(raw))
        self.max_events = max_events
        if capture_mode not in ("prefix", "failure_tail"):
            raise ValueError("unknown capture mode")
        self.capture_mode = capture_mode
        self.public_modules = set(public_module_paths)
        if (
            not self.public_modules <= set(public_input_hashes or {})
            or self.public_modules
            and capture_mode != "failure_tail"
        ):
            raise ValueError("only explicitly frozen public modules may be traced")
        pins = {v["path"]: v["sha256"] for v in self.sources.values()}
        if any(
            p not in pins or pins[p] != (public_input_hashes or {})[p] for p in self.public_modules
        ):
            raise ValueError("public module SHA mismatch")
        self.anchor_limit = min(8, max(1, max_events // 12))
        self.tail_limit = max_events - self.anchor_limit
        self.event_sizes = []
        self.events_seen = 0
        self.evicted_events = 0
        self.stopped = False
        self.stop_reason = None
        self.frame_limit = 256
        self.unknown_frames = 0
        self.recent_frames = {}
        self.terminal_anchors = []
        self.terminal_status = "not_observed"
        self.terminal_depth_limited = False
        self.terminal_anchor_omissions = 0
        self.events, self.frames = [], {}
        self.truncated = False
        self.line_events = 0
        self.error = None
        self.record_bytes = 0
        self.filenames = {}
        self.object_ids = {}

    def _object_identity(self, value):
        # Integer keys only:no user hash/equality/getters and no strong object
        # references that would change candidate lifetime or destructor timing.
        key = id(value)
        if key in self.object_ids:
            return self.object_ids[key]
        if len(self.object_ids) >= 512:
            return None
        token = len(self.object_ids) + 1
        self.object_ids[key] = token
        return token

    def _describe(self, value):
        return describe_public_value(value, identity=self._object_identity)

    def _source(self, frame):
        filename = frame.f_code.co_filename
        if filename in self.filenames:
            return self.filenames[filename]
        source = None
        in_namespace = os.name == "nt" or filename.startswith(str(self.root) + os.sep)
        if in_namespace and not filename.startswith("<") and len(filename) < 4096:
            source = self.sources.get(str(Path(filename).resolve()))
        if len(self.filenames) < 512:
            self.filenames[filename] = source
        return source

    def capture_terminal(self, error=None, *, exit_code=1, deadline=None):
        """Bounded original traceback locations;no locals,repr,or second run."""
        if self.capture_mode != "failure_tail":
            return
        if error is None or exit_code == 0:
            self.terminal_status = "no_unhandled_exception"
            return
        self.terminal_status = "unhandled_exception"
        try:
            identity_known = not (self.stopped or self.error)
            kind = type.__getattribute__(type(error), "__name__")[:100]
            tb = BaseException.__dict__["__traceback__"].__get__(error, type(error))
            seen = 0
            anchors = []
            while tb is not None and seen < 64:
                if deadline is not None and time.monotonic() >= deadline:
                    self.terminal_status = "deadline_unknown"
                    break
                seen += 1
                frame = tb.tb_frame
                source = self._source(frame)
                if source is not None and (
                    frame.f_code.co_name != "<module>" or source["path"] in self.public_modules
                ):
                    item = dict(
                        kind="terminal_exception",
                        frame=self.recent_frames.get(id(frame)) if identity_known else None,
                        parent=self.recent_frames.get(id(frame.f_back)) if identity_known else None,
                        function=frame.f_code.co_name[:120],
                        definition_line=frame.f_code.co_firstlineno,
                        line=tb.tb_lineno,
                        exception_type=kind,
                        **source,
                        authority="unhandled public process exception;not proven root cause",
                    )
                    if not identity_known:
                        item["identity_status"] = "unknown_after_coverage_stop"
                    # Keep the first outer authorized location and the latest
                    # inner locations;never invent a missing call edge.
                    anchors.append(item)
                    while (
                        len(anchors) > self.anchor_limit
                        or len(json.dumps(anchors, ensure_ascii=False).encode()) > 32 * 1024
                    ):
                        anchors.pop(1 if self.anchor_limit > 1 else 0)
                        self.terminal_anchor_omissions += 1
                tb = tb.tb_next
            self.terminal_depth_limited = tb is not None
            if self.terminal_depth_limited:
                self.truncated = True
            self.terminal_anchors = anchors
        except Exception as failure:
            self.terminal_status = "collector_error_unknown"
            self.error = type(failure).__name__

    def trace(self, frame, event, arg):
        if self.error or self.stopped or self.truncated and self.capture_mode == "prefix":
            return None
        try:
            source = self._source(frame)
        except Exception as error:
            self.error = type(error).__name__
            return None
        if source is None or (
            frame.f_code.co_name == "<module>"
            and (self.capture_mode != "failure_tail" or source["path"] not in self.public_modules)
        ):
            return None
        try:
            if event == "line":
                self.line_events += 1
                if self.line_events >= 100000:
                    self.truncated = True
                    self.stopped = True
                    self.stop_reason = "line_budget_exhausted"
                    return None
                return self.trace
            if event not in ("call", "return", "exception"):
                return self.trace
            if self.capture_mode == "prefix" and len(self.events) >= self.max_events:
                self.truncated = True
                return None
            key = id(frame)
            sequence = self.events_seen
            self.events_seen += 1
            if event == "call":
                if self.capture_mode == "prefix" or len(self.frames) < self.frame_limit:
                    self.frames[key] = sequence
                else:
                    self.unknown_frames += 1
                if self.capture_mode == "failure_tail":
                    self.recent_frames.pop(key, None)
                    if key in self.frames:
                        if len(self.recent_frames) >= self.frame_limit:
                            self.recent_frames.pop(next(iter(self.recent_frames)))
                        self.recent_frames[key] = sequence
            item = dict(
                kind=event,
                frame=self.frames.get(key),
                parent=self.frames.get(id(frame.f_back)),
                function=frame.f_code.co_name[:120],
                definition_line=frame.f_code.co_firstlineno,
                line=frame.f_lineno,
                **source,
            )
            if self.capture_mode == "failure_tail":
                item["sequence"] = sequence
                if frame.f_code.co_flags & (0x20 | 0x80 | 0x200):
                    item["continuity"] = "generator_or_coroutine_resume_unknown"
            if event == "call":
                code = frame.f_code
                n = (
                    code.co_argcount
                    + code.co_kwonlyargcount
                    + bool(code.co_flags & 4)
                    + bool(code.co_flags & 8)
                )
                item["parameters"] = {
                    name: self._describe(frame.f_locals[name])
                    for name in code.co_varnames[: min(n, 16)]
                    if name in frame.f_locals
                }
            elif event == "return":
                item["returned"] = self._describe(arg)
                item["semantics"] = "return_or_exception_unwind"
                self.frames.pop(key, None)
            else:
                item["exception_type"] = type.__getattribute__(arg[0], "__name__")[:100]
                item["locals"] = {
                    k: self._describe(v) for k, v in islice(frame.f_locals.items(), 12)
                }
            size = len(json.dumps(item, ensure_ascii=False, allow_nan=False).encode())
            if self.capture_mode == "failure_tail":
                while self.events and (
                    len(self.events) >= self.tail_limit or self.record_bytes + size > 160 * 1024
                ):
                    self.events.pop(0)
                    self.record_bytes -= self.event_sizes.pop(0)
                    self.evicted_events += 1
                    self.truncated = True
                if size > 160 * 1024:
                    self.truncated = True
                    self.stopped = True
                    self.stop_reason = "event_byte_budget_exhausted"
                    return None
            elif self.record_bytes + size > 192 * 1024:
                self.truncated = True
                return None
            self.record_bytes += size
            self.events.append(item)
            if self.capture_mode == "failure_tail":
                self.event_sizes.append(size)
        except Exception as error:
            # Collector failure cannot become the candidate's exception.
            self.error = type(error).__name__
        return self.trace

    def __enter__(self):
        self.prior = sys.gettrace()
        if self.prior is not None:
            raise ValueError(
                "public replay requires an isolated process without an existing tracer"
            )
        sys.settrace(self.trace)
        return self

    def __exit__(self, *_):
        sys.settrace(self.prior)

    def report(self):
        value = dict(
            identity="bounded_public_python_execution_v1",
            events=self.events,
            sources=list(self.sources.values()),
            truncated=self.truncated,
            collector_error=self.error,
            line_events=self.line_events,
            limitations=[
                "Python current-thread frames only",
                "C/native/IO/thread/process effects unknown",
                "object values not expanded;diagnostic timing is not a performance measurement",
            ],
        )
        if self.capture_mode == "failure_tail":
            value.update(
                capture_mode=self.capture_mode,
                terminal_anchors=self.terminal_anchors,
                terminal_status=self.terminal_status,
                events_seen=self.events_seen,
                evicted_events=self.evicted_events,
                tail_event_limit=self.tail_limit,
                terminal_anchor_limit=self.anchor_limit,
                coverage_exhausted=self.stopped,
                stop_reason=self.stop_reason,
                unknown_frame_calls=self.unknown_frames,
                terminal_depth_limited=self.terminal_depth_limited,
                terminal_anchor_omissions=self.terminal_anchor_omissions,
                call_ancestry="partial;evicted calls and untracked deep frames unknown",
                return_semantics="return_or_exception_unwind;generator continuity unknown",
            )
        return value


@dataclass(frozen=True)
class PublicReplayCapability:
    spec: PublicReplaySpec
    operation: object
    model_root: str = "/workspace"
    namespace_binding: object = None
    readonly_sources: object = None

    def __post_init__(self):
        from infinicoder.runtime.agent_execution import CallerPublicOperation

        if not isinstance(self.spec, PublicReplaySpec) or not isinstance(
            self.operation, CallerPublicOperation
        ):
            raise ValueError("explicit replay spec and caller-owned public operation required")
        if (
            self.operation.source_mutation_allowed
            or tuple(self.operation.observed_paths) != self.spec.source_paths
        ):
            raise ValueError("replay must observe the declared sources without mutating source")
        if self.operation.timeout_seconds > self.spec.timeout_seconds:
            raise ValueError("public executor cannot exceed replay timeout")
        root = PurePosixPath(self.model_root)
        if not root.is_absolute() or ".." in root.parts or str(root) != self.model_root:
            raise ValueError("explicit model workspace root required")
        if self.namespace_binding is not None and not callable(self.namespace_binding):
            raise ValueError("caller namespace binding must be callable")
        if self.readonly_sources is not None:
            if type(self.readonly_sources) is not dict or not set(self.readonly_sources) <= set(
                self.spec.public_input_hashes
            ):
                raise ValueError("external sources must be explicitly frozen public inputs")
            for relative, path in self.readonly_sources.items():
                if not Path(path).is_absolute():
                    raise ValueError("absolute caller public source paths required")

    def source_path(self, relative):
        if relative not in self.spec.source_paths:
            raise ValueError("source outside replay observation scope")
        return Path(
            (self.readonly_sources or {}).get(relative, Path(self.operation.workspace) / relative)
        )
