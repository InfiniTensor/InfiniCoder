"""Version-bind ordinary source reads without changing or replaying actions.

Thin migration from173/bbfb567 of exact-read association only. No projection. All
other output remains an ordinary observation, including tests and diagnostics.
"""

from collections import Counter
from dataclasses import dataclass
import hashlib
import re
import shlex
import uuid

from infinicoder.runtime.shell_control import classify_shell_control
from infinicoder.planning.task_source_plan import dump
from infinicoder.memory.source.task_source_roots import relative_public_path


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else dump(value).encode()).hexdigest()


def literal_read(command, aliases):
    """A negative syntax filter plus a small positive grammar;never execution."""
    if (
        not isinstance(command, str)
        or "\n" in command
        or "\r" in command
        or "#" in command
        or classify_shell_control(command)["status"] != "foreground_syntax"
    ):
        return None
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        args = list(lexer)
    except ValueError:
        return None
    if len(args) >= 4 and args[0] == "cd" and args[2] == "&&":
        if args[1].rstrip("/") not in tuple(a.rstrip("/") for a in aliases):
            return None
        args = args[3:]
    if not args or any(any(c in a for c in ";&|<>$`*?[]") for a in args):
        return None
    if args[0] in ("cat", "/bin/cat", "/usr/bin/cat"):
        paths = args[1:]
        terminated = paths[:1] == ["--"]
        if terminated:
            paths = paths[1:]
        if not 1 <= len(paths) <= 4 or not terminated and any(p.startswith("-") for p in paths):
            return None
        requested = [(p, 1, None) for p in paths]
    elif args[0] in ("sed", "/bin/sed", "/usr/bin/sed") and len(args) == 4 and args[1] == "-n":
        match = re.fullmatch(r"([1-9][0-9]*)(?:,([1-9][0-9]*))?p", args[2])
        if not match:
            return None
        first, last = int(match[1]), int(match[2] or match[1])
        if last < first or last - first >= 8192:
            return None
        requested = [(args[3], first, last)]
    else:
        return None
    result = []
    for path, first, last in requested:
        relative = relative_public_path(path, aliases)
        if relative is None:
            return None
        result.append((relative, first, last))
    return result


def lf_lines(raw):
    """Unix cat/sed byte ranges, including CRLF and an unterminated last line."""
    parts = raw.split(b"\n")
    return [p + b"\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


@dataclass(frozen=True)
class SourceSlice:
    path: str
    source_sha256: str
    observed_lines: tuple
    observed_bytes: tuple
    raw: bytes

    def metadata(self):
        return dict(
            path=self.path,
            source_sha256=self.source_sha256,
            observed_lines=list(self.observed_lines),
            observed_bytes=list(self.observed_bytes),
            fragment_sha256=sha(self.raw),
        )


@dataclass(frozen=True)
class ReadToken:
    namespace_id: str
    command: str
    slices: tuple


class ReadObservationWorkset:
    """Link prepared source feedback;the observer separately confirms delivery."""

    def __init__(self, reader, task_state, *, namespace_id, remaining):
        if not re.fullmatch(r"[0-9a-f]{64}", namespace_id):
            raise ValueError("identified task namespace required")
        if not reader.namespace.get("environment") or not reader.namespace.get("root"):
            raise ValueError("caller-declared source namespace required")
        self.reader, self.task_state, self.remaining = reader, task_state, remaining
        self.index, self.db = task_state.index, task_state.db
        self.namespace_id = namespace_id
        self.stats = Counter()
        self.pending = {}
        self.binding_id = sha(dict(reader_identity=reader.identity, namespace=reader.namespace))
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS read_observation_meta(namespace TEXT PRIMARY KEY,binding TEXT)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS read_observations(execution_id INTEGER PRIMARY KEY,record TEXT)"
        )
        namespaces = [
            tuple(r) for r in self.db.execute("SELECT namespace,binding FROM read_observation_meta")
        ]
        if namespaces and namespaces != [(namespace_id, self.binding_id)]:
            raise ValueError("cannot mix task source namespaces")
        if self.db.execute("SELECT COUNT(*) FROM read_observations").fetchone()[0]:
            raise ValueError("fresh read-observation store required;resume is unsupported")
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO read_observation_meta VALUES(?,?)",
                (namespace_id, self.binding_id),
            )
        self.records = {}

    def _namespace(self):
        self.remaining()
        if (
            sha(dict(reader_identity=self.reader.identity, namespace=self.reader.namespace))
            != self.binding_id
        ):
            raise ValueError("source namespace binding changed during the task")

    def _save(self, record):
        self.records[record["execution_id"]] = record
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO read_observations VALUES(?,?)",
                (record["execution_id"], dump(record)),
            )

    def begin_read(self, action):
        self._namespace()
        command = action.get("command", "")
        requested = literal_read(command, self.reader.workspace_aliases)
        if requested is None:
            self.stats["unsupported_syntax"] += 1
            return None
        slices = []
        total = 0
        lines = 0
        try:
            for path, first, last in requested:
                self.remaining()
                raw, _, version = self.reader.session.read(path, max_bytes=1_000_000)
                if self.reader.current_version(path) != version:
                    raise ValueError("namespace read version differs")
                chunks = lf_lines(raw)
                last = min(len(chunks), last if last is not None else len(chunks))
                fragment = b"".join(chunks[first - 1 : last])
                fragment.decode("utf-8")  # No replacement decoding into code facts.
                offset = sum(map(len, chunks[: first - 1]))
                total += len(fragment)
                lines += max(0, last - first + 1)
                if total > 64 * 1024 or lines > 8192:
                    raise ValueError("bounded observed source output required")
                slices.append(
                    SourceSlice(
                        path, version, (first, last), (offset, offset + len(fragment)), fragment
                    )
                )
        except (OSError, ValueError, UnicodeError):
            self.stats["source_not_bound"] += 1
            return None
        self.stats["begun_reads"] += 1
        return ReadToken(self.namespace_id, command, tuple(slices))

    def finish_read(self, token, result, execution):
        if token is None:
            return None
        self._namespace()
        if token.namespace_id != self.namespace_id or execution["command"] != token.command:
            raise ValueError("read result belongs to a different action/namespace")
        record = dict(
            namespace_id=self.namespace_id,
            execution_id=execution["id"],
            command_sha256=sha(token.command.encode()),
            result_sha256=sha(result),
            pre_versions={s.path: s.source_sha256 for s in token.slices},
            post_versions={},
            source_slices=[s.metadata() for s in token.slices],
            status="unverified_runtime_observation",
            source_view_ref=None,
            linked_message_ids=None,
            authority="Observed bytes only;not edit permission,dispatch or acceptance",
        )
        expected = b"".join(s.raw for s in token.slices).decode("utf-8")
        healthy = (
            isinstance(result, dict)
            and type(result.get("returncode")) is int
            and result["returncode"] == 0
            and not result.get("exception_info")
            and not result.get("truncated")
            and not result.get("timed_out")
            and result.get("output") == expected
            and execution.get("error") is None
            and execution.get("observer_complete")
        )
        if healthy:
            try:
                for s in token.slices:
                    self.remaining()
                    record["post_versions"][s.path] = self.reader.current_version(s.path)
                healthy = all(
                    record["post_versions"][s.path] == s.source_sha256 for s in token.slices
                )
            except (OSError, ValueError, UnicodeError):
                healthy = False
        if healthy and expected:
            record["status"] = "exact_source_output_waiting_for_actual_SDK_link"
            record["raw_output_sha256"] = sha(expected.encode())
            self.pending[execution["id"]] = (token, expected, record)
        else:
            self.stats["original_output_preserved"] += 1
        self._save(record)
        return record

    def link_result(self, execution, raw_message, prepared_message):
        """Called only after NativeEpisodeMemory links the actual SDK pair."""
        pending = self.pending.get(execution["id"])
        ids = execution.get("message_ids")
        if pending is None or not ids:
            return None
        token, expected, record = pending
        extra = raw_message.get("extra", {})
        content = prepared_message.get("content", "")
        if (
            not execution.get("observer_complete")
            or execution.get("error") is not None
            or raw_message.get("role") != "user"
            or extra.get("raw_output") != expected
            or extra.get("returncode") != 0
            or not isinstance(content, str)
            or content.count(expected) != 1
        ):
            record["status"] = "SDK_feedback_not_exact;original_preserved"
            self._save(record)
            self.pending.pop(execution["id"])
            return None
        start = content.index(expected)
        ref = "source-view:" + uuid.uuid4().hex
        records = []
        fragments = []
        for s in token.slices:
            for offset, line in enumerate(lf_lines(s.raw)):
                text = line.decode("utf-8").removesuffix("\n").removesuffix("\r")
                records.append(
                    dict(
                        kind="source_line",
                        path=s.path,
                        source_sha256=s.source_sha256,
                        line=s.observed_lines[0] + offset,
                        source=text,
                    )
                )
            fragments.append(dict(s.metadata(), text=s.raw.decode("utf-8")))
        view = dict(
            view="actual_shell_source_read",
            read_event_ref=ref,
            namespace_id=self.namespace_id,
            source_files=record["pre_versions"],
            records=records,
            literal_source_texts=fragments,
            execution_id=execution["id"],
            linked_message_ids=ids,
            authority="Exact source bytes observed at these versions;historical until current namespace revalidation",
        )
        self.reader._record_view(
            dict(op="actual_shell_read", command_sha256=record["command_sha256"]), view
        )
        record.update(
            status="linked_exact_source_observation",
            source_view_ref=ref,
            linked_message_ids=list(ids),
            prepared_message_sha256=sha(prepared_message),
            raw_message_sha256=sha(raw_message),
            feedback_span=[start, start + len(expected)],
            source_text_refs=[ref + ":" + str(i) for i in range(len(fragments))],
        )
        self._save(record)
        self.pending.pop(execution["id"])
        self.stats["linked_source_observations"] += 1
        return record
