"""Optional source queries and an expandable journal for the real native core.

The SDK retains its full history. Only the next model input is projected when
it will not fit; original task/system and the latest action/feedback stay exact.
No model-authored summaries become source facts and no extra generation runs.
"""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import shlex
import time

from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.analysis.code_context import CodeIndex


class MemoryCapacityError(ValueError):
    """Mandatory task and recent evidence cannot fit the declared input budget."""


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


class NativeRepositoryMemory:
    identity = "native_repository_memory_v1"
    expose_tools = True
    command_names = ("ic-memory",)
    HELP = (
        "Optional memory commands: use ONE standalone command, without a shell prefix. "
        'ic-memory find "question or symbol"; ic-memory route "module question"; '
        "ic-memory symbol SYMBOL_ID; ic-memory source PATH [START_LINE END_LINE]; "
        "ic-memory structure SYMBOL_ID; ic-memory relations SYMBOL_ID; "
        'ic-memory flow PATH [NAME]; ic-memory history "query"; '
        "ic-memory event ID [START_CHAR END_CHAR]. "
        "Source views refresh the index, identify versions and retain unknown relations. "
        "History contains original historical messages, including unverified agent claims; "
        "it is not proof of the current code. Re-read changed source. "
        "These commands do not execute or replay archived actions."
    )

    def __init__(self, count_messages, *, tokenizer_identity, reserve=256, tool_tokens=1000):
        self.count_messages = count_messages
        self.tokenizer_identity = tokenizer_identity
        self.reserve = reserve
        self.tool_tokens = tool_tokens

    def validate(self, request):
        if (
            not callable(self.count_messages)
            or not isinstance(self.tokenizer_identity, str)
            or not self.tokenizer_identity
        ):
            raise UnsupportedCore(
                "memory requires an actual message-tokenizer callback and its identity"
            )
        if type(self.reserve) is not int or self.reserve < 0:
            raise ValueError("nonnegative memory reserve required")
        if type(self.tool_tokens) is not int or self.tool_tokens < 64:
            raise ValueError("memory tool budget must be at least64 tokens")
        if request.model_profile.input_budget() <= self.reserve:
            raise UnsupportedCore("no input room after the declared output and memory reserves")

    def _remaining(self):
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("memory shares the original task deadline")
        return left

    def _count(self, messages):
        self._remaining()
        # Use the pinned SDK's actual API preparation, including removal of its
        # private extra/usage fields, before the caller's actual chat tokenizer.
        count = self.count_messages(self.prepare(messages), self._remaining())
        if type(count) is not int or count < 0:
            raise ValueError("message tokenizer must return a nonnegative int")
        self._remaining()
        return count

    @contextmanager
    def bind(self, request, model, environment, deadline):
        self.validate(request)
        self.deadline = deadline
        self.prepare = model._prepare_messages_for_api
        self.state = Path(request.state_dir)
        self.limit = request.model_profile.input_budget() - self.reserve
        self.tool_limit = min(self.tool_tokens, max(64, self.limit // 3))
        self.index = CodeIndex(
            request.workspace,
            self.state / "native-memory.sqlite",
            native_index_policy="on_demand",
            remaining=self._remaining,
        )
        self.index.db.execute(
            "CREATE TABLE IF NOT EXISTS native_history(id INTEGER PRIMARY KEY, sha TEXT, message TEXT)"
        )
        self.index.db.execute(
            "CREATE TABLE IF NOT EXISTS native_projections(id INTEGER PRIMARY KEY, messages TEXT, accounting TEXT)"
        )
        self.index.db.commit()
        if self.index.db.execute("SELECT COUNT(*) FROM native_history").fetchone()[0]:
            self.index.close()
            raise UnsupportedCore(
                "memory is one-shot; an existing history cannot be replayed or resumed"
            )
        self.seen = []
        query, execute = model.query, environment.execute

        def memory_query(messages, **kwargs):
            projected = self.project(messages)
            return query(projected, **kwargs)

        def memory_execute(action, *args, **kwargs):
            command = action.get("command", "")
            # Only a standalone explicit tool invocation is intercepted. Shell
            # operators, permissions and ordinary commands keep the original path.
            pattern = (
                r"^\s*(?:" + "|".join(re.escape(name) for name in self.command_names) + r")(?:\s|$)"
            )
            if re.match(pattern, command):
                try:
                    output = self.execute_tool_command(command)
                    return dict(output=output, returncode=0, exception_info="")
                except (ValueError, KeyError) as error:
                    return dict(
                        output=_json(dict(error=str(error), help=self.HELP)),
                        returncode=2,
                        exception_info="",
                    )
            return execute(action, *args, **kwargs)

        model.query, environment.execute = (
            memory_query,
            memory_execute if self.expose_tools else execute,
        )
        try:
            yield self
        finally:
            model.query, environment.execute = query, execute
            self.index.close()
            (self.state / "native-memory-info.json").write_text(
                _json(
                    dict(
                        identity=self.identity,
                        tokenizer_identity=self.tokenizer_identity,
                        journal=str(self.state / "native-memory.sqlite"),
                        projections=str(self.state / "native-memory-context.jsonl"),
                        retained_messages=len(self.seen),
                        resume_supported=False,
                        source_facts="Only explicit fresh source queries;agent claims remain historical unverified messages.",
                    )
                )
            )

    def execute_tool_command(self, command):
        args = self.parse_tool_command(command)
        return self.tool(args[1:])

    def parse_tool_command(self, command):
        args = shlex.split(command)
        if any(x in (";", "&&", "||", "|", ">", "<") for x in args):
            raise ValueError("ic-memory requires a standalone command")
        return args

    def _sync(self, messages):
        prepared = self.prepare(messages)
        serial = [_json(message) for message in prepared]
        hashes = [_sha(message) for message in serial]
        if hashes[: len(self.seen)] != self.seen:
            raise ValueError(
                "native history prefix changed; refusing to invent or replay a history"
            )
        with self.index.db:
            for i in range(len(self.seen), len(serial)):
                self.index.db.execute(
                    "INSERT INTO native_history VALUES(?,?,?)", (i, hashes[i], serial[i])
                )
        self.seen = hashes
        return prepared

    def project(self, messages):
        prepared = self._sync(messages)
        if len(prepared) < 2 or [m["role"] for m in prepared[:2]] != ["system", "user"]:
            raise UnsupportedCore("memory requires the pinned text agent system/task prefix")
        if any(not isinstance(m.get("content"), str) for m in prepared):
            raise UnsupportedCore("memory projection currently supports text histories only")
        # Copies: the SDK keeps every original message and its private fields.
        full = [dict(m) for m in prepared]
        full[1]["content"] += "\n\nMEMORY TOOLS\n" + self.HELP
        original_tokens = self._count(full)
        selected = list(range(len(prepared)))
        projected = full
        compacted = original_tokens > self.limit
        if compacted:
            tail = max(2, len(prepared) - 2)
            selected = [0, 1, *range(tail, len(prepared))]
            omitted = list(range(2, tail))
            note = dict(
                archived_messages=omitted,
                policy="Full original messages remain in ic-memory history/event;none were replayed. "
                "Past agent text is unverified. Current task and latest action/feedback are retained exactly.",
                recent_archived=[
                    dict(id=i, role=prepared[i]["role"], sha256=self.seen[i]) for i in omitted[-4:]
                ],
            )
            projected = [dict(full[i]) for i in selected]
            projected[1]["content"] += "\n\nHISTORICAL MEMORY INDEX\n" + _json(note)
        final_tokens = self._count(projected)
        record = dict(
            original_messages=len(prepared),
            original_tokens=original_tokens,
            projected_tokens=final_tokens,
            input_limit=self.limit,
            compacted=compacted,
            selected=selected,
            omitted=[i for i in range(len(prepared)) if i not in selected],
            original_task_sha256=_sha(prepared[1]["content"]),
            tokenizer_identity=self.tokenizer_identity,
            status="capacity_rejected"
            if final_tokens > self.limit
            else "ready_for_native_query;HTTP ledger determines actual forwarding",
        )
        with self.index.db:
            self.index.db.execute(
                "INSERT INTO native_projections(messages,accounting) VALUES(?,?)",
                (_json(projected), _json(record)),
            )
        with (self.state / "native-memory-context.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(_json(record) + "\n")
        if final_tokens > self.limit:
            raise MemoryCapacityError(
                "Original task/system plus latest action/feedback do not fit;no generation forwarded"
            )
        return projected

    def _bounded(self, kind, records, **metadata):
        def prefix(count):
            return dict(
                view=kind, records=records[:count], omitted=len(records) - count, **metadata
            )

        def fits(value):
            return self._count([dict(role="user", content=_json(value))]) <= self.tool_limit

        if fits(prefix(len(records))):
            return _json(prefix(len(records)))
        if not fits(prefix(0)):
            raise MemoryCapacityError(
                "memory response metadata exceeds the tool budget;request a smaller view"
            )
        low, high = 0, len(records)
        while low < high:
            middle = (low + high + 1) // 2
            if fits(prefix(middle)):
                low = middle
            else:
                high = middle - 1
        return _json(prefix(low))

    def tool(self, args):
        self._remaining()
        if not args:
            return _json(dict(help=self.HELP))
        command, *values = args
        if len(values) > 3 or any(len(value) > 2000 for value in values):
            raise ValueError("bounded memory arguments required")
        if command == "history" and len(values) == 1:
            terms = set(re.findall(r"\w+", values[0].lower()))
            found = []
            for row in self.index.db.execute(
                "SELECT id,sha,message FROM native_history WHERE id>=2"
            ):
                message = json.loads(row["message"])
                text = message.get("content", "")
                score = sum(term in text.lower() for term in terms)
                if score:
                    found.append(
                        (
                            score,
                            row["id"],
                            dict(
                                id=row["id"],
                                role=message["role"],
                                sha256=row["sha"],
                                chars=len(text),
                                status="historical;agent claims unverified",
                                next="ic-memory event " + str(row["id"]),
                            ),
                        )
                    )
            found.sort(key=lambda value: (value[0], value[1]), reverse=True)
            return self._bounded("history", [row[2] for row in found[:8]], matches=len(found))
        if command == "event" and len(values) in (1, 3):
            number = int(values[0])
            start, end = map(int, values[1:]) if len(values) == 3 else (0, 1600)
            if number < 2 or not 0 <= start < end or end - start > 4000:
                raise ValueError(
                    "event requires id>=2 and an explicit character range of at most4000"
                )
            row = self.index.db.execute(
                "SELECT sha,message FROM native_history WHERE id=?", (number,)
            ).fetchone()
            if row is None:
                raise KeyError("unknown history id")
            message = json.loads(row["message"])
            text = message["content"]
            return self._bounded(
                "historical_message",
                [
                    dict(
                        id=number,
                        role=message["role"],
                        sha256=row["sha"],
                        start_char=start,
                        end_char=min(end, len(text)),
                        total_chars=len(text),
                        exact_fragment=text[start:end],
                        status="historical;agent claims unverified",
                    )
                ],
            )

        # No stored source view is served until current files have been checked.
        self.index.refresh()
        metadata = dict(
            index_revision=self.index.revision(),
            limitations="Static candidates;unknown/unresolved relationships are not absent dependencies.",
        )
        if command == "route" and len(values) == 1:
            groups = self.index.memory_route(values[0])
            return self._bounded(
                "modules", groups["groups"], next="ic-memory find QUERY", **metadata
            )
        if command == "find" and len(values) == 1:
            rows = self.index.search(values[0], limit=8)
            keys = (
                "id",
                "path",
                "name",
                "qualname",
                "kind",
                "line",
                "end",
                "signature",
                "confidence",
            )
            records = [{k: row[k] for k in keys if k in row} for row in rows]
        elif command == "symbol" and len(values) == 1:
            symbol = self.index.symbol(values[0])
            row = self.index.db.execute(
                "SELECT hash,source FROM files WHERE path=?", (symbol["path"],)
            ).fetchone()
            records = [
                dict(
                    symbol={k: v for k, v in symbol.items() if k not in ("doc", "fields")},
                    source_sha256=row["hash"],
                )
            ]
            lines = row["source"].splitlines()
            stop = min(symbol["end"], len(lines), symbol["line"] + 59)
            metadata["source_lines_outside_view"] = max(0, symbol["end"] - stop)
            records += [
                dict(path=symbol["path"], line=i + 1, source_sha256=row["hash"], source=lines[i])
                for i in range(max(0, symbol["line"] - 1), stop)
            ]
        elif command == "source" and len(values) in (1, 3):
            path = values[0]
            start, end = map(int, values[1:]) if len(values) == 3 else (1, 40)
            if not 1 <= start <= end or end - start >= 80:
                raise ValueError("source needs1-based line numbers and at most80 lines")
            row = self.index.db.execute(
                "SELECT hash,source FROM files WHERE path=?", (path,)
            ).fetchone()
            if row is None:
                raise KeyError("path is not indexed")
            lines = row["source"].splitlines()
            metadata.update(path=path, source_sha256=row["hash"], total_lines=len(lines))
            records = [
                dict(path=path, line=i + 1, source_sha256=row["hash"], source=lines[i])
                for i in range(start - 1, min(end, len(lines)))
            ]
        elif command == "relations" and len(values) == 1:
            records = self.index.relationships(values[0], limit=24)
        elif command == "structure" and len(values) == 1:
            view = self.index.structure(values[0])
            records = [
                dict(
                    id=view["id"],
                    path=view["path"],
                    fields=view["fields"],
                    native_analysis=view.get("native_analysis"),
                )
            ] + view["field_accesses"]
        elif command == "flow" and len(values) in (1, 2):
            view = self.index.use_def(values[0], values[1] if len(values) == 2 else None, limit=24)
            records = view["edges"]
            metadata.update(
                path=view["path"],
                source_sha256=view["source_sha256"],
                analysis_limits=view["limitations"],
            )
        else:
            raise ValueError("unknown memory command/arguments;" + self.HELP)
        paths = {record["path"] for record in records if "path" in record}
        metadata["source_files"] = {
            row["path"]: row["hash"] for row in self.index.files() if row["path"] in paths
        }
        for path, expected in metadata["source_files"].items():
            current = (self.index.root / path).resolve()
            if (
                not current.is_relative_to(self.index.root)
                or not current.is_file()
                or hashlib.sha256(current.read_bytes()).hexdigest() != expected
            ):
                raise ValueError("source changed during memory query;refresh and query again")
        return self._bounded(command, records, **metadata)
