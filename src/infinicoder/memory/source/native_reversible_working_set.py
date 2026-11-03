"""Reversible physical context scheduling,not a model-generated summary."""

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re

from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.memory.native_memory import MemoryCapacityError, _json, _sha
from infinicoder.runtime.native_plain import NativePlainObserver


@dataclass(frozen=True)
class SourceAttachment:
    """Caller explicitly identifies only the serialized source part of a task."""

    literal: str
    files: dict

    def validate(self, task):
        if (
            not self.literal
            or task.count(self.literal) != 1
            or json.loads(self.literal) != self.files
        ):
            raise ValueError(
                "one exact JSON source attachment required;constraints remain verbatim"
            )
        if (
            not isinstance(self.files, dict)
            or not self.files
            or len(self.literal.encode()) > 16 * 1024**2
        ):
            raise ValueError("bounded nonempty source mapping required")
        for path, source in self.files.items():
            p = PurePosixPath(path)
            if (
                not isinstance(source, str)
                or p.is_absolute()
                or ".." in p.parts
                or str(p) != path
                or "\\" in path
                or ":" in path
                or any(x in (".git", ".infinicoder") for x in p.parts)
            ):
                raise ValueError("normalized public source attachment required")


class NativeWorkingSet(NativePlainObserver):
    identity = "native_reversible_working_set_v3_readonly_output_wrapper"
    expose_tools = True
    command_names = ("ic-workset",)
    HELP = (
        "Readonly queries,using the original agent action format:\n"
        "ic-workset files [QUERY]\nic-workset initial PATH START END\nic-workset source PATH START END\n"
        "ic-workset find QUERY\nic-workset graph SYMBOL_ID\nic-workset flow PATH [NAME]\n"
        "ic-workset event ID START END\n"
        "Use relative caller-authorized paths. Source lines are1-based,inclusive,at most80. "
        "Event IDs must be>=2;characters are0-based,end-exclusive,at most4000. "
        "One query per action;optional2>&1 and one head line filter are supported,for example "
        "ic-workset files 2>&1 | head -n 20. No other pipes,redirection,compound commands,expansion or byte filters. "
        "The underlying tool error code is preserved;head does not hide a failed query. "
        "Initial attachments and events are historical;source reads revalidate live bytes. "
        "Graph edges are static candidates;unknown edges are not absent dependencies. No command replay."
    )

    def execute_tool_command(self, command):
        # Other tool families retain their existing parser and status contract.
        if not re.match(r"^\s*ic-workset(?:\s|$)", command):
            return super().execute_tool_command(command)
        from infinicoder.memory.source.workset_output import parse_workset_output

        request = parse_workset_output(command)
        # The bound dispatcher has admitted this tool family. Preserve the
        # family's routing state without re-lexing away quoted literal operators.
        self._tool_family = request.argv[0]
        output = self.tool(list(request.argv[1:]))  # One readonly invocation,never replayed.
        if request.head_lines is None:
            return output
        lines = output.splitlines(keepends=True)
        selected = "".join(lines[: request.head_lines])
        if len(lines) <= request.head_lines:
            return output
        return self._output(
            dict(
                view="partial_tool_output",
                filter=dict(kind="head_lines", limit=request.head_lines),
                truncated=True,
                text=selected,
                complete_output_sha256=_sha(output),
                authority="Partial display only;not complete source evidence;request a narrower view",
            )
        )

    def __init__(
        self,
        *args,
        source_capability,
        attachment=None,
        namespace_binding=None,
        target_occupation=0.78,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sources = source_capability
        self.attachment = attachment
        self.namespace_binding = namespace_binding
        self.target_occupation = target_occupation

    def validate(self, request):
        super().validate(request)
        self.sources.validate(request.workspace, self.sources.paths)
        if self.attachment is not None:
            if not isinstance(self.attachment, SourceAttachment):
                raise ValueError("explicit SourceAttachment required")
            self.attachment.validate(request.spec.task)
            if not set(self.attachment.files).issubset(self.sources.paths):
                raise ValueError("attachment source capability incomplete")
        if not 0.7 <= self.target_occupation <= 0.8:
            raise ValueError("target occupation must be between70% and80%")

    def _namespace(self):
        callback = self.namespace_binding or getattr(
            self.environment, "implementation_workspace_binding", None
        )
        if not callable(callback):
            raise UnsupportedCore("explicit live source namespace binding required")
        value = callback()
        if Path(value["root"]).resolve() != self.workspace or not value.get("model_root"):
            raise UnsupportedCore("source namespace differs from actual candidate")
        return value

    @contextmanager
    def bind(self, request, model, environment, deadline):
        self.environment = environment
        self.working_paths = []
        self.invalidated = []
        self.records = []
        self.count_cache = {}
        previous_remaining = self.sources.remaining
        self.sources.remaining = self._remaining
        try:
            with super().bind(request, model, environment, deadline):
                self._namespace()
                self.target = int(self.limit * self.target_occupation)
                self.index.db.execute(
                    "CREATE TABLE working_initial(path TEXT PRIMARY KEY,sha TEXT,source TEXT)"
                )
                if self.attachment:
                    self.index.db.executemany(
                        "INSERT INTO working_initial VALUES(?,?,?)",
                        [(p, _sha(s), s) for p, s in self.attachment.files.items()],
                    )
                self.index.db.commit()
                try:
                    yield self
                finally:
                    (self.state / "native-working-set.json").write_text(
                        _json(
                            dict(
                                identity=self.identity,
                                records=self.records,
                                canonical_history_changed=False,
                                extra_model_calls=0,
                                initial_attachment_paths=list(self.attachment.files)
                                if self.attachment
                                else [],
                                capability_improvement_proven=False,
                            )
                        ),
                        encoding="utf8",
                    )
        finally:
            self.sources.remaining = previous_remaining

    def _refresh(self, paths):
        self._namespace()
        paths = list(dict.fromkeys(paths))
        previous = {r["path"]: r["hash"] for r in self.index.files() if r["path"] in paths}
        result = self.index.refresh_paths(paths, read_source=self.sources.read)
        changed = [p for p in result["changed"] if p in previous]
        self.invalidated = list(dict.fromkeys(self.invalidated + changed + result["deleted"]))[-8:]
        for path in paths:
            if path in self.working_paths:
                self.working_paths.remove(path)
            self.working_paths.append(path)
        self.working_paths = self.working_paths[-8:]
        return result

    def _interfaces(self):
        if self.working_paths:
            self._refresh(self.working_paths)
        rows = []
        for path in reversed(self.working_paths):
            file = self.index.db.execute(
                "SELECT hash,error FROM files WHERE path=?", (path,)
            ).fetchone()
            if file is None:
                continue
            syms = self.index.db.execute(
                "SELECT id,signature,line,end FROM symbols WHERE path=? ORDER BY line LIMIT 8",
                (path,),
            ).fetchall()
            rows.append(
                dict(
                    path=path,
                    sha256=file["hash"],
                    parse_error=file["error"],
                    interfaces=[dict(s) for s in syms],
                    coverage="declarations_only;request source for bodies",
                )
            )
        return rows

    def task_view(self, task):
        """Only caller-identified source data is externalized;no SDK template."""
        if self.attachment:
            literal = self.attachment.literal
            if task.count(literal) != 1:
                raise ValueError("explicit task attachment lost its canonical boundary")
            catalog = dict(
                layer="initial_source_archive",
                files=len(self.attachment.files),
                modules=sorted({str(PurePosixPath(p).parent) for p in self.attachment.files})[:8],
                source_set_sha256=_sha(_json(self.attachment.files)),
                retrieve="ic-workset files QUERY;ic-workset initial PATH START END",
                status="initial historical source;read current source before editing",
            )
            task = task.replace(literal, _json(catalog), 1)
        return task

    def _prefix(self, prepared):
        full = [dict(m) for m in prepared]
        full[1]["content"] = self.task_view(full[1]["content"])
        full[1]["content"] += "\n\nWORKING SET TOOLS\n" + self.HELP
        return full

    def _count(self, messages):
        self._remaining()
        key = _sha(_json(self.prepare(messages)))
        if key not in self.count_cache:
            result = super()._count(messages)
            if len(self.count_cache) >= 128:
                self.count_cache.pop(next(iter(self.count_cache)))
            self.count_cache[key] = result
        return self.count_cache[key]

    def _block_receipt(self, prepared, ids, characters=240, *, kind="action_and_actual_feedback"):
        quotes = []
        for i in ids:
            text = prepared[i]["content"]
            head = characters // 2
            tail = characters - head
            head_end = min(head, len(text))
            tail_start = max(head_end, len(text) - tail)
            quotes.append(
                dict(
                    event_id=i,
                    original_role=prepared[i]["role"],
                    sha256=self.seen[i],
                    characters=len(text),
                    head=dict(start=0, end=head_end, text=text[:head_end]),
                    tail=dict(start=tail_start, end=len(text), text=text[tail_start:]),
                    retrieve=f"ic-workset event {i} START END",
                )
            )
        executions = [
            e["id"]
            for e in self.executions
            if e.get("message_ids") and set(e["message_ids"]).issubset(ids)
        ]
        # A receipt is controller-authored USER data. It never impersonates a
        # former assistant response or pairs one command with another result.
        return dict(
            role="user",
            content=_json(
                dict(
                    layer="historical_action_feedback_receipt",
                    kind=kind,
                    canonical_event_ids=ids,
                    linked_executions=executions,
                    quotes=quotes,
                    provenance="quoted original action and associated SDK feedback;not a new model action",
                    completeness="exact referenced records remain retrievable;quotes are explicitly ranged",
                    commands_replayed=False,
                )
            ),
        )

    @staticmethod
    def _estimate(messages):
        # Conservative sizing chooses a trial only. Actual tokenizer admission
        # below is authoritative;UTF8 bytes are never reported as token counts.
        return sum(32 + len(m["content"].encode()) / 2 for m in messages)

    def project(self, messages):
        prepared = self._sync(messages)
        if (
            len(prepared) < 2
            or [m["role"] for m in prepared[:2]] != ["system", "user"]
            or any(not isinstance(m.get("content"), str) for m in prepared)
        ):
            raise UnsupportedCore("text SDK canonical task prefix required")
        exact_queries = 0

        def measure(value):
            nonlocal exact_queries
            exact_queries += 1
            if exact_queries > 9:
                raise MemoryCapacityError(
                    "bounded physical context assembly exhausted its nine exact-count queries"
                )
            return self._count(value)

        full = self._prefix(prepared)
        original_tokens = measure(prepared)
        full_tokens = measure(full)
        interfaces = self._interfaces()
        selected = list(range(len(full)))
        receipts = []
        prefix_ids, blocks = self._history_blocks(prepared)
        prefix = full[:2]
        packet = full
        optional = []
        mandatory = list(prefix)
        if full_tokens > self.target:
            selected = [0, 1]
            errors = prefix_ids[2:]
            if errors:
                mandatory.append(
                    self._block_receipt(
                        prepared, errors, 120, kind="initial_rejected_output_feedback"
                    )
                )
                receipts.append(errors)
            latest = blocks[-1] if blocks else []
            latest_whole = [full[i] for i in latest]
            if latest and self._estimate([*mandatory, *latest_whole]) <= self.target:
                mandatory += latest_whole
                selected += latest
            elif latest:
                mandatory.append(self._block_receipt(prepared, latest))
                receipts.append(latest)
            if measure(mandatory) > self.limit:
                # Drop quoted characters,but preserve the exact action/result
                # identities,association and references as one indivisible unit.
                mandatory = list(prefix)
                if errors:
                    mandatory.append(
                        self._block_receipt(
                            prepared, errors, 0, kind="initial_rejected_output_feedback"
                        )
                    )
                if latest:
                    mandatory.append(self._block_receipt(prepared, latest, 0))
                selected = [0, 1]
                receipts = ([errors] if errors else []) + ([latest] if latest else [])
            # Select older complete blocks using estimates,then fit the whole
            # trial in batches. No per-message full-history tokenizer loop.
            for ids in reversed(blocks[:-1]):
                trial = [full[i] for group in [ids, *optional] for i in group]
                if self._estimate([*mandatory, *trial]) <= self.target:
                    optional.insert(0, ids)
            lead = 1 if errors else 0

            def assemble():
                return [
                    *prefix,
                    *mandatory[2 : 2 + lead],
                    *[full[i] for group in optional for i in group],
                    *mandatory[2 + lead :],
                ]

            packet = assemble()
            trial_tokens = measure(packet)
            for _ in range(3):
                if trial_tokens <= self.limit:
                    break
                if not optional:
                    break
                optional = optional[len(optional) // 2 + 1 :]
                packet = assemble()
                trial_tokens = measure(packet)
            selected = sorted(set(selected + [i for group in optional for i in group]))
            if trial_tokens > self.limit:
                raise MemoryCapacityError(
                    "verbatim task and associated latest event receipt exceed capacity"
                )
        omitted = [i for i in range(2, len(full)) if i not in selected]
        note = dict(
            layer="working_set_directory",
            exact_archived_events=dict(first=2, last=len(full) - 1),
            omitted=len(omitted),
            recent_omitted=omitted[-6:],
            current_sources=interfaces,
            changed_or_deleted_sources=self.invalidated,
            history_is_not_current_source_proof=True,
        )
        # One directory trial,not an unbounded source-by-source fit loop. The
        # final exact count is retained even if the directory is omitted.
        extra = dict(role="user", content=_json(note))
        if exact_queries < 8 and measure([*packet, extra]) <= self.limit:
            packet = [*packet, extra]
        count = measure(packet)
        if count > self.limit:
            raise MemoryCapacityError("physical working set exceeds actual tokenizer limit")
        record = dict(
            original_tokens=original_tokens,
            projected_tokens=count,
            input_limit=self.limit,
            target=self.target,
            selected=selected,
            receipt_blocks=receipts,
            archived=omitted,
            exact_count_queries=exact_queries,
            selection_unit="assistant_action_and_all_associated_feedback",
            canonical_task_sha256=self.seen[1],
            attachment_archived=self.attachment is not None,
            invalidated=self.invalidated,
            namespace=self._namespace(),
            status="prepared;HTTP ledger establishes forwarding",
        )
        self.records.append(record)
        with self.index.db:
            self.index.db.execute(
                "INSERT INTO native_projections(messages,accounting) VALUES(?,?)",
                (_json(packet), _json(record)),
            )
        return packet

    def _output(self, value):
        text = _json(value)
        if self._count([dict(role="user", content=text)]) > min(1200, self.limit // 3):
            raise ValueError("view too large;request a narrower explicit range")
        return text

    def tool(self, args):
        if not args:
            raise ValueError(self.HELP)
        command, *values = args
        if command == "files" and len(values) <= 1:
            query = values[0].lower() if values else ""
            names = [p for p in self.sources.paths if query in p.lower()]
            return self._output(
                dict(
                    view="source_catalog",
                    paths=names[:40],
                    omitted=max(0, len(names) - 40),
                    coverage="caller readable paths;not edit permissions or repository closure",
                )
            )
        if command == "event" and len(values) == 3:
            number, start, end = map(int, values)
            if number < 2 or not 0 <= start < end <= start + 4000:
                raise ValueError("event id>=2;range at most4000")
            row = self.index.db.execute(
                "SELECT sha,message FROM native_history WHERE id=?", (number,)
            ).fetchone()
            if row is None:
                raise KeyError("unknown event")
            value = json.loads(row["message"])
            text = value["content"]
            return self._output(
                dict(
                    view="exact_historical_event",
                    id=number,
                    sha256=row["sha"],
                    role=value["role"],
                    start=start,
                    end=min(end, len(text)),
                    characters=len(text),
                    text=text[start:end],
                    commands_replayed=False,
                )
            )
        if command in ("initial", "source") and len(values) == 3:
            path, start, end = values[0], int(values[1]), int(values[2])
            if not 1 <= start <= end < start + 80:
                raise ValueError("line range1-based;at most80")
            if command == "source":
                self._refresh([path])
                row = self.index.db.execute(
                    "SELECT hash AS sha,source FROM files WHERE path=?", (path,)
                ).fetchone()
            else:
                row = self.index.db.execute(
                    "SELECT sha,source FROM working_initial WHERE path=?", (path,)
                ).fetchone()
            if row is None:
                raise KeyError("unknown source")
            lines = row["source"].splitlines()
            return self._output(
                dict(
                    view=command + "_source",
                    path=path,
                    sha256=row["sha"],
                    start=start,
                    end=min(end, len(lines)),
                    total_lines=len(lines),
                    complete_file=start == 1 and end >= len(lines),
                    exact_lines=[
                        dict(line=i + 1, text=lines[i])
                        for i in range(start - 1, min(end, len(lines)))
                    ],
                    historical=command == "initial",
                )
            )
        if command == "find" and len(values) == 1:
            query = values[0]
            words = set(re.findall(r"\w+", query.lower()))
            paths = sorted(
                self.sources.paths,
                key=lambda p: (-len(words & set(re.findall(r"\w+", p.lower()))), p),
            )[:8]
            self._refresh(paths)
            return self._output(
                dict(
                    view="symbol_candidates",
                    records=self.index.search(query, limit=6),
                    indexed_paths=paths,
                    coverage="selected sources only",
                )
            )
        if command == "graph" and len(values) == 1:
            path = values[0].split("::", 1)[0]
            self._refresh([path])
            return self._output(
                dict(
                    view="static_relation_candidates",
                    records=self.index.relationships(values[0], limit=12),
                    coverage="unobserved targets may be stale or unknown",
                )
            )
        if command == "flow" and len(values) in (1, 2):
            self._refresh([values[0]])
            return self._output(
                self.index.use_def(values[0], values[1] if len(values) == 2 else None, limit=12)
            )
        raise ValueError(self.HELP)
