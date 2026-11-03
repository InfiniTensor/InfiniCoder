"""Versioned source/hypothesis/observation state and whole-block projection.

All raw evidence stays in the existing task SQLite tables. This module never
generates, executes a candidate, infers acceptance from stdout or replays history.
"""

import copy
import hashlib
import json

from infinicoder.memory.native_memory import MemoryCapacityError
from infinicoder.memory.source.source_packet_budget import bounded_prefix
from infinicoder.planning.task_source_plan import SourcePlanPacket, PlanDraft, dump


def sha(value):
    return hashlib.sha256(dump(value).encode()).hexdigest()


def ranges(numbers):
    result = []
    for number in numbers:
        if result and result[-1][1] + 1 == number:
            result[-1][1] = number
        else:
            result.append([number, number])
    return result


class TaskEvidenceContext:
    def __init__(self, index, *, remaining):
        self.index, self.db, self.remaining = index, index.db, remaining
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS task_evidence_hypotheses(ref TEXT PRIMARY KEY,record TEXT);
          CREATE TABLE IF NOT EXISTS task_evidence_observations(ref TEXT PRIMARY KEY,execution_id INTEGER,record TEXT);
        """)
        self._view_cache = []
        self._view_rowid = 0

    def capture_plan(self, packet):
        if packet is None:
            return
        conditions = {f["path"]: f["source_sha256"] for f in packet.source_facts}
        for candidate in packet.static_candidates:
            conditions.update(candidate["support_sources"])
        record = dict(
            kind="model_hypothesis",
            text=packet.draft.text,
            status="unverified",
            response_ref=packet.response_ref,
            source_conditions=conditions,
            read_refs=sorted(
                {
                    v.get("read_event_ref")
                    for v in packet.source_facts + packet.static_candidates
                    if v.get("read_event_ref")
                }
            ),
            authority="Model suggestion;neither code fact,edit permission nor verified result",
        )
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO task_evidence_hypotheses VALUES(?,?)",
                (sha(record), dump(record)),
            )

    def capture_observations(self, prepared, blocks, executions, checks=()):
        by_message = {number: block for block in blocks for number in block}
        references = dict(executions={}, checks={})
        with self.db:
            for execution in executions:
                ids = execution.get("message_ids")
                if not ids or any(number not in by_message for number in ids):
                    continue
                complete = sorted({number for i in ids for number in by_message[i]})
                record = dict(
                    kind="runtime_observation",
                    execution_id=execution["id"],
                    message_ids=complete,
                    feedback_message_ids=[i for i in complete if prepared[i]["role"] == "user"],
                    message_sha256={str(i): sha(prepared[i]) for i in complete},
                    command=execution.get("command"),
                    returncode=execution.get("returncode"),
                    error=execution.get("error"),
                    observer_complete=execution.get("observer_complete"),
                    before=execution["before"],
                    after=execution["after"],
                    changed=execution.get("changed", []),
                    coverage="Explicit observed sources only;other changes unknown",
                    authority="Actual SDK action/feedback;self-test text is not independent acceptance",
                    feedback_observed_through=max(complete),
                )
                self.db.execute(
                    "INSERT OR IGNORE INTO task_evidence_observations VALUES(?,?,?)",
                    (sha(record), execution["id"], dump(record)),
                )
                references["executions"][execution["id"]] = sha(record)
            for check in checks:
                record = dict(
                    kind="trusted_public_check_observation",
                    event=copy.deepcopy(check),
                    authority="Actual caller checker callback;not model output or private final",
                )
                self.db.execute(
                    "INSERT OR IGNORE INTO task_evidence_observations VALUES(?,?,?)",
                    (sha(record), check.get("execution_id"), dump(record)),
                )
                references["checks"][check["id"]] = sha(record)
        return references

    def _views(self):
        tables = {
            r[0] for r in self.db.execute('SELECT name FROM sqlite_master WHERE type="table"')
        }
        if "source_plan_views" not in tables:
            return []
        for row in self.db.execute(
            "SELECT rowid,ref,sha256,record FROM source_plan_views WHERE rowid>? ORDER BY rowid",
            (self._view_rowid,),
        ):
            self.remaining()
            if hashlib.sha256(row["record"].encode()).hexdigest() != row["sha256"]:
                raise ValueError("stored exact source view hash differs")
            self._view_cache.append(json.loads(row["record"]))
            self._view_rowid = row["rowid"]
        # Source views are append-only task-owned records. Current source
        # versions are still independently rechecked on every projection.
        return self._view_cache

    def observed_paths(self):
        return tuple(
            dict.fromkeys(path for row in self._views() for path in row["view"]["source_files"])
        )

    def source_state(self, reader, *, refresh=False):
        views = self._views()
        latest = {}
        for row in views:
            for path, digest in row["view"]["source_files"].items():
                latest[path] = (digest, row)
        # A working set is bounded coverage, never a repository closure claim.
        selected = list(latest)[-8:]
        versions = {}
        invalid = []
        refreshed = []
        if reader is None:
            return dict(
                source_facts=[],
                static_candidates=[],
                invalidated_sources=[],
                current_versions={},
                refreshes=[],
                namespace=None,
                coverage="No actual source reader;source unknown",
            )
        for path in selected:
            expected, row = latest[path]
            try:
                observed = reader.current_version(path)
            except (OSError, ValueError, KeyError, UnicodeError):
                observed = None
            versions[path] = observed
            if observed == expected:
                continue
            invalid.append(
                dict(
                    path=path,
                    old_sha256=expected,
                    current_sha256=observed,
                    status="stale" if observed else "unknown_or_missing",
                )
            )
            if not refresh or observed is None or len(refreshed) >= 2:
                continue
            records = [
                r
                for r in row["view"]["records"]
                if r.get("path") == path and type(r.get("line")) is int
            ]
            if not records:
                continue
            first = min(r["line"] for r in records)
            last = min(first + 79, max(r["line"] for r in records))
            query = dict(op="inspect", target=path, start=first, end=last, view="source")
            try:
                view = reader.query(query)
            except (OSError, ValueError, KeyError, UnicodeError):
                continue
            versions.update(view["source_files"])
            refreshed.append(
                dict(
                    query=query,
                    read_event_ref=view["read_event_ref"],
                    semantic_alignment="Same historical location;interface alignment remains unverified",
                )
            )
        if refreshed:
            views = self._views()
        packet = SourcePlanPacket(reader.namespace, PlanDraft(""), "task-source-views")
        # Merge exact repeated lines by file/version; retain all read references.
        lines = {}
        static = {}
        for row in views:
            view = row["view"]
            support = view["source_files"]
            if not support or any(versions.get(path) != digest for path, digest in support.items()):
                continue
            for record in view["records"]:
                if record.get("kind") == "source_line":
                    key = (record["path"], record["source_sha256"], record["line"])
                    item = lines.setdefault(key, dict(text=record["source"], refs=[]))
                    if item["text"] != record["source"]:
                        raise ValueError("same source version has different exact text")
                    item["refs"].append(view["read_event_ref"])
                else:
                    key = sha(dict(record=record, support_sources=support))
                    static[key] = dict(
                        record=record,
                        support_sources=support,
                        read_event_ref=view["read_event_ref"],
                        status="Static candidate;runtime dispatch,values and ownership unknown",
                    )
        grouped = {}
        for (path, digest, line), value in sorted(lines.items()):
            grouped.setdefault((path, digest), []).append(
                dict(line=line, text=value["text"], read_refs=sorted(set(value["refs"])))
            )
        for (path, digest), values in grouped.items():
            for start in range(0, len(values), 8):
                packet.source_facts.append(
                    dict(
                        kind="source_record",
                        path=path,
                        source_sha256=digest,
                        lines=values[start : start + 8],
                        source_complete=False,
                        authority="Exact current read excerpt;not correctness proof",
                    )
                )
        return dict(
            source_facts=packet.source_facts,
            static_candidates=list(static.values()),
            invalidated_sources=invalid,
            current_versions=versions,
            refreshes=refreshed,
            namespace=reader.namespace,
            coverage="Selected observed files/excerpts only;unread modules and dynamic ownership unknown",
        )

    def hypotheses(self, versions):
        result = []
        for row in self.db.execute(
            "SELECT ref,record FROM task_evidence_hypotheses ORDER BY rowid DESC LIMIT 2"
        ):
            record = json.loads(row["record"])
            conditions = record["source_conditions"]
            stale = [
                path for path, expected in conditions.items() if versions.get(path) != expected
            ]
            result.append(
                dict(
                    ref=row["ref"],
                    preview=record["text"].encode()[:400].decode("utf8", errors="ignore"),
                    complete=False,
                    status="source_premise_changed_or_unknown;unverified"
                    if stale
                    else "unverified",
                    changed_or_unknown_premises=stale,
                    read_refs=record["read_refs"],
                )
            )
        return result

    def project(
        self,
        prepared,
        *,
        mandatory,
        blocks,
        episode,
        sources,
        archive,
        count,
        limit,
        edit_receipts=(),
    ):
        if any(type(i) is not int or not 0 <= i < len(prepared) for i in mandatory):
            raise ValueError("canonical mandatory message ids required")
        mandatory = set(mandatory)
        base = [copy.deepcopy(prepared[i]) for i in sorted(mandatory)]
        mandatory_tokens = count(base)
        if mandatory_tokens > limit:
            raise MemoryCapacityError("Exact task/latest/edit/check blocks exceed input capacity")
        hypotheses = self.hypotheses(sources["current_versions"])
        common = dict(
            origin="task_evidence_context_v1",
            source_facts=[],
            static_candidates=[],
            hypotheses=hypotheses,
            source_coverage=sources["coverage"],
            source_namespace=sources.get("namespace"),
            invalidated_sources=sources["invalidated_sources"],
            episode=episode,
            authority="Source excerpts are versioned;hypotheses unverified. Self-tests are observations,not final acceptance. "
            "Changed source is historical;ordinary coding remains available.",
        )
        if edit_receipts:
            common["current_edit_receipts"] = copy.deepcopy(list(edit_receipts))

        def assemble(ids, value):
            view = [copy.deepcopy(prepared[i]) for i in sorted(ids)]
            omitted = [i for i in range(2, len(prepared)) if i not in ids]
            context = copy.deepcopy(value)
            context["history"] = dict(
                command=archive["command"],
                omitted_ranges=ranges(omitted),
                omitted_messages=len(omitted),
                recent_refs=[dict(id=i, sha256=sha(prepared[i])) for i in omitted[-4:]],
                record_kinds=["event", "source", "hypothesis", "observation"],
                policy="Read exact archived records;commands are never replayed",
            )
            view.insert(2, dict(role="user", content="TASK EVIDENCE WORKING SET\n" + dump(context)))
            return view

        # Whole source records only. Each trial is counted as its actual full
        # chat template, not by character estimates or individual-record sums.
        records = [("source_facts", r) for r in sources["source_facts"]] + [
            ("static_candidates", r) for r in sources["static_candidates"]
        ]

        def source_packet(chosen, length):
            value = copy.deepcopy(common)
            for kind, record in chosen:
                value[kind].append(record)
            value["omitted_source_records"] = len(sources["source_facts"]) - len(
                value["source_facts"]
            )
            value["omitted_static_records"] = len(sources["static_candidates"]) - len(
                value["static_candidates"]
            )
            return value

        value = bounded_prefix(
            records,
            source_packet,
            lambda text: max(0, count(assemble(mandatory, json.loads(text))) - mandatory_tokens),
            max(1, limit - mandatory_tokens),
            count_limit=4,
        )
        if value is None:
            raise MemoryCapacityError("Mandatory evidence/history references exceed input capacity")
        optional = [
            dict(ids=block, messages=[prepared[i] for i in block])
            for block in reversed(blocks)
            if not set(block).issubset(mandatory)
        ]

        def selection(chosen, length):
            return dict(ids=sorted(mandatory | {i for record in chosen for i in record["ids"]}))

        selected = bounded_prefix(
            optional,
            selection,
            lambda text: count(assemble(set(json.loads(text)["ids"]), value)),
            limit,
            count_limit=4,
        )
        if selected is None:
            raise MemoryCapacityError("Mandatory task workset exceeds input capacity")
        ids = set(selected["ids"])
        projected = assemble(ids, value)
        tokens = count(projected)
        if tokens > limit:
            raise MemoryCapacityError("Final whole-message projection exceeds capacity")
        return projected, dict(
            compacted=True,
            selected=sorted(ids),
            omitted=[i for i in range(len(prepared)) if i not in ids],
            projected_tokens=tokens,
            input_limit=limit,
            canonical_history_changed=False,
            source_records=len(value["source_facts"]),
            static_records=len(value["static_candidates"]),
            current_source_versions=sources["current_versions"],
            source_refreshes=sources["refreshes"],
            hypothesis_states=[h["status"] for h in hypotheses],
            history_retrievable=True,
            model_input_sha256=sha(projected),
            status="ready;original_HTTP_proves_forwarding",
        )
