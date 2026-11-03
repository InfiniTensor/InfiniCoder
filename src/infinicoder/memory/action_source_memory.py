"""Action-anchored source facts and a separate bounded working view.

This component is an offline prototype. It does not change the running agent,
obtain editing permission,call a model,or manufacture missing source history.
"""

import ast
from dataclasses import dataclass
import hashlib
import json
import re
import textwrap

from infinicoder.memory.task_checkpoint_sources import _definitions, _relative
from infinicoder.memory.task_state_checkpoint import digest, dump


def _namespace(value):
    if not isinstance(value, dict) or not all(
        value.get(k) for k in ("environment", "root", "binding_identity")
    ):
        raise ValueError("verified source namespace required")
    return dump(value)


@dataclass(frozen=True)
class SourceSnapshot:
    path: str
    namespace_json: str
    raw: bytes
    version: str

    def __post_init__(self):
        if not _relative(self.path) or not isinstance(self.raw, bytes) or len(self.raw) > 1_000_000:
            raise ValueError("bounded immutable relative source required")
        if _namespace(json.loads(self.namespace_json)) != self.namespace_json:
            raise ValueError("canonical source namespace required")
        if hashlib.sha256(self.raw).hexdigest() != self.version:
            raise ValueError("source bytes/version differ")
        self.raw.decode("utf-8-sig")

    @property
    def text(self):
        return self.raw.decode("utf-8-sig")

    def record(self):
        return dict(
            path=self.path,
            namespace=json.loads(self.namespace_json),
            whole_file_sha256=self.version,
            source_utf8=self.raw.decode("utf8"),
            authority="Bytes at an attributed read;not runtime behavior or editing permission",
        )


def read_snapshot(reader, path, remaining):
    """Only the caller-bound reader can produce a new current observation."""
    remaining()
    namespace = _namespace(reader.namespace)
    raw, text, version = reader.session.read(path, max_bytes=1_000_000)
    if raw.decode("utf-8-sig") != text or reader.current_version(path) != version:
        raise ValueError("source changed during namespace read")
    if _namespace(reader.namespace) != namespace:
        raise ValueError("source namespace changed")
    return SourceSnapshot(path, namespace, raw, version)


@dataclass(frozen=True)
class SourceRevisionRecord:
    path: str
    namespace_json: str
    before: SourceSnapshot | None
    after: SourceSnapshot | None
    execution_id: int
    message_ids: tuple
    before_version: str | None


def observe_revision(
    path,
    namespace,
    before_bytes_ref,
    after_bytes_ref,
    execution_id,
    linked_feedback,
    *,
    before_version=None,
):
    """A before body is usable only if it matches the actual pre-action version.

    A prior read/starter is not automatically the immediate pre-action body.
    Missing after bytes mean unknown;this API does not infer a deletion.
    """
    identity = _namespace(namespace)
    ids = tuple(linked_feedback)
    if not _relative(path) or type(execution_id) is not int or execution_id < 0 or not ids:
        raise ValueError("attributed source action required")
    if any(type(i) is not int or i < 2 for i in ids):
        raise ValueError("linked feedback ids required")
    for snapshot in (before_bytes_ref, after_bytes_ref):
        if snapshot is not None and (
            not isinstance(snapshot, SourceSnapshot)
            or snapshot.path != path
            or snapshot.namespace_json != identity
        ):
            raise ValueError("source action crosses path or namespace")
    if before_bytes_ref is not None and before_bytes_ref.version != before_version:
        raise ValueError("historical body does not establish the actual before version")
    return SourceRevisionRecord(
        path, identity, before_bytes_ref, after_bytes_ref, execution_id, ids, before_version
    )


@dataclass(frozen=True)
class FocusRecord:
    snapshot: SourceSnapshot
    definition_json: str
    execution_id: int
    message_ids: tuple
    anchor: str
    before: SourceSnapshot | None = None

    @property
    def definition(self):
        return json.loads(self.definition_json)

    def record(self):
        definition = self.definition
        return dict(
            path=self.snapshot.path,
            definition=definition,
            whole_file_sha256=self.snapshot.version,
            definition_sha256=hashlib.sha256(definition["text"].encode()).hexdigest(),
            namespace=json.loads(self.snapshot.namespace_json),
            execution_id=self.execution_id,
            message_ids=list(self.message_ids),
            anchor=self.anchor,
            current_at_read=True,
            after=self.snapshot.record(),
            before=self.before.record() if self.before else None,
            write_authority="original_TaskSpec_and_public_contract_only",
            authority="Complete syntax at an attributed read;aliasing,ownership and runtime relationships remain unknown",
        )


def _all_definitions(snapshot, reader, remaining):
    remaining()
    rows = _definitions(snapshot.path, snapshot.text, reader, remaining)
    # State declarations are also complete syntax units even in modules with functions.
    if snapshot.path.endswith((".py", ".pyi")):
        lines = snapshot.text.splitlines(keepends=True)
        for node in ast.parse(snapshot.text).body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and not any(
                r["start"] == node.lineno and r["end"] == node.end_lineno for r in rows
            ):
                rows.append(
                    dict(
                        symbol="module-definition@" + str(node.lineno),
                        kind="module_definition",
                        start=node.lineno,
                        end=node.end_lineno,
                        text="".join(lines[node.lineno - 1 : node.end_lineno]),
                        boundary="python_ast_complete_definition",
                        lexical_interface=None,
                    )
                )
    return rows


def _innermost(rows, first, last):
    """Return complete units actually intersecting a range,not the shortest in a file."""
    selected = {}
    for line in range(first, last + 1):
        candidates = [r for r in rows if r["start"] <= line <= r["end"]]
        if not candidates:
            continue
        row = min(candidates, key=lambda r: (r["end"] - r["start"], -r["start"], r["symbol"]))
        selected[(row["symbol"], row["start"], row["end"])] = row
    result = list(selected.values())
    # A changed class header requires the whole class,which already contains its methods.
    return [
        r
        for r in result
        if not any(
            p is not r
            and p["start"] <= r["start"]
            and r["end"] <= p["end"]
            and (p["start"], p["end"]) != (r["start"], r["end"])
            for p in result
        )
    ]


def _definition_key(snapshot, row):
    if row["kind"] == "module_definition" and snapshot.path.endswith((".py", ".pyi")):
        node = ast.parse(textwrap.dedent(row["text"])).body[0]
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return (
            "module_definition",
            tuple(ast.dump(target, include_attributes=False) for target in targets),
        )
    return (row["kind"], row["symbol"])


def _own_syntax(snapshot, row):
    if not snapshot.path.endswith((".py", ".pyi")):
        return row["text"]
    root = ast.parse(textwrap.dedent(row["text"])).body[0]

    class LocalBody(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            return self.generic_visit(node) if node is root else ast.Pass()

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef

    # A child implementation changing does not falsely identify its entire
    # enclosing class/function as the changed implementation.
    return ast.dump(LocalBody().visit(root), include_attributes=False)


def _changed(revision, rows, reader, remaining):
    if (
        revision.before is None
        or revision.after is None
        or revision.before.version == revision.after.version
    ):
        return []
    old = {}
    new = {}
    for snapshot, definitions, target in (
        (revision.before, _all_definitions(revision.before, reader, remaining), old),
        (revision.after, rows, new),
    ):
        for row in definitions:
            target.setdefault(_definition_key(snapshot, row), []).append(row)
    selected = []
    for key, candidates in new.items():
        remaining()
        # Multiple lexical definitions/overloads require more evidence. Never
        # guess which same-name implementation is the actual changed target.
        if len(candidates) != 1 or len(old.get(key, [])) > 1:
            continue
        row = candidates[0]
        previous = old.get(key, [])
        if not previous or _own_syntax(revision.after, row) != _own_syntax(
            revision.before, previous[0]
        ):
            selected.append(row)
    return sorted(selected, key=lambda r: (r["start"], r["end"], r["symbol"]))


def choose_focus(
    events,
    revision_records,
    snapshots,
    public_task,
    *,
    reader=None,
    remaining=lambda: None,
    diagnostics=None,
):
    """Choose at most two actual action targets;ambiguous whole-file reads abstain.

    Action recency determines working-set movement. Within one action,verified
    changed ranges precede an exact delivered range. Public path::qualname is
    only a final exact tie-breaker;unverified memo paths never participate.
    """
    if not isinstance(public_task, str):
        raise ValueError("original public task required")
    snapshots = tuple(snapshots)
    current = {s.path: s for s in snapshots}
    namespaces = {s.namespace_json for s in snapshots}
    if len(namespaces) > 1:
        raise ValueError("working set crosses source namespace")
    if len(current) != len(snapshots):
        raise ValueError("one explicit current revision per path required")
    revisions = {}
    for record in revision_records:
        revisions.setdefault(record.execution_id, []).append(record)
    cache = {}
    focus = []
    seen = set()
    diagnostics = [] if diagnostics is None else diagnostics

    def definitions(snapshot):
        key = (snapshot.path, snapshot.version)
        if key not in cache:
            try:
                cache[key] = _all_definitions(snapshot, reader, remaining)
            except TimeoutError:
                raise
            except (SyntaxError, ValueError, ImportError, OSError, UnicodeError) as error:
                cache[key] = []
                diagnostics.append(
                    dict(path=snapshot.path, status="unknown", reason=type(error).__name__)
                )
        return cache[key]

    def add(snapshot, row, event, anchor, before=None):
        key = (snapshot.path, row["symbol"], row["start"], row["end"], snapshot.version)
        if key not in seen and len(focus) < 2:
            seen.add(key)
            focus.append(
                FocusRecord(
                    snapshot, dump(row), event["id"], tuple(event["message_ids"]), anchor, before
                )
            )

    for event in sorted(events, key=lambda e: e.get("id", -1), reverse=True):
        remaining()
        if len(focus) == 2:
            break
        if (
            not event.get("message_ids")
            or not event.get("observer_complete")
            or event.get("error") is not None
        ):
            continue
        anchored = False
        for revision in revisions.get(event["id"], ()):
            snapshot = current.get(revision.path)
            if (
                snapshot is None
                or revision.after != snapshot
                or revision.message_ids != tuple(event["message_ids"])
            ):
                continue
            try:
                changed = _changed(revision, definitions(snapshot), reader, remaining)
            except TimeoutError:
                raise
            except (SyntaxError, ValueError, ImportError, OSError, UnicodeError) as error:
                changed = []
                diagnostics.append(
                    dict(
                        path=snapshot.path,
                        status="changed_definition_unknown",
                        reason=type(error).__name__,
                    )
                )
            for row in changed:
                add(snapshot, row, event, "observed_changed_definition", revision.before)
                anchored = True
        observation = event.get("source_read") or {}
        snapshot = current.get(observation.get("path"))
        if anchored or snapshot is None or observation.get("status") != "linked_exact_output":
            continue
        first, last = observation["lines"]
        lines = snapshot.raw.splitlines(keepends=True)
        if (
            type(first) is not int
            or first < 1
            or last is not None
            and (type(last) is not int or last < first)
        ):
            diagnostics.append(
                dict(path=snapshot.path, status="unknown", reason="invalid delivered source range")
            )
            continue
        if b"".join(lines[first - 1 : last]).decode("utf8") != observation["output"]:
            continue
        rows = definitions(snapshot)
        if not observation.get("whole_file"):
            for row in _innermost(rows, first, last or len(lines)):
                add(snapshot, row, event, "delivered_read_range")
        else:
            # A whole file containing one unique definition is unambiguous.
            if len(rows) == 1:
                add(snapshot, rows[0], event, "delivered_unique_definition")
            for row in rows:
                anchor = snapshot.path + "::" + row["symbol"]
                if re.search(r"(?<![\w./])" + re.escape(anchor) + r"(?!\w|\.\w)", public_task):
                    add(snapshot, row, event, "explicit_public_symbol")
    return tuple(focus)


class SourceFocusArchive:
    """Use the existing source-plan table and readonly history reader;no new index."""

    def __init__(self, reader):
        self.reader = reader
        self.db = reader.index.db
        self.namespace_json = _namespace(reader.namespace)
        self.namespace = digest(json.loads(self.namespace_json))
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS action_source_aliases(id INTEGER PRIMARY KEY,namespace TEXT,ref TEXT,UNIQUE(namespace,ref))"
        )
        self.db.commit()

    def put(self, focus, remaining=lambda: None):
        remaining()
        if (
            _namespace(self.reader.namespace) != self.namespace_json
            or focus.snapshot.namespace_json != self.namespace_json
        ):
            raise ValueError("focus belongs to another namespace")
        definition = focus.definition
        if definition["boundary"] == "python_ast_complete_definition":
            lines = focus.snapshot.text.splitlines(keepends=True)
            if "".join(lines[definition["start"] - 1 : definition["end"]]) != definition["text"]:
                raise ValueError("definition bytes do not match the current source record")
        payload = focus.record()
        ref = "source-view:" + digest(payload)[:32]
        existing = self.db.execute(
            "SELECT record FROM source_plan_views WHERE ref=?", (ref,)
        ).fetchone()
        if existing:
            if json.loads(existing[0])["view"]["records"] != [payload]:
                raise ValueError("source reference collision")
        else:
            view = dict(
                read_event_ref=ref,
                view="action_anchored_definition",
                source_files={focus.snapshot.path: focus.snapshot.version},
                records=[payload],
                authority=payload["authority"],
                origin="actual_action_source_record",
            )
            self.reader._record_view(
                dict(
                    op="action_source_focus", execution_id=focus.execution_id, anchor=focus.anchor
                ),
                view,
            )
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO action_source_aliases(namespace,ref) VALUES(?,?)",
                (self.namespace, ref),
            )
        row = self.db.execute(
            "SELECT id FROM action_source_aliases WHERE namespace=? AND ref=?",
            (self.namespace, ref),
        ).fetchone()
        return "s" + str(row[0]), ref

    def get(self, alias):
        if not re.fullmatch(r"s[1-9][0-9]*", alias):
            raise ValueError("exact task-local source alias required")
        if _namespace(self.reader.namespace) != self.namespace_json:
            raise ValueError("source reader namespace changed")
        row = self.db.execute(
            "SELECT v.record,v.sha256 FROM action_source_aliases a JOIN source_plan_views v ON v.ref=a.ref WHERE a.id=? AND a.namespace=?",
            (int(alias[1:]), self.namespace),
        ).fetchone()
        if row is None:
            raise ValueError("source alias absent from this namespace")
        if hashlib.sha256(row[0].encode()).hexdigest() != row[1]:
            raise ValueError("durable source record changed")
        record = json.loads(row[0])
        if _namespace(record["namespace"]) != self.namespace_json:
            raise ValueError("source record namespace differs")
        return record


def render_focus(records, archive, current_versions, budget):
    """Separate durable provenance from a whole-definition model view."""
    if len(records) > 2:
        raise ValueError("at most two source focus slots")
    text = "Current source observations. Syntax only;runtime/ownership unknown. Original task alone controls edits.\n"
    count = budget["count"]
    remaining = budget["remaining"]
    limit = budget["source_tokens"]
    included = []
    omitted = []
    remaining()
    tokens = count([dict(role="user", content=text)])
    if tokens > limit:
        return dict(
            status="N_A;source header does not fit", text="", tokens=tokens, included=[], omitted=[]
        )
    for focus in records:
        remaining()
        alias, ref = archive.put(focus, remaining)
        definition = focus.definition
        identity = focus.snapshot.path + "::" + definition["symbol"]
        if current_versions.get(focus.snapshot.path) != focus.snapshot.version:
            omitted.append(
                dict(
                    alias=alias,
                    ref=ref,
                    path=focus.snapshot.path,
                    status="unknown;source revision changed",
                )
            )
            continue
        block = (
            f"[{alias} {ref}] {identity} L{definition['start']}-{definition['end']} current_at_read; "
            f"{focus.anchor} event:{focus.execution_id}\n" + definition["text"]
        )
        if definition.get("lexical_interface"):
            block += (
                "\nDirect lexical class header (runtime dispatch unknown):\n"
                + definition["lexical_interface"]["text"]
            )
        trial = text + "\n" + block
        remaining()
        measured = count([dict(role="user", content=trial)])
        if measured <= limit:
            text, tokens = trial, measured
            included.append(
                dict(alias=alias, ref=ref, path=focus.snapshot.path, version=focus.snapshot.version)
            )
        else:
            omitted.append(
                dict(
                    alias=alias,
                    ref=ref,
                    path=focus.snapshot.path,
                    status="omitted;complete definition exceeds budget",
                )
            )
    return dict(
        status="ready" if included else "N_A;no complete current focus fits",
        text=text if included else "",
        tokens=tokens,
        included=included,
        omitted=omitted,
        provenance="Full namespace,revisions and source bytes in readonly task history source records",
    )
