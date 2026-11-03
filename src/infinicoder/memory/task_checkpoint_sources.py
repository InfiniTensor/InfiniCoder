"""Bounded current implementation bytes for a task checkpoint;never edit scope."""

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import textwrap

from infinicoder.memory.task_state_checkpoint import digest, dump


AUTHORITY = (
    "Complete definition bytes at a verified read;not runtime behavior,aliasing,"
    "ownership,permission or acceptance. Original task controls editing."
)


@dataclass(frozen=True)
class SourceStateBundle:
    payload_json: str

    @property
    def payload(self):
        return json.loads(self.payload_json)

    @property
    def paths(self):
        return tuple(row["path"] for row in self.payload["sources"])


def _relative(path):
    if not isinstance(path, str) or not path or "\\" in path or ":" in path or "\0" in path:
        return False
    value = PurePosixPath(path)
    return (
        not value.is_absolute()
        and ".." not in value.parts
        and str(value) == path
        and value.parts[0] not in (".git", ".infinicoder")
    )


def _python_definitions(path, source):
    """Whole definitions,including decorators;no execution or inferred ownership."""
    tree = ast.parse(source, filename=path)
    lines = source.splitlines(keepends=True)
    found = []

    def walk(node, parents=()):
        for child in ast.iter_child_nodes(node):
            named = isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            if named:
                start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                end = child.end_lineno
                row = dict(
                    symbol=".".join([p.name for p in parents] + [child.name]),
                    kind="class" if isinstance(child, ast.ClassDef) else "function",
                    start=start,
                    end=end,
                    text="".join(lines[start - 1 : end]),
                    boundary="python_ast_complete_definition",
                    lexical_interface=None,
                )
                if parents:
                    parent = parents[-1]
                    if isinstance(parent, ast.ClassDef):
                        first = parent.body[0]
                        body_start = min(
                            [first.lineno]
                            + [d.lineno for d in getattr(first, "decorator_list", ())]
                        )
                        if body_start > parent.lineno:
                            header = "".join(lines[parent.lineno - 1 : body_start - 1])
                        else:
                            header = lines[parent.lineno - 1].encode()[: first.col_offset].decode()
                        row["lexical_interface"] = dict(
                            symbol=".".join(p.name for p in parents),
                            text=header,
                            authority="Direct lexical class header only;not ownership or runtime dispatch",
                        )
                found.append(row)
                walk(child, parents + (child,))
            else:
                walk(child, parents)

    walk(tree)
    # A module consisting only of a single assignment still has a real bounded
    # definition. Do not turn a line preview or function header into a body.
    if not found:
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                found.append(
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
    return found


def _native_definitions(path, source, reader, remaining):
    """Only a trusted actual-namespace parser can establish native extents."""
    parser = getattr(reader, "checkpoint_native_definitions", None)
    if not callable(parser):
        return []
    return parser(path, source, remaining)


def _definitions(path, source, reader, remaining):
    if Path(path).suffix.lower() in (".py", ".pyi"):
        return _python_definitions(path, source)
    if Path(path).suffix.lower() in (
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
        ".h",
        ".hh",
        ".hpp",
        ".cu",
        ".cuh",
    ):
        return _native_definitions(path, source, reader, remaining)
    return []


def _candidate_paths(events):
    changed = []
    read = []
    for event in reversed(events):
        if (
            not event.get("message_ids")
            or not event.get("observer_complete")
            or event.get("error") is not None
        ):
            continue
        for path in event.get("changed", ()):
            if _relative(path) and path not in [p[0] for p in changed]:
                changed.append((path, event, "recent_observed_file_change"))
        observation = event.get("source_read")
        if observation and observation.get("status") == "linked_exact_output":
            path = observation["path"]
            if _relative(path) and path not in [p[0] for p in read]:
                read.append((path, event, "recent_delivered_source_read"))
    return (changed + [row for row in read if row[0] not in [p[0] for p in changed]])[:2]


def validate_checkpoint_sources(bundle, reader=None):
    if not isinstance(bundle, SourceStateBundle):
        raise ValueError("immutable source-state bundle required")
    payload = bundle.payload
    rows = payload["sources"]
    namespace = payload["namespace"]
    if len(rows) > 2 or len({row["path"] for row in rows}) != len(rows):
        raise ValueError("at most two distinct source slots")
    if (
        not namespace.get("environment")
        or not namespace.get("root")
        or not namespace.get("binding_identity")
    ):
        raise ValueError("caller-verified current source namespace required")
    if reader is not None and namespace != reader.namespace:
        raise ValueError("source namespace changed")
    for row in rows:
        if (
            not _relative(row["path"])
            or row["write_authority"] != "original_TaskSpec_and_public_contract_only"
        ):
            raise ValueError("invalid source path or write authority")
        if row["status"] != "current_at_read":
            if "definition" in row:
                raise ValueError("unknown/stale source slot retains current body")
            continue
        definition = row["definition"]
        if (
            not row["current_at_read"]
            or definition["boundary"]
            not in ("python_ast_complete_definition", "clang_ast_complete_definition")
            or hashlib.sha256(definition["text"].encode()).hexdigest() != row["definition_sha256"]
            or len(row["whole_file_sha256"]) != 64
            or definition["end"] < definition["start"]
        ):
            raise ValueError("incomplete or inconsistent definition")
        if reader is not None and reader.current_version(row["path"]) != row["whole_file_sha256"]:
            raise ValueError("source revision changed during bundle transaction")
    return payload


def _read_definition(reader, path, event, selection, remaining, *, symbol=None, prior_reads=()):
    remaining()
    raw, source, version = reader.session.read(path, max_bytes=1_000_000)
    if (
        hashlib.sha256(raw).hexdigest() != version
        or raw.decode("utf-8-sig") != source
        or reader.current_version(path) != version
    ):
        raise ValueError("actual namespace source version differs")
    if selection == "recent_delivered_source_read":
        from infinicoder.memory.source.observed_source_reads import lf_lines

        observed = event["source_read"]
        lines = lf_lines(raw)
        first, last = observed["lines"]
        fragment = b"".join(lines[first - 1 : last])
        if fragment.decode("utf8") != observed["output"]:
            raise ValueError("delivered historical source does not match this current revision")
    if Path(path).suffix.lower() in (".py", ".pyi"):
        reader.session.refresh_paths((path,))
    definitions = _definitions(path, source, reader, remaining)
    if symbol is not None:
        definitions = [row for row in definitions if row["symbol"] == symbol]
    if not definitions:
        raise ValueError("complete definition unavailable")
    changed_since = None
    if symbol is None and Path(path).suffix.lower() in (".py", ".pyi"):
        for previous in reversed(prior_reads):
            observed = previous.get("source_read", {})
            if (
                previous["id"] >= event["id"]
                or observed.get("path") != path
                or observed.get("status") != "linked_exact_output"
                or not observed.get("whole_file")
            ):
                continue

            def syntax_key(row):
                return ast.dump(ast.parse(textwrap.dedent(row["text"])), include_attributes=False)

            try:
                old = {
                    row["symbol"]: syntax_key(row)
                    for row in _python_definitions(path, observed["output"])
                }
            except (SyntaxError, ValueError):
                continue
            different = [row for row in definitions if old.get(row["symbol"]) != syntax_key(row)]
            if different:
                definitions = different
                changed_since = previous["id"]
            break
    # Changed-file provenance is known;the exact changed definition is unknown
    # without an older body. Prefer the smallest whole callable,never a preview.
    definition = min(
        definitions,
        key=lambda row: (
            row["kind"] == "class",
            len(row["text"].encode()),
            row["start"],
            row["symbol"],
        ),
    )
    if reader.current_version(path) != version:
        raise ValueError("source changed while selecting a complete definition")
    row = dict(
        path=path,
        status="current_at_read",
        current_at_read=True,
        whole_file_sha256=version,
        definition=definition,
        definition_sha256=hashlib.sha256(definition["text"].encode()).hexdigest(),
        selection=selection,
        selection_execution_id=event["id"],
        selection_message_ids=list(event["message_ids"]),
        changed_definition="differs_from_delivered_definition"
        if changed_since is not None
        else "unknown;file-level observation only",
        prior_definition_read_execution_id=changed_since,
        origin="controller_fresh_read",
        read_authority="caller_bound_namespace_read_capability",
        write_authority="original_TaskSpec_and_public_contract_only",
        authority=AUTHORITY,
    )
    row["ref"] = "source-view:" + digest(dict(namespace=reader.namespace, row=row))[:32]
    return row


def _unknown_slot(row, reason):
    result = dict(
        path=row["path"],
        status="unknown",
        current_at_read=False,
        selected_symbol=row.get("definition", {}).get("symbol") or row.get("selected_symbol"),
        selection_execution_id=row["selection_execution_id"],
        selection_message_ids=row["selection_message_ids"],
        write_authority="original_TaskSpec_and_public_contract_only",
        reason=reason,
    )
    if row.get("historical_ref"):
        result["historical_ref"] = row["historical_ref"]
    return result


def _bounded_payload(namespace, rows, omissions, budget):
    def payload(selected):
        return dict(
            namespace=namespace,
            sources=selected,
            omissions=omissions,
            coverage="At most two selected source files;all other files and ownership unknown",
            authority=AUTHORITY,
            refresh_policy="Selected slots only;source bodies invalidated on revision change;memo remains historical",
        )

    count = budget["count"]
    limit = budget["source_tokens"]
    budget["remaining"]()
    selected = [
        _unknown_slot(row, "complete definition omitted by source budget")
        if row["status"] == "current_at_read"
        else row
        for row in rows
    ]
    measured = count([dict(role="user", content=dump(payload(selected)))])
    if measured > limit:
        return None, measured
    # At most one whole-definition admission per fixed slot. Never trim a body
    # or repeatedly tokenize source fragments to fill a quota.
    for position, row in enumerate(rows):
        if row["status"] != "current_at_read":
            continue
        trial = list(selected)
        trial[position] = row
        budget["remaining"]()
        tokens = count([dict(role="user", content=dump(payload(trial)))])
        if tokens <= limit:
            selected, measured = trial, tokens
    return SourceStateBundle(dump(payload(selected))), measured


def build_checkpoint_sources(public_task, linked_events, source_capability, versions, budget):
    """Return current whole syntax,using only event-selected paths and a reader."""
    if not isinstance(public_task, str) or not public_task.strip():
        raise ValueError("original public task required")
    reader = source_capability
    if not getattr(reader, "namespace", {}).get("binding_identity"):
        return None, dict(status="N_A;no_verified_source_namespace")
    rows = []
    omissions = []
    for path, event, selection in _candidate_paths(linked_events):
        try:
            rows.append(
                _read_definition(
                    reader, path, event, selection, budget["remaining"], prior_reads=linked_events
                )
            )
        except (OSError, ValueError, UnicodeError, SyntaxError, ImportError) as error:
            omissions.append(
                dict(
                    path=path,
                    status="unknown",
                    reason=type(error).__name__ + ": " + str(error)[:160],
                )
            )
    if not rows:
        return None, dict(status="N_A;no_complete_current_definition", omissions=omissions)
    bundle, measured = _bounded_payload(reader.namespace, rows, omissions, budget)
    if bundle is not None and not any(
        row["status"] == "current_at_read" for row in bundle.payload["sources"]
    ):
        bundle = None
    if bundle is not None:
        validate_checkpoint_sources(bundle, reader)
    return bundle, dict(
        status="ready" if bundle else "N_A;whole_source_bundle_exceeds_budget",
        source_tokens=measured,
        source_limit=budget["source_tokens"],
        selected_paths=[row["path"] for row in rows],
        declared_versions_are="identity hints only;fresh source reader is authority",
        observed_version_paths=sorted(versions),
    )


def refresh_checkpoint_sources(bundle, reader, budget):
    """Refresh only fixed chosen slots;never add files or retain a stale body."""
    old = validate_checkpoint_sources(bundle)
    rows = []
    omissions = []
    changed = False
    if old["namespace"] != reader.namespace:
        raise ValueError("source capability namespace changed")
    for before in old["sources"]:
        path = before["path"]
        row = before
        try:
            budget["remaining"]()
            version = reader.current_version(path)
            if before["status"] != "current_at_read" or version != before["whole_file_sha256"]:
                changed = True
                event = dict(
                    id=before["selection_execution_id"], message_ids=before["selection_message_ids"]
                )
                symbol = before.get("definition", {}).get("symbol") or before.get("selected_symbol")
                row = _read_definition(
                    reader,
                    path,
                    event,
                    "refresh_selected_source",
                    budget["remaining"],
                    symbol=symbol,
                )
                row["historical_ref"] = before.get("ref") or before.get("historical_ref")
        except (OSError, ValueError, UnicodeError, SyntaxError, ImportError) as error:
            changed = True
            row = dict(
                path=path,
                status="unknown",
                current_at_read=False,
                selected_symbol=before.get("definition", {}).get("symbol")
                or before.get("selected_symbol"),
                selection_execution_id=before["selection_execution_id"],
                selection_message_ids=before["selection_message_ids"],
                write_authority="original_TaskSpec_and_public_contract_only",
                reason=type(error).__name__ + ": " + str(error)[:160],
                historical_ref=before.get("ref") or before.get("historical_ref"),
            )
        rows.append(row)
    if not changed:
        return bundle, dict(status="unchanged", source_reads="selected versions only")
    updated, measured = _bounded_payload(reader.namespace, rows, omissions, budget)
    if updated is None:
        rows = [
            dict(
                path=r["path"],
                status="unknown",
                current_at_read=False,
                selected_symbol=r.get("definition", {}).get("symbol") or r.get("selected_symbol"),
                selection_execution_id=r["selection_execution_id"],
                selection_message_ids=r["selection_message_ids"],
                write_authority="original_TaskSpec_and_public_contract_only",
                reason="new complete source bundle does not fit",
                historical_ref=r.get("ref") or r.get("historical_ref"),
            )
            for r in rows
        ]
        updated, measured = _bounded_payload(reader.namespace, rows, [], budget)
        if updated is None:
            raise ValueError("source unknown markers exceed original source budget")
    validate_checkpoint_sources(updated, reader)
    return updated, dict(
        status="refreshed", source_tokens=measured, source_limit=budget["source_tokens"]
    )
