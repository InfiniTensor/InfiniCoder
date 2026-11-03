"""Question-selected source facts,independent of a writable body or proposal.

The caller supplies its existing namespace reader and actual tokenizer. This
module neither executes repository code nor interprets reading as edit scope.
"""

import ast
from dataclasses import asdict
import math
from pathlib import PurePosixPath
import re
import time
import uuid

from infinicoder.memory.source.source_question import NeedContextRequest
from infinicoder.analysis.source_question_syntax import (
    FUNCTIONS,
    _walk,
    _parents,
    _fields,
    _piece,
    _receiver,
)
from infinicoder.analysis.symbol_tools import _locate
from infinicoder.planning.task_source_plan import dump, WorkspaceSourcePlanReader


TARGET = re.compile(
    r"(?<![\w./])([A-Za-z0-9_][A-Za-z0-9_./-]*\.py)::([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)"
)
FILE = re.compile(r"(?<![\w./])([A-Za-z0-9_][A-Za-z0-9_./-]*\.py)(?![\w./])")
AUTHORITY = (
    "Exact source syntax and static import alternatives only;runtime "
    "dispatch,values,ownership,concurrency and correctness remain unknown. "
    "Reading grants no editing permission."
)


class SourceAnswerCapacityError(ValueError):
    """A real next-input count exceeds capacity;never a synthetic token count."""

    def __init__(self, standalone_tokens, next_input_tokens, input_limit):
        values = (standalone_tokens, next_input_tokens, input_limit)
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("actual capacity counts required")
        self.standalone_tokens, self.next_input_tokens, self.input_limit = values
        super().__init__("complete source answer exceeds current SDK input capacity")


def answer_source_question(question, anchor, read_session, counter, deadline, *, max_tokens=2048):
    """Return one whole-record view;no planner,model call or repository import.

    ``counter(messages,remaining_seconds)`` must use the actual chat tokenizer.
    ``read_session`` is the caller's WorkspaceSourcePlanReader (or its verified
    namespace subclass). Its regular-file capability is the only read entry.
    A new view reads/indexes at most three files;it never refreshes the whole repo.
    """
    if not isinstance(read_session, WorkspaceSourcePlanReader) or not callable(counter):
        raise ValueError("caller-owned namespace reader and actual tokenizer required")
    if not isinstance(question, str) or not 1 <= len(question.encode()) <= 8192:
        raise ValueError("one bounded literal source question required")
    if anchor is not None and (not isinstance(anchor, str) or TARGET.fullmatch(anchor) is None):
        raise ValueError(
            "anchor must be an explicit relative Python path::symbol;not an edit scope"
        )
    if type(max_tokens) is not int or not 128 <= max_tokens <= 2048:
        raise ValueError("source answer limit must be128..2048 actual tokens")
    if (
        isinstance(deadline, bool)
        or not isinstance(deadline, (int, float))
        or not math.isfinite(deadline)
    ):
        raise ValueError("finite original action deadline required")

    def remaining():
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("source question shares the original action deadline")
        return left

    remaining()
    reader = read_session
    requests = []
    unresolved = []
    files = {}
    records = []
    attempted = set()
    for match in TARGET.finditer(question):
        item = dict(kind="definition", path=match[1], symbol=match[2])
        if item not in requests:
            requests.append(item)
    fields = tuple(dict.fromkeys(re.findall(r"\bself\.([A-Za-z_]\w*)\b", question)))
    if fields:
        if anchor and "." in anchor.split("::", 1)[1]:
            path, name = anchor.split("::", 1)
            requests.insert(
                0, dict(kind="fields", path=path, symbol=name.rsplit(".", 1)[0], fields=fields)
            )
        else:
            unresolved.append(dict(fields=fields, status="explicit_owner_anchor_required"))
    if not requests and anchor:
        path, name = anchor.split("::", 1)
        requests.append(dict(kind="definition", path=path, symbol=name))
    for match in FILE.finditer(question):
        if not any(row["path"] == match[1] for row in requests):
            requests.append(dict(kind="file", path=match[1]))
    if len(requests) > 3:
        unresolved.append(dict(status="named_request_bound", omitted=len(requests) - 3))
        requests = requests[:3]
    need = NeedContextRequest(question, tuple(requests), ())
    base = dict(
        identity="source_question_action_v1",
        read_event_ref="source-view:" + uuid.uuid4().hex,
        namespace=reader.namespace,
        request=asdict(need),
        authority=AUTHORITY,
    )

    def view(status="answered", items=None, missing=None):
        return dict(
            base,
            status=status,
            records=records if items is None else items,
            unresolved=unresolved if missing is None else missing,
            source_files={path: value[2] for path, value in files.items()},
        )

    def count(value):
        result = counter([dict(role="user", content=dump(value))], remaining())
        if type(result) is not int or result < 0:
            raise ValueError("actual nonnegative tokenizer count required")
        return result

    def offer(record):
        try:
            tokens = count(view(items=records + [record]))
        except SourceAnswerCapacityError:
            unresolved.append(
                dict(
                    path=record["path"],
                    symbol=record.get("symbol"),
                    status="whole_record_does_not_fit_current_input",
                )
            )
            return False
        if tokens > max_tokens:
            unresolved.append(
                dict(
                    path=record["path"],
                    symbol=record.get("symbol"),
                    status="whole_record_does_not_fit",
                )
            )
            return False
        records.append(record)
        return True

    def read(path):
        remaining()
        if path in files:
            return files[path]
        if path in attempted:
            return None
        if len(attempted) >= 3:
            unresolved.append(dict(path=path, status="source_file_bound"))
            return None
        attempted.add(path)
        try:
            value = reader.session.read(path, max_bytes=131072)
            # Reuse the existing index,refreshing only bytes actually requested.
            reader.session.refresh_paths((path,))
        except (OSError, ValueError, UnicodeError, RecursionError) as error:
            unresolved.append(dict(path=path, status="unavailable", reason=type(error).__name__))
            return None
        files[path] = value
        return value

    def location(path, node, kind, **extra):
        _, text, digest = files[path]
        start = node.lineno
        end = node.end_lineno
        if isinstance(node, (*FUNCTIONS, ast.ClassDef)):
            start = min([start, *[item.lineno for item in node.decorator_list]])
        return dict(
            kind=kind,
            path=path,
            source_sha256=digest,
            start=start,
            end=end,
            complete=True,
            source="".join(text.splitlines(keepends=True)[start - 1 : end]),
            **extra,
        )

    def module_candidates(path, node):
        parent = list(PurePosixPath(path).parent.parts)
        if node.level:
            if node.level > len(parent):
                return []
            parts = parent[: len(parent) - node.level + 1]
        else:
            parts = []
        parts += node.module.split(".") if node.module else []
        if not parts:
            return []
        module = "/".join(parts)
        return [module + ".py", module + "/__init__.py"]

    def definition(path, name, seen=()):
        key = (path, name)
        if key in seen:
            unresolved.append(dict(path=path, symbol=name, status="static_import_cycle"))
            return
        value = read(path)
        if value is None:
            return
        try:
            tree = ast.parse(value[1])
        except (SyntaxError, RecursionError):
            unresolved.append(dict(path=path, symbol=name, status="unparsed"))
            return
        try:
            node = _locate(tree, name)
        except ValueError:
            node = None
        head = name.split(".")[0]
        imports = [
            (item, alias)
            for item in tree.body
            if isinstance(item, ast.ImportFrom)
            for alias in item.names
            if (alias.asname or alias.name) == head
        ]
        # A literal import chain is a static reading alternative. Rebindings,
        # wildcard imports and conditional exports cannot establish a target.
        rebound = any(
            isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store) and item.id == head
            for statement in tree.body
            if not isinstance(statement, (*FUNCTIONS, ast.ClassDef, ast.ImportFrom))
            for item in ast.walk(statement)
        )
        rebound = rebound or any(
            isinstance(statement, ast.Import)
            and any((a.asname or a.name.split(".")[0]) == head for a in statement.names)
            for statement in tree.body
        )
        rebound = rebound or any(
            isinstance(item, (*FUNCTIONS, ast.ClassDef)) and item.name == head
            for statement in tree.body
            if not isinstance(statement, (*FUNCTIONS, ast.ClassDef, ast.ImportFrom))
            for item in ast.walk(statement)
        )
        wildcard = any(
            isinstance(item, ast.ImportFrom) and any(a.name == "*" for a in item.names)
            for item in tree.body
        )
        if rebound or wildcard or node is not None and imports or len(imports) > 1:
            unresolved.append(dict(path=path, symbol=name, status="ambiguous_or_rebound_export"))
            return
        if node is not None:
            offer(location(path, node, "definition", symbol=path + "::" + name))
            return
        if len(imports) != 1:
            unresolved.append(
                dict(path=path, symbol=name, status="definition_or_static_export_unknown")
            )
            return
        statement, alias = imports[0]
        candidates = []
        # Inspect only namespace paths;do not import code or pick a basename.
        for candidate in module_candidates(path, statement):
            try:
                file = reader.workbench.workspace.path(candidate)
                if file.is_file() and not file.is_symlink():
                    candidates.append(candidate)
            except (OSError, ValueError):
                continue
        target = alias.name + ("." + name.split(".", 1)[1] if "." in name else "")
        offer(
            location(
                path,
                statement,
                "static_export",
                symbol=path + "::" + head,
                candidate_targets=[p + "::" + target for p in candidates],
                certainty="literal_import_syntax;runtime_import_resolution_unknown",
            )
        )
        if len(candidates) == 1:
            definition(candidates[0], target, (*seen, key))
        else:
            unresolved.append(
                dict(
                    path=path,
                    symbol=name,
                    status="module_path_ambiguous_or_missing",
                    candidates=candidates,
                )
            )

    for request in requests:
        remaining()
        path = request["path"]
        if request["kind"] == "definition":
            definition(path, request["symbol"])
            continue
        value = read(path)
        if value is None:
            continue
        _, text, digest = value
        if request["kind"] == "file":
            if not text:
                unresolved.append(dict(path=path, status="empty_file"))
                continue
            offer(
                dict(
                    kind="whole_file",
                    path=path,
                    source_sha256=digest,
                    start=1,
                    end=len(text.splitlines()),
                    complete=True,
                    source=text,
                )
            )
            continue
        try:
            tree = ast.parse(text)
            node = _locate(tree, request["symbol"])
        except (ValueError, SyntaxError, RecursionError):
            unresolved.append(dict(path=path, symbol=request["symbol"], status="owner_unknown"))
            continue
        if not isinstance(node, ast.ClassDef):
            unresolved.append(dict(path=path, status="owner_is_not_class"))
            continue
        constructors = [n for n in node.body if isinstance(n, FUNCTIONS) and n.name == "__init__"]
        if node.bases:
            unresolved.append(dict(path=path, status="inherited_fields_and_overrides_unknown"))
        if len(constructors) != 1:
            unresolved.append(dict(path=path, status="constructor_missing_or_ambiguous"))
            continue
        constructor = constructors[0]
        parents = _parents(tree)
        receiver = _receiver(constructor)
        found = set()
        for statement in _walk(constructor):
            if not isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                continue
            overlap = _fields(statement, receiver, ast.Store) & set(request["fields"])
            if overlap:
                fact = dict(
                    _piece(
                        dict(path=path, hash=digest, source=text), statement, parents, constructor
                    ),
                    kind="field_origin",
                    symbol=path + "::" + request["symbol"] + ".__init__",
                    fields=sorted(overlap),
                    complete=True,
                    certainty="constructor_receiver_syntax;later_writes_and_runtime_values_unknown",
                )
                if offer(fact):
                    found.update(overlap)
        for field in set(request["fields"]) - found:
            unresolved.append(
                dict(path=path, identifier="self." + field, status="field_origin_not_provided")
            )
    # Verify only queried files,not a speculative whole-repository snapshot.
    for path, value in files.items():
        remaining()
        if reader.current_version(path) != value[2]:
            return view(
                "stale_during_question",
                items=[],
                missing=[dict(path=path, status="source_changed;ask_again_if_needed")],
            )
    result = view("partial" if unresolved else "answered")
    if not requests:
        result = view("unknown;name_a_path_symbol_or_owner")
    if not records and requests:
        limited = any(
            row.get("status")
            in ("whole_record_does_not_fit", "whole_record_does_not_fit_current_input")
            for row in unresolved
        )
        result = view("budget_limited" if limited else "unknown;no_complete_source_facts")
    try:
        exceeds = count(result) > max_tokens
    except SourceAnswerCapacityError:
        exceeds = True
    if exceeds:
        # Retain the question and explicit failure;never cut a source record.
        result = view(
            "budget_limited", items=[], missing=[dict(status="whole_answer_does_not_fit")]
        )
        try:
            exceeds = count(result) > max_tokens
        except SourceAnswerCapacityError:
            exceeds = True
        if exceeds:
            result = dict(
                identity=base["identity"],
                read_event_ref=base["read_event_ref"],
                status="budget_limited",
                records=[],
                source_files={},
                unresolved=[dict(status="question_metadata_exceeds_answer_budget")],
            )
            if count(result) > max_tokens:
                raise ValueError("minimal failure metadata exceeds the actual answer budget")
    return result
