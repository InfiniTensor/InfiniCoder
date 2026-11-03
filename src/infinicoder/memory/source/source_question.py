"""One explicit source question answered with bounded,versioned syntax facts.

The existing index locates named definitions;the current read capability supplies
facts. No project code executes,and no inferred algorithm or type is an answer.
"""

import ast
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import re
import time

from infinicoder.analysis.code_context import CodeIndex
from infinicoder.analysis.symbol_tools import _locate
from infinicoder.analysis.source_question_syntax import (
    FUNCTIONS,
    _walk,
    _parents,
    _fields,
    _piece,
    _receiver,
)


@dataclass(frozen=True)
class NeedContextRequest:
    question: str
    requests: tuple
    unresolved: tuple


@dataclass(frozen=True)
class SourceAnswer:
    request: NeedContextRequest
    records: tuple
    unresolved: tuple
    source_versions: dict
    source_namespace: str
    added_tokens: int
    input_tokens: int
    status: str


def _named(text, name):
    return re.search(r"(?<![\w.])" + re.escape(name) + r"(?![\w.])", text)


def answer_context_request(owner, preparation, proposal, request, deadline):
    """Return (continuation preparation or None,audit). Never requests a model.

    Limits apply to exposed source files/requests;index freshness validation uses
    the original admitted read set. All facts are whole statements/definitions.
    """
    from infinicoder.runtime.implementation_proposal import delivery, dump
    from infinicoder.runtime.proposal_delivery import strict_json

    def remaining():
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("Original deadline exhausted during source answer")
        return left

    def no(status, **fields):
        return None, dict(status=status, **fields)

    remaining()
    if proposal.get("finish_reason") != "stop" or proposal.get("tool_calls_present"):
        return no("not_a_complete_source_question")
    try:
        value, _ = delivery(strict_json(proposal["text"]))
    except (ValueError, KeyError, TypeError):
        return no("not_a_complete_source_question")
    if (
        set(value) != {"need_context"}
        or not isinstance(value["need_context"], str)
        or not value["need_context"].strip()
    ):
        return no("not_a_complete_source_question")
    question = value["need_context"]
    target_path, qualified = owner.body_scope.split("::")
    owner_name = qualified.rsplit(".", 1)[0] if "." in qualified else None
    named = []
    unknown = []
    fields = tuple(dict.fromkeys(re.findall(r"\bself\.([A-Za-z_]\w*)\b", question)))
    if fields:
        if owner_name:
            named.append(dict(kind="fields", path=target_path, symbol=owner_name, fields=fields))
        else:
            unknown.append(dict(fields=fields, status="no_direct_target_class"))
    # Resolve only explicitly written identifiers against the admitted index.
    # An unqualified duplicate stays ambiguous;never pick the target file first.
    with CodeIndex(
        request.workspace,
        Path(request.state_dir) / "implementation-proposal.sqlite",
        remaining=remaining,
    ) as index:
        candidates = {}
        for row in index.db.execute("SELECT id,path,qualname FROM symbols ORDER BY id").fetchall():
            remaining()
            exact = (
                row["id"]
                if _named(question, row["id"])
                else row["qualname"]
                if _named(question, row["qualname"])
                else None
            )
            if exact and row["id"] != owner.body_scope:
                candidates.setdefault(exact, []).append(dict(row))
        for term, rows in sorted(candidates.items(), key=lambda item: question.find(item[0])):
            if len(rows) != 1:
                unknown.append(dict(identifier=term, status="ambiguous_definition"))
                continue
            row = rows[0]
            if not any(
                n.get("symbol") == row["qualname"]
                and n["path"] == row["path"]
                and n["kind"] == "definition"
                for n in named
            ):
                named.append(dict(kind="definition", path=row["path"], symbol=row["qualname"]))
        for path in owner.sources.paths:
            if _named(question, path):
                named.append(dict(kind="file", path=path))
    if len(named) > 3:
        unknown.extend(dict(request=n, status="named_request_bound") for n in named[3:])
    named = named[:3]
    need = NeedContextRequest(question, tuple(named), tuple(unknown))
    if not named:
        return no("no_resolvable_named_source", request=asdict(need))
    if not owner.current(preparation.plan):
        return no("stale_initial_source", request=asdict(need))
    records = []
    versions = dict(preparation.plan["source_versions"])
    files = {}
    unresolved = list(unknown)
    base_messages = preparation.messages + [dict(role="assistant", content=proposal["text"])]

    def payload(items, missing):
        return dict(
            origin="controller source answer;read-only syntax facts",
            source_namespace=owner.source_namespace,
            original_question=question,
            records=items,
            unresolved=missing,
            interpretation="Exact source syntax only;runtime types,dispatch,values,ownership,concurrency and correctness unknown;reading grants no editing",
            next="Return one complete body implementation using the original protected interface. If still insufficient return need_context;no further automatic source answer will follow.",
        )

    def messages(items, missing):
        return base_messages + [dict(role="user", content=dump(payload(items, missing)))]

    def offer(record):
        candidate = records + [record]
        added = owner.counter(
            [dict(role="user", content=dump(payload(candidate, unresolved)))], remaining()
        )
        total = owner.counter(messages(candidate, unresolved), remaining())
        if added > 2048 or total > request.model_profile.input_budget():
            unresolved.append(
                dict(
                    path=record["path"],
                    symbol=record.get("symbol"),
                    start=record["start"],
                    end=record["end"],
                    status="whole_record_does_not_fit",
                )
            )
            return False
        records.append(record)
        return True

    for item in named:
        remaining()
        path = item["path"]
        if path not in files:
            if len(files) >= 3:
                unresolved.append(dict(request=item, status="source_file_bound"))
                continue
            raw, text, digest = owner.sources.read(path, max_bytes=131072)
            if path in versions and versions[path] != digest:
                return no("stale_answer_source", request=asdict(need))
            versions[path] = digest
            files[path] = (text, digest)
        text, digest = files[path]
        lines = text.splitlines(keepends=True)

        def location(start, end, **extra):
            return dict(
                path=path,
                source_sha256=digest,
                start=start,
                end=end,
                complete=True,
                source="".join(lines[start - 1 : end]),
                **extra,
            )

        if item["kind"] == "file":
            offer(location(1, len(lines), kind="whole_file"))
            continue
        try:
            tree = ast.parse(text)
            node = _locate(tree, item["symbol"])
        except (ValueError, SyntaxError, RecursionError):
            unresolved.append(dict(request=item, status="current_definition_ambiguous_or_unparsed"))
            continue
        if item["kind"] == "definition":
            first = min([node.lineno, *[d.lineno for d in node.decorator_list]])
            offer(
                location(
                    first, node.end_lineno, kind="definition", symbol=path + "::" + item["symbol"]
                )
            )
            continue
        if not isinstance(node, ast.ClassDef):
            unresolved.append(dict(request=item, status="target_owner_is_not_class"))
            continue
        if node.bases:
            unresolved.append(
                dict(symbol=item["symbol"], status="inherited_fields_and_overrides_unknown")
            )
        constructors = [n for n in node.body if isinstance(n, FUNCTIONS) and n.name == "__init__"]
        if len(constructors) != 1:
            unresolved.append(dict(request=item, status="constructor_missing_or_ambiguous"))
            continue
        constructor = constructors[0]
        receiver = _receiver(constructor)
        parents = _parents(tree)
        found = set()
        file = dict(path=path, hash=digest, source=text)
        for statement in _walk(constructor):
            remaining()
            if not isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                continue
            overlap = _fields(statement, receiver, ast.Store) & set(item["fields"])
            if not overlap:
                continue
            piece = _piece(file, statement, parents, constructor)
            fact = dict(
                piece,
                kind="field_origin",
                symbol=path + "::" + item["symbol"] + ".__init__",
                fields=sorted(overlap),
                complete=True,
                certainty="constructor_receiver_syntax;later_writes_and_runtime_values_unknown",
            )
            if offer(fact):
                found.update(overlap)
        for field in item["fields"]:
            if field not in found:
                unresolved.append(
                    dict(identifier="self." + field, status="no_complete_field_origin_provided")
                )
    if not records:
        return no("no_new_source_facts_fit", request=asdict(need), unresolved=unresolved)
    # Never re-label source already present in the original target/dependencies
    # as new evidence sufficient to justify another physical model request.
    old = [preparation.plan["target"], *preparation.plan["direct_dependencies"]]
    fresh = [
        r
        for r in records
        if not any(
            r["path"] == o["path"] and r["start"] >= o["start"] and r["end"] <= o["end"]
            for o in old
        )
    ]
    if not fresh:
        return no("no_new_source_facts", request=asdict(need), unresolved=unresolved)
    answer_messages = messages(records, unresolved)
    added = owner.counter(
        [dict(role="user", content=dump(payload(records, unresolved)))], remaining()
    )
    count = owner.counter(answer_messages, remaining())
    answer = SourceAnswer(
        need,
        tuple(records),
        tuple(unresolved),
        versions,
        owner.source_namespace,
        added,
        count,
        "answered",
    )
    if added > 2048 or count > request.model_profile.input_budget():
        return no("complete_answer_metadata_does_not_fit", answer=asdict(answer))
    plan = dict(preparation.plan, source_versions=versions)
    if not owner.current(plan):
        return no("stale_before_continuation", answer=asdict(answer))
    remaining()
    return replace(
        preparation,
        plan=plan,
        messages=answer_messages,
        input_tokens=count,
        reason="One bounded answer to explicit source question",
    ), asdict(answer)
