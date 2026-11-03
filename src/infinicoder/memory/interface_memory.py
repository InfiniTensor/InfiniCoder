"""Task-local interface intents linked to versioned declarations and consumers.

An intent is a model claim. Source syntax and resolved indexed references are
observations, not behavioral compatibility, ownership or acceptance proofs.
"""

from infinicoder._resources import package_root

import ast
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json


GUIDANCE = """interface_contract {symbol_id,intent?,check_ids?}: inspect an interface
record,or save/update an intent after reading its current declaration. intent is
a requirement/design claim,not source truth. Existing caller check IDs describe
intended validation,not passed tests. interface_forget {symbol_id} drops a model
note,never a caller requirement. Changed declarations,consumers or requirements
need review;inspect the affected callers before changing an interface. These
notes never expand write scope or replace independent final verification."""


def identity():
    root = package_root()
    return hashlib.sha256(
        b"".join(
            (root / name).read_bytes()
            for name in (
                "memory/interface_memory.py",
                "memory/task_memory.py",
                "analysis/native_effects.py",
                "memory/inspection_memory.py",
                "memory/inspection_ledger.py",
                "analysis/code_context.py",
                "planning/task_contract.py",
            )
        )
    ).hexdigest()


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _requirements(spec):
    return _digest(asdict(spec))


def schemas():
    symbol = dict(type="string", minLength=1, maxLength=500)

    def obj(props, required):
        return dict(type="object", properties=props, required=required, additionalProperties=False)

    return dict(
        interface_contract=obj(
            dict(
                symbol_id=symbol,
                intent=dict(type="string", minLength=1, maxLength=800),
                check_ids=dict(
                    type="array", maxItems=12, uniqueItems=True, items=dict(type="string")
                ),
            ),
            ["symbol_id"],
        ),
        interface_forget=obj(dict(symbol_id=symbol), ["symbol_id"]),
    )


def _symbol_name(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 500 or "::" not in value:
        raise ValueError("symbol_id must be an indexed path::qualified_name")
    return value


def _declaration(index, workspace, symbol_id):
    from infinicoder.memory.task_memory import TaskMemory
    from infinicoder.analysis.symbol_tools import _locate

    symbol = index.symbol(_symbol_name(symbol_id))
    path = symbol["path"]
    # Workspace validates current containment/symlinks; the index is not authority.
    raw = workspace.path(path).read_bytes()
    row = index.db.execute(
        "SELECT hash,source,language,build FROM files WHERE path=?", (path,)
    ).fetchone()
    if row is None or hashlib.sha256(raw).hexdigest() != row["hash"]:
        raise ValueError("Refresh and read changed/deleted source before remembering an interface")
    facts = TaskMemory(index).summary(symbol_id, include_links=False)
    if facts.get("status") != "syntax_observations":
        raise ValueError(
            "Current unambiguous function facts required;native build/header inputs must be valid"
        )
    first, last = symbol["line"], symbol["end"]
    if row["language"] == "python":
        node = _locate(ast.parse(row["source"]), symbol["qualname"])
        if (
            not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            or node.lineno != symbol["line"]
        ):
            raise ValueError("Unambiguous indexed function declaration required")
        first = min([node.lineno, *[d.lineno for d in node.decorator_list]])
        last = max(node.lineno, node.body[0].lineno - 1)
    # Native declarations currently require observed complete symbol ranges;
    # don't guess a header boundary through macros/default lambdas/templates.
    parameters = deepcopy(facts.get("parameters", []))
    declaration = dict(
        symbol=symbol_id,
        path=path,
        source_sha256=row["hash"],
        line=symbol["line"],
        end=symbol["end"],
        signature=symbol["signature"],
        parameters=parameters,
        language=row["language"],
        declaration_read_range=[first, last],
        analysis_identity=facts.get("analysis_identity"),
        parser_inputs_sha256=_digest(row["build"]),
    )
    return declaration


def _consumers(index, workspace, symbol_id, cache):
    digest = hashlib.sha256()
    preview = []
    paths = set()
    stale = []
    count = 0
    rows = index.db.execute(
        """SELECT r.src,r.path,r.line,r.target,r.confidence,f.hash,f.build,f.language
        FROM refs r JOIN files f ON f.path=r.path WHERE r.dst=? AND r.kind='calls'
        ORDER BY r.path,r.line,r.src,r.target,r.confidence""",
        (symbol_id,),
    )
    for row in rows:
        path = row["path"]
        paths.add(path)
        count += 1
        if path not in cache:
            try:
                actual = hashlib.sha256(workspace.path(path).read_bytes()).hexdigest()
            except (OSError, ValueError):
                actual = None
            cache[path] = actual
        if cache[path] != row["hash"]:
            stale.append(path)
        if row["language"] != "python":
            from infinicoder.analysis.native_effects import current_native_inputs

            key = ("native_inputs", path)
            if key not in cache:
                _, cache[key] = current_native_inputs(index, dict(path=path), row, cache)
            if cache[key] is not None:
                stale.append(path)
        record = dict(
            caller=row["src"],
            path=path,
            line=row["line"],
            target=row["target"],
            confidence=row["confidence"],
            source_sha256=row["hash"],
        )
        digest.update(json.dumps([record, _digest(row["build"])], sort_keys=True).encode())
        if len(preview) < 24:
            preview.append(record)
    return dict(
        fingerprint=digest.hexdigest(),
        references=preview,
        total=count,
        files=len(paths),
        omitted=max(0, count - len(preview)),
        stale_paths=sorted(set(stale)),
        limitations="Resolved indexed call references only;dynamic,unresolved and unindexed consumers remain unknown.",
    )


def _read_proof(state, index, workspace, declaration):
    from infinicoder.memory.inspection_memory import current_coverage
    from infinicoder.memory.inspection_ledger import InspectionLedger

    rows = current_coverage(state, workspace, max_chars=200000) + InspectionLedger(index).current(
        [declaration["path"]]
    )
    lo, hi = declaration["declaration_read_range"]
    spans = []
    for row in rows:
        if (
            row["path"] == declaration["path"]
            and row.get("source_sha256") == declaration["source_sha256"]
        ):
            spans.extend(row.get("read_spans", []))
    cursor = lo
    for start, end in sorted(spans):
        if start > cursor:
            break
        cursor = max(cursor, end + 1)
        if cursor > hi:
            return
    raise ValueError("Read the current declaration range before recording an interface intent")


def _view(entry, state, index, workspace, spec, cache):
    result = dict(
        symbol=entry["declaration"]["symbol"],
        intent=dict(text=entry["intent"], origin="model_design_claim_not_fact"),
        intended_check_ids=entry["check_ids"],
        acceptance_proof=False,
        write_scope_unchanged=True,
        baseline_source_sha256=entry["declaration"]["source_sha256"],
        next="Inspect referenced source and verify all relevant consumers;caller final checks remain mandatory.",
    )
    reasons = []
    if entry["requirements_sha256"] != _requirements(spec):
        reasons.append("caller requirements changed")
    if entry["analysis_identity"] != identity():
        reasons.append("analysis implementation changed")
    try:
        declaration = _declaration(index, workspace, result["symbol"])
        consumers = _consumers(index, workspace, result["symbol"], cache)
        if declaration != entry["declaration"]:
            reasons.append("declaration source or syntax changed")
        if consumers["stale_paths"]:
            reasons.append("consumer source differs from index;refresh required")
        if consumers["fingerprint"] != entry["consumers"]["fingerprint"]:
            reasons.append("consumer sources or reference set changed")
    except (KeyError, OSError, ValueError, SyntaxError, UnicodeError) as error:
        reasons.append("source unavailable or analysis incomplete: " + str(error)[:160])
    if reasons:
        result.update(
            status="review_required",
            reasons=reasons,
            references=[
                dict(path=r["path"], line=r["line"], historical=True)
                for r in entry["consumers"]["references"][:6]
            ],
            source_facts_omitted=True,
        )
    else:
        result.update(
            status="current_syntax_observations",
            declaration=declaration,
            consumers=consumers,
            limitation="Unchanged source syntax is not behavioral compatibility or proof that the intent holds.",
        )
    return result


def bounded_view(value, max_chars=4800):
    result = deepcopy(value)
    while len(json.dumps(result, ensure_ascii=False)) > max_chars:
        if result.get("consumers", {}).get("references"):
            result["consumers"]["references"].pop()
            result["consumers"]["omitted"] += 1
        elif result.get("declaration"):
            result.pop("declaration")
            result.pop("consumers", None)
            result["source_facts_omitted"] = True
            result["expand"] = "Use versioned source/function_summary tools for omitted facts"
        elif result.get("references"):
            result["references"].pop()
        else:
            raise ValueError("Interface intent metadata exceeds observation bound")
    return result


def action(name, args, state, index, workspace, spec):
    if not isinstance(args, dict) or "symbol_id" not in args:
        raise ValueError("symbol_id required")
    sid = _symbol_name(args["symbol_id"])
    entries = state.get("interface_memory", {}).get("entries", {})
    if name == "interface_forget":
        if set(args) != {"symbol_id"}:
            raise ValueError("interface_forget accepts only symbol_id")
        entries.pop(sid, None)
        return dict(forgotten=sid, caller_requirements_unchanged=True, acceptance_proof=False)
    if name != "interface_contract" or set(args) - {"symbol_id", "intent", "check_ids"}:
        raise ValueError("Unknown interface memory action/arguments")
    if "intent" not in args:
        if "check_ids" in args:
            raise ValueError("check_ids needs an intent")
        if sid not in entries:
            raise ValueError("No saved interface intent;read source and provide intent")
        return bounded_view(_view(entries[sid], state, index, workspace, spec, {}))
    intent = args["intent"]
    checks = args.get("check_ids", [])
    if not isinstance(intent, str) or not intent.strip() or len(intent) > 800:
        raise ValueError("intent must be1..800 characters")
    if (
        not isinstance(checks, list)
        or len(checks) > 12
        or any(not isinstance(c, str) for c in checks)
        or len(checks) != len(set(checks))
    ):
        raise ValueError("check_ids must be at most12 distinct existing caller check IDs")
    if set(checks) - {check.id for check in spec.checks}:
        raise ValueError("Unknown caller check ID")
    if sid not in entries and len(entries) >= 12:
        raise ValueError("At most12 interface intents;explicitly forget irrelevant notes")
    declaration = _declaration(index, workspace, sid)
    _read_proof(state, index, workspace, declaration)
    consumers = _consumers(index, workspace, sid, {})
    if consumers["stale_paths"]:
        raise ValueError("Refresh changed consumers before remembering an interface")
    entry = dict(
        intent=intent,
        check_ids=list(checks),
        requirements_sha256=_requirements(spec),
        analysis_identity=identity(),
        declaration=declaration,
        consumers=consumers,
    )
    state.setdefault("interface_memory", dict(version=1, entries={}))["entries"][sid] = entry
    return bounded_view(_view(entry, state, index, workspace, spec, {}))


def pack(state, index, workspace, spec, question, max_tokens, estimator, focus_paths=()):
    from infinicoder.analysis.code_context import words

    if type(max_tokens) is not int or max_tokens < 0:
        raise ValueError("max_tokens must be nonnegative int")
    entries = state.get("interface_memory", {}).get("entries", {})
    report = dict(
        registered=len(entries),
        selected=[],
        omitted=len(entries),
        tokens=0,
        budget=max_tokens,
        origin="model_intents_plus_versioned_source_observations",
        acceptance_proof=False,
    )
    if not entries or max_tokens == 0:
        return "", report
    cache = {}
    views = [_view(e, state, index, workspace, spec, cache) for e in entries.values()]
    terms = set(words(question))
    focus = set(focus_paths)
    views.sort(
        key=lambda v: (
            v["status"] != "review_required",
            v["symbol"].split("::", 1)[0] not in focus,
            -len(terms & set(words(v["symbol"] + " " + v["intent"]["text"]))),
            v["symbol"],
        )
    )
    chosen = []

    def render():
        return json.dumps(
            dict(
                interface_memory=chosen,
                omitted=len(entries) - len(chosen),
                policy="Model intents are unverified;source versions and consumers do not prove behavior or acceptance.",
            ),
            ensure_ascii=False,
        )

    for view in views:
        candidate = deepcopy(view)
        while True:
            chosen.append(candidate)
            fits = estimator.estimate(render()) <= max_tokens
            chosen.pop()
            if fits:
                chosen.append(candidate)
                report["selected"].append(view["symbol"])
                break
            if candidate.get("consumers", {}).get("references"):
                candidate["consumers"]["references"].pop()
                candidate["consumers"]["omitted"] += 1
            elif candidate.get("declaration"):
                candidate.pop("declaration")
                candidate.pop("consumers", None)
                candidate["source_facts_omitted"] = True
                candidate["expand"] = "interface_contract {symbol_id} or versioned source tools"
            elif candidate.get("references"):
                candidate["references"].pop()
            else:
                break
    text = render() if chosen else ""
    report.update(
        omitted=len(entries) - len(chosen), tokens=estimator.estimate(text) if text else 0
    )
    return text, report
