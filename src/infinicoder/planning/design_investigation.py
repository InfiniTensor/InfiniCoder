"""Bounded source/probe investigation before recording a domain hypothesis.

Source facts, finite-model observations and model plans retain different types.
Nothing here executes candidate code, edits files, or satisfies acceptance.
"""

from infinicoder._resources import package_root

import hashlib
import json
from infinicoder.planning.planning import domain_contract as contract

MAX_REQUESTS = 6
MAX_INSPECTIONS = 4
MAX_INVALID_REQUESTS = 2
READ_TOOLS = ("read", "search", "use_def")


def identity():
    root = package_root()
    return hashlib.sha256(
        b"".join(
            (root / name).read_bytes()
            for name in (
                "planning/design_investigation.py",
                "planning/domain_contract.py",
                "planning/design_probes.py",
            )
        )
    ).hexdigest()


def pending(spec, unit, domain, state):
    mode = contract.pending(spec, unit, domain, state)
    if mode is None:
        return None
    record = state.get("domain_design_records", {}).get(mode["key"], {})
    if record.get("status") == "design_unavailable":
        return None
    if record.get("requests", 0) >= MAX_REQUESTS:
        record["status"] = "design_unavailable"
        return None
    mode.update(investigation=True, attempt=record.get("requests", 0) + 1)
    return mode


def begin(state, mode):
    records = state.setdefault("domain_design_records", {})
    while len(records) >= 32 and mode["key"] not in records:
        records.pop(next(iter(records)))
    record = records.setdefault(mode["key"], dict(attempts=0, plan=None))
    if record.get("requests", 0) >= MAX_REQUESTS or record["plan"] is not None:
        raise ValueError("design investigation window already spent")
    record["requests"] = record.get("requests", 0) + 1
    record.setdefault("investigation_observations", [])


def tools(domain, record):
    from infinicoder.planning.design_probes import TOOLS

    return (
        (*READ_TOOLS, *TOOLS.get(domain, ()))
        if len(record.get("investigation_observations", [])) < MAX_INSPECTIONS
        else ()
    )


def schema(domain, ids, state, mode):
    from infinicoder.runtime.protocol import action_schema

    record = state.get("domain_design_records", {}).get(mode["key"], {})
    names = tools(domain, record)
    plan = contract.schema(domain, ids)
    return {"anyOf": [plan, action_schema(names, design_probes=True)]} if names else plan


def decode(value, state, mode):
    if isinstance(value, dict) and set(value) == {"tool", "arguments"}:
        record = state["domain_design_records"][mode["key"]]
        if value["tool"] not in tools(mode["domain"], record) or not isinstance(
            value["arguments"], dict
        ):
            raise ValueError(
                "pre-code investigation permits only the advertised read-only source/probe tools;return the design when the window is spent"
            )
        return value
    return dict(tool="domain_design", arguments=value)


def _fresh(bindings, workspace):
    result = []
    for binding in bindings:
        try:
            file = workspace.path(binding["path"])
            current = (
                file.is_file()
                and hashlib.sha256(file.read_bytes()).hexdigest() == binding["sha256"]
            )
        except (OSError, ValueError):
            current = False
        result.append(dict(binding, source_version_current=current))
    return result


def _preview(result, tool):
    """Whole small results or explicit partial displays, never clipped JSON."""
    text = json.dumps(result, ensure_ascii=False)
    if len(text) <= 1600:
        return result
    if tool == "read":
        rows = result["text"].split("\n")
        shown = []
        for row in rows:
            if sum(len(v) + 1 for v in shown) + len(row) > 1000:
                break
            shown.append(row)
        partial = not shown and bool(rows)
        return dict(
            path=result["path"],
            sha256=result["sha256"],
            start=result["start"],
            total_lines=result["total_lines"],
            text="\n".join(shown) if shown else rows[0][:1000],
            fully_visible_line_count=len(shown),
            partial_first_line=partial,
            display_omitted=True,
            interpretation="Only displayed complete lines/prefix were supplied;request a narrower range for omitted source.",
        )
    # A truncated relation is not a new fact. Keep the original result in the
    # event log;request a narrower query rather than invent a shortened object.
    return dict(
        display_omitted=True,
        result_characters=len(text),
        interpretation="Structured result exceeds display budget;repeat search with a narrower query,or use_def with a name and smaller limit.",
    )


def view(state, mode, workspace):
    record = state.get("domain_design_records", {}).get(mode["key"], {})
    observations = []
    for item in record.get("investigation_observations", []):
        bindings = _fresh(item["source_bindings"], workspace)
        current = all(v["source_version_current"] for v in bindings)
        if item.get("repository_revision") is not None:
            current &= item["repository_revision"] == workspace.revision()
        observations.append(
            dict(
                id=item["id"],
                kind=item["kind"],
                tool=item["tool"],
                source_bindings=bindings,
                result_sha256=item["result_sha256"],
                result=item["display"] if current else None,
                bounded_probe_outcome=item.get("bounded_probe_outcome") if current else None,
                stale=not current,
                interpretation="Source/probe observation only,not implementation correctness. Stale observations are references only.",
            )
        )
    return dict(
        requests_spent=record.get("requests", 0),
        request_limit=MAX_REQUESTS,
        inspection_limit=MAX_INSPECTIONS,
        available_tools=list(tools(mode["domain"], record)),
        invalid_requests=record.get("invalid_requests", 0),
        invalid_request_limit=MAX_INVALID_REQUESTS,
        observations=observations,
        last_error=record.get("error"),
        all_budgets_shared=True,
        acceptance_proof=False,
        limitations="Index facts may be incomplete. Layout checks do not validate numerical semantics;finite lifecycle models do not prove actual concurrency. A recorded plan remains a model hypothesis.",
    )


def context(spec, unit, state, workspace, index, path, assembler, mode):
    from infinicoder.models.budget import BudgetExceeded

    observed = view(state, mode, workspace)
    record = state.get("domain_design_records", {}).get(mode["key"], {})
    current_ids = {v["id"] for v in observed["observations"] if not v["stale"]}
    current = [v for v in record.get("investigation_observations", []) if v["id"] in current_ids]
    mode["investigation_leases"] = [
        binding for item in current for binding in item["source_bindings"]
    ]
    mode["repository_leases"] = [
        item["repository_revision"]
        for item in current
        if item.get("repository_revision") is not None
    ]
    while True:
        try:
            return contract.context(
                spec, unit, state, workspace, index, path, assembler, mode, investigation=observed
            )
        except BudgetExceeded:
            # Keep outcome labels and source/version references. Older complete
            # bodies remain in events;never replace them with invented facts.
            old = next((v for v in observed["observations"] if v.get("result") is not None), None)
            if old is None:
                raise
            old["result"] = None
            old["display_omitted_for_context_budget"] = True


def inspect(tool, args, spec, state, mode, workspace, index):
    from infinicoder.planning.design_probes import TOOLS, run as run_probe

    record = state["domain_design_records"][mode["key"]]
    if tool not in tools(mode["domain"], record):
        raise ValueError("design inspection window or tool unavailable")
    if tool in ("read", "use_def"):
        path = args.get("path")
        file = workspace.path(path)
        if str(file.resolve()) in state.get("checker_files", {}):
            raise ValueError("protected acceptance source is not a design target")
    if tool == "read":
        if set(args) - {"path", "start", "count"} or not isinstance(args.get("path"), str):
            raise ValueError("read needs path,start?,count?")
        if any(
            type(args.get(name, default)) is not int
            for name, default in (("start", 1), ("count", 80))
        ):
            raise ValueError("integer read range required")
        if not 1 <= args.get("count", 80) <= 120:
            raise ValueError("design read count must be1..120")
        result = workspace.read(
            args["path"], start=args.get("start", 1), count=args.get("count", 80)
        )
        bindings = [dict(path=result["path"], sha256=result["sha256"])]
        kind = "source_excerpt"
    elif tool == "search":
        if (
            set(args) - {"query", "limit"}
            or not isinstance(args.get("query"), str)
            or not 1 <= len(args["query"]) <= 512
        ):
            raise ValueError("bounded search query required")
        if type(args.get("limit", 5)) is not int or not 1 <= args.get("limit", 5) <= 8:
            raise ValueError("design search limit must be1..8")
        repository_before = workspace.revision()
        index.refresh()
        result = index.search(args["query"], limit=args.get("limit", 5))
        bindings = []
        kind = "static_index"
        for item in result:
            path = item["path"]
            file = workspace.path(path)
            if str(file.resolve()) in state.get("checker_files", {}):
                raise ValueError("protected acceptance source excluded")
            row = index.db.execute("SELECT hash FROM files WHERE path=?", (path,)).fetchone()
            if row is None:
                raise ValueError("search returned an unbound source")
            bindings.append(dict(path=path, sha256=row[0]))
        if workspace.revision() != repository_before:
            raise ValueError("repository changed during search;repeat against current sources")
    elif tool == "use_def":
        if set(args) - {"path", "name", "limit"} or not isinstance(args.get("path"), str):
            raise ValueError("use_def needs path,name?,limit?")
        if type(args.get("limit", 12)) is not int or not 1 <= args.get("limit", 12) <= 24:
            raise ValueError("design use_def limit must be1..24")
        result = index.use_def(args["path"], name=args.get("name"), limit=args.get("limit", 12))
        bindings = [dict(path=result["path"], sha256=result["source_sha256"])]
        kind = "static_dataflow"
    elif tool in TOOLS.get(mode["domain"], ()):
        visible = [dict(path=v["path"], sha256=v["sha256"]) for v in mode["sources"].values()]
        for item in record.get("investigation_observations", []):
            if item["kind"] == "source_excerpt":
                visible.extend(item["source_bindings"])
        result = run_probe(tool, args, spec, state, workspace, mode["domain"], visible)
        bindings = [dict(path=args["path"], sha256=args["expected_hash"])]
        kind = "bounded_design_probe"
    else:
        raise ValueError("unsupported design tool")
    bindings = list({(v["path"], v["sha256"]): v for v in bindings}.values())
    if not all(v["source_version_current"] for v in _fresh(bindings, workspace)):
        raise ValueError("source changed during inspection;inspect the current version")
    raw = json.dumps(result, ensure_ascii=False)
    item = dict(
        id="investigation_" + str(len(record["investigation_observations"])),
        tool=tool,
        kind=kind,
        source_bindings=bindings,
        result_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        display=_preview(result, tool),
    )
    if tool == "search":
        item["repository_revision"] = repository_before
    if kind == "bounded_design_probe":
        item["bounded_probe_outcome"] = result["symbolic_observation"]["status"]
    record["investigation_observations"].append(item)
    return item, result


def accept(value, state, mode, workspace):
    record = state["domain_design_records"][mode["key"]]
    record["attempts"] += 1
    if not all(
        v["source_version_current"] for v in _fresh(mode.get("investigation_leases", []), workspace)
    ) or any(v != workspace.revision() for v in mode.get("repository_leases", [])):
        raise ValueError(
            "investigation source changed during request;inspect the current version before recording the design"
        )
    result = contract.accept(value, state, mode, workspace)
    result["investigation"] = view(state, mode, workspace)
    return result


def reject(state, mode, error):
    record = state["domain_design_records"][mode["key"]]
    record["invalid_requests"] = record.get("invalid_requests", 0) + 1
    record.update(
        error=str(error)[:500],
        status="design_unavailable"
        if record["requests"] >= MAX_REQUESTS
        or record["attempts"] >= 2
        or record["invalid_requests"] >= MAX_INVALID_REQUESTS
        else "design_investigation_pending",
    )
