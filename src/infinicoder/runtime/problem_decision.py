"""Bounded source-supported next-action decisions,inside the original budget.

Nodes certify current observations and action scope,never a model hypothesis.
The original edit executor and independent acceptance remain authoritative.
"""

import ast
from copy import deepcopy
from pathlib import Path
from infinicoder.runtime.problem_workflow import contract, digest, event


def schema():
    from infinicoder.runtime.problem_workflow import schema as probe_schema

    text = dict(type="string", minLength=1, maxLength=500)
    return dict(
        type="object",
        properties=dict(
            mode=dict(type="string", enum=["edit", "read", "probe"]),
            target_id=dict(type="string", pattern="^T[0-9]+$"),
            hypothesis=text,
            dependency_ids=dict(
                type="array",
                maxItems=4,
                uniqueItems=True,
                items=dict(type="string", pattern="^S[0-9]+$"),
            ),
            read_ref=dict(type="string", pattern="^[SD][0-9]+$"),
            start=dict(type="integer", minimum=1),
            count=dict(type="integer", minimum=1, maximum=100),
            hypotheses=probe_schema()["properties"]["hypotheses"],
        ),
        required=["mode", "target_id", "hypothesis", "dependency_ids"],
        additionalProperties=False,
    )


def control(state):
    return state.setdefault(
        "problem_decision", dict(status="observing", repeat_rounds=0, decisions=0, nodes=[])
    )


def available(state, ordinary):
    value = state.get("problem_decision", {})
    status = value.get("status")
    if status == "due":
        return list(dict.fromkeys([*ordinary, "problem_decide"]))
    if status == "active":
        return [
            dict(edit="edit", read="read", probe="problem_probe")[value["node"]["mode"]],
            "ask_user",
        ]
    return [name for name in ordinary if name != "problem_decide"]


def _covered(spans, start, end):
    return any(a <= start and b >= end for a, b in spans)


def offer(state, spec, workspace, index):
    """Only actually observed,versioned function bodies become edit targets."""
    from infinicoder.memory.inspection_ledger import InspectionLedger

    records = InspectionLedger(index).current()
    sources = []
    targets = []
    reads = []
    preferred = state.get("read_paths", [])
    records.sort(
        key=lambda r: preferred.index(r["path"]) if r["path"] in preferred else -1, reverse=True
    )
    for record in records[:8]:
        source = dict(
            id="S" + str(len(sources)),
            path=record["path"],
            sha256=record["source_sha256"],
            spans=record["read_spans"],
        )
        sources.append(source)
        lines = workspace.path(source["path"]).read_text(encoding="utf-8-sig").splitlines()
        # Import destinations are source-derived read suggestions,not edit leases.
        try:
            tree = ast.parse("\n".join(lines))
            for item in ast.walk(tree):
                if not isinstance(item, (ast.Import, ast.ImportFrom)) or not _covered(
                    source["spans"], item.lineno, item.end_lineno
                ):
                    continue
                modules = (
                    [alias.name for alias in item.names]
                    if isinstance(item, ast.Import)
                    else [item.module or ""]
                )
                for module in modules:
                    base = Path(source["path"]).parent
                    if isinstance(item, ast.ImportFrom) and item.level:
                        for _ in range(item.level - 1):
                            base = base.parent
                    else:
                        base = Path(".")
                    path = (
                        (base / Path(module.replace(".", "/"))).with_suffix(".py").as_posix()
                        if module
                        else ""
                    )
                    if (
                        path
                        and workspace.path(path).is_file()
                        and path not in {r["path"] for r in reads}
                    ):
                        reads.append(
                            dict(
                                id="D" + str(len(reads)),
                                path=path,
                                support_id=source["id"],
                                support_line=item.lineno,
                                origin="covered_import_syntax_not_runtime_proof",
                            )
                        )
        except (SyntaxError, UnicodeError, ValueError, OSError):
            pass
        try:
            workspace.path(source["path"], write=True)
        except ValueError:
            continue
        indexed = index.db.execute(
            "SELECT hash FROM files WHERE path=?", (source["path"],)
        ).fetchone()
        if not indexed or indexed["hash"] != source["sha256"]:
            continue
        declarations = index.db.execute(
            "SELECT qualname,kind,line,end FROM symbols WHERE path=? AND kind='function' ORDER BY line",
            (source["path"],),
        ).fetchall()
        for declaration in declarations:
            start, end = declaration["line"], declaration["end"]
            if declaration["kind"] != "function" or not _covered(source["spans"], start, end):
                continue
            excerpt = "\n".join(lines[start - 1 : end])
            if len(excerpt) > 1800:
                continue
            targets.append(
                dict(
                    source_id=source["id"],
                    path=source["path"],
                    sha256=source["sha256"],
                    symbol=declaration["qualname"],
                    start=start,
                    end=end,
                    excerpt=excerpt,
                )
            )
    if not targets:
        return None
    # An explicit caller symbol scope outranks unrelated early declarations.
    targets.sort(key=lambda t: len(t["symbol"]) if t["symbol"] in spec.task else 0, reverse=True)
    targets = targets[:4]
    for i, target in enumerate(targets):
        target["id"] = "T" + str(i)
    return dict(
        revision=workspace.revision(),
        contract_ref=contract(state, spec),
        sources=sources,
        targets=targets,
        read_dependencies=reads[:8],
    )


def context(state, spec, workspace, index):
    value = control(state)
    if value["status"] == "due" and state.get("problem_worker", {}).get("status") not in (
        "returned",
        "declined",
        "interrupted",
    ):
        current = offer(state, spec, workspace, index)
        if current is None:
            raise ValueError("No current observed implementation fits the decision capacity")
        value["offered"] = current
        return dict(
            status="choose_one_bounded_next_action",
            **current,
            decisions_left=2 - value["decisions"],
            rule="Two read-only rounds added no source evidence. Use problem_decide: choose one target and a hypothesis (not a fact),then edit;or name one unread range/read_dependency;or supply two discriminating hypotheses for problem_probe. Dependencies use current S IDs. Original checks and total call/time budgets stay unchanged.",
        )
    if value["status"] == "active":
        node = value["node"]
        check_lease(state, spec, workspace, node)
        return dict(
            status="execute_selected_next_action",
            node=node,
            rule="Execute this one ordinary guarded edit,exact targeted read,or declared problem_probe now. Source leases prove no hypothesis. All original independent checks remain required.",
        )
    return None


def check_lease(state, spec, workspace, node):
    from infinicoder.planning.source_plan import stale_sources

    certificate = dict(
        units=[
            dict(
                sources=[
                    dict(path=path, sha256=sha) for path, sha in node["source_versions"].items()
                ]
            )
        ]
    )
    if contract(state, spec) != node["contract_ref"] or stale_sources(certificate, workspace):
        raise ValueError("Decision source/contract lease changed;do not execute the stale node")


def choose(runtime, state, spec, workspace, args):
    from infinicoder.tools.native_tools import dependency

    dependency()(schema()).validate(args)
    value = control(state)
    if value["status"] != "due" or value["decisions"] >= 2:
        raise ValueError("No remaining bounded decision")
    offered = value.get("offered")
    assert offered
    if (
        offered["contract_ref"] != contract(state, spec)
        or offered["revision"] != workspace.revision()
    ):
        raise ValueError("Decision offer changed during the model request")
    target = next((t for t in offered["targets"] if t["id"] == args["target_id"]), None)
    if target is None:
        raise ValueError("Target must be a currently offered observed implementation")
    workspace.path(target["path"], write=True)
    sources = {s["id"]: s for s in offered["sources"]}
    ids = list(dict.fromkeys([target["source_id"], *args["dependency_ids"]]))
    if any(key not in sources for key in ids):
        raise ValueError("Dependency ID lacks current source evidence")
    node = dict(
        mode=args["mode"],
        target=deepcopy(target),
        hypothesis=args["hypothesis"],
        hypothesis_is_fact=False,
        source_versions={sources[key]["path"]: sources[key]["sha256"] for key in ids},
        contract_ref=offered["contract_ref"],
        check_ids=[c.id for c in spec.checks],
        phase=state.get("milestone", 0),
        status="pending",
        acceptance_proof=False,
    )
    if args["mode"] == "read":
        if set(args) - {
            "mode",
            "target_id",
            "hypothesis",
            "dependency_ids",
            "read_ref",
            "start",
            "count",
        }:
            raise ValueError("Read node cannot contain probe hypotheses")
        if not {"read_ref", "start", "count"} <= set(args):
            raise ValueError("A missing read reference and exact range are required")
        source = sources.get(args["read_ref"]) or next(
            (r for r in offered["read_dependencies"] if r["id"] == args["read_ref"]), None
        )
        if source is None:
            raise ValueError("Read must be an offered source or source-derived dependency")
        total = len(workspace.path(source["path"]).read_text(encoding="utf-8-sig").splitlines())
        end = min(total, args["start"] + args["count"] - 1)
        known = next((s for s in sources.values() if s["path"] == source["path"]), None)
        if args["start"] > total or (known and _covered(known["spans"], args["start"], end)):
            raise ValueError("The selected range adds no missing source evidence")
        node["read"] = dict(path=source["path"], start=args["start"], count=args["count"])
        if source.get("support_id"):
            support = sources[source["support_id"]]
            node["source_versions"][support["path"]] = support["sha256"]
    elif args["mode"] == "probe":
        if (
            set(args) - {"mode", "target_id", "hypothesis", "dependency_ids", "hypotheses"}
            or "hypotheses" not in args
        ):
            raise ValueError("Probe node needs competing hypotheses,not a read request")
        node["hypotheses"] = deepcopy(args["hypotheses"])
        if len({digest(h["prediction"]) for h in node["hypotheses"]}) < 2:
            raise ValueError("Probe predictions must discriminate")
    elif set(args) - {"mode", "target_id", "hypothesis", "dependency_ids"}:
        raise ValueError("Edit node cannot contain a read or probe request")
    check_lease(state, spec, workspace, node)
    node["id"] = digest([node, state.get("model_calls"), len(value["nodes"])])
    value["decisions"] += 1
    value.update(status="active", node=node)
    value["nodes"].append(node)
    event(
        runtime,
        state,
        "decision_selected",
        node_id=node["id"],
        mode=node["mode"],
        hypothesis_is_fact=False,
    )
    return dict(node_id=node["id"], selected=True, next_action=node["mode"], acceptance_proof=False)


def guard(state, spec, workspace, action):
    value = state.get("problem_decision", {})
    if action["tool"] == "ask_user":
        return
    # A due observation never revokes ordinary read/edit/verify capabilities.
    if value.get("status") != "active":
        return
    node = value["node"]
    check_lease(state, spec, workspace, node)
    args = action["arguments"]
    expected = dict(edit="edit", read="read", probe="problem_probe")[node["mode"]]
    if action["tool"] != expected:
        raise ValueError("Execute the selected node action;do not restart broad inspection")
    if expected == "edit" and args.get("path") != node["target"]["path"]:
        raise ValueError("Edit target differs from the selected implementation")
    if expected == "read" and args != node["read"]:
        raise ValueError("Read the exact selected missing range once")
    if expected == "problem_probe" and args.get("hypotheses") != node["hypotheses"]:
        raise ValueError("Probe must use the selected competing hypotheses")


def after_action(runtime, state, spec, workspace, index, action, result):
    from infinicoder.tools.read_batch import observations

    value = control(state)
    name = action["tool"]
    if value["status"] == "due" and result.get("error") and name == "problem_decide":
        value["invalid_proposals"] = value.get("invalid_proposals", 0) + 1
        if value["invalid_proposals"] >= 2:
            value["status"] = "closed"
            if state.get("status") == "running":
                state.update(status="failed", reason="bounded_problem_decision_invalid_proposals")
        return
    if value["status"] == "active" and name != "problem_decide":
        node = value["node"]
        changed = (
            name == "edit"
            and not result.get("error")
            and result.get("before") != result.get("after")
            and bool(result.get("after"))
        )
        if changed:
            node.update(
                status="candidate_constructed",
                candidate=dict(
                    path=result["path"],
                    before=result["before"],
                    after=result["after"],
                    revision=workspace.revision(),
                    candidate_id=result.get("problem_candidate_id"),
                ),
            )
            value["status"] = "closed"
            event(
                runtime,
                state,
                "decision_candidate",
                node_id=node["id"],
                candidate=node["candidate"],
                original_accepted=result.get("accepted"),
            )
        elif (
            name in ("read", "problem_probe") and not result.get("error") and value["decisions"] < 2
        ):
            node["status"] = "observation_returned"
            value["status"] = "due"
        else:
            node["status"] = "no_candidate"
            value["status"] = "closed"
            if state.get("status") == "running":
                state.update(status="failed", reason="bounded_problem_decision_no_candidate")
        return
    if name == "problem_decide" or value["status"] != "observing":
        return
    children = list(observations(action, result))
    reads = [r for a, r in children if a["tool"] == "read" and not r.get("error")]
    read_round = bool(reads) and len(reads) == len(children) and not result.get("error")
    if not read_round:
        value["repeat_rounds"] = 0
        return
    progress = any(
        not r.get("inspection_progress", {}).get("repeated_unchanged_range", False) for r in reads
    )
    revision = workspace.revision()
    if progress or value.get("revision") != revision:
        value["repeat_rounds"] = 0
    else:
        value["repeat_rounds"] += 1
    value["revision"] = revision
    if value["repeat_rounds"] >= 2:
        offered = offer(state, spec, workspace, index)
        if offered:
            value.update(status="due", offered=offered)
            event(
                runtime,
                state,
                "decision_due",
                repeat_rounds=value["repeat_rounds"],
                source_revision=revision,
                original_budget_unchanged=True,
            )
