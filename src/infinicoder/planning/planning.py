"""Opt-in import-ordered Python source stages with caller-owned acceptance.

This is structural planning, not architectural or semantic task decomposition.
Provide independent phase checks for meaningful intermediate acceptance; syntax
alone never establishes a module's correctness. Final checks remain mandatory.
"""

import ast, re
from dataclasses import asdict
from pathlib import PurePosixPath
from infinicoder.planning.task_contract import Check, TaskSpec


def bounded_planning_context(assembler, system, data):
    """Keep fitting packets byte-identical;budget optional source facts on overflow.

    Caller requirements,checks,updates and prior interfaces stay mandatory.
    Source records are retained or omitted whole,with their original versions.
    """
    import copy, json
    from infinicoder.models.budget import BudgetExceeded

    original = json.dumps(data, ensure_ascii=False)
    try:
        return assembler.assemble(original, {}, fixed_overhead_text=system)
    except BudgetExceeded as overflow:
        original_error = str(overflow)
    optional = {"read_coverage", "latest_inspection", "source_evidence"}
    required = {k: copy.deepcopy(v) for k, v in data.items() if k not in optional}
    # A genuinely oversized caller contract still fails before any request.
    text, report = assembler.assemble(
        json.dumps(required, ensure_ascii=False), {}, fixed_overhead_text=system
    )
    required_tokens = report["mandatory_tokens"]
    packet = required
    reads = []
    for pair in reversed(data.get("latest_inspection", [])):
        try:
            if not isinstance(pair, list) or not pair or not isinstance(pair[-1], dict):
                continue
            value = json.loads(pair[-1]["content"])
            if isinstance(value, dict):
                reads.append(value)
        except (ValueError, KeyError, TypeError):
            continue
    facts = []
    anchors = []
    routes = []
    for line in str(data.get("source_evidence", "")).splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if not isinstance(value, dict):
            continue
        if "qualified_anchor" in value:
            anchors.append(value)
        elif "route_map" in value:
            routes.append(value)
        else:
            facts.append(value)
    candidates = [("source_evidence", v) for v in anchors]
    if reads:
        candidates.append(("latest_inspection", reads[0]))
    if facts:
        candidates.append(("source_evidence", facts[0]))
    candidates += [("read_coverage", v) for v in data.get("read_coverage", [])]
    for i in range(1, max(len(reads), len(facts))):
        if i < len(reads):
            candidates.append(("latest_inspection", reads[i]))
        if i < len(facts):
            candidates.append(("source_evidence", facts[i]))
    candidates += [("source_evidence", v) for v in routes]
    omitted = {key: sum(name == key for name, _ in candidates) for key in sorted(optional)}
    try:
        probe = dict(packet, evidence_omissions=dict(omitted))
        text, report = assembler.assemble(
            json.dumps(probe, ensure_ascii=False), {}, fixed_overhead_text=system
        )
        packet = probe
    except BudgetExceeded:
        pass
    selected = []
    dropped = []
    for name, value in candidates:
        probe = dict(packet)
        probe[name] = [*packet.get(name, []), value]
        counts = dict(omitted)
        counts[name] -= 1
        if "evidence_omissions" in packet:
            probe["evidence_omissions"] = counts
        identity = value.get("source", value)
        identity = identity if isinstance(identity, dict) else value
        item = dict(
            name=name,
            path=identity.get("path"),
            source_sha256=identity.get("sha256")
            or identity.get("file_hash")
            or identity.get("source_sha256"),
        )
        try:
            proposal, new_report = assembler.assemble(
                json.dumps(probe, ensure_ascii=False), {}, fixed_overhead_text=system
            )
        except BudgetExceeded:
            dropped.append(dict(item, reason="optional evidence exceeds remaining planning budget"))
            continue
        packet = probe
        omitted = counts
        text = proposal
        report = new_report
        selected.append(item)
    report.update(
        adaptive_planning_context=True,
        original_budget_error=original_error,
        immutable_required_tokens=required_tokens,
        required_parts=list(required),
        evidence_selected=selected,
        evidence_dropped=dropped,
        evidence_omitted=omitted,
        evidence_policy="whole source-versioned JSON records;latest observations and ranked index facts interleaved;constraints and prior plan never trimmed",
    )
    return text, report


def plan_schema(check_ids):
    check_ids = list(check_ids)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["units"],
        "properties": {
            "units": {
                "type": "array",
                "minItems": 1,
                "maxItems": 12,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["goal", "write_set", "check_ids", "interfaces", "dependencies"],
                    "properties": {
                        "goal": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 2000,
                            "description": "Concrete implementation/repair deliverable. Include inspection/testing inside it;do not make read-only investigation milestones.",
                        },
                        "write_set": {"type": "array", "minItems": 1, "items": {"type": "string"}},
                        "check_ids": {
                            "type": "array",
                            "items": {"type": "string", "enum": check_ids},
                        },
                        "interfaces": {"type": "array", "items": {"type": "string"}},
                        "dependencies": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 0, "maximum": 11},
                        },
                    },
                },
            }
        },
    }
    if not check_ids:
        schema["properties"]["units"]["items"]["properties"]["check_ids"] = {
            "type": "array",
            "maxItems": 0,
            "items": {"type": "string"},
        }
    return schema


def proposed_plan(spec, arguments, workspace):
    """Validate model-proposed phases without permitting changes to acceptance."""
    if (
        set(arguments) != {"units"}
        or not isinstance(arguments["units"], list)
        or not 1 <= len(arguments["units"]) <= 12
    ):
        raise ValueError("plan requires 1..12 units")
    units = arguments["units"]
    result = TaskSpec.from_dict(asdict(spec))
    result.milestones = []
    all_ids = {c.id for c in spec.checks}
    validated = []
    for i, unit in enumerate(units):
        if not isinstance(unit, dict) or set(unit) - {
            "goal",
            "write_set",
            "check_ids",
            "interfaces",
            "dependencies",
        }:
            raise ValueError("unknown plan fields")
        deps = unit.get("dependencies", [])
        if not isinstance(deps, list) or any(type(d) is not int or d < 0 or d >= i for d in deps):
            raise ValueError("dependencies must refer to earlier units; cycles are forbidden")
        paths = unit.get("write_set")
        if not isinstance(paths, list) or not paths:
            raise ValueError("each unit requires a bounded write_set")
        for path in paths:
            workspace.path(path, write=True)
        checks = unit.get("check_ids", [])
        if not isinstance(checks, list) or any(c not in all_ids for c in checks):
            raise ValueError("plan cannot create or replace acceptance checks")
        interfaces = unit.get("interfaces", [])
        if not isinstance(interfaces, list) or any(
            not isinstance(x, str) or len(x) > 1000 for x in interfaces
        ):
            raise ValueError("interfaces must be bounded strings")
        goal = unit.get("goal")
        if not isinstance(goal, str) or not goal.strip() or len(goal) > 4000:
            raise ValueError("bounded goal required")
        inspection = re.match(
            r"^\s*(?:(?:inspect|read|explore|understand|analy[sz]e|review|search|locate|investigate|run\s+(?:the\s+)?tests)\b|检查|阅读|分析|查找|理解)",
            goal,
            re.I,
        )
        delivers = re.search(
            r"\b(?:implement|fix|repair|add|update|optimize|optimise|change|modify|remove|replace|integrate|write|build|create|refactor|migrate|extend)\b|实现|修复|新增|修改|优化|集成|构建|编写|重构",
            goal,
            re.I,
        )
        validated.append(
            dict(
                goal=goal,
                write_set=list(paths),
                check_ids=list(checks),
                interfaces=list(interfaces),
                inspection_only=bool(inspection and not delivers),
            )
        )
    if all(unit["inspection_only"] for unit in validated):
        raise ValueError(
            "read-only investigation is not an implementation milestone;inspect before planning and place those steps inside a concrete repair/deliverable unit"
        )
    # A common otherwise useful plan starts with investigation and ends with
    # testing. Fold these into an actual deliverable,after validating EVERY
    # original scope,dependency and checker. They never become standalone
    # phases that could be marked complete without changing the implementation.
    normalized = []
    pending = []

    def combine(target, investigations):
        target = dict(target)
        target["goal"] += "\nInvestigation/validation inside this deliverable: " + "; ".join(
            v["goal"] for v in investigations
        )
        for field in ["write_set", "check_ids", "interfaces"]:
            target[field] = list(
                dict.fromkeys(
                    [*target[field], *[item for v in investigations for item in v[field]]]
                )
            )
        return target

    for unit in validated:
        if unit["inspection_only"]:
            pending.append(unit)
            continue
        normalized.append(combine(unit, pending) if pending else unit)
        pending = []
    if pending:
        normalized[-1] = combine(normalized[-1], pending)
    for unit in normalized:
        paths = unit["write_set"]
        checks = unit["check_ids"]
        interfaces = unit["interfaces"]
        goal = unit["goal"]
        item = dict(
            goal=spec.task + "\nPHASE: " + goal + "\nINTERFACES: " + "; ".join(interfaces),
            write_set=paths,
            check_ids=checks,
            provisional=not checks,
        )
        if (
            len(paths) == 1
            and workspace.path(paths[0]).is_file()
            and workspace.path(paths[0]).stat().st_size <= 24000
        ):
            item["worker_path"] = paths[0]
        result.milestones.append(item)
    # The final phase is always bound to ALL caller-owned checks.
    result.milestones[-1]["check_ids"] = [c.id for c in spec.checks]
    result.milestones[-1]["provisional"] = False
    return TaskSpec.from_dict(asdict(result))


def python_file_order(sources):
    if not isinstance(sources, dict) or not sources:
        raise ValueError("sources must be a nonempty path-to-source mapping")
    modules = {}
    for path, source in sources.items():
        if not isinstance(path, str) or not isinstance(source, str):
            raise ValueError("source paths and contents must be strings")
        p = PurePosixPath(path)
        if p.is_absolute() or ".." in p.parts or p.suffix != ".py" or "\\" in path:
            raise ValueError("sources require safe relative POSIX Python paths")
        parts = list(p.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        modules[".".join(parts)] = path
    dependencies = {p: set() for p in sources}
    for path, source in sources.items():
        p = PurePosixPath(path)
        package = list(p.parent.parts)
        for node in ast.walk(ast.parse(source, filename=path)):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    if node.level > len(package):
                        raise ValueError("relative import leaves declared package: " + path)
                    prefix = package[: len(package) - node.level + 1]
                    module = ".".join(prefix + ([node.module] if node.module else []))
                else:
                    module = node.module or ""
                names = [module] + [module + "." + alias.name for alias in node.names]
            for name in names:
                # Include package initializers before imported submodules.
                components = name.split(".")
                for i in range(1, len(components) + 1):
                    candidate = modules.get(".".join(components[:i]))
                    if candidate and candidate != path:
                        dependencies[path].add(candidate)
    order, remaining = [], set(sources)
    while remaining:
        ready = sorted(p for p in remaining if dependencies[p] <= set(order))
        if not ready:
            raise ValueError(
                "cyclic imports require an explicit caller plan: " + ", ".join(sorted(remaining))
            )
        order.append(ready[0])
        remaining.remove(ready[0])
    return order


def stage_python_task(spec, sources, syntax_check, phase_checks=None):
    """Return a new validated TaskSpec; never alters caller task/check objects.

    sources: exact explicit editable files. syntax_check/phase_checks are trusted
    caller-supplied Check instances, outside the model's control. Each stage is
    bound to one source worker, carrying current imported interfaces. All checks
    execute again at the final stage. No automatic backward repair is promised.
    """
    result = TaskSpec.from_dict(asdict(spec))
    if result.milestones:
        raise ValueError("refusing to overwrite caller milestones")
    if not result.checks:
        raise ValueError("an independent final acceptance check is required")
    if set(sources) != set(result.allowed_paths):
        raise ValueError("staging requires exact explicit allowed files")
    if any(not isinstance(source, str) or not source.strip() for source in sources.values()):
        raise ValueError("source workers require existing nonempty source; use comments for stubs")
    if not isinstance(syntax_check, Check):
        raise ValueError("syntax_check must be a caller-provided Check")
    phase_checks = phase_checks or {}
    if set(phase_checks) - set(sources) or any(
        not isinstance(c, Check) for c in phase_checks.values()
    ):
        raise ValueError("phase checks must map known files to Check instances")
    order = python_file_order(sources)
    result.checks += [Check(**asdict(syntax_check))]
    result.checks += [Check(**asdict(phase_checks[p])) for p in order if p in phase_checks]
    all_ids = [c.id for c in result.checks]
    result.milestones = [
        dict(
            goal=result.task
            + "\nCurrent phase: implement only "
            + path
            + "; respect read-only imported interfaces from earlier phases.",
            worker_path=path,
            write_set=[path],
            check_ids=all_ids
            if i == len(order) - 1
            else [syntax_check.id] + ([phase_checks[path].id] if path in phase_checks else []),
        )
        for i, path in enumerate(order)
    ]
    return TaskSpec.from_dict(asdict(result))
