"""Compile source-grounded plans without turning inspected dependencies into edits.

The certificate proves current filesystem/read provenance and bounded scopes.
Goals,requirements and interfaces remain model claims;independent final checks
still determine correctness. Planned artifacts are explicitly not source facts.
"""

from infinicoder._resources import package_root

import copy
import hashlib
import json
import re
from infinicoder.planning.planning import plan_schema, proposed_plan


class SourceEvidenceRequired(ValueError):
    """The next planning step needs a current source observation."""

    def __init__(self, message, *, path=None, existing=None):
        super().__init__(message)
        self.path, self.existing = path, existing


def record_planning_rejection(state, error, workspace):
    """Controller feedback is a rejected proposal,never independent test evidence."""
    feedback = dict(
        origin="controller_plan_validation",
        plan_accepted=False,
        error_type=type(error).__name__,
        error=str(error)[:640],
        at_step=state.get("steps"),
        next="Revise the rejected proposal using current sources;all caller checks remain mandatory.",
    )
    if isinstance(error, SourceEvidenceRequired):
        feedback.update(required_path=error.path, existing=error.existing)
        state["planning_inspection_required"] = True
        if error.existing:
            try:
                feedback["source_sha256"] = hashlib.sha256(
                    workspace.path(error.path).read_bytes()
                ).hexdigest()
            except (OSError, ValueError, TypeError):
                pass
        feedback["next"] = (
            "Read the named current dependency before planning again."
            if error.existing
            else "Locate actual paths with files/search,or explicitly declare a required new file;do not read a nonexistent path."
        )
    elif state.get("planning_inspection_required"):
        prior = state.get("planning_feedback", {})
        for key in ("required_path", "existing", "source_sha256", "observed_sha256"):
            if key in prior:
                feedback[key] = prior[key]
    state["planning_feedback"] = feedback
    return feedback


def planning_feedback_view(state, workspace):
    feedback = copy.deepcopy(state.get("planning_feedback", {}))
    if feedback.get("required_path") and feedback.get("existing") is True:
        try:
            path = workspace.path(feedback["required_path"])
            current = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        except (OSError, ValueError):
            current = None
        feedback.update(
            target_exists=current is not None,
            current_source_sha256=current,
            inspection_current=current is not None and current == feedback.get("observed_sha256"),
        )
        if feedback["inspection_current"]:
            feedback["next"] = (
                "The named dependency has a current read observation;revise the proposal or choose a justified local repair."
            )
        elif current is None:
            feedback["next"] = (
                "The previously named dependency no longer exists;locate actual paths or revise its declaration."
            )
        else:
            feedback["next"] = (
                "Read the named dependency at its current version before planning again."
            )
    return feedback


def record_planning_read(state, result, workspace):
    if result.get("total_lines", 0) > 0 and not result.get(
        "covered_line_count", len(result.get("text", "").splitlines())
    ):
        return
    feedback = state.get("planning_feedback", {})
    if feedback.get("existing") is True:
        path = workspace.path(result["path"]).relative_to(workspace.root).as_posix()
        if path != feedback.get("required_path"):
            return
        feedback["observed_sha256"] = result["sha256"]
    state.pop("planning_inspection_required", None)


def identity():

    folder = package_root()
    return hashlib.sha256(
        b"".join(
            (folder / name).read_bytes()
            for name in [
                "planning/source_plan.py",
                "analysis/local_edit.py",
                "memory/source/source_scope.py",
                "execution/native_scope.py",
                "runtime/engine.py",
                "runtime/protocol.py",
                "runtime/worker.py",
                "runtime/work_packet.py",
                "analysis/code_context.py",
                "languages/clang_adapter.py",
                "languages/native_adapter.py",
                "analysis/native_effects.py",
            ]
        )
    ).hexdigest()


def grounded_plan_schema(check_ids):
    schema = plan_schema(check_ids)
    unit = schema["properties"]["units"]["items"]
    unit["required"] += ["read_set", "creations"]
    unit["properties"]["write_set"].update(
        minItems=0,
        maxItems=24,
        description="Concrete files to modify. Reading a dependency does not grant a reason to edit it. Inspection-only units have an empty write_set.",
    )
    unit["properties"]["read_set"] = dict(
        type="array",
        maxItems=48,
        items=dict(type="string"),
        description="Existing source dependencies to consult;these are not modification targets.",
    )
    unit["properties"]["creations"] = dict(
        type="array",
        maxItems=24,
        items=dict(
            type="object",
            additionalProperties=False,
            required=["path", "requirement", "interface"],
            properties={
                name: dict(type="string", minLength=1, maxLength=2000)
                for name in ["path", "requirement", "interface"]
            },
        ),
        description="Every new file needs a requirement and an interface. Existing files must be grounded in current inspected source.",
    )
    return schema


def _inspection_only(goal):
    inspect = re.match(
        r"^\s*(?:(?:inspect|read|explore|understand|analy[sz]e|review|search|locate|investigate|verify|test|run\s+(?:the\s+)?tests)\b|检查|阅读|分析|查找|理解|验证|测试)",
        goal,
        re.I,
    )
    change = re.search(
        r"\b(?:implement|fix|repair|add|update|optimize|optimise|change|modify|remove|replace|integrate|write|build|create|refactor|migrate|extend)\b|实现|修复|新增|修改|优化|集成|构建|编写|重构",
        goal,
        re.I,
    )
    return bool(inspect and not change)


def _path(workspace, path, write=False):
    if not isinstance(path, str) or not path or any(c in path for c in "*?["):
        raise ValueError("plan paths must identify concrete relative files")
    value = workspace.path(path, write=write)
    if value.is_dir():
        raise ValueError("plan targets must be files,not directories")
    return value.relative_to(workspace.root).as_posix(), value


def grounded_proposed_plan(spec, arguments, workspace, inspected):
    """inspected is runtime-owned read/index evidence,never model arguments."""
    if (
        set(arguments) != {"units"}
        or not isinstance(arguments["units"], list)
        or not 1 <= len(arguments["units"]) <= 12
    ):
        raise ValueError("plan requires1..12 units")
    seen = {}
    for record in inspected:
        if not isinstance(record, dict) or record.get("source_omitted"):
            continue
        try:
            path, file = _path(workspace, record["path"])
            digest = record.get("sha256", record.get("source_sha256"))
            if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != digest:
                continue
        except (OSError, ValueError, KeyError):
            continue
        fact = dict(
            path=path,
            sha256=digest,
            spans=copy.deepcopy(record.get("spans", record.get("read_spans", []))),
            origin=record.get("origin", "current_read_or_source_excerpt"),
        )
        if (
            type(record.get("excerpt_line")) is int
            and type(record.get("excerpt_end")) is int
            and record["excerpt_line"] <= record["excerpt_end"]
        ):
            fact["spans"].append([record["excerpt_line"], record["excerpt_end"]])
        for key in ("partial_prefix_characters", "total_lines", "full_file_visible"):
            if key in record:
                fact[key] = copy.deepcopy(record[key])
        if path in seen:
            previous = seen[path]
            fact["spans"] += previous["spans"]
            fact.update(
                {
                    key: previous[key]
                    for key in ("partial_prefix_characters", "total_lines", "full_file_visible")
                    if key in previous
                }
            )
            fact["origin"] = previous["origin"]
        merged = []
        for first, last in sorted(fact["spans"]):
            if merged and first <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], last)
            else:
                merged.append([first, last])
        fact["spans"] = merged
        seen[path] = fact
    allowed_checks = {c.id for c in spec.checks}
    normalized = []
    proof = []
    pending = []
    closure = []
    produced = {}
    for ordinal, raw in enumerate(arguments["units"]):
        required = {
            "goal",
            "write_set",
            "read_set",
            "creations",
            "check_ids",
            "interfaces",
            "dependencies",
        }
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError("grounded plan unit fields are incomplete or unknown")
        goal = raw["goal"]
        if not isinstance(goal, str) or not goal.strip() or len(goal) > 4000:
            raise ValueError("bounded implementation goal required")
        deps = raw["dependencies"]
        if not isinstance(deps, list) or any(
            type(d) is not int or d < 0 or d >= ordinal for d in deps
        ):
            raise ValueError("dependencies must reference earlier units")
        ancestors = set(deps)
        for dependency in deps:
            ancestors.update(closure[dependency])
        closure.append(ancestors)
        checks = raw["check_ids"]
        interfaces = raw["interfaces"]
        if not isinstance(checks, list) or any(c not in allowed_checks for c in checks):
            raise ValueError("caller acceptance checks cannot be invented or replaced")
        if not isinstance(interfaces, list) or any(
            not isinstance(v, str) or not v.strip() or len(v) > 1000 for v in interfaces
        ):
            raise ValueError("bounded interfaces required")
        for key, limit in [("write_set", 24), ("read_set", 48), ("creations", 24)]:
            if not isinstance(raw[key], list) or len(raw[key]) > limit:
                raise ValueError("bounded " + key + " required")
        inspection = _inspection_only(goal)
        if inspection and raw["creations"]:
            raise ValueError("inspection cannot declare a new implementation file")
        reads = (
            list(raw["read_set"]) + list(raw["write_set"]) if inspection else list(raw["read_set"])
        )
        writes = [] if inspection else list(raw["write_set"])
        if not inspection and not writes:
            raise ValueError("implementation unit needs a concrete write_set")
        creations = {}
        for declaration in raw["creations"]:
            if not isinstance(declaration, dict) or set(declaration) != {
                "path",
                "requirement",
                "interface",
            }:
                raise ValueError("creation requires path,requirement and interface")
            if any(
                not isinstance(declaration[k], str)
                or not declaration[k].strip()
                or len(declaration[k]) > 2000
                for k in declaration
            ):
                raise ValueError("creation claims must be bounded strings")
            path, file = _path(workspace, declaration["path"], write=True)
            if path in creations or file.exists() or path in produced:
                raise ValueError("creation target already exists or has a producer")
            creations[path] = dict(declaration, path=path)
        facts = []
        read_paths = []
        write_paths = []
        for kind, paths in [("read", reads), ("write", writes)]:
            for requested in paths:
                path, file = _path(workspace, requested, write=kind == "write")
                if file.is_file():
                    if path not in seen:
                        raise SourceEvidenceRequired(
                            "inspect current source before planning " + kind + ": " + path,
                            path=path,
                            existing=True,
                        )
                    fact = dict(seen[path], role=kind)
                elif kind == "write" and path in creations:
                    fact = dict(
                        path=path,
                        role=kind,
                        origin="declared_new_file",
                        sha256=None,
                        model_claim=creations[path],
                    )
                elif path in produced and produced[path] in ancestors:
                    fact = dict(
                        path=path,
                        role=kind,
                        origin="planned_artifact",
                        sha256=None,
                        producer_unit=produced[path],
                    )
                else:
                    raise SourceEvidenceRequired(
                        "missing source requires an explicit creation or earlier dependency: "
                        + path,
                        path=path,
                        existing=False,
                    )
                facts.append(fact)
                (read_paths if kind == "read" else write_paths).append(path)
        if set(creations) - set(write_paths):
            raise ValueError("creation must be an explicit modification target")
        item = dict(
            goal=goal,
            write_set=list(dict.fromkeys(write_paths)),
            check_ids=list(checks),
            interfaces=list(interfaces),
            dependencies=[],
        )
        record = dict(
            original_unit=ordinal,
            goal=goal,
            read_set=list(dict.fromkeys(read_paths)),
            write_set=item["write_set"],
            sources=facts,
            requirement_and_interface_claims=list(creations.values()),
            dependencies=deps,
        )
        if inspection:
            pending.append((item, record))
            continue
        for path in creations:
            produced[path] = ordinal
        if pending:
            item["goal"] += "\nInspection supporting this deliverable: " + "; ".join(
                v[0]["goal"] for v in pending
            )
            item["check_ids"] = list(
                dict.fromkeys(
                    [*item["check_ids"], *[c for v, _ in pending for c in v["check_ids"]]]
                )
            )
            item["interfaces"] = list(
                dict.fromkeys(
                    [*item["interfaces"], *[c for v, _ in pending for c in v["interfaces"]]]
                )
            )
            record["read_set"] = list(
                dict.fromkeys(
                    [*record["read_set"], *[p for _, v in pending for p in v["read_set"]]]
                )
            )
            record["sources"] = [f for _, v in pending for f in v["sources"]] + record["sources"]
            pending = []
        normalized.append(item)
        proof.append(record)
    if not normalized:
        raise ValueError("inspection alone is not an implementation plan")
    if pending:
        normalized[-1]["goal"] += "\nValidation within this deliverable: " + "; ".join(
            v["goal"] for v, _ in pending
        )
        proof[-1]["read_set"] = list(
            dict.fromkeys([*proof[-1]["read_set"], *[p for _, v in pending for p in v["read_set"]]])
        )
        proof[-1]["sources"] += [f for _, v in pending for f in v["sources"]]
    for ordinal, item in enumerate(normalized):
        item["dependencies"] = list(range(ordinal))
        if proof[ordinal]["read_set"]:
            item["goal"] += (
                "\nRead dependencies (preserve their existing interfaces unless separately declared as writes): "
                + ", ".join(proof[ordinal]["read_set"])
            )
    planned = proposed_plan(spec, dict(units=normalized), workspace)
    certificate = dict(
        schema=1,
        analysis_identity=identity(),
        units=proof,
        source_facts="Current source hashes/read locations only. Planned artifacts and model interface/requirement claims are unverified.",
        validity="At plan installation;check source leases again before later execution. Final external checks remain mandatory.",
    )
    certificate["digest"] = hashlib.sha256(
        json.dumps(certificate, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return planned, certificate


def runtime_sources(state, workspace, index=None, arguments=None):
    """Collect only controller-owned observations;the validator rechecks hashes."""
    from infinicoder.memory.inspection_memory import current_coverage

    if index is not None and getattr(index, "task_focused_memory", False):
        from infinicoder.memory.inspection_ledger import InspectionLedger

        paths = None
        if (
            isinstance(arguments, dict)
            and isinstance(arguments.get("units"), list)
            and len(arguments["units"]) <= 12
        ):
            paths = []
            for unit in arguments["units"]:
                if not isinstance(unit, dict):
                    continue
                for key, bound in [("read_set", 48), ("write_set", 24)]:
                    values = unit.get(key, [])
                    if isinstance(values, list) and len(values) <= bound:
                        paths.extend(p for p in values if isinstance(p, str))
        records = InspectionLedger(index).current(paths)
    else:
        records = [
            dict(record, origin="versioned_read")
            for record in current_coverage(state, workspace, max_chars=200000)
        ]
    records += [
        dict(record, origin="indexed_source_excerpt")
        for record in state.get("code_memory_report", {}).get("selected", [])
        if not record.get("source_omitted")
    ]
    return records


def stale_sources(certificate, workspace):
    """Return invalidated observed sources;planned artifacts have no source lease."""
    stale = set()
    for unit in certificate["units"]:
        for record in unit["sources"]:
            if not record.get("sha256"):
                continue
            try:
                _, path = _path(workspace, record["path"])
                current = hashlib.sha256(path.read_bytes()).hexdigest()
            except (OSError, ValueError):
                current = None
            if current != record["sha256"]:
                stale.add(record["path"])
    return sorted(stale)
