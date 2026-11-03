"""task compiler — contracts become validated work graphs.

Responsibilities (single, per review doc §4.2/§7):
- Convert TaskSpec (+ optional milestones) into WorkUnits with dependencies,
  check bindings and write sets.
- Validate the graph: unique ids, known dependencies, acyclicity, and
  requirement/check coverage. Invalid graphs compile to ZERO executable
  units (the runtime refuses to start).
- Classify failures into categories that drive the next discriminating
  step instead of blind retries.
"""

from dataclasses import dataclass, field

KINDS = ("investigate", "implement", "verify")


class CompileError(ValueError):
    pass


@dataclass
class WorkUnit:
    id: str
    goal: str
    kind: str = "implement"
    check_ids: list = field(default_factory=list)
    dependencies: list = field(default_factory=list)
    write_set: list = field(default_factory=list)
    worker_path: str = None

    def validate(self):
        if not isinstance(self.id, str) or not self.id.strip():
            raise CompileError("unit id must be a non-empty string")
        if not isinstance(self.goal, str) or not self.goal.strip():
            raise CompileError(f"{self.id}: goal must be non-empty")
        if self.kind not in KINDS:
            raise CompileError(f"{self.id}: kind must be one of {KINDS}")


@dataclass
class WorkGraph:
    units: list
    order: list  # topological execution order (linear subset)
    coverage: dict  # check_id -> unit_id

    def unit(self, unit_id):
        for u in self.units:
            if u.id == unit_id:
                return u
        raise KeyError(unit_id)


def _topological_order(units):
    ids = {u.id for u in units}
    indeg = {u.id: 0 for u in units}
    children = {u.id: [] for u in units}
    for u in units:
        for dep in u.dependencies:
            if dep not in ids:
                raise CompileError(f"{u.id}: unknown dependency {dep!r}")
            if dep == u.id:
                raise CompileError(f"{u.id}: self-dependency")
            indeg[u.id] += 1
            children[dep].append(u.id)
    ready = sorted(u.id for u in units if indeg[u.id] == 0)
    order, seen = [], set()
    while ready:
        node = ready.pop(0)
        order.append(node)
        seen.add(node)
        for child in sorted(children[node]):
            indeg[child] -= 1
            if indeg[child] == 0:
                ready.append(child)
    if len(order) != len(units):
        stuck = sorted(set(ids) - seen)
        raise CompileError(f"dependency cycle involving: {stuck}")
    return order


def compile_task(spec):
    """TaskSpec -> WorkGraph. Caller milestones map 1:1 to units (linear
    chain, preserving order); absent milestones compile to a default
    implement+verify unit covering every check. Zero units are produced
    when validation fails — the runtime must not execute."""
    check_ids = [c.id for c in spec.checks]
    units = []
    if spec.milestones:
        for i, m in enumerate(spec.milestones):
            deps = [f"u{i - 1}"] if i else []
            units.append(
                WorkUnit(
                    id=f"u{i}",
                    goal=m["goal"],
                    kind="implement",
                    check_ids=list(m["check_ids"]),
                    dependencies=deps,
                    write_set=list(m.get("write_set", spec.allowed_paths)),
                    worker_path=m.get("worker_path"),
                )
            )
        # Final unit re-verifies every acceptance check (doc: final step
        # rechecks all requirements) when the last milestone doesn't.
        if set(units[-1].check_ids) != set(check_ids):
            units.append(
                WorkUnit(
                    id=f"u{len(units)}",
                    goal="Final verification of every acceptance check",
                    kind="verify",
                    check_ids=list(check_ids),
                    dependencies=[units[-1].id],
                )
            )
    else:
        units.append(
            WorkUnit(
                id="u0",
                goal=spec.task,
                kind="implement",
                check_ids=list(check_ids),
                write_set=list(spec.allowed_paths),
            )
        )
    for u in units:
        u.validate()
    order = _topological_order(units)
    coverage = {}
    for u in units:
        for cid in u.check_ids:
            if cid not in check_ids:
                raise CompileError(f"{u.id}: unknown check {cid!r}")
            coverage[cid] = u.id
    uncovered = [c for c in check_ids if c not in coverage]
    if uncovered:
        raise CompileError(f"acceptance checks not covered by any unit: {uncovered}")
    return WorkGraph(units=units, order=order, coverage=coverage)


# -- failure classification --------------------------------------------------
CATEGORIES = (
    "specification",
    "protocol",
    "provider_config",
    "environment",
    "compilation",
    "semantic",
    "regression",
    "resource",
    "conflict",
    "budget",
    "internal",
)


def classify_failure(result):
    """Map an action/check result to (category, guidance). Guidance names
    the next DISCRIMINATING step; equivalent-repeat avoidance stays in the
    runtime's stagnation counter."""
    if result.get("failure_origin") == "model_phase_scope":
        return (
            "conflict",
            "inspect the caller-authorized path and its model-plan owner;submit a validated replacement plan before any edit;do not expand caller permissions",
        )
    if result.get("failure_origin") == "response_protocol":
        return (
            "protocol",
            "repair the current response/tool schema;this proposal did not enter an action handler and is not an implementation-test failure",
        )
    if result.get("failure_origin") == "design_contract":
        return (
            "protocol",
            "correct design parameters/current source lease;do not treat a malformed symbolic probe as an implementation failure",
        )
    if result.get("evidence"):
        from infinicoder.runtime.feedback import failure_packet

        rows = failure_packet(result)
        if rows:
            return classify_failure(rows[0])
    text = str(
        result.get("error")
        or result.get("stdout_tail")
        or result.get("detail")
        or json.dumps(result)[:400]
    ).lower()
    if "commandargvformaterror" in text:
        return (
            "protocol",
            "pass command.argv as executable plus separate arguments;explicitly invoke a shell for compound shell syntax",
        )
    if result.get("rejected_because"):
        return (
            "semantic",
            f"acceptance rejected: {result['rejected_because']}; inspect failing case detail",
        )
    if (
        "can't open file" in text
        or "declared protected checker path does not exist" in text
        or "no such file or directory" in text
    ):
        return (
            "environment",
            "verify the missing executable, checker script or declared dependency path before changing source",
        )
    if "exit_code" in result and result.get("exit_code") == 125:
        return (
            "resource",
            "process output exceeded disk cap; reduce output volume or narrow the command",
        )
    if "exit_code" in result and result.get("exit_code") == 124:
        return "environment", "process timed out; check for hangs or raise the per-check timeout"
    if "syntaxerror" in text or "modulenotfounderror" in text:
        return "compilation", "fix the syntax/import error shown in the tail first"
    if "assertionerror" in text or '"actual"' in text and '"expected"' in text:
        return (
            "semantic",
            "a test assertion failed; reproduce the exact failing assertion before editing",
        )
    if "expected exactly tool and arguments" in text or "json" in text:
        return "protocol", "emit exactly one JSON action object"
    if "path leaves workspace" in text or "outside the task edit scope" in text:
        return "specification", "requested path is outside the allowed write set"
    if "authentication" in text or "provider" in text:
        return "provider_config", "model endpoint rejected the request"
    return "internal", "inspect the result tail; change one variable and re-run"


import json  # noqa: E402  (used by classify_failure)
