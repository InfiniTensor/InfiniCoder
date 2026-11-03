"""Separate writable permissions from mandatory architecture-planning evidence.

This does not infer task architecture or validate a local repair. Explicit
grounded planning,caller phases and all independent checks keep their gates.
"""

from infinicoder._resources import package_root

from dataclasses import replace
import hashlib


def identity():
    folder = package_root()
    return hashlib.sha256(
        b"".join(
            (folder / name).read_bytes()
            for name in (
                "planning/permission_planning.py",
                "routing/profiles.py",
                "runtime/engine.py",
                "runtime/shell_loop.py",
            )
        )
    ).hexdigest()


def entry(route, spec, grounded_planning):
    if grounded_planning or spec.milestones or not route.needs_plan:
        return route, None
    # profiles.route_task currently derives this gate only from writable-path
    # breadth. Permission is an upper bound,not a list of required modifications.
    decision = dict(
        basis="permission_derived_route",
        original_needs_plan=True,
        decision="defer_permission_inferred_plan",
        architecture_validated=False,
        caller_phases_preserved=True,
        final_checks_unchanged=True,
    )
    route = replace(
        route,
        needs_plan=False,
        reasons=(
            *route.reasons,
            "Broad writable permission alone does not require a pre-edit architecture plan.",
        ),
    )
    return route, decision


def guidance(state):
    if not state.get("permission_plan_entry") or state.get("planning_required"):
        return ""
    return """
Writable permissions are an upper bound,not an instruction to edit every file.
Inspect the actual interfaces and choose the implementation scope from evidence.
For a local repair,you may use edit/command and independent verify/finish directly.
For coordinated modules or new interfaces,call plan with implementation units,
write_set,check_ids,interfaces and dependencies;then follow its staged execution.
Reading a dependency does not make it a write target. You may inspect more or
revisit the plan when evidence changes. Caller phases,permissions,clarification,
domain stages,shared budgets and ALL final checks remain mandatory.
"""
