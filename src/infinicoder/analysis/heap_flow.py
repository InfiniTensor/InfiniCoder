"""Bounded same-root heap update candidates, separate from local use-def.

These links guide source inspection. They do not establish aliasing, execution,
field independence, ownership, or effects of calls. No source code is executed.
"""

import ast
import hashlib
from pathlib import Path


def identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def may_updates(nodes, guards, edges, writes, owner, limit=256, calls=(), priority_lines=()):
    """Connect visible item/field writes to later uses of the same local root.

    A shared positive reaching definition is required. Rebinding a local must
    not carry an earlier object's update to the new object. Alias copies and
    effects through other roots remain explicitly unknown, not independent.
    """
    definitions = {}
    for edge in edges:
        if edge["definition_line"]:
            definitions.setdefault((edge["name"], edge["use_line"]), set()).add(
                edge["definition_line"]
            )
    loops = {
        node.lineno: node for node in nodes if isinstance(node, (ast.For, ast.AsyncFor, ast.While))
    }
    candidates = []
    unresolved = 0
    for write in writes:
        try:
            target = ast.parse(write["target"], mode="eval").body
        except (SyntaxError, ValueError, RecursionError):
            unresolved += 1
            continue
        if not isinstance(target, (ast.Attribute, ast.Subscript)):
            unresolved += 1
            continue
        while isinstance(target, (ast.Attribute, ast.Subscript)):
            target = target.value
        if not isinstance(target, ast.Name):
            unresolved += 1
            continue
        bound = definitions.get((target.id, write["line"]), set())
        if not bound:
            unresolved += 1
            continue
        candidates.append((write, target.id, bound, owner(write["line"])))
    # [] is an exact local list construction, unlike list() which may refer to
    # a shadowed callable. Mixed reaching definitions are not enough to assert
    # that an arbitrary .append has list mutation semantics.
    lists = set()
    for node in nodes:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.List):
            for target in getattr(node, "targets", [getattr(node, "target", None)]):
                if isinstance(target, ast.Name):
                    lists.add((target.id, node.lineno))
    for call in calls:
        try:
            target = ast.parse(call["target"], mode="eval").body
        except (SyntaxError, ValueError, RecursionError):
            continue
        if not (
            isinstance(target, ast.Attribute)
            and target.attr == "append"
            and isinstance(target.value, ast.Name)
        ):
            continue
        root = target.value.id
        bound = definitions.get((root, call["line"]), set())
        if bound and all((root, definition) in lists for definition in bound):
            write = dict(line=call["line"], target=call["target"], operation="local_list_append")
            candidates.append((write, root, bound, owner(call["line"])))
    links = []
    omitted = 0
    priority_lines = set(priority_lines)
    for (base, use_line), use_defs in sorted(
        definitions.items(),
        key=lambda row: (owner(row[0][1]) not in priority_lines, row[0][1], row[0][0]),
    ):
        use_statement = owner(use_line)
        for write, root, write_defs, write_statement in candidates:
            shared = sorted(use_defs & write_defs)
            if base != root or not shared or write_statement == use_statement:
                continue
            common_loops = sorted(
                set(guards.get(write_statement, ()))
                & (
                    set(guards.get(use_statement, ()))
                    | (
                        {use_statement}
                        if isinstance(loops.get(use_statement), ast.While)
                        else set()
                    )
                )
                & loops.keys()
            )
            carried = write_statement > use_statement
            if carried and not common_loops:
                continue
            row = dict(
                kind="may_update",
                base=base,
                target=write["target"],
                operation=write["operation"],
                write_line=write["line"],
                use_line=use_line,
                root_definition_lines=shared,
                basis="same_local_root_and_shared_reaching_definition",
                control_lines=list(guards.get(write_statement, ())),
                loop_carried=carried,
            )
            if carried:
                row["loop_lines"] = common_loops
            if len(links) < limit:
                links.append(row)
            else:
                omitted += 1
    return links, dict(
        status="bounded_scope_local_may_updates",
        omitted_links=omitted,
        unresolved_writes=unresolved,
        limitations=[
            "May affect a later read/return;branches,exceptions and early exits may prevent execution.",
            "Root binding overlap is not object identity;aliases between different names,access paths,properties and dynamic keys remain unknown.",
            "No field independence,callee mutation,ownership or runtime value proof;missing links do not prove independence.",
        ],
    )
