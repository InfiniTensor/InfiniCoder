"""Bounded Python code artifacts. Draft syntax is never acceptance evidence.

The runtime owns requests, permissions, live writes and all independent checks.
Nothing in this module executes a proposal or salvages truncated output.
"""

from infinicoder._resources import package_file

import ast
import hashlib
import json
import re
from dataclasses import asdict

POLICY = "bounded_units_v1"
LIMIT = 3
DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
ORDINARY_SOURCE_ACTIONS = """
Ordinary JSON actions remain available instead of a source proposal:
{"tool":"read","arguments":{"path":"relative.py"}}, search {query},
edit {path,expected_hash,old,new}, command {argv:[str]}, verify {}, finish {},
or ask_user {question}. Original workspace permissions and budgets apply.
These actions operate on live code and can leave the source proposal workflow.
"""


def ordinary_action(text):
    """Only whole-response JSON is a control; JSON inside Python is data."""
    from infinicoder.runtime.protocol import final_payload

    payload = final_payload(text)
    if payload.startswith("```json\n") and payload.endswith("\n```"):
        payload = payload[len("```json\n") : -len("\n```")]
    try:
        value = json.loads(payload)
    except (ValueError, TypeError):
        return None
    return (
        value
        if isinstance(value, dict)
        and set(value) == {"tool", "arguments"}
        and isinstance(value["tool"], str)
        and isinstance(value["arguments"], dict)
        else None
    )


def identity():
    names = (
        "memory/source/source_units.py",
        "runtime/engine.py",
        "memory/source/source_scope.py",
        "execution/workspace_tools.py",
        "runtime/worker.py",
        "runtime/work_packet.py",
        "runtime/protocol.py",
        "runtime/feedback.py",
    )
    return dict(
        policy=POLICY,
        implementation={
            name: hashlib.sha256(package_file(name).read_bytes()).hexdigest() for name in names
        },
    )


def charge_proposal(episode):
    if episode["status"] != "active" or episode["proposals"] >= LIMIT:
        raise ValueError("Source-unit episode exhausted; use ordinary actions")
    episode["proposals"] += 1
    if episode["proposals"] >= LIMIT:
        episode["limit_reached"] = True


SYSTEM = """This SYSTEM defines the current response transport. The original task/reference
retains its functional,backend,interface and performance requirements;earlier
source-only output wording does not override this transport.
Implement ONE complete Python source unit in the supplied draft.
Return one complete function, small NEW class, or import statement as Python
source, optionally in one fence. Existing functions keep their exact interface.
For an existing/new class method use the JSON action source_unit with arguments
code and target (qualified Class.method); new methods use exact class indentation.
source_unit may also include submit:true to commit the composed draft explicitly.
source_submit with empty arguments submits an already composed draft.
Ordinary JSON actions read/search/edit/command/verify/finish/ask_user remain
available under the original permissions. They read/execute LIVE code, never
the draft. Ordinary live changes invalidate the draft. Do not repeat the whole
module. Define a needed helper directly, then connect the required entry to it.
The original task, backend, interfaces and failures still apply. Drafts have
NOT been executed. Structural completion is not correctness. Only the original
independent checks decide acceptance. Never return a truncated code fragment."""


def digest(source):
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def interfaces(value):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("source_unit_interfaces must map paths to qualified symbols")
    result = {}
    for path, names in value.items():
        if (
            not isinstance(path, str)
            or not path.endswith(".py")
            or not isinstance(names, (list, tuple))
        ):
            raise ValueError("Python paths and symbol lists are required")
        if any(
            not isinstance(n, str) or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", n)
            for n in names
        ):
            raise ValueError("Invalid required source-unit interface")
        result[path] = list(dict.fromkeys(names))
    return result


def _definitions(tree):
    result = {}

    def visit(owner, prefix=""):
        for node in owner.body:
            if isinstance(node, DEFINITIONS):
                key = prefix + node.name
                if key in result:
                    raise ValueError("Ambiguous duplicate definition: " + key)
                result[key] = node
                if isinstance(node, ast.ClassDef):
                    visit(node, key + ".")

    visit(tree)
    return result


def _bindings(tree):
    names = []
    for node in tree.body:
        if isinstance(node, DEFINITIONS):
            names.append(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if any(a.name == "*" for a in node.names):
                raise ValueError("Wildcard imports require ordinary editing")
            names.extend(
                a.asname or (a.name.split(".")[0] if isinstance(node, ast.Import) else a.name)
                for a in node.names
            )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                names.extend(
                    n.id
                    for n in ast.walk(target)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                )
        else:
            # Conditional/dynamic bindings cannot safely be treated as absent.
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                    names.append(child.id)
                elif isinstance(child, DEFINITIONS):
                    names.append(child.name)
                elif isinstance(child, (ast.Import, ast.ImportFrom)):
                    names.extend(a.asname or a.name.split(".")[0] for a in child.names)
    return names


def _stub(node):
    if isinstance(node, ast.ClassDef):
        return not any(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not _stub(n)
            for n in node.body
        )
    body = [
        n
        for n in node.body
        if not (
            isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Constant)
            and isinstance(n.value.value, str)
        )
    ]
    return not body or all(
        isinstance(n, ast.Pass)
        or isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Constant)
        and n.value.value is Ellipsis
        or isinstance(n, ast.Raise)
        and any(isinstance(v, ast.Name) and v.id == "NotImplementedError" for v in ast.walk(n))
        for n in body
    )


def binding(spec, phase):
    from infinicoder.memory.repair_memory import signature

    return dict(requirements=signature(spec), phase=phase)


def check_view(spec, state, workspace):
    checked = state.get("source_unit_last_check")
    if checked is None:
        return None
    from infinicoder.memory.repair_memory import signature

    return dict(
        checked,
        same_live_revision=checked["live_revision"] == workspace.revision(),
        same_requirements=checked["requirements"] == signature(spec),
    )


def prepare_unit(spec, state, workspace, path, required=(), qualified_target=None):
    """Start one durable episode. A closed episode cannot silently restart."""
    old = state.get("source_units")
    if old is not None:
        raise ValueError("Source-unit episode already used; continue ordinary actions")
    file = workspace.path(path, write=True)
    if not path.endswith(".py") or not file.is_file():
        raise ValueError("Source units require an existing Python target; use ordinary edits")
    source = file.read_bytes().decode("utf-8")
    if not source:
        raise ValueError("Empty files require ordinary creation/editing")
    tree = ast.parse(source, filename=path)
    definitions = _definitions(tree)
    missing = [n for n in required if n not in definitions or _stub(definitions[n])]
    if qualified_target is None:
        # Only explicit caller interfaces or a real diagnostic line can narrow
        # an existing scope. A missing interface permits a new helper directly.
        if not missing and len(required) == 1 and required[0] in definitions:
            qualified_target = required[0]
        if qualified_target is None and not missing:
            failure = json.dumps(state.get("last_failure"), ensure_ascii=False)
            lines = [
                int(n)
                for n in re.findall(re.escape(path) + r"[:\"](?:,?\s*line\s*)?(\d+)", failure)
            ]
            candidates = [
                (n.end_lineno - n.lineno, name)
                for name, n in definitions.items()
                if not isinstance(n, ast.ClassDef)
                and any(n.lineno <= line <= n.end_lineno for line in lines)
            ]
            if candidates:
                qualified_target = min(candidates)[1]
    mode = "construct" if missing or not tree.body else "unknown"
    # A failed joint/performance check is not proof of a semantic/admission cause.
    episode = dict(
        policy=POLICY,
        status="active",
        path=path,
        base_sha256=digest(source),
        base_revision=workspace.revision(),
        live_source=source,
        draft_source=source,
        contract=binding(spec, state["milestone"]),
        required=list(required),
        target=qualified_target,
        granularity="definition",
        proposals=0,
        incomplete=1,
        units=[],
        mode=mode,
        mode_reason="Missing/stub caller interface"
        if mode == "construct"
        else "No independent classified correctness evidence",
        acceptance_proof=False,
    )
    state["source_units"] = episode
    return episode


def validate(episode, spec, state, workspace):
    if episode["status"] != "active":
        raise ValueError("Source-unit episode is closed")
    file = workspace.path(episode["path"], write=True)
    if (
        episode["contract"] != binding(spec, state["milestone"])
        or workspace.revision() != episode["base_revision"]
        or digest(file.read_bytes().decode("utf-8")) != episode["base_sha256"]
    ):
        episode.update(status="invalidated", reason="Live source or caller contract changed")
        raise ValueError(episode["reason"] + "; draft was not applied")


def stop(episode, reason):
    episode.update(status="closed", reason=reason)


def incomplete(episode):
    charge_proposal(episode)
    episode["incomplete"] += 1
    # The initial whole-file length already consumed the one safe reduction.
    stop(episode, "Two incomplete replies; use ordinary actions without a new episode")


def _source(text):
    from infinicoder.runtime.protocol import final_payload

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Expected complete source text")
    # Leading method indentation is source data. Generic transport stripping
    # would turn an explicitly scoped method into an invalid top-level fragment.
    text = text.lstrip("\r\n").rstrip("\r\n")
    if text.lstrip().startswith("<think>"):
        text = final_payload(text)
    if text.startswith("```"):
        lines = text.splitlines(keepends=True)
        if lines[0].strip() not in ("```", "```py", "```python") or lines[-1].strip() != "```":
            raise ValueError("Expected one complete Python source fence")
        text = "".join(lines[1:-1])
    return text if text.endswith("\n") else text + "\n"


def stage_unit(episode, complete_response, finish_reason="stop", target=None, body=False):
    if episode["status"] != "active" or episode["proposals"] >= LIMIT:
        raise ValueError("Source-unit episode exhausted; use ordinary actions")
    if finish_reason != "stop":
        incomplete(episode)
        raise ValueError("Incomplete source unit was neither staged nor executed")
    charge_proposal(episode)
    # Charge invalid complete proposals, too; the runtime charges original usage.
    try:
        source = _source(complete_response)
        before = episode["draft_source"]
        tree = ast.parse(before)
        definitions = _definitions(tree)
        if body:
            if (
                not target
                or target not in definitions
                or isinstance(definitions[target], ast.ClassDef)
            ):
                raise ValueError("A body requires an exact existing function target")
            from infinicoder.memory.source.source_scope import packet, restore

            scope = packet(before, target)
            old = ast.parse(scope["source"]).body[0]
            header = "".join(scope["source"].splitlines(keepends=True)[: old.body[0].lineno - 1])
            if old.body[0].lineno == old.lineno:
                raise ValueError("Inline function bodies require ordinary edits")
            # Body has explicit indentation, no text slicing or algorithm synthesis.
            proposed = header + source
            fragment = restore(scope, proposed)
            lines = before.splitlines(keepends=True)
            after = "".join(lines[: scope["start"] - 1]) + fragment + "".join(lines[scope["end"] :])
            name = target
        else:
            if target and "." in target and target not in definitions:
                parent, name = target.rsplit(".", 1)
                owner = definitions.get(parent)
                if not isinstance(owner, ast.ClassDef):
                    raise ValueError("New method requires a known class owner")
                # Parse under an actual class wrapper; preserve the supplied bytes.
                wrapper = ast.parse("class _Owner:\n" + source)
                if len(wrapper.body) != 1 or not isinstance(wrapper.body[0], ast.ClassDef):
                    raise ValueError("Method unit cannot contain declarations outside its owner")
                members = wrapper.body[0].body
                if (
                    len(members) != 1
                    or not isinstance(members[0], (ast.FunctionDef, ast.AsyncFunctionDef))
                    or members[0].name != name
                ):
                    raise ValueError("Expected one complete class method")
                indent = owner.body[0].col_offset
                if members[0].col_offset != indent:
                    raise ValueError("New method must use the exact existing class indentation")
                if name in _bindings(owner):
                    raise ValueError("New method collides with an existing class binding")
                lines = before.splitlines(keepends=True)
                after = (
                    "".join(lines[: owner.end_lineno]).rstrip("\r\n")
                    + "\n"
                    + source
                    + "".join(lines[owner.end_lineno :])
                )
                name = target
            else:
                nodes = ast.parse(source).body
                if len(nodes) != 1 or not isinstance(
                    nodes[0], (*DEFINITIONS, ast.Import, ast.ImportFrom)
                ):
                    raise ValueError("Propose exactly one complete definition or import")
                node = nodes[0]
                for owner in (n for n in ast.walk(node) if isinstance(n, ast.ClassDef)):
                    bindings = _bindings(owner)
                    if len(bindings) != len(set(bindings)):
                        raise ValueError(
                            "New class contains conflicting bindings; use ordinary edits"
                        )
                name = target or getattr(node, "name", None)
                if target and getattr(node, "name", None) != target.split(".")[-1]:
                    raise ValueError("Unit name does not match its exact target")
                if name in definitions:
                    if isinstance(definitions[name], ast.ClassDef):
                        raise ValueError(
                            "Modify existing classes with method units or ordinary edits"
                        )
                    from infinicoder.memory.source.source_scope import packet, restore

                    scope = packet(before, name)
                    fragment = restore(scope, source)
                    lines = before.splitlines(keepends=True)
                    after = (
                        "".join(lines[: scope["start"] - 1])
                        + fragment
                        + "".join(lines[scope["end"] :])
                    )
                else:
                    added = _bindings(ast.parse(source))
                    if (
                        not added
                        or len(added) != len(set(added))
                        or set(added) & set(_bindings(tree))
                    ):
                        raise ValueError("Unit binding collision; use ordinary edits")
                    if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                        raise ValueError("Future imports require ordinary edits")
                    after = before + ("" if before.endswith("\n") else "\n") + source
                    name = name or ",".join(added)
        if len(after.encode("utf-8")) > 1_000_000:
            raise ValueError("Draft exceeds original edit bound")
        compile(after, episode["path"], "exec", dont_inherit=True)
        episode["draft_source"] = after
        episode["units"].append(
            dict(
                target=name,
                code=source,
                sha256=digest(source),
                draft_sha256=digest(after),
                body=body,
                finish_reason=finish_reason,
            )
        )
        episode["target"] = next(
            (
                n
                for n in episode["required"]
                if n not in _definitions(ast.parse(after))
                or _stub(_definitions(ast.parse(after))[n])
            ),
            None,
        )
        return structural_status(episode)
    finally:
        if episode["proposals"] >= LIMIT:
            episode["limit_reached"] = True


def structural_status(episode):
    definitions = _definitions(ast.parse(episode["draft_source"]))
    missing = [n for n in episode["required"] if n not in definitions or _stub(definitions[n])]
    reached = set(episode["required"]) - set(missing)
    pending = list(reached)
    while pending:
        name = pending.pop()
        node = definitions[name]
        for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
            target = call.func.id if isinstance(call.func, ast.Name) else None
            if (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id in ("self", "cls")
                and "." in name
            ):
                target = name.rsplit(".", 1)[0] + "." + call.func.attr
            if target in definitions and target not in reached:
                reached.add(target)
                pending.append(target)
    added = {row["target"] for row in episode["units"] if row["target"] in definitions}
    unconnected = sorted(added - reached)
    return dict(
        draft_sha256=digest(episode["draft_source"]),
        live_sha256=episode["base_sha256"],
        proposals=episode["proposals"],
        remaining_proposals=max(0, LIMIT - episode["proposals"]),
        missing_interfaces=missing,
        defined_not_statically_connected=unconnected,
        auto_submittable=bool(episode["required"]) and not missing and not unconnected,
        relation_limit="Direct syntactic calls only; dynamic dispatch and execution unknown",
        live_changed=False,
        acceptance_proof=False,
    )


def commit_action(episode, spec, state, workspace, explicit=False):
    """Lower to the ORIGINAL guarded edit; the runtime performs/checks the write."""
    validate(episode, spec, state, workspace)
    status = structural_status(episode)
    if (
        not episode["units"]
        or status["missing_interfaces"]
        or not (explicit or status["auto_submittable"])
    ):
        raise ValueError(
            "Draft structure is incomplete/ambiguous; connect interfaces or use ordinary edits"
        )
    compile(episode["draft_source"], episode["path"], "exec", dont_inherit=True)
    return dict(
        tool="edit",
        arguments=dict(
            path=episode["path"],
            expected_hash=episode["base_sha256"],
            old=episode["live_source"],
            new=episode["draft_source"],
        ),
    )


def context(runtime, spec, state, workspace):
    episode = state["source_units"]
    validate(episode, spec, state, workspace)
    from infinicoder.runtime.worker import interfaces as local_interfaces

    unit = runtime._graph.unit(runtime._graph.order[state["milestone"]])
    checked = check_view(spec, state, workspace)
    from infinicoder.memory.source.source_handoff import project_history

    view = dict(
        task_contract=asdict(spec),
        source_unit_status=structural_status(episode),
        mode=episode["mode"],
        mode_reason=episode["mode_reason"],
        selected_target=episode["target"],
        granularity=episode["granularity"],
        draft_source=episode["draft_source"],
        live_sha256=episode["base_sha256"],
        live_dependency_revision=episode["base_revision"],
        required_interfaces=episode["required"],
        original_check_failure=checked if checked and not checked["accepted"] else None,
        latest_original_check=checked,
        last_action_or_transport_feedback=state.get("last_failure"),
        direct_local_interfaces=local_interfaces(
            workspace, episode["path"], episode["draft_source"]
        ),
        caller_capabilities=runtime.registry.context(spec.capabilities, spec.target_facts)
        if spec.capabilities
        else "",
        domain_contract=runtime._domain_design_view(spec, state, workspace, unit),
        teaching_material=list(runtime.worker_cards),
        recent_actual_observations=project_history(state, "python_source_unit", window=2),
        remaining_model_calls=max(0, runtime.max_model_calls - state.get("model_calls", 0)),
        draft_is_not_live=True,
        independent_acceptance_still_required=True,
    )
    # All original constraints/dependencies/failure are mandatory. No trimming to fit.
    text, report = runtime.assembler.assemble(
        json.dumps(view, ensure_ascii=False), {}, fixed_overhead_text=SYSTEM
    )
    report.update(
        presentation="source_unit",
        source_unit_status=view["source_unit_status"],
        acceptance_proof=False,
    )
    state["budget_report"] = report
    return [dict(role="system", content=SYSTEM), dict(role="user", content=text)]
