"""Canonical protocol boundary."""

import json


def final_payload(text):
    """Discard leading complete reasoning frames before interpreting a proposal.

    Raw responses remain in the session log. Never recover a draft action from
    an unfinished reasoning frame or execute JSON found inside that frame.
    """
    if not isinstance(text, str):
        raise ValueError("Model payload must be text")
    text = text.strip()
    while text.startswith("<think>"):
        end = text.find("</think>", len("<think>"))
        if end < 0:
            raise ValueError("Incomplete reasoning frame; no proposal was executed")
        text = text[end + len("</think>") :].strip()
    if not text:
        raise ValueError("Model supplied no final proposal")
    return text


INSPECTION_TOOLS = (
    "files",
    "directories",
    "dependencies",
    "tests_for",
    "search",
    "text_search",
    "symbol",
    "relationships",
    "structure",
    "impact",
    "read",
    "workpack",
    "use_def",
    "memory_route",
    "memory_expand",
    "knowledge",
    "seeds",
    "note",
    "ask_user",
)


class ActionProtocolError(ValueError):
    """A runtime-proven rejection before entering an action handler."""


def validate_dispatch(
    runtime, action, *, command_mode=False, stage=None, source_focus=False, presented_tools=None
):
    """Validate availability and ordinary argument shapes before dispatch.

    Existing specialised decoders retain their own richer validation. This does
    not grant permissions: workspace, process and checker guards remain final.
    """
    from infinicoder.runtime.shell_loop import all_tools

    allowed = (
        set(all_tools(runtime, stage=stage))
        if stage in ("clarification", "planning", "diagnostic")
        else set(all_tools(runtime))
    )
    if runtime.grounded_planning:
        allowed.update(("focus_edit", "release_edit_focus"))
    if source_focus:
        allowed.add("edit_function_body")
    allowed.discard("replan")
    if runtime.adaptive_replanning or command_mode:
        allowed.add("replan")
    if not runtime.allow_commands:
        allowed.discard("command")
    for name, enabled in (
        ("plan", stage in (None, "planning")),
        ("tool_catalog", runtime.compact_tools),
        ("domain_design", stage == "design"),
        ("read_batch", runtime.native_tools and runtime.native_read_batch and stage is None),
        ("source_recall", runtime.source_archive),
        ("problem_decide", runtime.problem_workflow),
        ("problem_probe", runtime.problem_workflow),
        ("profile", runtime._profiling_session is not None),
    ):
        if enabled:
            allowed.add(name)
    if presented_tools is not None:
        # The ordinary catalog is presentation only. Recompute current stage
        # authority and intersect it with the exact request's advertised set.
        from infinicoder.tools.tool_palette import ordinary_available

        allowed = set(ordinary_available(runtime)) | {"tool_catalog"}
        allowed.intersection_update(presented_tools)
    if (
        not isinstance(action, dict)
        or set(action) != {"tool", "arguments"}
        or not isinstance(action.get("tool"), str)
        or not isinstance(action.get("arguments"), dict)
    ):
        raise ActionProtocolError("Expected exactly tool and arguments object")
    name = action["tool"]
    if name not in allowed:
        raise ActionProtocolError("Unavailable tool: " + name)
    try:
        schema_names = [name] + (
            ["problem_probe"] if name == "edit" and runtime.problem_workflow else []
        )
        schema = action_schema(
            schema_names,
            task_memory=runtime.task_memory,
            flow_memory=runtime.flow_memory,
            adaptive_replanning=runtime.adaptive_replanning,
            grounded_planning=runtime.grounded_planning,
            diagnostic_planning=runtime.diagnostic_planning,
            design_probes=bool(runtime._design_tools()),
            observed_edits=runtime.observed_edits,
        )
    except ValueError:
        # Stage-specific plan/design/native handlers validate their own schema.
        return
    try:
        _validate_shape(action, schema)
    except ValueError as error:
        raise ActionProtocolError(str(error)) from error


def _validate_shape(value, schema):
    """The schema subset emitted by action_schema; no optional dependency."""
    import re

    if "anyOf" in schema:
        for branch in schema["anyOf"]:
            try:
                _validate_shape(value, branch)
                return
            except ValueError:
                pass
        raise ValueError("Action arguments do not match an advertised schema")
    types = {
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: type(v) is int,
        "number": lambda v: type(v) in (int, float),
        "boolean": lambda v: type(v) is bool,
        "null": lambda v: v is None,
    }
    kind = schema.get("type")
    kinds = kind if isinstance(kind, list) else [kind] if kind else []
    if kinds and not any(types[t](value) for t in kinds):
        raise ValueError("Wrong argument type")
    if "const" in schema and value != schema["const"]:
        raise ValueError("Wrong tool")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError("Wrong enum argument")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        if set(schema.get("required", ())) - set(value):
            raise ValueError("Missing required argument")
        if schema.get("additionalProperties") is False and set(value) - set(props):
            raise ValueError("Unknown argument")
        for key, item in value.items():
            if key in props:
                _validate_shape(item, props[key])
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get(
            "maxItems", float("inf")
        ):
            raise ValueError("Wrong argument count")
        for item in value:
            _validate_shape(item, schema.get("items", {}))
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get(
            "maxLength", float("inf")
        ):
            raise ValueError("Wrong argument length")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            raise ValueError("Wrong argument pattern")
    if type(value) in (int, float):
        if value < schema.get("minimum", -float("inf")) or value > schema.get(
            "maximum", float("inf")
        ):
            raise ValueError("Argument outside bounds")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            raise ValueError("Argument outside bounds")


def action_schema(
    allowed=None,
    task_memory=False,
    adaptive_replanning=False,
    grounded_planning=False,
    diagnostic_planning=False,
    design_probes=False,
    flow_memory=False,
    observed_edits=False,
):
    """Provider-neutral schema; runtime validation and path checks remain final."""
    string = {"type": "string"}
    strings = {"type": "array", "items": string}
    integer = {"type": "integer", "minimum": 1}

    def obj(properties, required=()):
        return {
            "type": "object",
            "properties": properties,
            "required": list(required),
            "additionalProperties": False,
        }

    schemas = {name: obj({}) for name in ("files", "directories", "seeds", "verify", "finish")}
    if allowed is not None and "leave_implementation" in allowed:
        schemas["leave_implementation"] = obj(
            {"reason": {"type": "string", "minLength": 1, "maxLength": 512}}, ["reason"]
        )
    cell = {"type": "integer", "minimum": 0, "maximum": 9999}
    digest = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    schemas["notebook_cells"] = obj(
        dict(path=string, start=cell, count={"type": "integer", "minimum": 1, "maximum": 32}),
        ["path"],
    )
    schemas["notebook_read"] = obj(
        dict(
            path=string,
            cell=cell,
            start=integer,
            count={"type": "integer", "minimum": 1, "maximum": 120},
        ),
        ["path", "cell"],
    )
    schemas["notebook_edit"] = obj(
        dict(
            path=string,
            cell=cell,
            expected_hash=digest,
            expected_cell_hash=digest,
            old={"type": "string", "minLength": 1, "maxLength": 32768},
            new={"type": "string", "maxLength": 32768},
        ),
        ["path", "cell", "expected_hash", "expected_cell_hash", "old", "new"],
    )
    for name in ("dependencies", "read", "workpack", "use_def"):
        props = {"path": string}
        if name in ("read", "workpack"):
            props.update(start=integer, count={"type": "integer", "minimum": 1, "maximum": 300})
        if name == "use_def":
            props.update(name={"type": ["string", "null"]}, limit=integer)
            if flow_memory:
                props["scope"] = string
        schemas[name] = obj(props, ["path"])
    schemas["edit"] = {
        "anyOf": [
            obj(
                {
                    "path": string,
                    "expected_hash": {"type": ["string", "null"]},
                    "old": string,
                    "new": string,
                },
                ["path", "expected_hash", "old", "new"],
            ),
            obj(
                {
                    "path": string,
                    "expected_hash": string,
                    "start": integer,
                    "end": integer,
                    "new": string,
                },
                ["path", "expected_hash", "start", "end"],
            ),
            obj(
                {"path": string, "expected_hash": {"type": "null"}, "content": string},
                ["path", "expected_hash", "content"],
            ),
        ]
    }
    if observed_edits:
        for branch in schemas["edit"]["anyOf"][:2]:
            branch["required"].remove("expected_hash")
            branch["properties"]["expected_hash"] = {"type": ["string", "null"]}
    if allowed is not None and "reference_compare" in allowed:
        from infinicoder.verification.reference_compare import schema as reference_schema

        schemas["reference_compare"] = reference_schema()
    if allowed is not None and "profile" in allowed:
        schemas["profile"] = obj(
            {"workload": {"type": "string", "minLength": 1, "maxLength": 48}}, ["workload"]
        )
    if allowed is not None and "read_symbol" in allowed:
        schemas["read_symbol"] = obj({"symbol_id": string}, ["symbol_id"])
    if allowed is not None and "edit_symbol" in allowed:
        schemas["edit_symbol"] = obj(
            {"symbol_id": string, "expected_hash": string, "code": string},
            ["symbol_id", "expected_hash", "code"],
        )
    if allowed is not None and "read_function_body" in allowed:
        schemas["read_function_body"] = obj({"symbol_id": string}, ["symbol_id"])
    if allowed is not None and "edit_function_body" in allowed:
        schemas["edit_function_body"] = obj(
            {"symbol_id": string, "expected_hash": string, "body": string},
            ["symbol_id", "expected_hash", "body"],
        )
    if allowed is not None and "problem_decide" in allowed:
        from infinicoder.runtime.problem_decision import schema as decision_schema

        schemas["problem_decide"] = decision_schema()
    if allowed is not None and "problem_probe" in allowed:
        from infinicoder.runtime.problem_workflow import schema as problem_schema

        schemas["problem_probe"] = problem_schema()
        from infinicoder.runtime.problem_workflow import candidate_schema

        for branch in schemas["edit"]["anyOf"]:
            branch["properties"]["problem_context"] = candidate_schema()
    if allowed is not None and "source_recall" in allowed:
        schemas["source_recall"] = obj(
            {
                "id": {"type": "string", "pattern": "^A[0-9]{1,8}$"},
                "offset": {"type": "integer", "minimum": 0},
                "chars": {"type": "integer", "minimum": 1, "maximum": 6000},
            },
            ["id"],
        )
    schemas["command"] = obj(
        {
            "argv": {"type": "array", "minItems": 1, "items": string},
            "timeout": {"type": "number", "exclusiveMinimum": 0},
        },
        ["argv"],
    )
    for name in ("search", "memory_route"):
        schemas[name] = obj({"query": string, "limit": integer}, ["query"])
    schemas["text_search"] = obj(
        {"query": string, "path_globs": strings, "regex": {"type": "boolean"}, "max_hits": integer},
        ["query"],
    )
    for name in ("symbol", "structure"):
        schemas[name] = obj({"symbol_id": string}, ["symbol_id"])
    schemas["relationships"] = obj(
        {
            "symbol_id": string,
            "direction": {"type": "string", "enum": ["both", "incoming", "outgoing"]},
        },
        ["symbol_id"],
    )
    schemas["impact"] = obj(
        {"paths": strings, "depth": {"type": "integer", "minimum": 0}}, ["paths"]
    )
    schemas["tests_for"] = obj({"paths": strings}, ["paths"])
    schemas["memory_expand"] = obj(
        {"module": string, "symbol_id": {"type": ["string", "null"]}, "limit": integer}
    )
    schemas["knowledge"] = obj({"card": string}, ["card"])
    schemas["note"] = obj({"text": {"type": "string", "maxLength": 1500}}, ["text"])
    schemas["ask_user"] = obj({"question": {"type": "string", "minLength": 1}}, ["question"])
    if grounded_planning:
        reason = {"type": "string", "minLength": 1, "maxLength": 1500}
        schemas["focus_edit"] = obj(
            {
                "path": string,
                "symbol": string,
                "expected_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "reason": reason,
                "read_set": {"type": "array", "maxItems": 48, "items": string},
            },
            ["path", "symbol", "expected_hash", "reason"],
        )
        schemas["release_edit_focus"] = obj({"reason": reason}, ["reason"])
    if adaptive_replanning:
        schemas["replan"] = obj(
            {
                "reason": {"type": "string", "minLength": 1, "maxLength": 1500},
                "affected_paths": {"type": "array", "maxItems": 48, "items": string},
            },
            ["reason"],
        )
    if task_memory:
        schemas["function_summary"] = obj({"symbol_id": string}, ["symbol_id"])
        schemas["memory_inspections"] = obj(
            {
                "paths": {"type": "array", "maxItems": 100, "items": string},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            }
        )
        schemas["memory_focus"] = obj(
            {
                "question": {"type": "string", "minLength": 1, "maxLength": 1500},
                "view": {
                    "type": "string",
                    "enum": ["auto", "values", "state", "resources", "interfaces"],
                },
            },
            ["question"],
        )
    if diagnostic_planning:
        from infinicoder.planning.diagnostic_flow import schema

        schemas["diagnose"] = schema()
    if design_probes:
        from infinicoder.planning.design_probes import schemas as design_schemas

        schemas.update(design_schemas())
    if allowed and set(allowed) & {"interface_contract", "interface_forget"}:
        from infinicoder.memory.interface_memory import schemas as interface_schemas

        schemas.update(interface_schemas())
    names = list(schemas) if allowed is None else list(allowed)
    if not names or set(names) - set(schemas):
        raise ValueError("unknown or empty action schema tools")
    return {
        "anyOf": [
            obj(
                {"tool": {"type": "string", "const": name}, "arguments": schemas[name]},
                ["tool", "arguments"],
            )
            for name in names
        ]
    }


def _decode_action(text):
    """Strict-first JSON action decode; fall back to extracting the first
    balanced JSON object from wrapped output (markdown fences, prose).
    Schema validation stays strict: recovery never invents fields."""
    text = final_payload(text)
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        start = text.find("{")
        if start < 0:
            raise ValueError("no JSON object in response")
        depth, end = 0, -1
        in_str, esc = False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end < 0:
            raise ValueError("unbalanced JSON in response")
        obj = json.loads(text[start : end + 1])
    if (
        not isinstance(obj, dict)
        or set(obj) != {"tool", "arguments"}
        or not isinstance(obj["arguments"], dict)
    ):
        raise ValueError("Expected exactly tool and arguments")
    return obj, text.strip()[:1] != "{"
