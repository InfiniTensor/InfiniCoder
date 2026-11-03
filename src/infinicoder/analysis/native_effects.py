"""Bounded Clang body observations,never purity,dispatch or ownership proofs."""

import hashlib
import importlib.metadata
import json
from pathlib import Path


FUNCTIONS = {"FUNCTION_DECL", "CXX_METHOD", "CONSTRUCTOR", "DESTRUCTOR"}


def identity():
    try:
        version = importlib.metadata.version("libclang")
    except importlib.metadata.PackageNotFoundError:
        version = "unavailable"
    return hashlib.sha256(Path(__file__).read_bytes() + version.encode()).hexdigest()


def function_facts(cursor, source, filename):
    """Use the already parsed TU;one bounded traversal per function body."""
    raw = source.encode("utf-8")
    filename = Path(filename).resolve()
    facts = dict(
        parameters=[],
        returns=[],
        transformations=[],
        calls=[],
        writes=[],
        resources=[],
        omitted={},
        variadic=cursor.type.is_function_variadic(),
        analysis_kind="clang_body_syntax",
        limitations=[
            "Body syntax only;branches,exceptions and unevaluated expressions may prevent execution.",
            "Pointer/reference aliases,implicit destructors,ownership and mutation by callees remain unknown.",
            "Header fingerprints are checked after parsing;concurrent edits during parsing are not an atomic snapshot.",
        ],
    )
    start, end = cursor.extent.start, cursor.extent.end
    facts["source_extent"] = dict(
        bytes=[start.offset, end.offset],
        lines=[start.line, end.line],
        source_sha256=hashlib.sha256(raw).hexdigest(),
        complete_definition=bool(
            cursor.is_definition()
            and start.file
            and end.file
            and Path(start.file.name).resolve() == filename
            and Path(end.file.name).resolve() == filename
            and 0 <= start.offset < end.offset <= len(raw)
        ),
    )

    def expression(node):
        start, end = node.extent.start, node.extent.end
        if not start.file or not end.file:
            return None
        if Path(start.file.name).resolve() != filename or Path(end.file.name).resolve() != filename:
            return None
        if not 0 <= start.offset < end.offset <= len(raw):
            return None
        return raw[start.offset : end.offset].decode("utf-8", errors="strict")

    def add(key, value):
        if len(facts[key]) >= 64 or len(json.dumps(value, ensure_ascii=False)) > 2400:
            facts["omitted"][key] = facts["omitted"].get(key, 0) + 1
        else:
            facts[key].append(value)

    def at(node):
        return dict(line=node.location.line, column=node.location.column)

    def default(parameter):
        # Clang exposes declaration default expressions,including defaults
        # inherited from a header. Tokens retain syntax,not an evaluated value.
        children = [c for c in parameter.get_children() if c.kind.is_expression()]
        if not children:
            return None
        tokens = [t.spelling for t in children[-1].get_tokens()]
        if not tokens or len(json.dumps(tokens)) > 1600:
            return None
        return tokens

    if cursor.kind.name == "CXX_METHOD" and not cursor.is_static_method():
        add(
            "parameters",
            dict(
                name="this",
                kind="implicit_receiver",
                annotation=cursor.semantic_parent.type.spelling,
                alias="unknown",
            ),
        )
    for parameter in cursor.get_arguments():
        add(
            "parameters",
            dict(
                name=parameter.spelling,
                kind="positional",
                annotation=parameter.type.spelling,
                default_tokens=default(parameter),
                **at(parameter),
            ),
        )
    if cursor.kind.name in ("CONSTRUCTOR", "DESTRUCTOR"):
        facts["limitations"].append(
            "Constructor initializers and implicit destruction outside the body are not summarized."
        )

    def write(target, node, operation):
        nested = list(target.get_children())
        kind = target.kind.name
        while kind in ("UNEXPOSED_EXPR", "PAREN_EXPR") and len(nested) == 1:
            target = nested[0]
            nested = list(target.get_children())
            kind = target.kind.name
        value = expression(target)
        category = None
        if kind in ("MEMBER_REF_EXPR", "ARRAY_SUBSCRIPT_EXPR"):
            category = "member_or_subscript"
        elif kind == "UNARY_OPERATOR" and value and value.lstrip().startswith("*"):
            category = "pointee"
        elif kind == "DECL_REF_EXPR" and target.referenced:
            declaration = target.referenced
            if declaration.type.kind.name in ("LVALUEREFERENCE", "RVALUEREFERENCE"):
                category = "reference"
            elif declaration.kind.name == "VAR_DECL" and (
                declaration.semantic_parent.kind.name in ("TRANSLATION_UNIT", "NAMESPACE")
                or declaration.storage_class.name == "STATIC"
            ):
                category = "global_or_static"
        if category and value is not None:
            add(
                "writes",
                dict(
                    target=value, operation=operation, storage=category, alias="unknown", **at(node)
                ),
            )

    remaining = 12000

    def visit(node):
        nonlocal remaining
        if remaining <= 0:
            facts["omitted"]["ast_frontier"] = facts["omitted"].get("ast_frontier", 0) + 1
            return
        remaining -= 1
        kind = node.kind.name
        if kind in FUNCTIONS | {
            "LAMBDA_EXPR",
            "CLASS_DECL",
            "STRUCT_DECL",
            "FUNCTION_TEMPLATE",
            "CLASS_TEMPLATE",
        }:
            return
        children = list(node.get_children())
        text = expression(node)
        if kind == "RETURN_STMT":
            value = expression(children[0]) if children else None
            if not children or value is not None:
                add("returns", dict(expression=value, **at(node)))
            else:
                facts["omitted"]["returns"] = facts["omitted"].get("returns", 0) + 1
        elif kind == "VAR_DECL":
            initializers = [c for c in children if c.kind.is_expression()]
            if initializers:
                value = expression(initializers[-1])
                if value is not None:
                    add(
                        "transformations",
                        dict(
                            targets=[node.spelling],
                            expression=value,
                            operation="initialize",
                            **at(node),
                        ),
                    )
            if node.type.get_canonical().kind.name == "RECORD":
                add(
                    "resources",
                    dict(
                        kind="local_record_lifetime_candidate",
                        binding=node.spelling,
                        annotation=node.type.spelling,
                        ownership="unknown",
                        guaranteed=False,
                        **at(node),
                    ),
                )
        elif kind in ("BINARY_OPERATOR", "COMPOUND_ASSIGNMENT_OPERATOR") and len(children) == 2:
            left, right = children
            lhs, rhs = expression(left), expression(right)
            if (
                lhs is not None
                and rhs is not None
                and left.extent.end.offset <= right.extent.start.offset
            ):
                operators = [
                    t.spelling
                    for t in node.get_tokens()
                    if t.kind.name != "COMMENT"
                    and left.extent.end.offset <= t.extent.start.offset
                    and t.extent.end.offset <= right.extent.start.offset
                ]
                operator = operators[0] if len(operators) == 1 else None
                if operator in ("=", "+=", "-=", "*=", "/=", "%=", "<<=", ">>=", "&=", "|=", "^="):
                    add(
                        "transformations",
                        dict(
                            targets=[lhs],
                            expression=rhs,
                            operation=operator,
                            reads_previous_value=operator != "=",
                            **at(node),
                        ),
                    )
                    write(left, node, operator)
                elif operator is None:
                    facts["omitted"]["unclassified_operator"] = (
                        facts["omitted"].get("unclassified_operator", 0) + 1
                    )
        elif kind == "UNARY_OPERATOR" and children and text:
            operator = next(
                (
                    value
                    for value in ("++", "--")
                    if text.strip().startswith(value) or text.strip().endswith(value)
                ),
                None,
            )
            target = expression(children[0])
            if operator and target is not None:
                add(
                    "transformations",
                    dict(
                        targets=[target],
                        expression=text,
                        operation=operator,
                        reads_previous_value=True,
                        **at(node),
                    ),
                )
                write(children[0], node, operator)
        elif kind == "CALL_EXPR":
            target = node.referenced
            arguments = []
            for arg in node.get_arguments():
                arguments.append(expression(arg))
            call = dict(
                target=target.spelling if target else "",
                arguments=arguments,
                expression=text,
                **at(node),
            )
            if target and target.kind.name in FUNCTIONS:
                params = list(target.get_arguments())
                inputs = []
                for i, arg in enumerate(arguments):
                    tokens = default(params[i]) if i < len(params) and arg is None else None
                    inputs.append(
                        dict(
                            parameter_index=i,
                            expression=arg,
                            default_tokens=tokens,
                            origin="argument"
                            if arg is not None
                            else "declaration_default_syntax"
                            if tokens
                            else "unknown",
                        )
                    )
                call.update(argument_bindings=inputs, variadic=target.type.is_function_variadic())
            add("calls", call)
            if target and target.spelling in (
                "close",
                "release",
                "free",
                "unlock",
                "shutdown",
                "dispose",
            ):
                add(
                    "resources",
                    dict(
                        kind="named_release_candidate",
                        expression=text,
                        ownership="unknown",
                        guaranteed=False,
                        **at(node),
                    ),
                )
        elif kind in ("CXX_NEW_EXPR", "CXX_DELETE_EXPR"):
            if text is not None:
                add(
                    "resources",
                    dict(
                        kind="explicit_new" if kind == "CXX_NEW_EXPR" else "explicit_delete",
                        expression=text,
                        ownership="unknown",
                        guaranteed=False,
                        **at(node),
                    ),
                )
        for child in children:
            visit(child)

    for body in cursor.get_children():
        if body.kind.name == "COMPOUND_STMT":
            visit(body)
    return facts


def current_native_inputs(index, symbol, file, fingerprint_cache=None):
    """Refuse stale or incompletely versioned parser inputs,without reparsing."""

    def reject(reason, stale=False):
        return None, dict(status="stale" if stale else "unknown", reason=reason)

    build = json.loads(file["build"]) if file["build"] else {}
    if not build.get("semantic") or build.get("mode") != "clang-ast":
        return reject("semantic native analysis unavailable")
    if build.get("native_effects_identity") != identity():
        return reject("native effect analyzer changed;refresh", True)
    database = build.get("compile_database_path")
    expected = build.get("compile_database_sha256")
    try:
        if not database or hashlib.sha256(Path(database).read_bytes()).hexdigest() != expected:
            return reject("compile database changed;refresh", True)
    except OSError:
        return reject("compile database missing;refresh", True)
    state = index.db.execute(
        "SELECT revision,complete FROM native_input_state WHERE path=?", (symbol["path"],)
    ).fetchone()
    if (
        not state
        or not state["complete"]
        or state["revision"] != build.get("native_dependency_revision")
    ):
        return reject("native parser inputs incompletely versioned;refresh")
    from infinicoder.languages.clang_adapter import header_fingerprint

    cache = {} if fingerprint_cache is None else fingerprint_cache
    for filename, expected in index.db.execute(
        "SELECT file,hash FROM native_inputs WHERE path=?", (symbol["path"],)
    ):
        value = header_fingerprint(filename, cache)
        if value is None or value[0] != expected:
            return reject("native header changed or disappeared;refresh", True)
    return build, None


def current_native_facts(index, symbol, file, fingerprint_cache=None):
    build, problem = current_native_inputs(index, symbol, file, fingerprint_cache)
    if problem:
        return None, problem
    found = build.get("native_function_facts", {}).get("functions", {}).get(symbol["id"])
    if not found:
        return None, dict(
            status="unknown", reason="native function missing,ambiguous or outside bounded analysis"
        )
    return dict(
        found,
        native_dependency_revision=build["native_dependency_revision"],
        compile_database_sha256=build["compile_database_sha256"],
    ), None


def links(memory, symbol, facts):
    """Join already resolved per-call graph rows to independently live callees."""
    index = memory.index
    build = json.loads(
        index.db.execute("SELECT build FROM files WHERE path=?", (symbol["path"],)).fetchone()[0]
    )
    bindings = build.get("native_bindings", {}).get("refs", [])
    edges = list(
        index.db.execute("SELECT * FROM refs WHERE path=? ORDER BY rowid", (symbol["path"],))
    )
    pairs = list(zip(bindings, edges)) if len(bindings) == len(edges) else []
    output = []
    for call in facts["calls"]:
        link = dict(
            call_line=call["line"], call_column=call["column"], target_expression=call["target"]
        )
        matches = [
            edge
            for binding, edge in pairs
            if (binding["src"], binding["line"], binding["column"])
            == (symbol["id"], call["line"], call["column"])
            and (edge["src"], edge["line"], edge["target"])
            == (binding["src"], binding["line"], binding["target"])
        ]
        if (
            len(matches) != 1
            or not matches[0]["dst"]
            or matches[0]["confidence"] != "clang_static_candidate"
        ):
            link.update(
                status="unknown", reason="indirect,virtual,missing or ambiguous static call"
            )
        else:
            target = matches[0]["dst"]
            callee = memory.summary(target, include_links=False)
            if callee["status"] != "syntax_observations":
                link.update(
                    status="unknown", callee=target, reason="callee facts stale or unavailable"
                )
            elif call.get("variadic") or callee.get("variadic"):
                link.update(status="unknown", callee=target, reason="variadic arguments")
            else:
                parameters = [p for p in callee["parameters"] if p["kind"] == "positional"]
                inputs = {}
                unknown = False
                if len(call.get("argument_bindings", [])) != len(parameters) or callee.get(
                    "omitted", {}
                ).get("parameters"):
                    unknown = True
                for argument in call.get("argument_bindings", []):
                    i = argument["parameter_index"]
                    if i >= len(parameters) or argument["origin"] == "unknown":
                        unknown = True
                        continue
                    name = parameters[i]["name"] or "#" + str(i)
                    inputs[name] = dict(
                        expression=argument["expression"],
                        default_tokens=argument["default_tokens"],
                        origin=argument["origin"],
                    )
                if unknown:
                    link.update(
                        status="unknown", callee=target, reason="argument/default facts incomplete"
                    )
                else:
                    link.update(
                        status="static_candidate",
                        callee=target,
                        callee_path=callee["path"],
                        callee_line=callee["line"],
                        callee_source_sha256=callee["source_sha256"],
                        callee_native_dependency_revision=callee["native_dependency_revision"],
                        inputs=inputs,
                        return_expressions=callee["returns"],
                        may_write=callee["writes"],
                        resource_regions=callee["resources"],
                        callee_omissions=callee.get("omitted", {}),
                        receiver="unknown",
                        limitation="Caller expressions and declaration default tokens only;no evaluated values,alias,ownership or transitive effect proof.",
                    )
        output.append(link)
    return output
