"""Model-selected function repair with current source leases and caller scope.

Selection is a hypothesis about modification scope,never a correctness claim.
The existing source worker protects declarations and external final acceptance.
"""

import ast
import hashlib
import re


def _python_scope(source, symbol):
    from infinicoder.memory.source.source_scope import packet

    if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", symbol):
        raise ValueError("qualified Python function name required")
    counts = {}
    functions = set()

    def visit(nodes, prefix=""):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + node.name
                counts[name] = counts.get(name, 0) + 1
                if isinstance(node, ast.ClassDef):
                    visit(node.body, name + ".")
                else:
                    functions.add(name)

    visit(ast.parse(source).body)
    prefixes = [".".join(symbol.split(".")[:i]) for i in range(1, len(symbol.split(".")) + 1)]
    if symbol not in functions or any(counts.get(name) != 1 for name in prefixes):
        raise ValueError(
            "unknown or ambiguous Python function;inspect the source or use a broader plan"
        )
    return packet(source, symbol)


def scope_for(focus, workspace):
    path = focus["path"]
    raw = workspace.path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != focus["sources"][path]:
        raise ValueError("focused source changed;inspect its current version")
    if focus["language"] == "python":
        return _python_scope(raw.decode("utf-8-sig"), focus["symbol"])
    from infinicoder.execution.native_scope import packet

    scope = packet(workspace, path, focus["symbol"])
    if scope is None or scope["qualname"] != focus["symbol"]:
        raise ValueError(
            "native function cannot be resolved from current Clang source;use a broader plan"
        )
    if scope["resource_configuration"] != focus.get("resource_configuration") or scope[
        "compile_arguments_sha256"
    ] != focus.get("compile_arguments_sha256"):
        raise ValueError(
            "focused Clang configuration changed;inspect and select the current function again"
        )
    return scope


def select(spec, arguments, workspace, inspected, unit, phase):
    from infinicoder.execution.workspace_tools import Workspace
    from infinicoder.planning.source_plan import grounded_proposed_plan

    required = {"path", "symbol", "expected_hash", "reason"}
    if (
        not isinstance(arguments, dict)
        or not required <= set(arguments)
        or set(arguments) - required - {"read_set"}
    ):
        raise ValueError(
            "focus_edit requires path,symbol,expected_hash,reason and optional read_set"
        )
    if unit.kind == "verify" or unit.worker_path:
        raise ValueError(
            "caller verification/source-worker unit cannot be replaced by a model focus"
        )
    for name in required:
        if not isinstance(arguments[name], str) or not arguments[name].strip():
            raise ValueError("nonempty focus " + name + " required")
    if len(arguments["reason"]) > 1500 or len(arguments["symbol"]) > 500:
        raise ValueError("focus reason/name exceeds bound")
    if not re.fullmatch("[0-9a-f]{64}", arguments["expected_hash"]):
        raise ValueError("current read SHA256 required")
    scoped = Workspace(workspace.root, unit.write_set or workspace.allowed_paths)
    file = scoped.path(arguments["path"], write=True)
    path = file.relative_to(workspace.root).as_posix()
    if not file.is_file() or not 0 < file.stat().st_size <= 2_000_000:
        raise ValueError(
            "focus needs an existing bounded source file;create new modules through a plan"
        )
    raw = file.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != arguments["expected_hash"]:
        raise ValueError("focused source changed;read the current file before selecting it")
    reads = arguments.get("read_set", [])
    if not isinstance(reads, list) or len(reads) > 48 or any(not isinstance(p, str) for p in reads):
        raise ValueError("read_set must contain at most48 concrete source paths")
    # Reuse plan admission solely for source/path evidence. The caller work
    # graph and its acceptance checks are retained rather than replaced.
    declared = dict(
        goal="Repair the selected function: " + arguments["reason"],
        write_set=[path],
        read_set=reads,
        creations=[],
        check_ids=[c.id for c in spec.checks],
        interfaces=["Retain caller contract and protected declaration"],
        dependencies=[],
    )
    _, certificate = grounded_proposed_plan(spec, dict(units=[declared]), workspace, inspected)
    sources = {r["path"]: r["sha256"] for r in certificate["units"][0]["sources"]}
    symbol = arguments["symbol"]
    language = "python" if file.suffix == ".py" else "native"
    if language == "python":
        scope = _python_scope(raw.decode("utf-8-sig"), symbol)
    else:
        from infinicoder.execution.native_scope import packet

        scope = packet(scoped, path, symbol)
        if scope is None:
            raise ValueError(
                "native localization requires one unambiguous Clang function and compile database;use a broader plan"
            )
        sources.update(scope["evidence_source_versions"])
        sources[scope["compile_database_path"]] = scope["compile_database_sha256"]
    focus = dict(
        path=path,
        symbol=scope["qualname"],
        language=language,
        reason=arguments["reason"],
        phase=phase,
        unit_id=unit.id,
        sources=sources,
        source_locations=certificate["units"][0]["sources"],
        source_locations_valid_at="selection;historical after observed edits",
        declaration="signature/decorators protected;release_edit_focus before broader edits",
        origin="model_scope_hypothesis_with_current_source_evidence",
        final_checks=[c.id for c in spec.checks],
    )
    if language == "native":
        focus["external_compile_versions"] = scope["external_compile_versions"]
        focus["resource_configuration"] = scope["resource_configuration"]
        focus["compile_arguments_sha256"] = scope["compile_arguments_sha256"]
    if invalid(focus, workspace):
        raise ValueError("source changed while selecting the focused repair")
    return focus


def invalid(focus, workspace):
    stale = []
    for path, digest in focus["sources"].items():
        try:
            current = hashlib.sha256(workspace.path(path).read_bytes()).hexdigest()
        except (OSError, ValueError):
            current = None
        if current != digest:
            stale.append(path)
    if focus.get("language") == "native":
        try:
            from infinicoder.execution.native_scope import configuration

            resources, arguments = configuration(workspace, focus["path"])
            if resources != focus.get("resource_configuration") or arguments != focus.get(
                "compile_arguments_sha256"
            ):
                stale.append("native_analysis_configuration")
        except (OSError, ValueError, ImportError):
            stale.append("native_analysis_configuration")
    if focus.get("external_compile_versions"):
        from infinicoder.languages.clang_adapter import header_fingerprint

        cache = {}
        for path, digest in focus["external_compile_versions"].items():
            if (header_fingerprint(path, cache) or (None,))[0] != digest:
                stale.append("compile-input:" + path)
    return sorted(stale)


def advance(state, path, digest):
    focus = state.get("edit_focus")
    if focus and path == focus["path"]:
        focus["sources"][path] = digest


def view(focus):
    return {
        key: focus[key]
        for key in (
            "path",
            "symbol",
            "language",
            "reason",
            "phase",
            "unit_id",
            "declaration",
            "origin",
            "final_checks",
        )
    }
