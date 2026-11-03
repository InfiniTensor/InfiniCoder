"""Literal source views routed by explicit references;no source/command execution.

The routing decision is a hypothesis. Complete lexical bodies are not resolved
bindings,active configurations,tool-read receipts or edit authorization.
"""

import ast
import hashlib
from pathlib import Path
import re
from infinicoder.runtime.worker import _native_mask

SUFFIXES = {".py", ".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hh", ".hpp"}


def request(question, records, *, file_table=None):
    """Prefer explicit paths,then source IDs,then qualified symbol references."""
    paths = []
    names = []
    ids = []
    for raw in re.findall(
        r"(?<![\w/])(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.(?:py|c|cc|cpp|cxx|cu|cuh|h|hh|hpp)\b",
        question,
    ):
        if raw not in paths:
            paths.append(raw)
    file_names = {Path(p).name for p in paths}
    for name in re.findall(r"\b[A-Za-z_]\w*(?:(?:::|\.)[A-Za-z_]\w*)+\b", question):
        if name in file_names:
            continue
        name = name.replace(".", "::")
        if name not in names:
            names.append(name)
    for ref in re.findall(r"\b(?:D\d+|T0)\b", question):
        if ref not in ids:
            ids.append(ref)
    bound = [r for r in records if r.get("id") in ids]
    known = {r["name"].replace(".", "::") for r in records if isinstance(r.get("name"), str)}
    leaves = {n.split("::")[-1] for n in known}
    unresolved = [n for n in names if n not in known and n.split("::")[-1] not in leaves]
    names = [n for n in names if n not in unresolved]
    # A bare identifier is a routing anchor only when current source already
    # advertises it. Prose punctuation cannot create a nonexistent expert scope.
    for leaf in sorted(leaves):
        if re.fullmatch(r"[A-Za-z_]\w*", leaf) and re.search(
            r"(?<![\w:])" + re.escape(leaf) + r"(?![\w:])", question
        ):
            if leaf not in names and not any(n.split("::")[-1] == leaf for n in names):
                names.append(leaf)
    mentioned_files = list(dict.fromkeys(re.findall(r"\bF\d+\b", question)))
    file_table = file_table or {}
    alias_paths = [file_table[k]["path"] for k in mentioned_files if k in file_table]
    scope = paths or alias_paths or list(dict.fromkeys(r["path"] for r in bound))
    if not names:
        names = list(dict.fromkeys(r["name"] for r in bound if isinstance(r.get("name"), str)))
    if not scope and names:
        wanted = {n.split("::")[-1] for n in names}
        scope = list(
            dict.fromkeys(r["path"] for r in records if r.get("name", "").split("::")[-1] in wanted)
        )
    view = (
        "implementation"
        if re.search(
            r"\b(body|implementation|implementations|definition|definitions|performs|updates)\b|实现体|函数体|实现代码",
            question,
            re.I,
        )
        else "declaration"
    )
    return dict(
        paths=scope[:12],
        explicit_paths=paths[:12],
        source_ids=ids[:16],
        file_ids=mentioned_files,
        unresolved_file_ids=[k for k in mentioned_files if k not in file_table],
        symbols=names[:16],
        unresolved_symbols=unresolved[:16],
        view=view,
        selected=bool(scope),
        routing_is_hypothesis=True,
    )


def _python(source, names):
    tree = ast.parse(source)
    out = []
    lines = source.splitlines(keepends=True)

    def visit(nodes, parent=()):
        for node in nodes:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            qualified = "::".join((*parent, node.name))
            if not names or any(
                qualified == n or qualified.endswith("::" + n) or node.name == n.split("::")[-1]
                for n in names
            ):
                first = min([node.lineno, *[v.lineno for v in node.decorator_list]])
                out.append(
                    dict(
                        name=qualified,
                        line=first,
                        end=node.end_lineno,
                        text="".join(lines[first - 1 : node.end_lineno]),
                        parser="python_ast",
                    )
                )
            visit(node.body, (*parent, node.name))

    visit(tree.body)
    return out


def _native(source, names):
    from infinicoder.execution.native_scope import _closing, _header

    mask = _native_mask(source)
    out = []
    seen = set()
    for name in dict.fromkeys(n.split("::")[-1] for n in names):
        if not re.fullmatch(r"[A-Za-z_]\w*", name):
            continue
        for match in list(re.finditer(r"\b" + re.escape(name) + r"\s*\(", mask))[:16]:
            left = mask.rfind("\n", 0, match.start()) + 1
            opening = mask.find("{", match.end(), min(len(mask), match.end() + 900))
            if opening < 0 or ";" in mask[match.end() : opening]:
                continue
            try:
                right = _closing(mask, opening) + 1
                fragment = source[left:right]
                # The edit guard intentionally rejects directives anywhere in
                # a proposed function. A read-only view validates its literal
                # header separately;pragma text is evidence,not edit authority.
                _, _, tokens = _header(source[left:opening] + "{}")
                directives = re.findall(
                    r"(?m)^\s*#\s*(\w+)", _native_mask(fragment, mask_directives=False)
                )
                if any(d not in ("pragma", "line") for d in directives):
                    continue
                if tokens[tokens.index("(") - 1] != name or len(fragment) > 4200:
                    continue
            except (ValueError, IndexError):
                continue
            if (left, right) in seen:
                continue
            seen.add((left, right))
            out.append(
                dict(
                    name=name,
                    line=source.count("\n", 0, left) + 1,
                    end=source.count("\n", 0, right) + 1,
                    text=fragment,
                    parser="position_preserving_lexical_body",
                    qualified_binding_unknown=True,
                    directives=directives,
                )
            )
    return out


def lookup(plan, workspace, question, *, protected_paths=(), file_table=None):
    route = request(question, plan["records"], file_table=file_table)
    protected = {str(Path(p).resolve()) for p in protected_paths}
    versions = {}
    records = []
    total = used = 0
    if not route["selected"] or route["view"] != "implementation":
        return [], {}, dict(route, handled=False)
    names = route["symbols"]
    for relative in route["paths"]:
        if len(records) >= 4:
            break
        try:
            path = workspace.path(relative)
            if str(path.resolve()) in protected or path.suffix.lower() not in SUFFIXES:
                continue
            with path.open("rb") as stream:
                data = stream.read(min(131073, 1048576 - total))
            total += len(data)
            if len(data) > 131072 or len(data) != path.stat().st_size:
                continue
            source = data.decode("utf8")
            digest = hashlib.sha256(data).hexdigest()
            # A path-only question cannot guess a native symbol. Reuse names
            # already observed at that exact path,never enumerate other files.
            local_names = names or [
                r["name"]
                for r in plan["records"]
                if r["path"] == relative and r.get("kind") == "lexical_signature_candidate"
            ]
            bodies = (
                _python(source, local_names)
                if path.suffix == ".py"
                else _native(source, local_names)
            )
        except (OSError, ValueError, UnicodeError, SyntaxError):
            continue
        for body in bodies:
            if len(records) >= 4 or used + len(body["text"]) > 4800:
                break
            # Multiple definitions remain visible and explicitly ambiguous.
            records.append(
                dict(
                    body,
                    path=relative,
                    source_sha256=digest,
                    kind="literal_function_implementation",
                    view="complete_literal_function",
                    origin="explicit_source_question",
                    priority=0,
                    binding_unknown=True,
                    configuration_unknown=True,
                    definition_ambiguous=sum(b["name"] == body["name"] for b in bodies) > 1,
                )
            )
            used += len(body["text"])
            versions[relative] = digest
        if total >= 1048576:
            break
    try:
        current = all(
            hashlib.sha256(workspace.path(p).read_bytes()).hexdigest() == h
            for p, h in versions.items()
        )
    except (ValueError, OSError):
        current = False
    if not current:
        records = []
        versions = {}
    return (
        records,
        versions,
        dict(
            route,
            handled=True,
            records_added=len(records),
            source_snapshot_current=current,
            files_requested=len(route["paths"]),
            bytes_read=total,
            model_calls=0,
            semantic_proof=False,
        ),
    )
