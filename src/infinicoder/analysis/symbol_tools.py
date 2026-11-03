"""Versioned Python definition operations; syntax is not acceptance evidence."""

from infinicoder._resources import package_root

import ast
import difflib
import hashlib
import io
from pathlib import Path
import re
import tokenize

DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
GUIDANCE = """Optional symbol tools: read_symbol {"symbol_id":"file.py::Class.method"}
reads one Python definition, including decorators. edit_symbol needs symbol_id,
the current expected_hash, and code containing the complete replacement definition
with its original indentation and the same name/kind. No old-text matching is needed.
Signatures may change;inspect consumers and verify the resulting behavior. Partial
reads are marked incomplete;use ordinary read for omitted lines. Ambiguous or
unsupported definitions require ordinary read/edit/command tools. These tools do
not grant write permission or bypass planning,protected files or final checks.
For a body-only repair,use read_function_body {"symbol_id":"file.py::Class.method"}.
It returns interface,indented body,expected_hash,and omitted-docstring metadata.
Then edit_function_body takes symbol_id,expected_hash,body: replacement statements
with the ORIGINAL indentation,without def/decorators/docstring. It preserves the
function signature,docstring and other definitions. No automatic reindent.
Use this for behavior changes that must preserve an interface;ordinary symbol
editing remains available for explicitly needed interface changes."""


def identity():
    return dict(
        version=2,
        sha256=hashlib.sha256(
            Path(__file__).read_bytes()
            + (package_root() / "analysis/function_body.py").read_bytes()
        ).hexdigest(),
    )


def split_symbol(symbol_id):
    if not isinstance(symbol_id, str) or len(symbol_id) > 1500 or symbol_id.count("::") != 1:
        raise ValueError("Use one symbol_id of the form relative.py::Class.method")
    path, name = symbol_id.split("::")
    if not path or not name or any(not part.isidentifier() for part in name.split(".")):
        raise ValueError("Use the exact qualified Python definition name")
    return path, name


def _definitions(tree):
    result = []

    def visit(node, scope=()):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, DEFINITIONS):
                nested = (*scope, child.name)
                result.append((".".join(nested), child))
                visit(child, nested)
            else:
                visit(child, scope)

    visit(tree)
    return result


def callable_ranges(source):
    """Unique statically declared callables,using the same names as memory.

    Conditional/rebound and nested function declarations are deliberately
    unknown. Returned ranges include decorators and retain original text.
    """
    tree = _parse(source, "<current-source>")
    definitions = _definitions(tree)
    counts = {}
    for name, node in definitions:
        counts[name] = counts.get(name, 0) + 1
    rows = []
    lines = source.splitlines(keepends=True)

    def visit(container, scope=(), owner=None):
        bindings = {}
        for node in container.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bindings[node.name] = bindings.get(node.name, 0) + 1
            else:
                # Any other store/import of this name makes its binding
                # conditional or rebound. Do not descend into a callable.
                for child in ast.walk(node):
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                        bindings[child.id] = bindings.get(child.id, 0) + 1
                    elif isinstance(child, (ast.Import, ast.ImportFrom)):
                        for alias in child.names:
                            key = alias.asname or alias.name.split(".")[0]
                            bindings[key] = bindings.get(key, 0) + 1
        for node in container.body:
            if not isinstance(node, DEFINITIONS):
                continue
            name = ".".join((*scope, node.name))
            if counts[name] != 1 or bindings.get(node.name) != 1:
                continue
            if isinstance(node, ast.ClassDef):
                first = node.body[0].lineno
                header = "".join(lines[node.lineno - 1 : max(node.lineno, first - 1)])
                declared_fields = []
                for member in node.body:
                    if isinstance(member, (ast.Assign, ast.AnnAssign)):
                        try:
                            ast.literal_eval(member.value)
                            fragment = ast.get_source_segment(source, member)
                            if fragment and len(fragment.encode()) <= 256:
                                declared_fields.append(fragment)
                        except (ValueError, TypeError):
                            pass
                visit(
                    node,
                    (*scope, node.name),
                    dict(
                        qualname=name,
                        class_header=header,
                        static_field_declarations=declared_fields,
                        declared_callables=[
                            n.name
                            for n in node.body
                            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                        ],
                        relation_confidence="lexical declarations only;runtime dispatch and dependency closure unknown",
                    ),
                )
            else:
                start = min([node.lineno, *[n.lineno for n in node.decorator_list]])
                rows.append(
                    dict(
                        qualname=name,
                        line=node.lineno,
                        start=start,
                        end=node.end_lineno,
                        kind=type(node).__name__,
                        owner=owner,
                    )
                )

    visit(tree)
    return sorted(rows, key=lambda row: (row["start"], row["end"], row["qualname"]))


def _locate(tree, name):
    definitions = _definitions(tree)
    matches = [node for qualified, node in definitions if qualified == name]
    if len(matches) != 1:
        candidates = [
            dict(name=q, line=n.lineno, kind=type(n).__name__) for q, n in definitions if q == name
        ][:8]
        raise ValueError(
            "Definition must resolve uniquely;matches="
            + str(len(matches))
            + ", candidates="
            + str(candidates)
        )
    return matches[0]


def _parse(source, path):
    try:
        return ast.parse(source, filename=path)
    except (SyntaxError, ValueError, RecursionError) as error:
        raise ValueError("Python syntax unavailable: " + str(error)[:220]) from None


def _load(workspace, symbol_id, write=False):
    path, name = split_symbol(symbol_id)
    file = workspace.path(path, write=write)
    if file.suffix.lower() not in (".py", ".pyi"):
        raise ValueError(
            "Symbol tools currently support Python AST definitions;use ordinary tools for this language"
        )
    if file.stat().st_size > 2_000_000:
        raise ValueError("File too large for symbol tools")
    data = file.read_bytes()
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
        source = data.decode(encoding)
        if source.encode(encoding) != data:
            raise ValueError("Source encoding does not round-trip")
    except (SyntaxError, UnicodeError, LookupError) as error:
        raise ValueError("Unsupported Python source encoding: " + str(error)[:160]) from None
    tree = _parse(source, path)
    try:
        node = _locate(tree, name)
    except RecursionError:
        raise ValueError("Definition nesting exceeds bounded parser capacity") from None
    start = min([node.lineno, *[d.lineno for d in node.decorator_list]])
    end = node.end_lineno
    return path, name, file, data, source, encoding, tree, node, start, end


def _shape_without(tree, selected):
    class Mask(ast.NodeTransformer):
        def visit(self, node):
            return ast.Pass() if node is selected else super().visit(node)

    return ast.dump(Mask().visit(tree), include_attributes=False)


def read_symbol(workspace, symbol_id):
    path, name, file, data, source, encoding, tree, node, start, end = _load(workspace, symbol_id)
    lines = source.splitlines()
    # Match the ordinary read receipt so inspection/source-memory consumers can
    # validate the exact visible lines rather than assuming a whole definition.
    visible = lines[start - 1 : min(end, start + 299)]
    result = dict(
        path=path,
        sha256=hashlib.sha256(data).hexdigest(),
        total_lines=len(lines),
        start=start,
        text="\n".join(visible),
        symbol_id=symbol_id,
        symbol_end=end,
        symbol_kind=type(node).__name__,
        symbol_complete=len(visible) == end - start + 1,
        indentation=lines[node.lineno - 1][
            : len(lines[node.lineno - 1]) - len(lines[node.lineno - 1].lstrip())
        ],
        syntax_frontend="python_ast",
        encoding=encoding,
        acceptance_proof=False,
    )
    from infinicoder.memory.inspection_memory import bounded_read

    result = bounded_read(result)
    if result.get("read_truncated"):
        result["symbol_complete"] = False
    if not result["symbol_complete"]:
        result["symbol_guidance"] = (
            "Definition is incomplete in this observation;read the omitted source before reasoning about its full behavior."
        )
    return result


def edit_symbol(workspace, symbol_id, expected_hash, code):
    if not isinstance(expected_hash, str) or not re.fullmatch("[0-9a-f]{64}", expected_hash):
        raise ValueError("Copy the current complete file expected_hash from a source read")
    if not isinstance(code, str) or not code.strip() or len(code) > 1_000_000:
        raise ValueError("code must contain one bounded complete replacement definition")
    path, name, file, data, source, encoding, tree, node, start, end = _load(
        workspace, symbol_id, write=True
    )
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_hash:
        raise ValueError("File changed;read the current definition before editing")
    lines = source.splitlines(keepends=True)
    old = "".join(lines[start - 1 : end])
    match = re.search(r"\r\n|\n|\r", old)
    eol = match.group() if match else "\n"
    terminal = eol if old.endswith(("\n", "\r")) else ""
    # Do not dedent/indent: doing so would silently change multiline strings.
    replacement = (
        code.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n").replace("\n", eol) + terminal
    )
    updated = "".join(lines[: start - 1]) + replacement + "".join(lines[end:])
    encoded = updated.encode(encoding)
    if len(encoded) > 1_000_000:
        raise ValueError("Edit too large")
    candidate = _parse(updated, path)
    try:
        new_node = _locate(candidate, name)
        if type(new_node) is not type(node):
            raise ValueError("Replacement must keep the definition name and kind")
        if _shape_without(tree, node) != _shape_without(candidate, new_node):
            raise ValueError(
                "Replacement changes statements outside the selected definition;keep its original indentation"
            )
        compile(updated, path, "exec", dont_inherit=True)
    except (SyntaxError, RecursionError) as error:
        raise ValueError("Replacement does not compile: " + str(error)[:220]) from None
    # Recheck the full-file hash in the existing atomic writer after all parser
    # work;the model cannot silently overwrite a concurrent edit.
    result = (
        workspace._atomic_replace(file, encoded, expected_hash, actual)
        if encoded != data
        else dict(path=path, before=actual, after=actual)
    )
    diff = "".join(
        difflib.unified_diff(
            old.splitlines(True),
            replacement.splitlines(True),
            fromfile=path + "::" + name,
            tofile=path + "::" + name,
        )
    )
    result.update(
        symbol_id=symbol_id,
        syntax_valid=True,
        acceptance_proof=False,
        requires_independent_verification=True,
        diff=diff[:2600],
        diff_truncated=len(diff) > 2600,
        changed=encoded != data,
    )
    return result


def read_function_body(workspace, symbol_id):
    from infinicoder.analysis.function_body import read_function_body as read

    return read(workspace, symbol_id)


def edit_function_body(workspace, symbol_id, expected_hash, body):
    from infinicoder.analysis.function_body import edit_function_body as edit

    return edit(workspace, symbol_id, expected_hash, body)
