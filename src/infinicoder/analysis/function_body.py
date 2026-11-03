"""Versioned Python body views/edits;syntax and scope are not correctness proof."""

import ast
import copy
import difflib
import hashlib
from pathlib import Path
import re
from infinicoder.analysis.symbol_tools import _load, _locate, _parse


def identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _doc(node):
    first = node.body[0]
    return (
        first
        if isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
        else None
    )


def _layout(node, source):
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ValueError("Function-body operations require a Python function,not a class")
    doc = _doc(node)
    statements = node.body[1:] if doc else node.body
    first = statements[0] if statements else None
    # A semicolon/single-line suite cannot be replaced as whole source lines
    # while preserving its header/docstring. Refuse instead of guessing offsets.
    first_line = (
        min([first.lineno, *[v.lineno for v in getattr(first, "decorator_list", [])]])
        if first
        else node.end_lineno + 1
    )
    if node.body[0].lineno <= node.lineno or (doc and first and first_line <= doc.end_lineno):
        raise ValueError(
            "Inline header/docstring suite needs ordinary symbol editing;no body edit performed"
        )
    lines = source.splitlines(keepends=True)
    anchor = first or doc
    line = lines[anchor.lineno - 1]
    indent = line[: len(line) - len(line.lstrip(" \t"))]
    if not indent:
        raise ValueError("Function body indentation unavailable")
    return first_line, node.end_lineno, indent, doc


def _shell(tree, selected):
    class Mask(ast.NodeTransformer):
        def visit(self, node):
            if node is selected:
                result = copy.deepcopy(node)
                doc = _doc(node)
                result.body = ([copy.deepcopy(doc)] if doc else []) + [ast.Pass()]
                return result
            return super().visit(node)

    return ast.dump(Mask().visit(tree), include_attributes=False)


def read_function_body(workspace, symbol_id):
    path, name, file, data, source, encoding, tree, node, definition_start, end = _load(
        workspace, symbol_id
    )
    start, end, indent, doc = _layout(node, source)
    lines = source.splitlines()
    header_end = node.body[0].lineno - 1
    header = "\n".join(lines[definition_start - 1 : header_end])
    header_complete = len(header) <= 1200
    result = dict(
        path=path,
        sha256=hashlib.sha256(data).hexdigest(),
        expected_hash=hashlib.sha256(data).hexdigest(),
        total_lines=len(lines),
        start=start,
        text="\n".join(lines[start - 1 : min(end, start + 299)]),
        symbol_id=symbol_id,
        symbol_kind=type(node).__name__,
        symbol_complete=False,
        body_end=end,
        body_complete=end - start + 1 <= 300,
        indentation=indent,
        interface=dict(
            start=definition_start, end=header_end, text=header[:1200], complete=header_complete
        ),
        docstring_omitted=bool(doc),
        docstring_lines=[doc.lineno, doc.end_lineno] if doc else None,
        body_view_identity=identity(),
        encoding=encoding,
        syntax_frontend="python_ast",
        acceptance_proof=False,
        scope="Only text at start is contiguous inspected body source. Interface is a separate source span;omitted docstring is not inspected. edit_function_body preserves header,docstring,and other definitions.",
    )
    from infinicoder.memory.inspection_memory import bounded_read

    result = bounded_read(result)
    if result.get("read_truncated"):
        result["body_complete"] = False
    return result


def edit_function_body(workspace, symbol_id, expected_hash, body):
    if not isinstance(expected_hash, str) or not re.fullmatch("[0-9a-f]{64}", expected_hash):
        raise ValueError("Copy expected_hash from a current source read")
    if not isinstance(body, str) or not body.strip() or len(body) > 1_000_000:
        raise ValueError(
            "body must contain bounded replacement statements with their original indentation"
        )
    path, name, file, data, source, encoding, tree, node, definition_start, end = _load(
        workspace, symbol_id, write=True
    )
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_hash:
        raise ValueError("File changed;read its current function body")
    start, end, indent, doc = _layout(node, source)
    lines = source.splitlines(keepends=True)
    old = "".join(lines[start - 1 : end])
    match = re.search(r"\r\n|\n|\r", source)
    eol = match.group() if match else "\n"
    prefix = "".join(lines[: start - 1])
    suffix = "".join(lines[end:])
    # No implicit reindent: it would alter multiline string contents.
    replacement = body.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n").replace("\n", eol)
    if old.endswith(("\n", "\r")) or suffix:
        replacement += eol
    gap = eol if not old and prefix and not prefix.endswith(("\n", "\r")) else ""
    updated = prefix + gap + replacement + suffix
    encoded = updated.encode(encoding)
    if len(encoded) > 1_000_000:
        raise ValueError("Edit too large")
    candidate = _parse(updated, path)
    new_node = _locate(candidate, name)
    # Compare the complete module with only the selected executable body masked.
    # Existing docstring,signature,decorators,annotations and all other nodes stay.
    if _shell(tree, node) != _shell(candidate, new_node):
        raise ValueError(
            "Body edit changed the interface,docstring,or statements outside the selected function"
        )
    try:
        compile(updated, path, "exec", dont_inherit=True)
    except (SyntaxError, RecursionError) as error:
        raise ValueError("Replacement does not compile: " + str(error)[:220]) from None
    result = (
        workspace._atomic_replace(file, encoded, expected_hash, actual)
        if encoded != data
        else dict(path=path, before=actual, after=actual)
    )
    diff = "".join(
        difflib.unified_diff(
            old.splitlines(True),
            (gap + replacement).splitlines(True),
            fromfile=path + "::" + name + " body",
            tofile=path + "::" + name + " body",
        )
    )
    result.update(
        symbol_id=symbol_id,
        body_only=True,
        interface_preserved=True,
        syntax_valid=True,
        acceptance_proof=False,
        requires_independent_verification=True,
        changed=encoded != data,
        diff=diff[:2600],
        diff_truncated=len(diff) > 2600,
        body_view_identity=identity(),
    )
    return result
