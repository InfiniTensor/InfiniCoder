"""Lossless delivery shaping using the current module,never executing code."""

import ast
import hashlib
from infinicoder.analysis.function_body import _layout, _load, _shell
from infinicoder.analysis.symbol_tools import _locate, _parse


def strict_json(text):
    import json

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate JSON delivery key: " + str(key)[:80])
            value[key] = item
        return value

    return json.loads(text, object_pairs_hook=unique)


def normalize_body(workspace, symbol_id, expected_hash, text):
    """Accept statements or exactly one matching complete definition.

    The temporary if wrapper is solely for parsing indented delivery text.
    Nothing is dedented,unparsed,imported or executed. Complete definitions
    are also spliced into the original module in memory for a full protected
    interface/perimeter comparison before extracting executable source lines.
    """
    if not isinstance(text, str) or not text.strip() or len(text) > 1_000_000:
        raise ValueError("A bounded nonempty source delivery required")
    path, name, file, raw, source, encoding, tree, node, start, end = _load(workspace, symbol_id)
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError("Delivery source version is stale")
    metadata = dict(
        original_delivery_sha256=hashlib.sha256(text.encode()).hexdigest(),
        form="body_statements",
        source_mutated=False,
        acceptance_proof=False,
    )
    try:
        try:
            payload = ast.parse(text)
            roots = payload.body
        except IndentationError:
            payload = ast.parse("if True:\n" + text)
            roots = (
                payload.body[0].body
                if len(payload.body) == 1 and isinstance(payload.body[0], ast.If)
                else payload.body
            )
    except (SyntaxError, ValueError, RecursionError):
        # Preserve invalid ordinary body text;the original scope/compile
        # operation reports it. Never fish a code fragment out of a bad block.
        return text, metadata
    definitions = [n for n in roots if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    intended = [
        n
        for n in definitions
        if n.col_offset == node.col_offset or n.name == node.name and len(roots) == 1
    ]
    if not intended:
        return text, metadata
    if len(roots) != 1 or len(intended) != 1:
        raise ValueError(
            "Complete definition delivery cannot contain sibling statements or helpers"
        )
    offered = intended[0]
    if offered.col_offset != node.col_offset:
        raise ValueError(
            "Complete definition indentation differs from the actual target;no implicit reindent"
        )
    lines = source.splitlines(keepends=True)
    replacement = text
    if not replacement.endswith(("\n", "\r")):
        replacement += "\n"
    candidate = "".join(lines[: start - 1]) + replacement + "".join(lines[end:])
    if len(candidate.encode(encoding)) > 1_000_000:
        raise ValueError("Definition delivery exceeds source bound")
    proposed = _parse(candidate, path)
    selected = _locate(proposed, name)
    if _shell(tree, node) != _shell(proposed, selected):
        raise ValueError(
            "Complete definition changed the interface,decorators,docstring or protected module source"
        )
    body_start, body_end, indent, doc = _layout(selected, candidate)
    body = "".join(candidate.splitlines(keepends=True)[body_start - 1 : body_end])
    if not body.strip():
        raise ValueError("Complete definition contains no executable body delivery")
    metadata.update(
        form="complete_definition",
        normalization="original_module_splice_validate_extract_body",
        interface_preserved=True,
        signature_and_docstring_bytes="Retained from the actual original file by edit_function_body",
    )
    return body, metadata
