"""Source-versioned syntax spans on existing reads,not a new read or edit grant."""

from infinicoder._resources import package_root

import ast
import hashlib
import json
from pathlib import Path
from infinicoder.analysis.function_body import _layout


def identity():
    root = package_root()
    return hashlib.sha256(
        b"".join(
            (root / name).read_bytes()
            for name in ("memory/source/source_boundaries.py", "analysis/function_body.py")
        )
    ).hexdigest()


def augment_read(workspace, result):
    """Leave text/ranges unchanged;bound metadata before ordinary text budgeting.

    Only long leading function docstrings receive a hint. Whole-source AST analysis
    does not mean the body has been displayed or understood;the read ledger still
    records only actual visible text. No repository code is executed.
    """
    if Path(result.get("path", "")).suffix != ".py":
        return result
    result = dict(result)
    receipt = dict(
        version=1,
        source_sha256=result["sha256"],
        status="unknown",
        ranges=[],
        acceptance_proof=False,
    )
    result["source_boundaries"] = receipt
    try:
        file = workspace.path(result["path"])
        with file.open("rb") as stream:
            data = stream.read(2_000_001)
        if len(data) > 2_000_000:
            receipt["status"] = "source_budget_exceeded"
            return result
        if hashlib.sha256(data).hexdigest() != result["sha256"]:
            receipt["status"] = "source_changed_since_read"
            return result
        source = data.decode("utf-8-sig")
        tree = ast.parse(source)
        if sum(1 for _ in ast.walk(tree)) > 24000:
            receipt["status"] = "analysis_budget_exceeded"
            return result
    except (OSError, UnicodeError, ValueError, SyntaxError, RecursionError):
        receipt["status"] = "syntax_unavailable"
        return result
    read_start = result["start"]
    read_end = read_start + len(result["text"].splitlines()) - 1
    candidates = []

    class Functions(ast.NodeVisitor):
        def __init__(self):
            self.names = []

        def visit_ClassDef(self, node):
            self.names.append(node.name)
            self.generic_visit(node)
            self.names.pop()

        def visit_FunctionDef(self, node):
            self.names.append(node.name)
            try:
                start, end, indent, doc = _layout(node, source)
                first = min([node.lineno, *[d.lineno for d in node.decorator_list]])
                if (
                    doc
                    and doc.end_lineno - doc.lineno + 1 >= 24
                    and start <= end
                    and first <= read_end
                    and read_start <= end
                ):
                    candidates.append(
                        dict(
                            symbol=".".join(self.names)[:200],
                            interface=[first, doc.lineno - 1],
                            docstring=[doc.lineno, doc.end_lineno],
                            implementation=[start, end],
                            body_visible_in_this_read=False,
                        )
                    )
            except ValueError:
                pass
            self.generic_visit(node)
            self.names.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

    Functions().visit(tree)
    receipt["status"] = "syntax_spans" if candidates else "no_long_function_docstring_in_read_range"
    for fact in sorted(candidates, key=lambda v: (v["interface"][0], v["implementation"][0])):
        if len(receipt["ranges"]) >= 3:
            break
        proposed = [*receipt["ranges"], fact]
        if len(json.dumps(proposed, ensure_ascii=False)) > 1200:
            break
        receipt["ranges"] = proposed
    receipt["ranges_truncated"] = len(receipt["ranges"]) < len(candidates)
    if receipt["ranges"]:
        receipt["guidance"] = (
            "Implementation spans exclude signatures and leading docstrings. Request their exact lines;metadata does not establish that omitted code was read,or that only this region needs editing."
        )
        # A finite,explicit read suggestion. It is not executed automatically,
        # and does not overwrite the read ledger's next_unread_line.
        fact = receipt["ranges"][0]
        if fact:
            start, end = fact["implementation"]
            receipt["suggested_read"] = dict(
                path=result["path"], start=start, count=min(100, end - start + 1)
            )
    try:
        with file.open("rb") as stream:
            current = stream.read(2_000_001)
        if hashlib.sha256(current).hexdigest() != result["sha256"]:
            receipt.update(status="source_changed_during_analysis", ranges=[])
            receipt.pop("suggested_read", None)
            receipt.pop("guidance", None)
    except OSError:
        receipt.update(status="source_unavailable_after_analysis", ranges=[])
        receipt.pop("suggested_read", None)
        receipt.pop("guidance", None)
    return result


def finish_visible_read(result):
    """Recompute visibility AFTER the existing source text budget is applied."""
    receipt = result.get("source_boundaries")
    if not receipt or receipt.get("status") != "syntax_spans":
        return result
    start = result["start"]
    end = start + result.get("covered_line_count", len(result["text"].splitlines())) - 1
    for fact in receipt["ranges"]:
        a, b = fact["implementation"]
        fact["body_visible_in_this_read"] = start <= a <= b <= end
    missing = next((v for v in receipt["ranges"] if not v["body_visible_in_this_read"]), None)
    if missing:
        a, b = missing["implementation"]
        receipt["suggested_read"] = dict(path=result["path"], start=a, count=min(100, b - a + 1))
    else:
        receipt.pop("suggested_read", None)
    return result
