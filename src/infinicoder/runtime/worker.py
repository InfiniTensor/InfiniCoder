"""Bounded source worker; the controller retains path/hash/acceptance authority."""

import ast
import json
from pathlib import Path
import re
import symtable
from infinicoder.models.budget import BudgetExceeded
from infinicoder.runtime.protocol import final_payload


class SourceFormatError(ValueError):
    """A source transport envelope is invalid;not a task semantic failure."""


def requires_initial_clarification(task, state):
    """Enforce explicit user ordering, not guessed ambiguity in ordinary tasks."""
    if state.get("clarifications"):
        return False
    return bool(
        re.search(
            r"\bask_user\s+before\s+editing\b|\b(?:ask|clarify|confirm)\b.{0,35}\bbefore\s+(?:editing|implementing)\b|(?:修改|编辑|实现)前.{0,16}(?:询问|澄清|确认)|先.{0,16}(?:询问|澄清|确认).{0,16}(?:再|后).{0,16}(?:实现|修改|编辑)",
            task,
            re.I,
        )
    )


def _native_mask(source, *, mask_directives=True):
    """Position-preserving lexical mask;not a C++ semantic parser."""
    chars = list(source)
    i = 0

    def blank(a, b):
        for k in range(a, b):
            if chars[k] not in "\r\n":
                chars[k] = " "

    while i < len(source):
        if source.startswith("//", i):
            end = source.find("\n", i + 2)
            while end >= 0 and source[i:end].rstrip("\r").endswith("\\"):
                end = source.find("\n", end + 1)
            end = len(source) if end < 0 else end
            blank(i, end)
            i = end
            continue
        if source.startswith("/*", i):
            end = source.find("*/", i + 2)
            if end < 0:
                raise ValueError("unterminated native comment")
            end += 2
            blank(i, end)
            i = end
            continue
        raw = (
            re.match(r'(?:u8|u|U|L)?R"([^ ()\\\t\r\n]{0,16})\(', source[i:])
            if source[i] in "uULR"
            else None
        )
        if raw:
            tail = ")" + raw.group(1) + '"'
            end = source.find(tail, i + raw.end())
            if end < 0:
                raise ValueError("unterminated native raw string")
            end += len(tail)
            blank(i, end)
            i = end
            continue
        if source[i] in ('"', "'"):
            if (
                source[i] == "'"
                and i + 1 < len(source)
                and source[i + 1].isalnum()
                and re.search(
                    r"(?<!\w)(?:0[xX][0-9A-Fa-f']+|0[bB][01']+|[0-9][0-9.eEpP+\-']*)$", source[:i]
                )
            ):
                i += 1
                continue
            quote = source[i]
            end = i + 1
            while end < len(source):
                if source[end] == "\\":
                    end += 3 if source.startswith("\\\r\n", end) else 2
                    continue
                if source[end] == quote:
                    break
                if source[end] in "\r\n":
                    raise ValueError("unterminated native literal")
                end += 1
            if end >= len(source):
                raise ValueError("unterminated native literal")
            end += 1
            blank(i, end)
            i = end
            continue
        i += 1
    offset = 0
    continued = False
    for line in "".join(chars).splitlines(keepends=True):
        directive = continued or bool(re.match(r"^[ \t]*#", line))
        if directive and mask_directives:
            blank(offset, offset + len(line))
        continued = directive and line.rstrip("\r\n").rstrip().endswith("\\")
        offset += len(line)
    return "".join(chars)


def merge_native_fragment(before, proposal, path):
    """Recover ONE exact-signature native function body;preserve the module.

    Complete module proposals retain normal behavior. A unique literal token
    signature is required;overloads/defaults/macros are never semantically
    inferred. The caller still owns source hashes and independent acceptance.
    """
    if Path(path).suffix.lower() not in {
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
        ".cu",
        ".cuh",
        ".h",
        ".hh",
        ".hpp",
    }:
        return proposal, None
    try:
        masked = _native_mask(proposal)
    except ValueError as error:
        raise ValueError(
            "Native source has an incomplete or unsupported literal/comment;return complete source or use focused action edits"
        ) from error
    opening = masked.find("{")
    if opening < 0:
        return proposal, None
    header = masked[:opening].strip()
    if (
        re.search(r"(?m)^\s*#", proposal[:opening])
        or not header
        or any(c in header for c in "{};")
        or re.search(r"\b(namespace|class|struct|enum|if|else|for|while|switch|catch|do)\b", header)
    ):
        return proposal, None
    first = header.find("(")
    if first < 0 or len(re.findall(r"\b[A-Za-z_]\w*\b", header[:first])) < 2:
        return proposal, None
    depth = 0
    for char in header:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if depth < 0:
            return proposal, None
    if depth:
        return proposal, None

    def close(text, start):
        depth = 0
        for n in range(start, len(text)):
            if text[n] == "{":
                depth += 1
            elif text[n] == "}":
                depth -= 1
                if depth == 0:
                    return n
        raise ValueError("unbalanced native function body;use focused action edits")

    try:
        end = close(masked, opening)
    except ValueError as error:
        raise ValueError(
            "Incomplete native function fragment;return its complete definition"
        ) from error
    if masked[end + 1 :].strip().strip(";").strip():
        return proposal, None
    # A standalone function is now established. Unknown original syntax or
    # ambiguous matches must not turn that function into an entire module.
    try:
        old_mask = _native_mask(before)
    except ValueError as error:
        raise ValueError(
            "Native fragment recovery cannot locate original body;use focused action edits"
        ) from error
    if "{" not in old_mask:
        return proposal, None  # genuinely new file/module
    # A file already containing just one standalone function is a complete
    # module response. Do not turn recovery into a new signature/default-arg
    # restriction on the existing whole-file editing API.
    old_first = old_mask.find("{")
    old_header = old_mask[:old_first].strip()
    old_paren = old_header.find("(")
    if (
        old_paren >= 0
        and len(re.findall(r"\b[A-Za-z_]\w*\b", old_header[:old_paren])) >= 2
        and not any(c in old_header for c in "{};")
        and not re.search(
            r"\b(namespace|class|struct|enum|if|for|while|switch|catch)\b", old_header
        )
        and not re.search(r"(?m)^\s*#", before[:old_first])
    ):
        old_last = close(old_mask, old_first)
        if not old_mask[old_last + 1 :].strip().strip(";").strip():
            return proposal, None
    if any(c in header for c in "=[]"):
        raise ValueError(
            "Unsupported native fragment signature;return the complete module or use focused action edits"
        )
    tokens = re.findall(r"[A-Za-z_]\w*|\d+|::|&&|->|[^\s]", header)

    def token_pattern(token):
        return (
            r"(?<!\w)" + re.escape(token) + r"(?!\w)"
            if re.fullmatch(r"\w+", token)
            else re.escape(token)
        )

    pattern = r"\s*".join(token_pattern(token) for token in tokens) + r"\s*\{"
    # Do not match a suffix of a declaration with omitted return qualifiers,
    # attributes or templates. Out-of-line definitions have a lexical boundary.
    matches = [
        m
        for m in re.finditer(pattern, old_mask)
        if not old_mask[: m.start()].rstrip() or old_mask[: m.start()].rstrip()[-1] in ";{}"
    ]
    if len(matches) != 1:
        raise ValueError(
            "Native function fragment requires exactly one identical signature in current module;preserve parameter names or use focused action edits"
        )
    match = matches[0]
    old_open = match.end() - 1
    old_end = close(old_mask, old_open)
    if re.search(r"(?m)^\s*#", before[old_open : old_end + 1]) or re.search(
        r"(?m)^\s*#", proposal[opening : end + 1]
    ):
        raise ValueError("Native fragment contains preprocessor branches;use focused action edits")
    merged = before[: old_open + 1] + proposal[opening + 1 : end] + before[old_end:]
    return merged, dict(
        kind="unique_literal_native_signature",
        line=before.count("\n", 0, match.start()) + 1,
        end=before.count("\n", 0, old_end) + 1,
        signature=header,
        scope="function body only;lexical identity,not semantic correctness",
    )


class SourceInterfaceMismatch(ValueError):
    """A file proposal cannot use the declaration-preserving source merge."""


def preserve_bindings(before, after, path):
    """Source-worker contract: retain existing public top-level declarations.

    Accept equivalent assignment/import bindings; this is an interface-presence
    guard, not semantic verification. Action mode remains available for renames.
    """
    if not path.endswith(".py"):
        return
    try:
        old = ast.parse(before)
        new = ast.parse(after)
    except SyntaxError:
        return  # frozen independent compiler/checker owns syntax
    required = {
        node.name
        for node in old.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and not node.name.startswith("_")
    }
    bound = set()
    for node in new.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                bound.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
    missing = required - bound
    if missing:
        raise SourceInterfaceMismatch(
            "Source worker dropped required public bindings in "
            + path
            + ": "
            + ", ".join(sorted(missing))
            + ". This proposal was not applied. Declaration-preserving source mode cannot perform a public rename;use ordinary edits only within the original caller contract and write scope."
        )


def complete_module_imports(source, path, domain="general"):
    """Complete only unbound global names of known module namespaces.

    This never executes source, installs packages or fixes local/free variables.
    The externally verified proposal remains the sole acceptance authority.
    """
    if not path.endswith(".py"):
        return source, []
    known = {
        name: "import " + name
        for name in (
            "math",
            "itertools",
            "functools",
            "collections",
            "heapq",
            "bisect",
            "statistics",
        )
    }
    if domain in ("operator", "framework"):
        known.update(
            torch="import torch",
            np="import numpy as np",
            numpy="import numpy",
            triton="import triton",
            tl="import triton.language as tl",
        )
    try:
        tree = ast.parse(source)
        table = symtable.symtable(source, path, "exec")
    except (SyntaxError, ValueError):
        return source, []
    bound = {
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()
    }
    needed = set()
    pending = [table]
    while pending:
        scope = pending.pop()
        pending.extend(scope.get_children())
        for symbol in scope.get_symbols():
            if (
                symbol.is_referenced()
                and symbol.is_global()
                and symbol.get_name() in known
                and symbol.get_name() not in bound
            ):
                needed.add(symbol.get_name())
    if not needed:
        return source, []
    insertion = 0
    for node in tree.body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and insertion == 0
        ):
            insertion = node.end_lineno
        elif isinstance(node, ast.ImportFrom) and node.module == "__future__":
            insertion = node.end_lineno
        else:
            break
    lines = source.splitlines(keepends=True)
    imports = [known[name] for name in sorted(needed)]
    lines[insertion:insertion] = [line + "\n" for line in imports]
    return "".join(lines), imports


SYSTEM = """You are a source-code worker. The controller owns paths, edits,
compilation and verification. Return only the COMPLETE replacement source for
the supplied small file. No shell commands or explanations. Preserve
public interfaces and unrelated behavior. Use the supplied contract, current
source and first failing diagnostics. Resolve repeated failures by changing
the implementation, not by repeating it. A fenced code block is also accepted.
Check EVERY stated invalid, empty and boundary input before loops/allocation.
Local interface files are read-only evidence; replace ONLY the named path.
Diagnostics may describe a rejected proposal; keep the current source as your
starting point. Never emit plans or diagnostic prose as source.
If essential requirements are missing, return a JSON control object with tool
set to "ask_user" and arguments containing one string field "question".
Write an actual question naming the missing requirement; never copy a placeholder.
This is the only permitted control response; it pauses without changing source.
Read user requirement updates before deciding anything is missing. Later user
answers resolve earlier missing requirements; implement the resolved contract.
Only ask a further question when a NEW requirement is unresolved, explaining why
the supplied answer is insufficient. Do not repeat an answered question.
Otherwise return source, not JSON. Never print claims of success.
The independent checker decides acceptance."""

JSON_SYSTEM = """Implement the supplied contract in the named source file.
Return one JSON object with exactly one field "source": the COMPLETE replacement
source as a string. No reasoning, tools, plans or other fields. Preserve public
interfaces. Read-only dependencies are evidence, not files to replace. Fix the
first failing observation. Validate invalid and empty inputs before side effects.
If essential requirements remain unknown, you may instead return a control
object with tool "ask_user" and arguments containing the concrete "question".
Do not claim success; an independent checker will execute the proposal."""

LINES_SYSTEM = """Implement the contract. The current source contains bugs.
Return a JSON object {"source_lines": [one string per source line]} containing
the complete replacement file. Each array item is ONE physical line; do not
encode multiple lines into an item. No explanation, reasoning or tool calls.
Use the read-only interfaces. Respect all valid, invalid and empty inputs.
Fix the observed error, not just formatting. Do not claim success."""

PATCH_SYSTEM = """Implement the contract by one precise source replacement.
Return exactly one JSON object with string fields "old" and "new". "old" must
be a nonempty literal substring appearing exactly once in CURRENT SOURCE.
Replace it with "new", preserving exact source indentation and physical newlines.
No ellipses, line numbers, plans or unchanged copies. Change the actual missing
or defective behavior. For an unimplemented function, replace its stub body with
the working algorithm. The controller retains path/hash authority and checks
syntax, protected interfaces and independent correctness. If evidence is missing,
use an explicitly permitted read-only control;ask_user only for a genuinely
unspecified requirement. Never print claims of success."""

REVIEW_SYSTEM = """Review the CURRENT SOURCE against the CONTRACT and FAILING
OBSERVATION. Identify at most three concrete violations: types, boundaries,
indexing, mathematical scope or state transitions. Quote the faulty expression
and explain why the observed input fails. Do not write replacement code or
claim success. Keep the review under 120 words. The checker, not this review,
determines correctness."""


def response_schema(format="json_source"):
    """Caller may pass this provider-neutral schema to a constrained service."""
    if format == "json_patch":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "source_patch",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "old": {"type": "string", "minLength": 1},
                        "new": {"type": "string"},
                    },
                    "required": ["old", "new"],
                    "additionalProperties": False,
                },
            },
        }
    if format == "json_lines":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "source_lines",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "source_lines": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string", "pattern": r"^[^\r\n]*$"},
                        }
                    },
                    "required": ["source_lines"],
                    "additionalProperties": False,
                },
            },
        }
    if format != "json_source":
        raise ValueError("Unknown constrained worker format")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "source",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"source": {"type": "string", "minLength": 1}},
                "required": ["source"],
                "additionalProperties": False,
            },
        },
    }


def clarification(text, path=None, format="source"):
    """Program-source envelopes may request a safe pause; JSON data is exempt."""
    if (
        format not in ("source", "json_source", "json_lines", "json_patch")
        or path
        and Path(path).suffix in (".json", ".jsonl")
    ):
        return None
    try:
        from infinicoder.runtime.protocol import _decode_action

        value, unused = _decode_action(text)
    except (ValueError, TypeError):
        return None
    if (
        isinstance(value, dict)
        and set(value) == {"tool", "arguments"}
        and value["tool"] == "ask_user"
    ):
        args = value["arguments"]
        if (
            isinstance(args, dict)
            and set(args) == {"question"}
            and isinstance(args["question"], str)
            and args["question"].strip()
        ):
            return args["question"].strip()
    return None


def inspection(
    text,
    task_memory=False,
    adaptive_replanning=False,
    grounded_planning=False,
    diagnostic_planning=False,
    design_probes=False,
):
    """Read-only controls for function-scoped workers; never grants edit/shell."""
    from infinicoder.runtime.protocol import INSPECTION_TOOLS, _decode_action

    try:
        value, _ = _decode_action(text)
    except (ValueError, TypeError):
        return None
    allowed = (
        *INSPECTION_TOOLS,
        *(["function_summary", "memory_focus", "memory_inspections"] if task_memory else []),
        *(["replan"] if adaptive_replanning else []),
        *(["focus_edit", "release_edit_focus"] if grounded_planning else []),
        *(["diagnose"] if diagnostic_planning else []),
        *(["layout_probe", "lifecycle_probe"] if design_probes else []),
    )
    if isinstance(value, dict) and value.get("tool") in allowed:
        return value
    return None


def _literal_source(text, path):
    if not path:
        return False
    suffix = Path(path).suffix
    try:
        if suffix == ".py":
            ast.parse(text, filename=path)
            return True
        if suffix == ".json":
            json.loads(text)
            return True
        if suffix == ".jsonl":
            for line in text.splitlines():
                if line.strip():
                    json.loads(line)
            return True
        if suffix in (".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hh", ".hpp"):
            return "```" not in _native_mask(text)
    except (ValueError, SyntaxError):
        pass
    return suffix in (".md", ".rst", ".txt", ".yaml", ".yml", ".toml")


def _source_envelope_value(text):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError) as error:
        error._infinicoder_source_envelope_rejection = True
        raise


def _source_envelope_error(message):
    error = SourceFormatError(message)
    error._infinicoder_source_envelope_rejection = True
    return error


def source_text(text, path=None, format="source"):
    text = final_payload(text)
    if format == "json_lines":
        value = _source_envelope_value(text)
        if not isinstance(value, dict) or set(value) != {"source_lines"}:
            raise _source_envelope_error("Expected exactly source_lines")
        lines = value["source_lines"]
        if (
            not isinstance(lines, list)
            or not lines
            or any(not isinstance(line, str) or "\n" in line or "\r" in line for line in lines)
        ):
            raise _source_envelope_error("source_lines must contain physical source lines")
        text = "\n".join(lines).strip()
    elif format == "json_source":
        value = _source_envelope_value(text)
        if (
            not isinstance(value, dict)
            or set(value) != {"source"}
            or not isinstance(value["source"], str)
        ):
            raise _source_envelope_error("Expected exactly one string field: source")
        text = value["source"].strip()
    elif format != "source":
        raise ValueError("Unknown worker format")
    if text.startswith("<think>"):
        end = text.find("</think>")
        if end < 0:
            raise SourceFormatError("Incomplete reasoning frame; no source was executed")
        text = text[end + len("</think>") :].strip()
    # Backticks inside a valid Python/JSON program are source data. Trying to
    # unwrap them as a transport fence can reject or replace a correct module.
    complete_source = _literal_source(text, path)
    lines = text.splitlines()
    if (
        not complete_source
        and len(lines) >= 3
        and lines[0].startswith("```")
        and lines[-1].strip() == "```"
    ):
        inner = "\n".join(lines[1:-1])
        if _literal_source(inner, path):
            text = inner
            complete_source = True
    if "```" in text and not complete_source:
        # One complete block is usual. Small models sometimes self-correct with
        # several complete alternatives; accept the final one only when every
        # Python alternative declares the same public interface.
        blocks = re.findall(r"(?m)^```[^\n]*\n(.*?)^```\s*$", text, re.S)
        if (
            len(blocks) > 1
            and text.count("```") == len(blocks) * 2
            and path
            and Path(path).suffix == ".py"
        ):
            interfaces = []
            for block in blocks:
                tree = ast.parse(block, filename=path)
                interfaces.append(
                    {
                        n.name
                        for n in tree.body
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                        and not n.name.startswith("_")
                    }
                )
            if not interfaces[0] or any(names != interfaces[0] for names in interfaces):
                raise SourceFormatError(
                    "Multiple source blocks have different interfaces; return one complete file"
                )
            text = blocks[-1]
        elif len(blocks) == 1 and text.count("```") == 2:
            text = blocks[0]
        else:
            raise SourceFormatError("Expected one complete source fence")
    if not text or "\x00" in text:
        raise SourceFormatError("Worker must return nonempty text source")
    if format == "source" and path and Path(path).suffix == ".py":
        try:
            value = json.loads(text)
        except ValueError:
            value = None
        if isinstance(value, dict) and set(value) == {"tool", "arguments"}:
            raise SourceFormatError("A control request is not replacement Python source")
    if path and Path(path).suffix == ".py":
        # Reject prose/tool requests before they can destroy the last valid source.
        ast.parse(text, filename=path)
    if (
        path
        and Path(path).suffix in (".cpp", ".cc", ".c", ".cu", ".h", ".hpp")
        and text.lstrip().startswith(("{", "["))
    ):
        raise SourceFormatError(
            "Native source worker returned a structured action instead of a translation unit"
        )
    return text + ("\n" if not text.endswith("\n") else "")


def scoped_source_text(text, path, format="source"):
    """Normalize a complete method's outer class indentation before parsing.

    Whole-file workers keep their strict syntax rule. The caller must still
    restore through source_scope's protected signature and hash-guarded merge.
    """
    import textwrap

    value = textwrap.dedent(source_text(text, format=format))
    ast.parse(value, filename=path)
    return value


def patch_source(text, current, path):
    value = json.loads(final_payload(text))
    if (
        not isinstance(value, dict)
        or set(value) != {"old", "new"}
        or not all(isinstance(v, str) for v in value.values())
    ):
        raise ValueError("Expected exactly two patch strings: old and new")
    old, new = value["old"], value["new"]
    if not old or old == new:
        raise ValueError("Patch must replace a nonempty unique substring with changed source")
    first = current.find(old)
    if first < 0 or current.find(old, first + 1) >= 0:
        raise ValueError(
            "Patch old substring is absent or ambiguous;inspect current source and quote one exact unique range"
        )
    proposal = current[:first] + new + current[first + len(old) :]
    if not proposal or "\x00" in proposal:
        raise ValueError("patch must retain nonempty text source")
    if path and Path(path).suffix == ".py":
        ast.parse(proposal, filename=path)
    return proposal


def _python_interface_slice(text, names, max_chars):
    """Exact static declaration slices, not a claim about module side effects."""
    tree = ast.parse(text)
    definitions = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    definitions[target.id] = node
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                definitions[alias.asname or alias.name.split(".")[0]] = node
    if not names or "*" in names or any(name not in definitions for name in names):
        raise BudgetExceeded(
            "Large required interface has unresolved exports; use action-mode source inspection"
        )
    selected, pending = {}, [definitions[name] for name in names]
    while pending:
        node = pending.pop()
        if node.lineno in selected:
            continue
        selected[node.lineno] = node
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Name)
                and isinstance(child.ctx, ast.Load)
                and child.id in definitions
            ):
                pending.append(definitions[child.id])
        if len(selected) > 100:
            raise BudgetExceeded("Required static interface closure exceeds 100 declarations")
    lines = text.splitlines()
    parts = [
        "# STATIC INTERFACE SLICE: exact declarations; other module code is omitted.\n"
        "# Dynamic exports and module side effects are not represented.\n"
    ]
    for _, node in sorted(selected.items()):
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        parts.append(
            f"# ORIGINAL LINES {start}-{node.end_lineno}\n"
            + "\n".join(lines[start - 1 : node.end_lineno])
            + "\n"
        )
    result = "\n".join(parts)
    if len(result) > max_chars:
        raise BudgetExceeded("Required static interface closure exceeds packet bound")
    return result


def interfaces(workspace, path, source, max_chars=7000):
    """Bounded direct local interface evidence; never read external oracle paths."""
    candidates = []
    requested = {}
    if Path(path).suffix in (".cpp", ".cc", ".c", ".cu", ".h", ".hpp"):
        for name in re.findall(r'^\s*#include\s*"([^"]+)"', source, re.M):
            # Quoted headers can be relative to the translation unit or to the
            # repository include root; inspect only paths that really exist.
            candidates += [(Path(path).parent / name).as_posix(), Path(name).as_posix()]
    elif Path(path).suffix == ".py":
        try:
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.level:
                        package_parts = list(Path(path).parent.parts)
                        if node.level > len(package_parts):
                            raise BudgetExceeded(
                                "Required relative interface leaves declared package"
                            )
                        parent = Path(*package_parts[: len(package_parts) - node.level + 1])
                    else:
                        parent = Path()
                    candidate = (parent / Path(*node.module.split("."))).as_posix() + ".py"
                    candidates.append(candidate)
                    requested.setdefault(candidate, set()).update(
                        alias.name for alias in node.names
                    )
        except SyntaxError:
            pass
    result, used = [], 0
    for candidate in dict.fromkeys(candidates):
        try:
            p = workspace.path(candidate)
            if p.is_file():
                if p.stat().st_size > 1_000_000:
                    raise BudgetExceeded("Required interface exceeds safe read bound: " + candidate)
                data = p.read_bytes()
                text = data.decode("utf-8-sig")
                if "\x00" in text:
                    raise BudgetExceeded("Required interface is not text: " + candidate)
                import hashlib

                total_lines = len(text.splitlines())
                sliced = len(text) > max_chars - used
                if sliced:
                    if p.suffix != ".py":
                        raise BudgetExceeded(
                            "Required direct interface exceeds packet bound: " + candidate
                        )
                    try:
                        text = _python_interface_slice(
                            text, requested.get(candidate, set()), max_chars - used
                        )
                    except SyntaxError:
                        raise BudgetExceeded(
                            "Required large interface cannot be parsed: " + candidate
                        ) from None
                value = {
                    "path": candidate,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "total_lines": total_lines,
                    "start": 1,
                    "text": text,
                }
                if sliced:
                    value.update(
                        view="static_interface_slice",
                        requested_symbols=sorted(requested[candidate]),
                        limitation="Unselected module code, dynamic exports and side effects are omitted; verify adaptations independently.",
                    )
                result.append(value)
                used += len(value["text"])
        except BudgetExceeded:
            raise
        except (ValueError, OSError, UnicodeError):
            pass
    return result
