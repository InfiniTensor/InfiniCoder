"""Resolve reading roots from public task text, without inventing edit scope."""

from pathlib import PurePosixPath
import re

from infinicoder.analysis.qualified_symbols import anchors

PATH = re.compile(
    r"(?<![\w./\\-])((?:/?[\w@+.-]+/)*[\w@+.-]+\.(?:py|c|cc|cpp|cxx|h|hh|hpp|cu|cuh|rs|js|ts|tsx|json|toml|yaml|yml|sh))(?::(\d+)(?:-(\d+))?)?(?![\w./-])"
)
NATIVE_NAME = re.compile(r"\b[A-Za-z_]\w*(?:::[A-Za-z_]\w*)+\b")


def relative_public_path(value, workspace_aliases=()):
    """Absolute paths require a caller-declared container workspace mapping."""
    for alias in workspace_aliases:
        if value.startswith(alias.rstrip("/") + "/"):
            value = value[len(alias.rstrip("/")) + 1 :]
            break
    p = PurePosixPath(value)
    if not value or p.is_absolute() or ".." in p.parts or "\\" in value or ":" in value:
        return None
    return p.as_posix()


def resolve_public_roots(index, public_task, public_targets=(), *, workspace_aliases=()):
    """Exact public files outrank qualified interfaces and lexical candidates.

    An unresolved explicit root never falls back to unrelated whole-repository
    functions. Ambiguous candidates remain alternatives, not a chosen binding.
    """
    paths = {r["path"] for r in index.files()}
    roots = []
    unknown = []
    resolutions = []
    hints = list(dict.fromkeys(NATIVE_NAME.findall(public_task)))
    requested = []
    for target in public_targets:
        if not isinstance(target, str):
            raise ValueError("public reading target must be text")
        if "::" in target:
            path, name = target.split("::", 1)
            if (
                relative_public_path(path, workspace_aliases) is not None
                and "." in PurePosixPath(path).name
            ):
                # Existing trusted callers canonicalize separately stated path
                # and interface as path::symbol. Both must be public; the
                # synthetic separator need not occur in the natural-language task.
                if path not in public_task or not re.search(
                    r"(?<![\w.])" + re.escape(name) + r"(?!\w)", public_task
                ):
                    raise ValueError("path and symbol must occur in the original public task")
                requested.append((path, name, None, None, "explicit_public_target"))
                continue
        if target not in public_task:
            raise ValueError("reading target must occur in the original public task")
        if PATH.fullmatch(target):
            match = PATH.fullmatch(target)
            requested.append((match[1], None, match[2], match[3], "explicit_public_target"))
        elif not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+", target):
            raise ValueError("target must be a public file, path::symbol or dotted interface")
    for match in PATH.finditer(public_task):
        requested.append((match[1], None, match[2], match[3], "literal_public_path"))
    for raw, name, start, end, origin in requested:
        path = relative_public_path(raw, workspace_aliases)
        if path is None:
            unknown.append(dict(query=raw, status="outside_declared_workspace_or_invalid"))
            continue
        if path not in paths:
            alternatives = sorted(p for p in paths if "/" not in path and p.endswith("/" + path))
            unknown.append(
                dict(
                    query=raw,
                    status="ambiguous_basename" if alternatives else "not_indexed",
                    alternatives=alternatives[:8],
                    alternatives_omitted=max(0, len(alternatives) - 8),
                )
            )
            continue
        previous = next((r for r in roots if r["path"] == path), None)
        if previous:
            if name and not previous.get("symbol_hint"):
                previous["symbol_hint"] = name
            continue
        root = dict(
            path=path,
            kind="public_file",
            selection=origin,
            public_anchor=raw,
            symbol_hint=name,
            identifier_hints=hints,
            line=int(start) if start else None,
            end_line=int(end or start) if start else None,
            status="exact_public_file;runtime_binding_unknown",
        )
        if (
            name
            and index.db.execute(
                "SELECT id FROM symbols WHERE id=?", (path + "::" + name,)
            ).fetchone()
        ):
            root["symbol"] = path + "::" + name
        roots.append(root)
    if not requested:
        resolutions = anchors(index, public_task)
        for resolution in resolutions:
            if resolution["status"] != "unique":
                unknown.append(dict(query=resolution["query"], status=resolution["status"]))
                continue
            answer = resolution["resolutions"][0]
            sid = answer["symbol"]
            path, name = sid.split("::", 1)
            roots.append(
                dict(
                    path=path,
                    kind="public_symbol",
                    symbol=sid,
                    symbol_hint=name,
                    selection="public_qualified_interface",
                    public_anchor=resolution["query"],
                    resolution_chain=answer["chain"],
                    status="static_candidate;runtime_binding_unknown",
                )
            )
    if not roots and not requested and not resolutions:
        # Native qualified names may not have an indexed method declaration.
        for hint in hints:
            rows = list(
                index.db.execute(
                    "SELECT id,path FROM symbols WHERE qualname=? OR name=?",
                    (hint, hint.split("::")[-1]),
                )
            )
            if len(rows) == 1:
                roots.append(
                    dict(
                        path=rows[0]["path"],
                        kind="public_symbol",
                        symbol=rows[0]["id"],
                        symbol_hint=hint,
                        public_anchor=hint,
                        selection="indexed_identifier_candidate",
                        status="static_candidate",
                    )
                )
            else:
                unknown.append(
                    dict(
                        query=hint,
                        status="ambiguous_identifier" if rows else "unresolved_identifier",
                    )
                )
        if not hints:
            for row in index.search(public_task, limit=2):
                roots.append(
                    dict(
                        path=row["path"],
                        kind="lexical_candidate",
                        symbol=row["id"],
                        symbol_hint=row["qualname"],
                        selection="lexical_candidate",
                        status="ambiguity_not_resolved",
                    )
                )
    omitted = max(0, len(roots) - 4)
    return dict(
        roots=roots[:4],
        unknown=unknown,
        resolutions=resolutions,
        roots_omitted=omitted,
        requested_paths=bool(requested),
        scope="reading hints only;no edit permission added",
    )
