"""Bounded current include/import/definition candidates with source evidence.

No commands, imports or new Clang invocations execute. Lexical candidates never
claim a compiler binding, runtime dispatch, ownership or correctness proof.
"""

import ast
import hashlib
from pathlib import Path, PurePosixPath
import re
import time

from infinicoder.languages.native_adapter import BuildProfile, NATIVE_EXTS
from infinicoder.analysis.qualified_symbols import _absolute, module_names


class CurrentSources:
    def __init__(self, index, remaining=None):
        self.index = index
        self.files = {r["path"]: r for r in index.files()}
        self.observed = {}
        self.remaining = remaining
        self.cutoff = time.monotonic() + 2

    def check_time(self):
        if time.monotonic() >= self.cutoff or self.remaining is not None and self.remaining() <= 0:
            raise TimeoutError("bounded dependency view analysis expired")

    def read(self, path):
        self.check_time()
        row = self.index.db.execute("SELECT * FROM files WHERE path=?", (path,)).fetchone()
        if row is None:
            return None
        p = self.index.root / path
        try:
            if p.is_symlink() or not p.is_file() or any(a.is_symlink() for a in p.parents):
                return None
            if not p.resolve().is_relative_to(self.index.root):
                return None
            raw = p.read_bytes()
            if hashlib.sha256(raw).hexdigest() != row["hash"]:
                return None
        except OSError:
            return None
        self.observed[path] = row["hash"]
        return dict(row)

    def stable(self):
        # No timeout here: retained facts are revalidated before returning.
        for path, digest in self.observed.items():
            p = self.index.root / path
            try:
                if p.is_symlink() or not p.is_file() or any(a.is_symlink() for a in p.parents):
                    return False
                if (
                    not p.resolve().is_relative_to(self.index.root)
                    or hashlib.sha256(p.read_bytes()).hexdigest() != digest
                ):
                    return False
            except OSError:
                return False
        return True


def location(file, start, end, kind, **fields):
    return dict(
        kind=kind, path=file["path"], source_sha256=file["hash"], start=start, end=end, **fields
    )


def _directives(file):
    """Whole logical directives plus their conditional context, never half macros."""
    lines = file["source"].splitlines(keepends=True)
    conditions = []
    i = 0
    while i < len(lines):
        start = i
        i += 1
        while i < len(lines) and lines[i - 1].rstrip("\r\n").endswith("\\"):
            i += 1
        source = "".join(lines[start:i])
        m = re.match(r"\s*#\s*(\w+)\b(.*)", source, re.S)
        if not m:
            continue
        directive, body = m.groups()
        if directive == "endif" and conditions:
            conditions.pop()
        if directive in ("elif", "else") and conditions:
            conditions[-1].append(dict(line=start + 1, source=source))
        condition = [item for frame in conditions for item in frame]
        if directive in ("if", "ifdef", "ifndef"):
            conditions.append([dict(line=start + 1, source=source)])
        yield directive, body, start + 1, i, source, condition


def _build(index, sources, unknown):
    # BuildProfile only parses existing configuration; reject linked/oversize
    # input before allowing the existing reader to inspect it.
    for path in (index.root / "compile_commands.json", index.root / "build/compile_commands.json"):
        if not path.exists() and not path.is_symlink():
            continue
        if (
            path.is_symlink()
            or any(a.is_symlink() for a in path.parents)
            or not path.is_file()
            or path.stat().st_size > 1024**2
        ):
            unknown.append(dict(kind="build", status="unsafe_or_oversize_compile_database"))
            return None
    try:
        profile = BuildProfile(index.root)
    except OSError:
        unknown.append(dict(kind="build", status="compile_database_changed_or_unavailable"))
        return None
    if profile.compile_commands:
        rel = profile.compile_commands.relative_to(index.root).as_posix()
        sources.observed[rel] = profile.compile_database_sha256
    return profile


def _include_candidates(profile, sources, path, header, quoted):
    candidates = []
    entry = profile.entry_for(path) if profile else None
    directories = [sources.index.root / PurePosixPath(path).parent] if quoted else []
    if entry:
        for raw in entry.get("includes", []):
            if not isinstance(raw, str):
                continue
            for prefix, target in entry.get("path_mappings", {}).items():
                if raw == prefix or raw.startswith(prefix + "/"):
                    raw = target + raw[len(prefix) :]
                    break
            p = Path(raw)
            directories.append(p if p.is_absolute() else Path(entry["directory"]) / p)
    for directory in directories:
        p = (directory / header).resolve()
        if p.is_relative_to(sources.index.root):
            name = p.relative_to(sources.index.root).as_posix()
            if name in sources.files and name not in candidates:
                candidates.append(name)
    if not candidates:
        candidates = [
            name for name in sources.files if name == header or name.endswith("/" + header)
        ]
    return sorted(candidates)


def expand(sources, roots, target_records, offer, *, max_depth=4, max_files=16):
    """Offer immutable dependency records after the common target packet.

    The caller owns the actual token budget. Analysis is bounded independently
    of exposure, so a missing or oversized record cannot trigger whole-repo IO.
    """
    unknown = []
    index = sources.index
    profile = None
    queue = []
    seen = set()
    scanned = []
    routes = {}
    records = []
    sources.dependency_paths = set()
    sources.dependency_edges = 0
    sources.dependency_max_files = max_files
    expose = offer
    offer = records.append
    module_map = {}
    paths = set(sources.files)
    for candidate in paths:
        sources.check_time()
        for name in module_names(candidate, paths):
            module_map.setdefault(name, []).append(candidate)
    for root in roots:
        file = sources.read(root["path"])
        if file:
            chain = [location(file, 1, 1, "public_root", public_anchor=root.get("public_anchor"))]
            queue.append((root["path"], 0, chain))
            sources.dependency_paths.add(root["path"])
    wanted = set()
    for record in target_records:
        text = record.get("source", "")
        if record["path"].endswith(".py"):
            try:
                wanted.update(
                    n.id
                    for n in ast.walk(ast.parse(text))
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                )
            except SyntaxError:
                wanted.update(re.findall(r"\b[A-Za-z_]\w*\b", text))
        else:
            # Drop comments/literals to avoid treating prose as definitions.
            clean = re.sub(
                r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', " ", text, flags=re.S
            )
            wanted.update(re.findall(r"\b[A-Za-z_]\w*\b", clean))
    if any(Path(r["path"]).suffix.lower() in NATIVE_EXTS for r in roots):
        profile = _build(index, sources, unknown)
    while queue and len(seen) < max_files:
        sources.check_time()
        path, depth, chain = queue.pop(0)
        if path in seen:
            continue
        seen.add(path)
        routes[path] = chain
        file = sources.read(path)
        if not file:
            unknown.append(dict(path=path, status="source_unavailable_or_changed"))
            continue
        scanned.append(path)
        if path.endswith(".py"):
            try:
                tree = ast.parse(file["source"])
            except SyntaxError:
                unknown.append(dict(path=path, status="python_parse_incomplete"))
                continue
            # Only direct module imports. Imports under runtime conditions remain unknown.
            imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
            for node in imports[:16]:
                text = "".join(
                    file["source"].splitlines(keepends=True)[node.lineno - 1 : node.end_lineno]
                )
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                else:
                    names = (
                        [_absolute(path, "." * node.level + node.module)]
                        if node.module
                        else [
                            _absolute(path, "." * node.level + a.name)
                            for a in node.names
                            if a.name != "*"
                        ]
                    )
                    for alias in node.names:
                        if alias.name == "*":
                            unknown.append(
                                dict(
                                    path=path,
                                    line=node.lineno,
                                    status="star_import_binding_unknown",
                                )
                            )
                        elif (alias.asname or alias.name) in wanted:
                            wanted.add(alias.name)
                for name in names:
                    candidates = module_map.get(name, [])
                    edge = location(
                        file, node.lineno, node.end_lineno, "import", source=text, module=name
                    )
                    _follow(
                        sources, candidates, depth, chain, edge, queue, offer, unknown, max_depth
                    )
        elif Path(path).suffix.lower() in NATIVE_EXTS:
            for ordinal, (directive, body, start, end, text, conditions) in enumerate(
                _directives(file)
            ):
                sources.check_time()
                if ordinal >= 64:
                    unknown.append(
                        dict(path=path, status="directive_bound;remaining_definitions_unknown")
                    )
                    break
                if directive == "include":
                    match = re.match(r'\s*([<"])([^>"\n]+)[>"]', body)
                    if not match:
                        unknown.append(
                            dict(path=path, line=start, status="computed_include_unknown")
                        )
                        continue
                    candidates = _include_candidates(
                        profile, sources, path, match[2], match[1] == '"'
                    )
                    edge = location(file, start, end, "include", source=text)
                    if conditions:
                        edge["conditions"] = conditions
                    if profile and profile.compile_database_sha256:
                        edge["build_evidence"] = dict(
                            identity=profile.identity,
                            compile_database_sha256=profile.compile_database_sha256,
                        )
                    _follow(
                        sources, candidates, depth, chain, edge, queue, offer, unknown, max_depth
                    )
                elif directive == "define":
                    match = re.match(r"\s*([A-Za-z_]\w*)", body)
                    if text.rstrip("\r\n").endswith("\\"):
                        unknown.append(
                            dict(path=path, line=start, status="incomplete_logical_directive")
                        )
                        continue
                    if match and match[1] in wanted:
                        offer(
                            location(
                                file,
                                start,
                                end,
                                "dependency_macro",
                                name=match[1],
                                source=text,
                                conditions=conditions,
                                evidence_chain=chain,
                                status="lexical_definition_candidate;preprocessor_selection_unknown",
                            )
                        )
    if queue:
        unknown.append(dict(status="dependency_file_or_depth_bound", pending=len(queue)))
    # Reuse existing declarations in visited files, without expanding native AST.
    for path in scanned:
        file = sources.read(path)
        if not file:
            continue
        rows = index.db.execute(
            "SELECT * FROM symbols WHERE path=? ORDER BY line LIMIT 129", (path,)
        ).fetchall()
        if len(rows) > 128:
            unknown.append(dict(path=path, status="declaration_bound;remaining_candidates_unknown"))
        for row in rows[:128]:
            sources.check_time()
            if row["name"] not in wanted:
                continue
            if any(r.get("symbol") == row["id"] for r in target_records):
                continue
            source = (
                "".join(file["source"].splitlines(keepends=True)[row["line"] - 1 : row["end"]])
                if path.endswith(".py")
                else row["signature"]
            )
            offer(
                location(
                    file,
                    row["line"],
                    row["end"],
                    "dependency_declaration",
                    symbol=row["id"],
                    source=source,
                    status="static_candidate;runtime_dispatch_and_effects_unknown",
                    evidence_chain=routes.get(path, []),
                )
            )
    # Spend the remaining packet budget on used definitions before navigation
    # routes. Analysis order must not crowd the relevant macro out of the view.
    priority = {"dependency_macro": 0, "dependency_declaration": 1, "dependency_route": 2}
    for record in sorted(
        records, key=lambda r: (priority[r["kind"]], len(r.get("evidence_chain", [])))
    ):
        expose(record)
    return dict(
        scanned_files=scanned,
        unknown=unknown,
        max_depth=max_depth,
        max_files=max_files,
        build_identity=profile.identity if profile else None,
        limits="Static candidates only;conditional selection,aliasing,ownership and correctness unknown",
    )


def _follow(sources, candidates, depth, chain, edge, queue, offer, unknown, max_depth):
    if not candidates:
        unknown.append(
            dict(path=edge["path"], line=edge["start"], status="external_or_unresolved_dependency")
        )
        return
    if len(candidates) > 4:
        unknown.append(
            dict(
                path=edge["path"],
                line=edge["start"],
                status="too_many_dependency_alternatives",
                count=len(candidates),
            )
        )
        return
    for candidate in candidates:
        if sources.dependency_edges >= 64:
            unknown.append(dict(path=edge["path"], status="edge_bound;remaining_routes_unknown"))
            return
        if (
            candidate not in sources.dependency_paths
            and len(sources.dependency_paths) >= sources.dependency_max_files
        ):
            unknown.append(dict(path=candidate, status="candidate_file_bound"))
            continue
        sources.dependency_paths.add(candidate)
        sources.dependency_edges += 1
        file = sources.read(candidate)
        if not file:
            continue
        # The next include/definition record already carries the selected
        # file's version. Avoid repeating an extra file node at every hop.
        hop = dict(edge, candidate_path=candidate)
        if len(candidates) > 1:
            hop["candidate_count"] = len(candidates)
        route = chain + [hop]
        offer(
            dict(
                kind="dependency_route",
                path=candidate,
                source_sha256=file["hash"],
                evidence_chain=route,
                status="static_candidate;compiler_or_import_binding_unknown",
                alternatives=len(candidates),
            )
        )
        if depth < max_depth:
            queue.append((candidate, depth + 1, route))
        else:
            unknown.append(dict(path=candidate, status="depth_bound;further_dependencies_unknown"))
