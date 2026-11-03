"""Persistent code facts. SQLite is the source of truth, packs are disposable views.

Python relationships are conservative static references, NOT a dynamic call graph.
C/C++/CUDA/Rust/JS declarations are explicitly heuristic. Unresolved calls remain
unresolved rather than being joined to unrelated same-name methods.
"""

from infinicoder._resources import package_root

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3

IGNORED = {
    ".git",
    ".infinicoder",
    ".kb",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    ".pytest_cache",
}
LANGUAGES = {
    ".py": "python",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hh": "cpp",
    ".hpp": "cpp",
    ".cu": "cuda",
    ".cuh": "cuda",
    ".rs": "rust",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".md": "markdown",
    ".rst": "restructuredtext",
    ".txt": "text",
    ".ipynb": "notebook",
    ".json": "json",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".ini": "config",
    ".cfg": "config",
    ".sh": "shell",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dataflow_identity():
    """On-demand views have an analysis identity even with unchanged source."""
    return digest((package_root() / "analysis/dataflow.py").read_bytes())


def words(text):
    # Split snake/camel identifiers and preserve Chinese terms/bigrams.
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    tokens = re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]+", text.lower())
    return [
        part
        for token in tokens
        for part in (
            [token] + [token[i : i + 2] for i in range(len(token) - 1)]
            if re.search(r"[\u4e00-\u9fff]", token)
            else [token]
        )
    ]


def python_facts(path, source):
    tree = ast.parse(source)
    symbols, refs, binding_sites = [], [], []
    imports = {}

    def local_names(function):
        names, globals_ = set(), set()

        class Locals(ast.NodeVisitor):
            def visit_Name(self, node):
                if isinstance(node.ctx, (ast.Store, ast.Del)):
                    names.add(node.id)

            def visit_FunctionDef(self, node):
                names.add(node.name)

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_ClassDef(self, node):
                names.add(node.name)

            def visit_Lambda(self, node):
                pass

            def visit_Global(self, node):
                globals_.update(node.names)

            def visit_Import(self, node):
                names.update(a.asname or a.name.split(".")[0] for a in node.names)

            def visit_ImportFrom(self, node):
                names.update(a.asname or a.name for a in node.names if a.name != "*")

            def visit_ExceptHandler(self, node):
                if node.name:
                    names.add(node.name)
                self.generic_visit(node)

        collector = Locals()
        for statement in function.body:
            collector.visit(statement)
        return names - globals_

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.scope = []
            self.owner = path + "::<module>"
            self.classes = []
            # lexical bindings per function scope (params, locals,
            # local imports) so calls through local names never link to globals.
            self.bindings = [set()]

        def _bind(self, names):
            for n in names:
                if isinstance(n, str):
                    self.bindings[-1].add(n)

        def visit_Import(self, node):
            inside_function = len(self.bindings) > 1
            for a in node.names:
                name = a.asname or a.name.split(".")[0]
                if not inside_function:
                    imports[name] = a.name if a.asname else a.name.split(".")[0]
                if inside_function:
                    self._bind([name])  # function-local import shadows globals
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            module = "." * node.level + (node.module or "")
            for a in node.names:
                if a.name != "*":
                    if len(self.bindings) == 1:
                        imports[a.asname or a.name] = module + ("." if node.module else "") + a.name
                    if len(self.bindings) > 1:
                        self._bind([a.asname or a.name])

        def visit_ClassDef(self, node):
            qual = ".".join(self.scope + [node.name])
            fields = []
            for child in node.body:
                if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                    fields.append({"name": child.target.id, "type": ast.unparse(child.annotation)})
            symbols.append(
                {
                    "id": path + "::" + qual,
                    "name": node.name,
                    "qualname": qual,
                    "kind": "class",
                    "line": node.lineno,
                    "end": node.end_lineno,
                    "signature": "class "
                    + qual
                    + "("
                    + ", ".join(ast.unparse(b) for b in node.bases)
                    + ")",
                    "doc": (ast.get_docstring(node) or "")[:600],
                    "fields": fields,
                    "confidence": "syntax",
                }
            )
            old = self.owner
            self.scope.append(node.name)
            self.classes.append(qual)
            self.owner = path + "::" + qual
            self.generic_visit(node)
            self.scope.pop()
            self.classes.pop()
            self.owner = old

        def visit_FunctionDef(self, node):
            qual = ".".join(self.scope + [node.name])
            sig = (
                ("async " if isinstance(node, ast.AsyncFunctionDef) else "")
                + "def "
                + qual
                + "("
                + ast.unparse(node.args)
                + ")"
            )
            if node.returns:
                sig += " -> " + ast.unparse(node.returns)
            symbols.append(
                {
                    "id": path + "::" + qual,
                    "name": node.name,
                    "qualname": qual,
                    "kind": "function",
                    "line": node.lineno,
                    "end": node.end_lineno,
                    "signature": sig,
                    "doc": (ast.get_docstring(node) or "")[:600],
                    "fields": [],
                    "confidence": "syntax",
                }
            )
            old = self.owner
            self.owner = path + "::" + qual
            self.scope.append(node.name)
            # parameters are bindings of this function scope.
            argnames = [
                a.arg for a in node.args.args + node.args.posonlyargs + node.args.kwonlyargs
            ]
            if node.args.vararg:
                argnames.append(node.args.vararg.arg)
            if node.args.kwarg:
                argnames.append(node.args.kwarg.arg)
            # Python determines locals for the entire function, not in source
            # visitation order (a later assignment shadows an earlier call).
            self.bindings.append(set(argnames) | local_names(node))
            for child in node.body:
                self.visit(child)
            self.scope.pop()
            self.bindings.pop()
            self.owner = old

        visit_AsyncFunctionDef = visit_FunctionDef

        def _assigned_names(self, target):
            """Collect simple names bound by an assignment/for/with/except target."""
            if isinstance(target, ast.Name):
                self._bind([target.id])
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    self._assigned_names(elt)
            elif isinstance(target, ast.Starred):
                self._assigned_names(target.value)

        def visit_Assign(self, node):
            for t in node.targets:
                self._assigned_names(t)
                self._binding_symbols(t, node)
            self.generic_visit(node)

        def visit_AnnAssign(self, node):
            if node.value is not None:
                self._assigned_names(node.target)
            self._binding_symbols(node.target, node)
            self.generic_visit(node)

        def _binding_symbols(self, target, node):
            if len(self.bindings) > 1:
                return
            if isinstance(target, (ast.Tuple, ast.List)):
                for item in target.elts:
                    self._binding_symbols(item, node)
                return
            if not isinstance(target, ast.Name):
                return
            qual = ".".join(self.scope + [target.id])
            sid = path + "::" + qual
            kind = "class_field" if self.classes else "variable"
            symbols.append(
                {
                    "id": sid,
                    "name": target.id,
                    "qualname": qual,
                    "kind": kind,
                    "line": node.lineno,
                    "end": node.end_lineno,
                    "signature": ast.unparse(node)[:500],
                    "doc": "",
                    "fields": [],
                    "confidence": "syntax-binding-candidate",
                }
            )
            binding_sites.append(
                {"symbol_id": sid, "line": node.lineno, "end": node.end_lineno, "kind": kind}
            )

        def visit_AugAssign(self, node):
            self._assigned_names(node.target)
            self.generic_visit(node)

        def visit_For(self, node):
            self._assigned_names(node.target)
            self.generic_visit(node)

        visit_AsyncFor = visit_For

        def visit_With(self, node):
            for item in node.items:
                if item.optional_vars is not None:
                    self._assigned_names(item.optional_vars)
            self.generic_visit(node)

        visit_AsyncWith = visit_With

        def visit_ExceptHandler(self, node):
            if node.name:
                self._bind([node.name])
            self.generic_visit(node)

        def visit_Global(self, node):
            # explicit global: remove from local bindings so globals resolve
            for name in node.names:
                self.bindings[-1].discard(name)
            self.generic_visit(node)

        def visit_Call(self, node):
            target = ast.unparse(node.func)
            # a call through a locally bound name (parameter,
            # local variable, local import) is NOT a global symbol reference.
            # Only BARE-name calls can be mis-linked to same-name globals;
            # attribute calls on unknown objects stay unresolved naturally.
            shadowed = "." not in target and len(self.bindings) > 1 and target in self.bindings[-1]
            refs.append(
                {
                    "src": self.owner,
                    "target": target,
                    "kind": "calls",
                    "line": node.lineno,
                    "class_scope": self.classes[-1] if self.classes else "",
                    "shadowed": bool(shadowed),
                }
            )
            self.generic_visit(node)

        def visit_Attribute(self, node):
            if (
                self.classes
                and isinstance(node.value, ast.Name)
                and node.value.id in ("self", "cls")
            ):
                cls = next(s for s in symbols if s["id"] == path + "::" + self.classes[-1])
                kind = "writes" if isinstance(node.ctx, ast.Store) else "reads"
                if kind == "writes" and not any(f["name"] == node.attr for f in cls["fields"]):
                    cls["fields"].append({"name": node.attr, "type": "unknown"})
                refs.append(
                    {
                        "src": self.owner,
                        "target": cls["id"] + "." + node.attr,
                        "kind": kind,
                        "line": node.lineno,
                        "class_scope": self.classes[-1],
                    }
                )
            self.generic_visit(node)

    Visitor().visit(tree)
    definitions = _collect_definitions(path, tree) + binding_sites
    return symbols, refs, imports, definitions


def _collect_definitions(path, tree):
    """every definition SITE (getter+setter overloads coexist)."""
    defs = []

    def walk(node, scope):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qual = ".".join(scope + [child.name])
                kind = "class" if isinstance(child, ast.ClassDef) else "function"
                defs.append(
                    {
                        "symbol_id": path + "::" + qual,
                        "line": child.lineno,
                        "end": child.end_lineno,
                        "kind": kind,
                    }
                )
                walk(child, scope + [child.name])
            else:
                walk(child, scope)

    walk(tree, [])
    return defs


def heuristic_facts(path, source):
    symbols = []
    # Deliberately no guessed call edges: a token match does not establish dispatch.
    pattern = re.compile(
        r"^\s*(?:(?:pub\s+)?(?:struct|class|enum|trait)\s+(\w+)|"
        r"(?:export\s+)?(?:async\s+)?(?:function|fn)\s+(\w+)\s*\(|"
        r"(?:[\w:*<>]+\s+)+([\w:]+)\s*\([^;]*\)\s*(?:const\s*)?\{)"
    )
    for line, text in enumerate(source.splitlines(), 1):
        match = pattern.search(text)
        if match:
            name = next(g for g in match.groups() if g)
            symbols.append(
                {
                    "id": f"{path}::{name}@{line}",
                    "name": name,
                    "qualname": name,
                    "kind": "type" if match.group(1) else "function",
                    "line": line,
                    "end": line,
                    "signature": text.strip()[:400],
                    "doc": "",
                    "fields": [],
                    "confidence": "heuristic",
                }
            )
    return symbols, [], {}, []


class CodeIndex:
    def __init__(
        self,
        root,
        database,
        *,
        native_index_policy="eager",
        native_resources=None,
        remaining=None,
        native_timeout=5,
        native_parent_guard=False,
    ):
        if native_index_policy not in ("eager", "on_demand"):
            raise ValueError("Unknown native index policy")
        if remaining is not None and not callable(remaining):
            raise ValueError("remaining must be a callable")
        if not 0 < native_timeout <= 30:
            raise ValueError("native_timeout must be 0..30 seconds")
        if type(native_parent_guard) is not bool:
            raise ValueError("native_parent_guard must be bool")
        self.native_parent_guard = native_parent_guard
        self.native_index_policy = native_index_policy
        self.native_resources = native_resources
        self.remaining = remaining
        self.native_timeout = native_timeout
        self.root = Path(root).resolve()
        self.database = Path(database).resolve()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.database, timeout=20)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, hash TEXT, language TEXT,
                source TEXT, error TEXT, imports TEXT);
            CREATE TABLE IF NOT EXISTS symbols(id TEXT PRIMARY KEY, path TEXT, name TEXT,
                qualname TEXT, kind TEXT, line INTEGER, end INTEGER, signature TEXT,
                doc TEXT, fields TEXT, confidence TEXT);
            CREATE INDEX IF NOT EXISTS symbol_path ON symbols(path);
            CREATE TABLE IF NOT EXISTS refs(path TEXT, src TEXT, target TEXT, kind TEXT,
                line INTEGER, class_scope TEXT, dst TEXT, confidence TEXT, shadowed INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS definitions(symbol_id TEXT, path TEXT, line INTEGER,
                end INTEGER, kind TEXT);
            CREATE INDEX IF NOT EXISTS def_symbol ON definitions(symbol_id);
            CREATE INDEX IF NOT EXISTS ref_src ON refs(src);
            CREATE INDEX IF NOT EXISTS ref_dst ON refs(dst);
            CREATE TABLE IF NOT EXISTS native_inputs(path TEXT,file TEXT,hash TEXT,
                PRIMARY KEY(path,file));
            CREATE TABLE IF NOT EXISTS native_input_state(path TEXT PRIMARY KEY,
                revision TEXT,complete INTEGER,generation INTEGER);
            CREATE TABLE IF NOT EXISTS partial_source_refresh(path TEXT PRIMARY KEY,analysis TEXT);
        """)
        previous = self.db.execute("SELECT value FROM metadata WHERE key='root'").fetchone()
        if previous and previous[0] != str(self.root):
            self.db.close()
            raise ValueError("Index belongs to a different workspace")
        # Schema migration: older databases lack refs.shadowed.
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(refs)")}
        if "shadowed" not in cols:
            self.db.execute("ALTER TABLE refs ADD COLUMN shadowed INTEGER DEFAULT 0")
            self.db.commit()
        fcols = {r[1] for r in self.db.execute("PRAGMA table_info(files)")}
        if "build" not in fcols:
            self.db.execute("ALTER TABLE files ADD COLUMN build TEXT")
            self.db.commit()
        self.db.execute("INSERT OR REPLACE INTO metadata VALUES ('root', ?)", (str(self.root),))
        self.db.commit()

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _check_analysis_budget(self):
        if self.remaining is not None and self.remaining() <= 0:
            from infinicoder.analysis.native_analysis import IndexBudgetExceeded

            self._search_cache = None
            raise IndexBudgetExceeded("Code analysis exhausted its original runtime budget")

    def expand_native(self, path):
        from infinicoder.languages.native_adapter import NATIVE_EXTS
        from infinicoder.execution.workspace_tools import Workspace

        Workspace(self.root).path(path)
        if self.native_index_policy != "on_demand" or Path(path).suffix.lower() not in NATIVE_EXTS:
            return None
        self.refresh(_native_paths=(path,))
        row = self.db.execute("SELECT build FROM files WHERE path=?", (path,)).fetchone()
        if not row:
            raise ValueError("path is not indexed")
        return json.loads(row[0] or "{}")

    def refresh_paths(self, paths, *, read_source):
        """Observe just the requested files; unseen rows are never deleted.

        The caller supplies a safe versioned read in its declared namespace.
        Native declarations stay heuristic here; semantic expansion retains its
        separate build/dependency contract. Full refresh will revisit these
        rows, even when their bytes match an older full-index generation.
        """
        from infinicoder.execution.workspace_tools import Workspace
        from infinicoder.languages.native_adapter import BuildProfile, native_context, NATIVE_EXTS

        paths = tuple(dict.fromkeys(paths))
        if len(paths) > 32 or not callable(read_source):
            raise ValueError("bounded explicit source paths required")
        analysis = digest(Path(__file__).read_bytes())
        changed = []
        deleted = []
        observed = []
        build = None
        with self.db:
            for path in paths:
                self._check_analysis_budget()
                Workspace(self.root).path(path)
                try:
                    raw, text, sha = read_source(path)
                except FileNotFoundError:
                    for table in (
                        "files",
                        "symbols",
                        "refs",
                        "definitions",
                        "native_inputs",
                        "native_input_state",
                        "partial_source_refresh",
                    ):
                        self.db.execute(f"DELETE FROM {table} WHERE path=?", (path,))
                    deleted.append(path)
                    continue
                if digest(raw) != sha or raw.decode("utf-8-sig") != text:
                    raise ValueError("exact observed source bytes and hash required")
                observed.append(path)
                suffix = Path(path).suffix.lower()
                error = ""
                info = None
                old = self.db.execute("SELECT hash FROM files WHERE path=?", (path,)).fetchone()
                partial = self.db.execute(
                    "SELECT analysis FROM partial_source_refresh WHERE path=?", (path,)
                ).fetchone()
                if (
                    suffix not in NATIVE_EXTS
                    and old
                    and old[0] == sha
                    and partial
                    and partial[0] == analysis
                ):
                    continue
                syms, refs, imports, definitions = [], [], {}, []
                try:
                    if suffix in (".py", ".pyi"):
                        syms, refs, imports, definitions = python_facts(path, text)
                    elif suffix in NATIVE_EXTS:
                        if build is None:
                            build = BuildProfile(self.root)
                        syms, refs, imports, definitions, info = native_context(
                            path, text, build, semantic=False
                        )
                    elif suffix == ".ipynb":
                        syms, refs, imports, definitions = [], [], {}, []
                    else:
                        syms, refs, imports, definitions = heuristic_facts(path, text)
                except SyntaxError as exc:
                    error = f"{exc.msg} at line {exc.lineno}"
                for table in (
                    "symbols",
                    "refs",
                    "definitions",
                    "native_inputs",
                    "native_input_state",
                ):
                    self.db.execute(f"DELETE FROM {table} WHERE path=?", (path,))
                self.db.execute(
                    "INSERT OR REPLACE INTO files VALUES(?,?,?,?,?,?,?)",
                    (
                        path,
                        sha,
                        LANGUAGES.get(suffix, "other"),
                        text,
                        error,
                        json.dumps(imports),
                        json.dumps(info) if info else None,
                    ),
                )
                self.db.executemany(
                    "INSERT OR REPLACE INTO symbols VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    [
                        (
                            s["id"],
                            path,
                            s["name"],
                            s["qualname"],
                            s["kind"],
                            s["line"],
                            s["end"],
                            s["signature"],
                            s["doc"],
                            json.dumps(s["fields"]),
                            s["confidence"],
                        )
                        for s in syms
                    ],
                )
                self.db.executemany(
                    "INSERT INTO refs VALUES(?,?,?,?,?,?,?,?,?)",
                    [
                        (
                            path,
                            r["src"],
                            r["target"],
                            r["kind"],
                            r["line"],
                            r["class_scope"],
                            None,
                            "unresolved",
                            int(bool(r.get("shadowed"))),
                        )
                        for r in refs
                    ],
                )
                self.db.executemany(
                    "INSERT INTO definitions VALUES(?,?,?,?,?)",
                    [(d["symbol_id"], path, d["line"], d["end"], d["kind"]) for d in definitions],
                )
                self.db.execute(
                    "INSERT OR REPLACE INTO partial_source_refresh VALUES(?,?)", (path, analysis)
                )
                changed.append(path)
            # Older callers of a changed target remain unknown until refreshed.
            for path in changed + deleted:
                self.db.execute(
                    "UPDATE refs SET dst=NULL,confidence=? WHERE substr(dst,1,?)=?",
                    ("target_changed;refresh_required", len(path) + 2, path + "::"),
                )
            # A source-preserving reread can restore a relation invalidated by
            # a changed target. Unobserved callers must remain unknown.
            if observed:
                self._resolve(paths=observed)
        self._search_cache = None
        return dict(
            changed=changed, deleted=deleted, coverage="selected_paths_only;unseen_paths_unchanged"
        )

    def refresh(
        self, max_bytes=1_000_000, _force_native=False, _already_native=(), _native_paths=()
    ):
        from infinicoder.languages.native_adapter import BuildProfile, native_context, NATIVE_EXTS

        bp = BuildProfile(self.root)
        import importlib.metadata

        try:
            clang_version = importlib.metadata.version("libclang")
        except importlib.metadata.PackageNotFoundError:
            clang_version = None
        parser_identity = digest(
            Path(__file__).read_bytes()
            + (package_root() / "analysis/qualified_symbols.py").read_bytes()
            + (package_root() / "analysis/native_effects.py").read_bytes()
            + str(clang_version).encode()
            + b"".join(
                (package_root() / "languages" / name).read_bytes()
                for name in ["native_adapter.py", "clang_adapter.py"]
            )
        )
        from infinicoder.analysis.native_analysis import policy_identity

        parser_identity = digest(
            (
                parser_identity
                + ":"
                + self.native_index_policy
                + ":"
                + str(self.native_parent_guard)
                + ":"
                + policy_identity(self.native_resources)
            ).encode()
            + (package_root() / "analysis/native_analysis.py").read_bytes()
        )
        previous_parser = self.db.execute(
            "SELECT value FROM metadata WHERE key='parser_identity'"
        ).fetchone()
        parser_changed = not previous_parser or previous_parser[0] != parser_identity
        previous_build = self.db.execute(
            "SELECT value FROM metadata WHERE key='build_identity'"
        ).fetchone()
        build_changed = not previous_build or previous_build[0] != bp.identity
        import os

        seen, changed, skipped = set(), [], []
        header_changed = False
        parsed_native = set()
        with self.db:
            for folder, dirs, names in os.walk(self.root, followlinks=False):
                excluded = getattr(self, "excluded_paths", ())
                dirs[:] = sorted(
                    d
                    for d in dirs
                    if d not in IGNORED
                    and not Path(folder, d).is_symlink()
                    and not any(Path(folder, d).resolve().is_relative_to(p) for p in excluded)
                )
                self._check_analysis_budget()
                for name in sorted(names):
                    self._check_analysis_budget()
                    p = Path(folder, name)
                    # A source-looking FIFO/device is not a readable source
                    # fact. Never block the caller while opening such entries.
                    if p.suffix.lower() not in LANGUAGES or p.is_symlink() or not p.is_file():
                        continue
                    if not p.resolve().is_relative_to(self.root) or p.resolve() == self.database:
                        continue
                    if any(p.resolve().is_relative_to(q) for q in excluded):
                        continue
                    rel = p.relative_to(self.root).as_posix()
                    if p.stat().st_size > max_bytes:
                        skipped.append(rel)
                        continue
                    raw = p.read_bytes()
                    if b"\x00" in raw:
                        skipped.append(rel)
                        continue
                    seen.add(rel)
                    sha = digest(raw)
                    old = self.db.execute("SELECT hash FROM files WHERE path=?", (rel,)).fetchone()
                    if p.suffix.lower() in (".h", ".hh", ".hpp", ".cuh") and (
                        not old or old[0] != sha
                    ):
                        header_changed = True
                    partial = self.db.execute(
                        "SELECT 1 FROM partial_source_refresh WHERE path=?", (rel,)
                    ).fetchone()
                    if (
                        old
                        and old[0] == sha
                        and not partial
                        and not parser_changed
                        and not (build_changed and p.suffix.lower() in NATIVE_EXTS)
                    ):
                        if p.suffix.lower() not in NATIVE_EXTS:
                            continue
                        if _force_native and rel in _already_native:
                            continue
                        dependency = self.db.execute(
                            "SELECT complete FROM native_input_state WHERE path=?", (rel,)
                        ).fetchone()
                        if dependency:
                            from infinicoder.languages.clang_adapter import header_fingerprint

                            same = bool(dependency[0])
                            for filename, expected in self.db.execute(
                                "SELECT file,hash FROM native_inputs WHERE path=?", (rel,)
                            ):
                                current = header_fingerprint(filename, bp.dependency_cache)
                                if current is None or current[0] != expected:
                                    same = False
                                    break
                            if same:
                                continue
                        elif not (_force_native or rel in _native_paths):
                            continue
                    changed.append(rel)
                    text = raw.decode("utf-8-sig", errors="replace")
                    error, syms, refs, imports, definitions = "", [], [], {}, []
                    build_info = None
                    try:
                        if p.suffix == ".py":
                            syms, refs, imports, definitions = python_facts(rel, text)
                        elif p.suffix.lower() == ".ipynb":
                            syms, refs, imports, definitions = [], [], {}, []
                        elif p.suffix.lower() in NATIVE_EXTS:
                            if self.native_index_policy == "on_demand" and rel in _native_paths:
                                from infinicoder.analysis.native_analysis import expand

                                seconds = (
                                    min(self.native_timeout, self.remaining())
                                    if self.remaining is not None
                                    else self.native_timeout
                                )
                                syms, refs, imports, definitions, build_info = expand(
                                    rel,
                                    text,
                                    bp,
                                    resources=self.native_resources,
                                    timeout=seconds,
                                    guard_parent=self.native_parent_guard,
                                )
                            elif (
                                self.native_index_policy == "eager"
                                and self.native_resources is None
                            ):
                                syms, refs, imports, definitions, build_info = native_context(
                                    rel, text, bp
                                )
                            else:
                                syms, refs, imports, definitions, build_info = native_context(
                                    rel,
                                    text,
                                    bp,
                                    semantic=self.native_index_policy == "eager",
                                    resource_dir=self.native_resources,
                                )
                        else:
                            syms, refs, imports, definitions = heuristic_facts(rel, text)
                    except SyntaxError as e:
                        error = f"{e.msg} at line {e.lineno}"
                    if p.suffix.lower() in NATIVE_EXTS:
                        parsed_native.add(rel)
                        inputs = build_info.pop("_native_inputs", None) if build_info else None
                        prior = self.db.execute(
                            "SELECT generation FROM native_input_state WHERE path=?", (rel,)
                        ).fetchone()
                        if inputs is not None:
                            self.db.execute("DELETE FROM native_inputs WHERE path=?", (rel,))
                            self.db.executemany(
                                "INSERT INTO native_inputs VALUES(?,?,?)",
                                [(rel, name, value) for name, value in inputs["files"].items()],
                            )
                        elif build_info and build_info.get("mode") == "compile_commands" and prior:
                            # Retain dependency candidates after a parse error so
                            # restoring a deleted/generated include can recover.
                            from infinicoder.languages.clang_adapter import header_fingerprint

                            observed = []
                            for filename, _ in self.db.execute(
                                "SELECT file,hash FROM native_inputs WHERE path=?", (rel,)
                            ):
                                value = header_fingerprint(filename, bp.dependency_cache)
                                observed.append((filename, value[0] if value else None))
                            inputs = dict(
                                files={},
                                complete=False,
                                unknown=[
                                    dict(
                                        reason="semantic parse unavailable;previous dependency candidates retained"
                                    )
                                ],
                                unknown_count=1,
                                revision=digest(json.dumps(observed).encode()),
                            )
                        else:
                            self.db.execute("DELETE FROM native_inputs WHERE path=?", (rel,))
                            self.db.execute("DELETE FROM native_input_state WHERE path=?", (rel,))
                        if inputs is not None:
                            generation = (prior[0] if prior else 0) + 1
                            dependency_revision = (
                                inputs["revision"]
                                if inputs["complete"]
                                else digest(
                                    (
                                        inputs["revision"] + ":unversioned:" + str(generation)
                                    ).encode()
                                )
                            )
                            self.db.execute(
                                "INSERT OR REPLACE INTO native_input_state VALUES(?,?,?,?)",
                                (rel, dependency_revision, int(inputs["complete"]), generation),
                            )
                            build_info.update(
                                native_dependency_revision=dependency_revision,
                                native_dependency_count=len(inputs["files"]),
                                native_dependency_complete=inputs["complete"],
                            )
                            if not inputs["complete"]:
                                build_info.update(
                                    native_dependency_unknown=inputs["unknown"],
                                    native_dependency_unknown_count=inputs["unknown_count"],
                                )
                    self.db.execute("DELETE FROM symbols WHERE path=?", (rel,))
                    self.db.execute("DELETE FROM refs WHERE path=?", (rel,))
                    self.db.execute("DELETE FROM definitions WHERE path=?", (rel,))
                    self.db.execute(
                        "INSERT OR REPLACE INTO files VALUES(?,?,?,?,?,?,?)",
                        (
                            rel,
                            sha,
                            LANGUAGES.get(p.suffix.lower(), "other"),
                            text,
                            error,
                            json.dumps(imports),
                            json.dumps(build_info) if build_info else None,
                        ),
                    )
                    for s in syms:
                        self.db.execute(
                            "INSERT OR REPLACE INTO symbols VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                            (
                                s["id"],
                                rel,
                                s["name"],
                                s["qualname"],
                                s["kind"],
                                s["line"],
                                s["end"],
                                s["signature"],
                                s["doc"],
                                json.dumps(s["fields"]),
                                s["confidence"],
                            ),
                        )
                    for r in refs:
                        self.db.execute(
                            "INSERT INTO refs VALUES(?,?,?,?,?,?,?,?,?)",
                            (
                                rel,
                                r["src"],
                                r["target"],
                                r["kind"],
                                r["line"],
                                r["class_scope"],
                                None,
                                "unresolved",
                                1 if r.get("shadowed") else 0,
                            ),
                        )
                    for d in definitions:
                        self.db.execute(
                            "INSERT INTO definitions VALUES(?,?,?,?,?)",
                            (d["symbol_id"], rel, d["line"], d["end"], d["kind"]),
                        )
                    self.db.execute("DELETE FROM partial_source_refresh WHERE path=?", (rel,))
            deleted = [r[0] for r in self.db.execute("SELECT path FROM files") if r[0] not in seen]
            for rel in deleted:
                if Path(rel).suffix.lower() in (".h", ".hh", ".hpp", ".cuh"):
                    header_changed = True
                for table in (
                    "files",
                    "symbols",
                    "refs",
                    "definitions",
                    "native_inputs",
                    "native_input_state",
                    "partial_source_refresh",
                ):
                    self.db.execute(f"DELETE FROM {table} WHERE path=?", (rel,))
            self._check_analysis_budget()
            self._resolve()
            self._check_analysis_budget()
            self.db.execute(
                "INSERT OR REPLACE INTO metadata VALUES ('build_identity', ?)", (bp.identity,)
            )
            self.db.execute(
                "INSERT OR REPLACE INTO metadata VALUES ('parser_identity', ?)", (parser_identity,)
            )
        # Initial/parser/build refresh already parsed every native source
        # against current headers. Only an otherwise incremental refresh needs
        # the second pass for unchanged translation units that include a header.
        if (
            self.native_index_policy == "eager"
            and header_changed
            and not (_force_native or parser_changed or build_changed)
        ):
            refreshed = self.refresh(max_bytes, _force_native=True, _already_native=parsed_native)
            changed = list(dict.fromkeys(changed + refreshed["changed"]))
        if changed or deleted or parser_changed or build_changed:
            self._search_cache = None
        return {
            "changed": changed,
            "deleted": deleted,
            "skipped": skipped,
            "files": len(seen),
            "revision": self.revision(),
            "native_headers_invalidated": header_changed,
            "native_index_policy": self.native_index_policy,
            "native_requested_paths": list(_native_paths),
        }

    def revision(self):
        rows = self.db.execute("SELECT path,hash FROM files ORDER BY path").fetchall()
        identity = self.db.execute(
            "SELECT key,value FROM metadata WHERE key IN ('build_identity','parser_identity') ORDER BY key"
        ).fetchall()
        native = self.db.execute(
            "SELECT path,revision,complete FROM native_input_state ORDER BY path"
        ).fetchall()
        partial = self.db.execute(
            "SELECT path,analysis FROM partial_source_refresh ORDER BY path"
        ).fetchall()
        return digest(
            json.dumps(
                {
                    "files": [tuple(r) for r in rows],
                    "analysis": [tuple(r) for r in identity],
                    "native_dependencies": [tuple(r) for r in native],
                    "partial_sources": [tuple(r) for r in partial],
                    "python_dataflow_analysis": dataflow_identity(),
                }
            ).encode()
        )

    def use_def(self, path, name=None, limit=100, scope=None, *, expand_semantics=True):
        from infinicoder.analysis.dataflow import python_use_def

        if scope is not None and (
            not isinstance(scope, str) or not scope.strip() or len(scope) > 500
        ):
            raise ValueError("scope must be a nonempty qualified name")
        if type(limit) is not int or not 1 <= limit <= 500:
            raise ValueError("limit must be 1..500")
        if type(expand_semantics) is not bool:
            raise ValueError("expand_semantics must be bool")
        if expand_semantics:
            self.expand_native(path)
        row = self.db.execute(
            "SELECT hash,source,build FROM files WHERE path=?", (path,)
        ).fetchone()
        if row is None:
            raise ValueError("path is not indexed")
        if not path.endswith(".py"):
            build = json.loads(row["build"]) if row["build"] else {}
            edges = [
                e
                for e in build.get("use_def", [])
                if (name is None or e["name"] == name)
                and (scope is None or e.get("scope") == scope)
            ]
            return {
                "path": path,
                "source_sha256": row["hash"],
                "edges": edges[:limit],
                "omitted": max(0, len(edges) - limit),
                "limitations": [
                    build.get(
                        "limitation",
                        "semantic Clang indexing unavailable; no native use-def claims",
                    )
                ],
            }
        # Persist expensive analysis independently from model-authored notes.
        # Source and analyzer identities jointly invalidate a cached view.
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS memory_dataflow(path TEXT PRIMARY KEY, source_sha256 TEXT, analysis_identity TEXT, facts TEXT)"
        )
        analysis = dataflow_identity()
        cached = self.db.execute(
            "SELECT source_sha256,analysis_identity,facts FROM memory_dataflow WHERE path=?",
            (path,),
        ).fetchone()
        if (
            cached
            and cached["source_sha256"] == row["hash"]
            and cached["analysis_identity"] == analysis
        ):
            result = json.loads(cached["facts"])
        else:
            result = python_use_def(row["source"])
            with self.db:
                self.db.execute(
                    "DELETE FROM memory_dataflow WHERE path NOT IN (SELECT path FROM files)"
                )
                self.db.execute(
                    "INSERT OR REPLACE INTO memory_dataflow VALUES(?,?,?,?)",
                    (path, row["hash"], analysis, json.dumps(result, ensure_ascii=False)),
                )
        wanted_scope = (
            scope
            if scope is None or scope == "<module>" or scope.startswith("<module>.")
            else "<module>." + scope
        )
        selected = [
            e
            for e in result["edges"]
            if (name is None or e["name"] == name)
            and (wanted_scope is None or e["scope"] == wanted_scope)
        ]
        return dict(
            path=path,
            source_sha256=row["hash"],
            index_revision=self.revision(),
            analysis_identity=dataflow_identity(),
            edges=selected[:limit],
            omitted=max(0, len(selected) - limit),
            limitations=result["limitations"],
        )

    def memory_route(self, query, limit=8):
        from infinicoder.memory.repository_memory import RepositoryMemory

        return RepositoryMemory(self).route(query, limit)

    def pack_tokens(self, query, max_tokens, estimator, focus_paths=(), exclude_paths=()):
        if getattr(self, "task_focused_memory", False) and getattr(self, "task_flow_memory", False):
            from infinicoder.memory.flow_memory import pack

            return pack(
                self,
                query,
                max_tokens,
                estimator,
                focus_paths,
                exclude_paths,
                view=getattr(self, "task_memory_view", "auto"),
            )
        if getattr(self, "task_focused_memory", False):
            from infinicoder.memory.task_memory import TaskMemory

            return TaskMemory(self).pack(
                query,
                max_tokens,
                estimator,
                focus_paths,
                exclude_paths,
                view=getattr(self, "task_memory_view", "auto"),
            )
        from infinicoder.memory.repository_memory import RepositoryMemory

        return RepositoryMemory(self).pack(query, max_tokens, estimator, focus_paths, exclude_paths)

    def record_inspection(self, result):
        from infinicoder.memory.inspection_ledger import InspectionLedger

        return InspectionLedger(self).record(result)

    def memory_inspections(self, paths=(), limit=20):
        from infinicoder.memory.inspection_ledger import InspectionLedger

        return InspectionLedger(self).query(paths, limit)

    def flow_slice(self, symbol_id, names=(), view="values", max_lines=16):
        from infinicoder.memory.flow_memory import slice_function, bounded_slice

        return bounded_slice(slice_function(self, symbol_id, names, view, max_lines))

    def function_summary(self, symbol_id):
        if (
            getattr(self, "task_flow_memory", False)
            or getattr(self, "flow_update_policy", None) == "same_root_v1"
        ):
            return self.flow_slice(symbol_id)
        from infinicoder.memory.task_memory import TaskMemory, bounded_summary

        return bounded_summary(TaskMemory(self).summary(symbol_id))

    def memory_expand(self, module=".", symbol_id=None, limit=20):
        from infinicoder.memory.long_memory import HierarchicalMemory

        return HierarchicalMemory(self).expand(module, symbol_id, limit)

    def _resolve(self, paths=None):
        symbols = {r["id"]: dict(r) for r in self.db.execute("SELECT * FROM symbols")}
        # Clang identities span translation units;short names do not establish
        # overload,namespace or dispatch bindings. Keep these facts in existing
        # versioned file metadata,so old indexes rebuild without schema changes.
        native = {
            r["path"]: json.loads(r["build"]) if r["build"] else {}
            for r in self.db.execute('SELECT path,build FROM files WHERE language!="python"')
        }
        native_symbols = {}
        symbol_counts = Counter()
        native_refs = {}
        for build in native.values():
            for binding in build.get("native_bindings", {}).get("symbols", []):
                sid = binding["symbol"]
                usr = binding["usr"]
                symbol_counts[sid] += 1
                if sid in symbols and usr:
                    native_symbols.setdefault(usr, set()).add(sid)
        # Binding records and rows share the parser's exact traversal order;
        # their full identities are checked before any semantic link is used.
        # This preserves distinct overloaded calls on the very same source line.
        for path, build in native.items():
            rows = list(
                self.db.execute("SELECT rowid,* FROM refs WHERE path=? ORDER BY rowid", (path,))
            )
            bindings = build.get("native_bindings", {}).get("refs", [])
            if len(rows) != len(bindings):
                continue
            for row, binding in zip(rows, bindings):
                if (row["src"], row["target"], row["line"], row["kind"]) == (
                    binding["src"],
                    binding["target"],
                    binding["line"],
                    "calls",
                ):
                    native_refs[row["rowid"]] = binding
        modules = {}
        from infinicoder.analysis.qualified_symbols import module_names

        python_paths = {
            row[0] for row in self.db.execute('SELECT path FROM files WHERE language="python"')
        }
        for path in sorted(python_paths):
            for module in module_names(path, python_paths):
                modules.setdefault(module, []).append(path)
        imports = {
            r["path"]: json.loads(r["imports"])
            for r in self.db.execute("SELECT path,imports FROM files")
        }
        selected = None if paths is None else set(paths)
        for row in list(self.db.execute("SELECT rowid,* FROM refs")):
            if selected is not None and row["path"] not in selected:
                continue
            dst = None
            target, path = row["target"], row["path"]
            confidence = "unresolved"
            if path in native:
                binding = native_refs.get(row["rowid"])
                if not binding:
                    confidence = "clang_binding_unavailable"
                elif symbol_counts.get(row["src"]) != 1:
                    confidence = "clang_ambiguous_symbol"
                elif binding["dispatch"] != "direct":
                    confidence = "clang_" + binding["dispatch"]
                else:
                    candidates = native_symbols.get(binding["usr"], set())
                    # Internal USRs may contain only a basename. Different
                    # translation units with the same filename are not one
                    # linkage scope. Header-local bodies remain conservative.
                    if binding.get("translation_unit_local"):
                        candidates = {sid for sid in candidates if symbols[sid]["path"] == path}
                    if not candidates:
                        confidence = "clang_no_indexed_definition"
                    elif len(candidates) != 1 or any(symbol_counts[sid] != 1 for sid in candidates):
                        confidence = "clang_ambiguous_definition"
                    else:
                        dst, confidence = next(iter(candidates)), "clang_static_candidate"
            elif row["shadowed"]:
                # locally bound name (param/local/local-import): not a global ref
                dst, confidence = None, "local_shadow"
            elif row["kind"] != "calls":
                dst, confidence = target, "syntax"
            elif target.startswith(("self.", "cls.")) and row["class_scope"]:
                candidate = path + "::" + row["class_scope"] + "." + target.split(".", 1)[1]
                if candidate in symbols:
                    dst, confidence = candidate, "static_candidate"
            elif re.fullmatch(r"\w+(?:\.\w+)*", target):
                first, *rest = target.split(".")
                imported = imports.get(path, {}).get(first)
                if imported:
                    resolved = imported + ("." + ".".join(rest) if rest else "")
                    if resolved.startswith("."):
                        level = len(resolved) - len(resolved.lstrip("."))
                        parent = path.removesuffix(".py").replace("/", ".").split(".")[:-1]
                        parent = parent[: len(parent) - level + 1]
                        resolved = ".".join(parent + [resolved.lstrip(".")])
                    bits = resolved.split(".")
                    for cut in range(len(bits) - 1, 0, -1):
                        candidates = modules.get(".".join(bits[:cut]), [])
                        hits = [
                            p + "::" + ".".join(bits[cut:])
                            for p in candidates
                            if p + "::" + ".".join(bits[cut:]) in symbols
                        ]
                        if len(hits) == 1:
                            dst, confidence = hits[0], "static_candidate"
                            break
                else:
                    scopes = row["src"].split("::", 1)[1].split(".")[:-1]
                    for length in range(len(scopes), -1, -1):
                        candidate = path + "::" + ".".join(scopes[:length] + [target])
                        if candidate in symbols:
                            dst, confidence = candidate, "static_candidate"
                            break
            self.db.execute(
                "UPDATE refs SET dst=?,confidence=? WHERE rowid=?", (dst, confidence, row["rowid"])
            )

    def search(self, query, limit=12):
        if not 1 <= limit <= 100:
            raise ValueError("limit must be in [1,100]")
        cached = getattr(self, "_search_cache", None)
        if cached is None:
            rows = [dict(r) for r in self.db.execute("SELECT * FROM symbols ORDER BY id")]
            docs = [words(r["path"] + " " + r["signature"] + " " + r["doc"]) for r in rows]
            df = Counter(w for doc in docs for w in set(doc))
            avg = sum(map(len, docs)) / max(len(docs), 1)
            cached = (rows, [Counter(doc) for doc in docs], [len(doc) for doc in docs], df, avg)
            self._search_cache = cached
        rows, frequencies, lengths, df, avg = cached
        qualified = {}
        if getattr(self, "qualified_context", False):
            from infinicoder.analysis.qualified_symbols import anchors

            for anchor in anchors(self, query):
                if anchor["status"] == "unique":
                    for resolution in anchor["resolutions"]:
                        qualified[resolution["symbol"]] = anchor["query"]
        query_words = set(words(query))
        identifiers = {
            token.lower()
            for token in re.findall(r"\b[A-Za-z_]\w*\b", query)
            if "_" in token or re.search(r"[a-z][A-Z]", token)
        }
        output = []
        for row, counts, length in zip(rows, frequencies, lengths):
            score = 0
            for word in query_words & counts.keys():
                tf = counts[word]
                idf = math.log(1 + (len(rows) - df[word] + 0.5) / (df[word] + 0.5))
                score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * length / max(avg, 1)))
            # Preserve exact program identifiers as well as natural-language
            # subwords. Long issue descriptions otherwise drown configuration
            # bindings under broad words such as file, memory and settings.
            names = {
                token.lower()
                for token in re.findall(r"\b[A-Za-z_]\w*\b", row["id"].split("::")[-1])
            }
            if identifiers & names:
                score += 20 * len(identifiers & names)
            if row["id"] in qualified:
                score += 100
            if score:
                row = dict(row)
                row["score"] = round(score, 5)
                row["fields"] = json.loads(row["fields"])
                output.append(row)
        return sorted(output, key=lambda r: (-r["score"], r["id"]))[:limit]

    def resolve_qualified(self, name, follow_forwarders=False):
        from infinicoder.analysis.qualified_symbols import resolve

        return resolve(self, name, follow_forwarders=follow_forwarders)

    def symbol(self, symbol_id):
        row = self.db.execute("SELECT * FROM symbols WHERE id=?", (symbol_id,)).fetchone()
        if not row:
            raise KeyError(symbol_id)
        data = dict(row)
        data["fields"] = json.loads(data["fields"])
        if self.native_index_policy == "on_demand" and Path(data["path"]).suffix.lower() in (
            ".c",
            ".cc",
            ".cpp",
            ".cxx",
            ".cu",
            ".h",
            ".hh",
            ".hpp",
            ".cuh",
        ):
            indexed = self.db.execute(
                "SELECT hash,build FROM files WHERE path=?", (data["path"],)
            ).fetchone()
            build = json.loads(indexed["build"] or "{}")
            data["native_analysis"] = dict(
                semantic=bool(build.get("semantic")),
                source_sha256=indexed["hash"],
                fields_known=bool(
                    build.get("semantic") and build.get("native_dependency_complete")
                ),
                rule="Catalog lookup performs no semantic expansion;unexpanded translation units have unknown relationships",
            )
        data["definitions"] = [
            dict(r)
            for r in self.db.execute(
                "SELECT path,line,end,kind FROM definitions WHERE symbol_id=? ORDER BY line",
                (symbol_id,),
            )
        ]
        return data

    def relationships(self, symbol_id, direction="both", limit=50):
        if direction not in ("both", "incoming", "outgoing") or not 1 <= limit <= 200:
            raise ValueError("invalid direction/limit")
        if self.native_index_policy == "on_demand":
            self.symbol(symbol_id)
        # Related fields are included in the structure view, not falsely joined to calls.
        clause = {"both": "src=? OR dst=?", "incoming": "dst=?", "outgoing": "src=?"}[direction]
        params = (symbol_id, symbol_id) if direction == "both" else (symbol_id,)
        return [
            dict(r)
            for r in self.db.execute(
                "SELECT * FROM refs WHERE " + clause + " ORDER BY path,line LIMIT ?",
                params + (limit,),
            )
        ]

    def structure(self, symbol_id):
        result = self.symbol(symbol_id)
        result["field_accesses"] = [
            dict(r)
            for r in self.db.execute(
                'SELECT * FROM refs WHERE kind IN ("reads","writes") AND substr(target,1,?)=? ORDER BY path,line LIMIT 100',
                (len(symbol_id) + 1, symbol_id + "."),
            )
        ]
        return result

    def files(self):
        return [
            dict(r)
            for r in self.db.execute("SELECT path,hash,language,error FROM files ORDER BY path")
        ]

    def text_search(self, query, path_globs=(), regex=False, max_hits=50, max_bytes=20000):
        """bounded literal/regex text search over indexed files.

        Locates configuration assignments, error strings, scripts and other
        non-symbol text without granting shell access. Root-constrained and
        budget-bounded; results carry file hash and truncation flags.
        """
        from fnmatch import fnmatch

        if not isinstance(query, str) or not query:
            raise ValueError("query must be a non-empty string")
        if not isinstance(path_globs, (list, tuple)) or any(
            not isinstance(g, str) or not g for g in path_globs
        ):
            raise ValueError("path_globs must be a list of non-empty strings")
        for g in path_globs:
            # Reject absolute paths and traversal outside the workspace root.
            normalized = g.replace("\\", "/")
            if (
                normalized.startswith(("/", "\\"))
                or normalized.startswith("..")
                or ":" in normalized
            ):
                raise ValueError(f"path_glob must be workspace-relative: {g!r}")
        if not isinstance(max_hits, int) or isinstance(max_hits, bool) or not 1 <= max_hits <= 200:
            raise ValueError("max_hits must be in [1,200]")
        if (
            not isinstance(max_bytes, int)
            or isinstance(max_bytes, bool)
            or not 500 <= max_bytes <= 200000
        ):
            raise ValueError("max_bytes must be in [500,200000]")
        if regex:
            try:
                pattern = re.compile(query)
            except re.error as e:
                raise ValueError(f"invalid regex: {e}")
            matcher = lambda line: pattern.search(line)
        else:
            matcher = lambda line: query in line
        hits, used, truncated = [], 0, False
        for row in self.db.execute("SELECT path,hash,source FROM files ORDER BY path"):
            if path_globs and not any(fnmatch(row["path"], g) for g in path_globs):
                continue
            for lineno, line in enumerate(row["source"].splitlines(), 1):
                if matcher(line):
                    text = line.strip()[:400]
                    entry = {
                        "path": row["path"],
                        "line": lineno,
                        "text": text,
                        "file_hash": row["hash"],
                        "evidence": "text",
                    }
                    cost = len(text) + 80
                    if len(hits) >= max_hits or used + cost > max_bytes:
                        truncated = True
                        break
                    hits.append(entry)
                    used += cost
            if truncated:
                break
        return {
            "hits": hits,
            "truncated": truncated,
            "bytes_used": used,
            "limitation": "Text matches only; not symbol resolution.",
        }

    def directories(self):
        groups = {}
        for row in self.files():
            parent = str(Path(row["path"]).parent).replace("\\", "/")
            item = groups.setdefault(parent, {"path": parent, "files": 0, "languages": {}})
            item["files"] += 1
            item["languages"][row["language"]] = item["languages"].get(row["language"], 0) + 1
        return list(groups.values())

    def dependencies(self, path):
        row = self.db.execute(
            "SELECT imports,error,hash FROM files WHERE path=?", (path,)
        ).fetchone()
        if not row:
            raise KeyError(path)
        return {
            "path": path,
            "hash": row["hash"],
            "imports": json.loads(row["imports"]),
            "parse_error": row["error"],
            "limitation": "Python import syntax; not a build-system dependency graph.",
        }

    def tests_for(self, paths):
        impact = self.impact(paths)
        related = {sid.split("::", 1)[0] for sid in impact["symbol_ids"]}
        names = {Path(p).stem.removeprefix("test_") for p in paths}
        results = []
        for row in self.files():
            path = row["path"]
            if "/tests/" in "/" + path or Path(path).name.startswith("test_"):
                reason = (
                    "static_reference"
                    if path in related
                    else "filename_candidate"
                    if any(n in Path(path).stem for n in names)
                    else None
                )
                if reason:
                    results.append({"path": path, "reason": reason, "coverage_proven": False})
        return results[:50]

    def impact(self, paths, depth=2, limit=60):
        if not 0 <= depth <= 5 or not 1 <= limit <= 200:
            raise ValueError("invalid impact bounds")
        frontier = {
            r[0]
            for p in paths
            for r in self.db.execute("SELECT id FROM symbols WHERE path=?", (p,))
        }
        visited, edges = set(frontier), []
        for _ in range(depth):
            next_frontier = set()
            for sid in sorted(frontier):
                for edge in self.relationships(sid, "incoming"):
                    if len(edges) >= limit:
                        return {"symbol_ids": sorted(visited), "edges": edges, "truncated": True}
                    edges.append(edge)
                    if edge["src"] not in visited:
                        next_frontier.add(edge["src"])
            visited |= next_frontier
            frontier = next_frontier
        return {
            "symbol_ids": sorted(visited),
            "edges": edges,
            "truncated": False,
            "limitation": "Static reference candidates only; dynamic dispatch and external consumers may be absent.",
        }

    def workpack(self, path, start=1, count=100):
        """Versioned source slice with file identity, interfaces and build facts."""
        self.expand_native(path)
        from infinicoder.execution.workspace_tools import Workspace

        source = Workspace(self.root).read(path, start, count)
        row = self.db.execute("SELECT build,hash FROM files WHERE path=?", (path,)).fetchone()
        matches = bool(row and row["hash"] == source["sha256"])
        interfaces = (
            [
                dict(r)
                for r in self.db.execute(
                    "SELECT id,signature,line,end,confidence FROM symbols WHERE path=? ORDER BY line LIMIT 30",
                    (path,),
                )
            ]
            if matches
            else []
        )
        return {
            "schema_version": 1,
            "index_revision": self.revision(),
            "source": source,
            "interfaces": interfaces,
            "build": json.loads(row["build"]) if matches and row["build"] else None,
            "index_source_matches": matches,
            "limitation": "Static candidates only;re-read hashes before edits. Native semantic availability is explicit in build;unexpanded translation units have unknown relationships.",
        }

    def outline(self, path, expected_hash, max_chars=1200):
        """Bounded declaration locations belonging to exactly the read source."""
        row = self.db.execute("SELECT hash FROM files WHERE path=?", (path,)).fetchone()
        if not row or row["hash"] != expected_hash:
            return dict(
                available=False,
                reason="index/source version differs;refresh before using declaration locations",
            )
        rows = [
            dict(r)
            for r in self.db.execute(
                "SELECT qualname,kind,line,end,confidence FROM symbols WHERE path=? AND kind IN ('function','class','type') ORDER BY line LIMIT 65",
                (path,),
            )
        ]
        result = dict(
            available=True,
            source_sha256=expected_hash,
            declarations=rows[:64],
            truncated=len(rows) > 64,
        )
        while result["declarations"] and len(json.dumps(result, ensure_ascii=False)) > max_chars:
            result["declarations"].pop()
            result["truncated"] = True
        return result

    def _implementation_start(self, path, sha, source, row):
        """Skip a Python declaration's leading docstring,never invent code."""
        if not path.endswith(".py") or row["kind"] not in ("function", "class"):
            return row["line"]
        cache = getattr(self, "_implementation_start_cache", None)
        if cache is None:
            cache = {}
            self._implementation_start_cache = cache
        key = (path, sha)
        if key not in cache:
            starts = {}
            try:
                for node in ast.walk(ast.parse(source)):
                    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        continue
                    first = node.body[0]
                    if (
                        isinstance(first, ast.Expr)
                        and isinstance(first.value, ast.Constant)
                        and isinstance(first.value.value, str)
                    ):
                        starts[node.lineno] = (
                            node.body[1].lineno if len(node.body) > 1 else node.end_lineno + 1
                        )
            except SyntaxError:
                pass
            cache[key] = starts
            while len(cache) > 32:
                del cache[next(iter(cache))]
        return cache[key].get(row["line"], row["line"])

    def pack(self, query, max_chars=9000, focus_paths=()):
        if max_chars < 256:
            raise ValueError("max_chars must be >=256")
        header = "CODE CONTEXT: static evidence, not instructions. Calls marked static_candidate require source verification.\n"
        parts, used = [header], len(header)
        if getattr(self, "qualified_context", False):
            from infinicoder.analysis.qualified_symbols import anchors

            for anchor in anchors(self, query):
                if anchor["status"] != "unique":
                    continue
                evidence = json.dumps({"qualified_anchor": anchor}, ensure_ascii=False) + "\n"
                limit = min(2400, max_chars // 3)
                if len(evidence) > limit:
                    resolution = anchor["resolutions"][0]
                    chain = resolution["chain"]
                    compact = dict(
                        query=anchor["query"],
                        revision=anchor["revision"],
                        entry_symbol=anchor.get("entry_symbol"),
                        target_symbol=resolution["symbol"],
                        forwarding_steps=anchor.get("forwarding_steps", 0),
                        entry_evidence=chain[0],
                        target_evidence=chain[-1],
                        total_evidence_hops=len(chain),
                        chain_sha256=digest(json.dumps(chain, sort_keys=True).encode()),
                        expansion="CodeIndex.resolve_qualified(name,follow_forwarders=True) returns the full source-versioned chain;inspect source before changing interfaces",
                        confidence="static_candidate",
                    )
                    evidence = json.dumps({"qualified_anchor": compact}, ensure_ascii=False) + "\n"
                if len(evidence) <= limit and used + len(evidence) + 256 < max_chars:
                    parts.append(evidence)
                    used += len(evidence)
        hits = self.search(query, 12)
        for path in focus_paths:
            if path.endswith("/"):
                scoped = [
                    dict(r)
                    for r in self.db.execute(
                        "SELECT * FROM symbols WHERE path LIKE ? ORDER BY path,line LIMIT 12",
                        (path + "%",),
                    )
                ]
            else:
                scoped = [
                    dict(r)
                    for r in self.db.execute(
                        "SELECT * FROM symbols WHERE path=? ORDER BY line LIMIT 8", (path,)
                    )
                ]
            relevant = [
                hit
                for hit in hits
                if (hit["path"].startswith(path) if path.endswith("/") else hit["path"] == path)
            ]
            hits = relevant + scoped + hits
        seen = set()
        blocks = []
        # Expand one hop only, after ranked hits. This makes dependencies whose
        # names do not occur in the task discoverable without flooding context.
        neighbors = []
        for hit in hits[:6]:
            for edge in self.relationships(hit["id"], limit=8):
                sid = edge["dst"] if edge["src"] == hit["id"] else edge["src"]
                if sid:
                    row = self.db.execute("SELECT * FROM symbols WHERE id=?", (sid,)).fetchone()
                    if row:
                        neighbors.append(dict(row))
        hits += neighbors
        # A bounded route map survives even when no lexical symbol matches.
        # It is derived from indexed facts, never a model-written summary.
        groups = Counter(str(Path(row["path"]).parent).replace("\\", "/") for row in self.files())
        route = {"groups": [], "omitted_groups": len(groups)}
        ranks = {}
        for rank, hit in enumerate(hits):
            ranks.setdefault(str(Path(hit["path"]).parent).replace("\\", "/"), rank)
        group_key = (
            (lambda item: (ranks.get(item[0], 10**6), -item[1], item[0]))
            if getattr(self, "qualified_context", False)
            else (lambda item: (-item[1], item[0]))
        )
        for group, count in sorted(groups.items(), key=group_key):
            candidate = {
                "groups": route["groups"] + [{"path": group, "files": count}],
                "omitted_groups": route["omitted_groups"] - 1,
            }
            if len(json.dumps(candidate, ensure_ascii=False)) > min(1200, max_chars // 4):
                break
            route = candidate
        route_text = json.dumps({"route_map": route}, ensure_ascii=False) + "\n"
        if used + len(route_text) + 80 <= max_chars:
            parts.append(route_text)
            used += len(route_text)
        for row in hits:
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            file = self.db.execute(
                "SELECT hash,source FROM files WHERE path=?", (row["path"],)
            ).fetchone()
            first = self._implementation_start(row["path"], file["hash"], file["source"], row)
            last = min(row["end"], first + 23)
            excerpt = "\n".join(file["source"].splitlines()[first - 1 : last])
            neighbors = self.relationships(row["id"], limit=8)
            block = (
                json.dumps(
                    {
                        "symbol": row["id"],
                        "path": row["path"],
                        "line": row["line"],
                        "file_hash": file["hash"],
                        "signature": row["signature"],
                        "confidence": row["confidence"],
                        "excerpt": excerpt,
                        "excerpt_line": first,
                        "excerpt_end": last,
                        "leading_docstring_omitted": first != row["line"],
                        "implementation_excerpt_complete": last == row["end"],
                        "relationships": neighbors,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            blocks.append(block)
        # Include paths and parse diagnostics even when no symbol matches (e.g. C++ parse gaps).
        for row in self.files():
            if row["error"] or set(words(query)) & set(words(row["path"])):
                blocks.append(json.dumps(row, ensure_ascii=False) + "\n")
        omitted = 0
        for block in blocks:
            if used + len(block) + 80 <= max_chars:
                parts.append(block)
                used += len(block)
            else:
                omitted += 1
        parts.append(f"OMITTED_BLOCKS={omitted}; use search/symbol/read tools for details.\n")
        return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--database", required=True)
    ap.add_argument(
        "view",
        choices=[
            "files",
            "directories",
            "dependencies",
            "tests",
            "search",
            "symbol",
            "structure",
            "incoming",
            "outgoing",
            "pack",
            "impact",
            "memory-route",
            "memory-expand",
            "use-def",
        ],
    )
    ap.add_argument("query", nargs="?", default="")
    ap.add_argument("--max-chars", type=int, default=9000)
    args = ap.parse_args()
    with CodeIndex(args.root, args.database) as idx:
        idx.refresh()
        if args.view == "files":
            result = idx.files()
        elif args.view == "directories":
            result = idx.directories()
        elif args.view == "dependencies":
            result = idx.dependencies(args.query)
        elif args.view == "tests":
            result = idx.tests_for([args.query])
        elif args.view == "search":
            result = idx.search(args.query)
        elif args.view in ("symbol", "structure"):
            result = getattr(idx, args.view)(args.query)
        elif args.view == "impact":
            result = idx.impact([args.query])
        elif args.view == "pack":
            result = idx.pack(args.query, args.max_chars)
        elif args.view == "memory-route":
            result = idx.memory_route(args.query)
        elif args.view == "memory-expand":
            result = idx.memory_expand(args.query or ".")
        elif args.view == "use-def":
            result = idx.use_def(args.query)
        else:
            result = idx.relationships(args.query, args.view)
        print(
            result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, indent=2)
        )


if __name__ == "__main__":
    main()
