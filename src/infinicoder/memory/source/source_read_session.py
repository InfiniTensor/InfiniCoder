"""One bounded source-read session; catalog coverage never grants edit scope."""

from collections import Counter, deque
import os
from pathlib import Path
import re
import time

from infinicoder.analysis.code_context import IGNORED, LANGUAGES, words
from infinicoder.memory.source.task_source_roots import (
    PATH,
    relative_public_path,
    resolve_public_roots,
)


class SourceReadSession:
    def __init__(self, index, workbench, *, workspace_aliases):
        self.index, self.workbench = index, workbench
        self.workspace_aliases = workspace_aliases
        self.remaining = workbench.remaining
        self._raw_read = workbench._read
        self._cache = None
        self._paths = None
        self.public_paths = []
        self.public_task = ""
        self.stats = Counter()
        self.events = []
        self.inventory_truncated = False
        workbench._read = self.read
        workbench.refresh_index = self.refresh_target
        workbench.page_count_limit = 4
        workbench.source_line_limit = 80
        workbench.expand_native = False

    def read(self, path, *, max_bytes=1_000_000):
        self.remaining()
        if self._cache is not None and path in self._cache:
            return self._cache[path]
        result = self._raw_read(path, max_bytes=min(max_bytes, 1_000_000))
        self.stats["host_source_reads"] += 1
        self.stats["host_source_bytes"] += len(result[0])
        if self._cache is not None:
            self._cache[path] = result
        return result

    def refresh_target(self, target=None):
        if target is None:
            raise ValueError("source session requires an explicit bounded query")
        path = target.split("::", 1)[0]
        return self.refresh_paths((path,))

    def refresh_paths(self, paths):
        started = time.monotonic()
        result = self.index.refresh_paths(paths, read_source=self.read)
        self.stats["indexed_path_updates"] += len(result["changed"])
        self.events.append(
            dict(
                phase="refresh_selected",
                paths=list(paths),
                started_monotonic=started,
                wall_seconds=time.monotonic() - started,
            )
        )
        return result

    def inventory(self):
        if self._paths is not None:
            return self._paths
        paths = []
        folders = 0
        entries = 0
        excluded = getattr(self.index, "excluded_paths", ())
        pending = deque([self.index.root])
        while pending:
            folder = pending.popleft()
            self.remaining()
            folders += 1
            try:
                children = sorted(folder.iterdir(), key=lambda path: path.name)
            except OSError:
                continue
            dirs = [
                p
                for p in children
                if not p.is_symlink()
                and p.is_dir()
                and p.name not in IGNORED
                and not any(p.resolve().is_relative_to(q) for q in excluded)
            ]
            names = [p.name for p in children if not p.is_dir()]
            pending.extend(dirs)
            entries += len(dirs) + len(names)
            for name in sorted(names):
                path = Path(folder, name)
                if (
                    path.suffix.lower() not in LANGUAGES
                    or path.is_symlink()
                    or not path.is_file()
                    or path.resolve() == self.index.database
                    or any(path.resolve().is_relative_to(p) for p in excluded)
                ):
                    continue
                paths.append(path.relative_to(self.index.root).as_posix())
                if len(paths) >= 4096:
                    break
            if len(paths) >= 4096 or folders >= 512 or entries >= 20000:
                self.inventory_truncated = True
                break
        self._paths = tuple(paths)
        self.stats["catalog_files"] = len(paths)
        self.stats["catalog_folders"] = folders
        return self._paths

    def public_module_paths(self, public_task):
        """Probe literal dotted-module suffixes before bounded discovery.

        A filesystem match is a reading alternative, not Python import binding.
        This also supports a public name which omits the outer source package.
        """
        hints = list(
            dict.fromkeys(re.findall(r"(?<![\w.])[A-Za-z_]\w*(?:\.[A-Za-z_]\w*){1,6}", public_task))
        )[:16]
        roots = [self.index.root]
        for path in sorted(self.index.root.iterdir(), key=lambda path: path.name):
            if len(roots) >= 65:
                break
            if path.name not in IGNORED and not path.is_symlink() and path.is_dir():
                roots.append(path)
        candidates = []
        for hint in hints:
            parts = hint.split(".")
            for length in range(len(parts), 0, -1):
                for root in roots:
                    stem = root.joinpath(*parts[:length])
                    for file in (stem.with_suffix(".py"), stem / "__init__.py"):
                        self.remaining()
                        if file.is_symlink() or not file.is_file():
                            continue
                        path = file.relative_to(self.index.root).as_posix()
                        try:
                            self.workbench.workspace.path(path)
                        except ValueError:
                            continue
                        if not any(row["path"] == path for row in candidates):
                            candidates.append(
                                dict(
                                    path=path,
                                    public_anchor=hint,
                                    kind="public_module_path_candidate",
                                    selection="literal_public_module_suffix;filesystem_candidate",
                                    status="reading_alternative;runtime_import_binding_unknown",
                                )
                            )
                if len(candidates) >= 8:
                    return candidates[:8]
        return candidates

    def seed(self, public_task):
        started = time.monotonic()
        self.public_task = public_task
        paths = self.inventory()
        known = set(paths)
        selected = []
        for match in PATH.finditer(public_task):
            path = relative_public_path(match[1], self.workspace_aliases)
            if path and path not in selected:
                # Explicit paths need no complete catalog or repository parse.
                try:
                    file = self.workbench.workspace.path(path)
                except ValueError:
                    continue
                if path in known or (not file.is_symlink() and file.is_file()):
                    selected.append(path)
            if len(selected) == 4:
                break
        self.public_paths = selected
        modules = self.public_module_paths(public_task)
        selected = list(dict.fromkeys(selected + [row["path"] for row in modules]))[:4]
        self.public_paths = selected
        if selected:
            self.refresh_paths(selected)
        roots = resolve_public_roots(
            self.index, public_task, workspace_aliases=self.workspace_aliases
        )
        for candidate in modules:
            if candidate["path"] in selected and not any(
                row["path"] == candidate["path"] for row in roots["roots"]
            ):
                roots["roots"].append(candidate)
        roots["roots"] = roots["roots"][:4]
        declarations = []
        for root in roots["roots"]:
            for row in self.index.db.execute(
                "SELECT id,kind,line,end,signature FROM symbols WHERE path=? ORDER BY line LIMIT 8",
                (root["path"],),
            ):
                declarations.append(dict(row))
        result = dict(
            public_roots=roots,
            declarations=declarations,
            directory_candidates=list(paths[:32]),
            other_files_omitted=max(0, len(paths) - 32),
            inventory_truncated=self.inventory_truncated,
            source_coverage="Only explicitly observed files are indexed;directory entries are reading hints",
            index_revision=self.index.revision(),
            interpretation="Partial catalog and static candidates,not runtime binding or permission to edit",
        )
        self.events.append(
            dict(phase="seed", wall_seconds=time.monotonic() - started, observed_paths=selected)
        )
        return result

    def _ordered_paths(self, query):
        tokens = set(words(query))

        def score(path):
            basename = words(Path(path).stem)
            prefix = max(
                (len(os.path.commonprefix((a, b))) for a in tokens for b in basename), default=0
            )
            return (
                (100 if path in self.public_paths else 0)
                + (20 if tokens.intersection(basename) else 0)
                + min(12, prefix if prefix >= 4 else 0)
            )

        # Existing static candidates are only hints until their bytes are read.
        names = re.findall(r"[A-Za-z_]\w*", query)
        lookup = names[-1] if names else query
        warm = [
            r[0]
            for r in self.index.db.execute(
                "SELECT DISTINCT path FROM symbols WHERE instr(name,?)>0 OR instr(qualname,?)>0 ORDER BY path LIMIT 24",
                (lookup, lookup),
            )
        ]
        paths = sorted(
            self.inventory(),
            key=lambda path: (
                -score(path),
                Path(path).suffix.lower()
                not in (
                    ".py",
                    ".pyi",
                    ".c",
                    ".cc",
                    ".cpp",
                    ".cu",
                    ".h",
                    ".hpp",
                    ".rs",
                    ".js",
                    ".ts",
                ),
                path,
            ),
        )
        return tuple(dict.fromkeys(warm + paths))

    def locate(self, query):
        if not isinstance(query, str) or not 1 <= len(query) <= 300:
            raise ValueError("bounded literal query required")
        hits = []
        scanned = []
        used = 0
        omitted = 0
        ordered = self._ordered_paths(query)
        for path in ordered:
            self.remaining()
            if len(scanned) >= 256 or used >= 2_000_000:
                break
            try:
                file = self.workbench.workspace.path(path)
                if file.stat().st_size > min(1_000_000, 2_000_000 - used):
                    omitted += 1
                    continue
                raw, source, digest = self.read(path)
            except (OSError, ValueError, UnicodeError):
                omitted += 1
                continue
            used += len(raw)
            scanned.append(path)
            matches = [
                dict(
                    kind="text_match_preview",
                    path=path,
                    line=line,
                    source_sha256=digest,
                    preview=text[:400],
                    source_complete=False,
                    next="Inspect this candidate;literal match is not runtime binding",
                )
                for line, text in enumerate(source.splitlines(), 1)
                if query in text
            ]
            hits.extend(matches[:8])
            if len(hits) >= 24:
                break
        selected = list(dict.fromkeys(hit["path"] for hit in hits))[:8]
        if selected:
            self.refresh_paths(selected)
        records = []
        for path in selected:
            for row in self.index.db.execute(
                "SELECT id,name,qualname,line,end,kind FROM symbols WHERE path=? ORDER BY line",
                (path,),
            ):
                if row["name"] in query or query in row["qualname"]:
                    records.append(
                        dict(
                            kind="symbol_candidate",
                            path=path,
                            source_sha256=self.read(path)[2],
                            symbol=dict(row),
                            binding="Static candidate in a partial scan;runtime dispatch unknown",
                        )
                    )
        records.extend(hits)
        # Only a unique file hit receives a small exact source excerpt. Other
        # matches remain alternatives, with no guessed function/dispatch owner.
        if len(selected) == 1:
            path = selected[0]
            source = self.read(path)[1].splitlines()
            digest = self.read(path)[2]
            line = hits[0]["line"]
            excerpt = [
                dict(
                    kind="source_line",
                    path=path,
                    line=i,
                    source=source[i - 1],
                    source_sha256=digest,
                )
                for i in range(max(1, line - 2), min(len(source), line + 5) + 1)
            ]
            records = excerpt + records
        self.stats["literal_scan_bytes"] += used
        self.stats["literal_scan_files"] += len(scanned)
        return self.workbench._page(
            "locate",
            dict(query=query),
            records,
            scanned_files=len(scanned),
            scanned_bytes=used,
            unscanned_or_skipped=max(0, len(ordered) - len(scanned)),
            unreadable_or_oversized=omitted,
            inventory_truncated=self.inventory_truncated,
            source_coverage="Bounded partial literal search;no match does not establish absence",
            unique_dispatch_proved=False,
        )

    def query(self, item):
        started = time.monotonic()
        self._cache = {}
        try:
            if item["op"] == "locate":
                result = self.locate(item["query"])
            else:
                result = self.workbench.inspect(
                    item["target"],
                    start=item.get("start"),
                    end=item.get("end"),
                    view=item.get("view", "source"),
                )
            # Revalidate only files actually present in this returned view.
            for path, digest in result["source_files"].items():
                raw = self._raw_read(path)
                self.stats["host_source_reads"] += 1
                self.stats["host_source_bytes"] += len(raw[0])
                if raw[2] != digest:
                    raise ValueError("Source changed during read transaction")
            return result
        finally:
            self._cache = None
            self.events.append(
                dict(phase="query", query=item, wall_seconds=time.monotonic() - started)
            )
