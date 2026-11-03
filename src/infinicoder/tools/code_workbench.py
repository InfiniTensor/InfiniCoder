"""On-demand current source observations and explicit versioned local edits.

This service reuses the caller's index and permissions. It has no model client,
task classifier,workflow loop,project import or correctness oracle.
"""

import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat

from infinicoder.analysis.function_body import edit_function_body
from infinicoder.analysis.symbol_tools import _definitions


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class CodeWorkbench:
    def __init__(
        self,
        index,
        workspace,
        *,
        count_output,
        remaining,
        output_limit=1600,
        scratch=None,
        record_event=None,
        refresh_index=None,
        page_count_limit=None,
        source_line_limit=120,
        expand_native=True,
    ):
        self.index, self.workspace = index, workspace
        self.count_output, self.remaining = count_output, remaining
        self.output_limit = output_limit
        self.scratch = Path(scratch) if scratch is not None else None
        self.record_event = record_event
        self.refresh_index, self.page_count_limit = refresh_index, page_count_limit
        self.source_line_limit, self.expand_native = source_line_limit, expand_native
        self.trials = None
        self.trial_output_limit = output_limit
        self.index.db.execute(
            "CREATE TABLE IF NOT EXISTS code_workbench_events(id INTEGER PRIMARY KEY,event TEXT)"
        )
        self.index.db.commit()

    def _event(self, kind, value):
        with self.index.db:
            self.index.db.execute(
                "INSERT INTO code_workbench_events(event) VALUES(?)",
                (_json(dict(kind=kind, **value)),),
            )
        if self.record_event is not None:
            self.record_event(kind, value)

    def _read(self, path, *, max_bytes=8 * 1024**2):
        if type(max_bytes) is not int or not 0 < max_bytes <= 8 * 1024**2:
            raise ValueError("positive bounded source byte limit required")
        self.remaining()
        file = self.workspace.path(path)
        if any(p.is_symlink() for p in (file, *file.parents)) or not file.is_file():
            raise ValueError("Current regular source without linked components required")
        fd = os.open(
            file, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
                raise ValueError("Source exceeds view bound;use ordinary reading")
            data = stream.read(max_bytes + 1)
            after = os.fstat(stream.fileno())
        # As in NativeEpisodeMemory,the same-handle change time is compared
        # separately. Windows path/fstat expose different ctime conventions.
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
        if (
            identity(before) != identity(after)
            or before.st_ctime_ns != after.st_ctime_ns
            or identity(after) != identity(file.lstat())
            or len(data) != after.st_size
        ):
            raise ValueError("Source changed while reading;inspect again")
        return data, data.decode("utf-8-sig"), _sha(data)

    def _page(self, kind, selector, records, cursor=None, _output_limit=None, **metadata):
        self.remaining()
        output_limit = (
            self.output_limit if _output_limit is None else min(self.output_limit, _output_limit)
        )
        fingerprint = _sha(_json(dict(kind=kind, selector=selector, records=records)).encode())
        start = 0
        if cursor:
            match = re.fullmatch(r"([0-9a-f]{64}):(\d+)", cursor)
            if not match or match[1] != fingerprint:
                raise ValueError("Stale or invalid page cursor;repeat the original view")
            start = int(match[2])
        if not 0 <= start <= len(records):
            raise ValueError("Page cursor outside view")

        def packet(selected, next_index):
            versions = {
                r["path"]: r["source_sha256"]
                for r in selected
                if "path" in r and "source_sha256" in r
            }
            for record in selected:
                versions.update(record.get("support_sources", {}))
            return dict(
                view=kind,
                selector=selector,
                records=selected,
                source_files=versions,
                returned_from=start,
                records_total=len(records),
                omitted_before=start,
                omitted_after=len(records) - next_index,
                next_cursor=fingerprint + ":" + str(next_index)
                if next_index < len(records)
                else None,
                facts=(
                    "Experiment observations;not acceptance"
                    if kind.startswith("candidate_trial")
                    else "Current source/observations;static candidates are not runtime dispatch or correctness proof"
                ),
                **(
                    {"read_event_ref": self.read_event_ref}
                    if getattr(self, "read_event_ref", None)
                    else {}
                ),
                **metadata,
            )

        if self.page_count_limit is not None:
            from infinicoder.memory.source.source_packet_budget import bounded_prefix

            value = bounded_prefix(
                records[start:],
                lambda chosen, length: packet(chosen, start + length),
                self.count_output,
                output_limit,
                count_limit=self.page_count_limit,
            )
            if value is None:
                raise ValueError("View metadata exceeds token budget;use a narrower query/range")
            for path, expected in value["source_files"].items():
                if self._read(path)[2] != expected:
                    raise ValueError("Source changed during view;inspect again")
            self._event(
                "source_view",
                dict(
                    view=kind,
                    selector=selector,
                    source_files=value["source_files"],
                    records=len(value["records"]),
                    next_cursor=value["next_cursor"],
                ),
            )
            return value
        chosen = []
        stop = start
        empty = packet([], start)
        if self.count_output(_json(empty)) > output_limit:
            raise ValueError("View metadata exceeds token budget;use a narrower query/range")
        while stop < len(records):
            self.remaining()
            trial = packet(chosen + [records[stop]], stop + 1)
            if self.count_output(_json(trial)) > output_limit:
                break
            chosen.append(records[stop])
            stop += 1
        if stop == start and start < len(records):
            record = records[start]
            chosen = [
                dict(
                    kind="record_not_displayed",
                    path=record.get("path"),
                    line=record.get("line"),
                    source_sha256=record.get("source_sha256"),
                    reason="One complete record exceeds tool budget;use ordinary read or a narrower view",
                    original_record_sha256=_sha(_json(record).encode()),
                    source_complete=False,
                )
            ]
            stop += 1
        value = packet(chosen, stop)
        if self.count_output(_json(value)) > output_limit:
            raise ValueError(
                "Record omission metadata exceeds token budget;ordinary reading remains available"
            )
        for path, expected in value["source_files"].items():
            if self._read(path)[2] != expected:
                raise ValueError("Source changed during view;inspect again")
        self._event(
            "source_view",
            dict(
                view=kind,
                selector=selector,
                source_files=value["source_files"],
                records=len(chosen),
                next_cursor=value["next_cursor"],
            ),
        )
        return value

    def locate(self, query, *, path_globs=(), cursor=None):
        if not isinstance(query, str) or not query or len(query) > 2000:
            raise ValueError("A bounded literal query or symbol name is required")
        self._refresh()
        records = []
        from fnmatch import fnmatch

        for symbol in self.index.search(query, limit=24):
            if path_globs and not any(fnmatch(symbol["path"], p) for p in path_globs):
                continue
            row = self.index.db.execute(
                "SELECT hash FROM files WHERE path=?", (symbol["path"],)
            ).fetchone()
            records.append(
                dict(
                    kind="symbol_candidate",
                    path=symbol["path"],
                    source_sha256=row["hash"],
                    symbol={
                        k: symbol[k]
                        for k in ("id", "name", "qualname", "kind", "line", "end", "score")
                        if k in symbol
                    },
                    binding="Search candidate,not unique runtime target or edit permission",
                )
            )
        text = self.index.text_search(query, path_globs=path_globs, max_hits=80, max_bytes=40000)
        for hit in text["hits"]:
            records.append(
                dict(
                    kind="text_match_preview",
                    path=hit["path"],
                    line=hit["line"],
                    source_sha256=hit["file_hash"],
                    preview=hit["text"],
                    source_complete=False,
                    next="inspect this path and line range",
                )
            )
        return self._page(
            "locate",
            dict(query=query, path_globs=list(path_globs)),
            records,
            cursor,
            candidate_limit=24,
            text_search_truncated=text["truncated"],
            unique_dispatch_proved=False,
        )

    def inspect(self, target, *, start=None, end=None, view="context", cursor=None):
        if view not in ("context", "source", "module", "relations", "flow"):
            raise ValueError("view must be context/source/module/relations/flow")
        self._refresh(target)
        symbol = None
        if "::" in target:
            symbol = self.index.symbol(target)
            path = symbol["path"]
            # Existing native expansion has its own parser process and remaining
            # time bound. Failure remains explicit text/unknown,not a guessed AST.
            if self.expand_native and Path(path).suffix.lower() not in (".py", ".pyi"):
                self.index.expand_native(path)
                symbol = self.index.symbol(target)
        else:
            path = target
        data, source, digest = self._read(path)
        lines = source.splitlines()
        indexed = self.index.db.execute("SELECT hash FROM files WHERE path=?", (path,)).fetchone()
        if indexed is not None and indexed["hash"] != digest:
            raise ValueError("Source changed after index refresh;inspect again")
        first = start if start is not None else symbol["line"] if symbol else 1
        last = (
            end
            if end is not None
            else min(symbol["end"] if symbol else len(lines), first + self.source_line_limit - 1)
        )
        if type(first) is not int or type(last) is not int or not 1 <= first <= last <= len(lines):
            raise ValueError("Explicit existing1-based source range required")
        if last - first > 999:
            raise ValueError("At most1000 requested lines per view;expand with another range")
        records = []

        def add(kind, **record):
            records.append(dict(kind=kind, path=path, source_sha256=digest, **record))

        if view in ("context", "source", "module"):
            for number in range(first, last + 1):
                add("source_line", line=number, source=lines[number - 1])
        limitation = "Text range;no complete dependency or dynamic dispatch claim"
        if Path(path).suffix.lower() in (".py", ".pyi"):
            try:
                tree = ast.parse(source, filename=path)
            except (SyntaxError, ValueError, RecursionError):
                tree = None
            if tree is not None:
                definitions = _definitions(tree)
                matches = [
                    node
                    for name, node in definitions
                    if symbol and name == target.split("::", 1)[1]
                ]
                selected = matches[0] if len(matches) == 1 else None
                if len(matches) > 1:
                    add(
                        "ambiguous_definition",
                        matching_definitions=len(matches),
                        binding="Text view only;no unique AST owner selected",
                    )
                if view in ("context", "module"):
                    referenced = (
                        {n.id for n in ast.walk(selected) if isinstance(n, ast.Name)}
                        if selected is not None
                        else None
                    )
                    for node in tree.body:
                        if isinstance(
                            node,
                            (
                                ast.Import,
                                ast.ImportFrom,
                                ast.Assign,
                                ast.AnnAssign,
                                ast.AugAssign,
                                ast.Expr,
                            ),
                        ):
                            if referenced is not None and view == "context":
                                bound = (
                                    {a.asname or a.name.split(".")[0] for a in node.names}
                                    if isinstance(node, (ast.Import, ast.ImportFrom))
                                    else {
                                        n.id
                                        for n in ast.walk(node)
                                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                                    }
                                )
                                if (
                                    not referenced & bound
                                    and "*" not in bound
                                    and not (
                                        isinstance(node, ast.ImportFrom)
                                        and node.module == "__future__"
                                    )
                                ):
                                    continue
                            add(
                                "module_statement",
                                line=node.lineno,
                                end=node.end_lineno,
                                node_type=type(node).__name__,
                                preview=lines[node.lineno - 1],
                                source_complete=node.lineno == node.end_lineno,
                                interpretation="Syntactic module statement;execution/effect not established",
                            )
                if selected is not None and view in ("context", "relations"):
                    qualname = target.split("::", 1)[1]
                    owners = [
                        (name, node)
                        for name, node in definitions
                        if isinstance(node, ast.ClassDef) and qualname.startswith(name + ".")
                    ]
                    owner = max(owners, key=lambda pair: len(pair[0])) if owners else None
                    if owner:
                        name, cls = owner
                        add(
                            "owner_class",
                            line=cls.lineno,
                            owner=name,
                            bases=[ast.unparse(b) for b in cls.bases],
                            interpretation="Declared lexical class/base expressions;runtime type and initialization completeness unknown",
                        )
                        for child in cls.body:
                            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                add(
                                    "owner_method_candidate",
                                    line=child.lineno,
                                    end=child.end_lineno,
                                    symbol=path + "::" + name + "." + child.name,
                                    dispatch="Same-class declaration candidate;subclasses/rebinding/aliases unresolved",
                                )
                    parents = {
                        child: node
                        for node in ast.walk(selected)
                        for child in ast.iter_child_nodes(node)
                    }
                    for node in ast.walk(selected):
                        if (
                            isinstance(node, ast.Attribute)
                            and isinstance(node.value, ast.Name)
                            and node.value.id in ("self", "cls")
                        ):
                            access = (
                                "write"
                                if isinstance(node.ctx, ast.Store)
                                else "delete"
                                if isinstance(node.ctx, ast.Del)
                                else "read"
                            )
                            if isinstance(parents.get(node), ast.AugAssign):
                                access = "read_write"
                            add(
                                "receiver_access",
                                line=node.lineno,
                                attribute=node.attr,
                                access=access,
                                owner=owner[0] if owner else None,
                                interpretation="Lexical receiver access;aliases,values and lifecycle ownership unknown",
                            )
                        if isinstance(node, ast.Call):
                            add(
                                "call_expression",
                                line=node.lineno,
                                expression=ast.unparse(node.func),
                                binding="Syntax only;not unique dispatch",
                            )
                if selected is not None and view in ("relations", "flow"):
                    flow = self.index.use_def(path, scope=target.split("::", 1)[1], limit=64)
                    if flow["source_sha256"] != digest:
                        raise ValueError("Dataflow source changed;inspect again")
                    for edge in flow["edges"]:
                        add("use_def_candidate", edge=edge, limitations=flow["limitations"])
                    for edge in self.index.relationships(target, limit=32):
                        add(
                            "static_relationship_candidate",
                            edge=edge,
                            support_sources=self._relationship_sources(edge),
                        )
                    add(
                        "analysis_boundary",
                        omitted_use_def=flow.get("omitted", 0),
                        relationship_limit=32,
                        limitations=flow["limitations"],
                    )
                limitation = "Python AST and declared source relations;dynamic dispatch/external state/lifecycle semantics unknown"
        else:
            row = self.index.db.execute("SELECT build FROM files WHERE path=?", (path,)).fetchone()
            build = json.loads(row["build"] or "{}") if row else {}
            add(
                "native_analysis_boundary",
                semantic=bool(build.get("semantic")),
                dependency_complete=bool(build.get("native_dependency_complete")),
                limitations=build.get(
                    "limitation",
                    "On-demand Clang metadata unavailable;only text/catalog candidates",
                ),
            )
            if symbol and view in ("context", "relations", "flow"):
                for edge in self.index.relationships(target, limit=32):
                    add(
                        "native_relationship_candidate",
                        edge=edge,
                        support_sources=self._relationship_sources(edge),
                    )
            limitation = "Native semantic facts only where parser/compilation metadata reports support;otherwise text/unknown"
        if view == "context":
            order = {
                "owner_class": 0,
                "module_statement": 1,
                "ambiguous_definition": 1,
                "source_line": 2,
                "receiver_access": 3,
                "owner_method_candidate": 4,
            }
            records.sort(key=lambda r: order.get(r["kind"], 5))
        return self._page(
            "inspect",
            dict(target=target, start=first, end=last, view=view),
            records,
            cursor,
            requested_source_range=[first, last],
            total_source_lines=len(lines),
            analysis_limits=limitation,
            outside_requested_range=max(0, len(lines) - (last - first + 1)),
        )

    def _refresh(self, target=None):
        if self.refresh_index is None:
            return self.index.refresh()
        return self.refresh_index(target)

    def _relationship_sources(self, edge):
        paths = {edge.get("path")}
        for key in ("src", "dst"):
            if isinstance(edge.get(key), str) and "::" in edge[key]:
                paths.add(edge[key].split("::", 1)[0])
        result = {}
        for path in paths:
            if not path:
                continue
            row = self.index.db.execute("SELECT hash FROM files WHERE path=?", (path,)).fetchone()
            if row:
                result[path] = row["hash"]
        return result

    def _payload(self, value):
        if self.scratch is None:
            raise ValueError("Caller has no restricted scratch mapping;use ordinary editing")
        if not isinstance(value, str):
            raise ValueError("Payload path must be text")
        path = PurePosixPath(value)
        if (
            not isinstance(value, str)
            or "\\" in value
            or "\0" in value
            or ".." in path.parts
            or not path.is_absolute()
            or str(path) != value
            or path.parts[:3] != ("/", "tmp", "task-scratch")
            or len(path.parts) < 4
        ):
            raise ValueError("Payload must be a normalized file below /tmp/task-scratch")
        root = self.scratch.resolve()
        file = root.joinpath(*path.parts[3:])
        if (
            self.scratch.is_symlink()
            or any(p.is_symlink() for p in (file, *file.parents))
            or not file.resolve().is_relative_to(root)
        ):
            raise ValueError("Linked or escaped scratch payload rejected")
        if not stat.S_ISREG(file.lstat().st_mode):
            raise ValueError(
                "Payload must be a regular file;FIFO/device/directory rejected before opening"
            )
        # A model can create a FIFO with an ordinary shell action. Nonblocking
        # open also closes the lstat/open replacement race;fstat below verifies
        # the same handle before any read. No host tool may wait for a writer.
        fd = os.open(
            file, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > 1_000_000:
                raise ValueError(
                    "Payload must be a complete regular file <=1000000bytes;no prefix applied"
                )
            data = stream.read(1_000_001)
            after = os.fstat(stream.fileno())
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
        if (
            identity(before) != identity(after)
            or before.st_ctime_ns != after.st_ctime_ns
            or identity(after) != identity(file.lstat())
            or len(data) != after.st_size
        ):
            raise ValueError("Payload changed/truncated during read;no edit applied")
        self.remaining()
        return data.decode("utf8"), dict(
            payload_sha256=_sha(data), payload_bytes=len(data), payload_path=value
        )

    def edit_body(self, symbol_id, expected_hash, body_file):
        body, provenance = self._payload(body_file)
        self.remaining()
        result = edit_function_body(self.workspace, symbol_id, expected_hash, body)
        value = dict(operation="edit-body", **result, **provenance)
        self._event("source_edit", value)
        self.index.refresh()
        return value

    def edit_range(self, path, expected_hash, start, end, new_file):
        if not re.fullmatch("[0-9a-f]{64}", expected_hash):
            raise ValueError("Copy current expected source hash")
        replacement, provenance = self._payload(new_file)
        data, source, digest = self._read(path)
        if digest != expected_hash:
            raise ValueError("File changed;inspect again")
        if (
            type(start) is not int
            or type(end) is not int
            or not 1 <= start <= end <= len(source.splitlines(True))
        ):
            raise ValueError("Explicit existing1-based line range required")
        syntax = "not_checked;range text has no interface or native semantic guarantee"
        if Path(path).suffix.lower() in (".py", ".pyi"):
            lines = source.splitlines(True)
            text = (
                replacement if not replacement or replacement.endswith("\n") else replacement + "\n"
            )
            candidate = "".join(lines[: start - 1]) + text + "".join(lines[end:])
            compile(candidate, path, "exec", dont_inherit=True)
            syntax = "Python syntax only;no interface/behavior guarantee"
        result = self.workspace.edit(path, expected_hash, new=replacement, start=start, end=end)
        value = dict(
            operation="edit-range",
            **result,
            **provenance,
            range=[start, end],
            syntax=syntax,
            acceptance_proof=False,
            requires_independent_verification=True,
        )
        self._event("source_edit", value)
        self.index.refresh()
        return value

    def trial(self, edit_script, probe_script):
        if self.trials is None:
            raise ValueError("candidate trials are not enabled by this caller/profile")
        result = self.trials.trial(edit_script, probe_script)
        # Initial feedback is a complete small outcome. The model can retrieve
        # detailed bindings/observations explicitly through trial-show.
        records = (
            self.trials.records(result["trial_id"])[:1]
            if result.get("trial_id")
            else [dict(kind="trial_unavailable", **result)]
        )
        return self._page(
            "candidate_trial",
            dict(trial_id=result.get("trial_id")),
            records,
            _output_limit=self.trial_output_limit,
            evidence_authority="model_authored_experiment;not_independent_acceptance",
        )

    def trial_show(self, identity, *, part="result", cursor=None):
        if self.trials is None:
            raise ValueError("candidate trials are not enabled by this caller/profile")
        return self._page(
            "candidate_trial_" + part,
            dict(trial_id=identity, part=part),
            self.trials.records(identity, part),
            cursor,
            _output_limit=self.trial_output_limit,
            evidence_authority="version_bound_experiment_observation;not_current_acceptance",
        )

    def trial_apply(self, identity):
        if self.trials is None:
            raise ValueError("candidate trials are not enabled by this caller/profile")
        result = self.trials.apply(identity)
        self.index.refresh()
        return result
