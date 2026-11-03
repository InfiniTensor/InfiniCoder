"""Bounded read-only candidate facts and explicitly historical initial bytes.

No project code, check, shell command or model executes here. Public reading
targets never imply write scope. Old source_archive.recall remains current-only.
"""

import ast
import difflib
import hashlib
from pathlib import Path
import time

from infinicoder.analysis.function_body import _layout, _shell
from infinicoder.memory.native_memory import _json
from infinicoder.memory.source.source_archive import _literal
from infinicoder.analysis.symbol_tools import _definitions
from infinicoder.memory.source.task_source_roots import relative_public_path, resolve_public_roots


class DiagnosticTimeout(Exception):
    """Local diagnostic allowance expired; ordinary task may continue."""


class HistoricalSources:
    """Initial unfinished caller-declared files, never a correct reference."""

    def __init__(self, folder, reader, paths, *, forbidden=(), remaining=None):
        self.folder = Path(folder)
        self.reader = reader
        self.remaining = remaining
        self.entries = {}
        self.folder.mkdir(exist_ok=False)
        forbidden = tuple(forbidden)
        total = 0
        for ordinal, path in enumerate(paths):
            if remaining is not None:
                remaining()
            reason = None
            if ordinal >= 8:
                reason = "file_count_bound"
            elif any(path == p or path.startswith(p.rstrip("/") + "/") for p in forbidden):
                reason = "protected_or_excluded"
            if reason:
                self.entries[path] = dict(status="unavailable", reason=reason)
                continue
            try:
                raw, _, digest = reader(path, max_bytes=1024**2)
                if total + len(raw) > 4 * 1024**2:
                    raise ValueError("total_initial_bytes_bound")
                target = self.folder / (digest + ".source")
                if target.exists():
                    if target.read_bytes() != raw:
                        raise ValueError("historical_content_identity_mismatch")
                else:
                    target.write_bytes(raw)
                total += len(raw)
                self.entries[path] = dict(
                    status="complete",
                    captured_revision=digest,
                    content_sha256=digest,
                    bytes=len(raw),
                    captured_unix_ns=time.time_ns(),
                    start=1,
                    end=len(raw.decode("utf-8-sig").splitlines()),
                    historical=True,
                    origin="actual_initial_unfinished_implementation;not_verified_reference",
                )
            except (OSError, ValueError, UnicodeError) as error:
                self.entries[path] = dict(
                    status="unavailable", reason=type(error).__name__ + ":" + str(error)
                )
        (self.folder / "manifest.json").write_text(_json(self.entries), encoding="utf8")

    def historical_source(self, path, current_sha256):
        """Return verified captured bytes with historical and current identities."""
        if self.remaining is not None:
            self.remaining()
        item = self.entries.get(path, {})
        if item.get("status") != "complete":
            return None
        raw = (self.folder / (item["content_sha256"] + ".source")).read_bytes()
        if len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != item["content_sha256"]:
            raise ValueError("Historical bytes failed identity verification")
        return dict(item, path=path, current_sha256=current_sha256, source=raw.decode("utf-8-sig"))


def _fragment(path, source, digest, center, *, width=18):
    lines = source.splitlines(keepends=True)
    start = max(1, min(center, max(1, len(lines))) - 4)
    end = min(len(lines), start + width - 1)
    return dict(
        kind="current_source_fragment",
        path=path,
        current_sha256=digest,
        start=start,
        end=end,
        source="".join(lines[start - 1 : end]),
        fragment_complete=True,
        file_complete=start == 1 and end == len(lines),
        interpretation="Exact current lines;not a complete dependency or semantic proof",
    )


def _body_audit(path, symbol, baseline, source, digest):
    before = ast.parse(baseline["source"])
    after = ast.parse(source)
    old = [n for q, n in _definitions(before) if q == symbol]
    new = [n for q, n in _definitions(after) if q == symbol]
    if len(old) != 1 or len(new) != 1:
        return None
    a, b, _, _ = _layout(old[0], baseline["source"])
    c, d, _, _ = _layout(new[0], source)
    old_lines = baseline["source"].splitlines(keepends=True)
    new_lines = source.splitlines(keepends=True)
    old_shell = "".join(old_lines[: a - 1]) + "<BODY>\n" + "".join(old_lines[b:])
    new_shell = "".join(new_lines[: c - 1]) + "<BODY>\n" + "".join(new_lines[d:])
    old_names = {q: n.lineno for q, n in _definitions(before)}
    new_names = {q: n.lineno for q, n in _definitions(after)}
    diff = list(
        difflib.unified_diff(
            old_shell.splitlines(),
            new_shell.splitlines(),
            fromfile="captured_initial_outside_body",
            tofile="current_outside_body",
            n=1,
        )
    )
    return dict(
        kind="explicit_body_scope_audit",
        path=path,
        symbol=symbol,
        captured_revision=baseline["captured_revision"],
        current_sha256=digest,
        baseline_complete=True,
        baseline_status="historical_unfinished_source;not_reference",
        write_scope_origin="explicit_caller_body_write_scope",
        outside_ast_changed=_shell(before, old[0]) != _shell(after, new[0]),
        outside_bytes_changed=old_shell != new_shell,
        removed_declarations=[
            dict(symbol=q, historical_line=old_names[q])
            for q in sorted(old_names.keys() - new_names.keys())
        ][:12],
        added_declarations=[
            dict(symbol=q, current_line=new_names[q])
            for q in sorted(new_names.keys() - old_names.keys())
        ][:12],
        outside_diff=diff[:32],
        outside_diff_complete=len(diff) <= 32,
        historical_body_range=[a, b],
        current_body_range=[c, d],
        interpretation="Byte/AST comparison only;no correctness finding or write permission added",
    )


def collect_candidate_observation(
    *,
    index,
    task,
    public_targets,
    paths,
    reader,
    historical,
    body_write_scopes,
    command,
    workspace_aliases,
    check_time,
):
    """Choose exactly one fact operation A/B/C, or return no new evidence."""
    check_time()
    roots = resolve_public_roots(index, task, public_targets, workspace_aliases=workspace_aliases)[
        "roots"
    ]
    allowed = set(paths) | {r["path"] for r in roots}

    def read(path):
        check_time()
        value = reader(path, max_bytes=1024**2)
        check_time()
        return value

    # A: Only an explicit machine binding authorizes a scope comparison.
    for target in body_write_scopes:
        path, symbol = target.split("::", 1)
        if path not in allowed or not path.endswith((".py", ".pyi")):
            continue
        _, source, digest = read(path)
        baseline = historical.historical_source(path, digest)
        if baseline is None:
            continue
        try:
            record = _body_audit(path, symbol, baseline, source, digest)
        except (SyntaxError, ValueError, RecursionError):
            record = None
        check_time()
        if record is not None:
            return dict(operation="explicit_scope", records=[record], source_files={path: digest})
    # B: Reject pipelines/expansion; empty arbitrary stdout is never EOF.
    literal = _literal(["bash", "-c", command]) if isinstance(command, str) else None
    if literal:
        path = relative_public_path(literal[0], workspace_aliases)
        if path in allowed:
            _, source, digest = read(path)
            lines = source.splitlines()
            start, end = literal[1:]
            record = dict(
                kind="literal_read_bounds",
                path=path,
                current_sha256=digest,
                actual_file_lines=len(lines),
                requested_start=start,
                requested_end=end,
                begins_past_eof=start > len(lines),
                interpretation="Computed bounds for this exact literal file read only",
            )
            root = next((r for r in roots if r["path"] == path), {})
            center = root.get("line") or 1
            hint = root.get("symbol_hint")
            if hint:
                positions = [
                    i
                    for i, line in enumerate(lines, 1)
                    if hint.split(".")[-1].split("::")[-1] in line
                ]
                if positions:
                    center = positions[0]
            return dict(
                operation="literal_bounds",
                records=[record, _fragment(path, source, digest, center)],
                source_files={path: digest},
            )
    # C: Public target and actual candidate change, never an error-number search.
    for root in roots:
        path = root["path"]
        if path not in paths:
            continue
        _, source, digest = read(path)
        baseline = historical.historical_source(path, digest)
        records = []
        center = root.get("line") or 1
        symbol = root.get("symbol")
        if symbol:
            row = index.db.execute("SELECT line FROM symbols WHERE id=?", (symbol,)).fetchone()
            if row:
                center = row["line"]
        elif root.get("symbol_hint"):
            needle = root["symbol_hint"].split(".")[-1].split("::")[-1]
            positions = [i for i, line in enumerate(source.splitlines(), 1) if needle in line]
            if positions:
                center = positions[0]
        if baseline is not None and baseline["captured_revision"] != digest:
            changes = list(
                difflib.unified_diff(
                    baseline["source"].splitlines(),
                    source.splitlines(),
                    fromfile="captured_initial_unfinished_source",
                    tofile="current_candidate",
                    n=2,
                )
            )
            records.append(
                dict(
                    kind="candidate_diff",
                    path=path,
                    captured_revision=baseline["captured_revision"],
                    current_sha256=digest,
                    baseline_complete=True,
                    baseline_status="historical;not_verified_reference",
                    diff=changes[:36],
                    diff_complete=len(changes) <= 36,
                    scope="write_scope_unknown;no_permission_inferred",
                )
            )
        records.append(_fragment(path, source, digest, center))
        support = {path: digest}
        if symbol:
            for edge in index.relationships(symbol, limit=8):
                if edge.get("src") != symbol and edge.get("dst") != symbol:
                    continue
                related = {
                    edge[k].split("::", 1)[0]
                    for k in ("src", "dst")
                    if isinstance(edge.get(k), str) and "::" in edge[k]
                }
                versions = {}
                for other in sorted(related):
                    if other not in allowed:
                        continue
                    versions[other] = read(other)[2]
                if len(versions) != len(related):
                    continue
                records.append(
                    dict(
                        kind="direct_static_relationship_candidate",
                        edge=edge,
                        source_files=versions,
                        binding="Static candidate only;dynamic receiver/ownership unknown",
                    )
                )
                support.update(versions)
                break
        return dict(operation="candidate_target", records=records, source_files=support)
    return dict(operation="no_new_evidence", records=[], source_files={})
