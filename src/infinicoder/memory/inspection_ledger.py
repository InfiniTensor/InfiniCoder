"""Persistent read provenance, separate from the bounded model-facing history.

Only complete visible lines or explicitly partial prefixes are observations.
Reading a source version never proves understanding or task completion.
"""

import hashlib
import json
from pathlib import Path


def identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class InspectionLedger:
    def __init__(self, index):
        self.index = index
        index.db.execute("""CREATE TABLE IF NOT EXISTS memory_inspections(
            path TEXT PRIMARY KEY,source_sha256 TEXT,total_lines INTEGER,
            spans TEXT,prefixes TEXT,analysis_identity TEXT)""")

    def _file(self, path):
        from infinicoder.execution.workspace_tools import Workspace

        file = Workspace(self.index.root).path(path)
        if not file.is_file() or file.stat().st_size > 2_000_000:
            raise ValueError("inspection source is not a readable text file")
        raw = file.read_bytes()
        return file.relative_to(self.index.root).as_posix(), raw

    def record(self, result):
        if not isinstance(result, dict) or not all(
            k in result for k in ("path", "sha256", "total_lines", "start", "text")
        ):
            raise ValueError("a controller read observation is required")
        try:
            path, raw = self._file(result["path"])
        except (OSError, ValueError):
            return False
        digest = hashlib.sha256(raw).hexdigest()
        if digest != result["sha256"]:
            return False
        try:
            lines = raw.decode("utf-8-sig").splitlines()
        except UnicodeError as error:
            raise ValueError("inspection source is not UTF-8") from error
        start = result["start"]
        text = result["text"]
        if (
            type(start) is not int
            or start < 1
            or not isinstance(text, str)
            or result["total_lines"] != len(lines)
        ):
            raise ValueError("inspection position does not match source")
        # splitlines() loses a visible trailing blank line. Empty text can
        # represent one blank source line,an empty file,or an out-of-range read.
        visible = (
            len(text.split("\n"))
            if text
            else int(1 <= start <= len(lines) and lines[start - 1] == "")
        )
        count = result.get("covered_line_count", visible)
        if type(count) is not int or count < 0 or count > max(0, len(lines) - start + 1):
            raise ValueError("invalid visible line count")
        partial = (
            result.get("partial_line") == start
            and result.get("read_truncated") is True
            and count == 0
        )
        if partial:
            if not 1 <= start <= len(lines) or not text or not lines[start - 1].startswith(text):
                raise ValueError("partial prefix does not match current source")
        elif text != "\n".join(lines[start - 1 : start - 1 + count]):
            raise ValueError("visible text does not match current source lines")
        if count == 0 and not partial and lines:
            return False
        old = self.index.db.execute(
            "SELECT * FROM memory_inspections WHERE path=?", (path,)
        ).fetchone()
        current = old and old["source_sha256"] == digest and old["analysis_identity"] == identity()
        spans = json.loads(old["spans"]) if current else []
        prefixes = json.loads(old["prefixes"]) if current else {}
        repeated = bool(
            current
            and (
                count
                and any(a <= start and b >= start + count - 1 for a, b in spans)
                or partial
                and (
                    prefixes.get(str(start), 0) >= len(text)
                    or any(a <= start <= b for a, b in spans)
                )
                or not lines
            )
        )
        if count:
            spans.append([start, start + count - 1])
        merged = []
        for first, last in sorted(spans):
            if merged and first <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], last)
            else:
                merged.append([first, last])
        if partial:
            prefixes[str(start)] = max(prefixes.get(str(start), 0), len(text))
        prefixes = {
            line: n
            for line, n in prefixes.items()
            if not any(a <= int(line) <= b for a, b in merged)
        }
        with self.index.db:
            self.index.db.execute(
                "INSERT OR REPLACE INTO memory_inspections VALUES(?,?,?,?,?,?)",
                (path, digest, len(lines), json.dumps(merged), json.dumps(prefixes), identity()),
            )
        next_line = 1
        for first, last in merged:
            if first > next_line:
                break
            next_line = max(next_line, last + 1)
        result["inspection_progress"] = dict(
            repeated_unchanged_range=repeated,
            next_unread_line=next_line if next_line <= len(lines) else None,
            store="persistent_read_ledger",
            guidance="Reuse current source observations or expand a different relevant range;visibility does not prove understanding.",
        )
        return True

    def current(self, paths=None):
        selected = None
        if paths is not None:
            if not isinstance(paths, (list, tuple)) or any(
                not isinstance(path, str) for path in paths
            ):
                raise ValueError("paths must be a list of relative file names")
            from infinicoder.execution.workspace_tools import Workspace

            ws = Workspace(self.index.root)
            selected = {ws.path(path).relative_to(self.index.root).as_posix() for path in paths}
        rows = []
        for record in self.index.db.execute(
            "SELECT * FROM memory_inspections ORDER BY path"
        ).fetchall():
            if selected is not None and record["path"] not in selected:
                continue
            try:
                path, raw = self._file(record["path"])
                valid = (
                    hashlib.sha256(raw).hexdigest() == record["source_sha256"]
                    and record["analysis_identity"] == identity()
                )
            except (OSError, ValueError):
                valid = False
            if not valid:
                with self.index.db:
                    self.index.db.execute(
                        "DELETE FROM memory_inspections WHERE path=?", (record["path"],)
                    )
                continue
            spans = json.loads(record["spans"])
            total = record["total_lines"]
            rows.append(
                dict(
                    path=path,
                    source_sha256=record["source_sha256"],
                    total_lines=total,
                    read_spans=spans,
                    partial_prefix_characters=json.loads(record["prefixes"]),
                    full_file_visible=total == 0 or spans == [[1, total]],
                    origin="controller_read_observation",
                    analysis_identity=identity(),
                )
            )
        return rows

    def query(self, paths=(), limit=20):
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be1..100")
        if (
            not isinstance(paths, (list, tuple))
            or len(paths) > 100
            or any(not isinstance(path, str) for path in paths)
        ):
            raise ValueError("paths must contain at most100 relative file names")
        rows = self.current(paths if paths else None)
        result = dict(
            observations=rows[:limit],
            omitted=max(0, len(rows) - limit),
            rule="Versioned visible-source observations,not understanding or current correctness. Read exact source to expand.",
        )
        while len(json.dumps(result, ensure_ascii=False)) > 4500:
            if not result["observations"]:
                break
            result["observations"].pop()
            result["omitted"] += 1
        return result
