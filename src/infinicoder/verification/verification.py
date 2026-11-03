"""frozen-snapshot verification with structured report parsing.

Checks execute against a frozen copy of the workspace, so edits during
verification cannot pollute results and vice versa. Pytest-style summaries
are parsed structurally; zero-test, all-skipped and collection-error
outcomes are rejected explicitly instead of counting as success.
"""

import json
import os
import re
import platform
import shutil
import sys
from pathlib import Path

SNAPSHOT_IGNORE = shutil.ignore_patterns(
    ".git", "__pycache__", "*.pyc", ".infinicoder", ".pytest_cache", "state.sqlite*"
)

_PYTEST_SUMMARY = re.compile(
    r"(?:(\d+) passed)?[, ]*(?:(\d+) failed)?[, ]*(?:(\d+) error)?[, ]*"
    r"(?:(\d+) skipped)?[, ]*(?:(\d+) deselected)?",
    re.I,
)
_NO_TESTS = re.compile(r"no tests (?:ran|collected)", re.I)
_COLLECTION_ERROR = re.compile(
    r"\berrors?\b.*during collection|"
    r"error[s]? collecting",
    re.I,
)


def environment_id():
    """Identity of the checking environment (bound into evidence)."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def parse_test_summary(stdout):
    """Best-effort structured parse of pytest-style output.
    Returns dict with counts and detected format; unknown formats keep
    exit-code-only semantics (marked format='exit-code-only')."""
    # Optional machine-readable report. Unknown JSON retains legacy exit-only
    # semantics; malformed/countless success reports are never accepted.
    try:
        report = json.loads(stdout)
        if isinstance(report, dict) and report.get("status") in ("passed", "failed"):
            counts = {k: report.get(k, 0) for k in ("passed", "failed", "error", "skipped")}
            valid = all(type(v) is int and v >= 0 for v in counts.values())
            if not valid or report["status"] == "failed":
                return {"format": "json", "passed": 0, "failed": 1}
            return {"format": "json", **counts}
    except (TypeError, ValueError):
        pass
    if _NO_TESTS.search(stdout or ""):
        return {"format": "pytest-approx", "zero_tests": True}
    if _COLLECTION_ERROR.search(stdout or ""):
        return {"format": "pytest-approx", "collection_error": True}
    unittest_count = re.search(r"Ran (\d+) tests? in ", stdout or "")
    if unittest_count:
        total = int(unittest_count.group(1))
        failed = re.search(r"failures=(\d+)", stdout or "")
        errors = re.search(r"errors=(\d+)", stdout or "")
        skipped = re.search(r"skipped=(\d+)", stdout or "")
        nf, ne, ns = [int(m.group(1)) if m else 0 for m in (failed, errors, skipped)]
        if "FAILED (" in (stdout or "") and not nf and not ne:
            nf = 1
        return {
            "format": "unittest",
            "zero_tests": total == 0,
            "passed": max(0, total - nf - ne - ns),
            "failed": nf,
            "error": ne,
            "skipped": ns,
        }
    m = re.search(
        r"\b((?:\d+ (?:passed|failed|errors?|skipped|deselected|xfailed|xpassed)[, ]*)+)(?:in |[=]|$)",
        stdout or "",
        re.M,
    )
    if not m:
        return {"format": "exit-code-only"}
    counts = {
        "passed": 0,
        "failed": 0,
        "error": 0,
        "skipped": 0,
        "deselected": 0,
        "xfailed": 0,
        "xpassed": 0,
    }
    for n, kind in re.findall(
        r"(\d+) (passed|failed|errors?|skipped|deselected|xfailed|xpassed)", m.group(1)
    ):
        kind = "error" if kind == "errors" else kind
        counts[kind] = counts.get(kind, 0) + int(n)
    return {"format": "pytest-approx", **counts}


def outcome_acceptable(summary):
    """Reject zero-test, all-skipped and collection-error results."""
    if summary.get("zero_tests") or summary.get("collection_error"):
        return False, "no tests collected or collection error"
    if summary.get("format") in ("pytest-approx", "unittest", "json"):
        passed = summary.get("passed", 0)
        failed = summary.get("failed", 0) + summary.get("error", 0)
        if failed:
            return False, "test report contains failures or errors"
        skipped = summary.get("skipped", 0)
        if passed == 0 and failed == 0 and skipped > 0:
            return False, "all tests skipped; nothing verified"
        if passed == 0 and failed == 0:
            return False, "no test outcome recorded"
    return True, ""


class Snapshot:
    """Frozen file-set copy of a workspace for isolated verification."""

    def __init__(self, workspace_root, snapshot_dir, excluded_paths=()):
        self.source = Path(workspace_root).resolve()
        self.dir = Path(snapshot_dir)
        if self.dir.resolve().is_relative_to(self.source.resolve()):
            raise ValueError("Snapshot must be outside workspace")
        excluded = tuple(str(p).replace("\\", "/").rstrip("/") for p in excluded_paths)
        if any(not p or p.startswith("/") or ".." in Path(p).parts for p in excluded):
            raise ValueError("snapshot exclusions must be bounded relative output paths")

        def ignored(folder, names):
            values = set(SNAPSHOT_IGNORE(folder, names))
            parent = Path(folder).relative_to(self.source)
            values.update(
                name
                for name in names
                if any(
                    (parent / name).as_posix() == p
                    or (parent / name).as_posix().startswith(p + "/")
                    for p in excluded
                )
            )
            return values

        links = {}
        # Preserve contained repository links as links, never as host-file copies.
        # Absolute internal links are rebased into the isolated snapshot.
        for folder, dirs, files in os.walk(self.source, followlinks=False):
            omit = ignored(folder, dirs + files)
            dirs[:] = [d for d in dirs if d not in omit]
            for name in dirs + files:
                if name in omit:
                    continue
                p = Path(folder, name)
                if hasattr(p, "is_junction") and p.is_junction():
                    raise ValueError(
                        "Verification tree contains an unsupported Windows junction: "
                        + str(p.relative_to(self.source))
                    )
                if p.is_symlink():
                    try:
                        target = p.resolve()
                    except RuntimeError:
                        raise ValueError(
                            "Cyclic workspace link: " + str(p.relative_to(self.source))
                        ) from None
                    if not target.is_relative_to(self.source):
                        raise ValueError(
                            "Workspace link leaves verification scope: "
                            + str(p.relative_to(self.source))
                        )
                    raw = os.readlink(p)
                    lexical = Path(os.path.abspath(p.parent / raw))
                    if not lexical.is_relative_to(self.source):
                        raise ValueError(
                            "Workspace link traverses outside source scope: "
                            + str(p.relative_to(self.source))
                        )
                    links[p.relative_to(self.source).as_posix()] = (
                        lexical.relative_to(self.source),
                        p.is_dir(),
                    )
        if self.dir.exists():
            shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir(parents=True)
        shutil.copytree(self.source, self.dir, ignore=ignored, dirs_exist_ok=True, symlinks=True)
        for relative, (target, is_directory) in links.items():
            p = self.dir / relative
            if not p.is_symlink():
                raise ValueError("Workspace link changed while snapshot was copied")
            p.unlink()
            p.symlink_to(
                os.path.relpath(self.dir / target, p.parent), target_is_directory=is_directory
            )
        for folder, dirs, files in os.walk(self.dir, followlinks=False):
            for name in dirs + files:
                p = Path(folder, name)
                if p.is_symlink():
                    if p.relative_to(
                        self.dir
                    ).as_posix() not in links or not p.resolve().is_relative_to(self.dir.resolve()):
                        raise ValueError("Snapshot contains an unverified link")

    def revision(self, excluded_paths=()):
        """Content hash over the snapshot (independent of live workspace)."""
        from infinicoder.execution.workspace_tools import Workspace

        return Workspace(self.dir).revision(excluded_paths)

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)
