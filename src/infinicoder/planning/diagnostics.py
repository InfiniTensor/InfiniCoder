"""Bounded compiler diagnostics and stable failure identities.

This extracts observations; it never prescribes a task-specific implementation.
"""

import re

_NVCC = re.compile(r"^(.+?)\((\d+)\): (?:fatal )?error(?: #\S+)?: (.*)$")
_GNU = re.compile(r"^(.+?):(\d+)(?::\d+)?: (?:fatal )?error: (.*)$")


def compiler_diagnostics(output, workspace, limit=3, *, namespace=None):
    diagnostics = []
    for line in (output or "").splitlines():
        match = _NVCC.match(line) or _GNU.match(line)
        if not match:
            continue
        path, number, message = match.groups()
        number = int(number)
        item = {"path": path, "line": number, "message": message[:600]}
        try:
            from pathlib import Path

            if namespace is not None:
                from infinicoder.planning.diagnostic_namespaces import relative_path

                item["reported_path"] = path[:512]
                path, mapping = relative_path(path, workspace, namespace)
                item["mapping"] = mapping
            else:
                raw = Path(path)
                if raw.is_absolute():
                    path = raw.resolve().relative_to(workspace.root).as_posix()
            item["source"] = workspace.read(path, max(1, number - 3), 7)
            if namespace is not None and not 1 <= number <= item["source"]["total_lines"]:
                raise ValueError("reported line lies outside source")
            item["path"] = path
        except (ValueError, OSError, UnicodeError):
            item["source_unavailable"] = True
        diagnostics.append(item)
        if len(diagnostics) == limit:
            break
    return diagnostics


def stable_failure(value):
    """Remove volatile timing/env values so equivalent failures can be counted."""
    if isinstance(value, list):
        return [stable_failure(v) for v in value]
    if isinstance(value, dict):
        return {
            k: stable_failure(v)
            for k, v in value.items()
            if k
            not in {
                "duration",
                "environment",
                "revision",
                "spool_bytes",
                "elapsed_seconds",
                "executed_argv",
                "inspection_progress",
            }
        }
    if isinstance(value, str):
        value = re.sub(r"snap-\d+", "snap-<id>", value)
        return re.sub(r"\b\d+(?:\.\d+)?s\b", "<time>", value)
    return value
