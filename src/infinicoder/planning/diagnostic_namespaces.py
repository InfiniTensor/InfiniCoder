"""Caller-declared compiler locations in a version-bound check snapshot.

A mapped location is an observation, not a root cause, edit permission, or
acceptance proof. Ordinary shell output cannot declare a check namespace.
"""

from infinicoder._resources import package_root

import hashlib
import json
from pathlib import Path, PurePosixPath


def normalize(value):
    if value is None:
        return None
    if not isinstance(value, dict) or not 1 <= len(value) <= 16:
        raise ValueError("diagnostic_namespaces requires1..16 check IDs and POSIX mount roots")
    result = {}
    for check, prefix in value.items():
        if (
            not isinstance(check, str)
            or not check
            or len(check) > 128
            or any(ord(c) < 32 for c in check)
        ):
            raise ValueError("diagnostic namespace needs a bounded check ID")
        if (
            not isinstance(prefix, str)
            or len(prefix) > 512
            or "\\" in prefix
            or any(ord(c) < 32 for c in prefix)
        ):
            raise ValueError("diagnostic namespace needs a canonical absolute POSIX mount root")
        path = PurePosixPath(prefix)
        if (
            not path.is_absolute()
            or str(path) in ("/", "//")
            or ".." in path.parts
            or str(path) != prefix
            or prefix.startswith("//")
        ):
            raise ValueError("diagnostic namespace needs a canonical non-root POSIX mount root")
        result[check] = prefix
    return dict(sorted(result.items()))


def from_cli(values):
    if values is None:
        return None
    result = {}
    for value in values:
        check, separator, prefix = value.partition("=")
        if not separator or check in result:
            raise ValueError("diagnostic namespace requires unique CHECK_ID=POSIX_ROOT entries")
        result[check] = prefix
    return normalize(result)


def validate_checks(value, spec):
    unknown = set(value or ()) - {c.id for c in spec.checks}
    if unknown:
        raise ValueError(
            "diagnostic_namespaces contains unknown check IDs: " + ", ".join(sorted(unknown))
        )


def identity():
    root = package_root()
    names = (
        "planning/diagnostic_namespaces.py",
        "planning/diagnostics.py",
        "verification/acceptance.py",
        "runtime/feedback.py",
        "runtime/engine.py",
    )
    return dict(
        version=1,
        sha256=hashlib.sha256(b"".join((root / n).read_bytes() for n in names)).hexdigest(),
    )


def relative_path(reported, workspace, prefix):
    if not isinstance(reported, str) or len(reported) > 4096 or any(ord(c) < 32 for c in reported):
        raise ValueError("invalid reported source path")
    raw = Path(reported)
    if ".." in raw.parts:
        raise ValueError("reported source path traverses directories")
    if raw.is_absolute():
        try:
            return raw.resolve().relative_to(workspace.root).as_posix(), "snapshot_absolute"
        except ValueError:
            pass
    if "\\" in reported:
        raise ValueError("unmapped reported source path")
    posix = PurePosixPath(reported)
    if ".." in posix.parts or str(posix) != reported or reported.startswith("//"):
        raise ValueError("noncanonical reported source path")
    if posix.is_absolute():
        path = posix.relative_to(PurePosixPath(prefix)).as_posix()
        return path, "caller_declared_check_namespace"
    if raw.is_absolute():
        raise ValueError("reported source lies outside snapshot")
    return reported, "snapshot_relative"


def observe(
    output,
    workspace,
    prefix,
    *,
    check_id,
    argv,
    checker_files,
    snapshot_revision,
    protected_revision,
    output_paths,
):
    """Consume one existing checker result; never dispatch a command or model."""
    from infinicoder.planning.diagnostics import compiler_diagnostics

    before = workspace.revision(output_paths)
    diagnostics = compiler_diagnostics(output, workspace, namespace=prefix)
    after = workspace.revision(output_paths)
    stable = before == protected_revision == after
    binding = dict(
        check_id=check_id,
        namespace=prefix,
        checker_sha256=hashlib.sha256(json.dumps(argv).encode()).hexdigest(),
        checker_files_sha256=hashlib.sha256(
            json.dumps(checker_files, sort_keys=True).encode()
        ).hexdigest(),
        snapshot_revision=snapshot_revision,
        protected_revision=protected_revision,
        protected_before=before,
        protected_after=after,
    )
    for item in diagnostics:
        available = stable and not item.get("source_unavailable")
        item["location_binding"] = dict(
            binding, status="snapshot_source_location" if available else "unknown"
        )
        if available:
            item["location_binding"]["source_sha256"] = item["source"]["sha256"]
        else:
            item.setdefault("reported_path", item.get("path", "")[:512])
            item.pop("path", None)
            item.pop("source", None)
            item.pop("mapping", None)
            item["source_unavailable"] = True
            item["location_binding"]["reason"] = (
                "snapshot_changed" if not stable else "source_location_unavailable"
            )
    return diagnostics
