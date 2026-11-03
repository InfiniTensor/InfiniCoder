"""Caller-declared checker identities, captured before candidate editing."""

import hashlib
from pathlib import Path


def checker_files(spec, workspace):
    result = {}
    for check in spec.checks:
        # Explicit protected paths are required for arbitrary custom oracles.
        # Common pytest tests and checker scripts also get protection by default.
        paths = list(check.protected_paths)
        for value in check.protected_paths:
            raw = Path(value)
            target = raw if raw.is_absolute() else workspace.path(value)
            if not target.exists():
                raise ValueError("Declared protected checker path does not exist: " + str(target))
        paths.extend(
            a
            for a in check.argv[1:]
            if Path(a).name == "tests"
            or (a.endswith(".py") and Path(a).name.startswith(("test_", "check_", "oracle")))
        )
        for value in paths:
            raw = Path(value)
            target = raw if raw.is_absolute() else workspace.path(value)
            candidates = target.rglob("*") if target.is_dir() else [target]
            for path in sorted(candidates):
                if path.is_file() and path.suffix != ".pyc" and "__pycache__" not in path.parts:
                    if path.is_symlink():
                        raise ValueError("Checker links are not supported")
                    result[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result
