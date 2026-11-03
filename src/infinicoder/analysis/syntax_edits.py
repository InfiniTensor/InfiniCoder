"""Compilation preflight for optional typed Python writes;never executes code."""

from infinicoder._resources import package_root

import hashlib
from pathlib import Path


def identity():
    root = package_root()
    return hashlib.sha256(
        b"".join(
            (root / name).read_bytes()
            for name in ("analysis/syntax_edits.py", "execution/workspace_tools.py")
        )
    ).hexdigest()


def preflight(path, encoded):
    if Path(path).suffix != ".py":
        return
    try:
        compile(encoded, str(path), "exec", dont_inherit=True)
    except (SyntaxError, ValueError, RecursionError) as error:
        if isinstance(error, SyntaxError):
            detail = f"{error.msg};line={error.lineno};offset={error.offset};source={str(error.text or '').strip()[:160]!r}"
        else:
            detail = type(error).__name__ + ": " + str(error)[:180]
        raise ValueError(
            "Python syntax preflight rejected;workspace unchanged. " + detail
        ) from None
