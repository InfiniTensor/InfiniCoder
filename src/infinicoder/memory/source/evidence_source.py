"""Explicit read-only source capability; no code tools or namespace requirement."""

from pathlib import Path, PurePosixPath
import re

from infinicoder.tools.code_workbench import CodeWorkbench
from infinicoder.execution.workspace_tools import Workspace
from infinicoder.memory.source.source_archive import _literal
from infinicoder.memory.source.task_source_roots import relative_public_path


def literal_source_read(command, workspace_aliases=("/workspace",)):
    """Recognize exact single-file cat/sed or nl|sed reads, never run them.

    A literal cd to a caller-bound workspace and one known line-numbering
    pipeline preserve file line indices. General pipelines,quoted operators,
    expansion,redirection and extra filtering remain unsupported.
    """
    if not isinstance(command, str):
        return None
    text = command.strip()
    aliases = "|".join(re.escape(root) for root in workspace_aliases)
    prefix = re.match(r"cd[ \t]+(?:" + aliases + r")[ \t]+&&[ \t]+", text) if aliases else None
    if prefix:
        text = text[prefix.end() :]
    literal = _literal(["bash", "-c", text])
    if literal is None:
        path = r"(?:/[A-Za-z0-9_./-]+|[A-Za-z0-9_][A-Za-z0-9_./-]*)"
        match = re.fullmatch(
            r"nl[ \t]+-ba[ \t]+(?P<path>" + path + r")[ \t]*\|[ \t]*"
            r"sed[ \t]+-n[ \t]+(?P<quote>[\'\"])(?P<start>[1-9][0-9]*),(?P<end>[1-9][0-9]*)p(?P=quote)",
            text,
        )
        if not match:
            return None
        start, end = int(match["start"]), int(match["end"])
        if start > end or end - start >= 300:
            return None
        literal = (match["path"], start, end)
    relative = relative_public_path(literal[0], workspace_aliases)
    return (relative, *literal[1:]) if relative is not None else None


class ReadOnlyTaskSources:
    identity = "caller_declared_regular_source_v1;no_project_import"

    def __init__(self, workspace, paths, *, excluded_paths=()):
        self.workspace = Workspace(workspace)
        self.paths = tuple(paths)
        self.excluded_paths = tuple(excluded_paths)
        self.remaining = lambda: None
        for path in self.paths:
            if (
                not isinstance(path, str)
                or not path
                or "\\" in path
                or ":" in path
                or "\0" in path
                or PurePosixPath(path).is_absolute()
                or ".." in PurePosixPath(path).parts
                or str(PurePosixPath(path)) != path
            ):
                raise ValueError("source-read capability requires normalized relative POSIX paths")
            self.workspace.path(path)

    def read(self, path, *, max_bytes=1024**2):
        if path not in self.paths or any(
            path == p or path.startswith(p.rstrip("/") + "/") for p in self.excluded_paths
        ):
            raise ValueError("source is outside the caller-declared read capability")
        # This method only needs workspace and remaining; no editing/index
        # object is constructed or exported. It validates a regular file,
        # symlink components and same-handle/hash consistency.
        return CodeWorkbench._read(self, path, max_bytes=max_bytes)

    def validate(self, workspace, paths):
        if self.workspace.root != Path(workspace).resolve() or not set(paths).issubset(self.paths):
            raise ValueError("source capability belongs to another workspace or observation set")
