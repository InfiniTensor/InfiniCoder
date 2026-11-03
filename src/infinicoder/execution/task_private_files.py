"""Task-owned temporary files for ordinary commands,not trusted checks.

Shells and namespaces remain fresh. Only explicit tmp/home/cache bind sources
persist until this task closes. No verifier or mutable source is mounted here.
"""

from pathlib import Path, PurePosixPath
import shutil


class TaskPrivateFiles:
    def __init__(self, scratch):
        self.scratch = Path(scratch).absolute()
        if not self.scratch.is_dir() or any(
            p.is_symlink() for p in (self.scratch, *self.scratch.parents)
        ):
            raise ValueError("existing task-owned scratch without linked parents required")
        self.root = self.scratch / "task-files"
        self.root.mkdir(mode=0o700, exist_ok=False)
        self.tmp = self.root / "tmp"
        self.home = self.root / "home"
        self.tmp.mkdir(mode=0o700)
        self.home.mkdir(mode=0o700)
        (self.tmp / "home").mkdir(mode=0o700)
        (self.home / ".cache").mkdir(mode=0o700)
        self.closed = False
        self.commands = 0

    @staticmethod
    def validate_builder(argv, command):
        if len(argv) <= len(command) or argv[-len(command) :] != list(command):
            raise ValueError("namespace builder must append exact command")
        positions = [
            i for i in range(len(argv) - len(command) - 1) if argv[i : i + 2] == ["--tmpfs", "/tmp"]
        ]
        if len(positions) != 1:
            raise ValueError(
                "task files require one explicit private /tmp tmpfs in the admitted builder"
            )
        return positions[0]

    def prepare(self, argv, command):
        if self.closed:
            raise ValueError("closed task temporary files")
        for path in (self.root, self.tmp, self.home, self.tmp / "home"):
            if not path.is_dir() or path.is_symlink():
                raise ValueError("task temporary root replaced;refuse namespace reuse")
        argv = list(argv)
        position = self.validate_builder(argv, command)
        # Replace the parent mount in place. Appending it after the caller's
        # readonly /tmp checker/client binds would hide those actual mounts.
        argv[position : position + 2] = [
            "--bind",
            str(self.tmp),
            "/tmp",
            "--bind",
            str(self.home),
            "/tmp/home",
        ]
        flags = []
        for key, value in (
            ("HOME", "/tmp/home"),
            ("XDG_CACHE_HOME", "/tmp/home/.cache"),
            ("TMPDIR", "/tmp"),
            ("TMP", "/tmp"),
            ("TEMP", "/tmp"),
        ):
            flags += ["--setenv", key, value]
        argv[-len(command) : -len(command)] = flags
        self.commands += 1
        return argv

    def check_mount_targets(self, argv):
        for index, flag in enumerate(argv[:-2]):
            if flag not in ("--bind", "--ro-bind", "--bind-try", "--ro-bind-try"):
                continue
            target = PurePosixPath(argv[index + 2])
            if not str(target).startswith("/tmp/"):
                continue
            relative = target.relative_to("/tmp")
            parts = relative.parts
            base = self.home if parts and parts[0] == "home" else self.tmp
            if base == self.home:
                parts = parts[1:]
            path = base
            for offset, part in enumerate(parts):
                path = path / part
                if path.is_symlink():
                    raise ValueError("task temporary mount target was replaced by a link")
                if path.exists() and offset < len(parts) - 1 and not path.is_dir():
                    raise ValueError("task temporary mount parent is not a directory")

    def close(self):
        if self.closed:
            return
        # The original task execution context has ended before cleanup; no
        # shell is retained. Never follow a candidate-replaced top-level link.
        if self.root.parent != self.scratch:
            raise ValueError("task cleanup outside its original scratch")
        if self.root.is_symlink():
            self.root.unlink()
        elif self.root.is_dir():
            shutil.rmtree(self.root)
        elif self.root.exists():
            self.root.unlink()
        self.closed = True
