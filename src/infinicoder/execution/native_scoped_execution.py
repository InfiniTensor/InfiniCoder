"""An explicit write-scope bridge for the existing Native sandbox.

The caller's namespace remains the execution substrate. Its workspace is made
read-only, with writable directory mounts only for the original TaskSpec scope.
There is no host-shell fallback and no second agent or acceptance loop.
"""

from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import threading
import time

from infinicoder.runtime.agent_execution import CoreRequest, UnsupportedCore
from infinicoder.execution.native_sandbox import NativeSandboxExecution


SCOPED_PROFILE = "native_scoped_v1"


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _plain_path(value, *, directory=False):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value or "\0" in value:
        raise UnsupportedCore("normalized workspace-relative scope required")
    text = value[:-1] if directory and value.endswith("/") else value
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or path.parts[0] in (".git", ".infinicoder")
        or str(path) != text
    ):
        raise UnsupportedCore("normalized workspace-relative scope required")
    return text


class ScopedNativeExecution:
    """Single-run directory scopes enforced by the original bubblewrap mounts.

    Exact-file contracts remain with Runtime: a writable file bind cannot
    preserve their create/delete/atomic-replace semantics. Rejecting them is
    explicit; the original TaskSpec is never widened or cleared.
    """

    identity = SCOPED_PROFILE

    def __init__(self, execution, *, cancelled=None):
        if not isinstance(execution, NativeSandboxExecution) or execution.snapshot_trials:
            raise UnsupportedCore("scoped Native needs the original single-run sandbox")
        if cancelled is not None and not callable(cancelled):
            raise ValueError("cancelled must be a caller callback")
        self.execution = execution
        self.cancelled = cancelled or (lambda: False)
        self.root = None
        self.active = False
        self.closed = False
        self.command_count = 0
        self.query_count = 0

    def _writable(self, relative):
        return any(relative.startswith(prefix) for prefix in self.paths)

    def _state(self):
        """Secondary content binding; OS read-only mounts are the write gate."""
        result = {}
        inodes = {}
        for base, directories, files in os.walk(self.root, followlinks=False):
            for name in sorted((*directories, *files)):
                path = Path(base) / name
                relative = path.relative_to(self.root).as_posix()
                info = path.lstat()
                if stat.S_ISREG(info.st_mode):
                    inodes.setdefault((info.st_dev, info.st_ino), []).append(
                        (relative, info.st_nlink)
                    )
                protected = not self._writable(relative) or relative in self.protected
                if not protected:
                    continue
                if path.is_symlink():
                    value = ("link", os.readlink(path))
                elif path.is_file():
                    with path.open("rb") as stream:
                        digest = hashlib.sha256()
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                        value = ("file", digest.hexdigest())
                elif path.is_dir():
                    value = ("directory",)
                else:
                    raise UnsupportedCore("special source files are unsupported")
                result[relative] = (stat.S_IMODE(info.st_mode), value)
        # A pre-existing hardlink through an authorized mount must not make a
        # protected inode writable. New cross-mount links are rejected by Linux.
        for entries in inodes.values():
            writable = [
                name for name, _ in entries if self._writable(name) and name not in self.protected
            ]
            if writable and (
                any(not self._writable(name) or name in self.protected for name, _ in entries)
                or entries[0][1] != len(entries)
            ):
                raise UnsupportedCore("writable source has a protected or external hardlink")
        return result

    def _paths_unchanged(self):
        for relative in (*self.directories, *self.protected):
            path = self.root / relative
            if not path.exists() or any(p.is_symlink() for p in (path, *path.parents)):
                raise UnsupportedCore("scope or protected mount target changed")
        info = self.root.stat()
        if (info.st_dev, info.st_ino) != self.root_inode:
            raise UnsupportedCore("original task workspace replaced")

    def validate(self, request):
        if self.closed:
            raise UnsupportedCore("scoped execution cannot be reused")
        paths = tuple(request.spec.allowed_paths)
        if not paths or any(not p.endswith("/") for p in paths):
            raise UnsupportedCore(
                "native_scoped_v1 needs existing directory prefixes;exact-file contracts use Runtime"
            )
        directories = tuple(_plain_path(p, directory=True) for p in paths)
        if len(set(directories)) != len(directories):
            raise UnsupportedCore("duplicate scope directories")
        for left in directories:
            if any(left != right and left.startswith(right + "/") for right in directories):
                raise UnsupportedCore("redundant nested scope directories")
        root = Path(request.workspace).absolute()
        if not root.is_dir() or any(p.is_symlink() for p in (root, *root.parents)):
            raise UnsupportedCore("regular original workspace without linked parents required")
        for relative in directories:
            path = root / relative
            if not path.is_dir() or any(p.is_symlink() for p in (path, *path.parents)):
                raise UnsupportedCore("scope directories must already exist without linked parents")
        protected = set()
        for check in request.spec.checks:
            for value in check.protected_paths:
                path = Path(value)
                if path.is_absolute():
                    try:
                        value = path.relative_to(root).as_posix()
                    except ValueError:
                        continue  # Caller acceptance owns external checker pins.
                relative = _plain_path(value)
                path = root / relative
                if any(relative.startswith(prefix) for prefix in paths):
                    raise UnsupportedCore(
                        "protected checkers must be outside writable directories;use Runtime for overlapping contracts"
                    )
                if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
                    raise UnsupportedCore(
                        "protected source checkers must be existing regular files"
                    )
                protected.add(relative)
        contract = _digest(asdict(request.spec))
        if self.root is None:
            self.root, self.paths, self.directories = root, paths, directories
            self.protected = tuple(sorted(protected))
            info = root.stat()
            self.root_inode = (info.st_dev, info.st_ino)
            self.contract = contract
            self.baseline = self._state()
        elif self.root != root or self.contract != contract:
            raise UnsupportedCore("scoped task or workspace changed")
        self._check(request)
        self.execution.validate(request)
        self._argv(self.execution.sandbox_argv, root, ["/bin/true"])

    def _check(self, request, *, active=False):
        if active and (not self.active or threading.get_ident() != self.owner):
            raise UnsupportedCore("scope lease expired or used on another thread")
        if (
            _digest(asdict(request.spec)) != self.contract
            or Path(request.workspace).absolute() != self.root
        ):
            raise UnsupportedCore("original TaskSpec or workspace changed")
        self._paths_unchanged()
        if self._state() != self.baseline:
            raise UnsupportedCore("protected workspace contents changed")

    def _argv(self, builder, workspace, command):
        argv = list(builder(workspace, command))
        if (
            not argv
            or Path(argv[0]).name not in ("bwrap", "bubblewrap")
            or argv[-len(command) :] != command
            or "--die-with-parent" not in argv
        ):
            raise UnsupportedCore("scope requires the original explicit bubblewrap namespace")
        end = len(argv) - len(command)
        mutable = []
        # Parse the admitted builder,not arbitrary shell text. Unknown flags or
        # a later child mount must not silently hide an original protection.
        arity = {
            key: 1
            for key in (
                "--unshare-user",
                "--unshare-user-try",
                "--unshare-pid",
                "--unshare-net",
                "--unshare-ipc",
                "--unshare-uts",
                "--unshare-cgroup",
                "--unshare-cgroup-try",
                "--die-with-parent",
                "--new-session",
                "--clearenv",
            )
        }
        arity.update(
            {
                key: 2
                for key in (
                    "--tmpfs",
                    "--dev",
                    "--proc",
                    "--dir",
                    "--chdir",
                    "--unsetenv",
                    "--uid",
                    "--gid",
                    "--hostname",
                    "--cap-drop",
                )
            }
        )
        arity.update({key: 3 for key in ("--bind", "--ro-bind", "--setenv")})
        index = 1
        while index < end:
            value = argv[index]
            size = arity.get(value)
            if size is None or index + size > end:
                raise UnsupportedCore("unsupported namespace operation in scoped builder")
            if value == "--bind":
                mutable.append((index, argv[index + 1 : index + 3]))
            if value in ("--bind", "--ro-bind", "--tmpfs", "--dev", "--proc", "--dir"):
                destination = argv[index + size - 1]
                if destination.startswith("/workspace/"):
                    raise UnsupportedCore(
                        "original nested workspace mounts need a separate supported contract"
                    )
            index += size
        if len(mutable) != 1 or mutable[0][1] != [str(workspace), "/workspace"]:
            raise UnsupportedCore("scope requires one original writable workspace mount")
        # Overlay only existing authorized directories. Their parent and all
        # other source remain read-only,including rename/link destinations.
        index = mutable[0][0]
        argv[index] = "--ro-bind"
        mounts = []
        for relative in self.directories:
            mounts += ["--bind", str(workspace / relative), "/workspace/" + relative]
        for relative in self.protected:
            if self._writable(relative):
                mounts += ["--ro-bind", str(workspace / relative), "/workspace/" + relative]
        if "--unshare-pid" not in argv[:end]:
            mounts += ["--unshare-pid"]
        mounts += ["--cap-drop", "ALL"]
        argv[-len(command) : -len(command)] = mounts
        return argv

    def public_contract(self, request):
        self._check(request, active=True)
        return dict(
            execution_profile=SCOPED_PROFILE,
            allowed_paths=list(self.paths),
            required_artifacts=request.spec.required_artifacts,
            checks=[asdict(check) for check in request.spec.checks],
            scope="Only these directory prefixes are writable;other workspace source is read-only.",
            task_spec_sha256=self.contract,
        )

    @contextmanager
    def bind_model(self, request, model):
        query = model.query

        def guarded(*args, **kwargs):
            self._check(request, active=True)
            if self.cancelled():
                raise KeyboardInterrupt("caller cancelled scoped Native task")
            self.query_count += 1
            result = query(*args, **kwargs)
            self._check(request, active=True)
            if self.cancelled():
                raise KeyboardInterrupt("caller cancelled scoped Native task")
            return result

        model.query = guarded
        try:
            yield
        finally:
            model.query = query

    @contextmanager
    def open(self, request, deadline):
        self.validate(request)
        if self.active or deadline <= time.monotonic():
            raise UnsupportedCore("fresh bounded scoped execution required")
        builder = self.execution.sandbox_argv
        self.owner, self.active = threading.get_ident(), True
        self.execution.sandbox_argv = lambda root, command: self._argv(builder, root, command)
        try:
            with NativeSandboxExecution.open(self.execution, request, deadline) as environment:
                execute = environment.execute

                def guarded(action, *args, **kwargs):
                    self._check(request, active=True)
                    if self.cancelled():
                        raise KeyboardInterrupt("caller cancelled scoped Native task")
                    self.command_count += 1
                    try:
                        return execute(action, *args, **kwargs)
                    finally:
                        self._check(request, active=True)

                environment.execute = guarded
                yield environment
                self._check(request, active=True)
        finally:
            self.execution.sandbox_argv = builder
            self.active, self.closed = False, True
            Path(request.state_dir).mkdir(parents=True, exist_ok=True)
            record = dict(
                execution_profile=SCOPED_PROFILE,
                task_spec_sha256=self.contract,
                allowed_paths=self.paths,
                protected_paths=self.protected,
                commands=self.command_count,
                model_queries=self.query_count,
                closed=True,
                original_namespace=self.execution.identity,
                separate_agent_loop=False,
            )
            (Path(request.state_dir) / "native-scope.json").write_text(
                json.dumps(record, indent=2), encoding="utf8"
            )

    def verify_scope(self, request):
        self._check(request)
        if self.cancelled():
            raise UnsupportedCore("caller cancelled before final acceptance")


def with_scoped_native_execution(request, *, cancelled=None):
    """Keep the complete CoreRequest and explicitly wrap its existing executor."""
    if not isinstance(request, CoreRequest):
        raise TypeError("CoreRequest required")
    if not request.spec.allowed_paths:
        raise UnsupportedCore("scoped execution needs an explicit caller write scope")
    return replace(
        request,
        execution_adapter=ScopedNativeExecution(request.execution_adapter, cancelled=cancelled),
    )
