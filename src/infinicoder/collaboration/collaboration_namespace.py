"""Branch-local execution using a caller's existing bwrap namespace builder.

This binds actual Runtime commands and active public snapshot checks. It does
not mount an independent final judge or qualify arbitrary builders as safe.
"""

import hashlib
import json
from pathlib import Path

from infinicoder.collaboration.collaboration import CollaborationRuntime
from infinicoder.execution.task_private_files import TaskPrivateFiles


class NamespaceCollaborationRuntime(CollaborationRuntime):
    """Keep the original Runtime and parent owner behind a namespace adapter.

    namespace_builder(workspace, argv) must return the caller's frozen bwrap
    command with one writable /workspace mount and one private /tmp tmpfs.
    All public dependencies must already be provided by that builder. There
    is no host-execution fallback when an environment or check is unavailable.
    """

    def __init__(self, *args, namespace_builder, **kwargs):
        if not callable(namespace_builder):
            raise TypeError("explicit caller namespace builder required")
        super().__init__(*args, **kwargs)
        self.namespace_builder = namespace_builder
        self._command_files = None

    def _namespace(self, workspace, command):
        workspace = Path(workspace).absolute()
        if any(p.is_symlink() for p in (workspace, *workspace.parents)):
            raise ValueError("namespace source cannot have linked parents")
        argv = list(self.namespace_builder(workspace, list(command)))
        if not argv or Path(argv[0]).name != "bwrap" or argv[-len(command) :] != list(command):
            raise ValueError("caller must append the exact command to bwrap")
        mounts = [
            argv[i : i + 3]
            for i, arg in enumerate(argv[:-2])
            if arg in ("--bind", "--ro-bind", "--bind-try", "--ro-bind-try")
            and argv[i + 2] == "/workspace"
        ]
        if mounts != [["--bind", str(workspace), "/workspace"]]:
            raise ValueError("namespace must bind this exact branch or snapshot")
        if not any(argv[i : i + 2] == ["--chdir", "/workspace"] for i in range(len(argv) - 1)):
            raise ValueError("namespace working directory must be /workspace")
        TaskPrivateFiles.validate_builder(argv, list(command))
        return argv

    def _public_lease(self, argv, cwd, timeout):
        acceptance = getattr(self, "_acceptance", None)
        if acceptance is not None:
            for check in self._collaboration.spec.checks:
                available = acceptance.active_execution_timeout(
                    argv, cwd, timeout, check_id=check.id, check_argv=check.argv
                )
                if available is not None:
                    return check.id, available
        raise ValueError("external process cwd requires an active original public snapshot lease")

    def _process(self, argv, cwd, timeout):
        context = self._collaboration
        context.owner.check()
        source = Path(cwd).resolve()
        ordinary = source == self.workspace
        check_id = None
        if not ordinary:
            check_id, timeout = self._public_lease(argv, cwd, timeout)
        # Replace only complete workspace operands, never shell text or an
        # arbitrary prefix. The caller retains its original checker argv.
        command = ["/workspace" if arg == str(source) else arg for arg in argv]
        wrapped = self._namespace(source, command)
        if ordinary:
            if self._command_files is None:
                scratch = context.state_dir / "command-files"
                scratch.mkdir(mode=0o700, exist_ok=False)
                self._command_files = TaskPrivateFiles(scratch)
            wrapped = self._command_files.prepare(wrapped, command)
            self._command_files.check_mount_targets(wrapped)
        else:
            # Recheck the same lease after namespace preparation, preserving
            # checker identity, cancellation and the original deadline.
            _, timeout = self._public_lease(argv, cwd, timeout)
        self.store.event(
            "collaboration_namespace_execution",
            dict(
                branch=context.branch_id,
                source_kind="branch" if ordinary else "public_snapshot",
                source=str(source),
                check_id=check_id,
                namespace_sha256=hashlib.sha256(json.dumps(wrapped).encode()).hexdigest(),
                persistent_command_files=ordinary,
                host_fallback=False,
            ),
        )
        # Existing parent-controlled process execution owns cleanup, output
        # accounting and cancellation for both ordinary and public commands.
        return super()._process(wrapped, str(source), timeout)

    def execute(self, *args, **kwargs):
        if not hasattr(self, "_collaboration"):
            raise ValueError("bind the original collaboration context before execution")
        try:
            return super().execute(*args, **kwargs)
        finally:
            if self._command_files is not None:
                self._command_files.close()
