"""Caller-predeclared read-only task history for an owned Docker environment.

The caller creates its normal container with these mount arguments. This helper
never creates a privileged container,restarts one,or mounts the runtime state.
"""

from contextlib import contextmanager
import json
from pathlib import Path
import re
import subprocess

from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.memory.task_history_archive import MODEL_ROOT


class DockerTaskHistoryMount:
    def __init__(self, state_dir, workspace):
        state = Path(state_dir).absolute()
        workspace = Path(workspace).absolute()
        directory = state / "plain" / "history-export"
        if (
            any(c in str(directory) for c in (",", "\0", "\n", "\r"))
            or state.resolve() != state
            or workspace.resolve() != workspace
            or state.is_relative_to(workspace)
            or workspace.is_relative_to(state)
            or any(p.is_symlink() for p in (directory, *directory.parents))
        ):
            raise UnsupportedCore("separate task-private history path required")
        # An existing export cannot be silently reused from another task.
        directory.mkdir(parents=True, mode=0o755, exist_ok=False)
        self.directory = directory.resolve()
        self.environment = None
        self.active = False

    @property
    def mount_arguments(self):
        return [
            "--mount",
            "type=bind,src=" + str(self.directory) + ",dst=" + MODEL_ROOT + ",readonly",
        ]

    def attach(self, environment, *, container_id, image_id, owner_label):
        if self.environment is not None or callable(
            getattr(environment, "bind_history_archive", None)
        ):
            raise UnsupportedCore("history capability already attached")
        if (
            not isinstance(container_id, str)
            or not re.fullmatch(r"[0-9a-f]{64}", container_id)
            or not isinstance(image_id, str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id)
            or not isinstance(owner_label, str)
            or "=" not in owner_label
            or not all(owner_label.split("=", 1))
        ):
            raise UnsupportedCore("actual owned container,image and exact owner label required")
        self.environment = environment
        self.container_id = container_id
        self.image_id = image_id
        self.owner_label = owner_label
        try:
            self.verify()
        except BaseException:
            self.environment = None
            raise
        environment.bind_history_archive = self.bind

    def verify(self):
        if self.environment is None or self.environment.container_id != self.container_id:
            raise UnsupportedCore("original container lease ended")
        process = subprocess.run(
            ["docker", "inspect", self.container_id],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        values = json.loads(process.stdout)
        if len(values) != 1:
            raise UnsupportedCore("ambiguous container identity")
        info = values[0]
        key, value = self.owner_label.split("=", 1)
        host = info["HostConfig"]
        if (
            info["Id"] != self.container_id
            or info["Image"] != self.image_id
            or info["Config"]["Labels"].get(key) != value
            or host.get("Privileged")
            or host.get("NetworkMode") != "none"
            or host.get("Devices")
            or host.get("DeviceRequests")
        ):
            raise UnsupportedCore("owned isolated Docker identity changed")
        mounts = [m for m in info["Mounts"] if m["Destination"] == MODEL_ROOT]
        if (
            len(mounts) != 1
            or mounts[0]["Type"] != "bind"
            or mounts[0]["RW"]
            or Path(mounts[0]["Source"]).resolve() != self.directory
        ):
            raise UnsupportedCore("exact read-only task history mount required")
        return dict(
            container_id=self.container_id,
            image_id=self.image_id,
            owner_label=self.owner_label,
            directory=str(self.directory),
            model_root=MODEL_ROOT,
            readonly=True,
        )

    @contextmanager
    def bind(self, directory, *, phase=None, namespace=None):
        if (
            self.active
            or Path(directory).resolve() != self.directory
            or phase != "plain"
            or not isinstance(namespace, str)
            or not re.fullmatch(r"[0-9a-f]{64}", namespace)
            or any(p.is_symlink() for p in (self.directory, *self.directory.parents))
        ):
            raise UnsupportedCore("task history phase or namespace differs")
        binding = self.verify()
        identity = json.loads((self.directory / "identity.json").read_bytes())
        if identity != dict(phase=phase, namespace=namespace):
            raise UnsupportedCore("export identity differs")
        self.active = True
        try:
            yield dict(
                binding,
                identity="actual_owned_Docker_readonly_task_history_mount_v1",
                phase=phase,
                namespace=namespace,
            )
        finally:
            self.active = False
