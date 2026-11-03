"""Readonly pinned public CPU runtime for a separately supplied diagnostic.

Only the public role processes see this venv. Ordinary shell execution and
independent final acceptance keep their original runtimes and permissions.
"""

from dataclasses import replace
import copy, hashlib, re, sys, time
from pathlib import Path, PurePosixPath

from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.verification.public_transition_probe import digest, encoded
from infinicoder.verification.transition_public_cpu import (
    CpuTransitionOperation,
    TransitionCpuExecution,
)


class PublicCpuRuntime:
    def __init__(self, directory, manifest):
        if sys.platform != "linux":
            raise UnsupportedCore("public CPU runtime requires Linux namespace execution")
        path = Path(directory)
        if not path.is_absolute() or ".." in path.parts or str(path) != str(path.resolve()):
            raise ValueError("explicit canonical public runtime directory required")
        if (
            len(path.parts) < 4
            or not path.is_dir()
            or any(p.is_symlink() for p in (path, *path.parents))
        ):
            raise ValueError("bounded regular public runtime directory required")
        self.directory = path
        self.manifest = copy.deepcopy(manifest)
        if (
            not isinstance(manifest, dict)
            or set(manifest)
            != {
                "kind",
                "python_abi",
                "torch_version",
                "wheel_sha256",
                "critical_files",
                "dependencies",
            }
            or manifest["kind"] != "public_torch_CPU_venv_v1"
            or manifest["python_abi"] != f"{sys.version_info.major}.{sys.version_info.minor}"
            or not isinstance(manifest["torch_version"], str)
            or "+cpu" not in manifest["torch_version"]
            or not re.fullmatch("[0-9a-f]{64}", manifest["wheel_sha256"])
            or not isinstance(manifest["dependencies"], dict)
        ):
            raise ValueError("explicit pinned public CPU distribution identity required")
        files = manifest["critical_files"]
        required = {
            "pyvenv.cfg",
            "lib/python" + manifest["python_abi"] + "/site-packages/torch/__init__.py",
            "lib/python" + manifest["python_abi"] + "/site-packages/torch/optim/adam.py",
            "lib/python" + manifest["python_abi"] + "/site-packages/torch/lib/libtorch_cpu.so",
        }
        if not isinstance(files, dict) or not required.issubset(files) or not 4 <= len(files) <= 8:
            raise ValueError(
                "four to8 explicit runtime file pins including CPU binary and optimizer required"
            )
        for name, value in files.items():
            p = PurePosixPath(name)
            if (
                p.is_absolute()
                or ".." in p.parts
                or p.as_posix() != name
                or not re.fullmatch("[0-9a-f]{64}", value)
            ):
                raise ValueError("bounded public runtime relative paths and SHA256 pins required")
        self._manifest_bytes = encoded(self.manifest)
        self.identity = digest(self.manifest)
        self.validate(time.monotonic() + 30)

    def validate(self, deadline):
        if encoded(self.manifest) != self._manifest_bytes:
            raise ValueError("public runtime manifest changed")
        total = 0
        for name, pin in self.manifest["critical_files"].items():
            path = self.directory / name
            if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
                raise ValueError("runtime critical source/library must be a regular file")
            size = path.stat().st_size
            total += size
            if size > 768 * 1024**2 or total > 1024**3:
                raise ValueError("runtime pin observation exceeds bounded size")
            h = hashlib.sha256()
            with path.open("rb") as stream:
                while True:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("public runtime identity deadline")
                    block = stream.read(1024**2)
                    if not block:
                        break
                    h.update(block)
            if h.hexdigest() != pin:
                raise ValueError("public runtime critical file changed: " + name)


class _RuntimeCpuExecution(TransitionCpuExecution):
    def __init__(self, scratch, state, runtime, libraries):
        self.public_cpu_runtime = runtime
        super().__init__(
            scratch, state, task_environment="command-private", runtime_library_dirs=libraries
        )
        self.identity += ";readonly_public_CPU_runtime:" + runtime.identity

    def mounts(self):
        values = super().mounts()
        root = self.public_cpu_runtime.directory
        return values if any(root.is_relative_to(p) for p in values) else values + [root]


def public_cpu_transition_operation(contract, reference_script, workspace, state, *, runtime):
    """Return the existing CallerPublicOperation for explicit API injection.

    No device,host proc,private acceptance script or oracle is exposed. The
    role Python must be this venv;matching and library integrity are explicit.
    This is a bounded CPU diagnostic,not full source/binary compatibility.
    """
    if not isinstance(runtime, PublicCpuRuntime):
        raise ValueError("explicit pinned public CPU runtime required")
    data = contract.manifest()
    schema = data["public_input_schema"]
    root = runtime.directory
    if (
        schema.get("device", "cpu") != "cpu"
        or schema.get("python_executable") != str(root / "bin/python")
        or schema.get("diagnostic_runtime_sha256") != runtime.identity
        or schema.get("torch_runtime", {}).get("version") != runtime.manifest["torch_version"]
    ):
        raise ValueError("public diagnostic contract differs from the selected CPU runtime")
    for path in (Path(workspace).resolve(), Path(state).resolve()):
        if path.is_relative_to(root) or root.is_relative_to(path):
            raise ValueError("public runtime cannot overlap source or state")
    execution = _RuntimeCpuExecution(
        Path(state).parent / ("." + Path(state).name + "-public-runtime-scratch"),
        state,
        runtime,
        schema.get("runtime_library_dirs", ()),
    )
    runner = CpuTransitionOperation(contract, reference_script, workspace, state, execution)
    operation = runner.capability.operation
    provider_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def execute(deadline):
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != provider_sha:
            raise ValueError("public runtime provider changed")
        runtime.validate(deadline)
        result = runner.execute(deadline)
        result["diagnostic_runtime_sha256"] = runtime.identity
        return result

    return replace(
        operation,
        executor=execute,
        implementation_identity=dict(
            operation.implementation_identity,
            public_cpu_runtime=runtime.identity,
            public_runtime_provider_sha256=provider_sha,
        ),
        environment_identity=dict(
            operation.environment_identity,
            diagnostic_python=str(root / "bin/python"),
            public_runtime_manifest=runtime.manifest,
            ordinary_execution_unchanged=True,
            host_proc=False,
            GPU=False,
        ),
    )
