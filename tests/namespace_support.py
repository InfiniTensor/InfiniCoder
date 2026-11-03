"""Check host namespace support before running Linux sandbox integration tests."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


def namespace_command(binary, workspace, public, command):
    """Build fixture mounts in an empty root, without host mount-point assumptions."""
    argv = [
        binary,
        "--unshare-user",
        "--die-with-parent",
        "--tmpfs",
        "/",
        "--tmpfs",
        "/home",
        "--tmpfs",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]
    for name in ("/usr", "/bin", "/sbin", "/lib", "/lib64"):
        if Path(name).exists():
            argv += ["--ro-bind", name, name]
    for name in ("/etc/ld.so.cache", "/etc/alternatives"):
        if Path(name).exists():
            argv += ["--ro-bind", name, name]
    return argv + [
        "--ro-bind",
        str(public),
        str(public),
        "--bind",
        str(workspace),
        "/workspace",
        "--chdir",
        "/workspace",
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
        "--setenv",
        "CUDA_VISIBLE_DEVICES",
        "",
        *command,
    ]


def require_namespace():
    binary = os.environ.get("IC_OWNED_BWRAP") or shutil.which("bwrap")
    required = os.environ.get("IC_REQUIRE_BWRAP") == "1"
    reason = None
    if sys.platform != "linux" or not binary or not Path(binary).is_file():
        reason = "Linux bubblewrap is unavailable"
    else:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace, public = root / "workspace", root / "public"
            workspace.mkdir()
            public.mkdir()
            (workspace / "probe").write_text("workspace")
            (public / "probe").write_text("public")
            code = (
                "from pathlib import Path; "
                "assert Path('/workspace/probe').read_text() == 'workspace'; "
                f"assert Path({str(public / 'probe')!r}).read_text() == 'public'"
            )
            probe = subprocess.run(
                namespace_command(
                    binary, workspace, public, ["/usr/bin/python3", "-I", "-c", code]
                ),
                capture_output=True,
                text=True,
                timeout=10,
            )
        if probe.returncode:
            message = probe.stderr.strip() or probe.stdout.strip()
            # Only a denied host capability is optional. Invalid commands,
            # missing libraries and other setup errors must still fail.
            denied = (
                "Operation not permitted",
                "Permission denied",
                "No permissions to create",
                "No permissions to creating",
                "user namespaces are not enabled",
            )
            namespace_error = any(
                marker in message for marker in ("namespace", "uid map", "gid map")
            )
            if not namespace_error or not any(marker in message for marker in denied):
                raise RuntimeError(f"bubblewrap probe failed ({probe.returncode}): {message}")
            reason = f"host denies bubblewrap namespaces ({probe.returncode}): {message}"
    if reason:
        if required:
            raise RuntimeError(reason)
        raise unittest.SkipTest(reason)
    return binary
