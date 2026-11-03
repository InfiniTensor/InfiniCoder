"""Locate packaged source identities and data independently of module depth."""

from pathlib import Path


def package_root():
    return Path(__file__).resolve().parent


def package_file(relative_path):
    """Resolve a package-owned relative path, including source hash inputs."""
    root = package_root()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Package resource must remain inside InfiniCoder")
    return path
