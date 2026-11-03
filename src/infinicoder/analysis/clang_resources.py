"""Caller-selected parser resources,never a repository command or installer."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def arguments(directory, existing):
    if directory is None:
        return [], None
    root = Path(directory)
    if not root.is_absolute():
        raise ValueError("Clang resource directory must be caller-owned absolute configuration")
    root = root.resolve()
    header = root / "include/stddef.h"
    if not header.is_file():
        raise ValueError("Configured Clang resources lack stddef.h")
    with header.open("rb") as stream:
        header_data = stream.read(131073)
    if len(header_data) > 131072:
        raise ValueError("Configured stddef.h exceeds budget")
    if any(a == "-resource-dir" or a.startswith("-resource-dir=") for a in existing):
        raise ValueError("Do not override a compile database resource directory")
    record = dict(
        directory=str(root),
        stddef_sha256=hashlib.sha256(header_data).hexdigest(),
        origin="caller_configuration_not_repository_instruction",
        version_verified=False,
    )
    manifest = root / "receipt.json"
    if manifest.is_file():
        with manifest.open("rb") as stream:
            data = stream.read(524289)
        if len(data) > 524288:
            raise ValueError("Resource manifest exceeds budget")
        value = json.loads(data)
        if not isinstance(value, dict):
            raise ValueError("Resource manifest must be an object")
        if value.get("libclang_version") != importlib.metadata.version("libclang"):
            raise ValueError("Configured resource version differs from installed libclang")
        record.update(
            manifest_sha256=hashlib.sha256(data).hexdigest(),
            version_verified=True,
            llvm_commit=value.get("llvm_commit"),
            libclang_version=value["libclang_version"],
        )
    return ["-resource-dir", str(root)], record


def resolve(directory, existing, *, inherit_environment=False):
    """Keep an explicit compile database resource, otherwise caller config.

    Existing semantic arguments have already been mapped into the caller's
    namespace. Nothing from a repository instruction or model is executed.
    The original compiler flags are returned unchanged when they name a resource.
    """
    declared = []
    for i, arg in enumerate(existing):
        if arg == "-resource-dir":
            if i + 1 >= len(existing):
                raise ValueError("compile database resource directory is missing")
            declared.append(existing[i + 1])
        elif arg.startswith("-resource-dir="):
            declared.append(arg.split("=", 1)[1])
    if len(declared) > 1:
        raise ValueError("ambiguous compile database resource directories")
    if declared:
        # Validate and identify its actual header/receipt, without appending an
        # overriding resource flag or falling back if this directory is invalid.
        _, record = arguments(declared[0], [])
        return [], dict(record, effective_origin="compile_database")
    origin = "caller_configuration"
    if directory is None and inherit_environment:
        directory = os.environ.get("INFINICODER_CLANG_RESOURCE_DIR") or None
        origin = "caller_environment"
    extra, record = arguments(directory, existing)
    return extra, dict(record, effective_origin=origin) if record is not None else None
