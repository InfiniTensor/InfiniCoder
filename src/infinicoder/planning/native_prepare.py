"""Prerequisite provider for optional native analysis;never a success judge.

Configuration executes through the caller's original process executor. Like
ordinary commands,it requires command authority and an external sandbox for
untrusted projects. No compilation-database command is executed here.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from infinicoder.languages.native_adapter import BuildProfile


def inputs(workspace, path):
    for relative in ("compile_commands.json", "build/compile_commands.json"):
        workspace.path(relative)  # Reject links before BuildProfile reads metadata.
    build = BuildProfile(workspace.root)
    cmake = workspace.path("CMakeLists.txt")
    cmake_sha = None
    if cmake.is_file():
        with cmake.open("rb") as stream:
            data = stream.read(65537)
        if len(data) <= 65536:
            cmake_sha = hashlib.sha256(data).hexdigest()
    value = dict(
        compile_database_sha256=build.compile_database_sha256,
        compile_database_path=str(build.compile_commands) if build.compile_commands else None,
        target_compile_entry=build.entry_for(path) is not None,
        cmake_source_sha256=cmake_sha,
        diagnostics=build.errors,
    )
    value["fingerprint"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    value["status"] = (
        "necessary_compile_inputs_present_scope_unverified"
        if value["target_compile_entry"]
        else "missing_target_compile_inputs"
    )
    return value


def candidate(spec, workspace, observation, *, allow_commands, protected_paths=()):
    """One plain project-default configure,in an explicitly permitted directory.

    Do not invent project options,download/install dependencies,alter presets or
    overwrite an existing build tree. CMake scripts remain untrusted project code.
    """
    if not allow_commands or observation["compile_database_path"] is not None:
        return None
    if not observation["cmake_source_sha256"]:
        return None
    # BuildProfile currently understands only root and build/ databases. Require
    # an exact caller-declared output directory,not a source edit permission.
    if not any("build" in check.output_paths for check in spec.checks):
        return None
    output = workspace.path("build")
    protected = [Path(p).resolve() for p in protected_paths]
    if any(p == output or p.is_relative_to(output) for p in protected):
        return None
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        return None
    return dict(
        argv=["cmake", "-S", ".", "-B", "build", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"],
        output_path="build",
        source_sha256=observation["cmake_source_sha256"],
        origin="project_default_configuration_with_caller_command_and_output_authority",
        prerequisite_only=True,
        acceptance_proof=False,
    )


def perform(workspace, declaration, process, *, timeout):
    if timeout <= 0:
        raise ValueError("No original runtime budget for build metadata preparation")
    if (
        hashlib.sha256(workspace.path("CMakeLists.txt").read_bytes()).hexdigest()
        != declaration["source_sha256"]
    ):
        raise ValueError("Project configuration changed before preparation")
    before = workspace.revision(excluded_paths=[declaration["output_path"]])
    result = asdict(process(declaration["argv"], workspace.root, timeout))
    after = workspace.revision(excluded_paths=[declaration["output_path"]])
    return dict(
        declaration=declaration,
        execution=result,
        source_revision_before=before,
        source_revision_after=after,
        source_scope_preserved=before == after,
        configured=result["exit_code"] == 0 and not result["timed_out"] and before == after,
        model_calls=0,
        acceptance_proof=False,
    )
