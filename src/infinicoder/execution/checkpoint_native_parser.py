"""Read-only native definition parser inside the caller's existing namespace."""

from infinicoder._resources import package_root

import hashlib
import json
from pathlib import Path
import sys
import tempfile


def definitions(request, path, source, reader, remaining):
    """Only trusted parser code executes;candidate files are mounted read-only."""
    from infinicoder.execution.workspace_tools import run_process

    execution = request.execution_adapter
    if not callable(getattr(execution, "sandbox_argv", None)):
        raise ValueError("native source parser needs the actual caller namespace builder")
    package = package_root().parent
    payload = dict(
        path=path,
        source=source,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        host_root=str(Path(request.workspace).resolve()),
    )
    with tempfile.TemporaryDirectory(prefix="ic-checkpoint-native-") as directory:
        file = Path(directory) / "input.json"
        file.write_text(json.dumps(payload), encoding="utf8")
        command = [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import sys;sys.path.insert(0,'/tmp/ic-checkpoint-parser');from infinicoder.execution.checkpoint_native_parser import main;main()",
            "/tmp/ic-checkpoint-input.json",
        ]
        argv = list(execution.sandbox_argv(Path(request.workspace).resolve(), command))
        if argv[-len(command) :] != command:
            raise ValueError("native parser builder changed its trusted command")
        # Preserve the original model namespace and make its candidate bind RO
        # for this read-only parser invocation. This does not change SDK mounts.
        workspace = str(Path(request.workspace).resolve())
        matches = [
            i
            for i in range(len(argv) - 2)
            if argv[i : i + 3] == ["--bind", workspace, "/workspace"]
        ]
        if len(matches) != 1:
            raise ValueError("native parser needs the verified /workspace source bind")
        argv[matches[0]] = "--ro-bind"
        argv[-len(command) : -len(command)] = [
            "--ro-bind",
            str(package),
            "/tmp/ic-checkpoint-parser",
            "--ro-bind",
            str(file),
            "/tmp/ic-checkpoint-input.json",
        ]
        result = run_process(
            argv, workspace, timeout=remaining(), max_output=1024**2, spool_cap=1024**2
        )
    if result.exit_code or result.timed_out or result.truncated:
        raise ValueError("native parser unavailable in actual namespace")
    value = json.loads(result.output)
    if value["source_sha256"] != payload["source_sha256"] or not value["complete_inputs"]:
        raise ValueError("native parser source/dependencies not current")
    encoded = source.encode()
    found = []
    for row in value["definitions"]:
        start, end = row["bytes"]
        if not 0 <= start < end <= len(encoded):
            raise ValueError("native extent outside current source")
        found.append(
            dict(
                symbol=row["symbol"],
                kind=row["kind"],
                start=row["lines"][0],
                end=row["lines"][1],
                text=encoded[start:end].decode("utf8"),
                boundary="clang_ast_complete_definition",
                lexical_interface=None,
                parser_inputs=dict(
                    compile_database_sha256=value["compile_database_sha256"],
                    dependency_revision=value["dependency_revision"],
                    namespace="actual_caller_CPU_namespace",
                    semantic_relationships="not_claimed",
                ),
            )
        )
    return found


def main():
    """Stdlib plus installed libclang;no repository imports,commands or writes."""
    from infinicoder.memory.source.evidence_source import ReadOnlyTaskSources
    from infinicoder.languages.native_adapter import BuildProfile, native_context
    from infinicoder.languages.clang_adapter import header_fingerprint

    payload = json.loads(Path(sys.argv[1]).read_bytes())
    path = payload["path"]
    root = Path("/workspace")
    capability = ReadOnlyTaskSources(root, (path,))
    raw, source, version = capability.read(path)
    if (
        source != payload["source"]
        or hashlib.sha256(source.encode()).hexdigest() != payload["source_sha256"]
    ):
        raise ValueError("source changed before namespace parse")
    profile = BuildProfile(root)
    host = payload["host_root"]
    mapped = {}
    for name, entry in profile.entries.items():

        def remap(value):
            return (
                "/workspace" + value[len(host) :]
                if value == host or value.startswith(host + "/")
                else value
            )

        item = dict(entry)
        item["directory"] = remap(item["directory"])
        item["path_mappings"] = dict(item.get("path_mappings", {}), **{host: "/workspace"})
        mapped[remap(name)] = item
    profile.entries = mapped
    symbols, _, _, _, build = native_context(path, source, profile)
    inputs = build.get("_native_inputs", {})
    complete = bool(
        build.get("semantic") and build.get("mode") == "clang-ast" and inputs.get("complete")
    )
    if complete:
        for name, expected in inputs.get("files", {}).items():
            if (header_fingerprint(name, {}) or (None,))[0] != expected:
                complete = False
                break
    if capability.read(path)[2] != version:
        complete = False
    if (
        profile.compile_commands
        and hashlib.sha256(profile.compile_commands.read_bytes()).hexdigest()
        != profile.compile_database_sha256
    ):
        complete = False
    functions = build.get("native_function_facts", {}).get("functions", {})
    definitions = []
    if complete:
        for symbol in symbols:
            body = functions.get(symbol["id"])
            span = body.get("source_extent") if body else None
            if (
                span
                and span["complete_definition"]
                and span["source_sha256"] == payload["source_sha256"]
            ):
                definitions.append(
                    dict(
                        symbol=symbol["qualname"],
                        kind=symbol["kind"],
                        bytes=span["bytes"],
                        lines=span["lines"],
                    )
                )
    print(
        json.dumps(
            dict(
                source_sha256=payload["source_sha256"],
                complete_inputs=complete,
                definitions=definitions,
                compile_database_sha256=profile.compile_database_sha256,
                dependency_revision=inputs.get("revision"),
                project_commands=0,
                candidate_writes=0,
            )
        )
    )


if __name__ == "__main__":
    main()
