"""minimal native (C/C++/CUDA) adapter.

Contract (same as python_facts): (symbols, refs, imports, definitions).
Two modes with EXPLICIT confidence labels:
- compile_commands.json present: build identity = its content hash; per-file
  entry (defines/includes) recorded into the file's facts.
- absent: heuristic-only, every symbol carries confidence='heuristic-native'
  and queries surface the degraded mode (never presented as semantic truth).
"""

import hashlib
import json
import re
import shlex
from pathlib import Path

NATIVE_EXTS = {".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hh", ".hpp"}

_DEF_PATTERNS = [
    # CUDA kernel / device functions
    (re.compile(r"^\s*__global__\s+void\s+(\w+)\s*\(([^;{]*)\)\s*\{", re.M), "kernel"),
    (re.compile(r"^\s*__device__\s+[\w:<>*\s]+?\s+(\w+)\s*\(([^;{]*)\)\s*\{", re.M), "function"),
    # plain free functions (return type + name + params + brace, not decl)
    (
        re.compile(
            r"^\s*(?!else|return|delete)(?:const\s+)?(?:[\w:<>*&\s]+?)\s+"
            r"(\w+)\s*\(([^;{]*)\)\s*(?:const\s*)?\{",
            re.M,
        ),
        "function",
    ),
    (re.compile(r"^\s*(?:class|struct)\s+(\w+)(?!\w)[^;{]*\{", re.M), "type"),
    (re.compile(r"^\s*enum(?:\s+class)?\s+(\w+)\s*(?::\s*\w+)?\s*\{", re.M), "enum"),
    (re.compile(r"^\s*(?:typedef\s+)?using\s+(\w+)\s*=", re.M), "alias"),
]
_NAME_FILTER = {"if", "for", "while", "switch", "catch", "return", "sizeof", "sizeof_", "defined"}


class BuildProfile:
    """Locates and identities the build configuration for native files."""

    def __init__(self, root):
        self.root = Path(root)
        cc = self._find("compile_commands.json")
        self.compile_commands = cc
        content = cc.read_bytes() if cc else None
        self.compile_database_sha256 = (
            hashlib.sha256(content).hexdigest() if content is not None else None
        )
        self.identity = "cc:" + self.compile_database_sha256[:16] if cc else "heuristic"
        self.entries = {}
        self.errors = []
        self.dependency_cache = {}
        if cc:
            try:
                data = json.loads(content.decode("utf-8"))
                if not isinstance(data, list):
                    raise ValueError("compile database must be a list")
                for e in data:
                    if not isinstance(e, dict) or not isinstance(e.get("file"), str):
                        raise ValueError("compile database entry requires file")
                    path_mappings = (
                        {"/workspace": str(self.root.resolve())}
                        if cc.parent == self.root / "build"
                        else {}
                    )

                    def mapped(value):
                        if not isinstance(value, str):
                            raise ValueError("compile directory/path must be text")
                        for prefix, target in path_mappings.items():
                            if value == prefix or value.startswith(prefix + "/"):
                                return str(Path(target) / value[len(prefix) :].lstrip("/"))
                        return value

                    directory = Path(mapped(e.get("directory") or str(self.root)))
                    if not directory.is_absolute():
                        directory = self.root / directory
                    file = Path(mapped(e["file"]))
                    file = (file if file.is_absolute() else directory / file).resolve()
                    argv = e.get("arguments")
                    if argv is None:
                        argv = shlex.split(e.get("command") or "", posix=True)
                    if not isinstance(argv, list) or any(not isinstance(a, str) for a in argv):
                        raise ValueError("compile arguments must be a string list")
                    defines, includes = list(e.get("defines") or []), list(e.get("includes") or [])
                    for i, a in enumerate(argv):
                        if a.startswith("-D") and len(a) > 2:
                            defines.append(a[2:])
                        elif a == "-D" and i + 1 < len(argv):
                            defines.append(argv[i + 1])
                        elif a.startswith("-I") and len(a) > 2:
                            includes.append(a[2:])
                        elif a in ("-I", "-isystem", "-iquote") and i + 1 < len(argv):
                            includes.append(argv[i + 1])
                    self.entries[file.as_posix()] = {
                        "defines": list(dict.fromkeys(defines)),
                        "includes": list(dict.fromkeys(includes)),
                        "arguments": argv,
                        "directory": str(directory),
                        "path_mappings": path_mappings,
                        "command": (e.get("command") or "")[:200],
                    }
            except (ValueError, TypeError, OSError):
                self.errors = ["invalid compile_commands.json; using heuristic facts"]
                self.entries = {}

    def _find(self, name, depth=2):
        # Never borrow an unrelated ancestor project's compile database.
        for cand in (self.root / name, self.root / "build" / name):
            if cand.is_file():
                return cand
        return None

    def entry_for(self, rel_path):
        return self.entries.get((self.root / rel_path).resolve().as_posix())


def native_facts(path, source, build_profile=None):
    # Build flags do not turn regex declarations into semantic AST evidence.
    confidence = "heuristic-native"
    symbols, definitions = [], []
    for pattern, kind in _DEF_PATTERNS:
        for m in pattern.finditer(source):
            name = m.group(1)
            if name in _NAME_FILTER:
                continue
            line = source.count("\n", 0, m.start()) + 1
            sig = source[m.start() : source.find("\n", m.start())].strip()[:300]
            sid = f"{path}::{name}"
            if any(s["id"] == sid for s in symbols):
                continue
            symbols.append(
                {
                    "id": sid,
                    "name": name,
                    "qualname": name,
                    "kind": kind,
                    "line": line,
                    "end": line,
                    "signature": sig,
                    "doc": "",
                    "fields": [],
                    "confidence": confidence,
                }
            )
            definitions.append({"symbol_id": sid, "line": line, "end": line, "kind": kind})
    # includes as import-ish metadata (no call edges invented)
    imports = {}
    for m in re.finditer(r'^\s*#\s*include\s+[<"]([^>"]+)[>"]', source, re.M):
        header = m.group(1)
        imports.setdefault(header.split("/")[0], header)
    return symbols, [], imports, definitions


def native_context(path, source, build_profile, *, semantic=True, resource_dir=None):
    """Adapter entry used by CodeIndex: returns facts + build metadata."""
    symbols, refs, imports, definitions = native_facts(path, source, build_profile)
    entry = build_profile.entry_for(path)
    build_info = {
        "identity": build_profile.identity,
        "mode": "compile_commands" if entry else "heuristic",
        "semantic": False,
        "diagnostics": build_profile.errors,
    }
    if entry:
        build_info["compile_database_path"] = str(build_profile.compile_commands.resolve())
        build_info["compile_database_sha256"] = build_profile.compile_database_sha256
        build_info["path_mappings"] = entry.get("path_mappings", {})
        build_info["defines"] = entry["defines"][:20]
        build_info["includes"] = entry["includes"][:20]
        if not semantic:
            build_info.update(
                mode="deferred-clang",
                semantic_analysis_deferred=True,
                limitation="Coarse native catalog only;request a path to expand semantic facts",
            )
            return symbols, refs, imports, definitions, build_info
        try:
            from infinicoder.languages.clang_adapter import clang_facts
            from infinicoder.analysis.native_effects import identity as effects_identity

            symbols, refs, definitions, flow, diagnostics, inputs, bindings, functions = (
                clang_facts(
                    path,
                    source,
                    build_profile.root,
                    entry,
                    dependency_cache=build_profile.dependency_cache,
                    include_dependencies=True,
                    include_bindings=True,
                    include_function_facts=True,
                    resource_dir=resource_dir,
                )
            )
            build_info.update(
                semantic=True,
                mode="clang-ast",
                use_def=flow,
                diagnostics=diagnostics,
                _native_inputs=inputs,
                native_bindings=bindings,
                native_function_facts=functions,
                native_effects_identity=effects_identity(),
                limitation="AST references only; no proof of pointer aliasing or interprocedural reaching definitions",
            )
        except Exception as exc:
            build_info["diagnostics"] = build_info["diagnostics"] + [
                "semantic parser unavailable: " + str(exc)[:500]
            ]
    return symbols, refs, imports, definitions, build_info
