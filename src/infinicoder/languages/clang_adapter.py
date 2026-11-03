"""Optional libclang AST adapter. Compilation commands are parsed, never run."""

from pathlib import Path
import hashlib
import json
import re
import stat

MAX_HEADER_BYTES = 8 * 1024 * 1024
MAX_HEADER_TOTAL_BYTES = 64 * 1024 * 1024
MAX_HEADER_COUNT = 2048


def header_fingerprint(filename, cache):
    """Bounded content identity;cache is local to one index refresh."""
    try:
        path = Path(filename).resolve()
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_HEADER_BYTES:
            return None
        identity = (
            str(path),
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
        if identity in cache:
            return cache[identity]
        with path.open("rb") as stream:
            data = stream.read(MAX_HEADER_BYTES + 1)
        if len(data) > MAX_HEADER_BYTES:
            return None
        after = path.stat()
        if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            return None
        result = (hashlib.sha256(data).hexdigest(), len(data))
        cache[identity] = result
        return result
    except (OSError, ValueError, RuntimeError):
        return None


def header_inputs(unit, cache):
    files = {}
    unknown = []
    total = 0
    seen = set()
    for inclusion in unit.get_includes():
        if inclusion.include is None:
            continue
        try:
            name = str(Path(inclusion.include.name).resolve())
        except (OSError, ValueError, RuntimeError):
            unknown.append(dict(reason="unresolved include path"))
            break
        if name in seen:
            continue
        if len(seen) >= MAX_HEADER_COUNT:
            unknown.append(dict(reason="include count limit"))
            break
        seen.add(name)
        value = header_fingerprint(name, cache)
        if value is None:
            unknown.append(
                dict(path=name[:500], reason="unreadable,changing,nonregular or oversized include")
            )
            continue
        if total + value[1] > MAX_HEADER_TOTAL_BYTES:
            unknown.append(dict(reason="include byte limit"))
            break
        files[name] = value[0]
        total += value[1]
    revision = hashlib.sha256(json.dumps(sorted(files.items())).encode()).hexdigest()
    return dict(
        files=files,
        complete=not unknown,
        unknown=unknown[:8],
        unknown_count=len(unknown),
        revision=revision,
    )


def semantic_arguments(entry):
    """Allow semantic compiler options, never plugin/response/config loaders.

    Indexing is a read-only operation. Unsupported semantic flags cause an
    explicit heuristic fallback rather than guessing an equivalent build.
    """
    args = []
    argv = entry["arguments"][1:]
    i = 0
    paths = {
        "-I",
        "-isystem",
        "-iquote",
        "-idirafter",
        "-include",
        "-imacros",
        "-isysroot",
        "--sysroot",
        "-resource-dir",
        "--cuda-path",
        "-F",
        "-iframework",
    }
    values = paths | {"-D", "-U", "-x", "-target", "--target", "-std", "--std"}
    harmless = {
        "-c",
        "--compile",
        "-MD",
        "-MMD",
        "-MP",
        "-fPIC",
        "-fpic",
        "-fPIE",
        "-fpie",
        "-pipe",
    }
    semantic = {
        "-pthread",
        "-fopenmp",
        "-fno-exceptions",
        "-fexceptions",
        "-frtti",
        "-fno-rtti",
        "-fms-extensions",
        "-fms-compatibility",
        "-fno-ms-compatibility",
        "-fdelayed-template-parsing",
        "-fno-delayed-template-parsing",
        "-nostdinc",
        "-nostdinc++",
        "-nocudainc",
        "-nocudalib",
        "--cuda-host-only",
        "--cuda-device-only",
    }

    def mapped(value):
        for prefix, target in entry.get("path_mappings", {}).items():
            if value == prefix or value.startswith(prefix + "/"):
                return str(Path(target) / value[len(prefix) :].lstrip("/"))
        return value

    while i < len(argv):
        arg = argv[i]
        i += 1
        if arg in ("-o", "--output-file", "-MF", "-MT", "-MQ"):
            if i >= len(argv):
                raise ValueError("missing compiler output argument")
            i += 1
            continue
        if (
            arg.startswith("@")
            or arg.startswith(
                (
                    "-fplugin",
                    "-fpass-plugin",
                    "-Xclang",
                    "-Xpreprocessor",
                    "-mllvm",
                    "--config",
                    "-ivfsoverlay",
                    "-include-pch",
                    "-fmodule",
                )
            )
            or arg in ("-load", "-plugin", "-add-plugin")
        ):
            raise ValueError(
                "read-only Clang indexing rejects plugin/response/config/module loading: " + arg
            )
        if arg in values:
            if i >= len(argv):
                raise ValueError("missing semantic compiler argument")
            value = argv[i]
            i += 1
            if value.startswith("@"):
                raise ValueError("compiler response arguments are not expanded by indexing")
            if arg in paths:
                value = mapped(value)
                value = str(
                    Path(value) if Path(value).is_absolute() else Path(entry["directory"]) / value
                )
            args.extend([arg, value])
            continue
        if arg in semantic:
            args.append(arg)
            continue
        if re.fullmatch(r"-O(?:0|1|2|3|s|z|fast)", arg):
            args.append(arg)
            continue
        if arg in harmless or re.fullmatch(r"-(?:g\w*|W\S*)", arg):
            continue
        matched = False
        for flag in sorted(paths, key=len, reverse=True):
            if arg.startswith(flag) and len(arg) > len(flag):
                value = arg[len(flag) :].lstrip("=")
                value = mapped(value)
                args.extend(
                    [
                        flag,
                        str(
                            Path(value)
                            if Path(value).is_absolute()
                            else Path(entry["directory"]) / value
                        ),
                    ]
                )
                matched = True
                break
        if matched:
            continue
        if arg.startswith(
            (
                "-D",
                "-U",
                "-std=",
                "--std=",
                "--target=",
                "--cuda-gpu-arch=",
                "-march=",
                "-mcpu=",
                "-mavx",
                "-mno-",
                "-msse",
            )
        ):
            args.append(arg)
            continue
        if not arg.startswith("-") and Path(arg).suffix.lower() in (
            ".c",
            ".cc",
            ".cpp",
            ".cu",
            ".cxx",
        ):
            continue
        raise ValueError("unsupported semantic compile option;using heuristic facts: " + arg)
    return args + ["-fno-modules", "-working-directory=" + entry["directory"]]


def clang_facts(
    path,
    source,
    root,
    entry,
    dependency_cache=None,
    include_dependencies=False,
    include_bindings=False,
    include_function_facts=False,
    resource_dir=None,
):
    from clang import cindex

    full = (Path(root) / path).resolve()
    args = semantic_arguments(entry)
    if resource_dir is not None:
        from infinicoder.analysis.clang_resources import resolve as resource_arguments

        extra, _ = resource_arguments(resource_dir, args)
        args += extra
    unit = cindex.Index.create().parse(str(full), args=args, unsaved_files=[(str(full), source)])
    diagnostics = [str(d)[:500] for d in unit.diagnostics]
    if any(d.severity >= cindex.Diagnostic.Error for d in unit.diagnostics):
        raise ValueError("Clang translation unit has errors: " + "; ".join(diagnostics[:3]))
    symbols = []
    refs = []
    definitions = []
    flow = []
    bindings = dict(
        symbols=[],
        refs=[],
        limitations="Direct static candidates only;indirect,virtual,missing or ambiguous definitions remain unknown. No runtime dispatch or alias proof.",
    )
    callable_names = {
        "FUNCTION_DECL",
        "CXX_METHOD",
        "CONSTRUCTOR",
        "DESTRUCTOR",
        "FUNCTION_TEMPLATE",
    }
    functions = dict(functions={}, omitted=0)
    declaration_names = {
        "FUNCTION_DECL",
        "CXX_METHOD",
        "CONSTRUCTOR",
        "DESTRUCTOR",
        "STRUCT_DECL",
        "CLASS_DECL",
        "ENUM_DECL",
        "TYPEDEF_DECL",
        "TYPE_ALIAS_DECL",
        "VAR_DECL",
        "FIELD_DECL",
        "PARM_DECL",
        "FUNCTION_TEMPLATE",
        "CLASS_TEMPLATE",
    }

    def own(cursor):
        return cursor.location.file and Path(cursor.location.file.name).resolve() == full

    def qualified(cursor):
        names = [cursor.spelling]
        parent = cursor.semantic_parent
        while parent and parent.kind.name != "TRANSLATION_UNIT":
            if parent.spelling:
                names.insert(0, parent.spelling)
            parent = parent.semantic_parent
        return "::".join(names)

    def visit(cursor, scope=None):
        if not own(cursor):
            return
        kind = cursor.kind.name
        name = cursor.spelling
        # A lambda body belongs to a generated callable,not the containing
        # function. Do not invent outer execution from its lexical placement.
        if kind == "LAMBDA_EXPR":
            return
        if kind in declaration_names and name:
            qual = qualified(cursor)
            sid = f"{path}::{qual}@{cursor.location.line}"
            if cursor.is_definition() or kind in (
                "FIELD_DECL",
                "PARM_DECL",
                "TYPEDEF_DECL",
                "TYPE_ALIAS_DECL",
            ):
                symbols.append(
                    dict(
                        id=sid,
                        name=name,
                        qualname=qual,
                        kind=kind.lower(),
                        line=cursor.extent.start.line,
                        end=cursor.extent.end.line,
                        signature=(cursor.type.spelling + " " + qual)[:500],
                        doc=cursor.raw_comment or "",
                        fields=[],
                        confidence="clang-ast",
                    )
                )
                definitions.append(
                    dict(
                        symbol_id=sid,
                        line=cursor.location.line,
                        end=cursor.extent.end.line,
                        kind=kind.lower(),
                    )
                )
                if kind in callable_names:
                    scope = sid
                    bindings["symbols"].append(dict(symbol=sid, usr=cursor.get_usr()))
                    if include_function_facts:
                        from infinicoder.analysis.native_effects import FUNCTIONS, function_facts

                        if kind in FUNCTIONS:
                            if sid in functions["functions"]:
                                functions["functions"][sid] = None
                            elif len(functions["functions"]) < 256:
                                try:
                                    functions["functions"][sid] = function_facts(
                                        cursor, source, full
                                    )
                                except (ValueError, TypeError, AttributeError, RuntimeError):
                                    functions["functions"][sid] = None
                                    functions["omitted"] += 1
                            else:
                                functions["omitted"] += 1
        if kind in ("DECL_REF_EXPR", "MEMBER_REF_EXPR", "CALL_EXPR") and cursor.referenced:
            target = cursor.referenced
            location = target.location
            flow.append(
                dict(
                    name=target.spelling,
                    use_line=cursor.location.line,
                    definition_line=location.line,
                    definition_file=str(location.file) if location.file else None,
                    scope=scope,
                    confidence="clang-ast-reference",
                )
            )
            if kind == "CALL_EXPR" and scope:
                ref = dict(
                    src=scope,
                    target=target.spelling,
                    kind="calls",
                    line=cursor.location.line,
                    class_scope=None,
                )
                refs.append(ref)
                dispatch = "direct"
                if target.kind.name not in callable_names:
                    dispatch = "indirect_call"
                elif target.kind.name == "CXX_METHOD" and target.is_virtual_method():
                    dispatch = "virtual_dispatch"
                bindings["refs"].append(
                    dict(
                        src=scope,
                        target=target.spelling,
                        line=cursor.location.line,
                        column=cursor.location.column,
                        usr=target.get_usr(),
                        dispatch=dispatch,
                        translation_unit_local=target.linkage.name
                        in ("INTERNAL", "NO_LINKAGE", "UNIQUE_EXTERNAL"),
                    )
                )
        for child in cursor.get_children():
            visit(child, scope)

    for child in unit.cursor.get_children():
        visit(child)
    facts = (symbols, refs, definitions, flow, diagnostics)
    if include_dependencies:
        facts = (*facts, header_inputs(unit, {} if dependency_cache is None else dependency_cache))
    if include_bindings:
        facts = (*facts, bindings)
    return (*facts, functions) if include_function_facts else facts
