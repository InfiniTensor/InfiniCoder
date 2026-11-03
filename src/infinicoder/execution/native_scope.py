"""Optional Clang-backed native function packets, with exact body-only merges.

Compilation commands are data and pass the existing read-only flag whitelist.
Unknown/ambiguous syntax declines localization; it never invents an AST fact.
"""

import hashlib
import json
from pathlib import Path
import re
from infinicoder.runtime.worker import _native_mask

NATIVE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hh", ".hpp"}


def _closing(mask, start):
    depth = 0
    for i in range(start, len(mask)):
        if mask[i] == "{":
            depth += 1
        elif mask[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("Native function body is incomplete")


def _header(text):
    mask = _native_mask(text)
    opening = mask.find("{")
    if opening < 0:
        raise ValueError("Native worker requires one complete function")
    header = mask[:opening]
    # Literals/defaults, attributes, initializers, macros and operator names
    # need a richer declaration comparison; decline instead of guessing.
    if (
        any(c in header for c in "{};=[]")
        or "#" in mask
        or re.search(r"(?m)^\s*#", text)
        or re.search(
            r"\b(namespace|class|struct|enum|operator|if|for|while|switch|catch)\b", header
        )
        or "(" not in header
        or ")" not in header
    ):
        raise ValueError("Unsupported protected native declaration;use action mode")
    end = _closing(mask, opening)
    if mask[end + 1 :].strip():
        raise ValueError("Native worker must return only the protected function")
    tokens = re.findall(r"[A-Za-z_]\w*|\d+|::|&&|->|[^\s]", header)
    if len(tokens) < 4:
        raise ValueError("Unsupported protected native signature")
    return opening, end, tokens


def restore(scope, proposal):
    """Keep the exact original module and declaration, replacing only its body."""
    opening, end, tokens = _header(proposal)
    old_open, old_end, old_tokens = _header(scope["source"])
    if tokens != old_tokens and not _same_qualified_declaration(
        old_tokens, tokens, scope.get("qualname")
    ):
        raise ValueError(
            "Native worker changed its protected signature;use action mode for interface changes"
        )
    source = scope["original"]
    start = scope["offset"]
    fragment = scope["source"]
    if source[start : start + len(fragment)] != fragment:
        raise ValueError("Native scope does not match the protected original source")
    return source[: start + old_open + 1] + proposal[opening + 1 : end] + source[start + old_end :]


def _same_qualified_declaration(original, proposed, qualname):
    """Permit only adding known enclosing namespaces to the SAME function.

    Return type,parameters,names and qualifiers must remain token-identical.
    No alias inference or namespace guessing;the actual declaration is retained.
    """
    if not isinstance(qualname, str) or not re.fullmatch(
        r"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*", qualname
    ):
        return False

    def split(tokens):
        try:
            end = tokens.index("(")
        except ValueError:
            return None
        start = end - 1
        if start < 0 or not re.fullmatch(r"[A-Za-z_]\w*", tokens[start]):
            return None
        while (
            start >= 2
            and tokens[start - 1] == "::"
            and re.fullmatch(r"[A-Za-z_]\w*", tokens[start - 2])
        ):
            start -= 2
        return tokens[:start], tokens[start:end:2], tokens[end:]

    old, new = split(original), split(proposed)
    if old is None or new is None:
        return False
    prefix, names, suffix = old
    other_prefix, other_names, other_suffix = new
    known = qualname.split("::")
    return (
        prefix == other_prefix
        and suffix == other_suffix
        and len(names) <= len(other_names) <= len(known)
        and names == known[-len(names) :]
        and other_names == known[-len(other_names) :]
    )


def packet(
    workspace,
    path,
    task,
    max_context=6500,
    *,
    resource_dir=None,
    include_peers=False,
    diagnostics=None,
):
    """Select one explicitly named, unambiguous Clang function definition.

    Returns None when semantic parsing or exact source provenance is unavailable.
    Read-only evidence is bounded, versioned and explicitly incomplete.
    """
    if Path(path).suffix.lower() not in NATIVE_SUFFIXES:
        return None
    try:
        return _packet(
            workspace,
            path,
            task,
            max_context,
            resource_dir=resource_dir,
            include_peers=include_peers,
            diagnostics=diagnostics,
        )
    except Exception as error:
        # A missing optional parser, invalid translation unit, unsafe compile
        # flag or unsupported native range cannot authorize a localized edit.
        if diagnostics is not None:
            diagnostics.append(
                dict(
                    kind="parser_unavailable",
                    detail=(type(error).__name__ + ": " + str(error))[:500],
                )
            )
        return None


def _arguments(entry, resource_dir=None):
    from infinicoder.languages.clang_adapter import semantic_arguments
    from infinicoder.analysis.clang_resources import resolve

    args = semantic_arguments(entry)
    extra, resources = resolve(resource_dir, args, inherit_environment=True)
    args += extra
    return args, resources, hashlib.sha256(json.dumps(args).encode()).hexdigest()


def configuration(workspace, path):
    """Current read-only analysis identity, without another Clang parse."""
    from infinicoder.languages.native_adapter import BuildProfile

    workspace.path(path, write=True)
    build = BuildProfile(workspace.root)
    entry = build.entry_for(path)
    if entry is None:
        raise ValueError("focused compile database entry is unavailable")
    _, resources, identity = _arguments(entry)
    return resources, identity


def _packet(
    workspace, path, task, max_context, *, resource_dir=None, include_peers=False, diagnostics=None
):
    from clang import cindex
    from infinicoder.languages.native_adapter import BuildProfile
    from infinicoder.languages.clang_adapter import header_inputs, header_fingerprint

    full = workspace.path(path, write=True)
    raw = full.read_bytes()
    if not raw or len(raw) > 1024 * 1024:
        return None
    source = raw.decode("utf-8")
    build = BuildProfile(workspace.root)
    entry = build.entry_for(path)
    if entry is None:
        return None
    database = build.compile_commands.relative_to(Path(workspace.root)).as_posix()
    if workspace.path(database) != build.compile_commands:
        return None
    args, resource_configuration, compile_arguments_identity = _arguments(entry, resource_dir)
    unit = cindex.Index.create().parse(str(full), args=args, unsaved_files=[(str(full), source)])
    errors = [d for d in unit.diagnostics if d.severity >= cindex.Diagnostic.Error]
    if errors:
        if diagnostics is not None:
            diagnostics.extend(
                dict(
                    kind="clang_error",
                    severity=d.severity,
                    detail=d.spelling[:500],
                    location=str(d.location)[:500],
                )
                for d in errors[:8]
            )
        return None

    def own(cursor):
        return cursor.location.file and Path(cursor.location.file.name).resolve() == full.resolve()

    def qualified(cursor):
        parts = [cursor.spelling]
        parent = cursor.semantic_parent
        while parent and parent.kind.name != "TRANSLATION_UNIT":
            if parent.spelling:
                parts.insert(0, parent.spelling)
            parent = parent.semantic_parent
        return "::".join(parts)

    def mentioned(name):
        return bool(re.search(r"(?<![\w:])" + re.escape(name) + r"(?![\w:])", task))

    candidates = []
    all_functions = []
    pending = list(unit.cursor.get_children())
    visited = 0
    while pending:
        node = pending.pop()
        visited += 1
        if visited > 50000:
            return None
        if not own(node):
            continue
        if node.kind.name in ("FUNCTION_DECL", "CXX_METHOD") and node.is_definition():
            name = qualified(node)
            all_functions.append((node, name))
            parts = name.split("::")
            matches = [len(parts) - i for i in range(len(parts)) if mentioned("::".join(parts[i:]))]
            if matches:
                candidates.append((max(matches), node, name))
        elif node.kind.name in (
            "NAMESPACE",
            "CLASS_DECL",
            "STRUCT_DECL",
            "LINKAGE_SPEC",
            "UNEXPOSED_DECL",
        ):
            pending.extend(node.get_children())
    if not candidates:
        return None
    strongest = max(row[0] for row in candidates)
    matches = [row for row in candidates if row[0] == strongest]
    if len(matches) != 1:
        return None
    _, node, name = matches[0]
    # Same-name overloads cannot be disambiguated just by a qualified name.
    if sum(row[2] == name for row in candidates) != 1:
        return None
    bodies = [child for child in node.get_children() if child.kind.name == "COMPOUND_STMT"]
    if len(bodies) != 1:
        return None
    a, b = node.extent.start.offset, node.extent.end.offset
    fragment = raw[a:b].decode("utf-8")
    opening, end, _ = _header(fragment)
    body = bodies[0]
    if (
        body.extent.start.offset != a + len(fragment[:opening].encode("utf-8"))
        or body.extent.end.offset != b
    ):
        return None
    # Check cached Clang token spellings against the current target bytes;
    # macro-expanded/synthetic ranges are not editable source ranges.
    for token in node.get_tokens():
        if not token.location.file or Path(token.location.file.name).resolve() != full.resolve():
            return None
        if (
            raw[token.extent.start.offset : token.extent.end.offset].decode("utf-8")
            != token.spelling
        ):
            return None
    evidence = []
    seen = set()
    used = 0
    omitted = 0
    versions = {path: hashlib.sha256(raw).hexdigest()}
    inputs = header_inputs(unit, {})
    if not inputs["complete"]:
        return None
    external_versions = {}
    for filename, digest in inputs["files"].items():
        actual = Path(filename)
        if actual.is_relative_to(workspace.root):
            relative = actual.relative_to(workspace.root).as_posix()
            workspace.path(relative)  # reject linked/protected repository inputs
            versions[relative] = digest
        else:
            external_versions[filename] = digest

    def add(cursor, priority=False):
        nonlocal used, omitted
        if not cursor or not cursor.location.file:
            return
        actual = Path(cursor.location.file.name).resolve()
        try:
            relative = actual.relative_to(Path(workspace.root).resolve()).as_posix()
            p = workspace.path(relative)
            if p.stat().st_size > 1024 * 1024:
                return
            data = p.read_bytes()
            data.decode("utf-8")
            left, right = cursor.extent.start.offset, cursor.extent.end.offset
            if not (0 <= left < right <= len(data)):
                return
            key = (relative, left, right)
            if key in seen or actual == full.resolve() and left <= a and right >= b:
                return
            seen.add(key)
            value = data[left:right].decode("utf-8")
            view = "exact_source"
            # Other function bodies are optional dependency implementation.
            # Keep their exact declaration prefix when a full body is large.
            children = list(cursor.get_children())
            compound = next((v for v in children if v.kind.name == "COMPOUND_STMT"), None)
            if compound and (not priority or len(value) > max_context - used):
                value = data[left : compound.extent.start.offset].decode("utf-8")
                view = "declaration_prefix"
            record = dict(
                path=relative,
                line=cursor.extent.start.line,
                end=cursor.extent.end.line,
                name=qualified(cursor),
                kind=cursor.kind.name.lower(),
                view=view,
                source_sha256=hashlib.sha256(data).hexdigest(),
                confidence="clang-ast-reference",
            )
            if len(evidence) >= 32:
                omitted += 1
                return
            if used + len(value) <= max_context:
                record["text"] = value
                used += len(value)
            else:
                record.update(view="location_only", reason="read this larger dependency on demand")
            versions[relative] = record["source_sha256"]
            evidence.append(record)
        except (ValueError, OSError, UnicodeError):
            return

    pending = list(node.get_children())
    visited = 0
    while pending and visited < 10000:
        child = pending.pop()
        visited += 1
        if (
            child.kind.name in ("DECL_REF_EXPR", "MEMBER_REF_EXPR", "TYPE_REF", "CALL_EXPR")
            and child.referenced
        ):
            target = child.referenced
            if target.kind.name not in ("PARM_DECL", "VAR_DECL") or not own(target):
                add(target.get_definition() or target, priority=child.kind.name == "CALL_EXPR")
        pending.extend(child.get_children())
    owner = node.semantic_parent
    if owner and owner.kind.name in ("CLASS_DECL", "STRUCT_DECL"):
        definition = owner.get_definition() or owner
        members = list(definition.get_children())
        for member in sorted(
            members,
            key=lambda c: (
                c.kind.name
                not in ("FIELD_DECL", "TYPEDEF_DECL", "TYPE_ALIAS_DECL", "CXX_BASE_SPECIFIER"),
                c.extent.start.line,
            ),
        ):
            if member.kind.name in (
                "FIELD_DECL",
                "TYPEDEF_DECL",
                "TYPE_ALIAS_DECL",
                "CXX_BASE_SPECIFIER",
                "CXX_METHOD",
                "CONSTRUCTOR",
            ):
                add(
                    (member.get_definition() or member) if include_peers else member,
                    priority=include_peers,
                )
                if include_peers and member.kind.name == "CXX_BASE_SPECIFIER" and member.referenced:
                    base = member.referenced.get_definition() or member.referenced
                    for field in base.get_children():
                        if field.kind.name == "FIELD_DECL":
                            add(field)
    if include_peers:
        peers = [
            other
            for other, other_name in all_functions
            if other_name != name and other.spelling == node.spelling
        ]
        for peer in peers[:2]:
            add(peer, priority=True)
        omitted += max(0, len(peers) - 2)
    # Facts are only published if all read versions still match this packet.
    if any(
        hashlib.sha256(workspace.path(p).read_bytes()).hexdigest() != h for p, h in versions.items()
    ):
        return None
    cache = {}
    if any((header_fingerprint(p, cache) or (None,))[0] != h for p, h in external_versions.items()):
        return None
    return dict(
        kind="native_clang",
        qualname=name,
        start=node.extent.start.line,
        end=node.extent.end.line,
        offset=len(raw[:a].decode("utf-8")),
        source=fragment,
        original=source,
        source_sha256=versions[path],
        compile_database_path=database,
        compile_database_sha256=hashlib.sha256(build.compile_commands.read_bytes()).hexdigest(),
        compile_arguments_sha256=compile_arguments_identity,
        module_evidence=evidence,
        evidence_source_versions=versions,
        external_compile_versions=external_versions,
        omitted_evidence=omitted,
        confidence="clang-ast-exact-source-range",
        resource_configuration=resource_configuration,
        peer_evidence_enabled=include_peers,
        limitation="Only this exact function body is editable. Evidence is a bounded static view,not complete alias/dataflow or transitive dependency semantics;inspect omitted dependencies and run independent full acceptance. Interface changes require action mode.",
    )
