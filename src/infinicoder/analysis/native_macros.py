"""Literal macro-definition dependency view;does not emulate a preprocessor."""

import hashlib
from pathlib import Path
import re
from infinicoder.runtime.worker import _native_mask


def definitions(source):
    lines = source.splitlines(keepends=True)
    masked = _native_mask(source, mask_directives=False).splitlines(keepends=True)
    rows = []
    number = 0
    while number < len(lines):
        start = number
        line = lines[number]
        number += 1
        match = re.match(r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)", masked[start])
        if not match:
            continue
        text = line
        while (
            text.rstrip("\r\n").rstrip().endswith("\\")
            and number < len(lines)
            and len(text) <= 2000
        ):
            text += lines[number]
            number += 1
        if len(text) > 2000 or text.rstrip().endswith("\\"):
            continue
        name = match.group(1)
        mask = _native_mask(text, mask_directives=False)
        replacement = mask[match.end() :]
        if replacement.startswith("("):
            closing = replacement.find(")")
            if closing < 0:
                continue
            replacement = replacement[closing + 1 :]
        # Only invocation-shaped names are candidate dependency edges. Active
        # definitions,conditional branches and expansion semantics stay unknown.
        dependencies = sorted(
            set(re.findall(r"\b([A-Z_][A-Z_0-9]{2,})\s*\(", replacement)) - {name}
        )
        rows.append(
            dict(name=name, line=start + 1, end=number, text=text, dependencies=dependencies)
        )
        if len(rows) >= 128:
            break
    return rows


def collect(plan, workspace):
    invocations = [r for r in plan["records"] if r["kind"] == "lexical_registration_candidate"]
    if not invocations:
        return [], {}, dict(edges=[], missing=[], configuration_unknown=True)
    protected = set(plan.get("protected_paths", ()))
    pending = list(
        dict.fromkeys(
            [r["path"] for r in invocations] + [plan["path"]] + [r["path"] for r in plan["records"]]
        )
    )
    visited = set()
    total = 0
    catalog = {}
    while pending and len(visited) < 12 and total < 1048576:
        relative = pending.pop(0)
        if relative in visited:
            continue
        visited.add(relative)
        try:
            path = workspace.path(relative)
            if str(path.resolve()) in protected:
                continue
            with path.open("rb") as stream:
                data = stream.read(min(131073, 1048576 - total))
            total += len(data)
            if len(data) > 131072 or path.stat().st_size != len(data):
                continue
            source = data.decode("utf8")
            digest = hashlib.sha256(data).hexdigest()
            for row in definitions(source):
                catalog.setdefault(row["name"], []).append(
                    dict(row, path=relative, source_sha256=digest)
                )
            for raw, mask in zip(
                source.splitlines(), _native_mask(source, mask_directives=False).splitlines()
            ):
                if not re.match(r"^\s*#\s*include\b", mask):
                    continue
                include = re.match(r'^\s*#\s*include\s*"([^"\n]+)"', raw)
                if not include:
                    continue
                quote = include.group(1)
                for candidate in (quote, (Path(relative).parent / quote).as_posix()):
                    try:
                        if (
                            candidate not in visited
                            and candidate not in pending
                            and workspace.path(candidate).is_file()
                        ):
                            pending.append(candidate)
                    except ValueError:
                        continue
        except (OSError, ValueError, UnicodeError):
            continue
    wanted = [(r["name"], 0) for r in invocations]
    seen = set()
    records = []
    versions = {}
    edges = []
    missing = []
    truncated = False
    while wanted and len(records) < 8:
        name, depth = wanted.pop(0)
        if name in seen:
            continue
        seen.add(name)
        found = catalog.get(name, [])
        if not found:
            missing.append(name)
            continue
        if len(found) > 2:
            truncated = True
        for row in found[:2]:
            if len(records) >= 8:
                break
            records.append(
                dict(
                    row,
                    kind="macro_definition_source",
                    view="exact_logical_macro_definition",
                    origin="literal_macro_dependency",
                    priority=0,
                    binding_unknown=True,
                    configuration_unknown=True,
                    definition_ambiguous=len(found) > 1,
                )
            )
            versions[row["path"]] = row["source_sha256"]
            for dependency in row["dependencies"]:
                edges.append(
                    dict(
                        source=name,
                        target=dependency,
                        relation="literal_replacement_invocation",
                        path=row["path"],
                        line=row["line"],
                        configuration_unknown=True,
                    )
                )
                if depth < 2:
                    wanted.append((dependency, depth + 1))
                elif dependency not in seen:
                    truncated = True
    try:
        current = all(
            hashlib.sha256(workspace.path(p).read_bytes()).hexdigest() == h
            for p, h in versions.items()
        )
    except (ValueError, OSError):
        current = False
    if not current:
        return (
            [],
            {},
            dict(
                edges=[],
                missing=sorted(seen),
                configuration_unknown=True,
                source_changed_during_scan=True,
            ),
        )
    return (
        records,
        versions,
        dict(
            edges=edges,
            missing=sorted(set(missing)),
            configuration_unknown=True,
            complete_preprocessor_analysis=False,
            omitted=bool(wanted) or truncated,
            source_scan_truncated=bool(pending),
            files_examined=len(visited),
            bytes_read=total,
        ),
    )
