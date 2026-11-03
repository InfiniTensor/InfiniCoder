"""Bounded current-source lookups for an explicit unresolved native question.

Windows and macro text are literal evidence,not resolved dispatch/configuration
facts. No repository command,model call or code modification is performed.
"""

import hashlib
from pathlib import Path
import re
from infinicoder.runtime.worker import _native_mask


def registrations(source, tokens):
    mask = _native_mask(source)
    rows = []
    for match in list(re.finditer(r"\bREGISTER\w*\s*\(", mask))[:64]:
        start = match.start()
        opening = mask.find("(", start)
        depth = 0
        right = None
        for i in range(opening, min(len(mask), opening + 2000)):
            if mask[i] == "(":
                depth += 1
            elif mask[i] == ")":
                depth -= 1
                if depth == 0:
                    right = i + 1
                    break
        if right is None:
            continue
        if not any(
            re.search(r"\b" + re.escape(token) + r"\b", mask[start:right]) for token in tokens
        ):
            continue
        rows.append(
            dict(
                line=source.count("\n", 0, start) + 1,
                end=source.count("\n", 0, right) + 1,
                text=source[start:right],
                name=mask[start:opening].strip(),
                kind="lexical_registration_candidate",
                binding_unknown=True,
                configuration_unknown=True,
            )
        )
    return rows[:8]


def lookup(plan, workspace, question, *, protected_paths=(), diagnostics=None, file_table=None):
    from infinicoder.memory.source.source_questions import lookup as explicit_lookup

    records, versions, route = explicit_lookup(
        plan, workspace, question, protected_paths=protected_paths, file_table=file_table
    )
    if diagnostics is not None:
        diagnostics.append(route)
    if route["handled"]:
        return records, versions
    protected = {str(Path(p).resolve()) for p in protected_paths}
    pending = [plan["path"]]
    existing = {r["path"] for r in plan["records"]}
    visited = set()
    candidates = []
    total = 0
    # Actual quoted includes in the target precede less direct lookup files.
    for quote in re.findall(r'^\s*#\s*include\s*"([^"\n]+)"', plan["scope"]["original"], re.M):
        for candidate in (quote, (Path(plan["path"]).parent / quote).as_posix()):
            try:
                if workspace.path(candidate).is_file() and candidate not in pending:
                    pending.append(candidate)
            except ValueError:
                continue
    pending.extend(p for p in sorted(existing) if p not in pending)
    stop = {
        "the",
        "and",
        "for",
        "from",
        "with",
        "this",
        "that",
        "need",
        "actual",
        "exact",
        "provided",
        "source",
        "records",
        "expected",
        "specific",
        "which",
        "where",
        "what",
        "are",
        "not",
        "can",
        "only",
        "return",
        "void",
        "const",
    }
    words = {w for w in re.findall(r"\b[A-Za-z_]\w{2,63}\b", question) if w.lower() not in stop}
    words = sorted(words, key=lambda w: ("_" not in w and not any(c.isupper() for c in w[1:]), w))[
        :24
    ]
    if not words:
        return [], {}
    while pending and len(visited) < 12 and total < 1048576:
        relative = pending.pop(0)
        if relative in visited:
            continue
        visited.add(relative)
        try:
            path = workspace.path(relative)
            if str(path.resolve()) in protected or path.suffix.lower() not in (
                ".h",
                ".hh",
                ".hpp",
                ".cuh",
                ".c",
                ".cc",
                ".cpp",
                ".cxx",
                ".cu",
            ):
                continue
            with path.open("rb") as stream:
                data = stream.read(131073)
            if len(data) > 131072:
                continue
            total += len(data)
            if total > 1048576:
                break
            source = data.decode("utf8")
            mask = _native_mask(source)
            lines = source.splitlines(keepends=True)
        except (OSError, ValueError, UnicodeError):
            continue
        digest = hashlib.sha256(data).hexdigest()
        seen = set()
        for word in words:
            for match in list(re.finditer(r"\b" + re.escape(word) + r"\b", mask))[:2]:
                number = source.count("\n", 0, match.start())
                left = max(0, number - 5)
                right = min(len(lines), number + 15)
                text = "".join(lines[left:right])
                while len(text) > 1400 and right > number + 1:
                    right -= 1
                    text = "".join(lines[left:right])
                while len(text) > 1400 and left < number:
                    left += 1
                    text = "".join(lines[left:right])
                if len(text) > 1400 or (left, right) in seen:
                    continue
                seen.add((left, right))
                hits = sum(
                    bool(re.search(r"\b" + re.escape(w) + r"\b", _native_mask(text))) for w in words
                )
                row = dict(
                    path=relative,
                    source_sha256=digest,
                    line=left + 1,
                    end=right,
                    text=text,
                    name=word,
                    kind="literal_source_window",
                    origin="current_source_question_lookup",
                    view="exact_source_window",
                    binding_unknown=True,
                    configuration_unknown=True,
                    priority=0,
                )
                if any(r["path"] == relative and r.get("text") == text for r in plan["records"]):
                    continue
                candidates.append(
                    (
                        (
                            relative not in existing,
                            hits,
                            path.suffix.lower() in (".h", ".hpp", ".hh"),
                            -len(text),
                        ),
                        row,
                    )
                )
    records = []
    versions = {}
    used = 0
    for _, row in sorted(candidates, key=lambda p: p[0], reverse=True):
        if any(
            r["path"] == row["path"] and not (r["end"] < row["line"] or row["end"] < r["line"])
            for r in records
        ):
            continue
        if used + len(row["text"]) > 4800:
            continue
        records.append(row)
        versions[row["path"]] = row["source_sha256"]
        used += len(row["text"])
        if len(records) == 4:
            break
    # Publish only a self-consistent source snapshot.
    if any(
        hashlib.sha256(workspace.path(p).read_bytes()).hexdigest() != h for p, h in versions.items()
    ):
        return [], {}
    return records, versions
