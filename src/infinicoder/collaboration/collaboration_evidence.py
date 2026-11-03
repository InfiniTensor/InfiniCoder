"""A bounded, versioned public source packet shared by all collaboration arms.

The complete checkpoint stays in the controller. Selection is lexical evidence,
not a proof of relevance or a new permission. No commands or model calls occur.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from infinicoder.search.candidate_checkpoint import CheckpointConflict, revision
from infinicoder.analysis.code_context import CodeIndex, words
from infinicoder.memory.source.task_source_roots import PATH, NATIVE_NAME, relative_public_path
from infinicoder.execution.workspace_tools import Workspace


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class SourceEvidencePolicy:
    max_tokens: int = 2048
    max_files: int = 12
    scan_bytes: int = 16 * 1024**2

    def __post_init__(self):
        for key in ("max_tokens", "max_files", "scan_bytes"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError("positive source evidence limits required")
        if self.max_files > 12 or self.max_tokens > 2048:
            raise ValueError("source evidence v1 is limited to 12 files and 2048 tokens")


def _read(workspace, path, row):
    location = workspace.path(path)
    if location.stat().st_size > 2_000_000:
        return None
    raw = location.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row["sha256"]:
        raise CheckpointConflict("source evidence changed from the frozen baseline: " + path)
    if b"\x00" in raw:
        return None
    try:
        return raw, raw.decode("utf-8-sig"), row["sha256"]
    except UnicodeDecodeError:
        return None


def build_source_evidence(spec, workspace, baseline, database, *, count_text, guard, policy=None):
    """Use public anchors and a bounded lexical scan, then the existing index.

    Whole files remain precisely retrievable through Runtime's original read
    tool. A partial scan, excerpt, ambiguous match or native declaration never
    claims complete repository or semantic coverage.
    """
    policy = policy or SourceEvidencePolicy()
    if (
        not isinstance(policy, SourceEvidencePolicy)
        or not callable(count_text)
        or not callable(guard)
    ):
        raise TypeError("explicit source policy, tokenizer and parent guard required")
    ws = Workspace(workspace, spec.allowed_paths)
    entries = {p: row for p, row in baseline["entries"].items() if row["kind"] == "file"}
    # Metadata and hidden files cannot be read through the original file tool.
    eligible = {}
    for path, row in entries.items():
        try:
            ws.path(path)
        except ValueError:
            continue
        eligible[path] = row
    literal = set()
    for match in PATH.finditer(spec.task):
        value = relative_public_path(match[1])
        if value in eligible:
            literal.add(value)
        elif value and "/" not in value:
            literal.update(p for p in eligible if Path(p).name == value)
    declared = {p for p in spec.allowed_paths + spec.required_artifacts if p in eligible}
    quoted = set(re.findall(r"`([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)`", spec.task))
    terms = set(words(spec.task)) - {
        "py",
        "pyi",
        "c",
        "cc",
        "cpp",
        "cxx",
        "h",
        "hh",
        "hpp",
        "cu",
        "cuh",
        "rs",
        "js",
        "ts",
        "tsx",
        "json",
        "toml",
        "yaml",
        "yml",
        "sh",
        "txt",
    }
    identifiers = set(re.findall(r"\b[A-Za-z_]\w*\b", spec.task))
    identifiers = {
        w for w in identifiers if "_" in w or re.search(r"[a-z][A-Z]", w) or w[:1].isupper()
    }
    qualified = set(NATIVE_NAME.findall(spec.task))
    candidates = []
    # Scan public anchors first. Byte coverage is explicit and independent of
    # agent results. Paths are never selected just because they sort first.
    order = sorted(
        eligible,
        key=lambda p: (
            p not in literal,
            p not in declared,
            any(x in ("third_party", "vendor", "node_modules") for x in Path(p).parts),
            p,
        ),
    )
    scanned = 0
    scanned_bytes = 0
    sources = {}
    for path in order:
        guard()
        row = eligible[path]
        if row["bytes"] > min(2_000_000, policy.scan_bytes - scanned_bytes):
            continue
        value = _read(ws, path, row)
        scanned_bytes += row["bytes"]
        scanned += 1
        if value is None:
            continue
        raw, source, digest = value
        path_hits = terms & set(words(path))
        names = identifiers & set(re.findall(r"\b[A-Za-z_]\w*\b", source))
        # Public exact paths and identifiers outrank broad natural-language
        # words. Vendor sources remain eligible when the task names them.
        line_hits = len({term for term in terms if len(term) >= 4} & set(words(source)))
        qualified_hits = sum(name in source for name in qualified)
        quoted_module = Path(path).stem in quoted
        score = (
            1000 * (path in literal)
            + 400 * quoted_module
            + 300 * (path in declared)
            + 80 * qualified_hits
            + 30 * len(names)
            + 5 * len(path_hits)
            + min(20, line_hits)
        )
        if not score:
            continue
        if path not in literal and any(
            x in ("third_party", "vendor", "node_modules") for x in Path(path).parts
        ):
            score *= 0.2
        candidates.append((score, path))
        sources[path] = value
        # Keep bounded source storage even when every scanned file has hits.
        candidates.sort(key=lambda item: (-item[0], item[1]))
        for _, omitted in candidates[policy.max_files :]:
            sources.pop(omitted, None)
        del candidates[policy.max_files :]
    packet = dict(
        kind="public_source_evidence_v1",
        baseline_revision=revision(baseline),
        withdrawn_records=0,
        max_tokens=policy.max_tokens,
        total_files=len(entries),
        scanned_files=scanned,
        scanned_bytes=scanned_bytes,
        selection="public_paths_identifiers_and_lexical_candidates",
        records=[],
        limitations="Partial source view; omitted files and dependencies are unknown. "
        "Use original read/lookup tools for exact source; no edit permission added.",
    )

    def tokens(value):
        amount = count_text(_json(value))
        if type(amount) is not int or amount < 0:
            raise ValueError("source tokenizer must return a nonnegative integer")
        return amount

    if tokens(packet) > policy.max_tokens:
        raise ValueError("source evidence header exceeds its tokenizer budget")
    with CodeIndex(
        workspace, database, native_index_policy="on_demand", remaining=lambda: guard() or 1
    ) as index:
        index.refresh_paths([path for _, path in candidates], read_source=lambda p: sources[p])
        ranked = index.search(spec.task, limit=24)
        for _, path in candidates:
            guard()
            raw, source, digest = sources[path]
            lines = source.splitlines(keepends=True)
            if not lines:
                continue
            hits = [r for r in ranked if r["path"] == path]
            center = (
                hits[0]["line"]
                if hits
                else next(
                    (
                        n
                        for n, line in enumerate(lines, 1)
                        if identifiers & set(re.findall(r"\b[A-Za-z_]\w*\b", line))
                    ),
                    1,
                )
            )
            record = dict(
                path=path,
                sha256=digest,
                selection="public_path"
                if path in literal
                else "quoted_module_candidate;binding_unknown"
                if Path(path).stem in quoted
                else "caller_path"
                if path in declared
                else "lexical_candidate;binding_unknown",
            )
            for width in (48, 24, 12, 6, 2, 0):
                start = max(1, center - min(8, width // 4))
                end = min(len(lines), start + width - 1)
                offered = dict(
                    record,
                    start=start,
                    end=end,
                    source="".join(lines[start - 1 : end]) if width else "",
                    status="bounded_excerpt;omitted_code_unknown"
                    if width
                    else "body_omitted;read_original",
                )
                trial = dict(packet, records=[*packet["records"], offered])
                if tokens(trial) <= policy.max_tokens and (
                    width == 0 or count_text(_json(offered)) <= 700
                ):
                    packet = trial
                    break
        for path in sources:
            guard()
            _read(ws, path, eligible[path])
    packet["omitted_files"] = len(entries) - len(packet["records"])
    # Adding the count can cross a tokenizer boundary. Remove complete records,
    # never source characters or caller constraints.
    while tokens(packet) > policy.max_tokens and packet["records"]:
        packet["records"].pop()
        packet["omitted_files"] = len(entries) - len(packet["records"])
    if tokens(packet) > policy.max_tokens:
        raise ValueError("source evidence metadata exceeds its tokenizer budget")
    return packet


def current_source_evidence(packet, workspace, *, count_text):
    """Withdraw source facts after edits, deletion, rename or a link change."""
    ws = Workspace(workspace)
    records = []
    for record in packet["records"]:
        try:
            path = ws.path(record["path"])
            current = (
                path.stat().st_size <= 2_000_000
                and hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
            )
        except (OSError, ValueError):
            current = False
        if current:
            records.append(record)
    current = dict(
        packet,
        records=records,
        omitted_files=packet["total_files"] - len(records),
        withdrawn_records=len(packet["records"]) - len(records),
    )
    while count_text(_json(current)) > packet["max_tokens"] and current["records"]:
        current["records"].pop()
        current["omitted_files"] += 1
        current["withdrawn_records"] += 1
    if count_text(_json(current)) > packet["max_tokens"]:
        raise ValueError("current source metadata exceeds its tokenizer budget")
    return current
