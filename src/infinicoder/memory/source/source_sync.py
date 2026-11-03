"""Optional current read working set; immutable history and checks stay historical.

Only controller read/workpack observations register sources. Arbitrary command
output and model notes are never interpreted as file reads or refreshed facts.
"""

from infinicoder._resources import package_root

from copy import deepcopy
import hashlib
import json
import re


LABEL = "\nCURRENT SOURCE WORKING SET (fresh reads, not verified behavior)\n"


def identity():
    folder = package_root()
    return dict(
        version=1,
        sha256=hashlib.sha256(
            b"".join(
                (folder / name).read_bytes()
                for name in (
                    "memory/source/source_sync.py",
                    "runtime/engine.py",
                    "runtime/shell_loop.py",
                    "execution/workspace_tools.py",
                    "analysis/code_context.py",
                )
            )
        ).hexdigest(),
    )


def _read(workspace, path):
    file = workspace.path(path)
    if not file.is_file() or file.stat().st_size > 2_000_000:
        raise ValueError("working source unavailable or exceeds text-tool limit")
    raw = file.read_bytes()
    if b"\0" in raw:
        raise ValueError("working source is binary")
    return hashlib.sha256(raw).hexdigest(), raw.decode("utf-8-sig").splitlines()


def observe(state, action, result, index, workspace):
    """Register a validated visible read; no inferred read from command stdout."""
    name = action.get("tool")
    if name not in ("read", "read_symbol", "read_function_body", "workpack") or result.get("error"):
        return False
    source = result.get("source") if name == "workpack" else result
    if not isinstance(source, dict) or not all(
        k in source for k in ("path", "sha256", "start", "total_lines", "text")
    ):
        return False
    try:
        digest, lines = _read(workspace, source["path"])
        path = workspace.path(source["path"]).relative_to(workspace.root).as_posix()
        start, text = source["start"], source["text"]
        count = source.get(
            "covered_line_count",
            len(text.split("\n"))
            if text
            else int(1 <= start <= len(lines) and lines[start - 1] == ""),
        )
        if (
            digest != source["sha256"]
            or source["total_lines"] != len(lines)
            or type(start) is not int
            or start < 1
            or type(count) is not int
            or count < 0
            or count > max(0, len(lines) - start + 1)
            or not isinstance(text, str)
        ):
            return False
        partial = (
            source.get("partial_line") == start
            and source.get("read_truncated") is True
            and count == 0
        )
        if partial:
            if not 1 <= start <= len(lines) or not text or not lines[start - 1].startswith(text):
                return False
        elif text != "\n".join(lines[start - 1 : start - 1 + count]):
            return False
        if not count and not partial and lines:
            return False
    except (OSError, ValueError, UnicodeError, TypeError, KeyError):
        return False
    entry = dict(
        path=path,
        origin_sha256=digest,
        start=start,
        count=count,
        prefix_chars=len(text) if partial else None,
        whole_file=start == 1 and count == len(lines) and not partial,
        last_call=state.get("model_calls", 0),
        last_step=state.get("steps", 0),
        observation_sha256=hashlib.sha256(
            json.dumps(source, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
    )
    # Re-anchor a fully observed symbol only against the matching index version.
    # Changed partial line ranges never silently inherit the same numeric offset.
    indexed = index.db.execute("SELECT hash FROM files WHERE path=?", (path,)).fetchone()
    if indexed and indexed["hash"] == digest and count and not entry["whole_file"]:
        symbols = index.db.execute(
            "SELECT id,line,end FROM symbols WHERE path=? AND line>=? AND end<=? AND confidence IN ('syntax','clang-ast') AND kind IN ('function','class','type') ORDER BY (end-line) DESC",
            (path, start, start + count - 1),
        ).fetchall()
        exact = [row for row in symbols if row["line"] == start and row["end"] == start + count - 1]
        if len(exact) == 1:
            entry["symbol_id"] = exact[0]["id"]
    rows = state.setdefault("source_working_set", [])
    key = (path, start, count, entry["prefix_chars"])
    rows[:] = [
        row for row in rows if (row["path"], row["start"], row["count"], row["prefix_chars"]) != key
    ]
    rows.append(entry)
    del rows[:-24]
    return True


def _reference(source, workspace, cache):
    if not isinstance(source, dict) or not all(k in source for k in ("path", "sha256", "text")):
        return None
    try:
        if source["path"] not in cache:
            cache[source["path"]] = _read(workspace, source["path"])[0]
        current = cache[source["path"]]
        status = (
            "same_source_version" if current == source["sha256"] else "historical_source_changed"
        )
    except (OSError, ValueError, UnicodeError):
        current, status = None, "historical_source_unavailable"
    return dict(
        path=source["path"],
        observed_sha256=source["sha256"],
        current_sha256=current,
        start=source.get("start"),
        status=status,
        observation_sha256=hashlib.sha256(
            json.dumps(source, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        guidance="Use current working-set text or request a focused read. Original observation remains in session history; it does not certify current behavior.",
    )


def project_observation(action, observation, workspace, cache=None, covered=()):
    """Render references for tracked read payloads without mutating paid events."""
    if action.get("tool") == "read_batch" and isinstance(observation, dict):
        projected = deepcopy(observation)
        from infinicoder.tools.read_batch import observations

        executed = list(observations(action, observation))
        for item in projected.get("items", []):
            if (
                any(item.get("action") == a and item.get("observation") == r for a, r in executed)
                and item.get("executed") is True
            ):
                item["observation"] = project_observation(
                    item["action"], item["observation"], workspace, cache, covered
                )
        return projected
    if action.get("tool") not in (
        "read",
        "read_symbol",
        "read_function_body",
        "workpack",
    ) or not isinstance(observation, dict):
        return deepcopy(observation)
    source = observation.get("source") if action["tool"] == "workpack" else observation
    reference = _reference(source, workspace, cache if cache is not None else {})
    if reference is None:
        return deepcopy(observation)
    if reference["observation_sha256"] not in (covered or ()):
        # A bounded current view can omit this region. Do not erase its only
        # visible read text merely because a registry reference exists.
        original = deepcopy(observation)
        reference["projection"] = "historical_text_retained_without_complete_visible_replacement"
        reference["guidance"] = (
            "Original read text is retained because no complete replacement is visible in this request. "
            "It describes observed_sha256;check current_sha256/status before treating it as current."
        )
        original["source_reference"] = reference
        return original
    # Do not preserve derived workpack interfaces from a stale source version.
    return dict(source_reference=reference)


def project_pairs(recent, workspace, covered=()):
    pairs = deepcopy(recent)
    cache = {}
    for pair in pairs:
        action = None
        try:
            action = json.loads(pair[0]["content"])
            observation = json.loads(pair[-1]["content"])
            projected = project_observation(action, observation, workspace, cache, covered)
            pair[-1]["content"] = json.dumps(projected, ensure_ascii=False)
        except (ValueError, TypeError, KeyError, IndexError):
            continue  # Unparseable evidence cannot be proven redundant.
    return pairs


def project_record(record, workspace, cache=None, covered=()):
    projected = deepcopy(record)
    projected["observation"] = project_observation(
        record.get("action", {}), record["observation"], workspace, cache, covered
    )
    return projected


def pack(state, workspace, index, question, max_tokens, estimator):
    """Task-ranked current fragments within a caller's reserved token budget."""
    if type(max_tokens) is not int or max_tokens < 0:
        raise ValueError("nonnegative source budget required")
    words = set(re.findall(r"[A-Za-z_]\w*", question.lower()))
    recent = state.get("read_paths", [])[-3:]
    changed = state.get("changed_files", [])[-3:]
    rows = list(state.get("source_working_set", []))

    def rank(entry):
        terms = set(
            re.findall(r"[A-Za-z_]\w*", (entry["path"] + " " + entry.get("symbol_id", "")).lower())
        )
        return (
            entry["path"] in recent,
            entry["path"] in changed,
            len(words & terms),
            entry["last_call"],
            entry["last_step"],
        )

    rows.sort(key=rank, reverse=True)
    selected, references, seen, cache = [], [], set(), {}
    render = lambda: (
        LABEL
        + json.dumps(
            dict(
                fragments=selected,
                references=references,
                rule="Fresh source observations only. Omitted regions require expansion; historical tests do not validate refreshed code.",
            ),
            ensure_ascii=False,
        )
    )
    for entry in rows:
        try:
            if entry["path"] not in cache:
                cache[entry["path"]] = _read(workspace, entry["path"])
            digest, lines = cache[entry["path"]]
            start, count = entry["start"], entry["count"]
            if digest != entry["origin_sha256"]:
                if entry["whole_file"]:
                    start, count = 1, len(lines)
                elif entry.get("symbol_id"):
                    file = index.db.execute(
                        "SELECT hash FROM files WHERE path=?", (entry["path"],)
                    ).fetchone()
                    symbol = index.db.execute(
                        "SELECT line,end FROM symbols WHERE id=? AND path=? AND confidence IN ('syntax','clang-ast')",
                        (entry["symbol_id"], entry["path"]),
                    ).fetchone()
                    if not file or file["hash"] != digest or not symbol:
                        raise ValueError("symbol cannot be re-anchored to current source")
                    start, count = symbol["line"], symbol["end"] - symbol["line"] + 1
                else:
                    raise ValueError("changed partial read requires a new focused read")
            key = (entry["path"], digest, start, count, entry["prefix_chars"])
            if key in seen:
                continue
            seen.add(key)
            row = dict(
                path=entry["path"],
                sha256=digest,
                start=start,
                total_lines=len(lines),
                origin_read_sha256=entry["observation_sha256"],
                changed_since_read=digest != entry["origin_sha256"],
                status="source_read_at_context_assembly_not_behavior_verification",
            )
            if entry.get("symbol_id"):
                row["symbol_id"] = entry["symbol_id"]
            text_lines = lines[start - 1 : start - 1 + count]

            def candidate(n):
                return dict(
                    row,
                    text="\n".join(text_lines[:n]),
                    visible_line_count=n,
                    complete_file=start == 1 and n == len(lines) and entry["prefix_chars"] is None,
                    complete_registered_region=n == count and entry["prefix_chars"] is None,
                    omitted_lines=max(0, count - n),
                )

            low, high, best = 0, len(text_lines), None
            while low <= high:
                middle = (low + high) // 2
                selected.append(candidate(middle))
                fits = estimator.estimate(render()) <= max_tokens
                value = selected.pop()
                if fits:
                    best = value
                    low = middle + 1
                else:
                    high = middle - 1
            if best and (best["visible_line_count"] or not lines):
                selected.append(best)
            elif start <= len(lines):
                # Oversized first line: preserve a bounded prefix and mark it partial.
                line = lines[start - 1]
                limit = min(len(line), entry["prefix_chars"] or len(line))
                low, high, best = 1, limit, None
                while low <= high:
                    middle = (low + high) // 2
                    value = dict(
                        row,
                        text=line[:middle],
                        visible_line_count=0,
                        partial_line=start,
                        prefix_characters=middle,
                        complete_file=False,
                        complete_registered_region=False,
                        omitted_lines=count,
                    )
                    selected.append(value)
                    fits = estimator.estimate(render()) <= max_tokens
                    selected.pop()
                    if fits:
                        best = value
                        low = middle + 1
                    else:
                        high = middle - 1
                if best:
                    selected.append(best)
        except (OSError, ValueError, UnicodeError, KeyError, TypeError) as error:
            reference = dict(
                path=entry.get("path"), status="requires_focused_read", reason=str(error)[:120]
            )
            references.append(reference)
            if estimator.estimate(render()) > max_tokens:
                references.pop()
    text = render() if selected or references else ""
    tokens = estimator.estimate(text) if text else 0
    assert tokens <= max_tokens
    report = dict(
        version=1,
        registered=len(rows),
        selected=len(selected),
        references=len(references),
        tokens=tokens,
        budget=max_tokens,
        source_versions={row["path"]: row["sha256"] for row in selected},
        complete_paths=[row["path"] for row in selected if row["complete_file"]],
        covered_observations=[
            row["origin_read_sha256"] for row in selected if row["complete_registered_region"]
        ],
        historical_events_rewritten=False,
    )
    return text, report
