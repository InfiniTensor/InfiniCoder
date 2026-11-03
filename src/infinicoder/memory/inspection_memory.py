"""Bounded,source-versioned inspection coverage;never model-authored facts."""

import hashlib, json


def bounded_read(result, max_text_chars=3400):
    """Bound serialized text before recording coverage,keeping JSON metadata.

    Complete visible lines are observations. A prefix of an oversized single
    line is explicitly partial and never counts as having inspected that line.
    The full-file hash remains the edit authority.
    """
    metadata = {k: v for k, v in result.items() if k != "text"}
    max_text_chars = min(
        max_text_chars, max(32, 5000 - len(json.dumps(metadata, ensure_ascii=False)))
    )
    if (
        "text" not in result
        or len(json.dumps(result["text"], ensure_ascii=False)) <= max_text_chars
    ):
        return result
    result = dict(result)
    lines = result["text"].splitlines()
    visible = []
    for line in lines:
        if len(json.dumps("\n".join([*visible, line]), ensure_ascii=False)) > max_text_chars:
            break
        visible.append(line)
    result.update(
        text="\n".join(visible),
        read_truncated=True,
        requested_lines=len(lines),
        covered_line_count=len(visible),
        visible_end=result["start"] + len(visible) - 1 if visible else None,
    )
    if not visible and lines:
        low, high = 0, len(lines[0])
        while low < high:
            middle = (low + high + 1) // 2
            if len(json.dumps(lines[0][:middle], ensure_ascii=False)) <= max_text_chars:
                low = middle
            else:
                high = middle - 1
        result.update(
            text=lines[0][:low],
            partial_line=result["start"],
            guidance="Only a prefix of this oversized line is visible;use a focused symbol/text query. This line is not covered.",
        )
    return result


def bounded_workpack(result):
    """Preserve source identity/progress instead of cutting the outer JSON."""
    result = dict(result)
    result["source"] = bounded_read(result["source"], 2800)
    interfaces = list(result.get("interfaces", []))
    while (
        interfaces
        and len(json.dumps(dict(result, interfaces=interfaces), ensure_ascii=False)) > 4800
    ):
        interfaces.pop()
    if len(interfaces) < len(result.get("interfaces", [])):
        result["interfaces_truncated"] = True
    result["interfaces"] = interfaces
    return result


def remember_read(state, result):
    if not all(key in result for key in ("path", "sha256", "total_lines", "start", "text")):
        return
    path = result["path"]
    memory = state.setdefault("read_coverage", {})
    previous = memory.pop(path, None)
    spans = previous["spans"] if previous and previous["sha256"] == result["sha256"] else []
    count = result.get("covered_line_count", len(result["text"].splitlines()))
    begin = result["start"]
    end = min(result["total_lines"], begin + count - 1)
    repeated = bool(count and any(a <= begin and b >= end for a, b in spans))
    merged = []
    for a, b in sorted(spans + ([[begin, end]] if count else [])):
        if merged and a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    memory[path] = dict(
        sha256=result["sha256"], total_lines=result["total_lines"], spans=merged[-32:]
    )
    while len(memory) > 16:
        del memory[next(iter(memory))]
    next_line = 1
    for a, b in merged:
        if a > next_line:
            break
        next_line = max(next_line, b + 1)
    result["inspection_progress"] = dict(
        repeated_unchanged_range=repeated,
        next_unread_line=next_line if next_line <= result["total_lines"] else None,
        guidance="Already inspected at this source version;reuse its evidence or inspect a different relevant range."
        if repeated
        else "Coverage records observations only,not task completion.",
    )


def current_coverage(state, workspace, max_chars=1600):
    memory = state.get("read_coverage", {})
    rows = []
    for path, value in list(memory.items()):
        try:
            valid = hashlib.sha256(workspace.path(path).read_bytes()).hexdigest() == value["sha256"]
        except (OSError, ValueError):
            valid = False
        if not valid:
            del memory[path]
            continue
        rows.append(
            dict(
                path=path,
                source_sha256=value["sha256"],
                lines=value["total_lines"],
                read_spans=value["spans"],
            )
        )
    while rows and len(json.dumps(rows, ensure_ascii=False)) > max_chars:
        rows.pop(0)
    return rows


def retain_current_reads(recent, workspace, deduplicate=False):
    """Keep only read observations whose file bytes still match their receipt."""
    if type(deduplicate) is not bool:
        raise ValueError("deduplicate must be bool")
    kept = []
    for pair in recent:
        try:
            observation = json.loads(pair[-1]["content"])
            if not isinstance(observation, dict):
                continue
            if isinstance(observation.get("source"), dict):
                observation = observation["source"]
            if not all(k in observation for k in ("path", "sha256", "text")):
                continue
            if (
                hashlib.sha256(workspace.path(observation["path"]).read_bytes()).hexdigest()
                == observation["sha256"]
            ):
                kept.append(pair)
        except (KeyError, TypeError, ValueError, OSError):
            continue
    if not deduplicate:
        return kept[-3:]
    # Keep newest observations of each exact visible read. Different ranges,
    # partial lines,and compound workpack/analysis observations stay distinct.
    # Raw paid turns remain in recent/history/events;this only selects context.
    selected = []
    seen = set()
    for pair in reversed(kept):
        try:
            proposal = json.loads(pair[0]["content"])
            observation = json.loads(pair[-1]["content"])
            if (
                isinstance(proposal, dict)
                and proposal.get("tool") == "read"
                and all(k in observation for k in ("path", "sha256", "start", "text"))
            ):
                key = (
                    observation["path"],
                    observation["sha256"],
                    observation["start"],
                    observation["text"],
                    observation.get("covered_line_count"),
                    observation.get("partial_line"),
                )
                if key in seen:
                    continue
                seen.add(key)
        except (KeyError, TypeError, ValueError):
            pass
        selected.append(pair)
        if len(selected) == 3:
            break
    return list(reversed(selected))
