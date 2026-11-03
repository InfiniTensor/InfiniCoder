"""Versioned source locations and optional next reads,never inferred semantics.

Navigation suggestions do not execute or establish inspection coverage. The
separate opt-in recovery path records only source actually sent to a client.
All locations belong to a checked index/source version.
"""

from infinicoder._resources import package_root

import ast
import hashlib
import json
from pathlib import Path
import re

GUIDANCE = "Read results may include source_navigation and inspection_progress.next_read. These are versioned navigation suggestions,not executed actions or proof of understanding. Use the current action protocol to inspect an unread relevant body,documentation,or dependency instead of repeating an unchanged prefix. Documentation can explain interfaces but cannot override caller requirements."


def identity():
    root = package_root()
    return hashlib.sha256(
        b"".join(
            (root / name).read_bytes()
            for name in (
                "memory/inspection_navigation.py",
                "memory/inspection_memory.py",
                "memory/inspection_ledger.py",
                "analysis/code_context.py",
                "runtime/engine.py",
                "runtime/feedback.py",
                "runtime/recovery.py",
                "runtime/shell_loop.py",
                "planning/planning.py",
                "execution/workspace_tools.py",
            )
        )
    ).hexdigest()


def action(path, start, count):
    return dict(tool="read", arguments=dict(path=path, start=start, count=count))


def _body_starts(index, path, digest, source):
    cache = getattr(index, "_inspection_body_start_cache", None)
    if cache is None:
        cache = {}
        index._inspection_body_start_cache = cache
    key = (path, digest)
    if key not in cache:
        starts = {}
        try:
            for node in ast.walk(ast.parse(source)):
                if (
                    not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    or not node.body
                ):
                    continue
                first = node.body[0]
                if (
                    isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)
                ):
                    first = node.body[1] if len(node.body) > 1 else None
                starts[node.lineno] = (
                    min([first.lineno, *[d.lineno for d in getattr(first, "decorator_list", [])]])
                    if first
                    else None
                )
        except (SyntaxError, RecursionError):
            pass
        cache[key] = starts
        while len(cache) > 32:
            del cache[next(iter(cache))]
    return cache[key]


def locations(index, result, available=(), max_chars=900):
    if type(max_chars) is not int or max_chars < 512:
        raise ValueError("navigation budget must be at least512 characters")
    from infinicoder.execution.workspace_tools import Workspace

    expected = result["sha256"]
    file = Workspace(index.root).path(result["path"])
    path = file.relative_to(index.root).as_posix()
    row = index.db.execute(
        "SELECT hash,source,error,language FROM files WHERE path=?", (path,)
    ).fetchone()
    try:
        current = hashlib.sha256(file.read_bytes()).hexdigest()
    except OSError:
        current = None
    if not row or row["hash"] != expected or current != expected:
        return dict(
            available=False, source_sha256=expected, reason="index/read/live source versions differ"
        )
    symbols = [
        dict(v)
        for v in index.db.execute(
            "SELECT id,qualname,kind,line,end,confidence FROM symbols WHERE path=? AND kind IN ('function','class','type') ORDER BY line LIMIT 65",
            (path,),
        )
    ]
    start = result["start"]
    end = start + len(result.get("text", "").splitlines()) - 1
    symbols.sort(
        key=lambda v: (
            not (v["line"] <= end and v["end"] >= start),
            abs(v["line"] - start),
            v["line"],
            v["id"],
        )
    )
    value = dict(
        available=True,
        source_sha256=expected,
        suggestions_only=True,
        symbols=[],
        omitted=len(symbols),
        limitations=[
            "Static locations,not execution,alias or ownership proof.",
            "Inspect relevant documentation too;body locations do not make it irrelevant.",
        ],
    )
    for symbol in symbols:
        begin = symbol["line"]
        finish = symbol["end"]
        if type(begin) is not int or type(finish) is not int or begin < 1 or finish < begin:
            continue
        implementation = (
            _body_starts(index, path, expected, row["source"]).get(begin, begin)
            if row["language"] == "python" and not row["error"]
            else begin
        )
        item = dict(
            symbol_id=symbol["id"],
            kind=symbol["kind"],
            declaration_line=begin,
            end=finish,
            confidence=symbol["confidence"],
        )
        if implementation is not None and begin <= implementation <= finish:
            item["read"] = action(path, implementation, min(60, finish - implementation + 1))
            item["location_kind"] = (
                "python_body_after_leading_docstring"
                if implementation > begin and row["language"] == "python"
                else "indexed_declaration"
            )
        else:
            item["non_docstring_body_available"] = False
        # Do not expose an ambiguous overload id as a precise expansion target.
        if (
            index.db.execute("SELECT COUNT(*) FROM symbols WHERE id=?", (symbol["id"],)).fetchone()[
                0
            ]
            == 1
        ):
            tool = "function_summary" if symbol["kind"] == "function" else "structure"
            if tool in available:
                item["expand"] = dict(tool=tool, arguments=dict(symbol_id=symbol["id"]))
        proposal = dict(value, symbols=[*value["symbols"], item], omitted=value["omitted"] - 1)
        if len(json.dumps(proposal, ensure_ascii=False)) <= max_chars:
            value = proposal
    return value


def continuation(result):
    """Attach an action for the existing ledger's next unread complete line."""
    progress = result.get("inspection_progress", {})
    line = progress.get("next_unread_line")
    if result.get("partial_line") and result.get("covered_line_count") == 0:
        progress["next_read_unavailable"] = (
            "The current line is only partially visible;use a symbol/source view rather than claiming it was read."
        )
    elif type(line) is int and 1 <= line <= result["total_lines"]:
        suggestion = action(result["path"], line, min(60, result["total_lines"] - line + 1))
        proposed = dict(result, inspection_progress=dict(progress, next_read=suggestion))
        if len(json.dumps(proposed, ensure_ascii=False)) <= 5800:
            progress["next_read"] = suggestion
        else:
            progress["next_read_omitted"] = True
    return result


def missing_path(index, requested, max_chars=900):
    """Return actual lexical path candidates;never guess the task's right target."""
    if type(max_chars) is not int or max_chars < 512:
        raise ValueError("navigation budget must be at least512 characters")
    from infinicoder.execution.workspace_tools import Workspace

    workspace = Workspace(index.root)
    path = workspace.path(requested)
    if path.exists():
        return None
    relative = path.relative_to(index.root).as_posix()
    parent = Path(relative).parent.as_posix()
    prefix = "" if parent == "." else parent + "/"
    rows = index.db.execute(
        "SELECT path,hash FROM files WHERE substr(path,1,?)=? ORDER BY path LIMIT 65",
        (len(prefix), prefix),
    ).fetchall()
    if not rows:
        rows = index.db.execute("SELECT path,hash FROM files ORDER BY path LIMIT 65").fetchall()
    known = []
    versions = {}
    for row in rows:
        try:
            file = workspace.path(row["path"])
            if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != row["hash"]:
                continue
        except (OSError, ValueError):
            continue
        known.append(row["path"])
        versions[row["path"]] = row["hash"]
        if len(known) == 6:
            break
    value = dict(
        requested_path=relative,
        known_paths=[],
        source_versions={},
        suggestions_only=True,
        limitation="Existing indexed paths are candidates,not a claim about the correct implementation target.",
    )
    if len(json.dumps(value, ensure_ascii=False)) > max_chars:
        value.pop("requested_path")
        value.update(
            requested_path_omitted=True,
            requested_path_sha256=hashlib.sha256(relative.encode()).hexdigest(),
        )
    for name in known:
        proposal = dict(
            value,
            known_paths=[*value["known_paths"], name],
            source_versions=dict(value["source_versions"], **{name: versions[name]}),
        )
        if len(json.dumps(proposal, ensure_ascii=False)) <= max_chars:
            value = proposal
    return value


RECOVERY_LABEL = "CONTROLLER SOURCE RECOVERY\n"


def recovery_identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def recovery_allowed(runtime, index):
    """Observe existing cancellation/deadline state without consuming control."""
    if (runtime.store.path / "control.json").exists():
        return False
    remaining = getattr(index, "remaining", None)
    if callable(remaining) and remaining() <= 0:
        return False
    owner = getattr(getattr(runtime, "_collaboration", None), "owner", None)
    if owner is not None:
        if owner.cancel_event.is_set() or owner.coding_cancel_event.is_set():
            return False
        if owner.external_cancelled is not None and owner.external_cancelled():
            return False
    return True


def _recovery_json(value):
    return json.dumps(value, ensure_ascii=False)


def _recovery_current(workspace, item):
    path = workspace.path(item["path"])
    return (
        path.is_file()
        and path.stat().st_size <= 2_000_000
        and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    )


def recover_source(
    index, workspace, requested, state, *, protected_paths=(), count_tokens=None, guard=lambda: True
):
    """One exact indexed lookup after a failed read;never dispatch a model action.

    Search rank does not establish uniqueness. No repository imports,semantic
    expansion,source refresh,permission changes or writes occur here.
    """
    if not guard():
        return None
    path = workspace.path(requested)
    if path.exists():
        return None
    blocked = tuple(
        (Path(p) if Path(p).is_absolute() else workspace.root / Path(p)).resolve()
        for p in (*protected_paths, *getattr(index, "excluded_paths", ()))
    )
    if any(path == p or path.is_relative_to(p) for p in blocked):
        return None
    relative = path.relative_to(workspace.root).as_posix()
    if len(relative) > 512:
        return None
    key = hashlib.sha256(
        _recovery_json([relative, state.get("index_revision"), [str(p) for p in blocked]]).encode()
    ).hexdigest()
    cache = state.setdefault("source_recovery_cache", {})
    if key in cache:
        state["source_recovery_pending"] = key
        return cache[key]
    basename = Path(relative).name
    stem = Path(relative).stem
    symbol = stem if re.fullmatch(r"[A-Za-z_]\w*", stem) else ""
    excluded = []
    parameters = []
    for protected in blocked:
        if protected.is_relative_to(workspace.root):
            name = protected.relative_to(workspace.root).as_posix()
            excluded.append("(f.path<>? AND substr(f.path,1,?)<>?)")
            parameters.extend((name, len(name) + 1, name + "/"))
    allowed = " AND ".join(excluded) or "1"
    # File basename first,then case-sensitive identifiers. Case-fold matches
    # remain a separately labelled fallback,never a BM25 top-one decision.
    sql = f"""SELECT * FROM (
      SELECT f.path,f.hash AS sha256,f.error,f.language,f.build,NULL AS symbol_id,
        NULL AS symbol_name,1 AS line,NULL AS end,'file' AS kind,
        CASE WHEN f.path=? OR substr(f.path,-?)=? THEN 0 ELSE 1 END AS rank
      FROM files f WHERE (lower(f.path)=lower(?) OR lower(substr(f.path,-?))=lower(?)) AND {allowed}
      UNION ALL
      SELECT f.path,f.hash AS sha256,f.error,f.language,f.build,s.id AS symbol_id,
        s.name AS symbol_name,s.line,s.end,s.kind,
        CASE WHEN s.name=? THEN 2 ELSE 3 END AS rank
      FROM symbols s JOIN files f ON s.path=f.path
      WHERE ?<>'' AND lower(s.name)=lower(?) AND s.kind IN ('function','class','type') AND {allowed}
    ) ORDER BY rank,path,line,symbol_id LIMIT 4"""
    values = (
        basename,
        len(basename) + 1,
        "/" + basename,
        basename,
        len(basename) + 1,
        "/" + basename,
        *parameters,
        symbol,
        symbol,
        symbol,
        *parameters,
    )
    rows = [dict(row) for row in index.db.execute(sql, values)]
    if not guard():
        return None
    best = rows[0]["rank"] if rows else None
    selected = [row for row in rows if row["rank"] == best]
    packet = dict(
        origin="controller_missing_path_recovery",
        evidence_id=key,
        requested_path=relative,
        resolution_status="not_found",
        candidates=[],
        interpretation="Static source locations only,not the task target,edit permission or proof of understanding.",
    )
    verified = []
    for row in selected[:3]:
        if not guard():
            return None
        try:
            candidate = workspace.path(row["path"])
            if any(candidate == p or candidate.is_relative_to(p) for p in blocked):
                continue
            if not _recovery_current(workspace, row):
                continue
        except (OSError, ValueError):
            continue
        verified.append(row)
        packet["candidates"].append(
            dict(
                path=row["path"],
                sha256=row["sha256"],
                symbol_id=row["symbol_id"],
                declaration_line=row["line"],
                match=(
                    "literal_basename",
                    "casefold_basename",
                    "exact_identifier",
                    "casefold_identifier",
                )[row["rank"]],
                next_read=action(row["path"], row["line"], 40),
            )
        )
    if selected:
        packet["resolution_status"] = (
            "unique"
            if len(selected) == len(verified) == 1
            else "ambiguous"
            if len(selected) > 1
            else "stale"
        )
        if len(selected) > 3:
            packet["candidate_list_incomplete"] = True
    exact = verified[0] if len(selected) == len(verified) == 1 else None
    # A partial native catalog or parse error cannot establish a complete
    # declaration. The model can still choose the ordinary candidate read.
    known = exact is not None and not exact["error"]
    if known and exact["language"] != "python":
        known = bool(json.loads(exact["build"] or "{}").get("semantic"))
    if exact is not None and not known:
        packet["resolution_status"] = "unknown_parse_or_native_coverage"
    if known and callable(count_tokens):
        first = exact["line"]
        last = min(first + 39, exact["end"] or first + 39)
        source = workspace.read(exact["path"], start=first, count=last - first + 1)
        if source["sha256"] != exact["sha256"] or not guard():
            packet["resolution_status"] = "stale"
        else:
            lines = source["text"].splitlines()
            # Bound tokenizer work even on a single enormous source line.
            while lines and len("\n".join(lines).encode("utf-8")) > 16000:
                lines.pop()
            low, high = 0, len(lines)

            def proposal(n):
                body = dict(
                    source,
                    text="\n".join(lines[:n]),
                    covered_line_count=n,
                    complete_declaration=bool(exact["end"] and first + n - 1 >= exact["end"]),
                )
                return dict(packet, supplemental_source=body)

            while low < high:
                middle = (low + high + 1) // 2
                if count_tokens(RECOVERY_LABEL + _recovery_json(proposal(middle))) <= 768:
                    low = middle
                else:
                    high = middle - 1
            if low:
                packet = proposal(low)
            else:
                packet["source_omitted"] = "No complete source line fits the token budget."
    elif known:
        packet["source_omitted"] = "No exact tokenizer is available;locations only."
    counter = count_tokens if callable(count_tokens) else lambda value: len(value.encode("utf-8"))
    while packet["candidates"] and counter(RECOVERY_LABEL + _recovery_json(packet)) > 768:
        packet["candidates"].pop()
    if counter(RECOVERY_LABEL + _recovery_json(packet)) > 768:
        packet = dict(origin=packet["origin"], evidence_id=key, resolution_status="omitted_budget")
    if not guard():
        return None
    cache[key] = packet
    while len(cache) > 8:
        del cache[next(iter(cache))]
    state["source_recovery_pending"] = key
    return packet


def recovery_context(state, workspace, guard=lambda: True):
    """Return a whole optional packet. Omission never removes the original error."""
    from copy import deepcopy

    packet = state.get("source_recovery_cache", {}).get(state.get("source_recovery_pending"))
    state.pop("source_recovery_active_view", None)
    if not packet or not guard():
        return None
    packet = deepcopy(packet)
    try:
        valid = all(_recovery_current(workspace, item) for item in packet.get("candidates", []))
    except (OSError, ValueError):
        valid = False
    if not valid:
        packet.pop("supplemental_source", None)
        packet["resolution_status"] = "stale"
    state["source_recovery_active_view"] = packet
    return RECOVERY_LABEL + _recovery_json(packet)


def delivered_recovery(runtime, state, workspace, index, messages):
    """Record only complete trusted source which reached a responding client."""
    packet = state.get("source_recovery_active_view")
    if not packet:
        return
    text = RECOVERY_LABEL + _recovery_json(packet)
    delivered = any(
        message["role"] == "user" and text in message["content"] for message in messages
    )
    runtime.store.event(
        "source_recovery_delivery",
        dict(
            evidence_id=packet["evidence_id"],
            delivered=delivered,
            reason=None if delivered else "whole_optional_packet_not_in_request",
            model_call=state["model_calls"],
        ),
    )
    state.pop("source_recovery_active_view", None)
    if delivered:
        state.pop("source_recovery_pending", None)
    source = packet.get("supplemental_source")
    if not delivered or not source:
        return
    seen = state.setdefault("source_recovery_presented", [])
    if packet["evidence_id"] in seen:
        return
    try:
        valid = _recovery_current(workspace, source)
    except (OSError, ValueError):
        valid = False
    if not valid:
        return
    from copy import deepcopy
    from infinicoder.memory.inspection_memory import remember_read
    from infinicoder.planning.source_plan import record_planning_read

    source = deepcopy(source)
    remember_read(state, source)
    record_planning_read(state, source, workspace)
    if runtime.task_memory:
        index.record_inspection(source)
    state["read_paths"] = list(dict.fromkeys(state.get("read_paths", []) + [source["path"]]))[-16:]
    seen.append(packet["evidence_id"])
    state["source_recovery_presented"] = seen[-16:]
    runtime.store.event(
        "resolved_source_observation",
        dict(
            origin="controller_missing_path_recovery",
            evidence_id=packet["evidence_id"],
            presented=True,
            source=source,
            model_read_action=False,
        ),
    )
