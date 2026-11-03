"""Optional session-local public command observations;never acceptance proofs.

Reported locations and model-selected programs are untrusted. Boundary hashes
do not prove which source ran inside a command. Caller-declared mount aliases
only resolve paths;they confer no command permissions or verification status.
"""

from infinicoder._resources import package_root

from copy import deepcopy
import hashlib, json, re
from pathlib import Path, PurePosixPath
from infinicoder.memory.repair_memory import signature
from infinicoder.memory.revision_feedback import revision

_TRACE = re.compile(r"^\s*File [\"\'](.+?)[\"\'], line (\d+)(?:,.*)?$")
_PYTEST = re.compile(r"^\s*(.+?\.py):(\d+)(?::(?:\d+:)?(?:.*))?$")
_GNU = re.compile(r"^(.+?):(\d+)(?::\d+)?: (?:fatal )?error: (.*)$")
_NVCC = re.compile(r"^(.+?)\((\d+)\): (?:fatal )?error(?: #\S+)?: (.*)$")
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def identity():
    root = package_root()
    return dict(
        version=1,
        sha256=hashlib.sha256(
            b"".join(
                (root / n).read_bytes()
                for n in (
                    "runtime/command_evidence.py",
                    "runtime/engine.py",
                    "runtime/shell_loop.py",
                    "memory/revision_feedback.py",
                )
            )
        ).hexdigest(),
    )


def aliases(value):
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or len(value) > 8:
        raise ValueError("command workspace aliases need at most8 absolute POSIX mount roots")
    result = []
    for item in value:
        if (
            not isinstance(item, str)
            or len(item) > 512
            or not item.startswith("/")
            or "\\" in item
            or any(ord(c) < 32 for c in item)
        ):
            raise ValueError("command workspace alias must be an absolute POSIX mount root")
        path = PurePosixPath(item)
        if not path.is_absolute() or ".." in path.parts or str(path) == "/":
            raise ValueError("bounded non-root POSIX mount alias required")
        normalized = str(path)
        if normalized in result:
            raise ValueError("duplicate workspace alias")
        result.append(normalized)
    return tuple(result)


def _path(reported, workspace, mounts):
    if any(ord(c) < 32 for c in reported) or reported.startswith("<"):
        raise ValueError("non-source reported location")
    raw = Path(reported)
    if raw.is_absolute():
        try:
            return raw.resolve().relative_to(workspace.root).as_posix(), "workspace_absolute"
        except ValueError:
            pass
    posix = PurePosixPath(reported)
    if posix.is_absolute():
        if ".." in posix.parts:
            raise ValueError("reported path traverses mount")
        for prefix in sorted(mounts, key=len, reverse=True):
            try:
                return posix.relative_to(
                    PurePosixPath(prefix)
                ).as_posix(), "caller_declared_mount_alias"
            except ValueError:
                pass
        raise ValueError("unmapped absolute reported path")
    if raw.is_absolute():
        raise ValueError("reported path outside workspace")
    return reported, "workspace_relative"


def locations(output, workspace, mounts):
    matches = []
    for line in _ANSI.sub("", output).splitlines():
        match = _TRACE.match(line) or _PYTEST.match(line) or _GNU.match(line) or _NVCC.match(line)
        if match:
            matches.append((match.group(1), int(match.group(2))))
    resolved = []
    unresolved = []
    seen = set()
    # Deepest reported in-workspace frame first. Never guess an alias by basename.
    for reported, line in reversed(matches):
        if (reported, line) in seen:
            continue
        seen.add((reported, line))
        item = dict(reported_path=reported[:160], line=line, mapping="unresolved")
        try:
            relative, basis = _path(reported, workspace, mounts)
            source = workspace.read(relative, max(1, line - 2), 5)
            if line < 1 or line > source["total_lines"]:
                raise ValueError("reported line outside current file")
            item.update(
                path=relative,
                mapping=basis,
                sha256=source["sha256"],
                start=source["start"],
                source_excerpt=source["text"][:240],
                excerpt_truncated=len(source["text"]) > 240,
            )
        except (ValueError, OSError, UnicodeError):
            pass
        if item.get("path"):
            resolved.append(item)
        elif len(unresolved) < 2:
            unresolved.append(item)
        if len(resolved) == 2:
            break
    return (resolved + unresolved)[:2]


def observe(state, spec, workspace, argv, before, result, mounts=()):
    """Record the existing process result;never run another command or check."""
    memory = state.setdefault(
        "public_command_memory", dict(failures=[], observed_commands=0, omitted_failures=0)
    )
    memory["observed_commands"] += 1
    command_hash = hashlib.sha256(json.dumps(argv, ensure_ascii=False).encode()).hexdigest()
    failed = result["exit_code"] != 0 or bool(result.get("timed_out") or result.get("truncated"))
    requirements = signature(spec)
    phase = state["milestone"]
    relevant = [
        r
        for r in memory["failures"]
        if (r["requirements"], r["phase"], r["command_sha256"])
        == (requirements, phase, command_hash)
    ]
    if not failed and not relevant:
        return None
    after = revision(spec, workspace)
    if not failed:
        for row in relevant:
            if row["before_revision"] == row["after_revision"] == before == after:
                row["later_exit0_same_boundaries"] = row.get("later_exit0_same_boundaries", 0) + 1
                row["last_exit0_call"] = state.get("model_calls", 0)
        return None
    output = result.get("output", "")
    output_hash = hashlib.sha256(output.encode()).hexdigest()
    full = json.dumps(argv, ensure_ascii=False)
    record = dict(
        requirements=requirements,
        phase=phase,
        command_sha256=command_hash,
        command_preview=full[:220],
        command_preview_truncated=len(full) > 220,
        before_revision=before,
        after_revision=after,
        boundary_revisions_equal=before == after,
        execution_source_proven=False,
        exit_code=result["exit_code"],
        timed_out=bool(result.get("timed_out")),
        output_truncated=bool(result.get("truncated")),
        visible_output_sha256=output_hash,
        output_excerpt=_ANSI.sub("", output)[-480:],
        locations=locations(output, workspace, mounts),
        model_call=state.get("model_calls", 0),
        repeat_count=1,
        acceptance_proof=False,
        origin="model_selected_public_command;output_and_reported_locations_untrusted;not_independent_verification",
    )
    fingerprint = [
        requirements,
        phase,
        command_hash,
        before,
        after,
        record["exit_code"],
        record["timed_out"],
        record["output_truncated"],
        output_hash,
    ]
    record["observation_id"] = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True).encode()
    ).hexdigest()
    duplicate = next(
        (r for r in memory["failures"] if r["observation_id"] == record["observation_id"]), None
    )
    if duplicate:
        duplicate["repeat_count"] += 1
        duplicate["last_repeat_call"] = record["model_call"]
        return deepcopy(duplicate)
    memory["failures"].append(record)
    while len(memory["failures"]) > 4:
        memory["failures"].pop(0)
        memory["omitted_failures"] += 1
    return deepcopy(record)


def view(state, spec, workspace, max_chars=1400):
    if type(max_chars) is not int or max_chars < 320:
        raise ValueError("command evidence view needs at least320 chars")
    rows = [
        r
        for r in state.get("public_command_memory", {}).get("failures", [])
        if r["requirements"] == signature(spec) and r["phase"] == state["milestone"]
    ]
    if not rows:
        return None
    current = revision(spec, workspace)
    chosen = next((r for r in reversed(rows) if r["after_revision"] == current), rows[-1])
    same = chosen["after_revision"] == current
    result = {
        k: deepcopy(chosen[k])
        for k in (
            "observation_id",
            "exit_code",
            "timed_out",
            "output_truncated",
            "command_preview",
            "command_preview_truncated",
            "output_excerpt",
            "locations",
            "repeat_count",
            "boundary_revisions_equal",
            "execution_source_proven",
        )
    }
    result.update(
        origin="PUBLIC COMMAND OBSERVATION;not independent acceptance or a proven cause",
        source_state="same_post_command_boundary" if same else "historical_boundary_changed",
        observed_revision=chosen["after_revision"],
        later_exit0_same_boundaries=chosen.get("later_exit0_same_boundaries", 0),
    )
    if not same:
        for item in result["locations"]:
            item.pop("source_excerpt", None)
            item["historical_source"] = True
    for item in result["locations"]:
        if item.get("path"):
            try:
                item["source_current"] = (
                    hashlib.sha256(workspace.path(item["path"]).read_bytes()).hexdigest()
                    == item["sha256"]
                )
            except (ValueError, OSError):
                item["source_current"] = False
            if not item["source_current"]:
                item.pop("source_excerpt", None)
    return _bounded(result, max_chars)


def _bounded(value, max_chars):
    result = deepcopy(value)
    while len(json.dumps(result, ensure_ascii=False)) > max_chars:
        if any("source_excerpt" in r for r in result.get("locations", [])):
            for item in result["locations"]:
                item.pop("source_excerpt", None)
            result["source_excerpts_omitted"] = True
        elif len(result.get("output_excerpt", "")) > 120:
            result["output_excerpt"] = result["output_excerpt"][-120:]
            result["output_excerpt_shortened"] = True
        elif result.get("locations"):
            result["locations"].pop()
            result["locations_omitted"] = True
        elif "command_preview" in result:
            result.pop("command_preview")
            result["command_omitted"] = True
        elif "output_excerpt" in result:
            result.pop("output_excerpt")
            result["output_omitted"] = True
        else:
            result = {k: result[k] for k in ("observation_id", "exit_code", "source_state")}
            result.update(origin="public command;not acceptance", details_omitted=True)
            break
    return result


def pack(runtime, spec, state, workspace, max_tokens):
    """Optional actual-tokenizer-budgeted packet;caller contract retains priority."""
    delivery = dict(included=False, input_tokens=0, budget=max_tokens)
    full = view(state, spec, workspace)
    for chars in (1400, 1000, 700, 450, 320) if full else ():
        observed = _bounded(full, chars)
        text = "\nPUBLIC COMMAND DIAGNOSTIC MEMORY\n" + json.dumps(observed, ensure_ascii=False)
        cost = runtime.assembler.estimator.estimate(text)
        if cost <= max_tokens:
            delivery.update(
                included=True, input_tokens=cost, observation_id=observed["observation_id"]
            )
            state["command_evidence_delivery"] = delivery
            return text
    state["command_evidence_delivery"] = delivery
    return ""
