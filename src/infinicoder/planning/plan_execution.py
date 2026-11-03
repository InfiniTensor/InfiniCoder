"""Controller-owned source leases, phase observations and suffix replanning.

Observed writes/version transitions are not semantic correctness proofs. Prefix
completion is historical;all independent caller checks still run at the end.
"""

import copy
from dataclasses import asdict
import hashlib
import json


def identity():
    from pathlib import Path

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _hash(workspace, path):
    try:
        file = workspace.path(path)
        return hashlib.sha256(file.read_bytes()).hexdigest() if file.is_file() else None
    except (OSError, ValueError):
        return None


def _digest(certificate):
    body = {k: v for k, v in certificate.items() if k != "digest"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def fingerprints(certificate, phase, workspace):
    if not certificate or not 0 <= phase < len(certificate["units"]):
        return {}
    return {path: _hash(workspace, path) for path in certificate["units"][phase]["write_set"]}


def failure_paths(result, certificate, phase, workspace):
    """Reported diagnostic locations are candidates,not proven root causes."""
    stack = [result]
    paths = []
    while stack:
        row = stack.pop()
        if isinstance(row, list):
            stack.extend(row)
            continue
        if not isinstance(row, dict):
            continue
        for diagnostic in row.get("diagnostics", []):
            if not isinstance(diagnostic, dict) or not isinstance(diagnostic.get("path"), str):
                continue
            try:
                paths.append(
                    workspace.path(diagnostic["path"]).relative_to(workspace.root).as_posix()
                )
            except (ValueError, OSError):
                pass
        for key in ("evidence", "last_failure"):
            if key in row:
                stack.append(row[key])
    if not paths and certificate and 0 <= phase < len(certificate["units"]):
        paths = certificate["units"][phase]["write_set"]
    return list(dict.fromkeys(paths))[:48]


def record_write(state, workspace, path, expected_hash, phase, origin="hash_guarded_edit"):
    execution = state.get("phase_execution")
    if execution is None or _hash(workspace, path) != expected_hash:
        return False
    certificate = state.get("source_plan_certificate", {})
    if (
        not 0 <= phase < len(certificate.get("units", []))
        or path not in certificate["units"][phase]["write_set"]
    ):
        return False
    execution["writes"][path] = dict(path=path, sha256=expected_hash, phase=phase, origin=origin)
    return True


def preflight(certificate, execution, phase, workspace):
    """Check current phase only;known writes advance versions,not old summaries."""
    if not certificate or not execution or not 0 <= phase < len(certificate["units"]):
        return dict(valid=True, sources=[], changed=[], invalid=[])
    if execution["certificate"] != certificate["digest"]:
        raise ValueError("phase execution belongs to a different plan certificate")
    sources, changed, invalid = [], [], []
    seen = set()
    for fact in certificate["units"][phase]["sources"]:
        path = fact["path"]
        if path in seen:
            continue
        seen.add(path)
        current = _hash(workspace, path)
        expected = fact.get("sha256")
        observed = execution["writes"].get(path)
        own = observed and observed["sha256"] == current and observed["phase"] <= phase
        if (
            current is None
            and own
            and observed["phase"] == phase
            and path in certificate["units"][phase]["write_set"]
        ):
            row = dict(
                path=path,
                source_sha256=None,
                status="deletion_observed_after_runtime_action",
                role="write",
            )
            sources.append(row)
            changed.append(row)
            continue
        if expected is None and current is None and fact["origin"] == "declared_new_file":
            sources.append(dict(path=path, status="declared_not_yet_created", source_sha256=None))
            continue
        if current is not None and (current == expected or own):
            row = dict(
                path=path, source_sha256=current, status="current_source_version", role=fact["role"]
            )
            if current != expected:
                row.update(
                    origin=observed["origin"],
                    prior_source_sha256=expected,
                    rule="Earlier source evidence is invalidated;inspect current source. This version observation proves no semantics.",
                )
                changed.append(row)
            sources.append(row)
        else:
            invalid.append(
                dict(
                    path=path,
                    expected_sha256=expected,
                    actual_sha256=current,
                    reason="missing planned artifact"
                    if expected is None
                    else "unrecognized source change/deletion",
                )
            )
    return dict(valid=not invalid, sources=sources, changed=changed, invalid=invalid)


def context(preflight_result, phase, execution):
    rows = preflight_result["sources"]
    result = dict(
        phase=phase,
        sources=rows[:6],
        omitted_sources=max(0, len(rows) - 6),
        changed_versions=[
            dict(path=r["path"], source_sha256=r["source_sha256"])
            for r in preflight_result["changed"][:6]
        ],
        prior_phase_observations=[
            dict(phase=r["phase"], independent_phase_check=r["independent_phase_check"])
            for r in execution.get("completed", [])[-4:]
        ],
        rule="Controller version observations,not model inspection or semantic proof. Current source supersedes old excerpts;final caller checks remain mandatory.",
    )
    if execution.get("replan_observation"):
        result["historical_failure"] = execution["replan_observation"]
    return result


def complete_phase(state, workspace, phase, evidence, provisional=False):
    execution = state.get("phase_execution")
    certificate = state.get("source_plan_certificate")
    if not execution or not certificate:
        return
    row = dict(
        phase=phase,
        outputs=fingerprints(certificate, phase, workspace),
        independent_phase_check=not provisional,
        evidence=copy.deepcopy(evidence),
    )
    execution["completed"] = [r for r in execution["completed"] if r["phase"] != phase] + [row]


def begin_replan(state, spec, workspace, reason, paths=()):
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 1500:
        raise ValueError("replan requires a concrete reason of1..1500 characters")
    if not isinstance(paths, (list, tuple)) or len(paths) > 48:
        raise ValueError("affected_paths must contain at most48 concrete paths")
    if spec.milestones and not state.get("plan_milestones"):
        raise ValueError("caller milestones cannot be replaced by a model replan")
    from infinicoder.planning.source_plan import _path

    paths = list(dict.fromkeys(_path(workspace, p)[0] for p in paths))
    certificate = state.get("source_plan_certificate", {})
    units = certificate.get("units", [])
    current = min(state["milestone"], max(0, len(units) - 1))
    affected = [i for i, u in enumerate(units) if set(paths) & {f["path"] for f in u["sources"]}]
    first = min([current, *affected]) if paths else current
    pending = state.get("pending_replan")
    if pending:
        first = min(first, pending["from_phase"])
    old_plan = state.get("plan_milestones", [])
    event = dict(
        from_phase=first,
        failed_phase=state["milestone"],
        reason=reason,
        affected_paths=paths,
        previous_certificate=certificate.get("digest"),
        retained_prefix=first,
        semantics="Prefix completion is historical;tail may revise interfaces and all final checks are re-executed.",
    )
    state["pending_replan"] = dict(
        event,
        prefix=copy.deepcopy(old_plan[:first]),
        prefix_certificate=copy.deepcopy(units[:first]),
    )
    # Plan installation clears the current failure. Carry the discriminating
    # observation into later workers without presenting it as a current result.
    from infinicoder.runtime.feedback import failure_packet

    observed = failure_packet(state.get("last_failure"), limit=1600)
    prior_failure = (pending or {}).get("failure_context") or state.get("phase_execution", {}).get(
        "replan_observation"
    )
    if (
        prior_failure
        and any(r.get("check_id") for r in prior_failure["evidence"])
        and not any(r.get("check_id") for r in observed or [])
    ):
        # A later plan-envelope error must not erase the external observation
        # that originally made the implementation plan fail.
        state["pending_replan"]["failure_context"] = copy.deepcopy(prior_failure)
    elif observed:
        raw = state["last_failure"]
        records = raw if isinstance(raw, list) else raw.get("evidence", [raw])
        ids = {r.get("check_id") for r in observed}
        versions = [
            {k: r[k] for k in ("check_id", "revision", "checker") if k in r}
            for r in records
            if isinstance(r, dict) and r.get("check_id") in ids
        ][:2]
        state["pending_replan"]["failure_context"] = dict(
            kind="historical_failure_observation",
            evidence=observed,
            source_versions=versions,
            from_certificate=certificate.get("digest"),
            rule="Observed on an earlier source version;not current correctness evidence. Recheck after implementation.",
        )
    elif prior_failure:
        state["pending_replan"]["failure_context"] = copy.deepcopy(prior_failure)
    state["plan_invalidations"] = [*state.get("plan_invalidations", []), event][-12:]
    state.update(
        planning_required=True,
        milestone=first,
        evidence=[],
        failures={},
        no_progress={},
        plan_protocol_failures=0,
        replans=state.get("replans", 0) + 1,
    )
    for key in ("fallback_path", "worker_review", "worker_review_key", "memory_focus"):
        state.pop(key, None)
    state["progress_hint"] = (
        "Replan from phase "
        + str(first)
        + ": "
        + reason
        + ". Preserve the caller contract and review changed source versions."
    )
    return event


def install(planned, certificate, state):
    """Append a freshly grounded tail after controller-retained historical phases."""
    from infinicoder.planning.task_contract import TaskSpec

    pending = state.get("pending_replan", {})
    prefix = copy.deepcopy(pending.get("prefix", []))
    first = len(prefix)
    if first + len(planned.milestones) > 12:
        raise ValueError("retained prefix and new implementation tail exceed12 phases")
    result = TaskSpec.from_dict(asdict(planned))
    result.milestones = prefix + result.milestones
    # The fresh tail already binds every final caller check. Validate again
    # after combining so prefix retention cannot remove or replace that gate.
    result = TaskSpec.from_dict(asdict(result))
    certificate = copy.deepcopy(certificate)
    certificate["units"] = (
        copy.deepcopy(pending.get("prefix_certificate", [])) + certificate["units"]
    )
    if pending:
        certificate["replan"] = {
            k: v
            for k, v in pending.items()
            if k not in ("prefix", "prefix_certificate", "failure_context")
        }
    certificate["digest"] = _digest(certificate)
    old = state.get("phase_execution", {})
    execution = dict(
        certificate=certificate["digest"],
        writes={
            p: copy.deepcopy(r) for p, r in old.get("writes", {}).items() if r["phase"] < first
        },
        completed=[copy.deepcopy(r) for r in old.get("completed", []) if r["phase"] < first],
    )
    if pending.get("failure_context"):
        execution["replan_observation"] = copy.deepcopy(pending["failure_context"])
    return result, certificate, first, execution
