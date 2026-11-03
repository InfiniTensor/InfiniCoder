"""Session-local independent repair observations,kept separate from code facts."""

from infinicoder._resources import package_root

from dataclasses import asdict
from copy import deepcopy
import hashlib
import json
from infinicoder.runtime.feedback import failure_packet, failed_observation


def identity():
    root = package_root()
    return hashlib.sha256(
        b"".join(
            (root / name).read_bytes()
            for name in (
                "memory/repair_memory.py",
                "runtime/feedback.py",
                "runtime/work_packet.py",
                "runtime/engine.py",
                "runtime/worker.py",
            )
        )
    ).hexdigest()


def signature(spec):
    value = asdict(spec)
    value.pop("milestones", None)
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def observe(state, spec, ok, evidence, check_ids, require_artifacts):
    """Called only around the caller-owned verifier,never from model notes."""
    bound = [row for row in evidence if isinstance(row, dict) and row.get("revision")]
    if not bound:
        return None
    revisions = {row["revision"] for row in bound}
    revision = next(iter(revisions)) if len(revisions) == 1 else None
    failures = failure_packet(evidence, limit=1400) if not ok else None
    if not ok and failures is None:
        failures = [
            {key: row[key] for key in ("check_id", "accepted", "reason", "error") if key in row}
            for row in bound
            if row.get("accepted") is False or row.get("error")
        ][:2]
    record = dict(
        requirements=signature(spec),
        phase=state["milestone"],
        tested_revision=revision,
        passed=bool(ok),
        checks=list(check_ids),
        artifacts_required=bool(require_artifacts),
        failure=failures,
        observed_checks=[
            r["check_id"] for r in evidence if isinstance(r, dict) and r.get("check_id")
        ],
        failed_checks=list(
            dict.fromkeys(
                r["check_id"] for r in evidence if failed_observation(r) and r.get("check_id")
            )
        ),
        checker_identity=hashlib.sha256(
            json.dumps(state.get("checker_files", {}), sort_keys=True).encode()
        ).hexdigest(),
        receipt_sha256=hashlib.sha256(
            json.dumps(evidence, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        model_calls_observed=state.get("model_calls", 0),
    )
    record["observation_id"] = hashlib.sha256(
        json.dumps(record, sort_keys=True).encode()
    ).hexdigest()
    memory = state.setdefault("repair_evidence", dict(observations=[], omitted=0))
    memory["observations"].append(record)
    if len(memory["observations"]) > 8:
        memory["observations"].pop(0)
        memory["omitted"] += 1
    return record


def view(state, spec, workspace, last_failure=None, max_chars=2300):
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 128:
        raise ValueError("repair view max_chars must be an int >=128")
    memory = state.get("repair_evidence", {})
    records = memory.get("observations", [])
    if not records:
        return None
    requirements = signature(spec)
    current = workspace.revision()
    chosen = None
    # A passing subset cannot erase an independently failing larger check set.
    ordered = sorted(
        enumerate(records),
        key=lambda pair: (pair[1]["tested_revision"] == current, pair[0]),
        reverse=True,
    )
    for i, record in ordered:
        if record["requirements"] != requirements or record["passed"]:
            continue
        superseded = any(
            record["tested_revision"] is not None
            and later["requirements"] == requirements
            and later["checker_identity"] == record["checker_identity"]
            and later["tested_revision"] == record["tested_revision"]
            and later["passed"]
            and (
                set(later["observed_checks"]) >= set(record["failed_checks"])
                if record["failed_checks"]
                else set(later["checks"]) >= set(record["checks"])
            )
            and (not record["artifacts_required"] or later["artifacts_required"])
            for later in records[i + 1 :]
        )
        if not superseded:
            chosen = record
            break
    if chosen is None:
        return None
    same = chosen["tested_revision"] is not None and chosen["tested_revision"] == current
    result = {
        key: deepcopy(chosen[key])
        for key in (
            "observation_id",
            "tested_revision",
            "phase",
            "checks",
            "checker_identity",
            "receipt_sha256",
        )
    }
    result.update(
        status="observed_failure_on_current_source"
        if same
        else "historical_failure_source_changed",
        current_revision=current,
        failure=deepcopy(chosen["failure"]),
        observations_omitted=memory.get("omitted", 0),
        provenance="caller-owned independent verification;not a code fact,model hypothesis or completion proof",
    )
    if chosen["failure"] == failure_packet(last_failure, limit=1400):
        result.pop("failure")
        result["failure_already_in_last_failure"] = True
    if same:
        result["next"] = (
            "Change the failed implementation or obtain a discriminating observation;unchanged source and repeated finish do not establish correctness."
        )
    else:
        result["next"] = (
            "Diagnostic history only;re-run independent checks before applying it to changed source."
        )
    while len(json.dumps(result, ensure_ascii=False)) > max_chars:
        if result.get("failure"):
            result["failure"].pop()
            result["failure_omitted"] = True
        elif result.get("checks"):
            result["checks"].pop()
            result["check_ids_omitted"] = result.get("check_ids_omitted", 0) + 1
        else:
            compact = dict(
                observation_id=result["observation_id"],
                tested_revision=result["tested_revision"],
                status=result["status"],
                details_omitted=True,
            )
            if len(json.dumps(compact, ensure_ascii=False)) <= max_chars:
                return compact
            return dict(observation_id=result["observation_id"], details_omitted=True)
    return result
