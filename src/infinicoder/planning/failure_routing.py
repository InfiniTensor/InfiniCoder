"""Late,version-bound diagnosis guidance from independent verification evidence."""

from infinicoder._resources import package_root

import hashlib
import json
from infinicoder.memory.repair_memory import view as observed_failure
from infinicoder.analysis.task_compiler import classify_failure


def identity():
    root = package_root()
    return hashlib.sha256(
        b"".join(
            (root / name).read_bytes()
            for name in (
                "planning/failure_routing.py",
                "memory/repair_memory.py",
                "runtime/shell_loop.py",
                "runtime/engine.py",
            )
        )
    ).hexdigest()


def view(state, spec, workspace):
    """Model notes,old revisions and stdout PASSED cannot request this route."""
    observed = observed_failure(state, spec, workspace)
    if not observed or observed.get("status") != "observed_failure_on_current_source":
        return None
    current = observed["current_revision"]
    key = str(state["milestone"]) + ":" + current
    rejected = state.get("rejected_revisions", {}).get(key, 0)
    if rejected < 2:
        return None
    checker = hashlib.sha256(
        json.dumps(state.get("checker_files", {}), sort_keys=True).encode()
    ).hexdigest()
    if observed.get("checker_identity") != checker:
        return None
    category, _ = classify_failure(dict(evidence=observed.get("failure", [])))
    experiment = {
        "semantic": "Run a small boundary or differential example derived from the public requirement;show expected and actual intermediate values.",
        "compilation": "Inspect the reported source/import and run the smallest relevant compile or import check.",
        "timeout": "Run a bounded reproduction that distinguishes slow progress from a hang.",
        "specification": "Inspect the current interface and caller-owned allowed scope before proposing a repair.",
    }.get(
        category,
        "Inspect the relevant source and run a bounded experiment that distinguishes two plausible explanations.",
    )
    return dict(
        route="discriminating_observation",
        failure_id=observed["observation_id"],
        source_revision=current,
        checker_identity=checker,
        rejected_same_revision=rejected,
        failure_category=category if category != "internal" else "unclassified",
        next=experiment,
        avoid="Do not resubmit the identical failed replacement or repeat verify on unchanged source. First obtain new evidence,then make a justified repair.",
        constraints="Use existing read/command tools and public requirements;do not access protected checks. Predictions are hypotheses,not facts. Final independent acceptance remains required.",
    )
