"""One existing patch submission,leased to an observed public failure.

No model,checker,planner or editor lives here. The ordinary Runtime owns all
requests and commits;this module only changes one eligible output protocol.
"""

from infinicoder._resources import package_root

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from infinicoder.verification.check_identity import checker_files
from infinicoder.memory.repair_memory import signature, view
from infinicoder.execution.workspace_tools import Workspace

POLICY = "revision_patch_v1"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def identity():
    root = package_root()
    return hashlib.sha256(
        b"".join(
            (root / name).read_bytes()
            for name in (
                "memory/repair_submission.py",
                "memory/repair_memory.py",
                "runtime/worker.py",
                "runtime/work_packet.py",
                "runtime/engine.py",
            )
        )
    ).hexdigest()


def capture(spec, state, workspace, passed, observation, live_before):
    """Called only after the original caller-owned _verify has returned."""
    if passed:
        if view(state, spec, workspace, state.get("last_failure")) is None:
            state.pop("repair_submission_public_lease", None)
        return
    known_checks = {check.id for check in spec.checks}
    failed = set(observation.get("failed_checks", ())) if observation else set()
    if (
        not observation
        or observation.get("tested_revision") is None
        or not failed.intersection(known_checks)
        or live_before != workspace.revision()
    ):
        state.pop("repair_submission_public_lease", None)
        return
    lease = dict(
        requirements=signature(spec),
        phase=state["milestone"],
        live_revision=live_before,
        tested_snapshot_revision=observation["tested_revision"],
        observation_id=observation["observation_id"],
        receipt_sha256=observation["receipt_sha256"],
        checker_identity=observation["checker_identity"],
        failed_checks=sorted(failed.intersection(known_checks)),
        origin="original_caller_verify;live_revision_captured_before_and_after_check",
    )
    lease["lease_id"] = digest(lease)
    state["repair_submission_public_lease"] = lease


def _current(spec, state, workspace, lease):
    if not lease or lease.get("requirements") != signature(spec):
        return False
    if lease.get("phase") != state["milestone"] or workspace.revision() != lease.get(
        "live_revision"
    ):
        return False
    current_checkers = checker_files(spec, workspace)
    checker_identity = hashlib.sha256(
        json.dumps(current_checkers, sort_keys=True).encode()
    ).hexdigest()
    if (
        current_checkers != state.get("checker_files", {})
        or checker_identity != lease["checker_identity"]
    ):
        return False
    evidence = view(state, spec, workspace, state.get("last_failure"))
    return bool(
        evidence
        and evidence["observation_id"] == lease["observation_id"]
        and evidence["receipt_sha256"] == lease["receipt_sha256"]
        and evidence["tested_revision"] == lease["tested_snapshot_revision"]
    )


def select(runtime, spec, state, workspace, path, source, unit, *, candidate_format=None):
    """Never chooses a file. Runtime must already have selected its worker."""
    if (
        not runtime.repair_submission_policy
        or state.get("repair_submission_attempt")
        or not state.get("model_calls", 0)
    ):
        return None
    if (
        candidate_format is not None
        or runtime._worker_scope
        or runtime.review_before_repair
        or Path(path).suffix != ".py"
        or runtime.optimization is not None
    ):
        return None
    competing = (
        "source_unit_policy",
        "implementation_provider",
        "problem_workflow",
        "binding_candidates",
        "source_focus",
        "native_plan",
        "design_contracts",
        "design_probes",
        "observed_edits",
    )
    if any(getattr(runtime, name, None) for name in competing):
        return None
    if (
        runtime._route.profile == "operator"
        and runtime.domain_compiler
        and not state.get("domain_compiler_attempted")
    ):
        return None  # A local producer must keep its own original protocol.
    if any(check.output_paths for check in spec.checks):
        return None  # This first lease does not infer an excluded snapshot's source set.
    lease = state.get("repair_submission_public_lease")
    try:
        scoped = Workspace(workspace.root, unit.write_set or spec.allowed_paths)
        target = scoped.path(path, write=True)
        chain = [workspace.root / path]
        while chain[-1] != workspace.root:
            chain.append(chain[-1].parent)
        if any(p.is_symlink() or hasattr(p, "is_junction") and p.is_junction() for p in chain):
            return None
        if not target.is_file() or target.is_symlink() or target.read_bytes() != source.encode():
            return None
        ast.parse(source, filename=path)
        if not _current(spec, state, workspace, lease):
            return None
    except (ValueError, OSError, SyntaxError, UnicodeError):
        return None
    selection = dict(
        policy=POLICY,
        path=path,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        public_lease=deepcopy(lease),
        acceptance_proof=False,
        payload_fit="unknown_until_model_returns;original_output_cap_retained",
    )
    selection["selection_id"] = digest(selection)
    return selection


def validate(spec, state, workspace, selection):
    try:
        target = workspace.path(selection["path"], write=True)
        valid = (
            _current(spec, state, workspace, selection["public_lease"])
            and hashlib.sha256(target.read_bytes()).hexdigest() == selection["source_sha256"]
        )
    except (ValueError, OSError, KeyError):
        valid = False
    if not valid:
        raise ValueError("Public-failure source/checker lease changed;patch was not applied")


def claim(runtime, spec, state, workspace):
    selection = runtime._repair_submission_selection
    validate(spec, state, workspace, selection)
    if state.get("repair_submission_attempt"):
        raise ValueError("The session already used its one revision-patch opportunity")
    attempt = dict(
        selection_id=selection["selection_id"],
        path=selection["path"],
        source_sha256=selection["source_sha256"],
        lease_id=selection["public_lease"]["lease_id"],
        model_call=state.get("model_calls", 0) + 1,
        outcome="requested",
    )
    state["repair_submission_attempt"] = attempt
    runtime._repair_submission_active = True
    runtime.store.event("repair_submission_requested", attempt)


def applied(runtime, state, result):
    runtime._repair_submission_committed = True
    state["repair_submission_attempt"].update(
        outcome="applied", before=result["before"], after=result["after"]
    )
    runtime.store.event("repair_submission_applied", deepcopy(state["repair_submission_attempt"]))


def rejected(runtime, state, *, outcome="rejected_without_write", failure_origin="proposal"):
    attempt = state.get("repair_submission_attempt")
    if not attempt or runtime._repair_submission_committed:
        return
    attempt["outcome"] = outcome
    attempt["failure_origin"] = failure_origin
    unit = runtime._graph.unit(runtime._graph.order[state["milestone"]])
    # An explicit caller source worker keeps its original scope/protocol.
    # Automatic recovery returns to the existing ordinary action channel.
    if not runtime._explicit_worker_path and not unit.worker_path:
        blocked = state.setdefault("source_worker_blocked_paths", [])
        if attempt["path"] not in blocked:
            blocked.append(attempt["path"])
        state.pop("fallback_path", None)
        attempt["fallback"] = "ordinary_actions"
    else:
        attempt["fallback"] = "original_caller_source_worker"
    runtime.store.event("repair_submission_rejected", deepcopy(attempt))


def not_applied_control(runtime, state, control):
    attempt = state.get("repair_submission_attempt")
    if attempt and not runtime._repair_submission_committed:
        attempt.update(outcome="not_applied_control", control=control)
        runtime.store.event("repair_submission_not_applied", deepcopy(attempt))
