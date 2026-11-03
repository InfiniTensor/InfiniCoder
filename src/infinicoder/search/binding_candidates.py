"""Turn one bounded,source-derived binding hypothesis into a checked candidate.

No task IDs,reference solutions,candidate execution or model requests. The
original Runtime alone edits and verifies;only its full final checks accept.
"""

from infinicoder._resources import package_root

import ast
import copy
from dataclasses import dataclass
import hashlib
import json
import time
from infinicoder.planning.example_diagnosis import analyze, functions
from infinicoder.memory.repair_memory import signature


def identity():
    root = package_root()
    return hashlib.sha256(
        b"".join(
            (root / n).read_bytes()
            for n in (
                "search/binding_candidates.py",
                "planning/example_diagnosis.py",
                "runtime/engine.py",
            )
        )
    ).hexdigest()


@dataclass(frozen=True)
class Proposal:
    source: str
    module_before: str
    prior_failure: dict
    receipt: dict


def propose(runtime, spec, state, workspace, path):
    if (
        not path.endswith(".py")
        or not spec.checks
        or spec.amendments
        or state.get("clarifications")
        or runtime.optimization is not None
        or state.get("edit_focus")
        or runtime.shell_loop
        or state["milestone"] != len(runtime._graph.order) - 1
        or state.get("binding_candidate_count", 0) >= 2
    ):
        return None
    unit = runtime._graph.unit(runtime._graph.order[state["milestone"]])
    if set(unit.check_ids) != {c.id for c in spec.checks}:
        return None
    module = workspace.path(path).read_bytes()
    digest = hashlib.sha256(module).hexdigest()
    failure = state.get("last_failure")
    if (
        not isinstance(failure, dict)
        or failure.get("path") != path
        or failure.get("after") != digest
        or failure.get("accepted") is not False
        or failure.get("binding_failure_requirements") != signature(spec)
    ):
        return None
    outputs = list(dict.fromkeys(p for c in spec.checks for p in c.output_paths))
    revision = workspace.revision(outputs)
    if not any(
        e.get("revision") == revision
        and e.get("exit_code") not in (None, 0, 124)
        and e.get("checker_files") == state.get("checker_files")
        for e in failure.get("evidence", [])
    ):
        return None
    from infinicoder.verification.check_identity import checker_files

    if checker_files(spec, workspace) != state.get("checker_files"):
        return None
    view = runtime._worker_scope["source"] if runtime._worker_scope else module.decode("utf8")
    report = analyze(spec.task, view, path)
    if not report:
        return None
    row = report["results"][0]
    fits = row["argument_order_hypotheses"]
    if (
        len(fits) != 1
        or row["hypotheses_omitted"]
        or len({tuple(e["arguments"]) for e in row["examples"]}) < 2
    ):
        return None
    permutation = fits[0]
    formal, expr = functions(view)[row["function"]]
    mapping = dict(zip(formal, [formal[i] for i in permutation]))

    class Rename(ast.NodeTransformer):
        def visit_Name(self, node):
            return ast.copy_location(
                ast.Name(id=mapping.get(node.id, node.id), ctx=copy.deepcopy(node.ctx)), node
            )

    changed = ast.unparse(ast.fix_missing_locations(Rename().visit(copy.deepcopy(expr))))
    lines = view.encode("utf8").splitlines(keepends=True)
    start = sum(len(x) for x in lines[: expr.lineno - 1]) + expr.col_offset
    end = sum(len(x) for x in lines[: expr.end_lineno - 1]) + expr.end_col_offset
    candidate = (view.encode()[:start] + changed.encode() + view.encode()[end:]).decode()
    if (
        candidate == view
        or row["function"] not in functions(candidate)
        or analyze(spec.task, candidate, path) is not None
    ):
        return None
    key = hashlib.sha256(
        json.dumps(
            [path, digest, signature(spec), permutation, identity()], sort_keys=True
        ).encode()
    ).hexdigest()
    if key in state.get("binding_candidate_attempted", []):
        return None
    receipt = dict(
        id=key,
        path=path,
        module_source_sha256=digest,
        source_view_sha256=hashlib.sha256(view.encode()).hexdigest(),
        requirements=signature(spec),
        other_sources=workspace.revision([path]),
        function=row["function"],
        argument_order=permutation,
        old_expression=ast.unparse(expr),
        new_expression=changed,
        diagnosis=report,
        provider_identity=identity(),
        model_calls=0,
        hypothesis_not_proof=True,
    )
    return Proposal(candidate, module.decode("utf8"), copy.deepcopy(failure), receipt)


def verify(runtime, spec, state, workspace, proposal, result, remaining, require_artifacts=True):
    """Same original verifier;restore failed proposals without clobbering edits."""
    began = time.monotonic()
    receipt = proposal.receipt
    path = receipt["path"]
    accepted = False
    evidence = []
    error = None
    rollback = None
    candidate_hash = result["after"]

    def valid():
        return (
            signature(spec) == receipt["requirements"]
            and workspace.revision([path]) == receipt["other_sources"]
            and hashlib.sha256(workspace.path(path).read_bytes()).hexdigest() == candidate_hash
        )

    try:
        if not valid() or remaining() <= 0:
            raise ValueError("Candidate source,requirements or budget changed before acceptance")
        ok, evidence = runtime._verify(
            spec,
            workspace,
            [c.id for c in spec.checks],
            remaining,
            require_artifacts=require_artifacts,
        )
        control = runtime.store.take_control()
        if control:
            state.update(
                status="paused" if control == "pause" else "cancelled", reason="user_" + control
            )
            runtime.store.event(
                "control", dict(action=control, boundary="after_binding_candidate_check")
            )
        accepted = bool(ok and valid() and remaining() > 0 and not control)
        return accepted, evidence
    except BaseException as exc:
        error = type(exc).__name__ + ": " + str(exc)[:500]
        raise
    finally:
        if not accepted:
            current = workspace.path(path).read_bytes()
            if hashlib.sha256(current).hexdigest() == candidate_hash:
                rollback = workspace.edit(
                    path=path,
                    expected_hash=candidate_hash,
                    old=current.decode(),
                    new=proposal.module_before,
                )
                result.update(
                    rejected_source=True,
                    retained_sha256=rollback["after"],
                    binding_baseline_restored=True,
                )
                state["last_failure"] = (
                    copy.deepcopy(proposal.prior_failure)
                    if signature(spec) == receipt["requirements"]
                    else None
                )
            else:
                rollback = dict(skipped=True, reason="Concurrent source change;never overwrite it")
                state.update(status="awaiting_input", reason="binding_candidate_source_changed")
        result["binding_candidate_id"] = receipt["id"]
        outcome = dict(
            id=receipt["id"],
            path=path,
            source_before=receipt["module_source_sha256"],
            candidate_sha256=candidate_hash,
            requirements=receipt["requirements"],
            accepted=accepted,
            evidence=evidence,
            error=error,
            rollback=rollback,
            wall_s=time.monotonic() - began,
            model_calls=0,
        )
        state.setdefault("binding_candidate_trials", []).append(outcome)
        runtime.store.event("binding_candidate_trial", outcome)
        runtime.store.save(state)
