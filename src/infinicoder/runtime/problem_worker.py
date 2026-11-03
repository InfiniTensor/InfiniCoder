"""One evidence-triggered scoped proposal using the existing source worker.

No global Runtime configuration is changed. A dispatch survives interruption
but is never replayed. Source leases certify provenance,not semantic reasons.
"""

import hashlib, re
from copy import deepcopy
from infinicoder.runtime.problem_workflow import contract, digest, event


def finish(runtime, state, workspace, outcome, result=None):
    record = state.get("problem_worker")
    if not record or record["status"] in ("returned", "declined", "interrupted"):
        return
    result = result or {}
    record.update(status="returned", outcome=outcome, return_call=state.get("model_calls", 0))
    if record.get("applied"):
        try:
            retained = hashlib.sha256(
                workspace.path(record["target"]["path"]).read_bytes()
            ).hexdigest()
        except (OSError, ValueError):
            retained = None
        record.update(retained_sha256=retained, retained=retained == record["proposed_sha256"])
        if not record["retained"]:
            record["outcome"] = "candidate_not_retained"
    elif record.get("proposal_unchanged"):
        record["outcome"] = "no_source_change"
    elif result.get("error"):
        record["outcome"] = "invalid_or_unavailable_proposal"
    record["original_accepted"] = result.get("accepted")
    state["problem_decision"]["status"] = "closed"
    event(
        runtime,
        state,
        "scoped_proposal_returned",
        record=deepcopy(record),
        acceptance_is_original_verifier=True,
    )


def lease(state, spec, workspace, record):
    from infinicoder.planning.source_plan import stale_sources

    if record["contract_ref"] != contract(state, spec) or record["phase"] != state.get(
        "milestone", 0
    ):
        raise ValueError("Scoped proposal contract/phase changed")
    if stale_sources(
        dict(
            units=[
                dict(sources=[dict(path=p, sha256=s) for p, s in record["source_versions"].items()])
            ]
        ),
        workspace,
    ):
        raise ValueError("Scoped proposal source dependency changed")


def select(runtime, state, spec, workspace, index):
    """Unique caller-named current symbol with actual whole-target read coverage."""
    existing = state.get("problem_worker")
    if existing:
        if existing["status"] == "dispatched":
            existing.update(status="interrupted", outcome="dispatched_request_not_replayed")
            state["problem_decision"]["status"] = "closed"
            event(runtime, state, "scoped_proposal_interrupted", record=deepcopy(existing))
            return None
        if existing["status"] != "pending":
            return None
        try:
            lease(state, spec, workspace, existing)
        except ValueError as error:
            decline(runtime, state, str(error))
            return None
        return existing
    if state.get("problem_decision", {}).get("status") != "due":
        return None
    from infinicoder.memory.inspection_ledger import InspectionLedger
    from infinicoder.memory.source.source_scope import choose, packet, declarations

    unit = runtime._graph.unit(runtime._graph.order[state["milestone"]])
    matches = []
    current = InspectionLedger(index).current()
    for source in current:
        path = source["path"]
        if not path.endswith(".py"):
            continue
        try:
            workspace.path(path, write=True)
            if unit.write_set:
                from infinicoder.execution.workspace_tools import Workspace

                Workspace(workspace.root, unit.write_set).path(path, write=True)
            text = workspace.path(path).read_text(encoding="utf-8-sig")
            symbol = choose(text, spec.task)
            if symbol is None:
                # Public module functions may be explicitly named as file.func.
                names = [
                    name
                    for name in declarations(text)
                    if re.search(
                        r"(?<![\w.])"
                        + re.escape(workspace.path(path).stem + "." + name)
                        + r"(?!\w)",
                        spec.task,
                    )
                ]
                symbol = names[0] if len(names) == 1 else None
            if symbol is None:
                continue
            scope = packet(text, symbol)
            if not any(a <= scope["start"] and b >= scope["end"] for a, b in source["read_spans"]):
                continue
            matches.append(
                dict(
                    path=path,
                    symbol=symbol,
                    sha256=source["source_sha256"],
                    start=scope["start"],
                    end=scope["end"],
                )
            )
        except (OSError, ValueError, UnicodeError, SyntaxError):
            continue
    if len(matches) != 1:
        state["problem_worker"] = dict(
            version=1,
            status="declined",
            outcome="unique_observed_caller_symbol_required",
            matches=len(matches),
        )
        state["problem_decision"]["status"] = "closed"
        event(runtime, state, "scoped_proposal_declined", record=state["problem_worker"])
        return None
    if len(current) > 64:
        state["problem_worker"] = dict(
            version=1, status="declined", outcome="bounded_dependency_lease_capacity"
        )
        state["problem_decision"]["status"] = "closed"
        return None
    target = matches[0]
    versions = {r["path"]: r["source_sha256"] for r in current}
    record = dict(
        version=1,
        status="pending",
        target=target,
        source_versions=versions,
        contract_ref=contract(state, spec),
        phase=state["milestone"],
        semantic_reason=None,
        scope_is_controller_hypothesis=True,
        acceptance_proof=False,
        applied=False,
    )
    record["id"] = digest(record)
    state["problem_worker"] = record
    runtime.store.save(state)
    event(runtime, state, "scoped_proposal_selected", record=deepcopy(record))
    return record


def decline(runtime, state, reason):
    record = state.get("problem_worker")
    if record and record["status"] not in ("returned", "declined", "interrupted"):
        record.update(status="declined", outcome=reason)
        state["problem_decision"]["status"] = "closed"
        event(runtime, state, "scoped_proposal_declined", record=deepcopy(record))


def dispatch(runtime, state, spec, workspace):
    record = state["problem_worker"]
    lease(state, spec, workspace, record)
    if record["status"] != "pending":
        raise ValueError("Scoped proposal is not pending")
    record.update(status="dispatched", dispatch_call=state["model_calls"])
    runtime.store.save(state)
    event(runtime, state, "scoped_proposal_dispatched", record=deepcopy(record))


def applied(runtime, state, workspace, result):
    record = state.get("problem_worker")
    if (
        not record
        or record["status"] != "dispatched"
        or result.get("path") != record["target"]["path"]
    ):
        return
    if result.get("before") == result.get("after") or not result.get("after"):
        return
    record.update(
        applied=True,
        proposed_sha256=result["after"],
        proposed_revision=workspace.revision(),
        candidate_id=result.get("problem_candidate_id"),
    )
    event(runtime, state, "scoped_proposal_applied", record=deepcopy(record))


def action_request(runtime, spec, state, text, schema, finish_reason):
    """Only a complete original native action is a fallback request,not code."""
    from infinicoder.runtime.protocol import final_payload

    payload = final_payload(text)
    if runtime.native_tools and payload.lstrip().startswith("<function"):
        from infinicoder.tools.native_tools import decode

        return decode(
            text,
            schema,
            finish_reason=finish_reason,
            allow_rationale=True,
            allow_envelope_closure=True,
            allow_read_batch=runtime.native_read_batch,
        )
    return None
