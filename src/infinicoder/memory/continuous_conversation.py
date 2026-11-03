"""Opt-in projection of the canonical Runtime action history, not another agent.

No generated summaries, hidden requests, execution authority, or acceptance
certificates. Historical observations keep their actual revision identities.
"""

from infinicoder._resources import package_root

from dataclasses import asdict
import hashlib
import json

from infinicoder.models.budget import BudgetExceeded
from infinicoder.runtime.feedback import acceptance_packet
from infinicoder.memory.revision_feedback import revision
from infinicoder.execution.workspace_tools import Workspace


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def identity():
    root = package_root()
    return dict(
        version="continuous_v1",
        sha256=hashlib.sha256(
            b"".join(
                (root / name).read_bytes()
                for name in (
                    "memory/continuous_conversation.py",
                    "runtime/shell_loop.py",
                    "runtime/text_frontend.py",
                    "runtime/engine.py",
                )
            )
        ).hexdigest(),
    )


def validate(runtime):
    """Reject competing automatic routes before any request, never downgrade."""
    if not runtime.shell_loop or runtime.decision_driver != "text_task_first":
        raise ValueError("continuous_v1 requires shell_loop and text_task_first")
    # Runtime internally enables command_worker for every shell_loop. This is
    # the ordinary original executor, not a competing source-worker route.
    forbidden = (
        "automatic_workers",
        "worker_path",
        "worker_symbol",
        "source_focus",
        "inspection_recovery",
        "native_plan",
        "problem_workflow",
        "stalled_feedback_policy",
        "flow_update_policy",
        "design_contracts",
        "design_probes",
        "design_investigation",
        "grounded_planning",
        "diagnostic_planning",
        "development_workflows",
        "repair_memory",
        "source_sync",
        "source_archive",
        "command_evidence",
        "operator_protocol",
        "tensor_dataflow",
        "profiling",
        "optimization",
    )
    active = [name for name in forbidden if getattr(runtime, name, False)]
    if active:
        raise ValueError(
            "continuous_v1 does not support automatic context/routes: " + ",".join(active)
        )


def preflight(runtime, state):
    if state.get("planning_required"):
        raise ValueError(
            "continuous_v1 requires an explicit writable contract before the first request; use the original route for mandatory planning"
        )
    if any(unit.kind != "implement" or unit.worker_path for unit in runtime._graph.units):
        raise ValueError(
            "continuous_v1 supports ordinary implementation phases only; use the original route for mandatory stages/workers"
        )


def implementation_targets(workspace, paths, protected):
    """Only explicit exact caller paths identify implementation targets.

    A glob/directory permission is not an inferred candidate declaration. Do
    not guess from task IDs, extensions, test filenames, or arbitrary writes.
    """
    paths = sorted(set(paths))
    if not paths or any(any(char in path for char in "*?[") for path in paths):
        return dict(status="unknown", reason="no_explicit_exact_target_files")
    hashes = {}
    try:
        for name in paths:
            path = workspace.path(name, write=True)
            if path.is_dir() or str(path.resolve()) in protected:
                return dict(status="unknown", reason="directory_or_protected_target")
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    except (OSError, ValueError):
        return dict(status="unknown", reason="target_identity_unavailable")
    return dict(status="known", scope_identity=digest(paths), hashes=hashes)


def snapshot(runtime, spec, state):
    ordinal = min(state["milestone"], len(runtime._graph.order) - 1)
    unit = runtime._graph.unit(runtime._graph.order[ordinal])
    # Full caller data and work-unit interfaces stay mandatory; teaching cards,
    # automatically retrieved source and duplicate state logs are not a contract.
    work_unit = asdict(unit)
    if unit.goal == spec.task or unit.goal.startswith(spec.task + "\n"):
        work_unit.pop("goal")
        work_unit.update(task_ref="task", goal_suffix=unit.goal[len(spec.task) :])
    contract = dict(
        task=spec.task,
        allowed_paths=spec.allowed_paths,
        required_artifacts=spec.required_artifacts,
        target_facts=spec.target_facts,
        tensor_layouts=spec.tensor_layouts,
        acceptance=acceptance_packet(spec.checks),
        capabilities=spec.capabilities,
        amendments=spec.amendments,
        clarifications=state.get("clarifications", []),
        current_work_unit=work_unit,
    )
    if getattr(runtime, "reference_contract", None) is not None:
        from infinicoder.verification.reference_compare import view as reference_view

        contract["public_reference_experiment"] = reference_view(
            runtime.reference_contract, state["milestone"]
        )
    contract = json.loads(encoded(contract))
    workspace = Workspace(runtime.workspace, spec.allowed_paths)
    return dict(
        phase=state["milestone"],
        requirements=digest(contract),
        source_revision=revision(spec, workspace),
        contract=contract,
        implementation_targets=implementation_targets(
            workspace, unit.write_set or spec.allowed_paths, state.get("checker_files", {})
        ),
    )


def stamp(runtime, spec, state):
    """Attach provenance to the original record after original guarded dispatch."""
    records = state.get("shell_loop_history", [])
    if not records:
        return
    session = state.get("continuous_conversation", {})
    before = session.get("current_snapshot")
    after = snapshot(runtime, spec, state)
    old_targets = (before or {}).get("implementation_targets", {})
    new_targets = after["implementation_targets"]
    comparable = (
        old_targets.get("status") == new_targets.get("status") == "known"
        and old_targets["scope_identity"] == new_targets["scope_identity"]
        and before["requirements"] == after["requirements"]
        and before["phase"] == after["phase"]
    )
    changed = (
        [
            name
            for name in new_targets["hashes"]
            if old_targets["hashes"][name] != new_targets["hashes"][name]
        ]
        if comparable
        else None
    )
    records[-1]["conversation_provenance"] = dict(
        before={k: v for k, v in (before or {}).items() if k != "contract"},
        after={k: v for k, v in after.items() if k != "contract"},
        source_changed=bool(before and before["source_revision"] != after["source_revision"]),
        implementation_changed=bool(changed) if comparable else None,
        implementation_changed_paths=changed,
        acceptance_proof=False,
    )


def _ranges(calls):
    ranges = []
    for call in calls:
        if ranges and ranges[-1][1] + 1 == call:
            ranges[-1][1] = call
        else:
            ranges.append([call, call])
    return ranges


def context(runtime, spec, state, index):
    from infinicoder.runtime.shell_loop import _cost, _bounded_pair, tools

    current = snapshot(runtime, spec, state)
    session = state.setdefault("continuous_conversation", {})
    if not session:
        contract = dict(
            current["contract"],
            remaining_model_calls=runtime.max_model_calls - state.get("model_calls", 0),
        )
        text = (
            encoded(contract)
            + "\n"
            + runtime.registry.context(spec.capabilities, spec.target_facts)
        )
        guidance = runtime._inspection_guidance()
        if runtime.auto_verify_commands:
            from infinicoder.memory.revision_feedback import GUIDANCE

            guidance += "\n" + GUIDANCE
        initial = [
            dict(
                role="system",
                content=runtime._decision_frontend.system(
                    tools(runtime), extended=bool(runtime.task_memory or runtime.symbol_tools)
                )
                + guidance,
            ),
            dict(role="user", content=runtime._decision_frontend.instance(text)),
        ]
        session.update(
            policy="continuous_v1",
            initial_messages=initial,
            initial_contract_hash=current["requirements"],
            initial_messages_hash=digest(initial),
            initial_snapshot=current,
            current_snapshot=current,
            boundaries=[],
        )
    if (
        session["policy"] != "continuous_v1"
        or digest(session["initial_messages"]) != session["initial_messages_hash"]
    ):
        raise ValueError("Continuous conversation initial contract/policy integrity mismatch")
    prior = session["current_snapshot"]
    if any(prior[key] != current[key] for key in ("requirements", "phase", "source_revision")):
        boundary = dict(
            after_call=state.get("model_calls", 0),
            origin="runtime_revision_or_contract_boundary",
            previous={k: v for k, v in prior.items() if k != "contract"},
            current={k: v for k, v in current.items() if k != "contract"},
            historical_observations_remain_historical=True,
            acceptance_proof=False,
        )
        if prior["requirements"] != current["requirements"]:
            boundary["current_contract"] = current["contract"]
        if not session["boundaries"] or session["boundaries"][-1] != boundary:
            session["boundaries"].append(boundary)
    session["current_snapshot"] = current
    initial = [dict(message) for message in session["initial_messages"]]
    limit = runtime.assembler.profile.input_budget()
    records = state.get("shell_loop_history", [])
    calls = [record["call"] for record in records]

    # Source edits and their most recent acceptance observation cannot be
    # displaced by a suffix of read-only commands. They remain historical if
    # the current revision/requirements differ, never reusable acceptance.
    def current_scope(record):
        after = record.get("conversation_provenance", {}).get("after", {})
        targets = after.get("implementation_targets", {})
        return (
            after.get("phase") == current["phase"]
            and after.get("requirements") == current["requirements"]
            and targets.get("status") == current["implementation_targets"].get("status") == "known"
            and targets.get("scope_identity")
            == current["implementation_targets"].get("scope_identity")
        )

    candidate = next(
        (
            record
            for record in reversed(records)
            if record.get("conversation_provenance", {}).get("implementation_changed") is True
            and current_scope(record)
        ),
        None,
    )
    required = {candidate["call"]} if candidate else set()
    from infinicoder.runtime.feedback import command_has_check

    check = next(
        (
            record
            for record in reversed(records)
            if (not candidate or record["call"] >= candidate["call"]) and command_has_check(record)
        ),
        None,
    )
    if check:
        required.add(check["call"])
    if getattr(runtime, "reference_contract", None) is not None:
        reference = next(
            (
                r
                for r in reversed(records)
                if (r.get("action") or {}).get("tool") == "reference_compare" and current_scope(r)
            ),
            None,
        )
        if reference:
            required.add(reference["call"])
    if records:
        required.add(records[-1]["call"])
    selected = {}
    bounded = []
    versions = {}

    def version_ref(value):
        version = {key: value[key] for key in ("phase", "requirements", "source_revision")}
        key = digest(version)
        if key not in versions:
            versions[key] = (str(len(versions)), version)
        return versions[key][0]

    version_ref(session["initial_snapshot"])
    current_ref = version_ref(current)
    for record in records:
        provenance = record.get("conversation_provenance")
        if provenance:
            version_ref(provenance["before"])
            version_ref(provenance["after"])

    def compose(chosen):
        omitted = [call for call in calls if call not in chosen]
        needed_refs = {current_ref}
        for call in chosen:
            provenance = by_call[call].get("conversation_provenance")
            if provenance:
                needed_refs.update(
                    (version_ref(provenance["before"]), version_ref(provenance["after"]))
                )
        tail = dict(
            origin="runtime_conversation_projection",
            policy="continuous_v1",
            current_version_ref=current_ref,
            recorded_versions={
                label: value for label, value in versions.values() if label in needed_refs
            },
            remaining_model_calls=max(0, runtime.max_model_calls - state.get("model_calls", 0)),
            omitted_call_ranges=_ranges(omitted),
            bounded_calls=bounded,
            historical_observations_are_not_current_facts=True,
            acceptance_proof=False,
        )
        if getattr(runtime, "implementation_provider", None) is not None and state.get(
            "source_unit_last_check"
        ):
            from infinicoder.memory.source.source_units import check_view

            tail["latest_original_check"] = check_view(
                spec, state, Workspace(runtime.workspace, spec.allowed_paths)
            )
        tail["implementation_target_status"] = current["implementation_targets"]["status"]
        tail["target_candidate_call"] = candidate["call"] if candidate else None
        tail["latest_check_call"] = check["call"] if check else None
        # Current permissions/requirements must survive even if their boundary
        # predates the retained suffix. No model inference supplies this data.
        if current["requirements"] != session["initial_contract_hash"]:
            tail["current_contract"] = current["contract"]
        messages = list(initial)
        # Each turn carries its exact original versions. Only the latest
        # boundary needs a separate body; replaying every transition would
        # recreate the repeated state injection this policy is testing.
        boundaries = session["boundaries"][-1:]
        done = set()
        for call in sorted(chosen):
            for ordinal, event in enumerate(boundaries):
                if event["after_call"] < call and ordinal not in done:
                    messages.append(
                        dict(role="user", content="HISTORICAL RUNTIME BOUNDARY\n" + encoded(event))
                    )
                    done.add(ordinal)
            messages.extend(chosen[call])
        # Older boundary bodies can themselves grow. Retain the latest actual
        # boundary; omitted bodies remain durable with no invented summary.
        if boundaries and len(boundaries) - 1 not in done:
            messages.append(
                dict(role="user", content="RUNTIME BOUNDARY\n" + encoded(boundaries[-1]))
            )
        messages.append(dict(role="user", content="CURRENT RUNTIME ENVELOPE\n" + encoded(tail)))
        return messages

    # Store boundary identities on each record; all old boundary bodies need
    # not be repeated alongside old observations at every request.
    def pair(record):
        from infinicoder.runtime.shell_loop import render_pair

        result = render_pair(runtime, record)
        provenance = record.get("conversation_provenance")
        if provenance:
            result = [dict(message) for message in result]
            references = dict(
                before_version=version_ref(provenance["before"]),
                after_version=version_ref(provenance["after"]),
                source_changed=provenance["source_changed"],
                implementation_changed=provenance.get("implementation_changed"),
                implementation_changed_paths=provenance.get("implementation_changed_paths"),
                acceptance_proof=False,
            )
            result[-1]["content"] = (
                "HISTORICAL TURN PROVENANCE\n" + encoded(references) + "\n" + result[-1]["content"]
            )
        return result

    by_call = {record["call"]: record for record in records}
    if _cost(runtime, compose({})) > limit:
        raise BudgetExceeded(
            "Continuous conversation caller contract/current boundary exceeds input budget;zero request permitted"
        )
    for call in sorted(required):
        selected[call] = pair(by_call[call])
        if _cost(runtime, compose(selected)) > limit:
            selected.pop(call)
            # Existing bounded observation is explicitly partial user evidence;
            # it is never a truncated assistant action or a fake executed turn.
            budget = max(0, limit - _cost(runtime, compose(selected)) - 64)
            partial = _bounded_pair(runtime, by_call[call], budget)
            if partial is None:
                raise BudgetExceeded(
                    "Latest necessary continuous conversation evidence cannot fit;zero request permitted"
                )
            bounded.append(call)
            selected[call] = partial
            if _cost(runtime, compose(selected)) > limit:
                raise BudgetExceeded(
                    "Bounded continuous conversation evidence cannot fit;zero request permitted"
                )
    for record in reversed(records):
        call = record["call"]
        if call in selected:
            continue
        selected[call] = pair(record)
        if _cost(runtime, compose(selected)) > limit:
            selected.pop(call)
            break  # Retain a consecutive suffix plus explicitly pinned turns.
    messages = compose(selected)
    report = dict(
        shell_loop=True,
        conversation_policy="continuous_v1",
        initial_contract_hash=session["initial_contract_hash"],
        initial_messages_hash=session["initial_messages_hash"],
        selected_calls=sorted(selected),
        omitted_call_ranges=_ranges([call for call in calls if call not in selected]),
        target_candidate_call=candidate["call"] if candidate else None,
        latest_check_call=check["call"] if check else None,
        implementation_target_status=current["implementation_targets"]["status"],
        bounded_calls=bounded,
        total_history=len(records),
        input_tokens=_cost(runtime, messages),
        input_budget=limit,
        system_tokens=_cost(runtime, initial[:1]),
        initial_task_tokens=_cost(runtime, initial[1:]),
        history_tokens=sum(_cost(runtime, item) for item in selected.values()),
        optional_injection_tokens=0,
        memory_reserved_tokens=0,
        memory_included=False,
        estimator="caller tokenizer" if runtime.assembler.estimator.tokenizer else "heuristic",
    )
    report["boundary_and_envelope_tokens"] = (
        report["input_tokens"]
        - report["system_tokens"]
        - report["initial_task_tokens"]
        - report["history_tokens"]
    )
    state["budget_report"] = report
    session["projection"] = report
    runtime.store.event("continuous_conversation_projection", report)
    return messages
