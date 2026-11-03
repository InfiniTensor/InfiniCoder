"""One task question,versioned source views and a bounded discriminating probe.

Syntax,caller-executed observations and model hypotheses remain distinct.
Probe comparisons never resolve a task or replace caller-owned acceptance.
"""

from infinicoder._resources import package_root

from copy import deepcopy
import hashlib, json
from pathlib import Path
from infinicoder.memory.repair_memory import signature

MAX_QUESTIONS = 8
MAX_EXPERIMENTS = 2


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def contract(state, spec):
    return digest([signature(spec), state.get("clarifications", [])])


def identity():
    root = package_root()
    return hashlib.sha256(
        b"".join(
            (root / name).read_bytes()
            for name in (
                "runtime/problem_workflow.py",
                "runtime/problem_decision.py",
                "runtime/problem_worker.py",
                "analysis/compact_question_flow.py",
                "memory/question_memory.py",
                "memory/flow_memory.py",
                "planning/diagnostic_flow.py",
                "runtime/shell_loop.py",
                "runtime/engine.py",
            )
        )
    ).hexdigest()


def event(runtime, state, kind, **data):
    state["problem_sequence"] = state.get("problem_sequence", 0) + 1
    row = dict(
        kind=kind,
        call=state.get("model_calls", 0),
        step=state.get("steps", 0),
        sequence=state["problem_sequence"],
        acceptance_proof=False,
        **data,
    )
    state.setdefault("problem_events", []).append(row)
    del state["problem_events"][:-32]
    runtime.store.event("problem_artifact_" + kind, row)
    return row


def recent_records(questions, field, current_id, limit=2):
    rows = []
    for question in questions.values():
        values = (
            question.get("candidates", [])
            if field == "candidates"
            else [question["observation"]]
            if question.get("observation")
            else []
        )
        rows.extend((question, value) for value in values)
    rows.sort(
        key=lambda row: (row[1].get("sequence", 0), row[1].get("call", 0), row[1].get("step", 0))
    )
    current = [row for row in rows if row[0]["id"] == current_id]
    chosen = current[-1:] if current else []
    for row in reversed(rows):
        if len(chosen) >= limit:
            break
        if not any(row[1] is selected[1] for selected in chosen):
            chosen.append(row)
        if len(chosen) >= limit:
            break
    chosen.sort(
        key=lambda row: (row[1].get("sequence", 0), row[1].get("call", 0), row[1].get("step", 0))
    )
    return chosen, max(0, len(rows) - len(chosen))


def refresh(state, spec, workspace, runtime=None):
    """Invalidate source-derived artifacts;never refresh old runtime evidence."""
    requirements = contract(state, spec)
    cache = {}
    revision = None
    for question in state.get("problem_questions", {}).values():
        stale = question["requirements"] != requirements
        for path, sha in question.get("sources", {}).items():
            try:
                if path not in cache:
                    cache[path] = hashlib.sha256(workspace.path(path).read_bytes()).hexdigest()
                stale = stale or cache[path] != sha
            except (OSError, ValueError):
                stale = True
        observation = question.get("observation")
        if observation:
            if revision is None:
                revision = workspace.revision()
            stale = stale or observation["revision"] != revision
        for candidate in question.get("candidates", []):
            if revision is None:
                revision = workspace.revision()
            candidate["current"] = (
                candidate["revision"] == revision and candidate["requirements"] == requirements
            )
        if stale and question["status"] != "stale":
            question["status"] = "stale"
            question["invalidation_unreported"] = True
            if observation:
                observation["current"] = False
        if runtime and question.pop("invalidation_unreported", False):
            event(
                runtime,
                state,
                "invalidated",
                question_id=question["id"],
                artifact_id=question.get("artifact_id"),
                reason="source_or_public_contract_changed",
            )


def pack(index, spec, state, unit, workspace, max_tokens, estimator, exclude_paths=(), mounts=()):
    from infinicoder.memory.question_memory import select
    from infinicoder.memory.task_memory import TaskMemory
    from infinicoder.memory.source.working_set import pack as source_pack
    from infinicoder.memory.flow_memory import slice_function, bounded_slice

    refresh(state, spec, workspace)
    choice = select(index, spec, state, unit, workspace, mounts)
    question_id = digest([contract(state, spec), state.get("milestone", 0), choice["question"]])
    questions = state.setdefault("problem_questions", {})
    if question_id not in questions:
        if len(questions) >= MAX_QUESTIONS:
            return "", dict(included=False, reason="bounded_question_registry_full", tokens=0)
        questions[question_id] = dict(
            id=question_id,
            question=choice["question"],
            requirements=contract(state, spec),
            status="unresolved",
            sources={},
            exposed=False,
            phase=state.get("milestone", 0),
        )
    question = questions[question_id]
    # A changed question keeps its ID/history,while a new source view gets a new artifact.
    prior_status = question["status"]
    packet = dict(
        question=dict(
            id=question_id,
            text=choice["question"],
            origin="public_task_and_current_source_routing",
            selection_is_hypothesis=True,
            status="unresolved" if prior_status == "stale" else prior_status,
        ),
        sources=[],
        relations=[],
        source_facts=[],
        runtime_observations=[],
        model_hypotheses=[],
        candidate_states=[],
        missing=[],
        work_items=[
            dict(
                phase=state.get("milestone", 0),
                status="unverified",
                write_scope=list(unit.write_set),
                scope_is_permission_not_required_modification=True,
                check_ids=list(unit.check_ids),
            )
        ],
        rule="Read source and use original edit/verify/finish. A sufficient contract permits direct edits without an experiment. Optional edit.problem_context cites question_id,artifact_id,evidence_ids. If evidence is missing,use one problem_probe to distinguish hypotheses;comparisons are not causes or acceptance. Historical observations are not current facts.",
    )
    if max_tokens < 1600:
        packet["relation_source_refs"] = (
            "Zero-based indices into sources;path and version come from that source row."
        )
    render = lambda: encoded(dict(problem_work_packet=dict(packet, artifact_id=digest(packet))))
    required = required_context(state)
    observations, omitted_observations = recent_records(questions, "observation", question_id)
    candidates, omitted_candidates = recent_records(questions, "candidates", question_id)
    if omitted_candidates or omitted_observations:
        packet["omitted_lifecycle"] = dict(
            candidates=omitted_candidates,
            observations=omitted_observations,
            selection="latest current-question and newest event sequence;remaining history retained in session",
        )
    if estimator.estimate(render()) > max_tokens:
        return "", dict(
            included=False,
            reason="problem_header_exceeds_budget",
            tokens=0,
            required_evidence=required,
        )

    def offer(key, value):
        packet[key].append(value)
        if estimator.estimate(render()) > max_tokens:
            packet[key].pop()
            return False
        return True

    # Reserve results/candidate state BEFORE optional source and syntax views.
    for owner, observation in observations:
        value = {
            k: deepcopy(observation[k])
            for k in (
                "evidence_id",
                "revision",
                "outcome",
                "comparisons",
                "output_sha256",
                "current",
                "output_excerpt",
            )
            if k in observation
        }
        value["question_id"] = owner["id"]
        value["historical_not_current_fact"] = not observation["current"]
        if not offer("runtime_observations", value):
            return "", dict(
                included=False,
                reason="required_problem_evidence_exceeds_budget",
                tokens=0,
                required_evidence=True,
            )
        for hypothesis in observation["hypotheses"]:
            if not offer(
                "model_hypotheses",
                dict(
                    kind="model_hypothesis_not_fact",
                    evidence_id=observation["evidence_id"],
                    **hypothesis,
                ),
            ):
                return "", dict(
                    included=False,
                    reason="required_problem_evidence_exceeds_budget",
                    tokens=0,
                    required_evidence=True,
                )
    for owner, candidate in candidates:
        value = {
            k: deepcopy(candidate[k])
            for k in (
                "id",
                "question_id",
                "artifact_id",
                "evidence_ids",
                "explicit_reference",
                "path",
                "before",
                "after",
                "revision",
                "phase",
                "status",
                "current",
                "checked_ids",
            )
        }
        if not offer("candidate_states", value):
            return "", dict(
                included=False,
                reason="required_candidate_state_exceeds_budget",
                tokens=0,
                required_evidence=True,
            )
    focus = list(dict.fromkeys([row["path"] for row in choice["anchors"]] + list(unit.write_set)))
    source, base = source_pack(
        TaskMemory(index),
        choice,
        choice["question"],
        focus,
        max(0, max_tokens * 3 // 5),
        estimator,
        exclude_paths,
    )
    rows = [json.loads(line) for line in source.splitlines()]
    selected = []
    compact = max_tokens < 1600
    if compact:
        from infinicoder.analysis.compact_question_flow import first_pair

        candidates = [row for row in rows if "excerpt" in row]
        primary = first_pair(candidates, choice, index, packet, offer)
        if primary is not None:
            selected.append(primary)
    for row in rows:
        if "excerpt" in row or "current_unparsed_source" in row:
            if any(
                old.get("symbol") and old.get("symbol") == row.get("symbol") for old in selected
            ):
                continue
            if offer("sources", row):
                selected.append(row)
    seen = set()
    for row in selected:
        symbol = row.get("symbol")
        if not symbol or symbol in seen or len(seen) >= 2:
            continue
        seen.add(symbol)
        if compact and packet["relations"]:
            continue
        try:
            flow = slice_function(index, symbol, view=choice["view"], max_lines=8)
            if flow.get("heap_updates"):
                # Even larger packets have a bounded relation allowance. Keep
                # a complete source-linked update instead of dropping it while
                # retaining redundant full-slice metadata and scalar edges.
                from infinicoder.analysis.compact_question_flow import relation as compact_relation

                for groups, calls in ((3, 1), (2, 1), (1, 0)):
                    small = compact_relation(flow, packet["sources"].index(row), groups, calls)
                    if len(encoded(small)) <= 1800 and offer("relations", small):
                        break
                else:
                    offer(
                        "missing",
                        dict(
                            symbol=symbol,
                            reason="update_relation_exceeds_budget",
                            expand="function_summary",
                        ),
                    )
            else:
                relation = bounded_slice(flow, max_chars=1800)
                offer("relations", dict(kind="source_syntax_candidates", **relation))
        except (ValueError, KeyError, SyntaxError):
            offer("missing", dict(symbol=symbol, reason="static_relation_unavailable"))
    for row in rows:
        if "function_summary" in row or "use_def_of" in row:
            offer("source_facts", dict(kind="source_syntax_candidates", **row))
    if base.get("omitted") or not selected:
        offer(
            "missing",
            dict(
                reason="source_or_relations_omitted",
                expand="read/function_summary/use_def",
                omitted_is_not_irrelevant=True,
            ),
        )
    if not selected and not required:
        return "", dict(included=False, reason="no_current_source_fits", tokens=0)
    from infinicoder.analysis.compact_question_flow import dependency_versions

    try:
        versions = dependency_versions(packet, workspace)
    except (OSError, ValueError):
        return "", dict(included=False, reason="source_dependency_changed", tokens=0)
    artifact_id = digest(packet)
    question.update(sources=versions, artifact_id=artifact_id, status=packet["question"]["status"])
    return render(), dict(
        included=True,
        encoding="problem_workflow_v1",
        question_id=question_id,
        artifact_id=artifact_id,
        tokens=estimator.estimate(render()),
        max_tokens=max_tokens,
        source_records=len(selected),
        relation_records=len(packet["relations"]),
        source_versions=versions,
        prior_status=prior_status,
        experiment_available=not question.get("attempted"),
        compact_source_relation_pair=compact and bool(packet["relations"]),
        required_evidence=required,
    )


def publish(runtime, state, report, included):
    for question in state.get("problem_questions", {}).values():
        if question.pop("invalidation_unreported", False):
            event(
                runtime,
                state,
                "invalidated",
                question_id=question["id"],
                artifact_id=question.get("artifact_id"),
                reason="source_or_public_contract_changed",
            )
    if not included or not report.get("question_id"):
        return
    question = state["problem_questions"][report["question_id"]]
    question["exposed"] = True
    if question.get("last_exposed_artifact") != question["artifact_id"]:
        question["last_exposed_artifact"] = question["artifact_id"]
        event(
            runtime,
            state,
            "produced",
            question_id=question["id"],
            artifact_id=question["artifact_id"],
            sources=question["sources"],
            exposure_is_consumption=False,
        )


def schema():
    from infinicoder.planning.diagnostic_flow import schema as failure_schema

    value = deepcopy(failure_schema())
    value["properties"]["question_id"] = value["properties"].pop("failure_id")
    value["required"] = [
        "question_id" if name == "failure_id" else name for name in value["required"]
    ]
    return value


def prepare(state, spec, workspace, args, reads):
    from infinicoder.planning.diagnostic_flow import prepare_inputs

    if (
        not isinstance(args, dict)
        or not {"question_id", "sources", "hypotheses", "argv"} <= set(args)
        or set(args) - {"question_id", "sources", "hypotheses", "argv", "timeout"}
    ):
        raise ValueError(
            "problem_probe needs question_id,sources,hypotheses,argv and optional timeout"
        )
    refresh(state, spec, workspace)
    question = state.get("problem_questions", {}).get(args["question_id"])
    if (
        not question
        or not question.get("exposed")
        or question["status"] == "stale"
        or question.get("last_exposed_artifact") != question.get("artifact_id")
    ):
        raise ValueError("Reference a current actually exposed public-question packet")
    if question.get("attempted") or len(state.get("problem_attempts", [])) >= MAX_EXPERIMENTS:
        raise ValueError(
            "Problem experiment budget exhausted;do not repeat;use existing evidence or original final checks"
        )
    sources, hypotheses, argv, timeout = prepare_inputs(workspace, args, reads)
    for source in sources:
        if question["sources"].get(source["path"]) != source["sha256"]:
            raise ValueError("Experiment source must belong to this current question packet")
        path = workspace.path(source["path"]).resolve()
        for check in spec.checks:
            for protected in check.protected_paths:
                target = Path(protected)
                target = (
                    target.resolve()
                    if target.is_absolute()
                    else workspace.path(protected).resolve()
                )
                if path == target or path.is_relative_to(target):
                    raise ValueError("Acceptance checker is not public problem source")
    declaration = dict(
        requirements=signature(spec),
        public_contract=contract(state, spec),
        question_id=question["id"],
        artifact_id=question["artifact_id"],
        origin="public_contract_question_not_independent_failure",
        failure_id=None,
        failure_key=None,
        revision=workspace.revision(),
        checker_identity=digest(state.get("checker_files", {})),
        sources=sources,
        hypotheses=hypotheses,
        argv=argv,
        timeout=timeout,
        model_claims=True,
        acceptance_proof=False,
    )
    declaration["experiment_key"] = digest(
        [question["id"], sorted((row["path"], row["sha256"]) for row in sources), argv, timeout]
    )
    declaration["declaration_id"] = digest(declaration)
    return declaration


def reserve(runtime, state, declaration):
    question = state["problem_questions"][declaration["question_id"]]
    question["attempted"] = declaration["experiment_key"]
    question["status"] = "experiment_pending"
    state.setdefault("problem_attempts", []).append(declaration["declaration_id"])
    event(
        runtime,
        state,
        "reserved",
        question_id=question["id"],
        artifact_id=declaration["artifact_id"],
        declaration_id=declaration["declaration_id"],
        executed=False,
    )
    runtime.store.save(state)


def save(runtime, state, declaration, record, execution):
    if not isinstance(execution.get("process"), dict) or not execution.get("executed_argv"):
        raise ValueError(
            "A returned caller process record is required;reservation is not execution"
        )
    question = state["problem_questions"][declaration["question_id"]]
    record = dict(
        record,
        question_id=question["id"],
        origin=declaration["origin"],
        evidence_id=declaration["declaration_id"],
        current=True,
        process_returned=True,
        execution_sha256=digest(execution),
        output_excerpt=execution["process"].get("output", "")[:800],
    )
    question["observation"] = record
    question["status"] = (
        "investigated" if record["outcome"] == "observed_prediction_comparison" else "inconclusive"
    )
    observed_event = event(
        runtime,
        state,
        "consumed",
        question_id=question["id"],
        artifact_id=declaration["artifact_id"],
        consumer="returned_public_question_probe",
        declaration_id=declaration["declaration_id"],
        execution_sha256=record["execution_sha256"],
    )
    record.update(
        sequence=observed_event["sequence"],
        call=observed_event["call"],
        step=observed_event["step"],
    )
    event(
        runtime,
        state,
        "observed",
        question_id=question["id"],
        artifact_id=declaration["artifact_id"],
        evidence_id=record["evidence_id"],
        outcome=record["outcome"],
        question_resolved=False,
    )
    return record


def candidate_schema():
    text = dict(type="string", pattern="^[0-9a-f]{64}$")
    return dict(
        type="object",
        properties=dict(
            question_id=text,
            artifact_id=text,
            evidence_ids=dict(type="array", maxItems=2, uniqueItems=True, items=text),
        ),
        required=["question_id", "artifact_id"],
        additionalProperties=False,
    )


def prepare_candidate(state, spec, workspace, args):
    """Optional citation;never supplies a source read,hash,write lease or verdict."""
    args = deepcopy(args)
    reference = args.pop("problem_context", None)
    refresh(state, spec, workspace)
    if reference is None:
        return args, None
    from infinicoder.tools.native_tools import dependency

    dependency()(candidate_schema()).validate(reference)
    question = state.get("problem_questions", {}).get(reference["question_id"])
    if (
        not question
        or not question.get("exposed")
        or question["status"] == "stale"
        or question.get("last_exposed_artifact") != reference["artifact_id"]
        or question.get("artifact_id") != reference["artifact_id"]
    ):
        raise ValueError("Candidate citation requires a current actually exposed problem artifact")
    evidence_ids = reference.get("evidence_ids", [])
    observation = question.get("observation")
    for evidence_id in evidence_ids:
        if (
            not observation
            or observation["evidence_id"] != evidence_id
            or not observation.get("current")
            or not observation.get("process_returned")
        ):
            raise ValueError(
                "Candidate evidence citation requires a current returned public experiment"
            )
    return args, dict(
        question_id=question["id"],
        artifact_id=reference["artifact_id"],
        evidence_ids=list(evidence_ids),
        source_versions=deepcopy(question["sources"]),
        explicit_reference=True,
    )


def bind_candidate(runtime, state, spec, workspace, result, reference=None):
    """Called after the original guarded edit,BEFORE any automatic acceptance."""
    if not result.get("path") or not result.get("after") or result.get("error"):
        return
    if result.get("before") == result["after"]:
        result["problem_candidate_status"] = "no_source_change"
        event(runtime, state, "no_source_change", path=result["path"], changed=False)
        return
    path = workspace.path(result["path"]).relative_to(workspace.root).as_posix()
    if hashlib.sha256(workspace.path(path).read_bytes()).hexdigest() != result["after"]:
        return
    questions = state.get("problem_questions", {})
    if reference is None:
        # Controller association is distinct from an explicit model citation.
        for question in reversed(list(questions.values())):
            if (
                question.get("exposed")
                and question["status"] != "stale"
                and question.get("last_exposed_artifact") == question.get("artifact_id")
                and result.get("before") is not None
                and path in question.get("sources", {})
                and question["sources"][path] == result["before"]
            ):
                reference = dict(
                    question_id=question["id"],
                    artifact_id=question.get("last_exposed_artifact"),
                    evidence_ids=[],
                    source_versions=deepcopy(question["sources"]),
                    explicit_reference=False,
                )
                break
    if reference is None:
        return
    question = questions[reference["question_id"]]
    revision = workspace.revision()
    candidate = dict(
        id=digest([reference, revision, contract(state, spec)]),
        **reference,
        path=path,
        before=result.get("before"),
        after=result["after"],
        revision=revision,
        requirements=contract(state, spec),
        phase=state.get("milestone", 0),
        status="unverified",
        current=True,
        checked_ids=[],
        acceptance_proof=False,
    )
    candidates = question.setdefault("candidates", [])
    candidates.append(candidate)
    if len(candidates) > 4:
        question["candidates_omitted"] = question.get("candidates_omitted", 0) + len(candidates) - 4
        del candidates[:-4]
    result["problem_candidate_id"] = candidate["id"]
    bound_event = event(
        runtime,
        state,
        "candidate_bound",
        question_id=question["id"],
        candidate_id=candidate["id"],
        artifact_id=candidate["artifact_id"],
        explicit_reference=candidate["explicit_reference"],
        evidence_ids=candidate["evidence_ids"],
    )
    candidate.update(
        sequence=bound_event["sequence"], call=bound_event["call"], step=bound_event["step"]
    )
    if candidate["explicit_reference"]:
        event(
            runtime,
            state,
            "consumed",
            question_id=question["id"],
            candidate_id=candidate["id"],
            artifact_id=candidate["artifact_id"],
            consumer="guarded_candidate_reference",
            evidence_ids=candidate["evidence_ids"],
        )


def observe_action(runtime, state, workspace, action, result):
    """Only successful caller reads enter the question's source observations."""
    if action.get("tool") != "read" or result.get("error") or not result.get("sha256"):
        return
    path = result["path"]
    sha = result["sha256"]
    if hashlib.sha256(workspace.path(path).read_bytes()).hexdigest() != sha:
        return
    observation = dict(
        path=path,
        sha256=sha,
        start=result["start"],
        lines=len(result["text"].splitlines()),
        origin="runtime_read",
    )
    for question in state.get("problem_questions", {}).values():
        if question.get("sources", {}).get(path) != sha or question["status"] == "stale":
            continue
        reads = question.setdefault("reads", {})
        if reads.get(path) == observation:
            continue
        reads[path] = observation
        event(runtime, state, "source_observed", question_id=question["id"], source=observation)


def observe_acceptance(runtime, state, spec, workspace, ok, evidence, check_ids, require_artifacts):
    """Observe the original verifier;only actually covered checks complete nodes."""
    revision = workspace.revision()
    requirements = contract(state, spec)
    covered = sorted(
        {
            row["check_id"]
            for row in evidence
            if row.get("check_id") in check_ids
            and row.get("exit_code") == 0
            and not row.get("error")
            and not row.get("rejected_because")
        }
    )
    caller_ids = {check.id for check in spec.checks}
    callback = any(row.get("check_id") == "callback" and row.get("ok") is True for row in evidence)
    complete = bool(
        ok
        and require_artifacts
        and ((caller_ids and caller_ids <= set(covered)) or (not caller_ids and callback))
    )
    for question in state.get("problem_questions", {}).values():
        for candidate in question.get("candidates", []):
            if candidate["revision"] != revision or candidate["requirements"] != requirements:
                continue
            candidate.update(
                current=True,
                checked_ids=covered,
                acceptance_evidence_sha256=digest(evidence),
                status="accepted"
                if complete
                else "checked_partial"
                if ok and covered
                else "rejected"
                if not ok
                else "unverified",
                acceptance_proof=complete,
            )
            event(
                runtime,
                state,
                "candidate_checked",
                question_id=question["id"],
                candidate_id=candidate["id"],
                checked_ids=covered,
                original_verifier_ok=bool(ok),
                final_caller_coverage=complete,
            )


def required_context(state):
    return any(
        q.get("observation") or q.get("candidates")
        for q in state.get("problem_questions", {}).values()
    )


def stamp_history(state, spec, workspace, phase, action_schema=None):
    """Attach controller provenance to the existing paid-turn record."""
    record = state["shell_loop_history"][-1]
    record["working_state_provenance"] = dict(
        contract_ref=contract(state, spec),
        phase=phase,
        source_revision=workspace.revision(),
        origin="controller_returned_action",
        acceptance_proof=False,
    )
    if record.get("proposal_rejected_before_dispatch") and isinstance(action_schema, dict):
        for branch in action_schema.get("anyOf", []):
            if branch.get("properties", {}).get("tool", {}).get("const") == "read":
                properties = branch["properties"]["arguments"].get("properties", {})
                record["rejected_read_constraints"] = {
                    name: {
                        key: value
                        for key, value in schema.items()
                        if key in ("minimum", "maximum", "type")
                    }
                    for name, schema in properties.items()
                    if name in ("start", "count")
                }


def working_context(state, spec, workspace, index, token_budget, estimator):
    """Bounded projection of canonical questions,read ledger and actual turns.

    No model claim is minted from reading or exposure. No semantic fact is
    stored here. Historical results retain references instead of becoming
    current facts after edits,phase changes or requirement updates.
    """
    refresh(state, spec, workspace)
    requirements = contract(state, spec)
    phase = state.get("milestone", 0)
    history = state.get("shell_loop_history", [])
    revision = workspace.revision() if history else None
    questions = state.get("problem_questions", {})
    current = [
        q
        for q in questions.values()
        if q.get("requirements") == requirements
        and q.get("phase") == phase
        and q.get("status") != "stale"
    ]
    ident = state.get("question_memory_report", {}).get("question_id")
    question = next((q for q in current if q["id"] == ident), current[-1] if current else None)

    def bounded(value, limit=500):
        text = encoded(value)
        return (
            value
            if len(text) <= limit
            else dict(
                excerpt=text[:limit],
                complete=False,
                sha256=hashlib.sha256(text.encode()).hexdigest(),
                full_record="session_state_and_events",
            )
        )

    def result_view(record):
        result = record.get("observation", {})
        action = record.get("action") or {}
        provenance = record.get("working_state_provenance", {})
        rejected = record.get("proposal_rejected_before_dispatch") is True
        valid = (
            provenance.get("contract_ref") == requirements
            and provenance.get("phase") == phase
            and provenance.get("source_revision") == revision
        )
        kind = (
            "rejected_before_dispatch"
            if rejected
            else "caller_verifier_result"
            if "accepted" in result
            else "tool_returned"
        )
        view = dict(
            call=record["call"],
            step=record["step"],
            kind=kind,
            execution_status="not_executed"
            if rejected
            else "returned_error"
            if result.get("error")
            else "returned_result_not_acceptance",
            current=valid,
            source_revision=provenance.get("source_revision"),
            phase=provenance.get("phase", record.get("phase")),
            action_ref=digest(action),
            full_turn_ref=digest(record),
            historical_not_current_fact=not valid,
        )
        view["action"] = dict(
            tool=action.get("tool"),
            arguments={
                k: bounded(v, 160)
                for k, v in action.get("arguments", {}).items()
                if k in ("path", "start", "count", "symbol", "symbol_id", "name", "query")
            },
        )
        if rejected:
            view["rejected_proposal"] = bounded(record.get("proposal"), 600)
            if record.get("rejected_read_constraints"):
                view["actual_schema_constraints"] = record["rejected_read_constraints"]
        for name in (
            "path",
            "sha256",
            "start",
            "covered_line_count",
            "total_lines",
            "read_truncated",
            "inspection_progress",
            "before",
            "after",
            "retained_sha256",
            "accepted",
            "error",
            "failures",
            "exit_code",
            "timed_out",
            "problem_candidate_id",
        ):
            if name in result:
                view[name] = bounded(result[name], 650 if name in ("failures", "error") else 280)
        if valid and result.get("path") and result.get("after"):
            try:
                view["currently_retained_source_sha256"] = hashlib.sha256(
                    workspace.path(result["path"]).read_bytes()
                ).hexdigest()
            except (OSError, ValueError):
                view["currently_retained_source_sha256"] = None
        if action.get("tool") == "read_batch":
            view["batch_children"] = [
                dict(
                    ordinal=item["ordinal"],
                    executed=item["executed"],
                    action=bounded(item["action"], 200),
                    result_ref=digest(item.get("observation", {})),
                    error=item.get("observation", {}).get("error"),
                )
                for item in result.get("items", [])
            ]
        return view

    packet = dict(
        version=2,
        contract_ref=requirements,
        phase=phase,
        current_question=None,
        recent_results=[],
        current_reads=[],
        candidate_states=[],
        runtime_observations=[],
        model_hypotheses=[],
        omitted=dict(
            turns=max(0, len(history) - 1),
            reads=0,
            candidates=0,
            observations=0,
            questions=max(0, len(questions) - int(question is not None)),
        ),
        rule="Controller actions and source visibility are observations,not understanding or causes. Model claims remain hypotheses. Use ordinary reads,edits,verification or configured probes;no registration or worker switch is required.",
    )
    if question:
        packet["current_question"] = dict(
            id=question["id"],
            text=bounded(question["question"], 400),
            status=question["status"],
            origin="public_contract_scope_hypothesis",
            selection_is_hypothesis=True,
            artifact_id=question.get("last_exposed_artifact"),
            source_versions=question.get("sources", {}),
        )
    else:
        packet["current_question"] = dict(
            task_ref="task",
            origin="public_contract_scope_hypothesis",
            selection_is_hypothesis=True,
            status="no_current_versioned_question",
        )
    # Latest actual outcome is mandatory within the new view;never publish a
    # half-result as if it were a complete new state. The original caller core
    # still contains last_failure when the entire view must be omitted.
    if history:
        packet["recent_results"].append(result_view(history[-1]))
    render = lambda: encoded(dict(task_working_state=packet))
    if estimator.estimate(render()) > token_budget:
        return "", dict(
            included=False,
            reason="latest_real_result_and_state_do_not_fit",
            tokens=0,
            available_tokens=token_budget,
            ordinary_context_retained=True,
        )

    def offer(key, value):
        packet[key].append(value)
        if estimator.estimate(render()) > token_budget:
            packet[key].pop()
            return False
        return True

    candidates, unused = recent_records(
        questions, "candidates", question["id"] if question else None, limit=3
    )
    for owner, candidate in reversed(candidates):
        value = {
            k: deepcopy(candidate[k])
            for k in (
                "id",
                "path",
                "before",
                "after",
                "revision",
                "phase",
                "status",
                "current",
                "checked_ids",
                "acceptance_proof",
            )
            if k in candidate
        }
        value.update(
            question_id=owner["id"],
            current=bool(candidate.get("current") and owner.get("phase") == phase),
            historical_not_current_fact=not (
                candidate.get("current") and owner.get("phase") == phase
            ),
        )
        if not offer("candidate_states", value):
            packet["omitted"]["candidates"] += 1
    packet["omitted"]["candidates"] += unused
    observations, unused = recent_records(
        questions, "observation", question["id"] if question else None, limit=2
    )
    for owner, observation in reversed(observations):
        value = {
            k: bounded(observation[k], 320)
            for k in (
                "evidence_id",
                "revision",
                "outcome",
                "comparisons",
                "output_sha256",
                "current",
                "output_excerpt",
            )
            if k in observation
        }
        valid = bool(
            observation.get("current")
            and owner.get("phase") == phase
            and owner.get("requirements") == requirements
        )
        value.update(question_id=owner["id"], current=valid, historical_not_current_fact=not valid)
        if not offer("runtime_observations", value):
            packet["omitted"]["observations"] += 1
        for hypothesis in observation.get("hypotheses", []):
            offer(
                "model_hypotheses",
                dict(
                    kind="model_declared_hypothesis_not_fact",
                    current_dependencies=valid,
                    evidence_id=observation.get("evidence_id"),
                    claim=bounded(hypothesis, 240),
                ),
            )
    packet["omitted"]["observations"] += unused
    if index is not None:
        from infinicoder.memory.inspection_ledger import InspectionLedger

        reads = InspectionLedger(index).current()
    else:
        from infinicoder.memory.inspection_memory import current_coverage

        reads = current_coverage(state, workspace)
    for read in reversed(reads):
        value = {
            k: deepcopy(read[k])
            for k in (
                "path",
                "source_sha256",
                "total_lines",
                "read_spans",
                "partial_prefix_characters",
                "full_file_visible",
            )
            if k in read
        }
        value.update(origin="controller_read_observation", visibility_is_not_understanding=True)
        if not offer("current_reads", value):
            packet["omitted"]["reads"] += 1
    for record in reversed(history[-3:-1]):
        if offer("recent_results", result_view(record)):
            packet["omitted"]["turns"] -= 1
    # Omissions counters can grow in width;enforce the final serialized bound.
    if estimator.estimate(render()) > token_budget:
        return "", dict(
            included=False,
            reason="complete_working_state_omission_metadata_does_not_fit",
            tokens=0,
            ordinary_context_retained=True,
        )
    text = render()
    return text, dict(
        included=True,
        tokens=estimator.estimate(text),
        version=2,
        view_sha256=hashlib.sha256(text.encode()).hexdigest(),
        sources="canonical_problem_questions+InspectionLedger+shell_loop_history",
        omitted=deepcopy(packet["omitted"]),
        hypotheses_are_model_claims=True,
    )


def attach_working_context(spec, state, workspace, index, assembler, required, overhead):
    """Prioritize the shared state before optional teaching and old history."""
    unused, base = assembler.assemble(required, {}, fixed_overhead_text=overhead)
    text, report = working_context(
        state,
        spec,
        workspace,
        index,
        max(0, min(1600, base["remaining_tokens"] - 48)),
        assembler.estimator,
    )
    state["working_state_report"] = report
    return required + "\n" + text if report["included"] else required
