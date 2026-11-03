"""Two Native candidates, public counterexamples, one independent final.

No copied query loop, seed changes, private judge selection, or hidden retries.
The default native path is unchanged unless this policy is explicitly supplied.
"""

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import signal
import sys
import threading
import time

from infinicoder.runtime.agent_execution import (
    CoreOutcome,
    FinalAcceptance,
    ModelBinding,
    NativeTaskBinding,
    PublicTaskContext,
    NATIVE_CORE,
    UnsupportedCore,
)
from infinicoder.search.candidate_checkpoint import (
    CandidateCheckpointStore,
    CheckpointConflict,
    revision,
)
from infinicoder.verification.candidate_public_check import (
    CandidatePublicCheck,
    PublicCandidateVerdict,
    contract_hash,
)
from infinicoder.runtime.native_run import CoreDeadline, _deadline_binding, _initial_messages
from infinicoder.execution.native_sandbox import CANDIDATE_SOURCE_PATH, CANDIDATE_EVIDENCE_PATH


@dataclass(frozen=True)
class CandidateSearchPolicy:
    public_check: CandidatePublicCheck | None = None
    exclusive_workspace: bool = False
    phase_history_factory: object = None
    hypothesis_branch: bool = False

    @property
    def identity(self):
        return (
            "public_hypothesis_branch_v1"
            if self.hypothesis_branch
            else "public_counterexample_branch_v1"
        )

    max_candidates = 2
    first_call_fraction = 0.6
    first_time_fraction = 0.6

    def validate(self, request):
        if type(self.hypothesis_branch) is not bool:
            raise UnsupportedCore("hypothesis branching is an explicit boolean")
        if type(self.exclusive_workspace) is not bool:
            raise UnsupportedCore("explicit candidate workspace ownership required")
        if self.public_check is not None and not isinstance(
            self.public_check, CandidatePublicCheck
        ):
            raise UnsupportedCore("caller-issued public selector required")
        if any(
            x is not None
            for x in (request.context_adapter, request.repair_adapter, request.planning_adapter)
        ):
            raise UnsupportedCore(
                "first candidate search profile uses plain agent without other adapters"
            )
        if self.phase_history_factory is not None:
            counter = getattr(request.transport_adapter, "counter", None)
            if (
                not callable(self.phase_history_factory)
                or not callable(counter)
                or not getattr(counter, "identity", None)
            ):
                raise UnsupportedCore(
                    "phase history requires an explicit factory and identified actual counter"
                )


def _write(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8"
    )


def _public_task(request, command):
    if isinstance(command, NativeTaskBinding):
        if command.task != request.spec.task:
            raise UnsupportedCore("verbatim binding changed the public TaskSpec task")
        return command.task, "caller_verbatim", command.identity
    if isinstance(command, str) and command.strip():
        return (
            request.spec.task
            + "\nWorking directory: /workspace. Independent acceptance command:\n"
            + command,
            "legacy_public_command",
            request.acceptance_adapter.identity,
        )
    raise UnsupportedCore("public task context must yield a command or NativeTaskBinding")


def _sdk_phase(
    request, binding, environment, task, phase, deadline, calls, record, observed_paths=()
):
    """One NativeAgent.run, inside the already-open original caller contexts."""
    from infinicoder.runtime.native_agent import NativeAgent
    from infinicoder.models.infinilm_client import InfiniLMTextModel, default_config

    config = default_config()
    state = Path(request.state_dir) / phase
    state.mkdir(exist_ok=False)
    trajectory = state / "native-trajectory.json"
    kwargs = dict(config["model"].get("model_kwargs", {})) | binding.model_kwargs
    kwargs.update(
        max_tokens=request.model_profile.output_reserve,
        num_retries=0,
        timeout=min(request.budget.request_timeout, max(0.001, deadline - time.monotonic())),
    )
    model = InfiniLMTextModel(
        **{k: v for k, v in config["model"].items() if k not in ("model_kwargs", "model_name")},
        model_name=binding.model_name,
        model_kwargs=kwargs,
    )
    agent = NativeAgent(
        model,
        environment,
        **(
            config["agent"]
            | dict(
                step_limit=calls,
                wall_time_limit_seconds=max(1, int(deadline - time.monotonic())),
                cost_limit=0,
                output_path=trajectory,
            )
        ),
    )
    started = time.monotonic()
    result = {}
    error = None
    try:
        counter = getattr(request.transport_adapter, "counter", None)
        if callable(counter):
            count = counter(_initial_messages(agent, task), deadline - time.monotonic())
            record["initial_input_tokens"] = count
            if count > request.model_profile.input_budget():
                result = dict(exit_status="CandidateInitialContextExceeded", submission="")
                return result
        limits = getattr(request.transport_adapter, "limit_request", None)
        bound = (
            limits(
                output_tokens=request.model_profile.output_reserve,
                absolute_deadline=deadline,
                max_physical_calls=calls,
            )
            if callable(limits)
            else nullcontext()
        )
        history = nullcontext()
        factory = request.candidate_search.phase_history_factory
        if factory is not None:
            from infinicoder.memory.native_phase_history import (
                NativePhaseHistoryContext,
                PhaseHistoryBinding,
            )

            descriptor = PhaseHistoryBinding(
                Path(request.state_dir).resolve(),
                state.resolve(),
                phase,
                Path(request.workspace).resolve(),
                hashlib.sha256(request.spec.task.encode()).hexdigest(),
                hashlib.sha256(task.encode()).hexdigest(),
                tuple(observed_paths),
                deadline,
                counter,
                counter.identity,
            )
            context = factory(descriptor)
            if type(context) is not NativePhaseHistoryContext or context.binding != descriptor:
                raise UnsupportedCore(
                    "phase factory must return a fresh exact NativePhaseHistoryContext"
                )
            history = context.bind_phase(request, model, environment)
            record["phase_history"] = dict(
                identity=context.identity,
                binding=descriptor.manifest(),
                report_ref=str(state / "native-phase-history.json"),
            )
        hypothesis = nullcontext()
        if phase == "B" and request.candidate_search.hypothesis_branch:
            from infinicoder.search.branch_hypothesis import bind_first_response

            hypothesis = bind_first_response(
                model, request.state_dir, record.setdefault("hypothesis", {})
            )
        with bound, _deadline_binding(model, environment, deadline), history, hypothesis:
            result = agent.run(task)
    except CoreDeadline as exc:
        error = type(exc).__name__ + ": " + str(exc)
        result = dict(exit_status="CandidatePhaseDeadline", submission="")
    except KeyboardInterrupt:
        result = dict(exit_status="Interrupted", submission="")
        raise
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
        result = dict(exit_status=type(exc).__name__, submission="")
    finally:
        agent.save(trajectory)
        record.update(
            phase=phase,
            trajectory=str(trajectory),
            sdk_agent_calls=agent.n_calls,
            sdk_step_limit=calls,
            phase_deadline=deadline,
            wall_seconds=time.monotonic() - started,
            result=result,
            error=error,
            usage_ledger_ref=binding.usage_ledger_ref,
            physical_requests="original shared transport ledger;agent calls reported separately",
        )
    return result


def _observed_changes(store, baseline, candidate, paths):
    before, after = store.load(baseline), store.load(candidate)
    for path in paths:
        if path == "." and before != after:
            return True
        if CandidateCheckpointStore._part(before, path) != CandidateCheckpointStore._part(
            after, path
        ):
            return True
    return False


def _check_live(store, candidate, check, bound, state, deadline, contract_manifest):
    """Preserve exact public evidence and fail closed on changed/late versions."""
    folder = state / ("public-" + candidate.name)
    folder.mkdir(exist_ok=False)
    status, explanation = "unknown", ""
    raw = None
    error = None
    expected_contract = contract_hash(contract_manifest)
    before = revision(store.scan())
    started = time.monotonic()
    if before != candidate.revision:
        raise CheckpointConflict("candidate changed before public validation")
    try:
        if contract_hash(check.manifest()) != expected_contract:
            raise ValueError("public selector contract changed")
        raw = bound.invoke(deadline)
        raw_bytes = json.dumps(raw, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
        if len(raw_bytes) > 4 * 1024**2:
            raise ValueError("public evidence exceeded bounded complete record")
        (folder / "raw.json").write_bytes(raw_bytes)
        decoded = check.decoder(raw)
        if not isinstance(decoded, PublicCandidateVerdict):
            raise TypeError("caller decoder must return PublicCandidateVerdict")
        if contract_hash(check.manifest()) != expected_contract:
            raise ValueError("public selector contract changed during validation")
        status, explanation = decoded.status, decoded.counterexample
        if time.monotonic() > deadline:
            status, explanation = "unknown", "public operation completed after its allowed deadline"
    except TimeoutError as exc:
        error = type(exc).__name__ + ": " + str(exc)
    except Exception as exc:
        status = "env_error"
        error = type(exc).__name__ + ": " + str(exc)
    after = revision(store.scan())
    if after != before:
        status, explanation = (
            "unknown",
            "workspace changed during public operation;result cannot select a candidate",
        )
        store.restore(candidate, expected_revision=after)
    raw_path = folder / "raw.json"
    report = dict(
        candidate=candidate.name,
        status=status,
        counterexample=explanation,
        tested_revision=before,
        revision_after_operation=after,
        retained_revision=candidate.revision,
        contract_sha256=expected_contract,
        operation_contract_sha256=bound.contract_sha256,
        coverage=contract_manifest["operation"]["public_coverage"],
        environment=contract_manifest["operation"]["environment_identity"],
        raw_ref=str(raw_path) if raw_path.exists() else None,
        raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest() if raw_path.exists() else None,
        error=error,
        wall_seconds=time.monotonic() - started,
        controller_origin=True,
        independent_final_used=False,
    )
    _write(folder / "result.json", report)
    return report


def _branch_task(public_task, store, baseline, candidate, verdict, state, *, hypothesis=False):
    folder = state / "candidate-evidence"
    folder.mkdir(mode=0o700, exist_ok=False)
    before, after = store.load(baseline), store.load(candidate)
    changed = [
        key
        for key in sorted(set(before["entries"]) | set(after["entries"]))
        if before["entries"].get(key) != after["entries"].get(key)
    ]
    _write(
        folder / "changes.json",
        dict(
            baseline_revision=baseline.revision,
            A_revision=candidate.revision,
            changed_entries=changed,
            source_identity="complete bytes,modes,contained links;not external runtime state",
        ),
    )
    _write(folder / "public-A.json", verdict)
    (folder / "raw-public-A.json").write_bytes(Path(verdict["raw_ref"]).read_bytes())
    _write(folder / "A-manifest.json", after)
    # File names/summaries are bounded in the prompt;full artifacts stay readable.
    summary = dict(
        A_revision=candidate.revision,
        baseline_revision=baseline.revision,
        tested_revision=verdict["tested_revision"],
        public_contract_sha256=verdict["contract_sha256"],
        public_coverage=verdict["coverage"],
        counterexample=verdict["counterexample"],
        changed_entry_count=len(changed),
        changed_entries_preview=changed[:32],
        complete_changes_path=CANDIDATE_EVIDENCE_PATH + "/changes.json",
        public_result_path=CANDIDATE_EVIDENCE_PATH + "/public-A.json",
        raw_public_result_path=CANDIDATE_EVIDENCE_PATH + "/raw-public-A.json",
    )
    extra = ""
    if hypothesis:
        from infinicoder.search.branch_hypothesis import instructions

        extra = instructions()
    return (
        public_task
        + "\n\nPUBLIC COUNTEREXAMPLE FOR ALTERNATIVE CANDIDATE B\n"
        + json.dumps(summary, ensure_ascii=False)
        + (
            "\nThe live /workspace has been restored to the original baseline. Construct an alternative implementation "
            "of the original task, using this actual public failure as evidence. All original requirements, interfaces "
            "and scope still apply. Historical candidate A is readonly at "
            + CANDIDATE_SOURCE_PATH
            + ". "
            "Its code is historical, not the current workspace. Complete public evidence is readonly at "
            + CANDIDATE_EVIDENCE_PATH
            + ". /tmp/task-scratch is fresh for B. Only live /workspace is selected. "
            "Public coverage is incomplete; the independent final check has not run. Use the ordinary tools and submission protocol."
        )
        + extra
    )


def run_candidate_search(request):
    if sys.platform != "linux" or threading.current_thread() is not threading.main_thread():
        raise UnsupportedCore(
            "native core requires a Linux main-thread process for strict deadlines"
        )
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise UnsupportedCore("native core cannot replace an existing wall alarm")
    state = Path(request.state_dir).resolve()
    state.mkdir(parents=True, exist_ok=True)
    marker = state / "native-run.json"
    with marker.open("x", encoding="utf8") as stream:
        json.dump(
            dict(status="started", backend=NATIVE_CORE, strategy=request.candidate_search.identity),
            stream,
        )
    deadline = time.monotonic() + request.budget.wall_seconds
    manifest = state / "native-candidate-search.json"
    report = dict(
        strategy=request.candidate_search.identity,
        shared_deadline=deadline,
        max_model_calls=request.budget.max_model_calls,
        phases=[],
        events=[],
        checks=[],
        applicability="pending",
        selected=None,
        independent_final_calls=0,
    )
    if request.candidate_search.phase_history_factory is not None:
        report["phase_history_policy"] = dict(
            identity="phase_history_budget_v1",
            default_enabled=False,
            planner_enabled=False,
            phase_databases_independent=True,
        )

    def event(kind, **values):
        report["events"].append(dict(kind=kind, monotonic=time.monotonic(), **values))
        _write(manifest, report)

    result = {}
    final = None
    error = acceptance_error = None
    ledger_ref = ""
    interrupted = False
    store = selected = None
    selection_safe = False
    try:
        with request.transport_adapter.open(request, deadline) as binding:
            if not isinstance(binding, ModelBinding) or not binding.usage_ledger_ref:
                raise UnsupportedCore("original budgeted model binding and shared ledger required")
            ledger_ref = binding.usage_ledger_ref
            with request.execution_adapter.open(request, deadline) as environment:
                with request.acceptance_adapter.public_command(request, deadline) as public:
                    operation = context_check = None
                    if isinstance(public, PublicTaskContext):
                        operation, context_check, public = (
                            public.operation,
                            public.candidate_check,
                            public.binding,
                        )
                    task, mode, identity = _public_task(request, public)
                    _write(
                        state / "native-public-task.json",
                        dict(
                            task=task,
                            mode=mode,
                            identity=identity,
                            task_sha256=hashlib.sha256(task.encode()).hexdigest(),
                            spec_task_sha256=hashlib.sha256(request.spec.task.encode()).hexdigest(),
                            final_acceptance="Independent verifier still required;binding proves no result",
                        ),
                    )
                    policy = request.candidate_search
                    if (
                        policy.public_check is not None
                        and context_check is not None
                        and policy.public_check is not context_check
                    ):
                        raise UnsupportedCore(
                            "conflicting request and public-context candidate selectors"
                        )
                    check = (
                        policy.public_check if policy.public_check is not None else context_check
                    )
                    reason = None
                    if check is None or operation is not check.operation:
                        reason = "no_exact_caller_public_selector"
                    elif not policy.exclusive_workspace:
                        reason = "no_exclusive_workspace_authority"
                    elif not callable(getattr(environment, "bind_candidate_branch", None)):
                        reason = "no_fresh_scratch_readonly_A_binding"
                    elif not all(
                        callable(getattr(request.transport_adapter, x, None))
                        for x in ("limit_request", "remaining_physical_calls")
                    ):
                        reason = "no_shared_physical_phase_budget"
                    elif request.budget.max_model_calls < 2:
                        reason = "insufficient_calls_for_two_candidates"
                    baseline = None
                    restore_reserve = 1
                    if reason is None:
                        try:
                            started = time.monotonic()
                            store = CandidateCheckpointStore(
                                request.workspace,
                                state / "candidate-checkpoints",
                                exclusive_workspace=True,
                                remaining=lambda: deadline - time.monotonic(),
                            )
                            baseline = store.capture("baseline")
                            restore_reserve = max(1, 2 * (time.monotonic() - started))
                        except (OSError, ValueError, TimeoutError) as exc:
                            reason = "complete_checkpoint_unavailable:" + type(exc).__name__
                    available = deadline - time.monotonic()
                    generation = (
                        (available - 2 * operation.timeout_seconds - restore_reserve)
                        if reason is None
                        else 0
                    )
                    if reason is None and generation < 2:
                        reason = "insufficient_wall_for_two_public_checks_and_generation"
                    if reason is not None:
                        report["applicability"] = "N/A;plain"
                        event("pre_generation_NA", reason=reason)
                        phase = {}
                        report["phases"].append(phase)
                        result = _sdk_phase(
                            request,
                            binding,
                            environment,
                            task,
                            "plain",
                            deadline,
                            request.budget.max_model_calls,
                            phase,
                            operation.observed_paths if operation is not None else (),
                        )
                        selection_safe = True
                    else:
                        report["applicability"] = "eligible"
                        contract = json.loads(json.dumps(check.manifest()))
                        report["public_contract"] = contract
                        event("baseline_captured", revision=baseline.revision)
                        with operation.bind(request, deadline) as bound:
                            phase = {}
                            report["phases"].append(phase)
                            result = _sdk_phase(
                                request,
                                binding,
                                environment,
                                task,
                                "A",
                                time.monotonic() + generation * policy.first_time_fraction,
                                max(
                                    1,
                                    int(
                                        request.budget.max_model_calls * policy.first_call_fraction
                                    ),
                                ),
                                phase,
                                operation.observed_paths,
                            )
                            A = store.capture("A")
                            selected = A
                            event("candidate_captured", candidate="A", revision=A.revision)
                            verdict = _check_live(
                                store,
                                A,
                                check,
                                bound,
                                state,
                                min(
                                    deadline - restore_reserve,
                                    time.monotonic() + operation.timeout_seconds,
                                ),
                                contract,
                            )
                            report["checks"].append(verdict)
                            event("public_validation", candidate="A", status=verdict["status"])
                            sdk_left = request.budget.max_model_calls - sum(
                                p.get("sdk_agent_calls", 0) for p in report["phases"]
                            )
                            calls = min(
                                sdk_left, request.transport_adapter.remaining_physical_calls()
                            )
                            b_deadline = deadline - operation.timeout_seconds - restore_reserve
                            if (
                                verdict["status"] == "public_fail"
                                and _observed_changes(store, baseline, A, operation.observed_paths)
                                and calls > 0
                                and b_deadline > time.monotonic()
                            ):
                                task_B = _branch_task(
                                    task,
                                    store,
                                    baseline,
                                    A,
                                    verdict,
                                    state,
                                    hypothesis=policy.hypothesis_branch,
                                )
                                store.restore(baseline, expected_revision=A.revision)
                                event(
                                    "baseline_restored_for_B", baseline_revision=baseline.revision
                                )
                                phase = {}
                                report["phases"].append(phase)
                                with environment.bind_candidate_branch(
                                    A.tree_ref, state / "candidate-evidence"
                                ) as branch:
                                    phase["branch_binding"] = branch
                                    result = _sdk_phase(
                                        request,
                                        binding,
                                        environment,
                                        task_B,
                                        "B",
                                        b_deadline,
                                        calls,
                                        phase,
                                        operation.observed_paths,
                                    )
                                B = store.capture("B")
                                event("candidate_captured", candidate="B", revision=B.revision)
                                hypothesis_ok = (
                                    not policy.hypothesis_branch
                                    or phase.get("hypothesis", {}).get("status") == "proposed"
                                )
                                if not hypothesis_ok:
                                    other = dict(
                                        candidate="B",
                                        status="unknown",
                                        reason="alternative_hypothesis_not_established",
                                    )
                                    event("alternative_not_established", candidate="B")
                                elif B.revision == A.revision:
                                    other = dict(verdict, candidate="B", duplicate_of="A")
                                    event("duplicate_candidate", revision=B.revision)
                                else:
                                    other = _check_live(
                                        store,
                                        B,
                                        check,
                                        bound,
                                        state,
                                        min(
                                            deadline - restore_reserve,
                                            time.monotonic() + operation.timeout_seconds,
                                        ),
                                        contract,
                                    )
                                report["checks"].append(other)
                                if policy.hypothesis_branch:
                                    phase.setdefault("hypothesis", {}).update(
                                        source_differs_from_A=B.revision != A.revision,
                                        observed_source_changed=_observed_changes(
                                            store, baseline, B, operation.observed_paths
                                        ),
                                        public_status=other["status"],
                                        semantic_diversity_proven=False,
                                    )
                                selected = B if other["status"] == "public_pass" else A
                                store.restore(selected, expected_revision=B.revision)
                                event(
                                    "candidate_retained",
                                    candidate=selected.name,
                                    revision=selected.revision,
                                )
                            else:
                                event(
                                    "search_ended_after_A",
                                    status=verdict["status"],
                                    observed_source_changed=_observed_changes(
                                        store, baseline, A, operation.observed_paths
                                    ),
                                )
                            if revision(store.scan()) != selected.revision:
                                raise CheckpointConflict("retained candidate version mismatch")
                            report["selected"] = asdict(selected)
                            selection_safe = True
                            event(
                                "retained_version_verified",
                                candidate=selected.name,
                                revision=selected.revision,
                            )
    except CoreDeadline as exc:
        error = type(exc).__name__ + ": " + str(exc)
        result = dict(exit_status="TaskDeadline", submission="")
    except KeyboardInterrupt as exc:
        interrupted = True
        error = type(exc).__name__ + ": " + str(exc)
        result = dict(exit_status="Interrupted", submission="")
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
        result = dict(exit_status=type(exc).__name__, submission="")
    if not interrupted and selection_safe:
        try:
            report["independent_final_calls"] += 1
            event("independent_final_started")
            final = request.acceptance_adapter.verify(request)
            if not isinstance(final, FinalAcceptance):
                raise TypeError("independent verifier must return FinalAcceptance")
        except Exception as exc:
            final = None
            acceptance_error = type(exc).__name__ + ": " + str(exc)
        except KeyboardInterrupt as exc:
            interrupted = True
            error = type(exc).__name__ + ": " + str(exc)
    report.update(
        selection_safe=selection_safe,
        interrupted=interrupted,
        error=error,
        acceptance_error=acceptance_error,
    )
    _write(manifest, report)
    phases = report["phases"]
    trajectory = phases[-1].get("trajectory", "") if phases else ""
    result = dict(
        result,
        candidate_search_manifest_ref=str(manifest),
        phase_count=len(phases),
        applicability=report["applicability"],
        retained_candidate=selected.name if selected is not None else None,
        retained_candidate_sdk_result=next(
            (
                p["result"]
                for p in phases
                if selected is not None and p.get("phase") == selected.name
            ),
            None,
        ),
        controller_selection_is_not_agent_submission=True,
    )
    outcome = CoreOutcome(
        NATIVE_CORE,
        result.get("exit_status", "Unknown"),
        str(Path(request.workspace).resolve()),
        trajectory,
        ledger_ref,
        result,
        final,
        sum(p.get("sdk_agent_calls", 0) for p in phases),
        error,
        acceptance_error,
    )
    _write(marker, asdict(outcome))
    return outcome
