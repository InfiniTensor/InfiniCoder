"""Run the InfiniCoder text-action core with caller-owned budgets and verification."""

from contextlib import contextmanager, nullcontext
from dataclasses import asdict
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
    AuxiliaryCallBudget,
    PublicTaskContext,
    NATIVE_CORE,
    UnsupportedCore,
    validate_native_request,
)


class CoreDeadline(KeyboardInterrupt):
    """Non-retryable by the Native core, including its network retry backoff."""


def _initial_messages(agent, task):
    """Render the actual agent prefix for capacity checks, without changing it."""
    previous = agent.extra_template_vars
    agent.extra_template_vars = previous | dict(task=task)
    try:
        return [
            agent.model.format_message(role=role, content=agent._render_template(template))
            for role, template in (
                ("system", agent.config.system_template),
                ("user", agent.config.instance_template),
            )
        ]
    finally:
        agent.extra_template_vars = previous


@contextmanager
def _deadline_binding(model, environment, deadline):
    # Same Linux main-thread budget boundary as the existing reference runner.
    # Do not interrupt a running shell: its executor must kill children on timeout.
    query, execute = model.query, environment.execute

    def bounded_query(*args, **kwargs):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CoreDeadline("shared task deadline exhausted before query")
        previous = signal.getsignal(signal.SIGALRM)
        if signal.getitimer(signal.ITIMER_REAL)[0]:
            raise RuntimeError("refusing to replace an existing alarm")

        def expired(signum, frame):
            raise CoreDeadline("shared task deadline exhausted, including upstream retries")

        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, remaining)
        try:
            return query(*args, **kwargs)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

    def bounded_execute(*args, **kwargs):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CoreDeadline("shared task deadline exhausted before command")
        kwargs["timeout"] = min(kwargs.get("timeout") or environment.config.timeout, remaining)
        return execute(*args, **kwargs)

    model.query, environment.execute = bounded_query, bounded_execute
    try:
        yield
    finally:
        model.query, environment.execute = query, execute


class NativeRunCore:
    def run(self, request):
        validate_native_request(request)
        from infinicoder.execution.native_scoped_execution import ScopedNativeExecution

        scope_adapter = (
            request.execution_adapter
            if type(request.execution_adapter) is ScopedNativeExecution
            else None
        )
        if request.implementation_proposal is not None:
            from infinicoder.runtime.native_implementation_proposal import (
                run_implementation_proposal,
            )

            return run_implementation_proposal(request)
        if request.candidate_search is not None:
            from infinicoder.search.native_candidate_search import run_candidate_search

            return run_candidate_search(request)
        if request.repair_adapter is not None:
            from infinicoder.runtime.extensions.native_repair_run import run_with_repair

            return run_with_repair(request)
        if sys.platform != "linux" or threading.current_thread() is not threading.main_thread():
            raise UnsupportedCore(
                "native core requires a Linux main-thread process for strict deadlines"
            )
        if signal.getitimer(signal.ITIMER_REAL)[0]:
            raise UnsupportedCore("native core cannot replace an existing wall alarm")
        state = Path(request.state_dir).resolve()
        # One actual invocation, never a replay/resume of pending model actions.
        state.mkdir(parents=True, exist_ok=True)
        marker = state / "native-run.json"
        with marker.open("x", encoding="utf-8") as stream:
            json.dump(dict(status="started", backend=NATIVE_CORE), stream)
        deadline = time.monotonic() + request.budget.wall_seconds
        from infinicoder.runtime.native_agent import NativeAgent
        from infinicoder.models.infinilm_client import InfiniLMTextModel, default_config

        config = default_config()
        trajectory = state / "native-trajectory.json"
        final = None
        acceptance_error = error = None
        result = {}
        agent = None
        ledger_ref = ""
        auxiliary = None
        interrupted = False
        try:
            with request.transport_adapter.open(request, deadline) as binding:
                if not isinstance(binding, ModelBinding) or not binding.usage_ledger_ref:
                    raise ValueError(
                        "transport must yield ModelBinding with the original usage ledger reference"
                    )
                ledger_ref = binding.usage_ledger_ref
                kwargs = dict(config["model"].get("model_kwargs", {}))
                kwargs.update(binding.model_kwargs)
                kwargs.update(
                    max_tokens=request.model_profile.output_reserve,
                    timeout=min(
                        request.budget.request_timeout, max(0.001, deadline - time.monotonic())
                    ),
                    num_retries=0,
                )
                model = InfiniLMTextModel(
                    **{k: v for k, v in config["model"].items() if k != "model_kwargs"},
                    model_name=binding.model_name,
                    model_kwargs=kwargs,
                )
                bind_delivery = getattr(request.transport_adapter, "bind_model", None)

                def task_expired():
                    raise CoreDeadline("shared task deadline exhausted,including upstream retries")

                delivery = (
                    bind_delivery(model, task_expired=task_expired)
                    if callable(bind_delivery)
                    else nullcontext()
                )
                with delivery, request.execution_adapter.open(request, deadline) as environment:
                    agent = NativeAgent(
                        model,
                        environment,
                        **(
                            config["agent"]
                            | dict(
                                step_limit=request.budget.max_model_calls,
                                wall_time_limit_seconds=max(1, int(request.budget.wall_seconds)),
                                cost_limit=0,
                                output_path=trajectory,
                            )
                        ),
                    )
                    # The acceptance context exposes the caller original public
                    # command, not private tests or a model-defined pass signal.
                    with request.acceptance_adapter.public_command(request, deadline) as command:
                        public_operation = None
                        if isinstance(command, PublicTaskContext):
                            public_operation, command = command.operation, command.binding
                        if isinstance(command, NativeTaskBinding):
                            if command.task != request.spec.task:
                                raise UnsupportedCore(
                                    "verbatim binding changed the public TaskSpec task"
                                )
                            task, mode, identity = command.task, "caller_verbatim", command.identity
                        elif isinstance(command, str) and command.strip():
                            # Preserve the frozen139-143 public packet exactly.
                            task = (
                                request.spec.task
                                + "\nWorking directory: /workspace. Independent acceptance command:\n"
                                + command
                            )
                            mode, identity = (
                                "legacy_public_command",
                                request.acceptance_adapter.identity,
                            )
                        else:
                            raise UnsupportedCore(
                                "public task context must yield a command or NativeTaskBinding"
                            )
                        describe_scope = (
                            scope_adapter.public_contract if scope_adapter is not None else None
                        )
                        if callable(describe_scope):
                            task += "\nCaller task contract:\n" + json.dumps(
                                describe_scope(request), ensure_ascii=False
                            )
                        (state / "native-public-task.json").write_text(
                            json.dumps(
                                dict(
                                    mode=mode,
                                    identity=identity,
                                    task=task,
                                    task_sha256=hashlib.sha256(task.encode()).hexdigest(),
                                    spec_task_sha256=hashlib.sha256(
                                        request.spec.task.encode()
                                    ).hexdigest(),
                                    final_acceptance="Independent verifier still required;public task binding is not resolution evidence.",
                                ),
                                ensure_ascii=False,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                        from infinicoder.memory.native_single_history import (
                            NativeSinglePhaseHistory,
                        )

                        if type(request.context_adapter) is NativeSinglePhaseHistory:
                            memory = request.context_adapter.bind_for_task(
                                request, model, environment, deadline, public_task=task
                            )
                        else:
                            memory = (
                                request.context_adapter.bind(request, model, environment, deadline)
                                if request.context_adapter is not None
                                else nullcontext()
                            )
                        operation = (
                            public_operation.bind(request, deadline)
                            if public_operation is not None
                            else nullcontext(None)
                        )
                        original_query = model.query
                        bind_task = getattr(request.context_adapter, "bind_public_task", None)
                        task_binding = bind_task(task) if callable(bind_task) else nullcontext()
                        with task_binding, memory, operation as bound_public_operation:
                            bind_public = getattr(
                                request.context_adapter, "bind_public_operation", None
                            )
                            observation = (
                                bind_public(bound_public_operation)
                                if callable(bind_public)
                                else nullcontext()
                            )
                            if request.planning_adapter is not None:
                                request.planning_adapter.prepare_plan(
                                    request,
                                    model,
                                    environment,
                                    query=original_query,
                                    public_task=task,
                                    initial_messages=lambda: _initial_messages(agent, task),
                                )
                                agent.config.step_limit = (
                                    request.budget.max_model_calls
                                    - request.planning_adapter.planning_query_calls
                                )
                            bind_agent = getattr(request.context_adapter, "bind_sdk_agent", None)
                            sdk_context = (
                                bind_agent(agent) if callable(bind_agent) else nullcontext()
                            )
                            bind_auxiliary = getattr(
                                request.context_adapter, "bind_auxiliary_calls", None
                            )
                            if callable(bind_auxiliary):
                                auxiliary = AuxiliaryCallBudget(
                                    request,
                                    agent,
                                    original_query,
                                    deadline,
                                    raw_query=model._query,
                                    prepare=model._prepare_messages_for_api,
                                    public_task=task,
                                )
                            auxiliary_context = (
                                bind_auxiliary(auxiliary)
                                if callable(bind_auxiliary)
                                else nullcontext()
                            )
                            bind_scope = (
                                scope_adapter.bind_model if scope_adapter is not None else None
                            )
                            scope_queries = (
                                bind_scope(request, model)
                                if callable(bind_scope)
                                else nullcontext()
                            )
                            with (
                                observation,
                                sdk_context,
                                auxiliary_context,
                                scope_queries,
                                _deadline_binding(model, environment, deadline),
                            ):
                                from infinicoder.domains.operator.native_implementation_provider import (
                                    NativeImplementationProvider,
                                )
                                from infinicoder.collaboration.implementation_handoff import (
                                    prepare_specialist_handoff,
                                    consume_specialist_handoff,
                                )

                                receipt = None
                                if type(
                                    request.context_adapter
                                ) is NativeImplementationProvider and getattr(
                                    request.context_adapter.handoff_policy, "enabled", False
                                ):
                                    receipt = prepare_specialist_handoff(
                                        request.context_adapter, _initial_messages(agent, task)
                                    )
                                if receipt is not None and consume_specialist_handoff(
                                    request.context_adapter, receipt
                                ):
                                    # No agent query took place and no assistant
                                    # Submit was fabricated. Final is unchanged.
                                    agent.messages = _initial_messages(agent, task)
                                    result = dict(
                                        exit_status="SpecialistHandoff",
                                        submission="",
                                        specialist_handoff_ref=str(
                                            state / "native-specialist-handoff.json"
                                        ),
                                        soft_aspirations_not_established=receipt.soft_aspirations_not_established,
                                    )
                                else:
                                    result = agent.run(task)
        except CoreDeadline as exc:
            error = f"{type(exc).__name__}: {exc}"
            result = dict(exit_status="TaskDeadline", submission="")
        except KeyboardInterrupt as exc:
            interrupted = True
            error = f"{type(exc).__name__}: {exc}"
            result = dict(exit_status="Interrupted", submission="")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            result = dict(exit_status=type(exc).__name__, submission="")
        finally:
            # agent saves in its own finally; also preserve an interrupted state
            # that has no ordinary exit message. Never run another model step.
            if agent is not None:
                agent.save(trajectory)
        if not interrupted:
            try:
                verify_scope = scope_adapter.verify_scope if scope_adapter is not None else None
                if callable(verify_scope):
                    verify_scope(request)
                final = request.acceptance_adapter.verify(request)
                if not isinstance(final, FinalAcceptance):
                    raise TypeError("independent verifier must return FinalAcceptance")
                if callable(verify_scope):
                    verify_scope(request)
            except Exception as exc:
                final = None
                acceptance_error = f"{type(exc).__name__}: {exc}"
        if scope_adapter is not None:
            result = dict(result, execution_profile="native_scoped_v1")
        planning_calls = (
            request.planning_adapter.planning_query_calls
            if request.planning_adapter is not None
            else 0
        )
        auxiliary_calls = auxiliary.call_units if auxiliary is not None else 0
        if auxiliary is not None:
            result = dict(
                result,
                auxiliary_model_calls=auxiliary_calls,
                main_sdk_agent_calls=agent.n_calls,
                auxiliary_usage_authority="Original shared physical transport ledger;agent step budget reduced by auxiliary units",
            )
        if request.planning_adapter is not None:
            result = dict(
                result,
                planning_query_calls=planning_calls,
                main_sdk_agent_calls=agent.n_calls if agent is not None else 0,
                source_planning_report_ref=str(
                    state
                    / getattr(
                        request.planning_adapter, "report_filename", "native-source-plan.json"
                    )
                ),
                physical_request_costs="Original shared transport usage ledger",
            )
        outcome = CoreOutcome(
            backend=NATIVE_CORE,
            exit_reason=result.get("exit_status", "Unknown"),
            candidate_ref=str(Path(request.workspace).resolve()),
            native_trajectory_ref=str(trajectory),
            usage_ledger_ref=ledger_ref,
            native_result=result,
            final_acceptance=final,
            agent_calls=(agent.n_calls if agent is not None else 0)
            + planning_calls
            + auxiliary_calls,
            error=error,
            acceptance_error=acceptance_error,
        )
        marker.write_text(
            json.dumps(asdict(outcome), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return outcome
