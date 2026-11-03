"""One proposal then one original SDK run, on the original shared budget relay."""

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import signal
import sys
import threading
import time
from types import SimpleNamespace

from infinicoder.runtime.agent_execution import (
    CoreBudget,
    CoreOutcome,
    ModelBinding,
    NATIVE_CORE,
    UnsupportedCore,
    execute_agent,
    validate_native_request,
)
from infinicoder.runtime.implementation_proposal import ImplementationProposal, dump
from infinicoder.runtime.proposal_history import historical_event, project_history
from infinicoder.runtime.native_run import CoreDeadline, _deadline_binding
from infinicoder.execution.workspace_tools import Workspace


def run_implementation_proposal(request):
    validate_native_request(request)
    owner = request.implementation_proposal
    if not isinstance(owner, ImplementationProposal):
        raise UnsupportedCore("Explicit implementation proposal policy required")
    if (
        sys.platform != "linux"
        or threading.current_thread() is not threading.main_thread()
        or signal.getitimer(signal.ITIMER_REAL)[0]
    ):
        raise UnsupportedCore("Proposal requires Linux main thread without an existing alarm")
    started = time.monotonic()
    deadline = started + request.budget.wall_seconds
    ordinary = replace(request, implementation_proposal=None)
    state = Path(request.state_dir)
    state.mkdir(parents=True, exist_ok=True)
    evidence = state / "implementation-proposal.json"
    with evidence.open("x", encoding="utf8") as stream:
        stream.write("{}\n")
    record = dict(
        identity=owner.identity,
        source_namespace=owner.source_namespace,
        original_task_sha256=hashlib.sha256(request.spec.task.encode()).hexdigest(),
        original_budget=asdict(request.budget),
        automatic_retry=False,
        SDK_history=False,
        final_authority="original independent acceptance",
    )

    def save():
        record["wall_s"] = time.monotonic() - started
        evidence.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf8")

    preparation = owner.prepare(ordinary, deadline)
    record["preparation"] = asdict(preparation)
    save()
    if not preparation.applicable:
        record.update(status="N/A;ordinary input unchanged", proposal_call_units=0)
        save()
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("No remaining original task budget")
        return execute_agent(
            replace(ordinary, budget=replace(ordinary.budget, wall_seconds=left)),
            backend=NATIVE_CORE,
        )
    with ordinary.transport_adapter.open(ordinary, deadline) as binding:
        if not isinstance(binding, ModelBinding):
            raise UnsupportedCore("Original budgeted ModelBinding required")
        record["shared_usage_ledger_ref"] = binding.usage_ledger_ref
        unused = SimpleNamespace(
            execute=lambda *a, **k: None,
            config=SimpleNamespace(timeout=request.budget.request_timeout),
        )
        limiter = getattr(ordinary.transport_adapter, "limit_request", None)

        def query_once(current, name):
            query = SimpleNamespace(
                query=lambda: owner.propose(current, ordinary, binding, deadline, state / name)
            )
            limits = (
                limiter(
                    output_tokens=ordinary.model_profile.output_reserve,
                    absolute_deadline=deadline,
                    max_physical_calls=1,
                )
                if callable(limiter)
                else nullcontext()
            )
            try:
                with limits:
                    with _deadline_binding(query, unused, deadline):
                        return query.query()
            except (CoreDeadline, TimeoutError) as error:
                return dict(
                    finish_reason=None,
                    text=None,
                    error=type(error).__name__ + ": " + str(error),
                    proposal_call_units=1,
                )

        proposal = query_once(preparation, "implementation-proposal-response.json")
        # The alarm is fully released before starting the ordinary native core.
        if time.monotonic() >= deadline:
            proposal = dict(
                proposal, finish_reason=None, error="Proposal arrived after original deadline"
            )
        patch = owner.apply(
            preparation, proposal, Workspace(ordinary.workspace, ordinary.spec.allowed_paths)
        )
        used = 1
        record["initial_assembly"] = patch
        record["initial_response_text_sha256"] = hashlib.sha256(
            (proposal.get("text") or "").encode()
        ).hexdigest()
        if (
            owner.source_question_completion
            and patch["status"] == "need_context"
            and request.budget.max_model_calls >= 3
        ):
            try:
                continuation, answer = owner.answer_context_request(
                    preparation, proposal, ordinary, deadline
                )
                record["source_answer"] = answer
                if continuation is not None:
                    # Revalidate after tokenization and immediately before the
                    # second physical call. Never recursively answer a question.
                    if owner.current(continuation.plan) and time.monotonic() < deadline:
                        proposal = query_once(
                            continuation, "implementation-proposal-continuation-response.json"
                        )
                        used += 1
                        if time.monotonic() >= deadline:
                            proposal = dict(
                                proposal,
                                finish_reason=None,
                                error="Continuation arrived after original deadline",
                            )
                        patch = owner.apply(
                            continuation,
                            proposal,
                            Workspace(ordinary.workspace, ordinary.spec.allowed_paths),
                        )
                        record["continuation_preparation"] = asdict(continuation)
                    else:
                        record["source_answer"]["continuation_cancelled"] = (
                            "source_changed_or_deadline"
                        )
            except (
                CoreDeadline,
                TimeoutError,
                OSError,
                ValueError,
                KeyError,
                SyntaxError,
                RecursionError,
            ) as error:
                record["source_answer_failure"] = type(error).__name__ + ": " + str(error)[:300]
        record["historical_event"] = historical_event(preparation, proposal, patch)
        record["event_recorded_epoch_s"] = time.time()
        record.update(proposal_call_units=used, assembly=patch)
        left = deadline - time.monotonic()
        calls = request.budget.max_model_calls - used
        if left <= 0 or calls <= 0:
            final = ordinary.acceptance_adapter.verify(ordinary)
            record.update(status="proposal_budget_exhausted", final_acceptance=asdict(final))
            save()
            return CoreOutcome(
                NATIVE_CORE,
                "ProposalBudgetExhausted",
                str(ordinary.workspace),
                "",
                binding.usage_ledger_ref,
                dict(implementation_proposal_ref=str(evidence)),
                final,
                used,
            )

        class SharedTransport:
            identity = "same_open_original_proposal_budget_relay"

            def validate(self, current):
                if (
                    current.workspace != ordinary.workspace
                    or current.model_profile != ordinary.model_profile
                ):
                    raise UnsupportedCore("Shared proposal relay contract differs")

            @contextmanager
            def open(self, current, inner_deadline):
                yield binding

        base_context = ordinary.context_adapter

        class ProposalContext:
            identity = "one_controller_proposal_receipt;ordinary_history_unchanged"

            def validate(self, current):
                if base_context is not None:
                    base_context.validate(current)

            @contextmanager
            def bind_public_operation(self, operation):
                bind = getattr(base_context, "bind_public_operation", None)
                with bind(operation) if callable(bind) else nullcontext():
                    yield

            @contextmanager
            def bind(self, current, model, environment, inner_deadline):
                original = (
                    base_context.bind(current, model, environment, min(deadline, inner_deadline))
                    if base_context is not None
                    else nullcontext()
                )
                with original:
                    prepare = model._prepare_messages_for_api

                    def prepared(messages):
                        try:
                            current_hash = owner.sources.read(preparation.plan["target"]["path"])[2]
                        except (OSError, ValueError):
                            current_hash = None
                        result, currency, slot = project_history(
                            prepare(messages), record["historical_event"], current_hash
                        )
                        count = owner.counter(result, deadline - time.monotonic())
                        if count > current.model_profile.input_budget():
                            from infinicoder.memory.native_memory import MemoryCapacityError

                            raise MemoryCapacityError(
                                "Complete ordinary input plus proposal receipt exceeds original input budget"
                            )
                        record.setdefault("deliveries", []).append(
                            dict(
                                input_tokens=count,
                                messages_sha256=hashlib.sha256(dump(result).encode()).hexdigest(),
                                source_currency=currency,
                                historical_event_slot=slot,
                                event_id=record["historical_event"]["event_id"],
                                canonical_history_changed=False,
                            )
                        )
                        save()
                        return result

                    model._prepare_messages_for_api = prepared
                    try:
                        yield
                    finally:
                        model._prepare_messages_for_api = prepare

        current = replace(
            ordinary,
            budget=CoreBudget(calls, left, ordinary.budget.request_timeout),
            transport_adapter=SharedTransport(),
            context_adapter=ProposalContext(),
        )
        save()
        native = execute_agent(current, backend=NATIVE_CORE)
        record.update(
            status="single_proposal_then_original_native_complete",
            native_outcome=asdict(native),
            ordinary_agent_calls=native.agent_calls,
            total_call_units=used + native.agent_calls,
            cost_policy="One original raw physical ledger includes proposal and ordinary work;unknown usage remains unknown",
        )
        save()
        return replace(
            native,
            agent_calls=used + native.agent_calls,
            native_result=native.native_result
            | dict(
                implementation_proposal_ref=str(evidence),
                proposal_call_units=used,
                ordinary_agent_calls=native.agent_calls,
            ),
        )
