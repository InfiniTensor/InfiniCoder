"""Run two bounded repair phases with shared transport and independent final acceptance."""

from dataclasses import asdict, replace
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
    NATIVE_CORE,
    UnsupportedCore,
)
from infinicoder.runtime.native_run import CoreDeadline, _deadline_binding, _initial_messages
from infinicoder.runtime.extensions.native_repair import RepairHandoff, RepairStagnation


def run_with_repair(request):
    if sys.platform != "linux" or threading.current_thread() is not threading.main_thread():
        raise UnsupportedCore(
            "native core requires a Linux main-thread process for strict deadlines"
        )
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise UnsupportedCore("native core cannot replace an existing wall alarm")
    state = Path(request.state_dir).resolve()
    state.mkdir(parents=True, exist_ok=True)
    marker = state / "native-run.json"
    with marker.open("x", encoding="utf-8") as stream:
        json.dump(
            dict(
                status="started", backend=NATIVE_CORE, continuation=request.repair_adapter.identity
            ),
            stream,
        )
    deadline = time.monotonic() + request.budget.wall_seconds
    from infinicoder.runtime.native_agent import NativeAgent
    from infinicoder.models.infinilm_client import InfiniLMTextModel, default_config

    config = default_config()
    memory, repair = request.context_adapter, request.repair_adapter
    manifest = state / "native-phases.json"
    phases = []
    final = None
    error = acceptance_error = None
    result = {}
    total_calls = 0
    ledger_ref = ""
    trajectory = state / "ordinary/native-trajectory.json"
    interrupted = False
    try:
        # Open each original caller context exactly once for the whole task.
        with request.transport_adapter.open(request, deadline) as binding:
            if not isinstance(binding, ModelBinding) or not binding.usage_ledger_ref:
                raise ValueError("transport must yield the original ModelBinding and usage ledger")
            ledger_ref = binding.usage_ledger_ref
            with request.execution_adapter.open(request, deadline) as environment:
                with request.acceptance_adapter.public_command(request, deadline) as command:
                    if isinstance(command, NativeTaskBinding):
                        if command.task != request.spec.task:
                            raise UnsupportedCore(
                                "verbatim binding changed the public TaskSpec task"
                            )
                        public_task, mode, identity = (
                            command.task,
                            "caller_verbatim",
                            command.identity,
                        )
                    elif isinstance(command, str) and command.strip():
                        public_task = (
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
                    (state / "native-public-task.json").write_text(
                        json.dumps(
                            dict(
                                mode=mode,
                                identity=identity,
                                task=public_task,
                                task_sha256=hashlib.sha256(public_task.encode()).hexdigest(),
                                spec_task_sha256=hashlib.sha256(
                                    request.spec.task.encode()
                                ).hexdigest(),
                                final_acceptance="Independent verifier required;this binding proves no result.",
                            ),
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    task = public_task
                    for phase in ("ordinary", "repair"):
                        remaining_calls = request.budget.max_model_calls - total_calls
                        remaining_wall = deadline - time.monotonic()
                        if remaining_calls <= 0 or remaining_wall <= 0:
                            result = dict(
                                exit_status="LimitsExceeded"
                                if remaining_calls <= 0
                                else "TaskDeadline",
                                submission="",
                            )
                            break
                        phase_state = state / phase
                        phase_state.mkdir(exist_ok=False)
                        phase_request = replace(request, state_dir=phase_state)
                        trajectory = phase_state / "native-trajectory.json"
                        kwargs = (
                            dict(config["model"].get("model_kwargs", {})) | binding.model_kwargs
                        )
                        kwargs.update(
                            max_tokens=request.model_profile.output_reserve,
                            timeout=min(request.budget.request_timeout, remaining_wall),
                            num_retries=0,
                        )
                        model = InfiniLMTextModel(
                            **{
                                k: v
                                for k, v in config["model"].items()
                                if k not in ("model_kwargs", "model_name")
                            },
                            model_name=binding.model_name,
                            model_kwargs=kwargs,
                        )
                        agent = NativeAgent(
                            model,
                            environment,
                            **(
                                config["agent"]
                                | dict(
                                    step_limit=remaining_calls,
                                    wall_time_limit_seconds=max(1, int(remaining_wall)),
                                    cost_limit=0,
                                    output_path=trajectory,
                                )
                            ),
                        )

                        def preview(repair_task):
                            left = request.budget.max_model_calls - total_calls - agent.n_calls
                            if left <= 0 or deadline <= time.monotonic():
                                return False, dict(
                                    status="not_applicable;no_original_budget_remaining"
                                )
                            prepared = _initial_messages(agent, repair_task)
                            tokens = memory._count(prepared)
                            return tokens <= memory.limit, dict(
                                input_tokens=tokens,
                                input_limit=memory.limit,
                                remaining_agent_calls=left,
                                status="fits"
                                if tokens <= memory.limit
                                else "not_applicable;complete_repair_contract_does_not_fit",
                            )

                        handed_off = False
                        phase_result = {}
                        started = time.monotonic()
                        try:
                            with repair.bind_query(
                                phase_request,
                                model,
                                memory,
                                phase=phase,
                                public_task=public_task,
                                preview=preview,
                            ):
                                with memory.bind(phase_request, model, environment, deadline):
                                    with _deadline_binding(model, environment, deadline):
                                        phase_result = agent.run(task)
                        except RepairHandoff as transfer:
                            if phase != "ordinary":
                                raise RuntimeError("a repair phase cannot hand off again")
                            # All hooks and the outer SIGALRM have unwound.
                            if signal.getitimer(signal.ITIMER_REAL)[0]:
                                raise RuntimeError("parent alarm survived repair handoff")
                            handed_off = True
                            task = transfer.task
                            phase_result = dict(
                                exit_status="NativeRepairHandoff",
                                submission="",
                                seed_ref=str(state / "native-repair-seed.json"),
                                no_HTTP_for_handoff_query=True,
                            )
                        except RepairStagnation as exc:
                            phase_result = dict(
                                exit_status="NativeRepairStagnation", submission="", reason=str(exc)
                            )
                        except BaseException as exc:
                            phase_result = dict(exit_status=type(exc).__name__, submission="")
                            raise
                        finally:
                            agent.save(trajectory)
                            total_calls += agent.n_calls
                            phases.append(
                                dict(
                                    phase=phase,
                                    sdk_agent_calls=agent.n_calls,
                                    sdk_step_limit=remaining_calls,
                                    shared_deadline=deadline,
                                    trajectory=str(trajectory),
                                    state_dir=str(phase_state),
                                    result=phase_result,
                                    wall_seconds=time.monotonic() - started,
                                    usage_ledger_ref=ledger_ref,
                                    physical_requests="read_original_shared_ledger;not_agent_calls",
                                )
                            )
                            manifest.write_text(
                                json.dumps(
                                    dict(
                                        phases=phases,
                                        total_agent_calls=total_calls,
                                        max_agent_calls=request.budget.max_model_calls,
                                        shared_deadline=deadline,
                                        conservative_handoff_call_charge=True,
                                        usage_ledger_ref=ledger_ref,
                                    ),
                                    ensure_ascii=False,
                                    indent=2,
                                ),
                                encoding="utf-8",
                            )
                        result = phase_result
                        if not handed_off:
                            break
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
    if not interrupted:
        try:
            final = request.acceptance_adapter.verify(request)
            if not isinstance(final, FinalAcceptance):
                raise TypeError("independent verifier must return FinalAcceptance")
        except Exception as exc:
            final = None
            acceptance_error = f"{type(exc).__name__}: {exc}"
    result = dict(
        result,
        phase_manifest_ref=str(manifest),
        repair_handoff_requested=repair.used,
        repair_activated=any(p["phase"] == "repair" for p in phases),
        phase_count=len(phases),
        handoff_calls_are_not_HTTP=True,
    )
    outcome = CoreOutcome(
        NATIVE_CORE,
        result.get("exit_status", "Unknown"),
        str(Path(request.workspace).resolve()),
        str(trajectory),
        ledger_ref,
        result,
        final,
        total_calls,
        error,
        acceptance_error,
    )
    marker.write_text(json.dumps(asdict(outcome), ensure_ascii=False, indent=2), encoding="utf-8")
    return outcome
