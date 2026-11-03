"""Task-oriented selection over existing execution cores, without another agent loop."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

from infinicoder.runtime.agent_execution import CoreBudget, UnsupportedCore
from infinicoder.models.budget import ModelProfile
from infinicoder.routing.profiles import route_task
from infinicoder.planning.task_contract import validate_task_spec


@dataclass(frozen=True)
class SolvePolicy:
    backend: str = "runtime"
    domain: str = "auto"
    context: str = "auto"
    planning: str = "auto"
    enhancements: bool = True

    def __post_init__(self):
        choices = dict(
            backend=("runtime", "native"),
            domain=("auto", "general", "operator", "framework"),
            context=("auto", "source", "task", "off"),
            planning=("auto", "on", "off"),
        )
        for name, values in choices.items():
            if getattr(self, name) not in values:
                raise ValueError("unsupported " + name)
        if type(self.enhancements) is not bool:
            raise ValueError("enhancements must be boolean")
        if self.backend == "runtime" and (
            self.context == "task" or self.planning != "auto" or not self.enhancements
        ):
            raise UnsupportedCore("Runtime uses source memory and its existing planning contract")
        if not self.enhancements and (self.context != "auto" or self.planning != "auto"):
            raise ValueError("baseline cannot silently ignore explicit context or planning")


MODULES = {
    "source_memory": "Versioned source lookup and hierarchical working set",
    "task_memory": "Pressure-triggered exact action and feedback retention",
    "domain_workflow": "Operator or framework development guidance",
    "planning": "Dependency-aware implementation plan",
    "implementation_worker": "Existing file/function implementation mode",
    "operator_compiler": "Restricted source-derived operator proposal",
    "public_feedback": "Caller-bound public checks available during coding",
    "final_acceptance": "Independent caller-bound final acceptance",
    "performance": "Externally measured optimization (advanced explicit contract)",
    "fusion_expert": "Experimental linear/GELU expert",
    "alternative_branch": "Experimental alternative implementation hypothesis",
    "collaboration": "Explicit multi-agent coordination",
    "experience_bank": "Caller-owned cross-task verified experience",
}


def inspect_task(spec, workspace, *, policy=None, source_paths=(), plan_paths=(), budget=None):
    """Pure selection: no model, tokenizer, checker, write, or task execution."""
    policy = policy or SolvePolicy()
    if not isinstance(policy, SolvePolicy):
        raise TypeError("SolvePolicy required")
    validate_task_spec(spec)
    budget = budget or CoreBudget(30, 600, 180)
    route = route_task(spec, workspace, policy.domain)
    if policy.backend == "native" and any(
        (
            spec.checks,
            spec.milestones,
            spec.amendments,
            spec.capabilities,
            spec.target_facts,
            spec.tensor_layouts,
        )
    ):
        raise UnsupportedCore("Native cannot preserve these TaskSpec fields; select Runtime")
    scoped = bool(spec.allowed_paths)
    if policy.backend == "native" and any(not p.endswith("/") for p in spec.allowed_paths):
        raise UnsupportedCore(
            "Native scopes require existing directory prefixes; exact-file contracts use Runtime"
        )
    if (
        policy.backend == "native"
        and scoped
        and (policy.context == "source" or policy.planning == "on")
    ):
        raise UnsupportedCore(
            "scoped Native permits plain/task memory; use Runtime for source planning"
        )
    if policy.backend == "native" and policy.context == "task" and policy.planning == "on":
        raise UnsupportedCore(
            "task memory and source planning use mutually exclusive context owners"
        )
    native_profile = None
    if policy.backend == "native":
        native_profile = (
            ("cpu-scoped" if scoped else "cpu-plain")
            if scoped or policy.context == "task" or not policy.enhancements
            else "cpu-auto"
        )
    selected = dict(
        source_memory=policy.context != "off" if policy.backend == "runtime" else False,
        task_memory=policy.backend == "native" and policy.context == "task",
        domain_workflow=route.profile in ("operator", "framework")
        and (policy.backend == "runtime" or native_profile == "cpu-auto"),
        planning=route.needs_plan if policy.backend == "runtime" else False,
        implementation_worker=policy.backend == "runtime" and route.source_path is not None,
        operator_compiler=policy.backend == "runtime" and route.profile == "operator",
    )
    # Use the actual native selector, rather than approximating a second router.
    native_selection = None
    if native_profile == "cpu-auto":
        from types import SimpleNamespace
        from infinicoder.runtime.extensions.native_capability_pipeline import (
            CapabilityPipelinePolicy,
            capability_selection,
        )

        native_selection = capability_selection(
            SimpleNamespace(spec=spec, workspace=workspace, budget=budget),
            tuple(source_paths),
            CapabilityPipelinePolicy(
                profile=policy.domain,
                memory="on" if policy.context == "source" else policy.context,
                planning=policy.planning,
            ),
            artifact_write_paths=tuple(plan_paths),
        )
        for key, native_key in (
            ("source_memory", "memory"),
            ("planning", "planning"),
            ("domain_workflow", "domain_workflow"),
        ):
            selected[key] = native_selection["modules"][native_key]["active"]
    modules = {}
    for name, description in MODULES.items():
        reason = (
            "selected_by_task_and_contract"
            if selected.get(name)
            else "not_selected_for_this_contract"
        )
        if name in (
            "performance",
            "fusion_expert",
            "alternative_branch",
            "collaboration",
            "experience_bank",
        ):
            reason = "advanced_explicit_contract;not_part_of_standard_recipe"
        if native_profile in ("cpu-plain", "cpu-scoped") and name in (
            "source_memory",
            "domain_workflow",
            "planning",
        ):
            reason = "baseline_or_plain_context_owner;no_silent_composition"
        modules[name] = dict(
            description=description,
            selected=selected.get(name, False),
            reason=reason,
            activated=None,
            delivered=None,
            outcome="not_run",
        )
    text = spec.task.lower()
    operation = (
        "optimize"
        if any(w in text for w in ("optimiz", "优化", "speedup"))
        else ("repair" if any(w in text for w in ("fix", "debug", "bug", "修复")) else "implement")
    )
    return dict(
        schema=1,
        policy=asdict(policy),
        domain=route.profile,
        operation=operation,
        reasons=list(route.reasons),
        native_profile=native_profile,
        modules=modules,
        native_selection=native_selection,
        task_sha256=hashlib.sha256(json.dumps(asdict(spec), sort_keys=True).encode()).hexdigest(),
        source_paths=list(source_paths),
        plan_paths=list(plan_paths),
        budget=asdict(budget),
        status="selected",
        delivered_evidence="unknown unless backed by a physical request receipt",
    )


def _write_report(state_dir, report):
    path = Path(state_dir) / "capabilities.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".next")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _finish_report(state_dir, report):
    # Diagnostic persistence must not replace cancellation or the original failure.
    try:
        _write_report(state_dir, report)
    except OSError:
        pass


def solve(
    spec,
    workspace,
    state_dir,
    *,
    model_config,
    policy=None,
    budget=None,
    model_profile=None,
    tokenizer_path=None,
    python_environment=None,
    source_paths=(),
    implementation_paths=(),
    plan_paths=(),
    check_script=None,
    public_check_script=None,
    allow_commands=False,
    resume=False,
    client=None,
    estimator=None,
    event_handler=None,
):
    """Run one existing core and persist a common capability selection/activation receipt.

    Native paths and scripts are explicit authority. Runtime preserves the complete
    TaskSpec and its existing session contract. No profile silently widens permissions.
    """
    from infinicoder import __version__

    policy = policy or SolvePolicy()
    budget = budget or CoreBudget(30, 600, 180)
    model_profile = model_profile or ModelProfile(16384, model_config.max_output_tokens)
    if model_profile.output_reserve != model_config.max_output_tokens:
        raise ValueError("model output tokens must equal the context output reserve")
    if client is not None and client.config != model_config:
        raise ValueError("supplied client configuration differs from model_config")
    workspace, state_dir = Path(workspace).resolve(), Path(state_dir).resolve()
    if (
        not workspace.is_dir()
        or state_dir.is_relative_to(workspace)
        or workspace.is_relative_to(state_dir)
    ):
        raise ValueError("existing workspace and separate state directory required")
    if (
        policy.backend == "runtime"
        and not resume
        and any((state_dir / name).exists() for name in ("state.json", "capabilities.json"))
    ):
        raise ValueError("existing Runtime session; use resume with the original configuration")
    if policy.backend == "runtime" and (source_paths or implementation_paths):
        raise UnsupportedCore(
            "Runtime uses TaskSpec scope and its index; Native source path declarations are not consumed"
        )
    watched = tuple(dict.fromkeys(implementation_paths))
    readable = tuple(dict.fromkeys((*watched, *source_paths)))
    report = inspect_task(
        spec, workspace, policy=policy, source_paths=readable, plan_paths=plan_paths, budget=budget
    )
    report.update(version=__version__, model=model_config.model, context=asdict(model_profile))
    modules = report["modules"]
    started = time.monotonic()
    started_at = time.time()
    if policy.backend == "native":
        if (state_dir / "native-run.json").exists() or (state_dir / "capabilities.json").exists():
            raise ValueError("Native requires a fresh state directory; preserve the existing run")
        if resume or client is not None or estimator is not None or allow_commands:
            raise UnsupportedCore(
                "Native is one-shot and uses its own sandbox, transport and tokenizer"
            )
        if not tokenizer_path or not check_script:
            raise ValueError("Native requires tokenizer_path and an independent check_script")
        if policy.context == "task" and not watched:
            raise ValueError("task memory requires explicit implementation_paths")
        from infinicoder.app.native_api import create_cpu_request, run_native
        from infinicoder.runtime.extensions.native_capability_pipeline import (
            CapabilityPipelinePolicy,
        )

        options = dict(
            model_config=model_config,
            tokenizer_path=tokenizer_path,
            python_environment=python_environment,
            check_script=check_script,
            profile=report["native_profile"],
            model_profile=model_profile,
            budget=budget,
            implementation_paths=watched,
            public_check_script=public_check_script,
            public_check_paths=watched if public_check_script else (),
            public_check_coverage={"description": "caller public examples"}
            if public_check_script
            else None,
            public_check_feedback="reports" if public_check_script else "direct",
            task_working_memory=policy.context == "task",
        )
        if report["native_profile"] == "cpu-auto":
            options.update(
                source_read_paths=readable,
                architecture_write_paths=tuple(plan_paths),
                pipeline_policy=CapabilityPipelinePolicy(
                    profile=policy.domain,
                    memory="on" if policy.context == "source" else policy.context,
                    planning=policy.planning,
                ),
            )
        request = create_cpu_request(spec, workspace, state_dir, **options)
        execution = getattr(request.execution_adapter, "execution", request.execution_adapter)
        runtime_path = execution.runtime
        report["execution_environment"] = dict(
            python_environment=str(runtime_path),
            identity=request.execution_adapter.identity,
            pyvenv_sha256=hashlib.sha256((runtime_path / "pyvenv.cfg").read_bytes()).hexdigest()
            if (runtime_path / "pyvenv.cfg").is_file()
            else None,
        )
        modules["final_acceptance"].update(selected=True, reason="independent_check_script")
        modules["public_feedback"].update(
            selected=public_check_script is not None,
            reason="caller_public_check" if public_check_script else "not_supplied",
        )
        # Report only after complete contract validation; invalid calls must not create state.
        from infinicoder.runtime.agent_execution import validate_native_request

        validate_native_request(request)
        _write_report(state_dir, report)
        try:
            outcome = run_native(request)
            report.update(
                status="completed",
                success=outcome.success,
                exit_reason=outcome.exit_reason,
                usage_ledger_ref=outcome.usage_ledger_ref,
                agent_calls=outcome.agent_calls,
            )
            modules["final_acceptance"].update(
                activated=outcome.final_acceptance is not None,
                outcome="passed" if outcome.success else "failed_or_not_run",
            )
            observer = request.context_adapter
            for name, records in (
                ("source_memory", getattr(observer, "records", ())),
                ("task_memory", getattr(getattr(observer, "working_memory", None), "records", ())),
            ):
                if modules[name]["selected"]:
                    modules[name].update(
                        activated=bool(records),
                        outcome="context_assembler_executed" if records else "not_triggered",
                        compressed=any(
                            bool(r.get("archived") or r.get("omitted_events")) for r in records
                        ),
                        retrieval_used=None,
                    )
            if modules["planning"]["selected"]:
                modules["planning"].update(
                    activated=bool(getattr(observer, "planning_query_calls", 0))
                )
            if modules["domain_workflow"]["selected"]:
                modules["domain_workflow"].update(
                    activated=any(
                        r.get("kind") == "domain_workflow" and r.get("prepared")
                        for r in getattr(observer, "capability_records", ())
                    ),
                    outcome="projection_only;physical_delivery_not_inferred",
                )
            return outcome
        except BaseException as error:
            report.update(
                status="interrupted" if isinstance(error, KeyboardInterrupt) else "error",
                error_type=type(error).__name__,
            )
            raise
        finally:
            report["wall_seconds"] = time.monotonic() - started
            _finish_report(state_dir, report)
    if plan_paths:
        raise UnsupportedCore(
            "Runtime preserves TaskSpec planning; Native plan_paths are not consumed"
        )
    script_checks = None
    if check_script or public_check_script:
        if not (check_script and public_check_script):
            raise ValueError("Runtime script recipe requires public feedback and independent final")
        if spec.checks or resume or allow_commands:
            raise UnsupportedCore(
                "Runtime script recipe requires a fresh task without TaskSpec.checks or model commands"
            )
        from infinicoder.verification.runtime_checks import RuntimeScriptChecks

        script_checks = RuntimeScriptChecks(
            spec,
            workspace,
            state_dir,
            public_check_script,
            check_script,
            python_environment,
            started + budget.wall_seconds,
        )
    elif python_environment:
        raise UnsupportedCore("Runtime python_environment requires the explicit script recipe")
    if tokenizer_path:
        if estimator is not None:
            raise ValueError("supply tokenizer_path or estimator, not both")
        from transformers import AutoTokenizer
        from infinicoder.models.budget import TokenEstimator

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True, trust_remote_code=False
        )
        count = lambda text: len(tokenizer.encode(text, add_special_tokens=False))

        def count_request(messages, schema, output):
            framing = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **model_config.extra_body.get("chat_template_kwargs", {}),
            )
            return count(framing) + (count(json.dumps(schema)) if schema else 0) + output + 256

        estimator = TokenEstimator(count, count_request)
    from infinicoder.app.config import RuntimeConfig, create_runtime
    from infinicoder.models.model_client import CompatibleClient

    events = []

    def on_event(event):
        events.append(event["kind"])
        if event_handler:
            event_handler(event)

    runtime = create_runtime(
        workspace,
        client or CompatibleClient(model_config),
        state_dir,
        config=RuntimeConfig(
            profile=policy.domain,
            context_limit=model_profile.context_limit,
            max_steps=budget.max_model_calls,
            max_model_calls=budget.max_model_calls,
            time_limit=budget.wall_seconds,
            request_timeout=budget.request_timeout,
            allow_commands=allow_commands,
            code_memory=policy.context != "off",
        ),
        estimator=estimator,
        event_handler=on_event,
        verifier=script_checks.verify_public if script_checks else None,
    )
    modules["final_acceptance"].update(
        selected=bool(spec.checks or script_checks),
        reason="independent_check_script" if script_checks else "TaskSpec.checks",
    )
    modules["public_feedback"].update(selected=script_checks is not None)
    completed = False
    try:
        state = runtime.execute(spec, resume=resume, started_at=started_at)
        if script_checks:
            control = runtime.store.take_control()
            if control:
                state.update(
                    status="paused" if control == "pause" else "cancelled", reason="user_" + control
                )
                runtime.store.save(state)
            final = (
                script_checks.finish()
                if state.get("status") in ("succeeded", "failed", "budget_exhausted")
                else None
            )
            state = dict(
                state, success=bool(final and final["passed"]), independent_acceptance=final
            )
            report["execution_environment"] = script_checks.identity()
            report["independent_acceptance"] = final
        completed = True
        report.update(
            status="completed",
            success=state.get("success", state.get("status") == "succeeded"),
            exit_reason=state.get("reason"),
            model_calls=state.get("model_calls"),
            budget_report=state.get("budget_report"),
        )
        for name, active in dict(
            source_memory=bool(state.get("code_memory_report")),
            domain_workflow=bool(state.get("development_workflow")),
            planning="plan" in events,
            operator_compiler="domain_proposal" in events,
            implementation_worker=None,
            public_feedback=bool(script_checks.records) if script_checks else False,
            final_acceptance=(state.get("independent_acceptance") is not None)
            if script_checks
            else True
            if state.get("status") == "succeeded"
            else None,
        ).items():
            modules[name].update(
                activated=active,
                outcome="observed" if active else "unknown" if active is None else "not_observed",
            )
        if script_checks:
            modules["final_acceptance"]["outcome"] = (
                "not_run"
                if state["independent_acceptance"] is None
                else "passed"
                if state["success"]
                else "failed"
            )
        return state
    except BaseException as error:
        report.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "error",
            error_type=type(error).__name__,
        )
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        if completed or not resume:
            _finish_report(state_dir, report)
