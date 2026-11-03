"""Caller-authorized contract revisions; unavailable to model tool actions."""

from dataclasses import asdict
import hashlib, json, shutil, time
from infinicoder.planning.task_contract import TaskSpec
from infinicoder.analysis.task_compiler import compile_task
from infinicoder.verification.check_identity import checker_files
from infinicoder.execution.workspace_tools import Workspace
from infinicoder.routing.profiles import route_task


def revise_contract(runtime, spec, reason):
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("a caller-supplied revision reason is required")
    spec = TaskSpec.from_dict(spec if isinstance(spec, dict) else asdict(spec))
    compile_task(spec)
    if (
        route_task(spec, runtime.workspace, runtime.profile).profile == "operator"
        and "operator" not in spec.capabilities
    ):
        spec.capabilities.append("operator")
    ws = Workspace(runtime.workspace, spec.allowed_paths)
    if runtime._explicit_worker_path:
        ws.path(runtime.worker_path, write=True)
    if runtime.optimization and (
        spec.milestones or set(spec.allowed_paths) != set(runtime.optimization.paths)
    ):
        raise ValueError("new contract differs from the supplied optimization scope")
    document = asdict(spec)
    contract_document = dict(document)
    contract_document.pop("amendments", None)
    contract = hashlib.sha256(json.dumps(contract_document, sort_keys=True).encode()).hexdigest()
    if not runtime._execution_lock.acquire(blocking=False):
        raise RuntimeError("Runtime is active;pause it before revising the contract")
    try:
        runtime.store.acquire_lock()
        old = runtime.store.load()
        if old is None:
            raise ValueError("no existing session to revise")
        if old["workspace"] != str(runtime.workspace):
            raise ValueError("revision workspace differs")
        if old.get("pending_action"):
            raise ValueError(
                "an interrupted action has an unknown outcome;inspect it before changing the contract"
            )
        # A requirement change invalidates task evidence, not the policy and
        # budget under which that evidence was collected. Reject mismatches
        # before creating a backup or replacing the active session.
        from infinicoder.app.config import validate_session_configuration
        from infinicoder.memory.source.source_handoff import turn_identity

        validate_session_configuration(runtime, old, resume=True)
        expected = {
            "execution_history_policy": turn_identity(),
            "agent_context_policy": runtime._agent_context_policy(),
            "qualified_worker_policy": runtime.qualified_worker,
            "native_worker_policy": runtime.native_worker,
        }
        for name, value in expected.items():
            default = False if name.endswith("_worker_policy") else None
            if old.get(name, default) != value:
                raise ValueError(
                    f"Session {name} changed; preserve the original policy before revision"
                )
        generation = getattr(runtime.client, "generation_policy_identity", None)
        saved_generation = old.get("generation_policy_identity")
        if saved_generation != generation and (
            saved_generation is not None
            or generation is not None
            and old.get("model_calls", old["steps"]) > 0
        ):
            raise ValueError(
                "Session generation profile changed; preserve the original policy before revision"
            )
        if contract == old["contract"]:
            raise ValueError("contract is unchanged;append a user amendment and resume instead")
        epoch = old.get("contract_epoch", 0) + 1
        backup = (
            runtime.state_dir
            / "revisions"
            / ("contract-" + str(epoch).zfill(4) + "-" + old["contract"][:12])
        )
        backup.mkdir(parents=True, exist_ok=False)
        (backup / "state.json").write_text(
            json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (backup / "new-contract.json").write_text(
            json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        events = runtime.state_dir / "events.jsonl"
        if events.exists():
            shutil.copyfile(events, backup / "events.jsonl")
        now = time.time()
        starts = [
            old[k] for k in ("pause_started_epoch", "finished_epoch") if old.get(k) is not None
        ]
        wait_start = min(starts) if starts else None
        waited = max(0, now - wait_start) if wait_start is not None else 0
        history = old.get("contract_history", []) + [
            dict(
                epoch=epoch - 1, contract=old["contract"], backup=str(backup), reason=reason.strip()
            )
        ]
        state = dict(
            contract=contract,
            contract_document=document,
            contract_epoch=epoch,
            contract_history=history,
            workspace=str(runtime.workspace),
            steps=old["steps"],
            model_calls=old.get("model_calls", old["steps"]),
            usage=old.get("usage", {}),
            output_bytes=old.get("output_bytes", 0),
            session_deadline_epoch=old["session_deadline_epoch"] + waited,
            excluded_user_wait_s=old.get("excluded_user_wait_s", 0) + waited,
            milestone=0,
            notes=[],
            recent=[],
            changed_files=[],
            evidence=[],
            failures={},
            amendments=0,
            amendment_log=[],
            checker_files=checker_files(spec, ws),
            status="paused",
            reason="caller_contract_revision",
            pause_started_epoch=now,
        )
        for name in (
            "configuration_identity",
            "execution_history_policy",
            "generation_policy_identity",
            "agent_context_policy",
            "qualified_worker_policy",
            "native_worker_policy",
        ):
            if name in old:
                state[name] = old[name]
        runtime.store.save(state)
        runtime.store.event(
            "contract_revised",
            dict(
                epoch=epoch,
                old_contract=old["contract"],
                new_contract=contract,
                reason=reason.strip(),
                backup=str(backup),
                cumulative_model_calls=state["model_calls"],
                cumulative_steps=state["steps"],
                excluded_idle_s=waited,
            ),
        )
        runtime.worker_path = runtime._configured_worker_path
        runtime.worker_symbol = runtime._configured_worker_symbol
        runtime.worker_cards = list(runtime._configured_worker_cards)
        runtime.auto_verify_edits = runtime._configured_auto_verify
        runtime.direct_first = runtime._configured_direct_first
        runtime._worker_scope = None
        runtime._optimization_session = None
        return dict(
            status="paused",
            contract_epoch=epoch,
            backup=str(backup),
            model_calls=state["model_calls"],
            note="Resume with the new caller-owned TaskSpec;budgets remain cumulative;diff remains relative to original session input",
        )
    finally:
        runtime.store.release_lock()
        runtime._execution_lock.release()
