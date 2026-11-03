"""Explicit packaged Native entry;the existing infinicoder Runtime CLI is unchanged."""

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description="InfiniCoder one-shot Native CPU coding agent")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--state-dir",
        type=Path,
        required=True,
        help="fresh run state outside workspace;parent must exist",
    )
    tasks = parser.add_mutually_exclusive_group(required=True)
    tasks.add_argument("--task-file", type=Path, help="UTF-8 public task text")
    tasks.add_argument(
        "--spec-json",
        type=Path,
        help="complete TaskSpec;unsupported fields are rejected,never cleared",
    )
    parser.add_argument("--require-artifact", action="append", default=[])
    parser.add_argument(
        "--check-script",
        type=Path,
        required=True,
        help="trusted standalone Python final check outside workspace",
    )
    parser.add_argument("--base-url", required=True, help="caller-selected compatible endpoint")
    parser.add_argument("--model", required=True, help="raw serving model ID")
    parser.add_argument(
        "--tokenizer", type=Path, required=True, help="matching frozen local tokenizer directory"
    )
    parser.add_argument(
        "--api-key-env",
        help="optional existing provider credential environment variable;never saved",
    )
    parser.add_argument(
        "--profile",
        choices=(
            "cpu",
            "cpu-scoped",
            "cpu-trial",
            "cpu-anchored",
            "cpu-dependencies",
            "cpu-reselection",
            "cpu-plain",
            "cpu-auto",
            "cpu-workset",
            "cpu-architecture",
            "public-transition-probe",
            "implementation-proposal",
            "cpu-evidence",
            "cpu-planning",
            "cpu-task-context",
            "cpu-task-context-off",
            "cpu-reassessment",
            "cpu-reassessment-off",
            "cpu-candidate-search",
            "cpu-single-history",
        ),
        default="cpu",
        help="cpu-scoped:plain Native with OS-enforced existing directory scopes from --spec-json",
    )
    parser.add_argument(
        "--capability-policy",
        type=Path,
        help="cpu-auto:JSON semantic profile and memory/planning/diagnostic/knowledge switches",
    )
    parser.add_argument(
        "--task-working-memory",
        action="store_true",
        help="cpu-plain/cpu-scoped: activate reversible task memory only under context pressure",
    )
    parser.add_argument(
        "--fusion-provider-contract",
        type=Path,
        help="source-bound linear/GELU fusion contract; independently measured and verified",
    )
    parser.add_argument(
        "--fusion-schedule-workload", help="external workload identity for the two fusion schedules"
    )
    parser.add_argument(
        "--public-performance-contract",
        type=Path,
        help="cpu-auto:fixed external public workload and explicit rollback paths;requires public check and exclusive workspace",
    )
    parser.add_argument(
        "--source-attachment",
        type=Path,
        help="cpu-workset:exact serialized public FILES object present once in task",
    )
    parser.add_argument(
        "--architecture-write-path",
        action="append",
        default=[],
        help="cpu-architecture:explicit writable observed artifact;readable paths alone never grant writes",
    )
    parser.add_argument(
        "--architecture-context",
        choices=("reversible", "plain"),
        default="reversible",
        help="cpu-architecture:plain isolates planning from working-set source externalization and history compression",
    )
    parser.add_argument(
        "--proposal-body-scope",
        help="explicit relative.py::Class.method body for one optional implementation proposal",
    )
    parser.add_argument(
        "--proposal-read-path",
        action="append",
        default=[],
        help="additional caller-readable source for proposal;does not grant write permission",
    )
    parser.add_argument(
        "--source-question-completion",
        action="store_true",
        help="one bounded source answer and coding continuation after an explicit proposal question",
    )
    parser.add_argument(
        "--source-question-actions",
        action="store_true",
        help="optional read-only ask/locate/inspect tools in cpu-plain;requires caller source-read paths;does not grant edit scope",
    )
    parser.add_argument(
        "--observed-interface-map",
        action="store_true",
        help="cpu-plain only:represent actually delivered cross-file source uses;no extra planner or history crop",
    )
    parser.add_argument(
        "--observed-public-sources",
        type=Path,
        help="explicit public FILES JSON path->exact source text;each source must occur in the actual task FILES",
    )
    parser.add_argument(
        "--observed-scope-quote",
        action="append",
        default=[],
        help="exact original public read/write boundary quote;never inferred from source dependencies",
    )
    parser.add_argument(
        "--contract-review",
        action="store_true",
        help="cpu-plain only:one independent review and read-only diagnostic after the first real implementation edit;shares original budget",
    )
    parser.add_argument(
        "--contract-review-task-kind",
        choices=("auto", "lightweight", "implementation"),
        default="auto",
    )
    parser.add_argument(
        "--behavior-plan",
        action="store_true",
        help="cpu-plain only:ordinary Python behavior scenario and at most3 readonly CPU reruns across implementation versions;no extra model calls",
    )
    parser.add_argument(
        "--task-state-consolidation",
        action="store_true",
        help="cpu-plain only:one budgeted historical memo and fixed evidence checkpoint;default off",
    )
    parser.add_argument(
        "--revision-task-checkpoint",
        action="store_true",
        help="cpu-plain only:one checkpoint with current versioned complete source definitions;explicit readable paths;default off",
    )
    parser.add_argument(
        "--action-effect-memory",
        action="store_true",
        help="cpu-plain only:verified source effects for the next ordinary decision;no memo or history crop;default off",
    )
    parser.add_argument(
        "--convolution-provider-contract",
        type=Path,
        help="default off:explicit public-reference SHA,complete target authorization and final reserve JSON;cpu-plain with original public check;CPU profile does not grant GPU access",
    )
    parser.add_argument(
        "--reduction-provider-contract",
        type=Path,
        help="default off:explicit single-axis sum public-source contract;CPU profile does not grant GPU access",
    )
    parser.add_argument(
        "--specialist-handoff-policy",
        type=Path,
        help="default off:caller-declared hard coverage and soft objectives JSON;requires an ordinary typed implementation provider;original independent final remains required",
    )
    parser.add_argument(
        "--convolution-schedule-workload",
        help="default off:fixed workload ID emitted by the trusted public CPU performance checker;requires specialist handoff and explicit schedule ownership",
    )
    parser.add_argument(
        "--reduction-schedule-workload",
        help="default off:fixed workload ID from the trusted public checker;requires sum provider and specialist handoff",
    )
    parser.add_argument(
        "--schedule-exclusive-workspace",
        action="store_true",
        help="caller guarantees sole workspace ownership during fixed implementation schedule selection",
    )
    parser.add_argument(
        "--public-replay-contract",
        type=Path,
        help="default off:explicit public Python argv,readable sources,input hashes and provenance;one bounded diagnostic after that exact public command fails",
    )
    parser.add_argument(
        "--public-reproduction-contract",
        type=Path,
        help="default off:shared uninstrumented public bootstrap plus optional bounded dynamic evidence;explicit public argv/source hashes/provenance JSON",
    )
    parser.add_argument(
        "--without-public-diagnostics",
        action="store_true",
        help="keep the same common public bootstrap but omit dynamic evidence;requires --public-reproduction-contract",
    )
    parser.add_argument(
        "--public-reproduction-sources",
        type=Path,
        help="JSON mapping __ic_public__/filename.py to separate frozen public input files;readonly mount,not exported candidate source",
    )
    parser.add_argument(
        "--transition-contract",
        type=Path,
        help="explicit public tensor/state diagnostic contract JSON",
    )
    parser.add_argument(
        "--transition-reference-script",
        type=Path,
        help="caller-frozen public reference;separate from private final",
    )
    parser.add_argument(
        "--public-check-script",
        type=Path,
        help="explicit trusted public CPU tests;separate from private --check-script",
    )
    parser.add_argument(
        "--public-check-coverage", help="caller description of what these public tests cover"
    )
    parser.add_argument(
        "--public-check-path",
        action="append",
        default=[],
        help="workspace-relative source observed by the public checker",
    )
    parser.add_argument("--public-check-timeout", type=float, default=60)
    parser.add_argument(
        "--public-check-feedback",
        choices=("direct", "reports"),
        default="direct",
        help="reports: explicit run and paged read,with a fresh checker environment;requires cpu-plain or public-transition-probe",
    )
    parser.add_argument(
        "--exclusive-workspace",
        action="store_true",
        help="caller guarantees sole ownership during candidate search or cpu-auto measured rollback",
    )
    parser.add_argument(
        "--phase-history",
        action="store_true",
        help="experimental default-off per-phase history budget;requires cpu-candidate-search",
    )
    parser.add_argument(
        "--hypothesis-branch",
        action="store_true",
        help="experimental alternative hypothesis after a source-bound public failure; requires cpu-candidate-search",
    )
    parser.add_argument(
        "--task-environment",
        choices=("command-private", "task-private"),
        default="command-private",
        help="task-private keeps task tmp/HOME/cache across fresh-shell actions;requires cpu-plain or public-transition-probe;trusted checks stay separate",
    )
    parser.add_argument(
        "--planning-task-kind",
        choices=("repository", "lightweight"),
        default="repository",
        help="caller-declared applicability for cpu-planning;lightweight keeps the ordinary input",
    )
    parser.add_argument("--implementation-path", action="append", default=[])
    parser.add_argument(
        "--source-read-path",
        action="append",
        default=[],
        help="additional caller-readable source;does not grant write permission",
    )
    parser.add_argument("--public-target", action="append", default=[])
    parser.add_argument(
        "--body-write-scope",
        action="append",
        default=[],
        help="caller-explicit public path::Python.method body constraint for diagnosis;never inferred from reading targets",
    )
    parser.add_argument("--context-limit", type=int, default=16384)
    parser.add_argument("--output-tokens", type=int, default=4096)
    parser.add_argument("--max-calls", type=int, default=15)
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument(
        "--delivery-policy",
        choices=("legacy", "logical-query-v1"),
        default="legacy",
        help="logical-query-v1 shares a600s logical wait and never resubmits unknown upstream outcomes;requires request-timeout>=600",
    )
    parser.add_argument("--final-timeout", type=float, default=180)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="currently rejected before generation;select Runtime explicitly",
    )
    parser.add_argument(
        "--pause-required", action="store_true", help="currently rejected before generation"
    )
    args = parser.parse_args(argv)
    started = time.monotonic()
    from contextlib import redirect_stdout

    output = sys.stdout
    with redirect_stdout(sys.stderr):
        return _execute(args, started, output)


def _execute(args, started, output):
    try:
        from infinicoder.runtime.agent_execution import CoreBudget
        from infinicoder.models.budget import ModelProfile
        from infinicoder.models.model_client import ModelConfig
        from infinicoder.app.native_api import create_cpu_request, run_native
        from infinicoder.models.native_delivery_policy import DeliveryPolicy
        from infinicoder.execution.native_sandbox import SANDBOX_NOTICE, PERSISTENT_SANDBOX_NOTICE
        from infinicoder.planning.contract_review import ContractReviewPolicy
        from infinicoder.planning.task_contract import TaskSpec

        if args.convolution_provider_contract and not args.spec_json:
            raise ValueError(
                "--convolution-provider-contract requires --spec-json to preserve exact public REFERENCE bytes"
            )
        if args.spec_json:
            if args.require_artifact:
                raise ValueError("--require-artifact belongs to --task-file;preserve JSON contract")
            spec = TaskSpec.from_dict(json.loads(args.spec_json.read_text(encoding="utf-8-sig")))
        else:
            spec = TaskSpec(
                args.task_file.read_text(encoding="utf8")
                + "\n"
                + (
                    PERSISTENT_SANDBOX_NOTICE
                    if args.task_environment == "task-private"
                    else SANDBOX_NOTICE
                ),
                required_artifacts=args.require_artifact,
            )
        key = ""
        if args.api_key_env:
            if args.api_key_env not in os.environ:
                raise ValueError("selected provider credential environment variable is missing")
            key = os.environ[args.api_key_env]
        config = ModelConfig(
            args.base_url,
            args.model,
            api_key=key,
            max_output_tokens=args.output_tokens,
            temperature=0,
            extra_body=dict(seed=42, chat_template_kwargs=dict(enable_thinking=False)),
        )
        request = create_cpu_request(
            spec,
            args.workspace,
            args.state_dir,
            model_config=config,
            pipeline_policy=json.loads(args.capability_policy.read_text(encoding="utf-8-sig"))
            if args.capability_policy
            else None,
            public_performance=json.loads(
                args.public_performance_contract.read_text(encoding="utf-8-sig")
            )
            if args.public_performance_contract
            else None,
            tokenizer_path=args.tokenizer,
            check_script=args.check_script,
            profile=args.profile,
            model_profile=ModelProfile(args.context_limit, args.output_tokens),
            budget=CoreBudget(args.max_calls, args.seconds, args.request_timeout),
            implementation_paths=args.implementation_path,
            public_targets=args.public_target,
            body_write_scopes=args.body_write_scope,
            task_working_memory=args.task_working_memory,
            architecture_write_paths=args.architecture_write_path,
            architecture_context=args.architecture_context,
            source_attachment=(
                dict(
                    literal=args.source_attachment.read_text(encoding="utf8"),
                    files=json.loads(args.source_attachment.read_text(encoding="utf8")),
                )
                if args.source_attachment
                else None
            ),
            source_read_paths=args.source_read_path,
            public_check_script=args.public_check_script,
            public_check_coverage=dict(description=args.public_check_coverage)
            if args.public_check_coverage
            else None,
            public_check_paths=args.public_check_path,
            public_check_timeout=args.public_check_timeout,
            public_check_feedback=args.public_check_feedback,
            transition_contract=json.loads(args.transition_contract.read_text(encoding="utf-8-sig"))
            if args.transition_contract
            else None,
            transition_reference_script=args.transition_reference_script,
            exclusive_workspace=args.exclusive_workspace,
            phase_history=args.phase_history,
            hypothesis_branch=args.hypothesis_branch,
            task_environment=args.task_environment,
            planning_repository_task=args.planning_task_kind == "repository",
            proposal_body_scope=args.proposal_body_scope,
            proposal_read_paths=args.proposal_read_path,
            source_question_completion=args.source_question_completion,
            source_question_actions=args.source_question_actions,
            delivery_policy=DeliveryPolicy()
            if args.delivery_policy == "logical-query-v1"
            else None,
            observed_interface_map=args.observed_interface_map,
            observed_public_sources=json.loads(
                args.observed_public_sources.read_text(encoding="utf-8-sig")
            )
            if args.observed_public_sources
            else None,
            observed_scope_quotes=args.observed_scope_quote,
            contract_review=args.contract_review,
            contract_review_policy=ContractReviewPolicy(task_kind=args.contract_review_task_kind)
            if args.contract_review
            else None,
            behavior_plan=args.behavior_plan,
            task_state_consolidation=args.task_state_consolidation,
            revision_task_checkpoint=args.revision_task_checkpoint,
            action_effect_memory=args.action_effect_memory,
            convolution_provider=json.loads(
                args.convolution_provider_contract.read_text(encoding="utf-8-sig")
            )
            if args.convolution_provider_contract
            else None,
            reduction_provider=json.loads(
                args.reduction_provider_contract.read_text(encoding="utf-8-sig")
            )
            if args.reduction_provider_contract
            else None,
            fusion_provider=json.loads(
                args.fusion_provider_contract.read_text(encoding="utf-8-sig")
            )
            if args.fusion_provider_contract
            else None,
            public_replay=json.loads(args.public_replay_contract.read_text(encoding="utf-8-sig"))
            if args.public_replay_contract
            else None,
            public_reproduction=json.loads(
                args.public_reproduction_contract.read_text(encoding="utf-8-sig")
            )
            if args.public_reproduction_contract
            else None,
            public_diagnostics=not args.without_public_diagnostics,
            public_reproduction_sources=json.loads(
                args.public_reproduction_sources.read_text(encoding="utf-8-sig")
            )
            if args.public_reproduction_sources
            else None,
            final_timeout=args.final_timeout,
        )
        request = replace(request, resume=args.resume, pause_required=args.pause_required)
        if args.specialist_handoff_policy:
            from infinicoder.collaboration.implementation_handoff import with_specialist_handoff

            request = with_specialist_handoff(
                request, json.loads(args.specialist_handoff_policy.read_text(encoding="utf-8-sig"))
            )
        if (
            sum(
                bool(v)
                for v in (
                    args.convolution_schedule_workload,
                    args.reduction_schedule_workload,
                    args.fusion_schedule_workload,
                )
            )
            > 1
        ):
            raise ValueError("choose one matching implementation schedule workload")
        schedule_workload = (
            args.convolution_schedule_workload
            or args.reduction_schedule_workload
            or args.fusion_schedule_workload
        )
        if args.schedule_exclusive_workspace and not schedule_workload:
            raise ValueError(
                "--schedule-exclusive-workspace requires an implementation schedule workload"
            )
        if schedule_workload:
            from infinicoder.domains.operator.convolution_schedule_search import (
                with_convolution_schedule_search,
            )
            from infinicoder.domains.operator.reduction_schedule_search import (
                with_reduction_schedule_search,
            )
            from infinicoder.domains.operator.fusion_schedule_search import (
                with_fusion_schedule_search,
            )
            from infinicoder.verification.public_performance_report import PublicPerformanceReport

            owner = request.context_adapter
            if not hasattr(owner, "contract") or not args.specialist_handoff_policy:
                raise ValueError(
                    "schedule selection requires an explicit provider and specialist handoff"
                )
            report = PublicPerformanceReport(schedule_workload, (owner.contract.target,))
            decoder, identity = report.decoder(owner.public_check)
            attach = (
                with_fusion_schedule_search
                if args.fusion_schedule_workload
                else with_reduction_schedule_search
                if args.reduction_schedule_workload
                else with_convolution_schedule_search
            )
            request = attach(
                request,
                performance_decoder=decoder,
                decoder_identity=identity,
                exclusive_workspace=args.schedule_exclusive_workspace,
            )
        selection = getattr(
            request.context_adapter,
            "selection",
            getattr(request.context_adapter, "capability_selection", None),
        )
        if selection is not None:
            print(
                json.dumps(
                    dict(event="capability_selection", selection=selection), ensure_ascii=False
                ),
                file=sys.stderr,
            )
        setup = time.monotonic() - started
        outcome = run_native(request)
        value = dict(
            outcome=asdict(outcome),
            success=outcome.success,
            profile=args.profile,
            setup_wall_s=setup,
            total_wall_s=time.monotonic() - started,
        )
        value["task_environment"] = args.task_environment
        value["public_check_feedback"] = args.public_check_feedback
        value["source_question_actions"] = args.source_question_actions
        value["action_effect_memory"] = args.action_effect_memory
        value["convolution_provider"] = args.convolution_provider_contract is not None
        value["reduction_provider"] = args.reduction_provider_contract is not None
        value["specialist_handoff"] = args.specialist_handoff_policy is not None
        value["convolution_schedule_search"] = args.convolution_schedule_workload is not None
        value["reduction_schedule_search"] = args.reduction_schedule_workload is not None
        value["capability_pipeline"] = args.profile == "cpu-auto"
        value["public_replay"] = args.public_replay_contract is not None
        value["public_reproduction"] = args.public_reproduction_contract is not None
        value["public_diagnostics"] = (
            args.public_reproduction_contract is not None and not args.without_public_diagnostics
        )
        value["delivery_policy"] = args.delivery_policy
        value["observed_interface_map"] = args.observed_interface_map
        value["contract_review"] = args.contract_review
        value["behavior_plan"] = args.behavior_plan
        value["task_state_consolidation"] = args.task_state_consolidation
        value["revision_task_checkpoint"] = args.revision_task_checkpoint
        if args.phase_history or args.profile == "cpu-single-history":
            value["phase_history"] = "phase_history_budget_v1"
        (request.state_dir / "native-entry.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf8"
        )
        print(json.dumps(value, ensure_ascii=False), file=output)
        return 0 if outcome.success else 1
    except (ValueError, OSError, ImportError) as error:
        print(
            json.dumps(
                dict(
                    error=type(error).__name__ + ": " + str(error),
                    selected_entry="explicit Native;unsupported contracts require explicit Runtime selection",
                ),
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
