"""Inspect or execute a task through the common capability selector."""

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

from infinicoder.runtime.agent_execution import CoreBudget
from infinicoder.models.budget import ModelProfile
from infinicoder.models.model_client import ModelConfig
from infinicoder.app.task_solver import SolvePolicy, inspect_task, solve
from infinicoder.planning.task_contract import TaskSpec


def main(argv=None):
    parser = argparse.ArgumentParser(description="InfiniCoder task workflows and activation report")
    parser.add_argument("operation", choices=("inspect", "solve"))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True, help="complete TaskSpec JSON")
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--backend", choices=("runtime", "native"), default="runtime")
    parser.add_argument(
        "--workflow", choices=("auto", "general", "operator", "framework"), default="auto"
    )
    parser.add_argument("--context", choices=("auto", "source", "task", "off"), default="auto")
    parser.add_argument("--planning", choices=("auto", "on", "off"), default="auto")
    parser.add_argument(
        "--baseline", action="store_true", help="Native: same tools with enhancements disabled"
    )
    parser.add_argument("--source-path", action="append", default=[])
    parser.add_argument("--implementation-path", action="append", default=[])
    parser.add_argument("--plan-path", action="append", default=[])
    parser.add_argument(
        "--check-script", type=Path, help="Independent final script, outside workspace"
    )
    parser.add_argument(
        "--public-check-script", type=Path, help="Public feedback script, outside workspace"
    )
    parser.add_argument("--tokenizer", type=Path, help="Matching local tokenizer")
    parser.add_argument(
        "--python-env",
        type=Path,
        help="Project virtual environment for isolated checks; separate from controller dependencies",
    )
    parser.add_argument("--base-url", default=os.environ.get("INFINICODER_API_BASE"))
    parser.add_argument("--model", default=os.environ.get("INFINICODER_MODEL"))
    parser.add_argument("--api-key-env", default="INFINICODER_API_KEY")
    parser.add_argument("--context-limit", type=int, default=16384)
    parser.add_argument("--output-tokens", type=int, default=4096)
    parser.add_argument("--max-calls", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=180)
    parser.add_argument(
        "--allow-commands", action="store_true", help="Runtime workspace code execution"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Runtime only, same task and settings"
    )
    args = parser.parse_args(argv)
    try:
        spec = TaskSpec.from_dict(json.loads(args.spec.read_text(encoding="utf-8-sig")))
        policy = SolvePolicy(
            args.backend, args.workflow, args.context, args.planning, not args.baseline
        )
        budget = CoreBudget(args.max_calls, args.seconds, args.request_timeout)
        readable = tuple(dict.fromkeys((*args.implementation_path, *args.source_path)))
        selection = inspect_task(
            spec,
            args.workspace,
            policy=policy,
            source_paths=readable,
            plan_paths=args.plan_path,
            budget=budget,
        )
        if args.operation == "inspect":
            print(json.dumps(selection, ensure_ascii=False, indent=2))
            return 0
        if not args.state_dir or not args.base_url or not args.model:
            raise ValueError(
                "solve requires --state-dir and model/base URL arguments or environment"
            )
        model = ModelConfig(
            args.base_url,
            args.model,
            api_key=os.environ.get(args.api_key_env, ""),
            max_output_tokens=args.output_tokens,
            temperature=0,
            extra_body=dict(seed=42, chat_template_kwargs=dict(enable_thinking=False)),
        )
        print(
            json.dumps(
                {
                    "event": "capabilities_selected",
                    "backend": args.backend,
                    "domain": selection["domain"],
                    "modules": [k for k, v in selection["modules"].items() if v["selected"]],
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        with redirect_stdout(sys.stderr):
            result = solve(
                spec,
                args.workspace,
                args.state_dir,
                model_config=model,
                policy=policy,
                budget=budget,
                model_profile=ModelProfile(args.context_limit, args.output_tokens),
                tokenizer_path=args.tokenizer,
                python_environment=args.python_env,
                source_paths=args.source_path,
                implementation_paths=args.implementation_path,
                plan_paths=args.plan_path,
                check_script=args.check_script,
                public_check_script=args.public_check_script,
                allow_commands=args.allow_commands,
                resume=args.resume,
            )
        if args.backend == "runtime":
            success = result.get("success", result.get("status") == "succeeded")
            summary = {k: result.get(k) for k in ("status", "reason", "model_calls")}
            summary.update(
                success=success, independent_acceptance=result.get("independent_acceptance")
            )
        else:
            success = result.success
            summary = dict(
                success=success,
                exit_reason=result.exit_reason,
                final_acceptance=asdict(result.final_acceptance)
                if result.final_acceptance
                else None,
                usage_ledger_ref=result.usage_ledger_ref,
            )
        summary["capability_report"] = str(args.state_dir.resolve() / "capabilities.json")
        print(json.dumps(summary, ensure_ascii=False))
        return 0 if success else 1
    except (ValueError, OSError, ImportError, TypeError) as error:
        print(
            json.dumps(
                dict(status="error", error_type=type(error).__name__, reason=str(error)),
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    except KeyboardInterrupt:
        return 130
