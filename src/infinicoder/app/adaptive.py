"""Explicit experimental API for task-conditioned Runtime composition."""

from dataclasses import asdict
import json
from pathlib import Path
from infinicoder.routing.capability_router import decide, identity, contract_identity
from infinicoder.planning.task_contract import TaskSpec


def run_routed(
    spec,
    workspace,
    client,
    state_dir,
    *,
    model="",
    context_limit=16384,
    allow_commands=False,
    enabled=(),
    disabled=(),
    resume=False,
    runtime_factory=None,
    policy="standard",
    **runtime_options,
):
    """Run existing capabilities selected before constructing the real Runtime.

    runtime_factory preserves caller-owned execution/sandbox subclasses. No new
    process, command permission, model budget or verifier is supplied by routing.
    This API is experimental; the ordinary Runtime/run defaults stay unchanged.
    """
    if isinstance(spec, dict):
        spec = TaskSpec.from_dict(spec)
    else:
        spec = TaskSpec.from_dict(asdict(spec))
    workspace = Path(workspace).resolve()
    state_dir = Path(state_dir).resolve()
    if state_dir.is_relative_to(workspace):
        raise ValueError("Session and routing evidence must be outside editable workspace")
    actual_model = getattr(getattr(client, "config", None), "model", "")
    if model and actual_model and model != actual_model:
        raise ValueError("Routing model differs from actual client")
    model = model or actual_model
    performance_search = runtime_options.get("optimization") is not None
    if performance_search and "checked_edits" in enabled:
        raise ValueError("checked_edits cannot terminate a performance search on correctness alone")
    route_path = state_dir / "capability-route.json"
    request = dict(
        contract=contract_identity(spec),
        model=model,
        context_limit=context_limit,
        allow_commands=allow_commands,
        enabled=list(enabled),
        disabled=list(disabled),
        policy=policy,
        performance_search=performance_search,
        router_identity=identity(),
    )
    if route_path.exists():
        if not resume:
            raise ValueError("Routing session already exists; use resume or a new state directory")
        record = json.loads(route_path.read_bytes())
        if record["request"] != request:
            raise ValueError(
                "Routing contract,model,permissions or policy changed; preserve session and use a new one"
            )
        route = record["decision"]
    else:
        if resume:
            raise ValueError(
                "No saved routing decision; ordinary sessions require an explicit migration"
            )
        route = decide(
            spec,
            workspace,
            model=model,
            context_limit=context_limit,
            allow_commands=allow_commands,
            enabled=enabled,
            disabled=disabled,
            policy=policy,
            performance_search=performance_search,
        )
        record = dict(request=request, decision=route)
    if "checked_edits" in route["modules"] and runtime_options.get("optimization") is not None:
        raise ValueError("checked_edits cannot terminate a performance search on correctness alone")
    conflicts = [
        name
        for name, value in runtime_options.items()
        if name in route["runtime_options"] and route["runtime_options"][name] != value
    ]
    if conflicts:
        raise ValueError(
            "Use enabled/disabled modules to resolve explicit routing option conflicts: "
            + ",".join(sorted(conflicts))
        )
    if runtime_factory is None:
        from infinicoder.runtime.engine import Runtime

        runtime_factory = Runtime
    from infinicoder.models.budget import ModelProfile

    profile = runtime_options.get("model_profile")
    if profile is not None and profile.context_limit != context_limit:
        raise ValueError("Routing context must equal the actual Runtime model profile")
    if profile is None:
        output = getattr(getattr(client, "config", None), "max_output_tokens", 1024)
        runtime_options["model_profile"] = ModelProfile(
            context_limit=context_limit, output_reserve=output
        )
    runtime = runtime_factory(
        workspace,
        client,
        state_dir,
        allow_commands=allow_commands,
        **dict(route["runtime_options"], **runtime_options),
    )
    if not route_path.exists():
        state_dir.mkdir(parents=True, exist_ok=True)
        with route_path.open("x", encoding="utf8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
    return runtime.execute(spec, resume=resume)


def main():
    import argparse, os
    from infinicoder.models.model_client import CompatibleClient, ModelConfig

    parser = argparse.ArgumentParser(
        description="Experimental task-conditioned capability composition"
    )
    for name in ("workspace", "spec"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--state-dir")
    parser.add_argument("--route-only", action="store_true")
    parser.add_argument("--model", default=os.environ.get("INFINICODER_MODEL", ""))
    parser.add_argument("--base-url", default=os.environ.get("INFINICODER_API_BASE", ""))
    parser.add_argument("--context-limit", type=int, default=16384)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--max-model-calls", type=int, default=8)
    parser.add_argument("--time-limit", type=float, default=120)
    parser.add_argument("--allow-commands", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--enable", action="append", default=[])
    parser.add_argument("--disable", action="append", default=[])
    parser.add_argument(
        "--policy",
        choices=(
            "standard",
            "native_candidate",
            "conditional_candidate",
            "evidence_candidate",
            "focus_candidate",
            "native_plan_candidate",
        ),
        default="standard",
    )
    args = parser.parse_args()
    spec = TaskSpec.from_dict(json.loads(Path(args.spec).read_bytes()))
    if args.route_only:
        print(
            json.dumps(
                decide(
                    spec,
                    args.workspace,
                    model=args.model,
                    context_limit=args.context_limit,
                    allow_commands=args.allow_commands,
                    enabled=args.enable,
                    disabled=args.disable,
                    policy=args.policy,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if not args.state_dir:
        parser.error("--state-dir is required when executing")
    client = CompatibleClient(
        ModelConfig(
            args.base_url,
            args.model,
            os.environ.get("INFINICODER_API_KEY", ""),
            max_output_tokens=args.max_output_tokens,
            temperature=0,
            extra_body=dict(
                seed=42, chat_template_kwargs=dict(enable_thinking=False), skip_special_tokens=False
            ),
        )
    )
    state = run_routed(
        spec,
        args.workspace,
        client,
        args.state_dir,
        model=args.model,
        context_limit=args.context_limit,
        allow_commands=args.allow_commands,
        enabled=args.enable,
        disabled=args.disable,
        resume=args.resume,
        policy=args.policy,
        max_steps=args.max_model_calls,
        max_model_calls=args.max_model_calls,
        time_limit=args.time_limit,
    )
    print(
        json.dumps(
            {key: state.get(key) for key in ("status", "reason", "model_calls", "changed_files")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
