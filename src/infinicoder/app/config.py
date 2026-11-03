"""Common configuration for the persistent Runtime backend."""

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math

from infinicoder.models.budget import ModelProfile


@dataclass(frozen=True)
class RuntimeConfig:
    """Session options; model credentials belong to ModelConfig, not this object.

    Advanced experiments remain available on Runtime. Changing configuration
    when resuming a session is subject to its existing policy identity checks.
    """

    profile: str = "auto"
    max_steps: int = 60
    max_model_calls: int = 60
    time_limit: float = 1800
    request_timeout: float = 180
    context_limit: int = 16384
    allow_commands: bool = False
    code_memory: bool = True
    development_workflows: bool = True
    automatic_workers: bool = True
    domain_compiler: bool = True
    fusion_expert: bool = False

    def __post_init__(self):
        if self.profile not in ("auto", "general", "operator", "framework", "legacy"):
            raise ValueError("Unknown Runtime profile")
        for name in ("max_steps", "max_model_calls", "context_limit"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("time_limit", "request_timeout"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        for name in (
            "allow_commands",
            "code_memory",
            "development_workflows",
            "automatic_workers",
            "domain_compiler",
            "fusion_expert",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")


def create_runtime(
    workspace,
    client,
    state_dir,
    *,
    config=None,
    estimator=None,
    execution_profile=None,
    event_handler=None,
    verifier=None,
):
    """Create the existing Runtime with a small, explicit configuration surface.

    No model call or task execution occurs here. Supply the actual serving
    tokenizer through estimator for accurate context accounting.
    """
    from infinicoder.runtime.engine import Runtime

    config = config if config is not None else RuntimeConfig()
    if not isinstance(config, RuntimeConfig):
        raise TypeError("config must be RuntimeConfig")
    options = asdict(config)
    context_limit = options.pop("context_limit")
    profile = ModelProfile(
        context_limit=context_limit, output_reserve=client.config.max_output_tokens
    )
    runtime = Runtime(
        workspace,
        client,
        state_dir,
        **options,
        model_profile=profile,
        estimator=estimator,
        execution_profile=execution_profile,
        event_handler=event_handler,
        verifier=verifier,
    )
    runtime.bind_session_configuration = True
    return runtime


def session_configuration_identity(runtime):
    """Hash actual model and common limits without persisting credentials.

    API key rotation is allowed. The remaining provider settings and limits
    must match; older sessions without this binding require the legacy API.
    Existing engine policy identities continue to cover advanced strategies.
    """
    config = runtime.client.config
    model = asdict(config) if is_dataclass(config) else dict(vars(config))
    model.pop("api_key", None)
    if model.get("temperature") is not None:
        model["temperature"] = float(model["temperature"])
    fields = (
        "max_steps",
        "max_model_calls",
        "time_limit",
        "request_timeout",
        "output_budget_bytes",
        "allow_commands",
        "profile",
        "domain_compiler",
        "automatic_workers",
        "code_memory",
        "development_workflows",
    )
    limits = {name: getattr(runtime, name) for name in fields}
    for name in ("time_limit", "request_timeout"):
        limits[name] = float(limits[name])
    payload = dict(schema=2, model=model, limits=limits, context=asdict(runtime.assembler.profile))
    if getattr(runtime, "fusion_expert", False):
        payload["fusion_expert"] = "linear_gelu_fusion_v1"
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_session_configuration(runtime, state, *, resume):
    """Validate the same common configuration before execution or revision."""
    saved = state.get("configuration_identity") if state else None
    if not getattr(runtime, "bind_session_configuration", False) and saved is None:
        return None
    current = session_configuration_identity(runtime)
    if resume and saved != current:
        raise ValueError(
            "Session model or budget configuration changed, or the session predates configuration binding; use the original configuration or the legacy API for an older session"
        )
    return current
