"""Explicit same-model generation profiles;global Runtime budgets stay authoritative."""

from dataclasses import dataclass, replace
import hashlib
import json
import re
import threading
from infinicoder.models.model_client import (
    CompatibleClient,
    ModelConfig,
    validate_model_config,
    _validate_number,
)


@dataclass(frozen=True)
class GenerationProfile:
    name: str
    client: object
    request_cap_seconds: float = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch("[a-z][a-z0-9_-]{0,31}", self.name):
            raise ValueError("bounded generation profile name required")
        if not isinstance(getattr(self.client, "config", None), ModelConfig) or not callable(
            getattr(self.client, "generate", None)
        ):
            raise ValueError("profile needs a configured compatible client")
        validate_model_config(self.client.config)
        if getattr(self.client, "supports_structured_output", False) and not callable(
            getattr(self.client, "generate_structured", None)
        ):
            raise ValueError("profile advertised structured output without a structured generator")
        if self.request_cap_seconds is not None:
            _validate_number(self.request_cap_seconds, "profile request cap", minimum=0.1)


class ScheduledClient:
    """One task's explicit profile schedule;the last stage repeats.

    The declared config reserves the largest output cap in context assembly.
    All profiles use the same endpoint,model,protocol and credential context.
    restore_generation_position restores only the schedule position;it never resets API counts
    or the Runtime's persisted call/time budget. One in-flight call is allowed.
    """

    def __init__(self, profiles, schedule):
        profiles = tuple(profiles)
        schedule = tuple(schedule)
        if not profiles or not all(isinstance(p, GenerationProfile) for p in profiles):
            raise ValueError("explicit generation profiles required")
        self.profiles = {p.name: p for p in profiles}
        if (
            len(self.profiles) != len(profiles)
            or not schedule
            or len(schedule) > 16
            or any(name not in self.profiles for name in schedule)
        ):
            raise ValueError("unique profiles and a valid bounded schedule required")
        first = self.profiles[schedule[0]].client.config

        def context(config):
            return (config.base_url.rstrip("/"), config.model, config.protocol, config.api_key)

        if any(context(p.client.config) != context(first) for p in profiles):
            raise ValueError(
                "scheduled profiles must keep the same endpoint,model,protocol and credential context"
            )
        self.schedule = schedule
        self.config = replace(
            first, max_output_tokens=max(p.client.config.max_output_tokens for p in profiles)
        )
        self.calls = 0
        self._position = 0
        self._last_profile = None
        self._lock = threading.Lock()
        self.generation_policy_identity = hashlib.sha256(
            json.dumps(self.receipt(), sort_keys=True).encode()
        ).hexdigest()

    @property
    def supports_structured_output(self):
        return all(
            getattr(p.client, "supports_structured_output", False) for p in self.profiles.values()
        )

    @property
    def last_profile(self):
        return dict(self._last_profile) if self._last_profile is not None else None

    def receipt(self):
        def settings(profile):
            c = profile.client.config
            template = c.extra_body.get("chat_template_kwargs", {})
            return dict(
                model=c.model,
                endpoint_sha256=hashlib.sha256(c.base_url.encode()).hexdigest(),
                protocol=c.protocol,
                max_output_tokens=c.max_output_tokens,
                temperature=c.temperature,
                provider_options_sha256=hashlib.sha256(
                    json.dumps(c.extra_body, sort_keys=True).encode()
                ).hexdigest(),
                thinking=template.get("enable_thinking") if isinstance(template, dict) else None,
                reasoning_prefix_open=c.reasoning_prefix_open,
                structured_outputs=c.structured_outputs,
                request_cap_seconds=profile.request_cap_seconds,
            )

        return dict(
            kind="same_model_scheduled_generation_v1",
            schedule=list(self.schedule),
            repeat_last=True,
            output_reserve=self.config.max_output_tokens,
            profiles={name: settings(profile) for name, profile in self.profiles.items()},
        )

    def restore_generation_position(self, used_model_calls=0):
        _validate_number(used_model_calls, "used_model_calls", minimum=0, integer=True)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("cannot restore a generation profile during an active call")
        try:
            self._position = used_model_calls
        finally:
            self._lock.release()

    def _generate(self, messages, timeout, schema=None):
        _validate_number(timeout, "scheduled request timeout", minimum=0.001)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("scheduled client supports one in-flight generation")
        try:
            profile = self.profiles[self.schedule[min(self._position, len(self.schedule) - 1)]]
            self._position += 1
            self.calls += 1
            applied = (
                min(timeout, profile.request_cap_seconds)
                if profile.request_cap_seconds is not None
                else timeout
            )
            self._last_profile = dict(
                name=profile.name,
                ordinal=self._position,
                object_api_attempt=self.calls,
                max_output_tokens=profile.client.config.max_output_tokens,
                requested_timeout_s=timeout,
                applied_timeout_s=applied,
                reasoning_prefix_open=profile.client.config.reasoning_prefix_open,
            )
            if schema is not None:
                return profile.client.generate_structured(messages, schema, timeout=applied)
            return profile.client.generate(messages, timeout=applied)
        finally:
            self._lock.release()

    def generate(self, messages, timeout):
        return self._generate(messages, timeout)

    def generate_structured(self, messages, schema, timeout):
        if not isinstance(schema, dict):
            raise ValueError("structured generation requires an explicit schema object")
        if not self.supports_structured_output:
            raise ValueError(
                "all scheduled profiles must support the requested structured protocol"
            )
        return self._generate(messages, timeout, schema)


def fast_reason_fast(config, fast_output_tokens=4096, fast_seconds=60, reasoned_seconds=180):
    """Opt-in for an explicitly verified enable_thinking-compatible service.

    No model-name inference or replacement;the caller's maximum output cap is
    never increased. A rejected fast draft can use one deliberative call,then
    subsequent tool/repair turns return to the fast profile.
    """
    validate_model_config(config)
    _validate_number(fast_output_tokens, "fast_output_tokens", minimum=1, integer=True)
    template = config.extra_body.get("chat_template_kwargs", {})
    if (
        config.protocol != "openai-compatible"
        or not isinstance(template, dict)
        or template.get("enable_thinking") is not True
    ):
        raise ValueError(
            "fast_reason_fast requires caller-declared,verified enable_thinking service support"
        )
    body = json.loads(json.dumps(config.extra_body))
    body["chat_template_kwargs"]["enable_thinking"] = False
    fast = CompatibleClient(
        replace(
            config,
            max_output_tokens=min(fast_output_tokens, config.max_output_tokens),
            extra_body=body,
            reasoning_prefix_open=False,
        )
    )
    reasoned = CompatibleClient(
        replace(config, extra_body=json.loads(json.dumps(config.extra_body)))
    )
    return ScheduledClient(
        [
            GenerationProfile("fast", fast, fast_seconds),
            GenerationProfile("deliberative", reasoned, reasoned_seconds),
        ],
        ["fast", "deliberative", "fast"],
    )
