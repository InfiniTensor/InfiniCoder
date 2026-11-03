"""Direct text-action client for InfiniLM's chat completions endpoint."""

import copy
import importlib.resources
import json
import math
import time
from types import SimpleNamespace
import urllib.error
import urllib.parse
import urllib.request

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from infinicoder.models.http_transport import open_request
from infinicoder.models.model_client import ProviderError
from infinicoder.runtime.native_protocol import (
    FormatError,
    format_observation_messages,
    parse_regex_actions,
)


ACTION_REGEX = r"```mswea_bash_command\s*\n(.*?)\n```"


def default_config():
    """Load packaged prompts, with no global configuration or environment lookup."""
    return json.loads(
        importlib.resources.files("infinicoder").joinpath("native_defaults.json").read_text("utf-8")
    )


class InfiniLMModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: str
    model_kwargs: dict = Field(default_factory=dict, repr=False)
    action_regex: str = ACTION_REGEX
    format_error_template: str = "{{ error }}"
    observation_template: str = "<returncode>{{output.returncode}}</returncode>\n{{output.output}}"


class Completion:
    """Keep the original JSON envelope for auxiliary calls and usage evidence."""

    def __init__(self, data):
        if not isinstance(data, dict) or not isinstance(data.get("choices"), list):
            raise ProviderError(200, "invalid_completion")
        if len(data["choices"]) != 1 or not isinstance(data["choices"][0], dict):
            raise ProviderError(200, "expected_one_choice")
        choice = data["choices"][0]
        message = choice.get("message")
        if (
            not isinstance(message, dict)
            or message.get("role") != "assistant"
            or not isinstance(message.get("content"), str)
            or message.get("tool_calls")
            or message.get("function_call")
        ):
            raise ProviderError(200, "expected_assistant_text")
        self._data = copy.deepcopy(data)
        self.choices = [
            SimpleNamespace(
                message=SimpleNamespace(content=message["content"]),
                finish_reason=choice.get("finish_reason"),
            )
        ]

    def model_dump(self, **kwargs):
        return copy.deepcopy(self._data)


class InfiniLMTextModel:
    """One HTTP dispatch per call; retry only a proven local non-dispatch."""

    abort_exceptions = [KeyboardInterrupt]

    def __init__(self, **kwargs):
        try:
            self.config = InfiniLMModelConfig(**kwargs)
        except ValidationError:
            raise ValueError("Invalid InfiniLM text client configuration") from None
        endpoint = self.config.model_kwargs.get("api_base", "")
        parsed = urllib.parse.urlsplit(endpoint)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("InfiniLM requires an explicit HTTP(S) base URL without credentials")
        base = endpoint.rstrip("/")
        self.url = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")

    def _prepare_messages_for_api(self, messages):
        # Controller annotations must never become service arguments.
        return [{k: copy.deepcopy(v) for k, v in m.items() if k != "extra"} for m in messages]

    def _query(self, messages, **kwargs):
        settings = self.config.model_kwargs | kwargs
        timeout = settings.get("timeout", 120)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("A finite positive request timeout is required")
        if settings.get("num_retries", 0) != 0:
            raise ValueError("HTTP retries must be controlled by the delivery policy")
        allowed = {
            "api_base",
            "api_key",
            "extra_headers",
            "timeout",
            "num_retries",
            "max_tokens",
            "temperature",
        }
        if settings.keys() - allowed:
            raise ValueError(
                "Unsupported InfiniLM client options: "
                + ", ".join(sorted(settings.keys() - allowed))
            )
        if settings.get("api_base") != self.config.model_kwargs["api_base"]:
            raise ValueError("A query cannot change its configured endpoint")
        headers = dict(settings.get("extra_headers") or {})
        if any(
            k.lower() in ("authorization", "host", "content-type", "content-length")
            for k in headers
        ):
            raise ValueError("Reserved HTTP header override")
        headers["Content-Type"] = "application/json"
        if settings.get("api_key"):
            headers["Authorization"] = "Bearer " + settings["api_key"]
        body = dict(model=self.config.model_name, messages=messages, stream=False)
        body.update(
            {k: settings[k] for k in ("temperature", "max_tokens") if settings.get(k) is not None}
        )
        request = urllib.request.Request(
            self.url, json.dumps(body, allow_nan=False).encode(), headers
        )
        try:
            with open_request(request, timeout=timeout) as response:
                data = json.loads(response.read())
        except urllib.error.HTTPError as error:
            # Never embed the service body, headers or credentials in trajectories.
            status = error.code
            code = None
            try:
                detail = json.loads(error.read(4096))
                if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
                    code = detail["error"].get("code")
            except (OSError, ValueError):
                pass
            finally:
                error.close()
            category = (
                "authentication"
                if status in (401, 403)
                else "context_overflow"
                if code == "context_length_exceeded"
                else "rate_limit"
                if status == 429
                else "http_error"
            )
            raise ProviderError(status, category) from None
        except (OSError, ValueError):
            raise ProviderError(None, "transport_or_response_error") from None
        return Completion(data)

    def query(self, messages, **kwargs):
        from infinicoder.models.native_delivery_policy import ConfirmedNotDispatched

        prepared = self._prepare_messages_for_api(messages)
        for attempt in range(2):
            try:
                response = self._query(prepared, **kwargs)
                break
            except ConfirmedNotDispatched:
                if attempt:
                    raise
        raw = response.model_dump()
        choice = response.choices[0]
        extra = dict(response=raw, cost=0.0, cost_status="unpriced", timestamp=time.time())
        try:
            if choice.finish_reason != "stop":
                from jinja2 import StrictUndefined, Template

                raise FormatError(
                    dict(
                        role="user",
                        content=Template(
                            self.config.format_error_template, undefined=StrictUndefined
                        ).render(
                            finish_reason=choice.finish_reason,
                            actions=[],
                            error="The response did not finish normally; a complete text action is required.",
                        ),
                        extra={},
                    )
                )
            actions = parse_regex_actions(
                choice.message.content,
                action_regex=self.config.action_regex,
                format_error_template=self.config.format_error_template,
                template_kwargs=dict(finish_reason=choice.finish_reason),
            )
            if not actions[0]["command"]:
                raise FormatError(
                    dict(role="user", content="An empty action cannot execute.", extra={})
                )
        except FormatError as error:
            error.messages[0].setdefault("extra", {}).update(extra)
            raise
        return dict(
            role="assistant", content=choice.message.content, extra=dict(extra, actions=actions)
        )

    def format_message(self, **kwargs):
        return kwargs

    def format_observation_messages(self, message, outputs, template_vars=None):
        return format_observation_messages(
            outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
        )

    def get_template_vars(self, **kwargs):
        config = self.config.model_dump()
        config["model_kwargs"] = {
            k: v for k, v in config["model_kwargs"].items() if k not in ("api_key", "extra_headers")
        }
        return config

    def serialize(self):
        return dict(
            info=dict(
                config=dict(model=self.get_template_vars(), model_type="infinilm_http_text"),
                cost_status="unpriced",
            )
        )
