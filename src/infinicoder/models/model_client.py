"""Explicit per-run client for caller-selected compatible endpoints.

No provider/model defaults or model-name capability guesses. This adapter uses
complete JSON text actions; a native-tool adapter can implement the same contract.
"""

from dataclasses import dataclass, field, replace
import json
import urllib.error
import urllib.request

from infinicoder.models.http_transport import open_request


@dataclass(frozen=True)
class ModelResponse:
    text: str
    finish_reason: str = "stop"
    usage: dict = None
    raw_text: str = None


class ProviderError(RuntimeError):
    def __init__(self, status, category, retryable=False):
        super().__init__(f"Provider request failed: status={status}, category={category}")
        self.status, self.category, self.retryable = status, category, retryable


@dataclass(frozen=True)
class ModelConfig:
    base_url: str
    model: str
    api_key: str = field(default="", repr=False)
    protocol: str = "openai-compatible"
    max_output_tokens: int = 2048
    temperature: float = None
    extra_body: dict = field(default_factory=dict)
    structured_outputs: bool = False
    reasoning_prefix_open: bool = False


def _validate_number(value, field, minimum=None, maximum=None, integer=False):
    """Finite numeric validation; bool is never an acceptable number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number, got {type(value).__name__}")
    if integer and not isinstance(value, int):
        raise ValueError(f"{field} must be an integer, got {value!r}")
    import math

    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be >= {minimum}, got {value!r}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field} must be <= {maximum}, got {value!r}")
    return value


def validate_model_config(config):
    """reject invalid model configuration before any request."""
    if not isinstance(config.base_url, str) or not config.base_url.strip():
        raise ValueError("ModelConfig.base_url must be a non-empty string")
    if not isinstance(config.model, str) or not config.model.strip():
        raise ValueError("ModelConfig.model must be a non-empty string")
    if not isinstance(config.api_key, str):
        raise ValueError("ModelConfig.api_key must be a string")
    if type(config.structured_outputs) is not bool:
        raise ValueError("structured_outputs must be boolean")
    if type(config.reasoning_prefix_open) is not bool:
        raise ValueError("reasoning_prefix_open must be boolean")
    if config.reasoning_prefix_open and config.protocol != "openai-compatible":
        raise ValueError("reasoning_prefix_open requires the explicitly probed compatible protocol")
    if config.protocol not in ("openai-compatible", "anthropic"):
        raise ValueError(f"ModelConfig.protocol unsupported: {config.protocol!r}")
    _validate_number(
        config.max_output_tokens, "ModelConfig.max_output_tokens", minimum=1, integer=True
    )
    if config.temperature is not None:
        _validate_number(config.temperature, "ModelConfig.temperature", minimum=0.0, maximum=2.0)
    protected = {"model", "messages", "system", "max_tokens", "stream", "temperature"}
    if not isinstance(config.extra_body, dict) or protected & config.extra_body.keys():
        raise ValueError("extra_body must be an object without protected request fields")
    try:
        json.dumps(config.extra_body, allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError("extra_body must contain finite JSON values") from None


class CompatibleClient:
    def __init__(self, config):
        validate_model_config(config)
        self.config = config

    @property
    def supports_structured_output(self):
        # Inline reasoning prefill and an ordinary native JSON grammar cannot
        # be assumed compatible: the grammar can suppress </think> entirely.
        # Preserve the caller's explicit inline-reasoning protocol, transport
        # unconstrained text, then strictly parse only its completed final frame.
        # Services with a separate reasoning field do not need this prefill flag.
        return (
            self.config.structured_outputs
            and self.config.protocol == "openai-compatible"
            and not self.config.reasoning_prefix_open
        )

    def generate_structured(self, messages, schema, timeout):
        if not self.supports_structured_output:
            raise ValueError("structured outputs were not enabled for this service")
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "infinicoder_output", "strict": True, "schema": schema},
        }
        config = replace(
            self.config, extra_body=dict(self.config.extra_body, response_format=response_format)
        )
        return CompatibleClient(config).generate(messages, timeout)

    def generate(self, messages, timeout):
        c = self.config
        headers = {"Content-Type": "application/json"}
        body = {"model": c.model, "max_tokens": c.max_output_tokens}
        body.update(c.extra_body)
        if c.temperature is not None:
            body["temperature"] = c.temperature
        base = c.base_url.rstrip("/")
        if c.protocol == "anthropic":
            body["system"] = "\n".join(m["content"] for m in messages if m["role"] == "system")
            body["messages"] = [m for m in messages if m["role"] != "system"]
            headers.update({"anthropic-version": "2023-06-01", "x-api-key": c.api_key})
            url = base + ("/messages" if base.endswith("/v1") else "/v1/messages")
        else:
            body["messages"] = messages
            if c.api_key:
                headers["Authorization"] = "Bearer " + c.api_key
            url = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
        req = urllib.request.Request(url, json.dumps(body).encode(), headers)
        try:
            with open_request(req, timeout=max(0.1, timeout)) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            # Read only for classification; never log response bodies or credentials.
            detail = e.read(4096).decode(errors="replace").lower()
            category = (
                "authentication"
                if e.code in (401, 403)
                else "context_overflow"
                if "context" in detail and ("length" in detail or "limit" in detail)
                else "rate_limit"
                if e.code == 429
                else "server"
                if e.code >= 500
                else "configuration"
            )
            raise ProviderError(e.code, category, e.code == 429 or e.code >= 500) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise ProviderError(None, "transport", True) from None
        except (ValueError, UnicodeError):
            raise ProviderError(None, "invalid_response", True) from None
        try:
            if c.protocol == "anthropic":
                text = "".join(
                    b.get("text", "") for b in data["content"] if b.get("type") == "text"
                )
                reason = data.get("stop_reason", "unknown")
            else:
                choice = data["choices"][0]
                text = choice["message"].get("content") or ""
                reason = choice.get("finish_reason", "unknown")
            usage = data.get("usage") or {}
            if not isinstance(text, str) or not isinstance(usage, dict):
                raise ValueError("invalid text/usage")
            raw_text = text
            # Some verified chat templates prefill <think> in the prompt and
            # return only its body/closing tag. This is an explicit endpoint
            # setting, never a model-name guess. Preserve raw evidence.
            if (
                c.reasoning_prefix_open
                and not text.lstrip().startswith("<think>")
                and not (
                    choice["message"].get("reasoning_content") or choice["message"].get("reasoning")
                )
            ):
                text = "<think>" + text
            return ModelResponse(text, reason, usage, raw_text=raw_text)
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            raise ProviderError(None, "invalid_response", True) from None
