"""Shared text-action parsing, observations and loop control messages."""

# Portions adapted from mini-SWE-agent 2.4.6 (MIT); see THIRD_PARTY_NOTICES.
import re
import time
from typing import Any


class InterruptAgentFlow(Exception):
    """Raised to interrupt the agent flow and add messages."""

    def __init__(self, *messages: dict):
        self.messages = messages
        super().__init__()


class Submitted(InterruptAgentFlow):
    """Raised when the agent has completed its task."""


class LimitsExceeded(InterruptAgentFlow):
    """Raised when the agent has exceeded its cost or step limit."""


class TimeExceeded(LimitsExceeded):
    """Raised when the agent has exceeded its wall-clock time limit."""


class UserInterruption(InterruptAgentFlow):
    """Raised when the user interrupts the agent."""


class FormatError(InterruptAgentFlow):
    """Raised when the LM's output is not in the expected format."""


UNSET = object()


def recursive_merge(*dictionaries: dict | None) -> dict:
    """Merge multiple dictionaries recursively.

    Later dictionaries take precedence over earlier ones.
    Nested dictionaries are merged recursively.
    UNSET values are skipped.
    """
    if not dictionaries:
        return {}
    result: dict[str, Any] = {}
    for d in dictionaries:
        if d is None:
            continue
        for key, value in d.items():
            if value is UNSET:
                continue
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = recursive_merge(result[key], value)
            elif isinstance(value, dict):
                # Recursively merge dict values to filter out nested UNSET values
                result[key] = recursive_merge(value)
            else:
                result[key] = value
    return result


def parse_regex_actions(
    content: str,
    *,
    action_regex: str,
    format_error_template: str,
    template_kwargs: dict | None = None,
) -> list[dict]:
    """Parse actions from text content using regex. Raises FormatError if not exactly one action.

    ``template_kwargs`` are extra variables exposed to ``format_error_template`` (e.g.
    ``{"finish_reason": ...}`` so a template can report a ``max_tokens`` truncation -- which shows
    up here as zero parsed actions -- instead of a generic format error).
    """
    from jinja2 import StrictUndefined, Template

    actions = [a.strip() for a in re.findall(action_regex, content, re.DOTALL)]
    if len(actions) != 1:
        error_msg = f"Expected exactly 1 action, found {len(actions)}."
        raise FormatError(
            {
                "role": "user",
                "content": Template(format_error_template, undefined=StrictUndefined).render(
                    actions=actions, error=error_msg, **(template_kwargs or {})
                ),
                "extra": {
                    "interrupt_type": "FormatError",
                    "n_actions": len(actions),
                    "model_response": content,
                },
            }
        )
    return [{"command": action} for action in actions]


def format_observation_messages(
    outputs: list[dict],
    *,
    observation_template: str,
    template_vars: dict | None = None,
    multimodal_regex: str = "",
) -> list[dict]:
    """Format execution outputs into user observation messages."""
    from jinja2 import StrictUndefined, Template

    if multimodal_regex:
        raise ValueError("Native execution supports text observations only")
    results = []
    for output in outputs:
        content = Template(observation_template, undefined=StrictUndefined).render(
            output=output, **(template_vars or {})
        )
        msg: dict = {
            "role": "user",
            "content": content,
            "extra": {
                "raw_output": output.get("output", ""),
                "returncode": output.get("returncode"),
                "timestamp": time.time(),
                "exception_info": output.get("exception_info"),
                **output.get("extra", {}),
            },
        }
        results.append(msg)
    return results
