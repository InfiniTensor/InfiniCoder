"""token budget estimation and model profiles.

Replaces character-approximate budgets with conservative token estimates.
No hard dependency on tokenizer libraries: when the caller provides a
tokenizer callable it is used; otherwise a documented conservative
heuristic bounds the estimate (over-estimating is safe, under-estimating
silently truncates).
"""

import math
import re
import hashlib
import json
from dataclasses import dataclass

# Conservative ASCII packing: services commonly pack 3.5-4.5 chars/token for
# code/JSON; assume 3.0 to over-estimate (safer for budget rejection).
_ASCII_CHARS_PER_TOKEN = 3.0
# CJK codepoints are ~1 token each on mainstream tokenizers; assume 1.0.
_CJK_RE = re.compile(r"[\u3000-\u9fff\uff00-\uffef\u4e00-\u9fff]")


def heuristic_tokens(text):
    """Approximate estimate, not a tokenizer upper bound.

    Supply the serving model's tokenizer for exact content accounting;
    chat framing still needs an explicit protocol overhead reserve.
    """
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    rest = len(text) - cjk
    return int(cjk + math.ceil(rest / _ASCII_CHARS_PER_TOKEN) + 1)


@dataclass(frozen=True)
class TokenEstimator:
    """Optional tokenizer adapter; falls back to the conservative heuristic."""

    tokenizer: object = None  # callable text -> token count
    request_counter: object = None  # optional exact chat/schema/output bound

    def estimate(self, text):
        if self.tokenizer is not None:
            value = int(self.tokenizer(text))
            if value < 0:
                raise ValueError("tokenizer returned negative count")
            return value
        return heuristic_tokens(text)


@dataclass(frozen=True)
class ModelProfile:
    """Declared service limits for budget arithmetic (tokens)."""

    context_limit: int = 32768
    output_reserve: int = 2048
    protocol_overhead: int = 256  # chat template / role framing slack

    def __post_init__(self):
        for name in ("context_limit", "output_reserve", "protocol_overhead"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"ModelProfile.{name} must be a positive int")
        if self.output_reserve + self.protocol_overhead >= self.context_limit:
            raise ValueError("ModelProfile: output+overhead must leave input room")

    def input_budget(self):
        """Max tokens available for message content (system+history+pack)."""
        return self.context_limit - self.output_reserve - self.protocol_overhead


class BudgetExceeded(ValueError):
    """Raised when even the mandatory contract cannot fit the input budget.

    Zero model requests must follow: the runtime surfaces the specific
    reason instead of silently truncating mandatory constraints."""


class ContextAssembler:
    """Assembles message parts under a token budget with drop priorities.

    Mandatory parts (task contract, acceptance, current work unit) are never
    dropped; if they alone exceed the budget, BudgetExceeded is raised.
    Optional parts drop in order: code pack, older history, notes.
    """

    PRIORITIES = (
        "mandatory",
        "source_recovery",
        "history",
        "notes",
        "interface_memory",
        "code_pack",
    )

    def __init__(self, estimator=None, profile=None):
        self.estimator = estimator or TokenEstimator()
        self.profile = profile or ModelProfile()
        self.extra_fixed_context = ""
        self.extra_request_tokens = 0
        self.request_reassembly = None

    def bind_reassembly(self, messages, rebuild):
        """Bind a pure same-stage rebuild, never a Runtime action or query."""
        identity = hashlib.sha256(
            json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        self.request_reassembly = (identity, rebuild)

    def reassemble_request(self, messages, reserve):
        identity = hashlib.sha256(
            json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        if self.request_reassembly is None or self.request_reassembly[0] != identity:
            raise BudgetExceeded(
                "Physical request exceeds context and has no matching same-stage assembly"
            )
        if type(reserve) is not int or reserve < 0:
            raise ValueError("nonnegative physical request reserve required")
        previous = self.extra_request_tokens
        try:
            self.extra_request_tokens = reserve
            return self.request_reassembly[1]()
        finally:
            self.extra_request_tokens = previous

    def assemble(self, mandatory_text, optional_parts, fixed_overhead_text=""):
        """mandatory_text: str. optional_parts: dict name -> str.
        fixed_overhead_text: always-sent content (e.g. system prompt) that
        consumes budget but is emitted separately by the caller.
        'history' is accounted but NOT joined into the assembled text —
        the runtime re-emits it as real chat turns.
        Returns (assembled_text, report)."""
        if type(self.extra_request_tokens) is not int or self.extra_request_tokens < 0:
            raise ValueError("nonnegative physical request reserve required")
        budget = self.profile.input_budget() - self.extra_request_tokens
        overhead = self.estimator.estimate(fixed_overhead_text + self.extra_fixed_context)
        if overhead > budget:
            raise BudgetExceeded(
                f"system/overhead {overhead}t alone exceeds input budget "
                f"{budget}t; raise context_limit"
            )
        budget -= overhead
        mand = self.estimator.estimate(mandatory_text)
        if mand > budget:
            raise BudgetExceeded(
                f"mandatory content {mand}t exceeds input budget {budget}t "
                f"(context_limit={self.profile.context_limit}, "
                f"output_reserve={self.profile.output_reserve}); "
                "split the task or raise context_limit"
            )
        report = {
            "mandatory_tokens": mand,
            "overhead_tokens": overhead,
            "budget": budget,
            "selected": [],
            "dropped": [],
        }
        if self.extra_request_tokens:
            report["physical_request_extra_tokens"] = self.extra_request_tokens
        remaining = budget - mand
        # Drop order (first dropped first): code_pack, older history, notes.
        for name in ("source_recovery", "history", "notes", "interface_memory", "code_pack"):
            part = optional_parts.get(name)
            if part is None:
                continue
            cost = self.estimator.estimate(part)
            if cost <= remaining:
                remaining -= cost
                report["selected"].append({"name": name, "tokens": cost})
            else:
                report["dropped"].append({"name": name, "tokens": cost, "reason": "budget"})
        report["remaining_tokens"] = remaining
        selected = {s["name"] for s in report["selected"]}
        # Only non-history parts join the text; history becomes chat turns.
        pieces = [mandatory_text] + [
            optional_parts[n]
            for n in ("source_recovery", "notes", "interface_memory", "code_pack")
            if n in selected
        ]
        return "\n".join(p for p in pieces if p), report
