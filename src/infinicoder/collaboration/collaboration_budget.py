"""One durable budget owner for a finite set of single-request model roles.

CompatibleClient performs one HTTP request per generate entry and no retries.
Arbitrary SDK clients are deliberately not accepted by this transport adapter.
Unknown usage and late results remain charged and stop subsequent dispatch.
"""

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import threading
import time

from infinicoder.models.model_client import CompatibleClient, ProviderError


class CollaborationStopped(RuntimeError):
    """Parent cancellation, deadline, cost uncertainty or resource exhaustion."""


@dataclass(frozen=True)
class CollaborationLimits:
    max_model_calls: int
    wall_seconds: float
    final_reserve_seconds: float = 30
    max_total_tokens: int | None = None

    def __post_init__(self):
        if type(self.max_model_calls) is not int or self.max_model_calls < 3:
            raise ValueError("collaboration needs at least three shared call slots")
        for key in ("wall_seconds", "final_reserve_seconds"):
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(key + " must be finite and positive")
        if self.final_reserve_seconds >= self.wall_seconds:
            raise ValueError("final reserve must leave a coding interval")
        if self.max_total_tokens is not None and (
            type(self.max_total_tokens) is not int or self.max_total_tokens < 1
        ):
            raise ValueError("optional total token cap must be a positive integer")


class CollaborationBudget:
    """Exclusive parent owner; fixed role quotas never acquire a second budget.

    The caller must supply the real chat/schema tokenizer reservation function.
    A reservation includes maximum output tokens. Persistent journal failure is
    fatal before dispatch; resume is rejected by the collaboration entry.
    """

    def __init__(
        self,
        limits,
        journal,
        *,
        request_token_bound,
        tokenizer_identity,
        external_cancelled=None,
        request_context_limit=None,
    ):
        if not isinstance(limits, CollaborationLimits) or not callable(request_token_bound):
            raise TypeError("explicit collaboration limits and tokenizer counter required")
        if not isinstance(tokenizer_identity, str) or not tokenizer_identity.strip():
            raise ValueError("explicit tokenizer identity required")
        self.limits = limits
        self.deadline = time.monotonic() + limits.wall_seconds
        self.coding_deadline = self.deadline - limits.final_reserve_seconds
        self.request_token_bound = request_token_bound
        self.tokenizer_identity = tokenizer_identity
        if external_cancelled is not None and not callable(external_cancelled):
            raise TypeError("external cancellation must be a caller callback")
        if request_context_limit is not None and (
            type(request_context_limit) is not int or request_context_limit < 1
        ):
            raise ValueError("request context limit must be a positive integer")
        self.external_cancelled = external_cancelled
        self.request_context_limit = request_context_limit
        self.cancel_event = threading.Event()
        self.coding_cancel_event = threading.Event()
        self.coding_reason = None
        self._lock = threading.RLock()
        self._stream = Path(journal).open("x", encoding="utf-8", newline="\n")
        self._sequence = 0
        self._calls = 0
        self._known_tokens = 0
        self._reserved_tokens = 0
        self._unknown = 0
        self._inflight = {}
        self._quotas = {}
        self._role_calls = {}
        self.reason = None
        self._closed = False
        try:
            self._event("opened", limits=asdict(limits), tokenizer_identity=tokenizer_identity)
        except BaseException:
            self._stream.close()
            raise

    def _event(self, kind, **value):
        self._sequence += 1
        try:
            self._stream.write(
                json.dumps(dict(sequence=self._sequence, kind=kind, **value), allow_nan=False)
                + "\n"
            )
            self._stream.flush()
            os.fsync(self._stream.fileno())
        except Exception:
            self.reason = "budget_journal_unavailable"
            self.cancel_event.set()
            raise

    def allocate(self, quotas):
        with self._lock:
            if (
                self._quotas
                or self._calls
                or not quotas
                or any(
                    not isinstance(k, str) or not k or type(v) is not int or v < 0
                    for k, v in quotas.items()
                )
            ):
                raise ValueError("role quotas must be declared once before dispatch")
            if sum(quotas.values()) > self.limits.max_model_calls:
                raise ValueError("role quotas exceed the one parent call limit")
            self._quotas = dict(quotas)
            self._event("allocated", quotas=self._quotas)

    def stop(self, reason="cancelled"):
        with self._lock:
            if not self.cancel_event.is_set():
                self.reason = reason
                self.cancel_event.set()
                self.coding_cancel_event.set()
                self._event("stopped", reason=reason)

    def stop_coding(self, reason="coding_deadline"):
        """End search without consuming the caller's final-only reserve."""
        with self._lock:
            if not self.coding_cancel_event.is_set():
                self.coding_reason = reason
                self.coding_cancel_event.set()
                self._event("coding_stopped", reason=reason)

    def check(self, *, final=False):
        with self._lock:
            if self.external_cancelled is not None and self.external_cancelled():
                self.stop("parent_cancel")
            if self._closed or self.cancel_event.is_set():
                raise CollaborationStopped(self.reason or "budget owner closed")
            if not final and self.coding_cancel_event.is_set():
                raise CollaborationStopped(self.coding_reason or "coding_stopped")
            if time.monotonic() >= (self.deadline if final else self.coding_deadline):
                raise CollaborationStopped("parent_deadline" if final else "coding_deadline")

    def remaining(self, *, final=False):
        return max(0, (self.deadline if final else self.coding_deadline) - time.monotonic())

    def reserve(self, role, messages, schema, config):
        # Counting may be expensive; no request can start until the atomic
        # admission below rechecks cancellation, the deadline and every quota.
        bound = self.request_token_bound(messages, schema, config.max_output_tokens)
        if type(bound) is not int or not config.max_output_tokens <= bound:
            raise ValueError("tokenizer must reserve input framing/schema plus maximum output")
        if self.request_context_limit is not None and bound > self.request_context_limit:
            raise CollaborationStopped("request_context_overflow")
        with self._lock:
            self.check()
            if role not in self._quotas or self._role_calls.get(role, 0) >= self._quotas[role]:
                raise CollaborationStopped("role_call_quota")
            if self._calls >= self.limits.max_model_calls:
                raise CollaborationStopped("parent_call_quota")
            if self.limits.max_total_tokens is not None and (
                self._known_tokens + self._reserved_tokens + bound > self.limits.max_total_tokens
            ):
                raise CollaborationStopped("parent_token_reservation")
            self._calls += 1
            ordinal = self._calls
            self._role_calls[role] = self._role_calls.get(role, 0) + 1
            self._reserved_tokens += bound
            self._inflight[ordinal] = dict(role=role, tokens=bound)
            try:
                self._event("request_reserved", ordinal=ordinal, role=role, token_bound=bound)
            except BaseException:
                # Persistence failed before the client is invoked. Keep the
                # call slot spent and parent halted,but no request is in flight.
                self._reserved_tokens -= bound
                self._inflight.pop(ordinal)
                raise
            return ordinal

    def settle(self, ordinal, usage, *, error=None):
        with self._lock:
            reservation = self._inflight.pop(ordinal)
            self._reserved_tokens -= reservation["tokens"]
            valid = isinstance(usage, dict) and all(
                type(usage.get(k)) is int and usage[k] >= 0
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            )
            valid = (
                valid
                and usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
            )
            if valid:
                charged = usage["total_tokens"]
                self._known_tokens += charged
            else:
                charged = None
                self._unknown += 1
            late = (
                self.coding_cancel_event.is_set()
                or self.cancel_event.is_set()
                or time.monotonic() >= self.coding_deadline
            )
            self._event(
                "request_settled",
                ordinal=ordinal,
                role=reservation["role"],
                known_tokens=charged,
                usage_unknown=not valid,
                late=late,
                error=error,
            )
            if not valid:
                self.stop("unknown_model_usage")
            elif charged > reservation["tokens"]:
                self.stop("token_reservation_underestimated")
            elif (
                self.limits.max_total_tokens is not None
                and self._known_tokens > self.limits.max_total_tokens
            ):
                self.stop("total_token_limit_exceeded")
            self.check()

    def snapshot(self):
        with self._lock:
            return dict(
                physical_request_attempts=self._calls,
                known_tokens=self._known_tokens,
                unknown_usage_requests=self._unknown,
                reserved_tokens=self._reserved_tokens,
                inflight=len(self._inflight),
                role_calls=dict(self._role_calls),
                role_quotas=dict(self._quotas),
                stopped=self.cancel_event.is_set(),
                reason=self.reason,
                coding_stopped=self.coding_cancel_event.is_set(),
                coding_reason=self.coding_reason,
            )

    def close(self):
        with self._lock:
            if self._inflight:
                raise RuntimeError("cannot close parent budget with requests in flight")
            if not self._closed:
                try:
                    self._event("closed", totals=self.snapshot())
                finally:
                    self._closed = True
                    self._stream.close()


class BudgetedRoleClient:
    """Both public model entries are charged exactly once before transport.

    This is not a generic SDK decorator. Only the exact, non-retrying built-in
    CompatibleClient is admitted; it may construct its internal structured
    client, but still performs one HTTP request for this reserved entry.
    """

    def __init__(self, client, owner, role):
        if type(client) is not CompatibleClient:
            raise TypeError("collaboration requires the built-in single-request CompatibleClient")
        if (
            type(client.config.extra_body.get("n", 1)) is not int
            or client.config.extra_body.get("n", 1) != 1
        ):
            raise ValueError("collaboration does not admit multi-sample HTTP requests")
        self._client, self.owner, self.role = client, owner, role
        self.config = client.config

    @property
    def supports_structured_output(self):
        return self._client.supports_structured_output

    def _call(self, messages, timeout, schema=None):
        try:
            ordinal = self.owner.reserve(self.role, messages, schema, self.config)
        except CollaborationStopped as exc:
            raise ProviderError(None, "collaboration_" + str(exc), False) from None
        try:
            timeout = min(timeout, self.owner.remaining())
            if timeout <= 0:
                raise TimeoutError("coding interval expired after reservation")
            response = (
                self._client.generate_structured(messages, schema, timeout)
                if schema is not None
                else self._client.generate(messages, timeout)
            )
        except BaseException as exc:
            try:
                self.owner.settle(ordinal, None, error=type(exc).__name__)
            except CollaborationStopped:
                pass
            raise
        try:
            self.owner.settle(ordinal, response.usage)
        except CollaborationStopped as exc:
            # No late/unknown-cost proposal reaches Runtime's editing path.
            raise ProviderError(None, "collaboration_" + str(exc), False) from None
        return response

    def generate(self, messages, timeout):
        return self._call(messages, timeout)

    def generate_structured(self, messages, schema, timeout):
        if not self.supports_structured_output or not isinstance(schema, dict):
            raise ValueError("explicit supported structured schema required")
        return self._call(messages, timeout, schema)
