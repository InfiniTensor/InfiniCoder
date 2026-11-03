"""One logical SDK query,one absolute wait,and explicit physical outcomes.

The policy never rewrites messages,executes model actions,or retries an unknown
upstream result. It is opt-in and does not replace the SDK's agent loop.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import copy
import math
import signal
import sys
import threading
import time
import uuid


class NativeDeliveryExhausted(Exception):
    """Ordinary failure:preserve the candidate and run original final once."""


class ConfirmedNotDispatched(Exception):
    """A trusted relay component failed before it attempted upstream dispatch."""


@dataclass(frozen=True)
class DeliveryPolicy:
    max_wait_seconds: float = 600
    max_confirmed_not_dispatched_retries: int = 1
    retry_after_unknown: bool = False

    def __post_init__(self):
        if (
            isinstance(self.max_wait_seconds, bool)
            or not isinstance(self.max_wait_seconds, (int, float))
            or not math.isfinite(self.max_wait_seconds)
            or self.max_wait_seconds <= 0
        ):
            raise ValueError("finite positive logical query wait required")
        if (
            type(self.max_confirmed_not_dispatched_retries) is not int
            or not 0 <= self.max_confirmed_not_dispatched_retries <= 1
        ):
            raise ValueError("at most one confirmed-not-dispatched retry")
        if self.retry_after_unknown is not False:
            raise ValueError("unknown upstream results cannot be retried")


TERMINAL = {
    "delivered",
    "unknown",
    "expired",
    "task_expired",
    "not_dispatched_exhausted",
    "rejected",
    "context_rejected",
    "cancelled_before_dispatch",
}


class QueryDelivery:
    identity = "native_delivery_policy_v1"

    def __init__(
        self,
        policy,
        task_deadline,
        *,
        lock=None,
        save=None,
        phase_deadline=None,
        clock=time.monotonic,
    ):
        if not isinstance(policy, DeliveryPolicy):
            raise TypeError("typed delivery policy required")
        self.policy = policy
        self.task_deadline = task_deadline
        self.lock = lock or threading.RLock()
        self.save = save or (lambda: None)
        self.phase_deadline = phase_deadline or (lambda: task_deadline)
        self.clock = clock
        self.queries = []
        self.active = None
        self.owner = None
        self.task_expired = None

    def raise_task_expired(self, lease=None, *, forced=False):
        if forced or self.clock() >= self.task_deadline:
            if lease is not None and lease["outcome"] not in TERMINAL:
                lease.update(
                    outcome="task_expired",
                    usage_unknown=any(a.get("dispatched") for a in lease["attempts"]),
                )
                self.save()
            # Relay threads only record outcomes;the Core's original exception
            # is raised in the owning main thread,never in an HTTP worker.
            if (
                callable(self.task_expired)
                and threading.current_thread() is threading.main_thread()
            ):
                self.task_expired()

    def snapshot(self):
        with self.lock:
            return dict(
                identity=self.identity,
                policy=asdict(self.policy),
                queries=copy.deepcopy(self.queries),
            )

    def start(self):
        with self.lock:
            self.raise_task_expired()
            if self.active is not None:
                raise NativeDeliveryExhausted("logical model queries cannot nest")
            tick = self.clock()
            cutoff = min(
                self.task_deadline, self.phase_deadline(), tick + self.policy.max_wait_seconds
            )
            if cutoff <= tick:
                raise NativeDeliveryExhausted("logical query deadline exhausted before dispatch")
            lease = dict(
                query_id=uuid.uuid4().hex,
                started_monotonic=tick,
                deadline_monotonic=cutoff,
                deadline_source="task"
                if cutoff == self.task_deadline
                else "phase"
                if cutoff == self.phase_deadline()
                else "logical_wait",
                fingerprint=None,
                outcome="open",
                attempts=[],
                confirmed_not_dispatched=0,
                closed=False,
                usage_unknown=False,
                late_usage=[],
                rejections=[],
            )
            self.queries.append(lease)
            self.active = lease
            self.owner = threading.get_ident()
            self.save()
            return lease

    def lookup(self, query_id):
        with self.lock:
            return next((q for q in self.queries if q["query_id"] == query_id), None)

    def remaining(self, lease):
        with self.lock:
            self.raise_task_expired(lease)
            left = (
                min(lease["deadline_monotonic"], self.task_deadline, self.phase_deadline())
                - self.clock()
            )
            if left <= 0:
                if lease["outcome"] not in TERMINAL:
                    lease.update(
                        outcome="expired",
                        usage_unknown=any(a.get("dispatched") for a in lease["attempts"]),
                    )
                self.save()
                raise NativeDeliveryExhausted("absolute logical query deadline exhausted")
            return left

    def accept(self, lease):
        with self.lock:
            if lease is None:
                return False
            if lease is not self.active or lease["closed"] or lease["outcome"] in TERMINAL:
                lease["rejections"].append(
                    dict(reason="terminal_or_inactive_query", at=self.clock())
                )
                self.save()
                return False
            try:
                self.remaining(lease)
            except NativeDeliveryExhausted:
                return False
            return True

    def begin_attempt(self, lease, fingerprint, attempt):
        with self.lock:
            if not self.accept(lease):
                raise NativeDeliveryExhausted("terminal logical query cannot dispatch")
            if lease["fingerprint"] not in (None, fingerprint):
                lease["outcome"] = "rejected"
                self.save()
                raise NativeDeliveryExhausted("request changed inside one logical query")
            if lease["outcome"] == "pending":
                raise NativeDeliveryExhausted("concurrent duplicate logical request")
            lease["fingerprint"] = fingerprint
            lease["outcome"] = "pending"
            lease["attempts"].append(
                dict(
                    attempt=attempt,
                    started_monotonic=self.clock(),
                    dispatched=False,
                    outcome="pending",
                )
            )
            self.save()

    def dispatched(self, lease):
        with self.lock:
            if not self.accept(lease):
                raise NativeDeliveryExhausted("logical query revoked before upstream dispatch")
            lease["attempts"][-1]["dispatched"] = True
            self.save()

    def not_dispatched(self, lease, reason):
        with self.lock:
            if lease["outcome"] in TERMINAL:
                return
            if lease["attempts"] and lease["attempts"][-1]["dispatched"]:
                return self.unknown(lease, "cannot confirm no upstream dispatch")
            lease["confirmed_not_dispatched"] += 1
            lease["outcome"] = (
                "confirmed_not_dispatched_retry"
                if lease["confirmed_not_dispatched"]
                <= self.policy.max_confirmed_not_dispatched_retries
                else "not_dispatched_exhausted"
            )
            if lease["attempts"]:
                lease["attempts"][-1].update(outcome="confirmed_not_dispatched", reason=reason)
            if self.clock() >= lease["deadline_monotonic"]:
                lease["outcome"] = "expired"
            self.save()

    def unknown(self, lease, reason):
        with self.lock:
            lease.update(
                outcome="unknown",
                usage_unknown=not any(isinstance(a.get("usage"), dict) for a in lease["attempts"]),
            )
            if lease["attempts"]:
                lease["attempts"][-1].update(outcome="unknown", reason=reason)
            self.save()

    def rejected(self, lease, *, context=False):
        with self.lock:
            if lease["outcome"] in TERMINAL:
                return
            lease["outcome"] = "context_rejected" if context else "rejected"
            if lease["attempts"]:
                lease["attempts"][-1]["outcome"] = lease["outcome"]
            self.save()

    def delivered(self, lease, usage=None):
        with self.lock:
            # Never revive a cancelled/expired/unknown query with a late action.
            if (
                lease["closed"]
                or lease["outcome"] in TERMINAL
                or self.clock() >= lease["deadline_monotonic"]
            ):
                if isinstance(usage, dict):
                    lease["late_usage"].append(copy.deepcopy(usage))
                    lease["usage_unknown"] = False
                if lease["outcome"] not in TERMINAL:
                    lease.update(outcome="unknown", usage_unknown=not isinstance(usage, dict))
                self.save()
                return False
            lease["outcome"] = "delivered"
            lease["usage_unknown"] = not isinstance(usage, dict)
            if lease["attempts"]:
                lease["attempts"][-1].update(outcome="delivered", usage=copy.deepcopy(usage))
            self.save()
            return True

    def finish(self, lease, error=None):
        with self.lock:
            if lease["outcome"] not in TERMINAL:
                dispatched = any(a["dispatched"] for a in lease["attempts"])
                lease.update(
                    outcome="unknown" if dispatched else "cancelled_before_dispatch",
                    usage_unknown=dispatched,
                )
            lease["closed"] = True
            if error is not None:
                lease["caller_exit_type"] = type(error).__name__
            if self.active is lease:
                self.active = None
                self.owner = None
            self.save()

    @contextmanager
    def query_alarm(self, lease):
        # The supported Native core is a Linux main-thread process. The alarm
        # also bounds SDK backoff,DNS and a slow/trickling response body.
        if sys.platform != "linux" or threading.current_thread() is not threading.main_thread():
            raise NativeDeliveryExhausted(
                "logical delivery requires the Linux main-thread Native core"
            )
        old_handler = signal.getsignal(signal.SIGALRM)
        old_timer = signal.getitimer(signal.ITIMER_REAL)
        tick = self.clock()
        left = self.remaining(lease)

        def expired(signum, frame):
            self.raise_task_expired(lease, forced=lease["deadline_source"] == "task")
            raise NativeDeliveryExhausted(
                "absolute logical query wait exhausted,including SDK retry backoff"
            )

        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, min(left, old_timer[0]) if old_timer[0] else left)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
            if old_timer[0]:
                signal.setitimer(
                    signal.ITIMER_REAL,
                    max(0.001, old_timer[0] - (self.clock() - tick)),
                    old_timer[1],
                )

    @contextmanager
    def bind_model(self, model, *, task_expired=None):
        """Instance-local wrappers;the actual SDK still parses and runs actions."""
        original_query = model.query
        original_attempt = model._query
        original_abort = model.abort_exceptions
        self.task_expired = task_expired
        prior = {name: model.__dict__.get(name) for name in ("query", "_query", "abort_exceptions")}
        present = set(model.__dict__) & set(prior)
        original_abort_types = tuple(original_abort)

        def attempt(messages, **kwargs):
            lease = self.active
            if lease is None or threading.get_ident() != self.owner:
                raise NativeDeliveryExhausted("no calling-thread logical query lease")
            if lease["outcome"] in TERMINAL:
                raise NativeDeliveryExhausted("logical query is already terminal")
            headers = dict(kwargs.get("extra_headers") or {})
            if "X-InfiniCoder-Query-Id" in headers:
                raise NativeDeliveryExhausted("reserved local query header")
            headers["X-InfiniCoder-Query-Id"] = lease["query_id"]
            kwargs = dict(kwargs, extra_headers=headers, timeout=self.remaining(lease))
            try:
                result = original_attempt(messages, **kwargs)
            except Exception as error:
                self.raise_task_expired(lease, forced=lease["outcome"] == "task_expired")
                # Keep existing non-retryable authentication/context classes.
                if isinstance(error, original_abort_types):
                    raise
                with self.lock:
                    state = lease["outcome"]
                    if state in ("open", "pending", "delivered"):
                        self.unknown(lease, "client exception without a complete SDK result")
                        state = lease["outcome"]
                if state == "confirmed_not_dispatched_retry":
                    self.remaining(lease)
                    raise ConfirmedNotDispatched("relay confirmed no upstream dispatch") from error
                raise NativeDeliveryExhausted("logical query ended: " + state) from error
            if lease["outcome"] != "delivered":
                raise NativeDeliveryExhausted(
                    "complete response arrived after the logical query ended"
                )
            return result

        def query(*args, **kwargs):
            lease = self.start()
            error = None
            try:
                with self.query_alarm(lease):
                    return original_query(*args, **kwargs)
            except BaseException as caught:
                error = caught
                raise
            finally:
                self.finish(lease, error)

        model.query = query
        model._query = attempt
        model.abort_exceptions = [*original_abort, NativeDeliveryExhausted]
        try:
            yield
        finally:
            self.task_expired = None
            for name in prior:
                if name in present:
                    setattr(model, name, prior[name])
                else:
                    delattr(model, name)
