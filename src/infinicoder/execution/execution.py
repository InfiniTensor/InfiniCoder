"""declarative execution profiles with hard resource limits.

run_process already kills process trees and bounds RETURNED output, but the
spool file itself can grow unbounded (a runaway child streaming GBs fills
the disk before anyone reads it). ExecutionProfile adds a hard spool cap
enforced DURING execution, plus caller-approved environment policy and a
session-cumulative output budget.
"""

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class ExecutionProfile:
    """Resource policy for one managed process execution."""

    max_output_bytes: int = 12000  # bytes returned to the model
    spool_cap_bytes: int = 64 * 1024 * 1024  # hard disk cap on the spool
    default_timeout: float = 60
    max_timeout: float = 120
    env_extra: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ("max_output_bytes", "spool_cap_bytes"):
            v = getattr(self, name)
            minimum = 1000 if name == "spool_cap_bytes" else 100
            if isinstance(v, bool) or not isinstance(v, int) or v < minimum:
                raise ValueError(f"ExecutionProfile.{name} must be int >= {minimum}")
        if (
            any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in (self.default_timeout, self.max_timeout)
            )
            or not 0 < self.default_timeout <= self.max_timeout
        ):
            raise ValueError("ExecutionProfile: default_timeout must be within max_timeout")
        for k, v in (self.env_extra or {}).items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise ValueError("ExecutionProfile.env_extra must map str->str")

    def clamp_timeout(self, requested):
        try:
            value = float(requested)
        except (TypeError, ValueError):
            value = self.default_timeout
        if value != value or value == float("inf"):  # NaN/inf guard
            value = self.default_timeout
        return max(0.1, min(value, self.max_timeout))


class OutputBudget:
    """Session-cumulative spool accounting across many executions."""

    def __init__(self, total_bytes=512 * 1024 * 1024):
        if isinstance(total_bytes, bool) or total_bytes < 1000:
            raise ValueError("total_bytes must be a sane positive int")
        self.total = total_bytes
        self.used = 0

    def charge(self, nbytes):
        self.used += max(0, int(nbytes))
        return self.used < self.total

    def remaining_bytes(self):
        return max(0, self.total - self.used)
