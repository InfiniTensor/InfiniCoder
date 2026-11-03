"""Opt-in schema for an identified external CPU public measurement script."""

from dataclasses import dataclass
import hashlib, json

from infinicoder.optimization.native_measured_optimization import (
    PublicPerformanceSample,
    PublicPerformancePolicy,
)
from infinicoder.optimization.optimization import Measurement


@dataclass(frozen=True)
class PublicPerformanceReport:
    workload_identity: str
    rollback_paths: tuple
    max_candidates: int = 2
    final_reserve_seconds: float = 30
    min_speedup: float = 1.02

    def __post_init__(self):
        if not isinstance(self.workload_identity, str) or not self.workload_identity.strip():
            raise ValueError("explicit fixed public workload required")
        if not isinstance(self.rollback_paths, tuple) or not self.rollback_paths:
            raise ValueError("explicit rollback paths required")

    def policy(self, operation, exclusive_workspace):
        return PublicPerformancePolicy(
            operation,
            self.rollback_paths,
            exclusive_workspace,
            self.max_candidates,
            self.final_reserve_seconds,
            self.min_speedup,
        )

    def decoder(self, check):
        script_sha = check.operation.implementation_identity.get("script_sha256")
        if not script_sha:
            raise ValueError("standard report decoding requires a frozen caller public script")

        def decode(raw):
            if (
                not isinstance(raw, dict)
                or raw.get("independent_final_used") is not False
                or raw.get("public_script_sha256") != script_sha
            ):
                raise ValueError("original external public script binding differs")
            process = raw["process"]
            if process.get("timed_out") or process.get("truncated"):
                raise ValueError("public measurement did not finish with complete output")
            report = json.loads(process["output"])
            sample = report["performance_sample"]
            if (
                not isinstance(sample, dict)
                or sample.get("workload_identity") != self.workload_identity
            ):
                raise ValueError("fixed public performance workload differs")
            correct = check.decoder(raw).status == "public_pass"
            if sample.get("correct") is not correct:
                raise ValueError("public timing sample contradicts original public correctness")
            if sample.get("memory_observed") is True and "peak_memory_bytes" not in sample:
                raise ValueError("observed memory needs an explicit external value")
            measurement = (
                Measurement(tuple(sample["samples_ms"]), sample.get("peak_memory_bytes", 0))
                if correct
                else None
            )
            return PublicPerformanceSample(
                sample["source_hashes"],
                correct,
                sample["independent_checks"],
                self.workload_identity,
                measurement,
                sample.get("warmup_count", 0),
                "caller-script:"
                + script_sha
                + ":report:"
                + hashlib.sha256(process["output"].encode()).hexdigest(),
                sample.get("failure"),
                sample.get("memory_observed", False),
            )

        identity = (
            "frozen_CPU_public_performance_report_v1:"
            + hashlib.sha256((script_sha + "\n" + self.workload_identity).encode()).hexdigest()
        )
        return decode, identity
