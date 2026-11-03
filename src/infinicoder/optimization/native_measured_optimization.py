"""Bounded public measurement of actual versions and hash-guarded best restore."""

from contextlib import contextmanager
from dataclasses import dataclass, asdict
import math, time
from pathlib import Path

from infinicoder.runtime.agent_execution import CallerPublicOperation
from infinicoder.memory.native_memory import _json, _sha
from infinicoder.memory.source.native_reversible_working_set import NativeWorkingSet
from infinicoder.optimization.optimization import Measurement
from infinicoder.execution.workspace_tools import Workspace


@dataclass(frozen=True)
class PublicPerformanceSample:
    source_hashes: dict
    correct: bool
    independent_checks: int
    workload_identity: str
    measurement: Measurement | None = None
    warmup_count: int = 0
    evidence_ref: str = ""
    failure: dict | None = None
    memory_observed: bool = False

    def __post_init__(self):
        if (
            type(self.correct) is not bool
            or type(self.independent_checks) is not int
            or self.independent_checks < 0
            or self.correct
            and self.independent_checks < 1
        ):
            raise ValueError(
                "typed external correctness;positive checks required for a correct version"
            )
        if (
            not isinstance(self.workload_identity, str)
            or not self.workload_identity
            or not self.evidence_ref
        ):
            raise ValueError("frozen public workload and independent evidence reference required")
        if not self.source_hashes or any(
            not isinstance(p, str)
            or not isinstance(h, str)
            or len(h) != 64
            or any(c not in "0123456789abcdef" for c in h)
            for p, h in self.source_hashes.items()
        ):
            raise ValueError("complete observed source hashes required")
        if self.correct:
            if (
                not isinstance(self.measurement, Measurement)
                or type(self.warmup_count) is not int
                or self.warmup_count < 3
            ):
                raise ValueError(
                    "correct versions need at least3 warmups and5 external timing samples"
                )
        elif self.measurement is not None:
            raise ValueError("incorrect candidate timing cannot become performance evidence")
        if type(self.memory_observed) is not bool:
            raise ValueError("explicit memory observation status required")
        if self.failure is not None and (
            not isinstance(self.failure, dict) or len(_json(self.failure).encode()) > 4096
        ):
            raise ValueError("bounded public failure observation required")


@dataclass(frozen=True)
class PublicPerformancePolicy:
    operation: CallerPublicOperation
    rollback_paths: tuple
    exclusive_workspace: bool
    max_candidates: int = 2
    final_reserve_seconds: float = 30
    min_speedup: float = 1.02

    def __post_init__(self):
        if (
            not isinstance(self.operation, CallerPublicOperation)
            or self.operation.source_mutation_allowed
        ):
            raise ValueError("identified source-preserving public operation required")
        if (
            self.exclusive_workspace is not True
            or not isinstance(self.rollback_paths, tuple)
            or not self.rollback_paths
            or len(set(self.rollback_paths)) != len(self.rollback_paths)
        ):
            raise ValueError("explicit exclusive ownership and writable rollback paths required")
        if len(self.operation.observed_paths) > 16 or not set(self.rollback_paths).issubset(
            self.operation.observed_paths
        ):
            raise ValueError(
                "at most16 observed sources;rollback permission separate from read scope"
            )
        if type(self.max_candidates) is not int or not 1 <= self.max_candidates <= 2:
            raise ValueError("at most2 newly produced candidate versions")
        if (
            not math.isfinite(self.min_speedup)
            or self.min_speedup <= 1
            or not math.isfinite(self.final_reserve_seconds)
            or self.final_reserve_seconds < 5
        ):
            raise ValueError("positive performance threshold and final reserve required")


class PublicPerformanceController:
    """No model/executor of its own. The caller operation owns isolated timing."""

    def __init__(self, policy, sources):
        if not isinstance(policy, PublicPerformancePolicy):
            raise TypeError("explicit performance policy required")
        self.policy, self.sources = policy, sources
        self.best = None
        self.baseline = None
        self.records = []
        self.measured = {}
        self.candidates = 0
        self.initialized = False
        self.disabled = False
        self.last_sdk_sources = None
        self.final_source_binding_verified = False

    @contextmanager
    def bind(self, request, owner, deadline):
        self.owner, self.deadline, self.request = owner, deadline, request
        self.sources.validate(request.workspace, self.policy.operation.observed_paths)
        self.ws = Workspace(request.workspace, allowed_paths=self.policy.rollback_paths)
        self.output = Path(request.state_dir) / "native-performance.json"
        with self.policy.operation.bind(request, deadline) as lease:
            self.lease = lease
            try:
                yield self
            finally:
                # No new measurement after the run. Only restore an exact
                # attributed,unmeasured SDK version when a known best exists.
                try:
                    if self.best is not None:
                        current = self._snapshot()
                        if (
                            self.last_sdk_sources is not None
                            and current == self.last_sdk_sources
                            and self._pins(current) != self.best["pins"]
                        ):
                            self._restore(current, "final_unmeasured_version")
                        self.final_source_binding_verified = (
                            self.best is not None
                            and self._pins(self._snapshot()) == self.best["pins"]
                        )
                except (ValueError, OSError, TimeoutError) as error:
                    self.records.append(
                        dict(
                            status="cleanup_source_binding_unknown", error_type=type(error).__name__
                        )
                    )
                self._save()

    def _snapshot(self):
        values = {}
        for path in self.policy.operation.observed_paths:
            raw, text, pin = self.sources.read(path)
            values[path] = dict(raw=raw, text=text, sha256=pin)
        return values

    @staticmethod
    def _pins(snapshot):
        return {p: v["sha256"] for p, v in snapshot.items()}

    def _save(self):
        data = dict(
            policy=self.policy.operation.manifest(),
            rollback_paths=self.policy.rollback_paths,
            exclusive_workspace=True,
            candidate_limit=self.policy.max_candidates,
            produced_versions=self.candidates,
            best={k: v for k, v in self.best.items() if k != "sources"} if self.best else None,
            records=self.records,
            disabled=self.disabled,
            private_final_used=False,
            extra_model_calls=0,
            final_source_binding_verified=self.final_source_binding_verified,
            coverage="Declared observed files/public workload only;not repository closure or final acceptance",
        )
        self.output.write_text(_json(data), encoding="utf8")

    def source_changed(self):
        try:
            self.last_sdk_sources = self._snapshot()
        except FileNotFoundError:
            self.last_sdk_sources = None

    def _restore(self, current, reason):
        if self.best is None:
            return
        # A non-writable dependency change invalidates the old best;do not
        # restore an apparently good target into a different dependency state.
        if any(
            current[p]["sha256"] != pin
            for p, pin in self.best["pins"].items()
            if p not in self.policy.rollback_paths
        ):
            self.disabled = True
            self.records.append(dict(status="best_invalidated_by_dependency_change", reason=reason))
            self.best = None
            return
        if self._pins(self._snapshot()) != self._pins(current):
            raise ValueError("concurrent source change prevents best restore")
        restored = []
        for path in self.policy.rollback_paths:
            if current[path]["sha256"] == self.best["pins"][path]:
                continue
            target = self.ws.path(path, write=True)
            raw = self.best["sources"][path]["raw"]
            self.ws._atomic_replace(target, raw, current[path]["sha256"], current[path]["sha256"])
            restored.append(path)
        self.records.append(
            dict(
                status="restored_best_public_correct",
                reason=reason,
                paths=restored,
                source_hashes=self.best["pins"],
                authority="This public workload only;independent final still required",
            )
        )
        self.owner._save_event(
            "controller_best_restore",
            dict(
                paths=restored,
                source_hashes=self.best["pins"],
                provenance="explicit_public_performance_policy;not_an_SDK_edit",
                reason=reason,
            ),
        )

    def before_query(self):
        if self.disabled:
            return
        self.owner._namespace()
        try:
            snapshot = self._snapshot()
        except FileNotFoundError:
            self.initialized = True
            self.records.append(
                dict(status="incomplete_source_set;wait_for_complete_attributed_version")
            )
            self._save()
            return
        pins = self._pins(snapshot)
        key = _sha(_json(pins))
        if self.best is not None and any(
            pins[p] != h
            for p, h in self.best["pins"].items()
            if p not in self.policy.rollback_paths
        ):
            self._restore(snapshot, "dependency_changed_before_measurement")
            self._save()
            return
        if key in self.measured:
            if self.best is not None and pins != self.best["pins"]:
                if self.last_sdk_sources is not None and self._pins(self.last_sdk_sources) == pins:
                    self._restore(snapshot, "cached_non_best_version;no_remeasurement")
                else:
                    self.disabled = True
                    self.records.append(
                        dict(status="cached_version_unattributed;no_restore", source_hashes=pins)
                    )
                self._save()
            return
        initial = not self.initialized
        if not initial and self.last_sdk_sources is None:
            self.disabled = True
            self.records.append(dict(status="unknown_source_change;no_attributed_candidate"))
            self._save()
            return
        if not initial and self._pins(self.last_sdk_sources) != pins:
            self.disabled = True
            self.records.append(dict(status="unattributed_or_concurrent_source_change"))
            self._save()
            return
        if not initial and self.candidates >= self.policy.max_candidates:
            self.records.append(
                dict(status="candidate_budget_exhausted;version_not_measured", source_hashes=pins)
            )
            self._restore(snapshot, "candidate_budget_exhausted")
            self._save()
            return
        if self.deadline - time.monotonic() <= self.policy.final_reserve_seconds + 1:
            self.records.append(dict(status="final_reserve;no_measurement", source_hashes=pins))
            self._restore(snapshot, "final_reserve")
            self._save()
            return
        if not initial:
            self.candidates += 1
        self.initialized = True
        operation = self.policy.operation
        started = time.monotonic()
        dispatch = dict(
            kind="public_performance",
            implementation_identity=operation.implementation_identity,
            scope=operation.public_scope,
            coverage=operation.public_coverage,
            candidate=0 if initial else self.candidates,
        )
        before = self.owner._snapshot()
        try:
            with self.owner.controller_public_dispatch(dispatch):
                sample = self.lease.invoke(
                    min(
                        self.deadline - self.policy.final_reserve_seconds,
                        time.monotonic() + operation.timeout_seconds,
                    )
                )
            after = self._snapshot()
            if self._pins(after) != pins:
                raise ValueError("measurement changed candidate source")
            if not isinstance(sample, PublicPerformanceSample) or sample.source_hashes != pins:
                raise ValueError("typed public external sample must bind every observed source")
            if (
                self.baseline is not None
                and sample.workload_identity != self.baseline["workload_identity"]
            ):
                raise ValueError("public performance workload changed")
            report = asdict(sample)
            report["measurement"] = sample.measurement.report() if sample.measurement else None
            if report["measurement"] is not None and not sample.memory_observed:
                report["measurement"]["peak_memory_bytes"] = None
            self.owner.record_controller_return(
                dispatch, before=before, after=self.owner._snapshot(), report=report
            )
            record = dict(
                status="measured_public_version",
                initial=initial,
                source_hashes=pins,
                correct=sample.correct,
                workload_identity=sample.workload_identity,
                measurement=report["measurement"],
                independent_checks=sample.independent_checks,
                evidence_ref=sample.evidence_ref,
                wall_seconds=time.monotonic() - started,
            )
            record["public_failure"] = sample.failure
            self.records.append(record)
            self.measured[key] = record
            if sample.correct:
                candidate = dict(
                    pins=pins,
                    sources=snapshot,
                    measurement=report["measurement"],
                    workload_identity=sample.workload_identity,
                )
                if self.baseline is None:
                    self.baseline = dict(candidate)
                if (
                    self.best is None
                    or self.best["measurement"]["median_ms"] / candidate["measurement"]["median_ms"]
                    >= self.policy.min_speedup
                ):
                    self.best = candidate
                    record["retained"] = True
                else:
                    record["retained"] = False
                    self._restore(snapshot, "no_confirmed_improvement")
            else:
                self._restore(snapshot, "public_correctness_failed")
        except Exception as error:
            self.records.append(
                dict(
                    status="measurement_unavailable",
                    error_type=type(error).__name__,
                    source_hashes=pins,
                    wall_seconds=time.monotonic() - started,
                )
            )
            self.disabled = True
            # Never overwrite a concurrent change discovered by a measurement.
            if self._pins(self._snapshot()) == pins:
                self._restore(snapshot, "measurement_unavailable")
        finally:
            self._save()

    def view(self):
        best = self.best
        return dict(
            layer="public_performance_feedback",
            produced_versions=self.candidates,
            max_candidates=self.policy.max_candidates,
            latest=self.records[-2:],
            best_source_hashes=best["pins"] if best else None,
            best_median_ms=best["measurement"]["median_ms"] if best else None,
            speedup_vs_first_correct=self.baseline["measurement"]["median_ms"]
            / best["measurement"]["median_ms"]
            if best and self.baseline
            else None,
            next="After correctness,produce a genuinely different optimization candidate if budget remains;inspect bottlenecks and recheck semantics. Repeated identical bytes reuse their measurement.",
            authority="Fixed public workload and observed sources only;diagnostic latency is not final acceptance",
        )


class NativeMeasuredOptimization(NativeWorkingSet):
    identity = "native_measured_optimization_v1"

    def __init__(self, *args, performance_policy, **kwargs):
        super().__init__(*args, **kwargs)
        self.performance = PublicPerformanceController(performance_policy, self.sources)

    def validate(self, request):
        super().validate(request)
        if not set(self.performance.policy.operation.observed_paths).issubset(self.sources.paths):
            raise ValueError("public performance source coverage unavailable")

    @contextmanager
    def bind(self, request, model, environment, deadline):
        with super().bind(request, model, environment, deadline):
            with self.performance.bind(request, self, deadline):
                yield self

    def execution_completed(self, event):
        super().execution_completed(event)
        if set(event.get("changed", [])).intersection(
            self.performance.policy.operation.observed_paths
        ):
            self.performance.source_changed()

    def before_query(self, messages):
        super().before_query(messages)
        self.performance.before_query()

    def project(self, messages):
        packet = super().project(messages)
        extra = dict(role="user", content=_json(self.performance.view()))
        return [*packet, extra] if self._count([*packet, extra]) <= self.limit else packet
