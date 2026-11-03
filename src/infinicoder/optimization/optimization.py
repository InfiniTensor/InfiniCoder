"""Independent correctness and repeated timing gates for bounded optimization."""

from dataclasses import dataclass
import math
import statistics
import itertools
from infinicoder.execution.workspace_tools import Workspace
import hashlib


@dataclass(frozen=True)
class Measurement:
    samples_ms: tuple
    peak_memory_bytes: int = 0

    def __post_init__(self):
        if len(self.samples_ms) < 5 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
            for v in self.samples_ms
        ):
            raise ValueError("at least five finite positive external timing samples required")
        if type(self.peak_memory_bytes) is not int or self.peak_memory_bytes < 0:
            raise ValueError("invalid peak memory")

    @property
    def median_ms(self):
        return statistics.median(self.samples_ms)

    def report(self):
        values = sorted(self.samples_ms)
        return dict(
            median_ms=self.median_ms,
            min_ms=values[0],
            max_ms=values[-1],
            samples_ms=list(self.samples_ms),
            peak_memory_bytes=self.peak_memory_bytes,
        )


@dataclass(frozen=True)
class OptimizationObjective:
    """Explicit trusted executor for a closed, initially correct file set.

    measure(workspace, remaining_seconds) must return Measurement, including
    synchronization/warmup in its own external executor. identity binds its
    version/hardware contract across resume. Candidate stdout is never a metric.
    """

    paths: tuple
    measure: object
    identity: str
    min_speedup: float = 1.1

    def __post_init__(self):
        if (
            not self.paths
            or len(set(self.paths)) != len(self.paths)
            or not all(isinstance(p, str) and p for p in self.paths)
        ):
            raise ValueError("explicit unique optimization paths required")
        if not callable(self.measure) or not isinstance(self.identity, str) or not self.identity:
            raise ValueError("trusted measurement callback and identity required")
        if (
            isinstance(self.min_speedup, bool)
            or not isinstance(self.min_speedup, (int, float))
            or not math.isfinite(self.min_speedup)
            or self.min_speedup <= 1
        ):
            raise ValueError("optimization speedup must exceed one")


class OptimizationSession:
    """Persistent best-correct checkpoint used by Runtime's existing edit loop."""

    def __init__(self, objective, workspace, state, store):
        self.objective, self.ws, self.state, self.store = objective, workspace, state, store
        self.contract = dict(
            paths=list(objective.paths),
            identity=objective.identity,
            min_speedup=objective.min_speedup,
        )
        saved = state.get("optimization")
        if saved and saved["contract"] != self.contract:
            raise ValueError("optimization contract changed across resume")

    def sources(self):
        return {
            p: self.ws.path(p, write=True).read_bytes().decode("utf-8")
            if self.ws.path(p).is_file()
            else None
            for p in self.objective.paths
        }

    def measure(self, remaining):
        before = self.sources()
        if remaining() <= 0:
            raise ValueError("no optimization measurement budget remains")
        result = self.objective.measure(self.ws.root, remaining())
        if before != self.sources():
            raise ValueError("measurement changed candidate source;result rejected")
        if not isinstance(result, Measurement):
            raise TypeError("trusted measure must return Measurement")
        if remaining() <= 0:
            raise ValueError("optimization measurement exceeded task deadline")
        return result.report()

    def initialize(self, remaining):
        if self.state.get("optimization"):
            return
        sources = self.sources()
        if any(source is None for source in sources.values()):
            raise ValueError("optimization requires existing baseline files")
        measured = self.measure(remaining)
        self.state["optimization"] = dict(
            contract=self.contract,
            baseline=measured,
            best=measured,
            best_sources=sources,
            candidates=[],
        )
        self.store.event(
            "optimization_baseline", dict(identity=self.objective.identity, measurement=measured)
        )

    def consider(self, correct, captured, remaining):
        saved = self.state["optimization"]
        row = dict(
            correct=bool(correct),
            source_hashes={
                p: hashlib.sha256(s.encode()).hexdigest() if s is not None else None
                for p, s in captured.items()
            },
        )
        retained = False
        try:
            if self.sources() != captured:
                raise ValueError(
                    "candidate source changed during verification;no rollback over concurrent edits"
                )
            if correct:
                if captured == saved["best_sources"]:
                    # A finish/reverification of identical source is not a new
                    # optimization. Remeasuring until timing noise looks faster
                    # could otherwise manufacture a gain without a code change.
                    row.update(
                        reverified_best=True, measurement_reused=True, measurement=saved["best"]
                    )
                else:
                    measured = self.measure(remaining)
                    row["measurement"] = measured
                    retained = measured["median_ms"] < saved["best"]["median_ms"]
                    if retained:
                        saved.update(best=measured, best_sources=captured)
        finally:
            row["retained"] = retained
            saved["candidates"].append(row)
            # Hash guard every restore. A user edit made during a check is
            # preserved and reported instead of being overwritten.
            if not retained:
                current = self.sources()
                if current != captured:
                    raise ValueError("concurrent source change prevents optimization rollback")
                for path, source in saved["best_sources"].items():
                    old = current[path]
                    if old != source:
                        if old is None:
                            self.ws.edit(path=path, expected_hash=None, content=source)
                        else:
                            self.ws.edit(
                                path=path,
                                expected_hash=hashlib.sha256(old.encode()).hexdigest(),
                                old=old,
                                new=source,
                            )
            row["speedup_vs_baseline"] = saved["baseline"]["median_ms"] / saved["best"]["median_ms"]
            self.store.event("optimization_candidate", row)
        speedup = saved["baseline"]["median_ms"] / saved["best"]["median_ms"]
        accepted = (
            correct
            and (retained or captured == saved["best_sources"])
            and speedup >= self.objective.min_speedup
        )
        return accepted, dict(
            check_id="optimization",
            accepted=accepted,
            speedup=speedup,
            required_speedup=self.objective.min_speedup,
            best=saved["best"],
            candidate=row,
            retained_best_correct=True,
            error=""
            if accepted
            else "Independent optimization goal not yet verified for this proposal;best correct source retained. Speedup="
            + str(speedup),
        )


def optimize_file(workspace, path, proposals, verify, measure, max_candidates=8, min_speedup=1.02):
    """Callbacks are caller-owned external executors, not generated programs' logs.

    Verify must return accepted=True plus a positive independent check count.
    measure must return Measurement. Every failed proposal is hash-guard restored.
    The caller must reserve an exclusive device and provide warmup/synchronization.
    """
    if type(max_candidates) is not int or not 1 <= max_candidates <= 100:
        raise ValueError("invalid candidate budget")
    if not math.isfinite(min_speedup) or min_speedup <= 1:
        raise ValueError("speedup threshold must exceed one")
    ws = workspace if isinstance(workspace, Workspace) else Workspace(workspace, [path])

    def correct():
        result = verify(ws.root)
        return (
            isinstance(result, dict)
            and result.get("accepted") is True
            and type(result.get("checks_run")) is int
            and result["checks_run"] > 0
        )

    if not correct():
        raise ValueError("a correct independently verified baseline is required")
    best = measure(ws.root)
    if not isinstance(best, Measurement):
        raise TypeError("measure must return Measurement")
    baseline = best.report()
    history = []
    for i, source in enumerate(itertools.islice(proposals, max_candidates)):
        if not isinstance(source, str) or not source:
            raise ValueError("proposal must contain complete source")
        before = ws.read(path)
        old = ws.path(path).read_bytes().decode("utf-8")
        edit = ws.edit(path=path, expected_hash=before["sha256"], old=old, new=source)
        keep = False
        row = {"candidate": i, "sha256": edit["after"]}
        try:
            row["correct"] = correct()
            if row["correct"]:
                timed = measure(ws.root)
                if not isinstance(timed, Measurement):
                    raise TypeError("measure must return Measurement")
                row["measurement"] = timed.report()
                row["speedup_vs_best"] = best.median_ms / timed.median_ms
                keep = row["speedup_vs_best"] >= min_speedup
                if keep:
                    best = timed
        finally:
            row["retained"] = keep
            if not keep:
                ws.edit(path=path, expected_hash=edit["after"], old=source, new=old)
            history.append(row)
    return dict(
        baseline=baseline,
        best=best.report(),
        candidates=history,
        speedup=baseline["median_ms"] / best.median_ms,
        limitation="Timing quality and device isolation are the trusted caller executor responsibility.",
    )
