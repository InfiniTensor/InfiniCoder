"""Caller-owned,source-bound performance observations;never acceptance evidence."""

from infinicoder._resources import package_root

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import time
import uuid


def identity():
    folder = package_root()
    return hashlib.sha256(
        b"".join(
            (folder / name).read_bytes()
            for name in (
                "optimization/profiling.py",
                "optimization/nsight.py",
                "runtime/engine.py",
                "runtime/protocol.py",
                "runtime/shell_loop.py",
            )
        )
    ).hexdigest()


def _text(value, label, limit=200):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError("Invalid " + label)
    return value


def _relative(value):
    _text(value, "relative path", 500)
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or "\\" in value or ":" in value or str(p) != value:
        raise ValueError("Explicit normalized relative paths required")
    return value


@dataclass(frozen=True)
class ProfileWorkload:
    name: str
    collect: object
    identity: str

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch(
            "[A-Za-z][A-Za-z0-9_-]{0,47}", self.name
        ):
            raise ValueError("Invalid profiling workload name")
        if not callable(self.collect):
            raise TypeError("Caller-owned profiling callback required")
        _text(self.identity, "workload identity", 500)


@dataclass(frozen=True)
class ProfilingObjective:
    paths: tuple
    workloads: tuple
    identity: str
    max_seconds: float = 60

    def __post_init__(self):
        if (
            not isinstance(self.paths, tuple)
            or not 1 <= len(self.paths) <= 64
            or len(set(self.paths)) != len(self.paths)
        ):
            raise ValueError("One to64 unique tracked source paths required as a tuple")
        for path in self.paths:
            _relative(path)
        if (
            not isinstance(self.workloads, tuple)
            or not 1 <= len(self.workloads) <= 16
            or not all(isinstance(w, ProfileWorkload) for w in self.workloads)
        ):
            raise ValueError("Explicit bounded profiling workloads required as a tuple")
        if len({w.name for w in self.workloads}) != len(self.workloads):
            raise ValueError("Duplicate profiling workload")
        _text(self.identity, "profiling identity", 500)
        if (
            isinstance(self.max_seconds, bool)
            or not isinstance(self.max_seconds, (int, float))
            or not 0 < self.max_seconds <= 3600
        ):
            raise ValueError("Profiling timeout must be finite and in(0,3600]")

    def contract(self):
        return dict(
            identity=self.identity,
            paths=list(self.paths),
            max_seconds=self.max_seconds,
            workloads=[dict(name=w.name, identity=w.identity) for w in self.workloads],
        )


@dataclass(frozen=True)
class ProfileReport:
    tool: str
    tool_version: str
    device: str
    metrics: dict
    hotspots: tuple = ()
    artifacts: tuple = ()
    status: str = "observed"
    error: str = ""

    def data(self, folder):
        for label in ("tool", "tool_version", "device"):
            _text(getattr(self, label), label)
        if (
            self.status not in ("observed", "unavailable", "error")
            or not isinstance(self.error, str)
            or len(self.error) > 1000
        ):
            raise ValueError("Invalid profiling status")
        if not isinstance(self.metrics, dict) or len(self.metrics) > 24:
            raise ValueError("Bounded named profile metrics required")
        for name, metric in self.metrics.items():
            _text(name, "metric name", 80)
            if not isinstance(metric, dict) or set(metric) != {"value", "unit"}:
                raise ValueError("A profile metric needs value and unit")
            value = metric["value"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("Finite numeric profile metric required")
            _text(metric["unit"], "metric unit", 48)
        if self.status != "observed" and self.metrics:
            raise ValueError("Unsuccessful profiling cannot supply current metrics")
        if self.status == "observed" and not self.metrics:
            raise ValueError("Observed profiling requires metrics")
        if not isinstance(self.hotspots, tuple) or len(self.hotspots) > 5:
            raise ValueError("At most five hotspot records required")
        if any(
            not isinstance(v, dict) or len(json.dumps(v, ensure_ascii=False)) > 1500
            for v in self.hotspots
        ):
            raise ValueError("Bounded diagnostic hotspot records required")
        if (
            not isinstance(self.artifacts, tuple)
            or len(self.artifacts) > 8
            or len(set(self.artifacts)) != len(self.artifacts)
        ):
            raise ValueError("At most eight unique profiler-owned artifacts required")
        files = {}
        for name in self.artifacts:
            _relative(name)
            path = folder / name
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(folder.resolve())
                or not path.is_file()
                or path.stat().st_size > 256 * 1024 * 1024
            ):
                raise ValueError("Profiler artifact must be an owned regular file")
            files[name] = dict(
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size
            )
        # Copy callback data;neither later callback mutation nor non-JSON values
        # may silently alter persisted evidence.
        return json.loads(
            json.dumps(
                dict(
                    tool=self.tool,
                    tool_version=self.tool_version,
                    device=self.device,
                    metrics=self.metrics,
                    hotspots=self.hotspots,
                    artifacts=files,
                    status=self.status,
                    error=self.error,
                ),
                allow_nan=False,
            )
        )


class ProfilingSession:
    def __init__(self, objective, workspace, state, store, artifact_root):
        self.objective, self.ws, self.state, self.store = objective, workspace, state, store
        self.root = Path(artifact_root).resolve()
        if self.root.is_relative_to(workspace.root):
            raise ValueError("Profiling evidence must be outside editable workspace")
        self.contract = objective.contract()
        saved = state.get("profiling")
        if saved is not None and saved["contract"] != self.contract:
            raise ValueError("Profiling contract changed across resume")
        if saved is None:
            state["profiling"] = dict(contract=self.contract, reports=[])

    def sources(self):
        result = {}
        for name in self.objective.paths:
            path = self.ws.path(name)
            if not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
                raise ValueError("Tracked profiling source missing or larger than8MiB: " + name)
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result

    def collect(self, workload, remaining):
        if not isinstance(workload, str) or workload not in {
            w.name for w in self.objective.workloads
        }:
            raise ValueError("Choose an explicitly configured profiling workload")
        if remaining() <= 0:
            raise ValueError("No task budget remains for profiling")
        before = self.sources()
        source_id = hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()
        for row in reversed(self.state["profiling"]["reports"]):
            if (
                row["workload"] == workload
                and row["tracked_sources"] == before
                and row["status"] == "observed"
            ):
                return dict(self._summary(row), reused=True)
        timeout = min(self.objective.max_seconds, remaining())
        if timeout <= 0:
            raise ValueError("No task budget remains for profiling")
        folder = self.root / uuid.uuid4().hex
        folder.mkdir(parents=True, exist_ok=False)
        row = dict(
            id=folder.name,
            workload=workload,
            source_fingerprint=source_id,
            tracked_sources=before,
            status="error",
            metrics={},
            hotspots=[],
            artifacts={},
            error="",
            diagnostic_only=True,
            acceptance_proof=False,
            selection_measurement=False,
            artifact_directory=str(folder),
        )
        self.store.event(
            "profiling_started",
            dict(id=row["id"], workload=workload, source_fingerprint=source_id, timeout_s=timeout),
        )
        tick = time.monotonic()
        try:
            collector = next(w.collect for w in self.objective.workloads if w.name == workload)
            report = collector(self.ws.root, folder, timeout)
            if not isinstance(report, ProfileReport):
                raise TypeError("Trusted profiler must return ProfileReport")
            row.update(report.data(folder))
        except Exception as exc:
            row.update(
                status="error",
                error=type(exc).__name__ + ": " + str(exc)[:700],
                metrics={},
                hotspots=[],
            )
        finally:
            row["wall_s"] = time.monotonic() - tick
        try:
            unchanged = self.sources() == before
        except (ValueError, OSError):
            unchanged = False
        row["wall_s"] = time.monotonic() - tick
        if not unchanged:
            row.update(
                status="source_changed",
                metrics={},
                hotspots=[],
                error="Tracked source changed during profiling;observations invalidated",
            )
        elif row["wall_s"] > timeout or remaining() <= 0:
            row.update(
                status="deadline_exceeded",
                metrics={},
                hotspots=[],
                error="Profiling exceeded its shared task budget",
            )
        with (folder / "receipt.json").open("x", encoding="utf-8") as stream:
            json.dump(row, stream, indent=2, allow_nan=False)
        self.state["profiling"]["reports"].append(row)
        del self.state["profiling"]["reports"][:-8]
        self.store.event("profiling_observation", row)
        return self._summary(row)

    @staticmethod
    def _summary(row):
        return json.loads(
            json.dumps(
                {k: v for k, v in row.items() if k not in ("tracked_sources", "artifact_directory")}
            )
        )

    def view(self, estimator, max_tokens=400):
        rows = self.state["profiling"]["reports"]
        result = dict(
            workloads=[w.name for w in self.objective.workloads],
            diagnostic_only=True,
            acceptance_proof=False,
        )
        if rows:
            latest = rows[-1]
            try:
                current = self.sources() == latest["tracked_sources"]
            except (ValueError, OSError):
                current = False
            result["latest"] = self._summary(latest)
            if not current:
                result["latest"] = dict(
                    id=latest["id"],
                    workload=latest["workload"],
                    status="stale_source_requires_profile",
                    historical_source_fingerprint=latest["source_fingerprint"],
                    metrics={},
                )
        omitted = 0
        while estimator.estimate(json.dumps(result, ensure_ascii=False)) > max_tokens:
            latest = result.get("latest", {})
            if latest.get("hotspots"):
                latest["hotspots"] = latest["hotspots"][:-1]
            elif latest.get("artifacts"):
                latest.pop("artifacts")
            elif latest.get("metrics"):
                latest["metrics"].pop(next(reversed(latest["metrics"])))
                omitted += 1
                latest["omitted_metrics"] = omitted
            else:
                return None
        return result


def guidance(objective):
    return (
        '\nprofile {"workload":name} -> run only a caller-configured profiler workload: '
        + ", ".join(w.name for w in objective.workloads)
        + ". Source-bound observations are diagnostic only;verify correctness and performance independently. Earlier observations refer to their recorded source,not current edits. Kernel duration sums and shares are not application wall time or GPU utilization.\n"
    )
