"""Nsight Systems adapter:parse profiler-owned SQLite,never application stdout."""

import hashlib
import json
from pathlib import Path
import sqlite3
import time
from urllib.parse import quote
from dataclasses import dataclass, field

from infinicoder.optimization.profiling import (
    ProfileReport,
    ProfileWorkload,
    ProfilingObjective,
    _text,
)
from infinicoder.execution.workspace_tools import clean_environment, run_process


def parse_nsight_sqlite(path):
    """2025.3 SQLite kernel/API summaries;unknown/missing schema stays unavailable."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Nsight export must be a regular SQLite file")
    db = sqlite3.connect(
        "file:" + quote(path.resolve().as_posix(), safe="/:") + "?mode=ro&immutable=1", uri=True
    )
    try:
        db.execute("PRAGMA query_only=ON")
        columns = {r[1] for r in db.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
        if not {"start", "end", "demangledName"} <= columns:
            raise ValueError("Unsupported Nsight kernel schema or no CUDA kernel trace")
        if db.execute(
            "SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE typeof(start)!='integer' OR typeof(end)!='integer' OR end<start OR start<0"
        ).fetchone()[0]:
            raise ValueError("Invalid Nsight kernel timestamps")
        count, total = db.execute(
            "SELECT count(*),sum(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchone()
        if not count:
            raise ValueError("No CUDA kernels captured;no performance conclusion")
        metrics = dict(
            cuda_kernel_instances=dict(value=count, unit="count"),
            cuda_kernel_total_duration=dict(value=total, unit="ns"),
            cuda_kernel_mean_duration=dict(value=total / count, unit="ns"),
        )
        hotspots = tuple(
            dict(name=str(name)[:240], total_ns=duration, instances=instances)
            for name, duration, instances in db.execute(
                "SELECT s.value,sum(k.end-k.start) AS duration,count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id GROUP BY k.demangledName ORDER BY duration DESC LIMIT 5"
            )
        )
        if not hotspots:
            raise ValueError("Missing Nsight kernel names")
        runtime_columns = {
            r[1] for r in db.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_RUNTIME)")
        }
        if {"start", "end"} <= runtime_columns:
            if db.execute(
                "SELECT count(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE typeof(start)!='integer' OR typeof(end)!='integer' OR end<start OR start<0"
            ).fetchone()[0]:
                raise ValueError("Invalid Nsight runtime timestamps")
            calls, duration = db.execute(
                "SELECT count(*),sum(end-start) FROM CUPTI_ACTIVITY_KIND_RUNTIME"
            ).fetchone()
            if calls:
                metrics.update(
                    cuda_api_calls=dict(value=calls, unit="count"),
                    cuda_api_total_duration=dict(value=duration, unit="ns"),
                )
        return metrics, hotspots
    finally:
        db.close()


@dataclass(frozen=True)
class NsightSystemsCollector:
    """Fixed caller argv/device/tool version;invocation never accepts model argv.

    Use an external sandbox and hardware admission lock when sharing resources.
    Tracing overhead and concurrent kernels make these diagnostics unsuitable as
    optimization selection timings. The existing independent measure stays final.
    """

    argv: tuple
    visible_devices: str
    tool_version: str
    nsys: str = "/usr/local/bin/nsys"
    allow_host_device_metadata: bool = False
    identity: str = field(init=False)

    def __post_init__(self):
        if (
            not isinstance(self.argv, tuple)
            or not 1 <= len(self.argv) <= 128
            or any(not isinstance(v, str) or not v or len(v) > 8192 for v in self.argv)
        ):
            raise ValueError("Explicit bounded profiler application argv tuple required")
        _text(self.visible_devices, "visible device identity", 200)
        _text(self.tool_version, "Nsight version", 200)
        _text(self.nsys, "Nsight executable", 1000)
        if type(self.allow_host_device_metadata) is not bool:
            raise ValueError("Explicit caller metadata permission must be bool")
        config = dict(
            argv=self.argv,
            visible_devices=self.visible_devices,
            tool_version=self.tool_version,
            nsys=self.nsys,
            allow_host_device_metadata=self.allow_host_device_metadata,
            adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        object.__setattr__(
            self,
            "identity",
            hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        )

    def __call__(self, workspace, folder, seconds):
        # CUDA_VISIBLE_DEVICES selects the application device. Nsight can still
        # enumerate host-wide static GPU metadata. Never mistake it for isolation.
        # Use a device-isolated worker or obtain permission for all host metadata.
        if not self.allow_host_device_metadata:
            return ProfileReport(
                "nsight-systems",
                "not-executed",
                self.visible_devices,
                {},
                status="unavailable",
                error="Nsight can enumerate host GPU metadata beyond CUDA_VISIBLE_DEVICES. Caller metadata permission is required;use a device-isolated worker when other host devices are out of scope.",
            )
        start = time.monotonic()
        env = clean_environment()
        env["CUDA_VISIBLE_DEVICES"] = self.visible_devices
        artifacts = []
        observed_version = "unavailable"

        def invoke(argv, name):
            result = run_process(
                argv,
                workspace,
                timeout=max(0, seconds - (time.monotonic() - start)),
                max_output=16000,
                env=env,
                spool_cap=4 * 1024 * 1024,
            )
            (folder / name).write_text(result.output, encoding="utf-8")
            artifacts.append(name)
            if result.exit_code or result.timed_out or result.truncated:
                raise ValueError("Nsight command failed/timed out/truncated;see " + name)
            return result

        try:
            version = invoke([self.nsys, "--version"], "tool-version.txt").output.strip()
            observed_version = version[:200] or "unavailable"
            if version != self.tool_version:
                raise ValueError("Installed Nsight version differs from caller contract")
            target = folder / "capture"
            invoke(
                [
                    self.nsys,
                    "profile",
                    "--trace=cuda,nvtx",
                    "--sample=none",
                    "--cpuctxsw=none",
                    "--force-overwrite=false",
                    "--output=" + str(target),
                    *self.argv,
                ],
                "profile-process.txt",
            )
            report = folder / "capture.nsys-rep"
            if not report.is_file() or report.is_symlink():
                raise ValueError("Nsight report not produced")
            artifacts.append(report.name)
            # profile may already produce SQLite. Export to a separate unique
            # path;never overwrite or parse application-produced stdout as data.
            database = folder / "profile-export.sqlite"
            invoke(
                [
                    self.nsys,
                    "export",
                    "--type=sqlite",
                    "--force-overwrite=false",
                    "--output=" + str(database),
                    str(report),
                ],
                "export-process.txt",
            )
            artifacts.append(database.name)
            metrics, hotspots = parse_nsight_sqlite(database)
            return ProfileReport(
                "nsight-systems",
                self.tool_version,
                self.visible_devices,
                metrics,
                hotspots=hotspots,
                artifacts=tuple(artifacts),
            )
        except (ValueError, OSError, sqlite3.Error) as exc:
            return ProfileReport(
                "nsight-systems",
                observed_version,
                self.visible_devices,
                {},
                artifacts=tuple(artifacts),
                status="unavailable",
                error=str(exc)[:900],
            )


def load_profiling_config(path):
    """Explicit CLI caller configuration only;no dynamic imports or model fields."""
    value = json.loads(Path(path).read_bytes())
    if (
        not isinstance(value, dict)
        or set(value)
        - {
            "paths",
            "workloads",
            "identity",
            "max_seconds",
            "nsys",
            "tool_version",
            "allow_host_device_metadata",
        }
        or not {"paths", "workloads", "identity", "tool_version"} <= set(value)
    ):
        raise ValueError("Invalid caller profiling configuration")
    workloads = []
    if not isinstance(value["workloads"], list) or not isinstance(value["paths"], list):
        raise ValueError("Explicit profiling workloads required")
    for row in value["workloads"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"name", "argv", "visible_devices"}
            or not isinstance(row["argv"], list)
        ):
            raise ValueError("Each configured workload needs name,argv,visible_devices")
        collector = NsightSystemsCollector(
            tuple(row["argv"]),
            row["visible_devices"],
            value["tool_version"],
            value.get("nsys", "/usr/local/bin/nsys"),
            value.get("allow_host_device_metadata", False),
        )
        workloads.append(ProfileWorkload(row["name"], collector, collector.identity))
    return ProfilingObjective(
        tuple(value["paths"]), tuple(workloads), value["identity"], value.get("max_seconds", 60)
    )
