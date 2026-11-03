"""Standalone worker; caller mounts this and public_replay.py in its namespace."""

import json
from pathlib import Path
import runpy
import sys
from contextlib import nullcontext

from public_replay import PublicReplaySpec, PublicTraceRecorder, digest, python_entry


def main():
    config = json.loads(Path(sys.argv[1]).read_bytes())
    spec = PublicReplaySpec(**config["spec"])
    root = Path(config.get("model_root", "/workspace"))
    tracing = config.get("trace_enabled", True)
    if type(tracing) is not bool:
        raise ValueError("explicit public instrumentation policy required")
    recorder = PublicTraceRecorder(
        root,
        spec.source_paths,
        max_events=spec.max_events,
        capture_mode=spec.capture_mode,
        public_input_hashes=spec.public_input_hashes,
        public_module_paths=spec.public_module_paths,
    )
    for relative, pin in spec.public_input_hashes.items():
        if digest((root / relative).read_bytes()) != pin:
            raise ValueError("public reproduction input changed")
    mode, entry, args = python_entry(spec.argv)
    sys.dont_write_bytecode = True
    sys.argv = [entry, *args]
    sys.path[0] = str(root / Path(entry).parent) if mode == "script" else str(root)
    status = "finished"
    exception = None
    exit_code = 0
    try:
        with recorder if tracing else nullcontext():
            if mode == "script":
                runpy.run_path(str(root / entry), run_name="__main__")
            else:
                runpy.run_module(entry, run_name="__main__", alter_sys=True)
    except SystemExit as error:
        exit_code = error.code if type(error.code) is int else 0 if error.code is None else 1
        status = "system_exit"
        if tracing:
            recorder.capture_terminal(error, exit_code=exit_code, deadline=config.get("deadline"))
    except BaseException as error:
        exception = type(error).__name__
        status = "exception"
        exit_code = 1
        if tracing:
            recorder.capture_terminal(error, deadline=config.get("deadline"))
        if not tracing:
            import traceback

            traceback.print_exc()
    finally:
        if tracing and recorder.terminal_status == "not_observed":
            recorder.capture_terminal()
        report = recorder.report()
        report["instrumented"] = tracing
        report.update(
            entry_status=status,
            exception_type=exception,
            exit_code=exit_code,
            public_argv=spec.argv,
            public_origin=spec.public_origin,
            source_after={p: digest((root / p).read_bytes()) for p in spec.source_paths},
        )
        encode = lambda: json.dumps(report, ensure_ascii=False, allow_nan=False)
        if spec.capture_mode == "failure_tail":
            report["report_budget_evicted_events"] = 0
            while report["events"] and len(encode().encode()) > 256 * 1024:
                report["events"].pop(0)
                report["report_budget_evicted_events"] += 1
                report["truncated"] = True
            if len(encode().encode()) > 256 * 1024:
                report = dict(
                    collector_error="ReportBudgetExceeded",
                    events=[],
                    terminal_anchors=[],
                    truncated=True,
                )
        Path(config["output"]).write_text(encode(), encoding="utf8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
