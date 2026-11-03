"""Late public failures and bounded terminal evidence;no benchmark or NN."""

from dataclasses import asdict, replace
import json, subprocess, sys, tempfile, time, unittest
from pathlib import Path
from infinicoder.verification.public_replay import PublicReplaySpec, PublicTraceRecorder, digest
from infinicoder.verification.public_replay_cpu import cpu_public_replay
from infinicoder.verification.native_public_failure_slice import NativePublicFailureSlice
import test_public_failure_slice as old_controls

CORE = "def compute(x):\n    return x*2\n"
PUBLIC = "from core import compute\nfor i in range(300):\n    compute(i)\nassert compute(9)==27\n"


def spec(mode):
    return PublicReplaySpec(
        ("python", "repro.py"),
        ("core.py", "repro.py"),
        {"repro.py": digest(PUBLIC.encode())},
        "Owned late public assertion",
        timeout_seconds=5,
        final_reserve_seconds=1,
        capture_mode=mode,
        public_module_paths=("repro.py",) if mode == "failure_tail" else (),
    )


def worker(workspace, output, mode):
    config = output.with_suffix(".config.json")
    config.write_text(
        json.dumps(
            dict(
                spec=asdict(spec(mode)),
                model_root=str(workspace),
                output=str(output),
                deadline=time.monotonic() + 5,
            )
        )
    )
    program = Path(cpu_public_replay.__code__.co_filename).with_name("public_replay_worker.py")
    process = subprocess.run(
        [sys.executable, "-B", str(program), str(config)], capture_output=True, timeout=8
    )
    return process, json.loads(output.read_bytes())


class FailureCaptureControls(unittest.TestCase):
    def test_late_module_assert_is_preserved_and_reaches_next_query_then_goes_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = old_controls.PublicSliceControls().setup_case(Path(tmp), command="python repro.py")
            (c.workspace / "core.py").write_bytes(CORE.encode())
            (c.workspace / "repro.py").write_bytes(PUBLIC.encode())
            prefix, prior = worker(c.workspace, c.state / "prefix.json", "prefix")
            self.assertEqual(prefix.returncode, 1)
            self.assertTrue(prior["truncated"])
            self.assertFalse(any(e["kind"] == "exception" for e in prior["events"]))
            calls = []
            reports = []

            def replay(deadline):
                process, trace = worker(c.workspace, c.state / "tail.json", "failure_tail")
                self.assertEqual(process.returncode, 1)
                calls.append(1)
                reports.append(trace)
                return dict(trace=trace)

            capability = replace(
                c.observer.capability,
                spec=spec("failure_tail"),
                operation=replace(c.observer.capability.operation, executor=replay),
            )
            observer = NativePublicFailureSlice(
                c.observer.count_messages, tokenizer_identity="owned counter", capability=capability
            )
            request = replace(c.request, context_adapter=observer)
            messages = [
                dict(role="system", content="owned"),
                dict(role="user", content=request.spec.task),
            ]
            with observer.bind(request, c.model, c.environment, time.monotonic() + 30):
                c.model.query(messages)
                c.environment.execute(dict(command="python repro.py"))
                c.model.query(messages)
                packet = c.inputs[-1]
                self.assertEqual(packet[:-1], messages)
                view = json.loads(packet[-1]["content"].split("\n", 1)[1])
                failure = view["events"][0]
                self.assertEqual(
                    (failure["kind"], failure["path"], failure["line"], failure["exception_type"]),
                    ("terminal_exception", "repro.py", 4, "AssertionError"),
                )
                self.assertIn("not proven root cause", view["failure_capture"]["failure_relation"])
                (c.workspace / "core.py").write_bytes(CORE.replace("x*2", "x*3").encode())
                c.model.query(messages)
                self.assertEqual(c.inputs[-1], messages)
            self.assertEqual(calls, [1])
            trace = reports[0]
            self.assertEqual(trace["terminal_status"], "unhandled_exception")
            self.assertGreater(trace["evicted_events"], 0)
            self.assertLessEqual(len(trace["events"]) + len(trace["terminal_anchors"]), 96)
            self.assertLessEqual(len(json.dumps(trace).encode()), 256 * 1024)
            ids = [
                e["frame"]
                for e in trace["events"]
                if e["kind"] == "call" and e["frame"] is not None
            ]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertGreater(max(ids), 96)
            self.assertEqual(observer.report["source_versions"], "historical")
            self.assertIsNone(sys.gettrace())

    def test_line_limit_still_permits_bounded_original_traceback_and_unknown_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = (
                "def descend(n):\n    if n: return descend(n-1)\n    return 1/0\ndescend(300)\n"
            )
            path = root / "public.py"
            path.write_text(source)
            for exhausted in (False, True):
                limit = 96 if exhausted else 8
                recorder = PublicTraceRecorder(
                    root,
                    ("public.py",),
                    max_events=limit,
                    capture_mode="failure_tail",
                    public_input_hashes={"public.py": digest(path.read_bytes())},
                    public_module_paths=("public.py",),
                )
                if exhausted:
                    recorder.line_events = 99999
                try:
                    with recorder:
                        exec(compile(source, str(path.resolve()), "exec"), {})
                except ZeroDivisionError as error:
                    recorder.capture_terminal(error, deadline=time.monotonic() + 1)
                report = recorder.report()
                self.assertTrue(report["terminal_anchors"])
                self.assertTrue(report["terminal_depth_limited"])
                self.assertLessEqual(len(report["events"]) + len(report["terminal_anchors"]), limit)
                self.assertEqual(report["coverage_exhausted"], exhausted)
                if exhausted:
                    self.assertTrue(
                        all(
                            e["frame"] is None and e["parent"] is None
                            for e in report["terminal_anchors"]
                        )
                    )
                if not exhausted:
                    self.assertGreater(report["unknown_frame_calls"], 0)
                self.assertIsNone(sys.gettrace())

    def test_caught_error_and_normal_return_are_not_terminal_failure_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = 'try:\n    raise RuntimeError("caught")\nexcept RuntimeError:\n    result=3\n'
            path = root / "public.py"
            path.write_text(source)
            recorder = PublicTraceRecorder(
                root,
                ("public.py",),
                capture_mode="failure_tail",
                public_input_hashes={"public.py": digest(path.read_bytes())},
                public_module_paths=("public.py",),
            )
            with recorder:
                exec(compile(source, str(path.resolve()), "exec"), {})
            recorder.capture_terminal()
            self.assertEqual(recorder.report()["terminal_status"], "no_unhandled_exception")
            self.assertEqual(recorder.report()["terminal_anchors"], [])
            with self.assertRaisesRegex(ValueError, "frozen"):
                PublicTraceRecorder(
                    root,
                    ("public.py",),
                    capture_mode="failure_tail",
                    public_module_paths=("public.py",),
                )
            self.assertIsNone(sys.gettrace())


if __name__ == "__main__":
    unittest.main()
