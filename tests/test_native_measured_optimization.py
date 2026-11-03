import hashlib, json, tempfile, time, unittest
from pathlib import Path
from types import SimpleNamespace
from infinicoder import (
    CoreRequest,
    CoreBudget,
    ModelProfile,
    TaskSpec,
    ReadOnlyTaskSources,
    CallerPublicOperation,
)
from infinicoder.optimization.native_measured_optimization import (
    PublicPerformancePolicy,
    PublicPerformanceSample,
    NativeMeasuredOptimization,
)
from infinicoder.optimization.optimization import Measurement


def count(messages, remaining):
    return len(json.dumps(messages)) // 4


class PerformanceControllerTests(unittest.TestCase):
    def case(self, root, *, mutate=False, untyped=False, unavailable=False):
        workspace = root / "workspace"
        workspace.mkdir()
        state = root / "state"
        state.mkdir()
        (workspace / "impl.py").write_bytes(b"baseline\n")
        (workspace / "api.py").write_bytes(b"fixed dependency\n")
        calls = []

        def measure(deadline):
            pins = {
                p: hashlib.sha256((workspace / p).read_bytes()).hexdigest()
                for p in ("impl.py", "api.py")
            }
            text = (workspace / "impl.py").read_text().strip()
            calls.append(text)
            if unavailable and text == "fast":
                raise RuntimeError("Owned public timer unavailable")
            if mutate and text == "fast":
                (workspace / "impl.py").write_bytes(b"concurrent edit\n")
            if untyped and text == "fast":
                return dict(correct=True, samples_ms=[0.01] * 5)
            correct = text != "wrong"
            latency = {"baseline": 10, "fast": 5, "slower": 20}.get(text, 25)
            return PublicPerformanceSample(
                pins,
                correct,
                3,
                "owned_synthetic_controller_workload",
                Measurement((latency,) * 5) if correct else None,
                3,
                str(state / "owned-public-evidence.json"),
            )

        op = CallerPublicOperation(
            "owned-public-perf",
            workspace,
            dict(executor="owned-controller-test;not-a-performance-benchmark"),
            "Owned positive-value fixture",
            dict(checks=3),
            10,
            dict(device="synthetic;not_GPU_timing"),
            ("impl.py", "api.py"),
            (),
            512,
            measure,
        )
        policy = PublicPerformancePolicy(op, ("impl.py",), True, final_reserve_seconds=5)
        c = NativeMeasuredOptimization(
            count,
            tokenizer_identity="owned-unit-counter",
            performance_policy=policy,
            implementation_paths=("impl.py", "api.py"),
            source_capability=ReadOnlyTaskSources(workspace, ("impl.py", "api.py")),
        )
        req = CoreRequest(
            TaskSpec("Optimize owned positive-value fixture;preserve api.py."),
            workspace,
            state,
            ModelProfile(8192, 1024),
            CoreBudget(15, 300, 20),
            context_adapter=c,
        )
        model = SimpleNamespace(
            _prepare_messages_for_api=lambda m: m,
            query=lambda m, **kw: dict(role="assistant", content="ordinary"),
        )

        def execute(action):
            (workspace / action.get("path", "impl.py")).write_bytes(
                (action["command"] + "\n").encode()
            )
            return dict(returncode=0, output="ordinary edit")

        env = SimpleNamespace(
            execute=execute,
            implementation_workspace_binding=lambda: dict(
                root=str(workspace.resolve()), model_root="/workspace"
            ),
        )
        history = [dict(role="system", content="rules"), dict(role="user", content=req.spec.task)]
        return c, req, model, env, calls, history

    def test_new_correct_version_retained_bad_version_restored_without_retiming_same_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, r, m, e, calls, h = self.case(Path(tmp))
            with c.bind(r, m, e, time.monotonic() + 300):
                m.query(h)
                e.execute(dict(command="fast"))
                m.query(h)
                e.execute(dict(command="wrong"))
                m.query(h)
                self.assertEqual((r.workspace / "impl.py").read_bytes(), b"fast\n")
                m.query(h)
                self.assertEqual(calls, ["baseline", "fast", "wrong"])
                self.assertEqual(c.performance.view()["speedup_vs_first_correct"], 2)
            self.assertTrue(
                json.loads((r.state_dir / "native-performance.json").read_bytes())[
                    "final_source_binding_verified"
                ]
            )

    def test_two_candidate_limit_keeps_best_and_does_not_measure_a_third(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, r, m, e, calls, h = self.case(Path(tmp))
            with c.bind(r, m, e, time.monotonic() + 300):
                m.query(h)
                for text in ("fast", "slower", "third_unmeasured"):
                    e.execute(dict(command=text))
                    m.query(h)
                self.assertEqual(calls, ["baseline", "fast", "slower"])
                self.assertEqual(c.performance.candidates, 2)
                self.assertEqual((r.workspace / "impl.py").read_bytes(), b"fast\n")

    def test_repeated_measured_rejected_version_restored_before_next_decision_without_retiming(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            c, r, m, e, calls, h = self.case(Path(tmp))
            with c.bind(r, m, e, time.monotonic() + 300):
                m.query(h)
                e.execute(dict(command="slower"))
                m.query(h)
                self.assertEqual((r.workspace / "impl.py").read_bytes(), b"baseline\n")
                e.execute(dict(command="slower"))
                m.query(h)
                self.assertEqual((r.workspace / "impl.py").read_bytes(), b"baseline\n")
                self.assertEqual(calls, ["baseline", "slower"])

    def test_cached_rejected_version_written_outside_sdk_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, r, m, e, calls, h = self.case(Path(tmp))
            with c.bind(r, m, e, time.monotonic() + 300):
                m.query(h)
                e.execute(dict(command="slower"))
                m.query(h)
                c.performance.last_sdk_sources = None
                (r.workspace / "impl.py").write_bytes(b"slower\n")
                m.query(h)
                self.assertTrue(c.performance.disabled)
                self.assertEqual((r.workspace / "impl.py").read_bytes(), b"slower\n")
                self.assertEqual(calls, ["baseline", "slower"])

    def test_readonly_dependency_change_invalidates_best_before_comparing_faster_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, r, m, e, calls, h = self.case(Path(tmp))
            with c.bind(r, m, e, time.monotonic() + 300):
                m.query(h)
                e.execute(dict(path="api.py", command="changed dependency"))
                e.execute(dict(command="fast"))
                m.query(h)
                self.assertTrue(c.performance.disabled)
                self.assertIsNone(c.performance.best)
                self.assertEqual(calls, ["baseline"])
                self.assertEqual((r.workspace / "api.py").read_bytes(), b"changed dependency\n")
                self.assertEqual((r.workspace / "impl.py").read_bytes(), b"fast\n")

    def test_measurement_source_mutation_is_rejected_without_overwriting_concurrent_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, r, m, e, calls, h = self.case(Path(tmp), mutate=True)
            with c.bind(r, m, e, time.monotonic() + 300):
                m.query(h)
                e.execute(dict(command="fast"))
                m.query(h)
                self.assertTrue(c.performance.disabled)
                self.assertEqual((r.workspace / "impl.py").read_bytes(), b"concurrent edit\n")
            self.assertFalse(c.performance.final_source_binding_verified)

    def test_self_report_dictionary_does_not_become_external_correctness(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, r, m, e, calls, h = self.case(Path(tmp), untyped=True)
            with c.bind(r, m, e, time.monotonic() + 300):
                m.query(h)
                e.execute(dict(command="fast"))
                m.query(h)
                self.assertTrue(c.performance.disabled)
                self.assertEqual((r.workspace / "impl.py").read_bytes(), b"baseline\n")

    def test_optional_timer_failure_restores_best_and_keeps_ordinary_coding_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, r, m, e, calls, h = self.case(Path(tmp), unavailable=True)
            with c.bind(r, m, e, time.monotonic() + 300):
                m.query(h)
                e.execute(dict(command="fast"))
                m.query(h)
                self.assertTrue(c.performance.disabled)
                self.assertEqual((r.workspace / "impl.py").read_bytes(), b"baseline\n")
                m.query(h)
                self.assertEqual(calls, ["baseline", "fast"])


if __name__ == "__main__":
    unittest.main()
