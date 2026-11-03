"""Owned lifecycle/cost/rollback controls,not neural or GPU capability scores."""

from contextlib import contextmanager
from dataclasses import replace
import hashlib, json, tempfile, time, unittest
from pathlib import Path

from infinicoder.optimization.native_optimizing_provider import (
    NativeOptimizingImplementationProvider,
    optimize_provider_request,
)
from infinicoder.optimization.native_measured_optimization import (
    PublicPerformancePolicy,
    PublicPerformanceSample,
)
from infinicoder.optimization.optimization import Measurement
import test_native_implementation_provider as provider_controls


class OptimizingProviderTests(unittest.TestCase):
    def case(self, root, *, unsupported=False, untyped=False):
        reference = provider_controls.REFERENCE
        c = provider_controls.ProviderControls().setup_case(
            root,
            reference=reference.replace(
                "nn.Conv2d(ci, co, kernel, bias=False)", "nn.AvgPool2d(kernel)"
            )
            if unsupported
            else reference,
        )
        measurements = []

        def execute(deadline):
            raw = c.target.read_bytes()
            measurements.append(raw)
            correct = b"# wrong" not in raw
            latency = 5 if b"# fast" in raw else 20 if b"# slow" in raw else 10
            return dict(
                status="public_pass" if correct else "public_fail",
                counterexample="" if correct else "Owned wrong-value fixture",
                source_hashes={"solution.py": hashlib.sha256(raw).hexdigest()},
                correct=correct,
                samples_ms=[latency] * 5,
            )

        operation = replace(c.operation, executor=execute)
        check = replace(c.check, operation=operation)

        def decode(raw):
            if untyped:
                return dict(correct=True, latency=0.01)
            return PublicPerformanceSample(
                raw["source_hashes"],
                raw["correct"],
                3,
                "owned-synthetic-producer-lifecycle",
                Measurement(tuple(raw["samples_ms"])) if raw["correct"] else None,
                3,
                "owned-caller-public-report",
            )

        policy = PublicPerformancePolicy(operation, ("solution.py",), True, final_reserve_seconds=5)
        provider = NativeOptimizingImplementationProvider(
            lambda m, t: len(json.dumps(m)) // 4,
            tokenizer_identity="owned-test-counter",
            implementation_paths=("solution.py",),
            contract=c.contract,
            public_check=check,
            performance_policy=policy,
            performance_decoder=decode,
            decoder_identity="owned-synthetic-decoder-v1",
        )
        c.provider, c.operation = provider, operation
        c.request = replace(c.request, context_adapter=provider)
        c.measurements = measurements

        def ordinary_edit(action):
            c.target.write_bytes(
                c.target.read_bytes() + ("\n# " + action["command"] + "\n").encode()
            )
            return dict(returncode=0, output="ordinary edit")

        c.environment.execute = ordinary_edit
        return c

    @contextmanager
    def bound(self, c):
        deadline = time.monotonic() + 30
        with c.provider.bind(c.request, c.model, c.environment, deadline):
            with (
                c.operation.bind(c.request, deadline) as public,
                c.provider.bind_public_operation(public),
            ):
                yield [
                    dict(role="system", content="Owned original rules"),
                    dict(role="user", content=c.task),
                ]

    def test_original_generated_measurement_reused_and_slower_or_wrong_edits_restored(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self.case(Path(tmp))
            with self.bound(c) as h:
                c.model.query(h)
                generated = c.target.read_bytes()
                self.assertEqual(len(c.measurements), 1)
                self.assertEqual(c.provider.reused_provider_measurements, 1)
                c.environment.execute(dict(command="slow"))
                c.model.query(h)
                self.assertEqual(c.target.read_bytes(), generated)
                c.environment.execute(dict(command="wrong"))
                c.model.query(h)
                self.assertEqual(c.target.read_bytes(), generated)
                self.assertEqual(len(c.measurements), 3)
                self.assertEqual(c.provider.additional_public_executions, 2)
                c.model.query(h)
                self.assertEqual(len(c.measurements), 3)
            self.assertTrue(
                json.loads((c.state / "native-performance.json").read_bytes())[
                    "final_source_binding_verified"
                ]
            )

    def test_faster_ordinary_edit_retained_and_final_unmeasured_edit_restored(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self.case(Path(tmp))
            with self.bound(c) as h:
                c.model.query(h)
                c.environment.execute(dict(command="fast"))
                c.model.query(h)
                fast = c.target.read_bytes()
                self.assertEqual(c.provider.performance.view()["speedup_vs_first_correct"], 2)
                c.environment.execute(dict(command="wrong"))
            self.assertEqual(c.target.read_bytes(), fast)
            self.assertEqual(len(c.measurements), 2)

    def test_unsupported_producer_adds_no_performance_input_or_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self.case(Path(tmp), unsupported=True)
            with self.bound(c) as h:
                c.model.query(h)
                self.assertEqual(c.inputs, [h])
            self.assertEqual(c.measurements, [])

    def test_untyped_candidate_claim_is_not_a_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self.case(Path(tmp), untyped=True)
            with self.bound(c) as h:
                c.model.query(h)
                self.assertTrue(c.provider.performance.disabled)
                self.assertIsNone(c.provider.performance.best)
            self.assertEqual(len(c.measurements), 1)

    def test_a_different_operation_cannot_be_substituted_for_the_original_public_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self.case(Path(tmp))
            other = replace(c.operation, operation_id="another-public-operation")
            with self.assertRaises(ValueError):
                NativeOptimizingImplementationProvider(
                    lambda m, t: 1,
                    tokenizer_identity="owned",
                    implementation_paths=("solution.py",),
                    contract=c.contract,
                    public_check=c.check,
                    performance_policy=PublicPerformancePolicy(other, ("solution.py",), True),
                    performance_decoder=lambda r: r,
                    decoder_identity="owned",
                )

    def test_public_api_retains_original_contract_budget_and_adapters(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = provider_controls.ProviderControls().setup_case(Path(tmp))
            new = optimize_provider_request(
                c.request,
                performance_decoder=lambda r: r,
                decoder_identity="owned",
                exclusive_workspace=True,
            )
            for field in (
                "spec",
                "budget",
                "model_profile",
                "execution_adapter",
                "transport_adapter",
                "acceptance_adapter",
            ):
                self.assertIs(getattr(new, field), getattr(c.request, field))
            self.assertIs(new.context_adapter.public_check, c.check)
            with self.assertRaises(ValueError):
                optimize_provider_request(
                    new,
                    performance_decoder=lambda r: r,
                    decoder_identity="owned",
                    exclusive_workspace=True,
                )


if __name__ == "__main__":
    unittest.main()
