"""Owned typed-sample controls;unit samples are not latency evidence."""

import ast
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
import unittest
import test_implementation_handoff as fixture

from infinicoder.domains.operator.convolution_provider import digest
from infinicoder.domains.operator.convolution_schedule_search import (
    with_convolution_schedule_search,
)
from infinicoder.collaboration.implementation_handoff import (
    with_specialist_handoff,
    prepare_specialist_handoff,
    consume_specialist_handoff,
)
from infinicoder.optimization.native_measured_optimization import PublicPerformanceSample
from infinicoder.optimization.optimization import Measurement


def tile(source):
    call = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Subscript)
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "_ic_conv"
    )
    return {k.arg: ast.literal_eval(k.value) for k in call.keywords}


class ScheduleSearchControls(unittest.TestCase):
    def case(self, root, *, mode="ordinary"):
        def public(target):
            source = target.read_text()
            knobs = tile(source)
            pin = digest(target.read_bytes())
            if mode == "concurrent" and knobs["BN"] == 64:
                target.write_text("# foreign concurrent revision\n")
            status = "public_fail" if knobs["BM"] == 32 else "public_pass"
            return dict(
                status=status,
                source_sha256=pin,
                knobs=knobs,
                counterexample="Owned source contract requires BM16"
                if status == "public_fail"
                else "",
            )

        c, policy = fixture.HandoffControls().case(root, behavior=public)
        c.request = replace(c.request, budget=replace(c.request.budget, wall_seconds=180))
        with_specialist_handoff(c.request, policy)

        def decode(raw):
            correct = raw["status"] == "public_pass"
            pin = raw["source_sha256"]
            if mode == "wrong_binding" and raw["knobs"]["BN"] == 64:
                pin = "0" * 64
            # Explicit unit input to test the existing selection/restore rules.
            millis = 0.5 if raw["knobs"]["BN"] == 64 else 1.0
            return PublicPerformanceSample(
                {"solution.py": pin},
                correct,
                3,
                "owned-source-contract",
                Measurement((millis,) * 5) if correct else None,
                3,
                "owned-unit-public-report",
            )

        self.assertIs(
            with_convolution_schedule_search(
                c.request,
                performance_decoder=decode,
                decoder_identity="owned-unit-decoder",
                exclusive_workspace=True,
            ),
            c.request,
        )
        return c

    def run_case(self, c):
        messages = [dict(role="system", content="owned initial"), dict(role="user", content=c.task)]
        deadline = time.monotonic() + 180
        with c.provider.bind(c.request, c.model, c.environment, deadline):
            with c.operation.bind(c.request, deadline) as op, c.provider.bind_public_operation(op):
                receipt = prepare_specialist_handoff(c.provider, messages)
                consumed = receipt is not None and consume_specialist_handoff(c.provider, receipt)
        return receipt, consumed

    def test_real_different_sources_select_correct_fast_unit_sample_and_restore_failed_variant(
        self,
    ):
        with tempfile.TemporaryDirectory() as folder:
            c = self.case(Path(folder))
            receipt, consumed = self.run_case(c)
            self.assertTrue(consumed)
            self.assertIsNotNone(receipt.schedule_selection_sha256)
            self.assertEqual(tile(c.target.read_text())["BN"], 64)
            self.assertEqual(tile(c.target.read_text())["BM"], 16)
            self.assertEqual(len(c.calls), 3)
            self.assertEqual(c.inputs, [])
            self.assertEqual(len({digest(v["source"]) for v in c.calls}), 3)
            search = c.provider.schedule_search
            self.assertEqual(search.selected.schedule_name, "channel_tile")
            self.assertEqual(search.reused, 1)
            self.assertEqual(search.additional, 2)
            self.assertEqual(search.performance.candidates, 2)
            report = json.loads((c.state / "native-performance.json").read_bytes())
            self.assertTrue(report["final_source_binding_verified"])
            self.assertTrue(
                any(v.get("reason") == "public_correctness_failed" for v in report["records"])
            )

    def test_wrong_external_source_binding_restores_baseline_and_stops_variants(self):
        with tempfile.TemporaryDirectory() as folder:
            c = self.case(Path(folder), mode="wrong_binding")
            receipt, consumed = self.run_case(c)
            self.assertTrue(consumed)
            self.assertEqual(tile(c.target.read_text())["BN"], 32)
            self.assertEqual(len(c.calls), 2)
            self.assertTrue(c.provider.schedule_search.performance.disabled)
            self.assertEqual(c.provider.schedule_search.selected.schedule_name, "baseline")

    def test_public_execution_mutation_is_not_overwritten_or_handed_off(self):
        with tempfile.TemporaryDirectory() as folder:
            c = self.case(Path(folder), mode="concurrent")
            receipt, consumed = self.run_case(c)
            self.assertIsNone(receipt)
            self.assertFalse(consumed)
            self.assertEqual(c.target.read_text(), "# foreign concurrent revision\n")
            self.assertTrue(c.provider.schedule_search.performance.disabled)


if __name__ == "__main__":
    unittest.main()
