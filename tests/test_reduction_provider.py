"""Owned source lifecycle controls;typed unit samples are not GPU timings."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
import unittest

from test_reduction import reference
from test_native_implementation_provider import ProviderControls, SCOPE, STARTER
from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.domains.operator.candidate_provider import ProviderLeaseError
from infinicoder.verification.candidate_public_check import contract_hash
from infinicoder.domains.operator.convolution_provider import digest
from infinicoder.domains.operator.convolution_schedule_search import (
    with_convolution_schedule_search,
)
from infinicoder.app.development_api import configure_development_request
from infinicoder.memory.source.evidence_source import ReadOnlyTaskSources
from infinicoder.collaboration.implementation_handoff import (
    ImplementationHandoffPolicy,
    with_specialist_handoff,
    prepare_specialist_handoff,
    consume_specialist_handoff,
)
from infinicoder.domains.operator.native_implementation_provider import NativeImplementationProvider
from infinicoder.optimization.native_measured_optimization import PublicPerformanceSample
from infinicoder.optimization.optimization import Measurement
from infinicoder.domains.operator.reduction_provider import ReductionProviderContract, assess, apply
from infinicoder.domains.operator.reduction_schedule_search import with_reduction_schedule_search
from infinicoder.planning.task_contract import TaskSpec
from infinicoder.execution.workspace_tools import Workspace


class ReductionProviderTests(unittest.TestCase):
    def case(self, root, *, mode="ordinary", source=None):
        def public(target):
            pin = digest(target.read_bytes())
            coalesced = "if True and adjacent < sr:" in target.read_text()
            status = "public_fail" if coalesced and mode == "numeric_fail" else "public_pass"
            return dict(
                status=status,
                source_sha256=pin,
                coalesced=coalesced,
                counterexample="owned numeric failure" if status == "public_fail" else "",
            )

        c = ProviderControls().setup_case(
            root, reference=source or reference(), behavior=public, reserve=5
        )
        c.contract = ReductionProviderContract(
            "solution.py", digest(source or reference()), SCOPE, True, 5
        )
        c.provider = NativeImplementationProvider(
            lambda messages, timeout: len(json.dumps(messages)) // 4,
            tokenizer_identity="owned-unit-counter-not-model-tokenizer",
            contract=c.contract,
            public_check=c.check,
            implementation_paths=("solution.py",),
        )
        c.request = replace(
            c.request,
            spec=TaskSpec(c.task, required_artifacts=["solution.py"]),
            context_adapter=c.provider,
            budget=replace(c.request.budget, wall_seconds=180),
        )
        policy = ImplementationHandoffPolicy(
            digest(c.task),
            contract_hash(c.check.manifest()),
            ("owned-source-ABI",),
            ("owned-source-ABI",),
            False,
            (),
            True,
        )
        with_specialist_handoff(c.request, policy)

        def decode(raw):
            pin = "0" * 64 if mode == "wrong_binding" and raw["coalesced"] else raw["source_sha256"]
            correct = raw["status"] == "public_pass"
            return PublicPerformanceSample(
                {"solution.py": pin},
                correct,
                1 if correct else 0,
                "owned-sum-source-contract",
                Measurement((0.5 if raw["coalesced"] else 1.0,) * 5) if correct else None,
                3 if correct else 0,
                "owned-unit-public-report",
            )

        with_reduction_schedule_search(
            c.request,
            performance_decoder=decode,
            decoder_identity="owned-unit-sum-decoder",
            exclusive_workspace=True,
        )
        return c

    def run_case(self, c, *, mutate=False):
        messages = [
            dict(role="system", content="owned original"),
            dict(role="user", content=c.task),
        ]
        deadline = time.monotonic() + 180
        with c.provider.bind(c.request, c.model, c.environment, deadline):
            with c.operation.bind(c.request, deadline) as op, c.provider.bind_public_operation(op):
                receipt = prepare_specialist_handoff(c.provider, messages)
                if mutate:
                    c.target.write_bytes(c.target.read_bytes() + b"\n# concurrent change\n")
                consumed = receipt is not None and consume_specialist_handoff(c.provider, receipt)
                if receipt is None:
                    c.model.query(messages)
        return receipt, consumed, messages

    def test_generated_sum_flows_through_same_owner_public_selection_and_handoff(self):
        with tempfile.TemporaryDirectory() as folder:
            c = self.case(Path(folder))
            receipt, consumed, _ = self.run_case(c)
            self.assertTrue(consumed)
            self.assertIsNotNone(receipt.schedule_selection_sha256)
            self.assertEqual(
                c.provider.report["identity"], "native_single_axis_sum_implementation_v1"
            )
            self.assertEqual(c.provider.report["facts"]["semantics"], "single_axis_sum")
            self.assertEqual(c.provider.schedule_search.selected.schedule_name, "coalesced")
            self.assertEqual(len(c.calls), 2)
            self.assertEqual(len({digest(v["source"]) for v in c.calls}), 2)
            self.assertEqual(c.provider.schedule_search.reused, 1)
            self.assertEqual(c.provider.schedule_search.additional, 1)
            self.assertEqual(c.inputs, [])
            self.assertTrue((c.state / "native-reduction-schedules.json").is_file())
            self.assertFalse((c.state / "native-convolution-schedules.json").exists())

    def test_failed_or_wrongly_bound_variant_restores_correct_baseline(self):
        for mode in ("numeric_fail", "wrong_binding"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                c = self.case(Path(folder), mode=mode)
                _, consumed, _ = self.run_case(c)
                self.assertTrue(consumed)
                self.assertIn("if False and adjacent < sr:", c.target.read_text())
                self.assertEqual(c.provider.schedule_search.selected.schedule_name, "scalar")
                self.assertEqual(len(c.calls), 2)

    def test_after_check_change_invalidates_handoff_without_restoring_other_edit(self):
        with tempfile.TemporaryDirectory() as folder:
            c = self.case(Path(folder))
            receipt, consumed, _ = self.run_case(c, mutate=True)
            self.assertIsNotNone(receipt)
            self.assertFalse(consumed)
            self.assertTrue(c.target.read_bytes().endswith(b"# concurrent change\n"))

    def test_nearby_unsupported_semantics_preserve_original_input(self):
        for source in (reference("mean"), reference(init="(0,2)")):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as folder:
                c = self.case(Path(folder), source=source)
                receipt, consumed, messages = self.run_case(c)
                self.assertIsNone(receipt)
                self.assertFalse(consumed)
                self.assertEqual(c.target.read_bytes(), STARTER)
                self.assertEqual(c.calls, [])
                self.assertEqual(c.inputs, [messages])

    def test_source_lease_rejects_changed_target_task_and_forged_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            c = self.case(Path(folder))
            ws = Workspace(c.workspace, ("solution.py",))
            admitted = assess(c.request.spec, ws, c.contract)
            self.assertEqual(admitted["status"], "Applicable")
            altered = dict(
                admitted, candidate_source=admitted["candidate_source"] + "\n# replaced\n"
            )
            altered["candidate_sha256"] = digest(altered["candidate_source"])
            with self.assertRaises(ProviderLeaseError):
                apply(c.request.spec, ws, c.contract, altered)
            with self.assertRaises(ProviderLeaseError):
                apply(replace(c.request.spec, task=c.task + "\nchanged"), ws, c.contract, admitted)
            c.target.write_bytes(b"# another owner edit\n")
            with self.assertRaises(ProviderLeaseError):
                apply(c.request.spec, ws, c.contract, admitted)

    def test_matching_producer_type_required_by_schedule_factory(self):
        with tempfile.TemporaryDirectory() as folder:
            c = self.case(Path(folder))
            c.provider.schedule_search = None
            with self.assertRaises(UnsupportedCore):
                with_convolution_schedule_search(
                    c.request,
                    performance_decoder=lambda raw: raw,
                    decoder_identity="owned",
                    exclusive_workspace=True,
                )

    def test_auto_factory_selects_explicit_sum_without_replacing_task_or_budget(self):
        from types import SimpleNamespace
        from infinicoder.runtime.native_plain import NativePlainObserver

        with tempfile.TemporaryDirectory() as folder:
            c = self.case(Path(folder))
            plain = NativePlainObserver(
                c.provider.count_messages,
                tokenizer_identity=c.provider.tokenizer_identity,
                implementation_paths=("solution.py",),
            )
            original = replace(
                c.request,
                context_adapter=plain,
                acceptance_adapter=SimpleNamespace(public_check=c.check),
            )
            selected = configure_development_request(
                original,
                source_capability=ReadOnlyTaskSources(c.workspace, ("solution.py",)),
                reduction_provider=c.contract,
            )
            self.assertIs(type(selected.context_adapter), NativeImplementationProvider)
            self.assertIs(type(selected.context_adapter.contract), ReductionProviderContract)
            self.assertTrue(
                selected.context_adapter.capability_selection["modules"]["implementation_provider"][
                    "active"
                ]
            )
            self.assertIs(selected.spec, original.spec)
            self.assertIs(selected.budget, original.budget)
            self.assertIs(selected.acceptance_adapter, original.acceptance_adapter)


if __name__ == "__main__":
    unittest.main()
