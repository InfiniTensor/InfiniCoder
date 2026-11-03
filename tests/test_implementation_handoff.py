"""Owned source-contract controls;not GPU mathematics or benchmark scores."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
import unittest

from test_native_implementation_provider import ProviderControls, REFERENCE
from infinicoder.verification.candidate_public_check import contract_hash
from infinicoder.verification.candidate_public_check import CandidatePublicCheck
from infinicoder.domains.operator.convolution_provider import digest
from infinicoder.collaboration.implementation_handoff import (
    ImplementationHandoffPolicy,
    prepare_specialist_handoff,
    consume_specialist_handoff,
    with_specialist_handoff,
)
from infinicoder.planning.task_contract import TaskSpec


class HandoffControls(unittest.TestCase):
    def case(self, root, **kwargs):
        case = ProviderControls().setup_case(root, **kwargs)
        quote = "Retain a correct implementation even when this speed goal is not reached."
        case.task = quote + "\n" + case.task
        case.request = replace(
            case.request, spec=TaskSpec(case.task, required_artifacts=["solution.py"])
        )
        policy = ImplementationHandoffPolicy(
            digest(case.task),
            contract_hash(case.check.manifest()),
            ("owned-source-ABI",),
            ("owned-source-ABI",),
            False,
            (quote,),
            True,
        )
        return case, policy

    def test_actual_producer_public_receipt_then_changed_source_cannot_handoff(self):
        with tempfile.TemporaryDirectory() as folder:
            c, policy = self.case(Path(folder))
            original = c.request
            self.assertIs(with_specialist_handoff(c.request, policy), original)
            messages = [
                dict(role="system", content="owned original"),
                dict(role="user", content=c.task),
            ]
            deadline = time.monotonic() + 30
            with c.provider.bind(c.request, c.model, c.environment, deadline):
                with (
                    c.operation.bind(c.request, deadline) as op,
                    c.provider.bind_public_operation(op),
                ):
                    receipt = prepare_specialist_handoff(c.provider, messages)
                    self.assertIsNotNone(receipt)
                    self.assertEqual(
                        receipt.soft_aspirations_not_established, policy.soft_aspiration_quotes
                    )
                    self.assertTrue(consume_specialist_handoff(c.provider, receipt))
                    c.target.write_bytes(
                        c.target.read_bytes() + b"\n# changed after public verification\n"
                    )
                    changed = c.target.read_bytes()
                    self.assertFalse(consume_specialist_handoff(c.provider, receipt))
                    self.assertEqual(c.target.read_bytes(), changed)
            self.assertEqual(len(c.calls), 1)
            self.assertEqual(c.inputs, [])

    def test_hard_goal_missing_coverage_and_task_mismatch_keep_ordinary_query(self):
        for reason in ("hard_performance", "missing_coverage", "task_mismatch"):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as folder:
                c, policy = self.case(Path(folder))
                if reason == "hard_performance":
                    policy = replace(policy, hard_performance_required=True)
                elif reason == "missing_coverage":
                    policy = replace(policy, public_pass_covers=())
                else:
                    policy = replace(policy, task_sha256="0" * 64)
                with_specialist_handoff(c.request, policy)
                messages = [
                    dict(role="system", content="owned original"),
                    dict(role="user", content=c.task),
                ]
                deadline = time.monotonic() + 30
                with c.provider.bind(c.request, c.model, c.environment, deadline):
                    with (
                        c.operation.bind(c.request, deadline) as op,
                        c.provider.bind_public_operation(op),
                    ):
                        self.assertIsNone(prepare_specialist_handoff(c.provider, messages))
                        self.assertFalse(c.calls)
                        c.model.query(messages)
                self.assertEqual(len(c.inputs), 1)
                self.assertEqual(c.inputs[0][:2], messages)
                self.assertEqual(len(c.calls), 1)

    def test_dependency_changed_during_public_check_cannot_receive_handoff(self):
        with tempfile.TemporaryDirectory() as folder:

            def change_dependency(target):
                (target.parent / "dependency.py").write_text("version = 2\n")
                return dict(status="public_pass")

            c, policy = self.case(Path(folder), behavior=change_dependency)
            dependency = c.workspace / "dependency.py"
            dependency.write_text("version = 1\n")
            c.operation = replace(c.operation, observed_paths=("solution.py", "dependency.py"))
            c.check = CandidatePublicCheck(c.operation, c.check.decoder, c.check.decoder_identity)
            c.provider.public_check = c.check
            c.provider.paths = c.operation.observed_paths
            policy = replace(policy, public_check_sha256=contract_hash(c.check.manifest()))
            with_specialist_handoff(c.request, policy)
            messages = [
                dict(role="system", content="owned original"),
                dict(role="user", content=c.task),
            ]
            deadline = time.monotonic() + 30
            with c.provider.bind(c.request, c.model, c.environment, deadline):
                with (
                    c.operation.bind(c.request, deadline) as op,
                    c.provider.bind_public_operation(op),
                ):
                    self.assertIsNone(prepare_specialist_handoff(c.provider, messages))
                    self.assertFalse(c.calls)
                    c.model.query(messages)
            self.assertEqual(dependency.read_text(), "version = 2\n")
            self.assertEqual(len(c.inputs), 1)
            record = json.loads((c.state / "native-specialist-handoff.json").read_bytes())
            self.assertEqual(record["reason"], "single_source_handoff_only")

    def test_unsupported_reference_is_exact_original_packet_and_public_failure_is_not_ready(self):
        for reason in ("not_applicable", "public_fail", "unknown"):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as folder:
                kwargs = (
                    dict(
                        reference=REFERENCE.replace(
                            "nn.Conv2d(ci, co, kernel, bias=False)", "nn.AvgPool2d(kernel)"
                        )
                    )
                    if reason == "not_applicable"
                    else dict(behavior=reason)
                )
                c, policy = self.case(Path(folder), **kwargs)
                with_specialist_handoff(c.request, policy)
                messages = [
                    dict(role="system", content="owned original"),
                    dict(role="user", content=c.task),
                ]
                deadline = time.monotonic() + 30
                with c.provider.bind(c.request, c.model, c.environment, deadline):
                    with (
                        c.operation.bind(c.request, deadline) as op,
                        c.provider.bind_public_operation(op),
                    ):
                        self.assertIsNone(prepare_specialist_handoff(c.provider, messages))
                        c.model.query(messages)
                self.assertEqual(len(c.inputs), 1)
                self.assertEqual(len(c.calls), 0 if reason == "not_applicable" else 1)
                if reason == "not_applicable":
                    self.assertEqual(c.inputs[0], messages)
                record = json.loads((c.state / "native-specialist-handoff.json").read_bytes())
                self.assertEqual(record["status"], "ordinary_fallback")


if __name__ == "__main__":
    unittest.main()
