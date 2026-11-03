"""Owned lifecycle tests;source control is not numerical or GPU performance proof."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from test_native_implementation_provider import ProviderControls, SCOPE
from infinicoder.domains.operator.candidate_provider import ProviderLeaseError
from infinicoder.domains.operator.convolution_provider import digest
from infinicoder.app.development_api import configure_development_request
from infinicoder.domains.operator.rmsnorm import generate
from infinicoder.memory.source.evidence_source import ReadOnlyTaskSources
from infinicoder.domains.operator.native_implementation_provider import NativeImplementationProvider
from infinicoder.runtime.native_plain import NativePlainObserver
from infinicoder.domains.operator.rmsnorm_provider import RMSNormProviderContract, assess, apply
from infinicoder.planning.task_contract import TaskSpec
from infinicoder.execution.workspace_tools import Workspace


REFERENCE = """import torch
from torch import nn
class Model(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.RMSNorm(normalized_shape, eps=eps)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)
"""


class RMSNormProviderTests(unittest.TestCase):
    def case(self, root):
        c = ProviderControls().setup_case(root, reference=REFERENCE)
        c.task = c.task.replace("CUDA forward only.", "CUDA inference-only forward only.")
        c.contract = RMSNormProviderContract("solution.py", digest(REFERENCE), SCOPE, True, 2)
        c.provider = NativeImplementationProvider(
            lambda messages, timeout: len(json.dumps(messages)) // 4,
            tokenizer_identity="owned-unit-counter-not-real-tokenizer",
            contract=c.contract,
            public_check=c.check,
            implementation_paths=("solution.py",),
        )
        c.request = replace(c.request, spec=TaskSpec(c.task), context_adapter=c.provider)
        return c

    def test_parameter_field_and_constructor_are_preserved(self):
        candidate = generate(REFERENCE)
        self.assertEqual(candidate.descriptor.norm_field, "norm")
        self.assertIn("self.norm = nn.RMSNorm(normalized_shape, eps=eps)", candidate.source)
        self.assertIn(
            "self.norm.weight, self.norm.eps, self.norm.normalized_shape", candidate.source
        )
        self.assertIn("col < N", candidate.source)

    def test_noncanonical_side_effects_and_changed_math_are_rejected(self):
        changes = [
            ("return self.norm(x)", "return self.norm(x) + x"),
            ("nn.RMSNorm", "nn.LayerNorm"),
            ("super().__init__()", "super().__init__()\n        self.extra = print(1)"),
            ("eps=eps", 'eps=eval("1e-5")'),
        ]
        for old, new in changes:
            with self.subTest(change=new), self.assertRaises(ValueError):
                generate(REFERENCE.replace(old, new))
        with self.assertRaises(ValueError):
            generate('raise AssertionError("must not execute")\n' + REFERENCE)

    def test_source_lease_rejects_concurrent_target_and_task_change(self):
        with tempfile.TemporaryDirectory() as folder:
            c = self.case(Path(folder))
            ws = Workspace(c.workspace, ("solution.py",))
            admitted = assess(c.request.spec, ws, c.contract)
            self.assertEqual(admitted["status"], "Applicable")
            c.target.write_bytes(c.target.read_bytes() + b"# concurrent\n")
            with self.assertRaises(ProviderLeaseError):
                apply(c.request.spec, ws, c.contract, admitted)
            c.target.write_bytes(admitted["target_before"])
            changed = replace(c.request.spec, task=c.task + "\nNew caller requirement")
            with self.assertRaises(ProviderLeaseError):
                apply(changed, ws, c.contract, admitted)

    def test_produced_code_is_consumed_and_later_edit_invalidates_feedback(self):
        with tempfile.TemporaryDirectory() as folder:
            c = self.case(Path(folder))
            ProviderControls().run_query(c, edit=True)
            self.assertEqual(c.provider.producer_family, "RMSNorm inference")
            self.assertEqual(len(c.calls), 1)
            self.assertIn(b"_ic_rmsnorm_kernel", c.calls[0]["source"])
            self.assertIn('"public_status":"public_pass"', c.inputs[0][-1]["content"])
            self.assertIn('"public_status":"unknown"', c.inputs[1][-1]["content"])

    def test_unified_entry_selects_typed_operator_and_preserves_request(self):
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
                rmsnorm_provider=asdict(c.contract),
            )
            self.assertIs(selected.spec, original.spec)
            self.assertIs(selected.acceptance_adapter, original.acceptance_adapter)
            self.assertIs(selected.transport_adapter, original.transport_adapter)
            self.assertIsInstance(selected.context_adapter, NativeImplementationProvider)
            self.assertEqual(selected.context_adapter.contract, c.contract)
            with self.assertRaises(ValueError):
                configure_development_request(
                    original,
                    source_capability=ReadOnlyTaskSources(c.workspace, ("solution.py",)),
                    convolution_provider={},
                    rmsnorm_provider=c.contract,
                )

    def test_missing_inference_declaration_does_not_apply(self):
        with tempfile.TemporaryDirectory() as folder:
            c = self.case(Path(folder))
            task = replace(c.request.spec, task=c.task.replace("inference-only", ""))
            before = c.target.read_bytes()
            result = assess(task, Workspace(c.workspace, ("solution.py",)), c.contract)
            self.assertEqual(result["status"], "Unknown")
            self.assertEqual(c.target.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
