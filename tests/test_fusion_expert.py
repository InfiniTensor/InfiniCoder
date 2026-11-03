import ast
from dataclasses import replace
from pathlib import Path
import tempfile
import sys
import unittest

from infinicoder import TaskSpec, RuntimeConfig
from infinicoder.domains.operator.candidate_provider import ProviderLeaseError
from infinicoder.domains.operator.convolution_provider import digest
from infinicoder.domains.operator.linear_gelu import generate
from infinicoder.domains.operator.fusion_provider import FusionProviderContract, assess, apply
from infinicoder.execution.workspace_tools import Workspace


REFERENCE = """import torch
import torch.nn.functional as F
def linear_gelu(x, weight, bias):
    return F.gelu(F.linear(x, weight, bias), approximate="tanh")
"""


class FusionExpertTests(unittest.TestCase):
    def contract(self):
        return FusionProviderContract(
            "operator_impl/solution.py",
            REFERENCE,
            digest(REFERENCE),
            "Optimize the operator",
            True,
            60,
        )

    def test_public_composition_has_two_distinct_candidates_and_original_fallback(self):
        candidates = [generate(REFERENCE, schedule=name) for name in ("compact", "wide")]
        self.assertNotEqual(candidates[0].source, candidates[1].source)
        original = ast.parse(REFERENCE).body[-1]
        for candidate in candidates:
            functions = {
                node.name: node
                for node in ast.walk(ast.parse(candidate.source))
                if isinstance(node, ast.FunctionDef)
            }
            self.assertEqual(ast.dump(functions[candidate.entry].args), ast.dump(original.args))
            self.assertEqual(
                ast.dump(functions["_ic_linear_gelu_fallback"].body[0]), ast.dump(original.body[0])
            )
            self.assertEqual(candidate.reference_sha256, digest(REFERENCE))

    def test_semantic_changes_and_shadowed_bindings_are_not_lowered(self):
        changes = [
            REFERENCE.replace('"tanh"', '"none"'),
            REFERENCE.replace("F.linear(x, weight, bias)", "F.linear(weight, x, bias)"),
            REFERENCE + "\nF = object()\n",
            REFERENCE.replace("import torch.nn.functional as F", "import other as F"),
        ]
        for source in changes:
            with self.assertRaises(ValueError):
                generate(source)

    def test_source_permission_and_version_lease_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "operator_impl").mkdir()
            path = root / "operator_impl/solution.py"
            path.write_text(REFERENCE)
            ws = Workspace(root, ("operator_impl/",))
            spec = TaskSpec("Optimize the operator", allowed_paths=["operator_impl/"])
            contract = self.contract()
            admitted = assess(spec, ws, contract)
            self.assertEqual(admitted["status"], "Applicable")
            path.write_text(REFERENCE + "# concurrent edit\n")
            with self.assertRaises(ProviderLeaseError):
                apply(spec, ws, contract, admitted)
            path.write_text(REFERENCE)
            apply(spec, ws, contract, admitted)
            self.assertEqual(digest(path.read_bytes()), admitted["candidate_sha256"])
            self.assertEqual(
                assess(spec, ws, replace(contract, mode="disabled"))["status"], "Unknown"
            )

    def test_common_configuration_keeps_expert_opt_in(self):
        self.assertFalse(RuntimeConfig().fusion_expert)
        self.assertTrue(RuntimeConfig(fusion_expert=True).fusion_expert)

    def test_schedule_selection_uses_external_source_bound_reports(self):
        from test_native_implementation_provider import ProviderControls
        from infinicoder.domains.operator.native_implementation_provider import (
            NativeImplementationProvider,
        )
        from infinicoder.verification.candidate_public_check import contract_hash
        from infinicoder.collaboration.implementation_handoff import (
            ImplementationHandoffPolicy,
            with_specialist_handoff,
            prepare_specialist_handoff,
            consume_specialist_handoff,
        )
        from infinicoder.domains.operator.fusion_schedule_search import with_fusion_schedule_search
        from infinicoder.optimization.native_measured_optimization import PublicPerformanceSample
        from infinicoder.optimization.optimization import Measurement
        import json
        import time

        with tempfile.TemporaryDirectory() as tmp:
            compact_hash = digest(generate(REFERENCE).source)

            def check(target):
                return dict(status="public_pass", pin=digest(target.read_bytes()))

            case = ProviderControls().setup_case(Path(tmp), behavior=check)
            quote = "Replace the complete solution.py module."
            task = quote + "\nREFERENCE:\n" + REFERENCE
            contract = FusionProviderContract(
                "solution.py", REFERENCE, digest(REFERENCE), quote, True, 60
            )
            provider = NativeImplementationProvider(
                lambda messages, timeout: len(json.dumps(messages)) // 4,
                tokenizer_identity="owned-unit-counter",
                contract=contract,
                public_check=case.check,
                implementation_paths=("solution.py",),
            )
            request = replace(
                case.request,
                spec=TaskSpec(task, required_artifacts=["solution.py"]),
                budget=replace(case.request.budget, wall_seconds=180),
                context_adapter=provider,
            )
            with_specialist_handoff(
                request,
                ImplementationHandoffPolicy(
                    digest(task),
                    contract_hash(case.check.manifest()),
                    ("owned-source",),
                    ("owned-source",),
                    False,
                    (),
                    True,
                ),
            )
            with_fusion_schedule_search(
                request,
                performance_decoder=lambda raw: PublicPerformanceSample(
                    {"solution.py": raw["pin"]},
                    True,
                    1,
                    "owned-source-check",
                    Measurement((1.0 if raw["pin"] == compact_hash else 0.5,) * 5),
                    3,
                    "owned-unit-samples-not-GPU-evidence",
                ),
                decoder_identity="owned-source-decoder",
                exclusive_workspace=True,
            )
            deadline = time.monotonic() + 180
            with provider.bind(request, case.model, case.environment, deadline):
                with (
                    case.operation.bind(request, deadline) as op,
                    provider.bind_public_operation(op),
                ):
                    receipt = prepare_specialist_handoff(
                        provider,
                        [dict(role="system", content="owned"), dict(role="user", content=task)],
                    )
                    self.assertIsNotNone(receipt)
                    self.assertTrue(consume_specialist_handoff(provider, receipt))
            self.assertEqual(provider.schedule_search.selected.schedule_name, "wide")
            self.assertEqual(len(case.calls), 2)
            self.assertEqual(
                digest(case.target.read_bytes()),
                digest(generate(REFERENCE, schedule="wide").source),
            )

    def test_runtime_routes_public_source_through_normal_edit_and_acceptance(self):
        from infinicoder import Runtime, Check, ModelConfig, ModelResponse

        class Client:
            config = ModelConfig("http://127.0.0.1:1/v1", "owned-control")
            calls = 0

            def generate(self, messages, timeout):
                self.calls += 1
                return ModelResponse('{"tool":"finish","arguments":{}}')

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "solution.py").write_bytes(REFERENCE.encode())
            client = Client()
            runtime = Runtime(
                workspace,
                client,
                root / "session",
                profile="operator",
                worker_path="solution.py",
                symbol_workers=False,
                automatic_workers=False,
                fusion_expert=True,
                max_model_calls=2,
                allow_commands=True,
                time_limit=30,
            )
            check = Check(
                "owned-source-contract",
                [
                    sys.executable,
                    "-c",
                    "import ast,pathlib; s=pathlib.Path('solution.py').read_text(); ast.parse(s); "
                    "assert 'def _ic_linear_gelu_kernel' in s; assert 'def linear_gelu' in s",
                ],
            )
            state = runtime.execute(
                TaskSpec(
                    "Optimize the existing linear GELU operator using Triton; preserve the complete module interface.",
                    allowed_paths=["solution.py"],
                    checks=[check],
                )
            )
            self.assertEqual(state["status"], "succeeded", state.get("reason"))
            self.assertIn("_ic_linear_gelu_kernel", (workspace / "solution.py").read_text())


if __name__ == "__main__":
    unittest.main()
