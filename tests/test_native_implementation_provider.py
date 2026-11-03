"""Owned CPU lifecycle controls;no benchmark answers,CUDA or neural requests."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

from infinicoder.runtime.agent_execution import (
    CallerPublicOperation,
    CoreBudget,
    CoreRequest,
    UnsupportedCore,
    validate_native_request,
)
from infinicoder.models.budget import ModelProfile
from infinicoder.domains.operator.candidate_provider import ProviderLeaseError
from infinicoder.verification.candidate_public_check import (
    CandidatePublicCheck,
    PublicCandidateVerdict,
)
from infinicoder.domains.operator.convolution_provider import (
    ConvolutionProviderContract,
    assess,
    apply,
    digest,
)
from infinicoder.runtime.native_run import CoreDeadline
from infinicoder.domains.operator.native_implementation_provider import NativeImplementationProvider
from infinicoder.planning.task_contract import TaskSpec
from infinicoder.execution.workspace_tools import Workspace


REFERENCE = """import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self, ci, co, kernel=3):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(ci, co, kernel, bias=False)
    def forward(self, x):
        return self.conv(x)
ci = 3
co = 4
def get_init_inputs():
    return [ci, co]
def get_inputs():
    x = torch.rand(2, ci, 8, 7)
    return [x]
"""
SCOPE = "Implement the complete ModelNew module in solution.py."
TASK = SCOPE + " Use Triton FP32 CUDA forward only.\nREFERENCE:\n" + REFERENCE
STARTER = b"# Owned implementation target\n"


class ProviderControls(unittest.TestCase):
    def setup_case(self, root, *, reference=REFERENCE, behavior="public_pass", reserve=2):
        workspace = root / "workspace"
        workspace.mkdir()
        state = root / "state"
        state.mkdir()
        target = workspace / "solution.py"
        target.write_bytes(STARTER)
        task = SCOPE + " Use Triton FP32 CUDA forward only.\nREFERENCE:\n" + reference
        contract = ConvolutionProviderContract(
            "solution.py", digest(reference), SCOPE, True, reserve
        )
        calls = []
        inputs = []

        def execute_public(deadline):
            calls.append(dict(source=target.read_bytes(), deadline=deadline))
            if isinstance(behavior, BaseException):
                raise behavior
            if callable(behavior):
                return behavior(target)
            return dict(
                status=behavior,
                counterexample="Owned public boundary failure" if behavior == "public_fail" else "",
            )

        operation = CallerPublicOperation(
            "owned-public",
            workspace,
            dict(checker="owned-source-contract"),
            "Owned public source-only lifecycle control",
            dict(scope="AST/source binding,not CUDA correctness"),
            5,
            dict(environment="owned CPU fixture"),
            ("solution.py",),
            (),
            256,
            execute_public,
        )
        check = CandidatePublicCheck(
            operation,
            lambda result: PublicCandidateVerdict(
                result["status"], result.get("counterexample", "")
            ),
            "owned-decoder",
        )
        provider = NativeImplementationProvider(
            lambda messages, timeout: len(json.dumps(messages)) // 4,
            tokenizer_identity="owned-unit-counter-not-model-tokenizer",
            contract=contract,
            public_check=check,
            implementation_paths=("solution.py",),
        )
        request = CoreRequest(
            TaskSpec(task),
            workspace,
            state,
            ModelProfile(16384, 4096),
            CoreBudget(4, 30, 10),
            transport_adapter=SimpleNamespace(delivery_policy=None),
            context_adapter=provider,
        )

        def query(messages, **kwargs):
            inputs.append(messages)
            return dict(role="assistant", content="Owned ordinary SDK-shaped response")

        def execute(action, **kwargs):
            target.write_bytes(target.read_bytes() + b"\n# Owned ordinary model edit\n")
            return dict(output="edited", returncode=0, exception_info="")

        environment = SimpleNamespace(
            execute=execute,
            implementation_workspace_binding=lambda: dict(
                root=str(workspace.resolve()),
                model_root="/workspace",
                namespace_sha256="owned-fixture",
            ),
        )
        model = SimpleNamespace(query=query, _prepare_messages_for_api=lambda messages: messages)
        return SimpleNamespace(**locals())

    def run_query(self, case, *, edit=False):
        messages = [
            dict(role="system", content="Owned original system"),
            dict(role="user", content=case.task),
        ]
        deadline = time.monotonic() + 30
        with case.provider.bind(case.request, case.model, case.environment, deadline):
            with (
                case.operation.bind(case.request, deadline) as bound,
                case.provider.bind_public_operation(bound),
            ):
                first = case.model.query(messages)
                if edit:
                    feedback = case.environment.execute(dict(command="owned edit"))
                    messages = [*messages, first, dict(role="user", content=feedback["output"])]
                    case.model.query(messages)
        return messages

    def test_produced_source_reaches_ordinary_query_and_edit_invalidates_public_result(self):
        with tempfile.TemporaryDirectory() as folder:
            c = self.setup_case(Path(folder))
            self.run_query(c, edit=True)
            self.assertEqual(len(c.calls), 1)
            self.assertEqual(len(c.inputs), 2)
            self.assertIn(b"class ModelNew", c.calls[0]["source"])
            self.assertEqual(
                c.inputs[0][:2],
                [
                    dict(role="system", content="Owned original system"),
                    dict(role="user", content=c.task),
                ],
            )
            self.assertIn('"public_result_version":"current"', c.inputs[0][-1]["content"])
            self.assertIn('"public_status":"public_pass"', c.inputs[0][-1]["content"])
            self.assertIn(
                '"public_result_version":"stale_or_unverified"', c.inputs[1][-1]["content"]
            )
            self.assertIn('"public_status":"unknown"', c.inputs[1][-1]["content"])
            self.assertEqual(c.provider.report["public_executions"], 1)
            self.assertTrue(c.target.read_bytes().endswith(b"# Owned ordinary model edit\n"))
            info = json.loads((c.state / "native-memory-info.json").read_bytes())
            self.assertEqual(info["prepared_provider_views"], 2)
            self.assertEqual(
                info["projection"],
                "Original canonical history plus implementation-module observation",
            )

    def test_unsupported_reference_is_exactly_the_original_input(self):
        with tempfile.TemporaryDirectory() as folder:
            c = self.setup_case(
                Path(folder),
                reference=REFERENCE.replace(
                    "nn.Conv2d(ci, co, kernel, bias=False)", "nn.AvgPool2d(kernel)"
                ),
            )
            original = self.run_query(c)
            self.assertEqual(c.inputs, [original])
            self.assertEqual(c.calls, [])
            self.assertEqual(c.target.read_bytes(), STARTER)
            self.assertFalse((c.state / "native-implementation-provider.json").exists())

    def test_explicit_reference_span_preserves_multisection_task_and_never_guesses_end(self):
        for explicit in (False, True):
            with self.subTest(explicit=explicit), tempfile.TemporaryDirectory() as folder:
                c = self.setup_case(Path(folder))
                end = len(c.task.encode())
                start = end - len(REFERENCE.encode())
                c.task += (
                    "\n\nFILES:\n"
                    + json.dumps({"solution.py": STARTER.decode()})
                    + "\nKeep original public constraints and the CPU-only shell."
                )
                c.request = replace(c.request, spec=TaskSpec(c.task))
                if explicit:
                    c.provider.contract = replace(c.contract, public_reference_span=(start, end))
                original = self.run_query(c)
                self.assertEqual(c.inputs[0][:2], original[:2])
                if explicit:
                    self.assertEqual(len(c.calls), 1)
                    self.assertIn('"public_status":"public_pass"', c.inputs[0][-1]["content"])
                else:
                    self.assertFalse(c.calls)
                    self.assertEqual(c.inputs, [original])
                    self.assertEqual(c.target.read_bytes(), STARTER)

    def test_final_reserve_and_unknown_namespace_do_not_produce_or_change_input(self):
        for reason in ("reserve", "namespace"):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as folder:
                c = self.setup_case(Path(folder), reserve=100 if reason == "reserve" else 2)
                if reason == "namespace":
                    del c.environment.implementation_workspace_binding
                original = self.run_query(c)
                self.assertEqual(c.inputs, [original])
                self.assertFalse(c.calls)
                self.assertEqual(c.target.read_bytes(), STARTER)

    def test_public_failure_retains_real_candidate_and_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            c = self.setup_case(Path(folder), behavior="public_fail")
            self.run_query(c)
            self.assertIn(b"class ModelNew", c.target.read_bytes())
            self.assertIn("Owned public boundary failure", c.inputs[0][-1]["content"])
            self.assertIn('"public_status":"public_fail"', c.inputs[0][-1]["content"])

    def test_environment_error_and_tool_error_restore_only_produced_version(self):
        for behavior in ("env_error", RuntimeError("Owned tool unavailable")):
            with self.subTest(behavior=str(behavior)), tempfile.TemporaryDirectory() as folder:
                c = self.setup_case(Path(folder), behavior=behavior)
                self.run_query(c)
                self.assertEqual(c.target.read_bytes(), STARTER)
                self.assertEqual(c.provider.report["restoration"], "restored_original_starter")
                self.assertIn('"public_status":"unknown"', c.inputs[0][-1]["content"])

    def test_original_cancel_identity_survives_restore_and_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            failure = CoreDeadline("Owned task deadline")
            c = self.setup_case(Path(folder), behavior=failure)
            with self.assertRaises(CoreDeadline) as error:
                self.run_query(c)
            self.assertIs(error.exception, failure)
            self.assertEqual(c.target.read_bytes(), STARTER)
            self.assertFalse(c.inputs)
            self.assertEqual(len(c.calls), 1)

    def test_public_contract_violation_does_not_overwrite_concurrent_source(self):
        def concurrent(target):
            target.write_bytes(b"# Owned concurrent revision\n")
            raise ValueError("Owned source-preservation violation")

        with tempfile.TemporaryDirectory() as folder:
            c = self.setup_case(Path(folder), behavior=concurrent)
            self.run_query(c)
            self.assertEqual(c.target.read_bytes(), b"# Owned concurrent revision\n")
            self.assertEqual(c.provider.report["restoration"], "skipped_current_version_changed")

    def test_scope_resume_and_source_lease_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            c = self.setup_case(Path(folder))
            with self.assertRaises(ValueError):
                replace(c.contract, complete_module_authorized=False)
            for request in (
                replace(c.request, spec=TaskSpec(c.task, allowed_paths=["solution.py"])),
                replace(c.request, resume=True),
            ):
                with self.assertRaises(UnsupportedCore):
                    validate_native_request(request)
            workspace = Workspace(c.workspace, ("solution.py",))
            admission = assess(c.request.spec, workspace, c.contract)
            self.assertEqual(admission["status"], "Applicable")
            c.target.write_bytes(b"# Owned newer source\n")
            with self.assertRaises(ProviderLeaseError):
                apply(c.request.spec, workspace, c.contract, admission)
            self.assertEqual(c.target.read_bytes(), b"# Owned newer source\n")


if __name__ == "__main__":
    unittest.main()
