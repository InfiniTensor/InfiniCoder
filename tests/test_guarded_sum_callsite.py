"""Owned integration controls;no target model,reference patch or private grader."""

from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from infinicoder import TaskSpec, GuardedSumContract, prepare_guarded_sum, apply_guarded_sum
from infinicoder.search.candidate_checkpoint import CandidateCheckpointStore
from infinicoder.domains.operator.candidate_provider import ProviderLeaseError, UnknownPrerequisite
from infinicoder.domains.operator.reduction import generate
from infinicoder.execution.workspace_tools import Workspace


class Fixture:
    def __init__(self, root):
        self.root = root
        self.workspace = root / "workspace"
        self.workspace.mkdir()
        package = self.workspace / "ic_owned223_pkg"
        package.mkdir()
        (package / "__init__.py").write_text("")
        self.source = (
            'import torch\nSENTINEL = "original caller"\n\n'
            'def reduce(x):\n    """Owned generic reduction."""\n    return torch.sum(x, dim=0)\n\n'
            "def consumer(x):\n    return SENTINEL, reduce(x)\n"
        )
        self.target = "ic_owned223_pkg/api.py"
        self.dispatch = "ic_owned223_pkg/_ic_dispatch.py"
        self.backend = "ic_owned223_pkg/_ic_backend.py"
        (self.workspace / self.target).write_bytes(self.source.encode())
        self.quote = (
            "Modify reduce in "
            + self.target
            + " and create "
            + self.dispatch
            + " and "
            + self.backend
            + " only."
        )
        self.numeric = json.dumps(
            dict(
                allow_fp32_sum_reassociation=True,
                rtol=1e-5,
                atol=1e-5,
                equal_nan=True,
                signed_zero="not_preserved",
            )
        )
        self.paths = (self.target, self.dispatch, self.backend)
        self.spec = TaskSpec(
            self.quote + "\nNumerical agreement: " + self.numeric, allowed_paths=self.paths
        )
        self.contract = GuardedSumContract(
            self.target,
            "reduce",
            hashlib.sha256(self.source.encode()).hexdigest(),
            (6, 11, 6, 30),
            self.dispatch,
            self.backend,
            self.quote,
            self.numeric,
            1e-5,
            1e-5,
            enabled=True,
            allow_reassociation=True,
        )
        self.ws = Workspace(self.workspace, self.paths, validate_python=True)

    def install(self, schedule="scalar", ws=None):
        bundle = prepare_guarded_sum(self.spec, self.ws, self.contract, schedule=schedule)
        store = CandidateCheckpointStore(
            self.workspace,
            self.root / "checkpoints",
            exclusive_workspace=True,
            remaining=lambda: 60,
        )
        return apply_guarded_sum(
            self.spec, ws or self.ws, self.contract, bundle, checkpoint_store=store
        )

    @contextmanager
    def imported(self):
        sys.path.insert(0, str(self.workspace))
        try:
            yield (
                importlib.import_module("ic_owned223_pkg.api"),
                importlib.import_module("ic_owned223_pkg._ic_dispatch"),
            )
        finally:
            sys.path.remove(str(self.workspace))
            for name in list(sys.modules):
                if name == "ic_owned223_pkg" or name.startswith("ic_owned223_pkg."):
                    sys.modules.pop(name)


class SourceContractControls(unittest.TestCase):
    def test_old_lowering_programs_are_byte_identical(self):
        examples = json.loads(
            (Path(__file__).parent / "fixtures/sum217_lowering_bytes.json").read_bytes()
        )
        for case in examples["examples"]:
            for schedule, pin in case["schedule_sha256"].items():
                self.assertEqual(
                    hashlib.sha256(generate(case["source"], schedule).source.encode()).hexdigest(),
                    pin,
                )

    def test_only_local_return_changes_and_helpers_are_lazy(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp))
            bundle = prepare_guarded_sum(f.spec, f.ws, f.contract, schedule="coalesced")
            text = dict(bundle.artifacts)[f.target].decode()
            self.assertEqual(text.split("def reduce", 1)[0], f.source.split("def reduce", 1)[0])
            self.assertEqual(text.split("def consumer", 1)[1], f.source.split("def consumer", 1)[1])
            self.assertIn('"""Owned generic reduction."""', text)
            self.assertEqual(len(bundle.artifacts), 3)
            self.assertFalse((f.workspace / f.dispatch).exists())

    def test_scope_numeric_and_source_drift_fail_before_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp))
            with self.assertRaises(UnknownPrerequisite):
                prepare_guarded_sum(f.spec, f.ws, replace(f.contract, enabled=False))
            with self.assertRaises(ValueError):
                replace(f.contract, numerical_quote="approximate is probably okay")
            with self.assertRaises(ValueError):
                prepare_guarded_sum(replace(f.spec, allowed_paths=(f.target,)), f.ws, f.contract)
            with self.assertRaises(ProviderLeaseError):
                prepare_guarded_sum(replace(f.spec, task="other requirement"), f.ws, f.contract)
            (f.workspace / f.target).write_text(f.source + "# caller changed\n")
            with self.assertRaises(ProviderLeaseError):
                prepare_guarded_sum(f.spec, f.ws, f.contract)
            self.assertFalse((f.workspace / f.dispatch).exists())


@unittest.skipUnless(
    os.name == "posix" and importlib.util.find_spec("torch"),
    "real installed Torch and POSIX checkpoint required",
)
class InstalledConsumerControls(unittest.TestCase):
    def test_generic_CPU_integer_empty_and_grad_preserve_original(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp))
            f.install()
            with f.imported() as (api, adapter):
                values = [
                    torch.arange(12).reshape(3, 4),
                    torch.empty(0, 4),
                    torch.randn(3, 4),
                    torch.randn(3, 4, requires_grad=True),
                ]
                for x in values:
                    tag, result = api.consumer(x)
                    reference = torch.sum(x, dim=0)
                    self.assertEqual(tag, "original caller")
                    torch.testing.assert_close(result, reference, rtol=0, atol=0)
                    self.assertEqual(result.requires_grad, reference.requires_grad)
                values[-1].grad = None
                api.reduce(values[-1]).sum().backward()
                torch.testing.assert_close(values[-1].grad, torch.ones_like(values[-1]))
                with torch.inference_mode():
                    for x in values[:3]:
                        torch.testing.assert_close(
                            api.reduce(x), torch.sum(x, dim=0), rtol=0, atol=0
                        )
                with torch.autograd.forward_ad.dual_level(), torch.inference_mode():
                    self.assertEqual(
                        adapter.fallback_reason(values[2], torch.sum),
                        "forward_AD_or_unknown_context",
                    )
                    torch.testing.assert_close(
                        api.reduce(values[2]), torch.sum(values[2], dim=0), rtol=0, atol=0
                    )
                self.assertNotIn("ic_owned223_pkg._ic_backend", sys.modules)

    def test_subclass_modes_and_callable_override_do_not_bypass_original(self):
        import torch
        from torch.overrides import TorchFunctionMode
        from torch.utils._python_dispatch import TorchDispatchMode

        class Sub(torch.Tensor):
            pass

        class FunctionMode(TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append(func)
                return func(*args, **(kwargs or {}))

        class DispatchMode(TorchDispatchMode):
            def __init__(self):
                self.calls = []

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                self.calls.append(func)
                return func(*args, **(kwargs or {}))

        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp))
            f.install()
            with f.imported() as (api, adapter):
                x = torch.randn(3, 4)
                self.assertIs(type(api.reduce(x.as_subclass(Sub))), Sub)
                for Mode in (FunctionMode, DispatchMode):
                    mode = Mode()
                    with torch.inference_mode(), mode:
                        result = api.reduce(x)
                    self.assertEqual(len(mode.calls), 1)
                    torch.testing.assert_close(result, torch.sum(x, dim=0))
                original = torch.sum
                calls = []

                def override(*args, **kwargs):
                    calls.append((args, kwargs))
                    return original(*args, **kwargs) + 7

                with patch.object(torch, "sum", override), torch.inference_mode():
                    result = api.reduce(x)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0][1], {"dim": 0})
                torch.testing.assert_close(result, original(x, dim=0) + 7)
                self.assertNotIn("ic_owned223_pkg._ic_backend", sys.modules)

    def test_missing_backend_falls_back_but_selected_failure_propagates(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp))
            f.install()
            (f.workspace / f.backend).unlink()
            with f.imported() as (api, adapter):
                x = torch.randn(3, 4)
                # Owned routing injection isolates backend failure policy;this is
                # not evidence that CPU inputs pass the real CUDA guard.
                with patch.object(adapter, "fallback_reason", lambda *args: None):
                    torch.testing.assert_close(api.reduce(x), torch.sum(x, dim=0), rtol=0, atol=0)

                    def broken(_):
                        raise RuntimeError("owned launch failure")

                    adapter._FAST = broken
                    with self.assertRaisesRegex(RuntimeError, "owned launch failure"):
                        api.reduce(x)

    def test_interrupted_multi_file_install_uses_original_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp))

            class Interrupted(Workspace):
                def edit(self, path, *args, **kwargs):
                    if path == f.backend:
                        raise KeyboardInterrupt("owned cancellation")
                    return super().edit(path, *args, **kwargs)

            with self.assertRaises(KeyboardInterrupt):
                f.install(ws=Interrupted(f.workspace, f.paths))
            self.assertEqual((f.workspace / f.target).read_text(), f.source)
            self.assertFalse((f.workspace / f.dispatch).exists())
            self.assertFalse((f.workspace / f.backend).exists())

    def test_CLI_installs_the_same_scoped_artifact_and_fallback_consumer(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp))
            task = f.root / "task.json"
            contract = f.root / "contract.json"
            task.write_text(json.dumps(asdict(f.spec)))
            contract.write_text(json.dumps(asdict(f.contract)))
            command = [
                sys.executable,
                "-B",
                "-m",
                "infinicoder.domains.operator.guarded_sum_callsite",
                "--workspace",
                str(f.workspace),
                "--task-spec",
                str(task),
                "--contract",
                str(contract),
                "--apply",
                "--exclusive-workspace",
                "--state-dir",
                str(f.root / "checkpoints"),
            ]
            result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
            self.assertEqual(json.loads(result.stdout)["status"], "installed_not_verified")
            with f.imported() as (api, adapter):
                x = torch.ones(3, 4)
                torch.testing.assert_close(api.consumer(x)[1], torch.sum(x, dim=0), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
