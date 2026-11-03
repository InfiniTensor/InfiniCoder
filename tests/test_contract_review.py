"""Source/proposal boundaries;actual SDK/readonly execution is one installed group."""

from contextlib import contextmanager
from types import SimpleNamespace
import unittest

from infinicoder.runtime.agent_execution import AuxiliaryCallBudget, UnsupportedCore
from infinicoder.planning.contract_review import changed_source, digest, lightweight, parse_review
from infinicoder.verification.readonly_diagnostic import ReadonlyCpuDiagnostic


class ContractReviewTest(unittest.TestCase):
    def test_complete_changed_member_and_public_basis_without_oracle(self):
        before = "class Model:\n    def value(self,x):\n        return x\n"
        after = before.replace("return x", "return x+1")
        view = changed_source("model.py", before, after)
        self.assertEqual(
            view["current_complete_units"][0]["text"], "    def value(self,x):\n        return x+1"
        )
        self.assertEqual(view["sha"], digest(after))
        task = "Inputs are ordinary Python lists. Preserve the public API."
        import json

        proposal = dict(
            public_basis=[dict(quote="Inputs are ordinary Python lists.")],
            concern="input hypothesis",
            probe=dict(kind="cpu_python", script="assert [1,2].size(0)==2"),
            expected_observation="unverified input assumption",
        )
        parsed = parse_review(
            json.dumps(proposal),
            complete=True,
            public_task=task,
            sources={"model.py": after},
            public_operation=False,
            cpu_probe=True,
        )
        self.assertEqual(parsed["authority"], "unverified_model_hypothesis")
        self.assertIn("not input applicability", parsed["basis_validation"])
        proposal["public_basis"] = [dict(quote="This nonexistent quote is not a public contract.")]
        with self.assertRaises(ValueError):
            parse_review(
                json.dumps(proposal),
                complete=True,
                public_task=task,
                sources={},
                public_operation=False,
                cpu_probe=True,
            )
        with self.assertRaises(ValueError):
            parse_review(
                '{"status":"none"}',
                complete=False,
                public_task=task,
                sources={},
                public_operation=False,
                cpu_probe=True,
            )

    def test_auxiliary_call_reduces_sdk_budget_once_even_when_parser_fails(self):
        class Transport:
            def remaining_physical_calls(self):
                return 8

            @contextmanager
            def limit_request(self, **kw):
                self.limits = kw
                yield

        t = Transport()
        agent = SimpleNamespace(config=SimpleNamespace(step_limit=10))
        calls = []

        def query(*args, **kw):
            calls.append(kw)
            raise ValueError("owned parse error")

        import time

        budget = AuxiliaryCallBudget(
            SimpleNamespace(transport_adapter=t),
            agent,
            query,
            time.monotonic() + 10,
            public_task="Original public contract without SDK action formatting",
        )
        self.assertEqual(
            budget.public_task, "Original public contract without SDK action formatting"
        )
        with self.assertRaises(ValueError):
            budget.query_once([], output_tokens=500, absolute_deadline=time.monotonic() + 5)
        self.assertEqual(budget.call_units, 1)
        self.assertEqual(agent.config.step_limit, 9)
        self.assertEqual(t.limits["max_physical_calls"], 1)
        with self.assertRaises(UnsupportedCore):
            budget.query_once([], output_tokens=500, absolute_deadline=time.monotonic() + 5)
        self.assertEqual(len(calls), 1)
        # Raw completion never enters the pinned SDK's own retry/action loop.
        raw_calls = []

        def raw(messages, **kw):
            raw_calls.append(kw)
            raise TimeoutError("owned upstream timeout")

        budget = AuxiliaryCallBudget(
            SimpleNamespace(transport_adapter=t),
            agent,
            query,
            time.monotonic() + 10,
            raw_query=raw,
            prepare=lambda m: m,
        )
        with self.assertRaises(TimeoutError):
            budget.query_once([], output_tokens=500, absolute_deadline=time.monotonic() + 5)
        self.assertEqual(len(raw_calls), 1)
        self.assertEqual(len(calls), 1)

    def test_readonly_namespace_uses_exact_builder_and_rejects_extra_write_alias(self):
        from pathlib import Path

        runner = object.__new__(ReadonlyCpuDiagnostic)
        runner.workspace = Path("/workspace-source")

        def builder(root, command):
            return [
                "bwrap",
                "--die-with-parent",
                "--dev",
                "/dev",
                "--bind",
                str(root),
                "/workspace",
            ] + command

        runner.execution = SimpleNamespace(sandbox_argv=builder)
        argv = runner._argv(["/bin/true"])
        self.assertIn("--unshare-net", argv)
        self.assertIn("--unshare-pid", argv)
        i = argv.index(str(runner.workspace))
        self.assertEqual(argv[i - 1], "--ro-bind")
        runner.execution.sandbox_argv = lambda root, command: (
            builder(root, command)[: -len(command)]
            + ["--bind", "/somewhere", "/another-write"]
            + command
        )
        with self.assertRaises(UnsupportedCore):
            runner._argv(["/bin/true"])
        self.assertTrue(lightweight({"one.py": "def f(x):\n    return x\n"}))
        self.assertFalse(lightweight({"one.py": "import torch\ndef f(x):\n    return x\n"}))


if __name__ == "__main__":
    unittest.main()
