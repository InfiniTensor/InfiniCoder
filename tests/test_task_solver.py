"""Public workflow selection preserves task authority and reports actual execution."""

from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
import shutil
import time
from unittest.mock import patch

from infinicoder import (
    Check,
    ModelConfig,
    ModelResponse,
    TaskSpec,
    SolvePolicy,
    inspect_task,
    solve,
)
from infinicoder.runtime.agent_execution import CoreBudget, UnsupportedCore


class TaskWorkflows(unittest.TestCase):
    def test_pause_resume_charges_setup_and_preserves_original_budget(self):
        from infinicoder.app.config import create_runtime
        from infinicoder.session.session import SessionStore

        class Client:
            config = ModelConfig("http://127.0.0.1:1/v1", "scripted", max_output_tokens=256)
            calls = 0

            def generate(self, messages, timeout):
                self.calls += 1
                return ModelResponse("def answer():\n    return 42\n")

        setup = iter((0.02, 0.04))

        def delayed_runtime(*args, **kwargs):
            time.sleep(next(setup))
            return create_runtime(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "answer.py").write_text("def answer():\n    return 0\n")
            spec = TaskSpec(
                "Make answer() return 42.",
                allowed_paths=["answer.py"],
                checks=[
                    Check(
                        "acceptance",
                        [sys.executable, "-c", "from answer import answer; assert answer()==42"],
                    )
                ],
            )
            client = Client()
            state_dir = root / "state"
            SessionStore(state_dir).request_control("pause")
            options = dict(
                model_config=client.config,
                client=client,
                budget=CoreBudget(3, 30, 10),
                allow_commands=True,
            )
            before = time.time()
            with patch("infinicoder.app.config.create_runtime", side_effect=delayed_runtime):
                paused = solve(spec, workspace, state_dir, **options)
                self.assertEqual(paused["status"], "paused")
                self.assertEqual(client.calls, 0)
                self.assertGreaterEqual(paused["session_deadline_epoch"], before + 30)
                self.assertLess(
                    paused["session_deadline_epoch"] - paused["pause_started_epoch"], 29.99
                )
                saved = (state_dir / "state.json").read_bytes()
                with self.assertRaisesRegex(ValueError, "configuration changed"):
                    changed = dict(options, budget=CoreBudget(3, 60, 10))
                    # Validate changed limits without consuming the controlled setup delays.
                    with patch("infinicoder.app.config.create_runtime", create_runtime):
                        solve(spec, workspace, state_dir, resume=True, **changed)
                self.assertEqual((state_dir / "state.json").read_bytes(), saved)
                resumed = solve(spec, workspace, state_dir, resume=True, **options)
            self.assertEqual(resumed["status"], "succeeded", resumed.get("reason"))
            self.assertEqual(client.calls, 1)
            self.assertEqual(resumed["configuration_identity"], paused["configuration_identity"])
            self.assertAlmostEqual(
                resumed["session_deadline_epoch"] - paused["session_deadline_epoch"],
                resumed["excluded_user_wait_s"],
                places=5,
            )
            self.assertEqual(resumed["model_calls"], 1)

    @unittest.skipUnless(sys.platform == "linux" and shutil.which("bwrap"), "Linux CPU namespace")
    def test_public_repairs_and_independent_final_have_separate_authority(self):
        class Client:
            config = ModelConfig("http://127.0.0.1:1/v1", "scripted", max_output_tokens=256)
            calls = 0
            requests = []

            def generate(self, messages, timeout):
                self.calls += 1
                self.requests.append(json.dumps(messages))
                return ModelResponse(f"def answer():\n    return {41 if self.calls == 1 else 42}\n")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "answer.py").write_text("def answer():\n    return 0\n")
            public, final = root / "public.py", root / "final.py"
            script = (
                "import unittest\nfrom answer import answer\n"
                "class TestAnswer(unittest.TestCase):\n"
                " def test_answer(self): self.assertEqual(answer(), VALUE, 'MARKER')\n"
                "unittest.main()\n"
            )
            public.write_text(script.replace("VALUE", "42").replace("MARKER", "public-repair"))
            final.write_text(script.replace("VALUE", "43").replace("MARKER", "private-final"))
            client = Client()
            result = solve(
                TaskSpec("Make answer() return 42.", allowed_paths=["answer.py"]),
                workspace,
                root / "state",
                model_config=client.config,
                client=client,
                budget=CoreBudget(4, 30, 10),
                public_check_script=public,
                check_script=final,
            )
            self.assertEqual(result["status"], "succeeded")
            self.assertFalse(result["success"])
            self.assertEqual(client.calls, 2)
            self.assertIn("public-repair", "\n".join(client.requests))
            self.assertNotIn("private-final", "\n".join(client.requests))
            report = json.loads((root / "state/capabilities.json").read_text())
            self.assertFalse(report["success"])
            self.assertTrue(report["modules"]["public_feedback"]["activated"])
            self.assertEqual(report["modules"]["final_acceptance"]["outcome"], "failed")

    def test_native_domain_and_memory_selection_preserve_contract(self):
        spec = TaskSpec("Repair the training framework across optimizer.py and checkpoint.py")
        original = asdict(spec)
        report = inspect_task(
            spec,
            ".",
            policy=SolvePolicy(backend="native"),
            source_paths=("optimizer.py", "checkpoint.py"),
            plan_paths=("optimizer.py", "checkpoint.py"),
        )
        self.assertEqual(report["native_profile"], "cpu-auto")
        self.assertTrue(report["modules"]["source_memory"]["selected"])
        self.assertTrue(report["modules"]["planning"]["selected"])
        self.assertTrue(report["modules"]["domain_workflow"]["selected"])
        self.assertIsNone(report["modules"]["planning"]["activated"])
        self.assertEqual(asdict(spec), original)

    def test_scopes_and_context_owners_are_not_silently_changed(self):
        scoped = TaskSpec("Fix a framework", allowed_paths=["package/"])
        report = inspect_task(scoped, ".", policy=SolvePolicy(backend="native", context="task"))
        self.assertEqual(report["native_profile"], "cpu-scoped")
        self.assertTrue(report["modules"]["task_memory"]["selected"])
        self.assertFalse(report["modules"]["source_memory"]["selected"])
        with self.assertRaises(UnsupportedCore):
            inspect_task(scoped, ".", policy=SolvePolicy(backend="native", context="source"))
        with self.assertRaises(UnsupportedCore):
            inspect_task(
                TaskSpec("Fix it", checks=[Check("test", ["python", "test.py"])]),
                ".",
                policy=SolvePolicy(backend="native"),
            )

    def test_baseline_and_insufficient_planning_budget_remain_explicit(self):
        spec = TaskSpec("Implement a framework across modules")
        baseline = inspect_task(spec, ".", policy=SolvePolicy(backend="native", enhancements=False))
        self.assertEqual(baseline["native_profile"], "cpu-plain")
        self.assertFalse(any(m["selected"] for m in baseline["modules"].values()))
        report = inspect_task(
            spec,
            ".",
            policy=SolvePolicy(backend="native"),
            source_paths=("a.py", "b.py"),
            plan_paths=("a.py",),
            budget=CoreBudget(5, 60, 20),
        )
        self.assertFalse(report["modules"]["planning"]["selected"])

    def test_shared_runtime_edits_checks_and_records_without_credentials(self):
        class Client:
            config = ModelConfig(
                "http://127.0.0.1:1/v1", "scripted", api_key="private", max_output_tokens=256
            )
            calls = 0

            def generate(self, messages, timeout):
                self.calls += 1
                return ModelResponse("def answer():\n    return 42\n", usage={"total_tokens": 12})

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "answer.py").write_text("def answer():\n    return 0\n")
            task = TaskSpec(
                "Make answer() return 42.",
                allowed_paths=["answer.py"],
                checks=[
                    Check(
                        "acceptance",
                        [sys.executable, "-c", "from answer import answer; assert answer()==42"],
                    )
                ],
            )
            client = Client()
            result = solve(
                task,
                workspace,
                root / "state",
                model_config=client.config,
                client=client,
                budget=CoreBudget(3, 30, 10),
                allow_commands=True,
            )
            self.assertEqual(result["status"], "succeeded", result.get("reason"))
            report = (root / "state/capabilities.json").read_text(encoding="utf-8")
            self.assertNotIn("private", report)
            value = json.loads(report)
            self.assertTrue(value["success"])
            self.assertIsNone(value["modules"]["implementation_worker"]["activated"])
            self.assertTrue(value["modules"]["final_acceptance"]["activated"])
            self.assertIsNone(value["modules"]["implementation_worker"]["delivered"])
            with self.assertRaises(ValueError):
                solve(
                    task,
                    workspace,
                    root / "state",
                    model_config=client.config,
                    client=client,
                    allow_commands=True,
                )
            self.assertEqual((root / "state/capabilities.json").read_text(encoding="utf-8"), report)


if __name__ == "__main__":
    unittest.main()
