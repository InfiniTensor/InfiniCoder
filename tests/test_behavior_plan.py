"""Literal provenance,task-based routing and bounded file admission."""

import tempfile
import unittest
from pathlib import Path
from infinicoder.planning.behavior_plan import BehaviorPlanPolicy, parse_script, read_script, route


class BehaviorPlanTests(unittest.TestCase):
    def test_public_task_requirements_drive_route_instead_of_two_line_stub(self):
        task = "Across calls, retain the running total. Use CPU Python and preserve interfaces."
        result = route(task, observed_paths=["impl.py"], cpu_capability=True)
        self.assertTrue(result["applicable"])
        self.assertTrue(all(i["quote"] in task for i in result["public_quotes"]))
        self.assertFalse(
            route(
                task + " All tensors require CUDA.", observed_paths=["impl.py"], cpu_capability=True
            )["applicable"]
        )
        self.assertFalse(route(task, observed_paths=[], cpu_capability=True)["applicable"])
        self.assertFalse(
            route(task, observed_paths=["impl.py"], cpu_capability=False)["applicable"]
        )
        self.assertFalse(
            route(
                "Complete this pure function. Files in /tmp do not persist between commands.",
                observed_paths=["one.py"],
                cpu_capability=True,
            )["applicable"]
        )

    def test_scenario_is_never_imported_and_expected_values_remain_hypotheses(self):
        task = "Across calls, retain the running total."
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "behavior-plan.py"
            text = (
                "PUBLIC_BASIS="
                + repr(task)
                + "\ndef exercise(workspace,scratch):\n    raise AssertionError('model expectation is not an oracle')\n"
            )
            path.write_text(text, encoding="utf8")
            value = read_script(folder)
            plan = parse_script(value, task)
            self.assertEqual(plan["public_basis"], task)
            self.assertIn("unverified", plan["authority"])
            self.assertIn("provenance only", plan["basis_validation"])
            with self.assertRaises(ValueError):
                parse_script(value, "An unrelated task without that public quote.")
            path.write_text(
                text.replace("exercise(workspace,scratch)", "exercise(workspace,scratch,extra)"),
                encoding="utf8",
            )
            with self.assertRaises(ValueError):
                parse_script(read_script(folder), task)
            path.write_bytes(b"x" * 16385)
            with self.assertRaises(ValueError):
                read_script(folder)

    def test_source_mentions_are_not_public_requirements(self):
        import json

        files = "\n\nFILES:\n" + json.dumps(
            {
                "impl.py": "# CUDA checkpoint helper mentioned in unrelated source\ndef f(x): return x\n"
            }
        )
        task = "Across calls, retain the running total." + files
        value = route(task, observed_paths=["impl.py"], cpu_capability=True)
        self.assertTrue(value["applicable"])
        self.assertTrue(
            all(
                task[i["offset"] : i["offset"] + len(i["quote"])] == i["quote"]
                for i in value["public_quotes"]
            )
        )
        self.assertFalse(
            route(
                "Complete a pure integer function." + files,
                observed_paths=["impl.py"],
                cpu_capability=True,
            )["applicable"]
        )

    def test_limits_cannot_expand_the_single_workflow_budget(self):
        for kwargs in (
            {"max_runs": 4},
            {"max_script_versions": 3},
            {"context_tokens": 769},
            {"run_seconds": 21},
            {"total_seconds": 61},
            {"max_runs": True},
        ):
            with self.assertRaises(ValueError):
                BehaviorPlanPolicy(**kwargs)
        self.assertEqual(BehaviorPlanPolicy().max_runs, 3)


if __name__ == "__main__":
    unittest.main()
