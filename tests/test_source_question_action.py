"""Owned source fixtures;no benchmark candidate,model,or acceptance execution."""

import hashlib
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

from infinicoder.analysis.code_context import CodeIndex
from infinicoder.memory.source.source_question_action import (
    answer_source_question,
    SourceAnswerCapacityError,
)
from infinicoder.memory.source.source_question_command import (
    run_source_question_command,
    question_file,
)
from infinicoder.memory.source.evidence_source import ReadOnlyTaskSources
from infinicoder.planning.task_source_plan import WorkspaceSourcePlanReader, dump


class SourceQuestionActionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.workspace = self.folder / "workspace"
        self.workspace.mkdir()
        self.deadline = time.monotonic() + 30
        self.index = CodeIndex(
            self.workspace, self.folder / "sources.sqlite", native_index_policy="on_demand"
        )
        self.addCleanup(self.index.close)
        # A deterministic fixture counter tests the selector contract only.
        # Installed integration must still supply its real chat tokenizer.
        self.counter = lambda messages, remaining: (len(dump(messages).encode()) + 3) // 4
        self.reader = WorkspaceSourcePlanReader(
            SimpleNamespace(workspace=self.workspace),
            self.index,
            count_output=lambda text: len(text.encode()),
            remaining=lambda: 30,
        )

    def write(self, path, text):
        file = self.workspace / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text, encoding="utf8")

    def ask(self, question, anchor=None, **kwargs):
        return answer_source_question(
            question, anchor, self.reader, self.counter, self.deadline, **kwargs
        )

    def test_export_chain_finds_definition_without_import_or_editing(self):
        self.write("api.py", "from pkg.bridge import compute\n")
        self.write("pkg/bridge.py", "from .impl import transform as compute\n")
        self.write(
            "pkg/impl.py",
            'raise AssertionError("repository code must not execute")\ndef transform(value):\n    return value + 1\n',
        )
        before = {
            p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.workspace.rglob("*.py")
        }
        result = self.ask("Locate api.py::compute")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(
            [r["kind"] for r in result["records"]], ["static_export", "static_export", "definition"]
        )
        self.assertEqual(result["records"][-1]["symbol"], "pkg/impl.py::transform")
        self.assertEqual(set(result["source_files"]), {"api.py", "pkg/bridge.py", "pkg/impl.py"})
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before})
        self.assertLessEqual(self.counter([dict(role="user", content=dump(result))], 30), 2048)

    def test_fields_keep_else_guard_and_runtime_unknown(self):
        self.write(
            "cache.py",
            "class Cache:\n    def __init__(self,enabled):\n        if enabled:\n            self.items = []\n        else:\n            self.items = None\n    def use(self):\n        return self.items\n",
        )
        result = self.ask("Where is self.items initialized?", "cache.py::Cache.use")
        fields = [r for r in result["records"] if r["kind"] == "field_origin"]
        self.assertEqual(len(fields), 2)
        self.assertEqual([r["conditions"][0]["branch"] for r in fields], ["body", "orelse"])
        self.assertTrue(all("runtime_values_unknown" in r["certainty"] for r in fields))
        unknown = self.ask("Who owns self.items?")
        self.assertEqual(unknown["records"], [])
        self.assertTrue(
            any(r.get("status") == "explicit_owner_anchor_required" for r in unknown["unresolved"])
        )

    def test_rebound_and_ambiguous_modules_are_not_chosen(self):
        self.write("api.py", "from impl import compute\nimport replacement as compute\n")
        self.write("impl.py", "def compute(x):\n    return x\n")
        result = self.ask("api.py::compute")
        self.assertEqual(result["records"], [])
        self.assertTrue(
            any(r["status"] == "ambiguous_or_rebound_export" for r in result["unresolved"])
        )
        self.write("api.py", "from impl import compute\n")
        self.write("impl/__init__.py", "def compute(x):\n    return x + 2\n")
        result = self.ask("api.py::compute")
        self.assertTrue(
            any(r["status"] == "module_path_ambiguous_or_missing" for r in result["unresolved"])
        )
        self.assertEqual(set(result["source_files"]), {"api.py"})

    def test_long_definition_is_omitted_whole(self):
        self.write(
            "large.py",
            "def compute(x):\n"
            + "".join("    x = x + 1\n" for _ in range(1200))
            + "    return x\n",
        )
        result = self.ask("large.py::compute")
        self.assertEqual(result["records"], [])
        self.assertTrue(
            any(r["status"] == "whole_record_does_not_fit" for r in result["unresolved"])
        )
        self.assertLessEqual(self.counter([dict(role="user", content=dump(result))], 30), 2048)

    def test_reading_fourth_file_is_bounded_and_stale_read_is_not_current(self):
        for i in range(3):
            self.write(f"m{i}.py", f"from m{i + 1} import compute\n")
        self.write("m3.py", "def compute(x):\n    return x\n")
        result = self.ask("m0.py::compute")
        self.assertEqual(len(result["source_files"]), 3)
        self.assertTrue(any(r["status"] == "source_file_bound" for r in result["unresolved"]))
        self.assertFalse(any(r["kind"] == "definition" for r in result["records"]))
        self.reader.current_version = lambda path: "0" * 64
        result = self.ask("m0.py::compute")
        self.assertEqual(result["status"], "stale_during_question")
        self.assertEqual(result["records"], [])

    def test_command_views_preserve_history_and_mark_changes(self):
        self.write("api.py", "from impl import compute\n")
        self.write("impl.py", "def compute(x):\n    return x + 1\n")
        invoke = lambda args: run_source_question_command(
            self.reader, args, scratch=self.folder, deadline=self.deadline
        )
        first = invoke(["ask", "api.py::compute"])
        ref = first["read_event_ref"]
        self.assertTrue(any("return x + 1" in r["source"] for r in first["records"]))
        self.write("impl.py", "def compute(x):\n    return x + 2\n")
        stale = invoke(["view", ref])
        self.assertEqual(stale["records"], [])
        self.assertIn("impl.py", stale["changed_sources"])
        historical = invoke(["view", ref, "--historical"])
        self.assertEqual(historical["status"], "historical_stale")
        self.assertTrue(any("return x + 1" in r["source"] for r in historical["records"]))
        current = invoke(["ask", "api.py::compute"])
        self.assertTrue(any("return x + 2" in r["source"] for r in current["records"]))
        self.assertEqual(
            self.reader.index.db.execute(
                "SELECT COUNT(*) FROM source_plan_views WHERE ref=?", (ref,)
            ).fetchone()[0],
            1,
        )

    def test_read_capability_and_no_tool_edits(self):
        self.write("api.py", "from private_impl import compute\n")
        self.write("private_impl.py", "SECRET_NOT_AUTHORIZED = 19\ndef compute(x):\n    return x\n")
        sources = ReadOnlyTaskSources(self.workspace, ("api.py",))
        self.reader.session._raw_read = sources.read
        self.reader.workbench._read = sources.read
        self.reader.session._paths = sources.paths
        result = self.ask("api.py::compute")
        self.assertEqual(set(result["source_files"]), {"api.py"})
        self.assertNotIn("SECRET_NOT_AUTHORIZED", dump(result))
        for args in (["edit-body", "api.py::compute"], ["trial", "run"], ["edit-range", "api.py"]):
            with self.assertRaises(ValueError):
                run_source_question_command(
                    self.reader, args, scratch=self.folder, deadline=self.deadline
                )
        self.assertEqual(
            (self.workspace / "api.py").read_text(), "from private_impl import compute\n"
        )

    def test_scratch_question_and_capacity_failure(self):
        self.write("impl.py", "def compute(x):\n    return x\n")
        payload = self.folder / "question.txt"
        payload.write_text("impl.py::compute", encoding="utf8")
        self.assertEqual(
            question_file("/tmp/task-scratch/question.txt", self.folder), "impl.py::compute"
        )
        with self.assertRaises(ValueError):
            question_file("/tmp/task-scratch/../question.txt", self.folder)
        with self.assertRaises(ValueError):
            question_file(str(payload), self.folder)
        self.reader.workbench.count_output = lambda text: 2049
        result = run_source_question_command(
            self.reader,
            ["ask", "--question-file", "/tmp/task-scratch/question.txt"],
            scratch=self.folder,
            deadline=self.deadline,
        )
        self.assertEqual(result["status"], "budget_limited")
        self.assertEqual(result["records"], [])

    def test_current_input_limit_is_separate_from_actual_answer_count(self):
        self.write("impl.py", "def compute(x):\n    return x\n")

        def counter(messages, remaining):
            payload = json.loads(messages[0]["content"])
            if any(row.get("kind") == "definition" for row in payload["records"]):
                raise SourceAnswerCapacityError(300, 3000, 2816)
            return self.counter(messages, remaining)

        result = answer_source_question(
            "impl.py::compute", None, self.reader, counter, self.deadline
        )
        self.assertEqual(result["records"], [])
        self.assertTrue(
            any(
                r["status"] == "whole_record_does_not_fit_current_input"
                for r in result["unresolved"]
            )
        )


if __name__ == "__main__":
    unittest.main()
