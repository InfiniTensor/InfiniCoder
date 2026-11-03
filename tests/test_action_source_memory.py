"""Selection/storage contracts only;no tokenizer or coding-capability claims."""

import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from infinicoder.memory.action_source_memory import (
    SourceFocusArchive,
    SourceSnapshot,
    choose_focus,
    observe_revision,
    read_snapshot,
    render_focus,
)
from infinicoder.analysis.code_context import CodeIndex
from infinicoder.planning.task_source_plan import WorkspaceSourcePlanReader
from infinicoder.memory.task_state_checkpoint import dump


NAMESPACE = dict(
    environment="owned-test-workspace", root="/workspace", binding_identity="fixture-only"
)


def snapshot(path, text, namespace=NAMESPACE):
    raw = text.encode()
    return SourceSnapshot(path, dump(namespace), raw, hashlib.sha256(raw).hexdigest())


def event(number, **extra):
    return dict(
        id=number,
        message_ids=[2 + number * 2, 3 + number * 2],
        observer_complete=True,
        error=None,
        **extra,
    )


class ActionSourceMemoryContracts(unittest.TestCase):
    def test_changed_function_and_read_range_move_focus_without_shortest_fallback(self):
        old = "def tiny():\n    return 0\n\ndef update(values):\n    total = 0\n    for value in values:\n        total += value\n    return total\n"
        before = snapshot("state.py", old)
        after = snapshot("state.py", old.replace("total += value", "total += value * 2"))
        edit = event(1, changed=["state.py"])
        revision = observe_revision(
            "state.py",
            NAMESPACE,
            before,
            after,
            1,
            edit["message_ids"],
            before_version=before.version,
        )
        focus = choose_focus([edit], [revision], [after], "Keep the original public interface.")
        self.assertEqual([r.definition["symbol"] for r in focus], ["update"])
        self.assertEqual(focus[0].before, before)
        with self.assertRaises(ValueError):
            observe_revision(
                "state.py",
                NAMESPACE,
                before,
                after,
                1,
                edit["message_ids"],
                before_version="f" * 64,
            )
        dependency = snapshot(
            "interface.py",
            "def short():\n    return 1\n\ndef adapt(value):\n    scaled = value * 3\n    return scaled\n",
        )
        read = event(
            2,
            changed=[],
            source_read=dict(
                path="interface.py",
                lines=[5, 5],
                whole_file=False,
                status="linked_exact_output",
                output="    scaled = value * 3\n",
            ),
        )
        moved = choose_focus(
            [edit, read], [revision], [after, dependency], "Keep the original public interface."
        )
        self.assertEqual([r.definition["symbol"] for r in moved], ["adapt", "update"])
        whole = event(
            3,
            changed=[],
            source_read=dict(
                path="interface.py",
                lines=[1, None],
                whole_file=True,
                status="linked_exact_output",
                output=dependency.text,
            ),
        )
        self.assertEqual(choose_focus([whole], [], [dependency], "Inspect the module."), ())
        explicit = choose_focus([whole], [], [dependency], "Inspect interface.py::adapt.")
        self.assertEqual([r.definition["symbol"] for r in explicit], ["adapt"])
        self.assertEqual(
            choose_focus([whole], [], [dependency], "Inspect interface.py::adapt.other."), ()
        )

    def test_changed_method_is_distinct_from_changed_enclosing_class_contract(self):
        old = "class Pool:\n    limit = 2\n    def tiny(self):\n        return 0\n    def release(self, value):\n        self.free.append(value)\n        return len(self.free)\n"
        before = snapshot("pool.py", old)
        after = snapshot(
            "pool.py", old.replace("self.free.append(value)", "self.free.append(value + 1)")
        )
        edit = event(0, changed=["pool.py"])
        revision = observe_revision(
            "pool.py",
            NAMESPACE,
            before,
            after,
            0,
            edit["message_ids"],
            before_version=before.version,
        )
        focus = choose_focus([edit], [revision], [after], "Keep the public Pool contract.")
        self.assertEqual([r.definition["symbol"] for r in focus], ["Pool.release"])
        changed_contract = snapshot("pool.py", old.replace("limit = 2", "limit = 3"))
        revision = observe_revision(
            "pool.py",
            NAMESPACE,
            before,
            changed_contract,
            0,
            edit["message_ids"],
            before_version=before.version,
        )
        focus = choose_focus(
            [edit], [revision], [changed_contract], "Keep the public Pool contract."
        )
        self.assertEqual([r.definition["symbol"] for r in focus], ["Pool"])

    def test_durable_namespace_record_and_whole_definition_budget(self):
        source = snapshot(
            "state.py", "def tiny():\n    return 0\n\ndef update(value):\n    return value + 1\n"
        )
        read = event(
            0,
            changed=[],
            source_read=dict(
                path="state.py",
                lines=[5, 5],
                whole_file=False,
                status="linked_exact_output",
                output="    return value + 1\n",
            ),
        )
        focus = choose_focus([read], [], [source], "Keep interfaces.")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "state.py").write_bytes(source.raw)
            index = CodeIndex(workspace, root / "memory.sqlite")
            try:
                reader = WorkspaceSourcePlanReader(
                    SimpleNamespace(workspace=workspace),
                    index,
                    count_output=len,
                    remaining=lambda: 20,
                )
                reader.namespace = dict(NAMESPACE)
                archive = SourceFocusArchive(reader)
                self.assertEqual(read_snapshot(reader, "state.py", lambda: None), source)
                budget = dict(
                    count=lambda messages: len(str(messages)),
                    remaining=lambda: None,
                    source_tokens=10000,
                )
                view = render_focus(focus, archive, {"state.py": source.version}, budget)
                self.assertEqual(view["status"], "ready")
                self.assertIn("def update(value):\n    return value + 1", view["text"])
                self.assertNotIn("source_utf8", view["text"])
                alias = view["included"][0]["alias"]
                record = archive.get(alias)
                self.assertEqual(
                    record["view"]["records"][0]["after"]["source_utf8"], source.raw.decode()
                )
                reader.namespace = dict(NAMESPACE, binding_identity="other-task")
                with self.assertRaises(ValueError):
                    SourceFocusArchive(reader).get(alias)
                with self.assertRaises(ValueError):
                    archive.put(focus[0])
                reader.namespace = dict(NAMESPACE)
                stale = render_focus(focus, archive, {"state.py": "0" * 64}, budget)
                self.assertEqual(stale["included"], [])
                self.assertEqual(stale["text"], "")
                self.assertEqual(archive.get(alias), record)
                small = render_focus(
                    focus, archive, {"state.py": source.version}, dict(budget, source_tokens=150)
                )
                self.assertEqual(small["included"], [])
                self.assertNotIn("def update", small["text"])
            finally:
                index.close()

    def test_invalid_source_abstains_without_turning_missing_history_into_a_change(self):
        broken = snapshot("state.py", "def update(value):\n    return (\n")
        read = event(
            0,
            changed=[],
            source_read=dict(
                path="state.py",
                lines=[1, None],
                whole_file=True,
                status="linked_exact_output",
                output=broken.text,
            ),
        )
        diagnostics = []
        self.assertEqual(
            choose_focus(
                [read], [], [broken], "Inspect state.py::update.", diagnostics=diagnostics
            ),
            (),
        )
        self.assertEqual(diagnostics[0]["status"], "unknown")
        current = snapshot(
            "state.py", "def tiny():\n    return 0\n\ndef update(value):\n    return value + 1\n"
        )
        edit = event(1, changed=["state.py"])
        revision = observe_revision(
            "state.py", NAMESPACE, None, current, 1, edit["message_ids"], before_version="0" * 64
        )
        self.assertEqual(
            choose_focus([edit], [revision], [current], "Inspect the changed function."), ()
        )
        other = snapshot("state.py", current.text, dict(NAMESPACE, binding_identity="other-task"))
        with self.assertRaises(ValueError):
            choose_focus([read], [], [current, other], "Inspect state.py::update.")


if __name__ == "__main__":
    unittest.main()
