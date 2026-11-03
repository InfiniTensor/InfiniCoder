"""Source identity and invalidation only;real tokenizer/SDK controls are separate."""

import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from infinicoder.analysis.code_context import CodeIndex
from infinicoder.planning.task_source_plan import WorkspaceSourcePlanReader
from infinicoder.memory.task_checkpoint_sources import (
    build_checkpoint_sources,
    refresh_checkpoint_sources,
    validate_checkpoint_sources,
)


class SourceCheckpointContracts(unittest.TestCase):
    def test_changed_definition_current_revision_and_missing_body(self):
        old = "class Pool:\n    def size(self):\n        return 0\n    def take(self):\n        return None\n"
        current = old.replace("return None", "return self.free.pop()")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            workspace = root / "workspace"
            workspace.mkdir()
            source = workspace / "pool.py"
            source.write_bytes(current.encode())
            index = CodeIndex(workspace, root / "index.sqlite")
            try:
                reader = WorkspaceSourcePlanReader(
                    SimpleNamespace(workspace=workspace),
                    index,
                    count_output=len,
                    remaining=lambda: 20,
                )
                reader.namespace["binding_identity"] = "a" * 64
                reader.session.inventory = lambda: (_ for _ in ()).throw(
                    AssertionError("whole repository scan is forbidden")
                )
                events = [
                    dict(
                        id=0,
                        message_ids=[2, 3],
                        observer_complete=True,
                        error=None,
                        changed=[],
                        source_read=dict(
                            status="linked_exact_output",
                            path="pool.py",
                            whole_file=True,
                            lines=[1, None],
                            output=old,
                        ),
                    ),
                    dict(
                        id=1,
                        message_ids=[4, 5],
                        observer_complete=True,
                        error=None,
                        changed=["pool.py"],
                    ),
                ]
                budget = dict(
                    remaining=lambda: 20,
                    count=lambda messages: len(str(messages)),
                    source_tokens=20000,
                )
                bundle, info = build_checkpoint_sources(
                    "Keep the public pool interface.", events, reader, {}, budget
                )
                row = bundle.payload["sources"][0]
                self.assertEqual(row["definition"]["symbol"], "Pool.take")
                self.assertEqual(
                    row["definition"]["text"],
                    "    def take(self):\n        return self.free.pop()\n",
                )
                self.assertEqual(
                    row["whole_file_sha256"], hashlib.sha256(source.read_bytes()).hexdigest()
                )
                self.assertEqual(row["prior_definition_read_execution_id"], 0)
                self.assertEqual(
                    row["write_authority"], "original_TaskSpec_and_public_contract_only"
                )
                source.write_bytes(current.replace("self.free.pop()", "self.ready.pop()").encode())
                with self.assertRaises(ValueError):
                    validate_checkpoint_sources(bundle, reader)
                updated, _ = refresh_checkpoint_sources(bundle, reader, budget)
                self.assertNotIn("self.free.pop()", str(updated.payload))
                self.assertIn(
                    "self.ready.pop()", updated.payload["sources"][0]["definition"]["text"]
                )
                source.unlink()
                missing, _ = refresh_checkpoint_sources(updated, reader, budget)
                self.assertEqual(missing.payload["sources"][0]["status"], "unknown")
                self.assertNotIn("definition", missing.payload["sources"][0])
            finally:
                index.close()


if __name__ == "__main__":
    unittest.main()
