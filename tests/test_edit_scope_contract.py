"""Caller scope mistakes must fail before paid model or process dispatch."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from infinicoder import Runtime, TaskSpec, CompatibleClient, ModelConfig
from infinicoder.runtime.agent_execution import create_runtime_request, execute_agent
from infinicoder.execution.workspace_tools import Workspace


class EditScopeContract(unittest.TestCase):
    def test_runtime_and_shared_api_reject_patterns_without_model_or_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            target = workspace / "code.py"
            target.write_text("value = 1\n", encoding="utf-8")
            for index, (scope, shared_api) in enumerate(
                ((["pkg/**"], False), (["pkg/*.py"], True), (["pkg/file?.py"], False))
            ):
                with self.subTest(scope=scope, shared_api=shared_api):
                    client = CompatibleClient(
                        ModelConfig("http://127.0.0.1:1/v1", "owned-never-called")
                    )
                    client.generate = Mock(side_effect=AssertionError("model must not be called"))
                    runtime = Runtime(
                        workspace,
                        client,
                        root / ("session" + str(index)),
                        profile="general",
                        domain_compiler=False,
                        automatic_workers=False,
                        development_workflows=False,
                        code_memory=False,
                    )
                    runtime._process = Mock(
                        side_effect=AssertionError("process must not be called")
                    )
                    spec = TaskSpec(
                        "Repair the source while keeping the caller scope.", allowed_paths=scope
                    )
                    with self.assertRaisesRegex(ValueError, "allowed_paths: wildcard paths"):
                        if shared_api:
                            execute_agent(create_runtime_request(runtime, spec), runtime=runtime)
                        else:
                            runtime.execute(spec)
                    client.generate.assert_not_called()
                    runtime._process.assert_not_called()
                    self.assertIsNone(runtime.store.load())
                    self.assertEqual(target.read_text(), "value = 1\n")

    def test_milestone_pattern_is_rejected_even_under_valid_parent_directory(self):
        with self.assertRaisesRegex(ValueError, r"milestones\[0\].write_set: wildcard paths"):
            TaskSpec.from_dict(
                dict(
                    task="Implement a module.",
                    allowed_paths=["pkg/"],
                    milestones=[
                        dict(goal="Implement one phase", check_ids=[], write_set=["pkg/*.py"])
                    ],
                )
            )

    def test_literal_paths_directory_permissions_and_empty_scope_keep_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "pkg" / "state[0].py"
            path.parent.mkdir()
            original = b"value = 1\n"
            path.write_bytes(original)
            for scope in (["pkg/"], ["pkg/state[0].py"], []):
                with self.subTest(scope=scope):
                    spec = TaskSpec.from_dict(
                        dict(task="Update the literal file.", allowed_paths=scope)
                    )
                    self.assertEqual(spec.allowed_paths, scope)
                    ws = Workspace(root, spec.allowed_paths)
                    result = ws.edit(
                        "pkg/state[0].py",
                        hashlib.sha256(original).hexdigest(),
                        old="value = 1",
                        new="value = 2",
                    )
                    self.assertEqual(path.read_bytes(), b"value = 2\n")
                    self.assertNotEqual(result["before"], result["after"])
                    if scope:
                        with self.assertRaisesRegex(ValueError, "outside the task edit scope"):
                            ws.path("other.py", write=True)
                    path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
