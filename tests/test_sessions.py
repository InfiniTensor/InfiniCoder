"""Public session lifecycle:explicit resume,pause and retained call limits."""

import json
import hashlib
from dataclasses import replace
from pathlib import Path
import tempfile
import sys
import unittest

from infinicoder import Check, Runtime, TaskSpec, run, RuntimeConfig, create_runtime
from infinicoder.models.model_client import ModelConfig, ModelResponse


class SessionLifecycle(unittest.TestCase):
    def test_requirement_revision_preserves_budget_and_policy(self):
        original = b"def value():\n    return 1\n"
        actions = iter(
            [
                dict(tool="finish", arguments={}),
                dict(tool="read", arguments=dict(path="value.py")),
                dict(
                    tool="edit",
                    arguments=dict(
                        path="value.py",
                        expected_hash=hashlib.sha256(original).hexdigest(),
                        old="return 1",
                        new="return 2",
                    ),
                ),
                dict(tool="finish", arguments={}),
            ]
        )

        class Client:
            config = ModelConfig("http://127.0.0.1:1/v1", "scripted")
            calls = 0

            def generate(self, messages, timeout):
                self.calls += 1
                return ModelResponse(
                    json.dumps(next(actions)),
                    usage=dict(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                )

        def task(expected):
            return TaskSpec(
                f"Make value() return {expected}.",
                allowed_paths=["value.py"],
                checks=[
                    Check(
                        "acceptance",
                        [
                            sys.executable,
                            "-c",
                            f"from value import value; assert value() == {expected}",
                        ],
                    )
                ],
            )

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "value.py").write_bytes(original)
            client = Client()
            options = RuntimeConfig(
                profile="general",
                max_model_calls=4,
                automatic_workers=False,
                domain_compiler=False,
                allow_commands=True,
            )
            runtime = create_runtime(workspace, client, root / "session", config=options)
            first = runtime.execute(task(1))
            self.assertEqual(first["status"], "succeeded")
            before = (root / "session/state.json").read_bytes()
            for changed in (
                replace(options, max_model_calls=5),
                replace(options, domain_compiler=True),
            ):
                other = create_runtime(workspace, client, root / "session", config=changed)
                with self.assertRaisesRegex(ValueError, "configuration changed"):
                    other.revise(task(2), "The caller now needs a different value.")
                self.assertEqual((root / "session/state.json").read_bytes(), before)
                self.assertFalse((root / "session/revisions").exists())
            runtime.revise(task(2), "The caller now needs a different value.")
            revised = runtime.store.load()
            self.assertEqual(revised["configuration_identity"], first["configuration_identity"])
            self.assertEqual(revised["execution_history_policy"], first["execution_history_policy"])
            final = runtime.execute(task(2), resume=True)
            self.assertEqual(final["status"], "succeeded", final["reason"])
            self.assertEqual(final["model_calls"], 4)
            self.assertEqual(client.calls, 4)

    def runtime(self, root, client):
        workspace = root / "workspace"
        workspace.mkdir(exist_ok=True)
        return Runtime(
            workspace,
            client,
            root / "session",
            profile="general",
            automatic_workers=False,
            domain_compiler=False,
            code_memory=False,
            development_workflows=False,
            max_steps=5,
            max_model_calls=1,
            time_limit=60,
        )

    def test_resume_requires_a_saved_session(self):
        class Client:
            config = ModelConfig("http://127.0.0.1:1/v1", "unused")

            def generate(self, *args, **kwargs):
                raise AssertionError("A missing session must fail before model dispatch")

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            runtime = self.runtime(root, Client())
            with self.assertRaisesRegex(FileNotFoundError, "no saved session"):
                runtime.execute(TaskSpec("Inspect the workspace."), resume=True)
            self.assertFalse((root / "session/state.json").exists())
            self.assertFalse((root / "session/writer.lock").exists())
            with self.assertRaisesRegex(ValueError, "state_dir"):
                run("Inspect the workspace.", root / "workspace", client=Client(), resume=True)

    def test_pause_and_resume_keep_the_consumed_model_budget(self):
        class Client:
            config = ModelConfig("http://127.0.0.1:1/v1", "scripted")
            calls = 0

            def generate(self, messages, timeout):
                self.calls += 1
                runtime.store.request_control("pause")
                return ModelResponse(
                    json.dumps(dict(tool="note", arguments=dict(text="Inspected the task."))),
                    usage=dict(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                )

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            client = Client()
            runtime = self.runtime(root, client)
            task = TaskSpec(
                "Inspect the workspace.",
                checks=[
                    Check(
                        "caller",
                        ["python", "-c", 'raise AssertionError("Final must not run during pause")'],
                    )
                ],
            )
            paused = runtime.execute(task)
            self.assertEqual(paused["status"], "paused")
            self.assertEqual(paused["model_calls"], 1)
            resumed = runtime.execute(task, resume=True)
            self.assertEqual(client.calls, 1)
            self.assertEqual(resumed["model_calls"], 1)
            self.assertNotEqual(resumed["status"], "succeeded")


if __name__ == "__main__":
    unittest.main()
