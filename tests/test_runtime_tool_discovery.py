"""Actual ordinary Runtime discovery/edit/check/resume controls,not model scores."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

from infinicoder import Runtime, TaskSpec, Check, ModelResponse
from infinicoder.tools import tool_palette  # noqa: F401 -- import compatibility smoke check


def names(schema):
    result = set()
    if "tool" in schema.get("properties", {}):
        result.add(schema["properties"]["tool"]["const"])
    for branch in schema.get("anyOf", []):
        result.update(names(branch))
    return result


def action(name, **args):
    return dict(tool=name, arguments=args)


class RuntimeDiscovery(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.ws = self.root / "workspace"
        self.ws.mkdir()
        (self.ws / "a.py").write_text("def transform(x):return -100\n", encoding="utf-8")
        checker = self.root / "check.py"
        checker.write_text(
            'import sys,runpy\nf=runpy.run_path(sys.argv[1]+"/a.py")["transform"]\nassert [f(x) for x in (-3,0,9)]==[4,7,16]\n',
            encoding="utf-8",
        )
        self.spec = TaskSpec(
            "Implement a.transform(x) as x+7;preserve the public interface.",
            allowed_paths=["a.py"],
            checks=[
                Check(
                    "external",
                    [sys.executable, str(checker), "{workspace}"],
                    protected_paths=[str(checker)],
                )
            ],
        )

    def client(self, respond, structured=True):
        class Client:
            supports_structured_output = structured
            config = SimpleNamespace(max_output_tokens=512)

            def __init__(self):
                self.requests = []

            def generate_structured(self, messages, schema, timeout):
                return self.reply(messages, schema)

            def generate(self, messages, timeout):
                return self.reply(messages, None)

            def reply(self, messages, schema):
                self.requests.append((messages, schema))
                value = respond(len(self.requests), messages, schema)
                return ModelResponse(
                    json.dumps(value),
                    usage=dict(prompt_tokens=3, completion_tokens=2, total_tokens=5),
                )

        return Client()

    def runtime(self, client, **extra):
        options = dict(
            tool_discovery=True,
            automatic_workers=False,
            allow_commands=False,
            profile="general",
            domain_compiler=False,
            max_model_calls=12,
            max_steps=15,
            time_limit=30,
        )
        options.update(extra)
        return Runtime(self.ws, client, self.root / "state", **options)

    def edit(self, path="a.py"):
        return action(
            "edit",
            path=path,
            expected_hash=hashlib.sha256((self.ws / "a.py").read_bytes()).hexdigest(),
            old="return -100",
            new="return x+7",
        )

    def test_real_discover_read_edit_verify_finish(self):
        def respond(n, messages, schema):
            offered = names(schema)
            if n == 1:
                self.assertNotIn("use_def", offered)
                self.assertIn("tool_catalog", offered)
                self.assertIn('"code"', messages[-1]["content"])
                return action("tool_catalog", group="code")
            if n == 2:
                self.assertIn("use_def", offered)
                return action("read", path="a.py")
            if n == 3:
                return self.edit()
            if n == 4:
                return action("verify")
            return action("finish")

        runtime = self.runtime(self.client(respond))
        state = runtime.execute(self.spec)
        self.assertEqual(state["status"], "succeeded", state.get("last_failure"))
        self.assertEqual(state["model_calls"], 5)
        self.assertFalse(runtime.shell_loop)
        self.assertEqual(state["tool_palette"]["group"], "code")
        self.assertTrue(
            any(row["kind"] == "tool_palette_selected" for row in runtime.store.events())
        )

    def test_hidden_tools_rejected_before_dispatch_and_all_restores(self):
        def respond(n, messages, schema):
            if n == 1:
                return action("use_def", path="a.py")
            if n == 2:
                return action("tool_catalog", group="all")
            if n == 3:
                self.assertIn("use_def", names(schema))
                return action("tool_catalog", group="basic")
            if n == 4:
                self.assertNotIn("use_def", names(schema))
                return self.edit()
            return action("finish")

        runtime = self.runtime(self.client(respond))
        state = runtime.execute(self.spec)
        self.assertEqual(state["status"], "succeeded", state.get("last_failure"))
        starts = [
            row["data"]["tool"] for row in runtime.store.events() if row["kind"] == "action_started"
        ]
        self.assertNotIn("use_def", starts)

    def test_catalog_does_not_grant_command_permission(self):
        def respond(n, messages, schema):
            self.assertNotIn("command", names(schema))
            return (
                action("tool_catalog", group="all")
                if n == 1
                else action(
                    "command", argv=[sys.executable, "-c", 'raise RuntimeError("must not run")']
                )
            )

        runtime = self.runtime(self.client(respond), max_model_calls=2)
        state = runtime.execute(self.spec)
        self.assertNotEqual(state["status"], "succeeded")
        self.assertFalse(
            any(
                row["kind"] == "action_started" and row["data"]["tool"] == "command"
                for row in runtime.store.events()
            )
        )

    def test_permission_revoked_during_request_cannot_execute(self):
        marker = self.root / "should-not-exist"

        def respond(n, messages, schema):
            self.assertIn("command", names(schema))
            runtime.allow_commands = False
            return action(
                "command",
                argv=[
                    sys.executable,
                    "-c",
                    f"from pathlib import Path;Path({str(marker)!r}).touch()",
                ],
            )

        runtime = self.runtime(self.client(respond), allow_commands=True, max_model_calls=1)
        runtime.execute(self.spec)
        self.assertFalse(marker.exists())
        self.assertFalse(
            any(
                row["kind"] == "action_started" and row["data"]["tool"] == "command"
                for row in runtime.store.events()
            )
        )

    def test_pause_resume_preserves_group_and_call_count(self):
        def first(n, messages, schema):
            if n == 1:
                return action("tool_catalog", group="code")
            runtime.store.request_control("pause")
            return action("read", path="a.py")

        runtime = self.runtime(self.client(first))
        paused = runtime.execute(self.spec)
        self.assertEqual(paused["status"], "paused")
        self.assertEqual(paused["model_calls"], 2)

        def resumed(n, messages, schema):
            self.assertIn("use_def", names(schema))
            return self.edit() if n == 1 else action("finish")

        runtime.client = self.client(resumed)
        state = runtime.execute(self.spec, resume=True)
        self.assertEqual(state["status"], "succeeded", state.get("last_failure"))
        self.assertEqual(state["model_calls"], 4)

    def test_planning_and_worker_schemas_are_unchanged(self):
        runtime = self.runtime(self.client(lambda *_: action("finish")))
        state = dict(no_progress={})
        from infinicoder.execution.workspace_tools import Workspace

        workspace = Workspace(self.ws, self.spec.allowed_paths)
        for stage in ("clarification", "planning"):
            runtime.tool_discovery = True
            enabled = runtime._response_schema(stage, self.spec, state, workspace)
            runtime.tool_discovery = False
            disabled = runtime._response_schema(stage, self.spec, state, workspace)
            self.assertEqual(enabled, disabled)
            self.assertNotIn("tool_catalog", names(enabled))

    def test_unstructured_client_gets_the_actual_schema_in_context(self):
        def respond(n, messages, schema):
            self.assertIsNone(schema)
            self.assertIn("response_schema", messages[-1]["content"])
            return self.edit() if n == 1 else action("finish")

        state = self.runtime(self.client(respond, structured=False)).execute(self.spec)
        self.assertEqual(state["status"], "succeeded", state.get("last_failure"))

    def test_off_is_default_and_legacy_backends_keep_their_contract(self):
        import inspect
        from infinicoder import run

        self.assertIs(inspect.signature(Runtime).parameters["tool_discovery"].default, False)
        self.assertIs(inspect.signature(run).parameters["tool_discovery"].default, False)
        with self.assertRaisesRegex(ValueError, "must be bool"):
            self.runtime(self.client(lambda *_: None), tool_discovery=1)
        with self.assertRaisesRegex(ValueError, "ordinary Runtime"):
            self.runtime(self.client(lambda *_: None), shell_loop=True, allow_commands=True)
        with self.assertRaisesRegex(ValueError, "ordinary Runtime"):
            self.runtime(self.client(lambda *_: None), command_worker=True, allow_commands=True)

    def test_public_run_api_keeps_the_complete_task_and_checks(self):
        from infinicoder import run

        def respond(n, messages, schema):
            self.assertIn("tool_catalog", names(schema))
            return self.edit() if n == 1 else action("finish")

        state = run(
            self.spec.task,
            self.ws,
            client=self.client(respond),
            checks=self.spec.checks,
            allowed_paths=self.spec.allowed_paths,
            state_dir=self.root / "api-state",
            tool_discovery=True,
            automatic_workers=False,
            profile="general",
            domain_compiler=False,
            max_model_calls=4,
            time_limit=30,
        )
        self.assertEqual(state["status"], "succeeded", state.get("last_failure"))


if __name__ == "__main__":
    unittest.main()
