"""Owned cross-file execution and next-query controls;no benchmark answers."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from infinicoder.runtime.agent_execution import CallerPublicOperation, CoreBudget, CoreRequest
from infinicoder.models.budget import ModelProfile
from infinicoder.verification.native_public_failure_slice import NativePublicFailureSlice
from infinicoder.verification.public_replay import (
    PublicReplaySpec,
    PublicReplayCapability,
    PublicTraceRecorder,
    describe_public_value,
    digest,
)
from infinicoder.planning.task_contract import TaskSpec


SOURCE = """class Store:
    def scale(self, value, divisor):
        return value / divisor
"""
ENTRY = """from core import Store
def reproduce():
    return Store().scale(12, 0)
reproduce()
"""


def owned_trace(workspace, max_events=96):
    name = "owned_public_trace_core"
    spec = importlib.util.spec_from_file_location(name, workspace / "core.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recorder = PublicTraceRecorder(workspace, ("core.py", "repro.py"), max_events=max_events)
    try:
        with recorder:
            module.Store().scale(12, 0)
    except ZeroDivisionError:
        pass
    report = recorder.report()
    report.update(
        source_after={p: digest((workspace / p).read_bytes()) for p in ("core.py", "repro.py")},
        public_argv=["python", "repro.py"],
    )
    return report


class PublicSliceControls(unittest.TestCase):
    def setup_case(self, root, *, command="python repro.py", error=None, counter=None):
        workspace = root / "workspace"
        workspace.mkdir()
        state = root / "state"
        state.mkdir()
        (workspace / "core.py").write_bytes(SOURCE.encode())
        (workspace / "repro.py").write_bytes(ENTRY.encode())
        spec = PublicReplaySpec(
            ("python", "repro.py"),
            ("core.py", "repro.py"),
            {"repro.py": digest(ENTRY.encode())},
            "Owned public cross-file reproduction",
            timeout_seconds=5,
            final_reserve_seconds=1,
        )
        calls = []
        inputs = []

        def replay(deadline):
            calls.append(deadline)
            if error:
                raise error
            return dict(trace=owned_trace(workspace))

        operation = CallerPublicOperation(
            "owned-replay",
            workspace,
            dict(worker="owned-trace"),
            spec.public_origin,
            dict(public=True),
            5,
            dict(namespace="owned"),
            spec.source_paths,
            (),
            1024,
            replay,
        )
        observer = NativePublicFailureSlice(
            counter or (lambda messages, timeout: len(json.dumps(messages)) // 4),
            tokenizer_identity="owned-unit-counter",
            capability=PublicReplayCapability(spec, operation),
        )
        task = "Owned public task. Run python repro.py,repair core.py;preserve repro.py."
        request = CoreRequest(
            TaskSpec(task),
            workspace,
            state,
            ModelProfile(16384, 4096),
            CoreBudget(4, 30, 5),
            transport_adapter=SimpleNamespace(delivery_policy=None),
            context_adapter=observer,
        )
        model = SimpleNamespace(
            query=lambda messages, **kw: (
                inputs.append(messages) or dict(role="assistant", content="ordinary")
            ),
            _prepare_messages_for_api=lambda messages: messages,
        )
        environment = SimpleNamespace(
            execute=lambda action, **kw: dict(
                returncode=1, output="ZeroDivisionError", exception_info=""
            ),
            implementation_workspace_binding=lambda: dict(
                root=str(workspace.resolve()), model_root="/workspace", mapping="owned"
            ),
        )
        return SimpleNamespace(**locals())

    def lifecycle(self, c, *, mutate=False):
        messages = [dict(role="system", content="Owned system"), dict(role="user", content=c.task)]
        with c.observer.bind(c.request, c.model, c.environment, time.monotonic() + 30):
            answer = c.model.query(messages)
            result = c.environment.execute(dict(command=c.command))
            messages = [*messages, answer, dict(role="user", content=result["output"])]
            if mutate:
                (c.workspace / "core.py").write_text(SOURCE + "# newer source\n")
            c.model.query(messages)
        return messages

    def test_real_method_parameters_and_exception_without_wrapping(self):
        with tempfile.TemporaryDirectory() as folder:
            c = self.setup_case(Path(folder))
            report = owned_trace(c.workspace)
            call = next(x for x in report["events"] if x["kind"] == "call")
            self.assertEqual(call["parameters"]["value"]["value"], 12)
            self.assertEqual(call["parameters"]["divisor"]["value"], 0)
            self.assertEqual(call["parameters"]["self"]["type"], "Store")
            self.assertTrue(
                any(x.get("exception_type") == "ZeroDivisionError" for x in report["events"])
            )
            self.assertIsNone(sys.gettrace())

    def test_same_original_error_reaches_next_query_with_real_trace_once(self):
        with tempfile.TemporaryDirectory() as folder:
            c = self.setup_case(Path(folder))
            original = self.lifecycle(c)
            self.assertEqual(c.inputs[0], original[:2])
            self.assertEqual(c.inputs[1][:-1], original)
            self.assertEqual(len(c.calls), 1)
            self.assertIn("PUBLIC FAILURE EXECUTION SLICE", c.inputs[1][-1]["content"])
            self.assertIn('"value":12', c.inputs[1][-1]["content"])
            self.assertIn("ZeroDivisionError", c.inputs[1][-1]["content"])
            self.assertEqual(c.observer.report["status"], "observed")
            self.assertTrue((c.state / "native-public-failure-slice.json").is_file())

    def test_unregistered_shell_command_never_replays(self):
        with tempfile.TemporaryDirectory() as folder:
            c = self.setup_case(Path(folder), command="python repro.py; echo additional")
            original = self.lifecycle(c)
            self.assertFalse(c.calls)
            self.assertEqual(c.inputs[1], original)

    def test_changed_source_or_public_input_never_replays_old_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            c = self.setup_case(Path(folder))
            original = self.lifecycle(c, mutate=True)
            self.assertFalse(c.calls)
            self.assertEqual(c.inputs[1], original)
            self.assertEqual(c.observer.report["status"], "unavailable")

    def test_trace_limit_never_changes_candidate_return_or_exception(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = "def add(n):\n    return n+1\n"
            (root / "owned.py").write_text(source)
            namespace = {}
            exec(compile(source, str((root / "owned.py").resolve()), "exec"), namespace)
            recorder = PublicTraceRecorder(root, ("owned.py",), max_events=8)
            with recorder:
                values = [namespace["add"](n) for n in range(30)]
            self.assertEqual(values, list(range(1, 31)))
            self.assertTrue(recorder.truncated)
            self.assertLessEqual(len(recorder.events), 8)
            self.assertIsNone(sys.gettrace())

    def test_no_custom_repr_getters_or_container_overrides(self):
        class Hostile(dict):
            def __repr__(self):
                raise AssertionError("repr called")

            def items(self):
                raise AssertionError("items called")

            def __len__(self):
                raise AssertionError("len called")

        value = describe_public_value(Hostile())
        self.assertEqual(value["type"], "Hostile")
        self.assertNotIn("items", value)
        self.assertTrue(describe_public_value(float("nan"))["nonfinite"])

    def test_original_cancel_identity_is_preserved(self):
        failure = KeyboardInterrupt("owned cancellation")
        with tempfile.TemporaryDirectory() as folder:
            c = self.setup_case(Path(folder), error=failure)
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.lifecycle(c)
            self.assertIs(caught.exception, failure)

    def test_no_context_room_retains_complete_original_messages(self):
        with tempfile.TemporaryDirectory() as folder:
            c = self.setup_case(
                Path(folder),
                counter=lambda messages, timeout: (
                    2000
                    if any("PUBLIC FAILURE EXECUTION SLICE" in str(x) for x in messages)
                    else 50
                ),
            )
            original = self.lifecycle(c)
            self.assertEqual(c.inputs[1], original)
            self.assertEqual(len(c.calls), 1)

    def test_collector_path_error_preserves_the_original_candidate_exception(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "owned.py").write_bytes(b'def fail():\n    raise LookupError("owned")\n')
            namespace = {}
            exec(
                compile((root / "owned.py").read_bytes(), str(root / "owned.py"), "exec"), namespace
            )
            recorder = PublicTraceRecorder(root, ("owned.py",))
            with patch(
                "infinicoder.verification.public_replay.Path.resolve",
                side_effect=OSError("owned lookup failure"),
            ):
                with self.assertRaisesRegex(LookupError, "owned"):
                    with recorder:
                        namespace["fail"]()
            self.assertEqual(recorder.error, "OSError")
            self.assertIsNone(sys.gettrace())

    def test_public_input_mismatch_is_rejected_before_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            c = self.setup_case(Path(folder))
            c.observer.validate(c.request)
            (c.workspace / "repro.py").write_bytes(b"# other public input\n")
            with self.assertRaisesRegex(ValueError, "frozen caller contract"):
                c.observer.validate(c.request)


if __name__ == "__main__":
    unittest.main()
