import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

from infinicoder import CoreBudget, CoreRequest, ModelProfile, NativeEvidenceRoute, TaskSpec
from infinicoder.memory.native_memory import MemoryCapacityError


def count(messages, remaining):
    return (len(json.dumps(messages)) + 3) // 4


class TaskMemoryTests(unittest.TestCase):
    def fixture(self, root):
        workspace = root / "workspace"
        workspace.mkdir()
        state = root / "state"
        state.mkdir()
        (workspace / "app.py").write_text("value = 1\n")
        observer = NativeEvidenceRoute(
            count,
            tokenizer_identity="engineering deterministic counter",
            implementation_paths=("app.py",),
            policy=None,
            task_working_memory=True,
        )
        request = CoreRequest(
            TaskSpec("Fix app.py; preserve its API and do not edit tests."),
            workspace,
            state,
            ModelProfile(4096, 1024),
            CoreBudget(20, 30, 5),
            context_adapter=observer,
        )
        packets = []
        model = SimpleNamespace(
            _prepare_messages_for_api=lambda m: copy.deepcopy(m),
            query=lambda m, **kw: (
                packets.append(copy.deepcopy(m)) or dict(role="assistant", content="edit app.py")
            ),
        )

        def execute(action):
            if action["command"] == "edit app.py":
                (workspace / "app.py").write_text("value = 2\n")
            return dict(returncode=0, output="action completed")

        return observer, request, model, SimpleNamespace(execute=execute), packets

    def test_short_request_is_unchanged_and_long_history_is_retrievable(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, r, m, e, packets = self.fixture(Path(tmp))
            history = [
                dict(role="system", content="SDK rules"),
                dict(role="user", content=r.spec.task),
            ]
            with c.bind(r, m, e, time.monotonic() + 30):
                m.query(history)
                self.assertEqual(packets[-1], history)
                self.assertFalse(c.working_memory.active)
                for i in range(10):
                    history += [
                        dict(role="assistant", content=f"read file {i}"),
                        dict(role="user", content=f"source {i}\n" + "old source\n" * 200),
                    ]
                history += [
                    dict(role="assistant", content="check current"),
                    dict(role="user", content="FAIL: current mismatch; fix the API"),
                ]
                original = copy.deepcopy(history)
                m.query(history)
                self.assertEqual(history, original)
                self.assertTrue(packets[-1][1]["content"].startswith(r.spec.task))
                self.assertEqual(packets[-1][-2:], history[-2:])
                self.assertLessEqual(count(packets[-1], 1), r.model_profile.input_budget())
                result = e.execute(dict(command="ic-task-memory event 3 0 50"))
                self.assertEqual(json.loads(result["output"])["text"], history[3]["content"][:50])
                self.assertEqual((r.workspace / "app.py").read_text(), "value = 1\n")

    def test_current_edit_action_survives_old_observation_eviction(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, r, m, e, packets = self.fixture(Path(tmp))
            history = [dict(role="system", content="rules"), dict(role="user", content=r.spec.task)]
            with c.bind(r, m, e, time.monotonic() + 30):
                response = m.query(history)
                e.execute(dict(command="edit app.py"))
                history += [response, dict(role="user", content="source updated")]
                for i in range(8):
                    history += [
                        dict(role="assistant", content=f"read {i}"),
                        dict(role="user", content="long old observation\n" * 200),
                    ]
                history += [
                    dict(role="assistant", content="last action"),
                    dict(role="user", content="FAIL"),
                ]
                m.query(history)
                self.assertIn(response, packets[-1])
                self.assertIn("app.py", packets[-1][1]["content"])
                self.assertIn(2, c.working_memory.records[-1]["protected_events"])

    def test_oversized_current_feedback_fails_without_silent_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, r, m, e, packets = self.fixture(Path(tmp))
            history = [
                dict(role="system", content="rules"),
                dict(role="user", content=r.spec.task),
                dict(role="assistant", content="current check"),
                dict(role="user", content="current error\n" * 3000),
            ]
            with c.bind(r, m, e, time.monotonic() + 30):
                with self.assertRaises(MemoryCapacityError):
                    m.query(history)
                self.assertEqual(packets, [])

    def test_manual_failure_and_delivered_detail_pages_remain_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, r, m, e, packets = self.fixture(Path(tmp))
            commands = iter(
                ["check", "read report", *[f"read source {i}" for i in range(9)], "finish"]
            )
            m.query = lambda messages, **kw: (
                packets.append(copy.deepcopy(messages))
                or dict(role="assistant", content=next(commands))
            )
            report = dict(report_id="a" * 32, record_sha256="b" * 64)

            def execute(action):
                command = action["command"]
                if command in ("check", "read report"):
                    mode = "run" if command == "check" else "read"
                    value = (
                        dict(report, summary="failed")
                        if mode == "run"
                        else dict(
                            report, offset=0, text="FIRST DIVERGENCE: restored cursor is wrong"
                        )
                    )
                    if mode == "run":
                        token = c.begin_check(
                            judge_identity="owned", scope="public", coverage="owned"
                        )
                        c.end_check(token, report=dict(summary=dict(status="failed")))
                    c.observe_public_report(mode, value)
                    return dict(
                        returncode=1 if mode == "run" else 0,
                        output=json.dumps(
                            json.loads(json.dumps(dict(ok=True, result=value), sort_keys=True)),
                            ensure_ascii=False,
                        ),
                    )
                return dict(returncode=0, output="ordinary source observation\n" * 100)

            e.execute = execute
            history = [
                dict(role="system", content="SDK rules"),
                dict(role="user", content=r.spec.task),
            ]
            with c.bind(r, m, e, time.monotonic() + 30):
                for _ in range(12):
                    response = m.query(history)
                    result = e.execute(dict(command=response["content"]))
                    history += [response, dict(role="user", content=result["output"])]
                self.assertTrue(c.working_memory.active)
                self.assertTrue(
                    {2, 3, 4, 5}.issubset(c.working_memory.records[-1]["protected_events"])
                )
                self.assertIn("FIRST DIVERGENCE", json.dumps(packets[-1]))
                self.assertEqual(c.operation_count, 0)


if __name__ == "__main__":
    unittest.main()
