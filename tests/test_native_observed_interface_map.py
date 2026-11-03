"""Simulated query-boundary controls;actual SDK/HTTP qualification is separate."""

import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

from infinicoder.memory.native_observed_interface_map import NativeObservedInterfaceMap, HEADER
from infinicoder.planning.native_source_planning import workspace_source_reader
from infinicoder.memory.source.observed_source_reads import ReadObservationWorkset
from infinicoder.memory.task_evidence_context import TaskEvidenceContext
from infinicoder.planning.task_source_plan import WorkspaceSourcePlanReader
from infinicoder.analysis.code_context import CodeIndex

FILES = {
    "api.py": "from provider import load\ndef run(path):\n    result = load(path)\n    return consume(result)\n",
    "provider.py": "def load(path):\n    return {'value': path}\n",
}


class Model:
    def __init__(self, commands):
        self.commands = iter(commands)
        self.inputs = []

    def _prepare_messages_for_api(self, messages):
        return [{k: m[k] for k in ("role", "content") if k in m} for m in messages]

    def query(self, messages, **kwargs):
        self.inputs.append(copy.deepcopy(messages))
        command = next(self.commands)
        return dict(
            role="assistant",
            content="```bash\n" + command + "\n```",
            actions=[dict(command=command)],
        )


class Environment:
    def __init__(self, root):
        self.root = root

    def execute(self, action, **kwargs):
        command = action["command"]
        text = ""
        if command.startswith("cat "):
            text = (self.root / command[4:]).read_bytes().decode()
        elif command == "change":
            (self.root / "provider.py").write_bytes(
                (FILES["provider.py"] + "# new version\n").encode()
            )
        return dict(output=text, returncode=0, exception_info="")


class NativeObservedMapTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.workspace = self.base / "workspace"
        self.workspace.mkdir()
        self.state = self.base / "state"
        self.state.mkdir()
        for path, text in FILES.items():
            (self.workspace / path).write_bytes(text.encode())
        self.task = "Implement the public interface.\n\nFILES:\n" + json.dumps(FILES)
        self.request = SimpleNamespace(
            workspace=self.workspace,
            state_dir=self.state,
            spec=SimpleNamespace(task=self.task),
            model_profile=SimpleNamespace(input_budget=lambda: 12000),
            planning_adapter=None,
            repair_adapter=None,
            candidate_search=None,
            implementation_proposal=None,
        )

    def observer(self, reader=workspace_source_reader):
        observer = NativeObservedInterfaceMap(
            lambda messages, timeout: 100,
            tokenizer_identity="scripted-count-only",
            reader_factory=reader,
            public_sources=FILES,
        )
        self.request.context_adapter = observer
        return observer

    def run_queries(self, observer, commands):
        model = Model(commands)
        environment = Environment(self.workspace)
        history = [dict(role="system", content="fixture"), dict(role="user", content=self.task)]
        with observer.bind(self.request, model, environment, time.monotonic() + 60):
            for _ in commands:
                response = model.query(history)
                history.append(response)
                result = environment.execute(response["actions"][0])
                history.append(
                    dict(
                        role="user",
                        content="Observation:\n" + result["output"],
                        extra=dict(raw_output=result["output"], returncode=0),
                    )
                )
        return model, history

    def test_delivery_lifetime_edit_withdrawal_and_fresh_reobservation(self):
        observer = self.observer()
        model, history = self.run_queries(
            observer, ["cat api.py", "true", "true", "change", "cat provider.py", "true", "true"]
        )
        shown = [any(HEADER in m.get("content", "") for m in packet) for packet in model.inputs]
        self.assertEqual(shown, [False, True, True, True, False, False, True])
        self.assertEqual([observer.map_slots[i] for i in range(len(observer.maps))], [3, 1])
        self.assertEqual(len(observer.maps), 2)
        self.assertNotEqual(
            observer.maps[0]["source_versions"]["provider.py"],
            observer.maps[1]["source_versions"]["provider.py"],
        )
        self.assertFalse(any(HEADER in m.get("content", "") for m in history))
        new_ref = observer.maps[1]["payload"]["files"]
        provider = next(v for v in new_ref.values() if v["path"] == "provider.py")
        self.assertTrue(provider["message_ref"].startswith("query:6:"))
        self.assertLess(observer.spent, 10)

    def test_unavailable_namespace_keeps_original_input(self):
        observer = self.observer(None)
        model, history = self.run_queries(observer, ["true"])
        self.assertEqual(model.inputs[0], history[:2])
        self.assertFalse(observer.maps)

    def test_full_raw_output_but_truncated_sdk_feedback_is_not_source(self):
        with CodeIndex(self.workspace, self.state / "isolated.sqlite") as index:
            reader = WorkspaceSourcePlanReader(
                self.request, index, count_output=lambda text: 100, remaining=lambda: 10
            )
            state = TaskEvidenceContext(index, remaining=lambda: 10)
            reads = ReadObservationWorkset(
                reader, state, namespace_id="a" * 64, remaining=lambda: 10
            )
            action = dict(command="cat provider.py")
            token = reads.begin_read(action)
            event = dict(
                id=0,
                command=action["command"],
                error=None,
                observer_complete=True,
                message_ids=[2, 3],
            )
            reads.finish_read(token, dict(output=FILES["provider.py"], returncode=0), event)
            raw = dict(
                role="user",
                content="Output was truncated",
                extra=dict(raw_output=FILES["provider.py"], returncode=0),
            )
            self.assertIsNone(
                reads.link_result(event, raw, dict(role="user", content=raw["content"]))
            )
            self.assertEqual(
                reads.records[0]["status"], "SDK_feedback_not_exact;original_preserved"
            )
            self.assertEqual(
                index.db.execute("SELECT COUNT(*) FROM source_plan_views").fetchone()[0], 0
            )


if __name__ == "__main__":
    unittest.main()
