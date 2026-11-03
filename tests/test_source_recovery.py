"""Owned source fixtures and ordinary Runtime consumers;no task-quality score."""

import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from infinicoder import Runtime, TaskSpec, Check, ModelResponse
from infinicoder.models.budget import ContextAssembler, ModelProfile, TokenEstimator
from infinicoder.analysis.code_context import CodeIndex
from infinicoder.memory.inspection_navigation import (
    RECOVERY_LABEL,
    recover_source,
    recovery_context,
    delivered_recovery,
)
from infinicoder.execution.workspace_tools import Workspace


def count(text):
    return len(re.findall(r"\w+|[^\w\s]", text))


def action(tool, **args):
    return dict(tool=tool, arguments=args)


class SourceRecovery(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.ws = self.root / "workspace"
        self.ws.mkdir()
        (self.ws / "src").mkdir()
        (self.ws / "src/library.py").write_text(
            "class Gadget:\n    def transform(self,x):\n        return -100\n", encoding="utf-8"
        )
        self.workspace = Workspace(self.ws, ["src/library.py"])

    def index(self):
        index = CodeIndex(self.ws, self.root / "index.sqlite")
        index.refresh()
        self.addCleanup(index.close)
        return index

    def runtime(self, respond, **extra):
        class Client:
            supports_structured_output = True
            config = SimpleNamespace(max_output_tokens=512)

            def __init__(self):
                self.requests = []

            def generate_structured(self, messages, schema, timeout):
                self.requests.append((messages, schema))
                value = respond(len(self.requests), messages, schema)
                return ModelResponse(
                    json.dumps(value),
                    usage=dict(prompt_tokens=3, completion_tokens=2, total_tokens=5),
                )

        options = dict(
            source_recovery=True,
            automatic_workers=False,
            allow_commands=False,
            profile="general",
            domain_compiler=False,
            task_memory=True,
            estimator=TokenEstimator(count),
            max_model_calls=8,
            max_steps=10,
            time_limit=30,
        )
        options.update(extra)
        return Runtime(self.ws, Client(), self.root / "state", **options)

    def spec(self):
        checker = self.root / "check.py"
        checker.write_text(
            'import runpy,sys\nf=runpy.run_path(sys.argv[1]+"/src/library.py")["Gadget"]().transform\nassert [f(x) for x in (-3,0,9)]==[4,7,16]\n',
            encoding="utf-8",
        )
        return TaskSpec(
            "Implement Gadget.transform(x) as x+7;preserve its interface.",
            allowed_paths=["src/library.py"],
            checks=[
                Check(
                    "external",
                    [sys.executable, str(checker), "{workspace}"],
                    protected_paths=[str(checker)],
                )
            ],
        )

    def test_real_missing_read_to_source_edit_verify_finish(self):
        def respond(n, messages, schema):
            if n == 1:
                return action("read", path="src/Gadget.py")
            if n == 2:
                text = "\n".join(m["content"] for m in messages)
                self.assertIn("FileNotFoundError", text)
                self.assertIn(RECOVERY_LABEL, text)
                body = next(
                    m["content"].split(RECOVERY_LABEL)[1]
                    for m in messages
                    if RECOVERY_LABEL in m["content"]
                )
                packet = json.JSONDecoder().raw_decode(body)[0]
                self.assertEqual(packet["resolution_status"], "unique")
                self.assertEqual(packet["supplemental_source"]["path"], "src/library.py")
                self.assertIn("return -100", packet["supplemental_source"]["text"])
                self.assertLessEqual(
                    count(RECOVERY_LABEL + json.dumps(packet, ensure_ascii=False)), 768
                )
                return action(
                    "edit",
                    path="src/library.py",
                    expected_hash=packet["supplemental_source"]["sha256"],
                    old="return -100",
                    new="return x+7",
                )
            return action("verify" if n == 3 else "finish")

        runtime = self.runtime(respond)
        state = runtime.execute(self.spec())
        self.assertEqual(state["status"], "succeeded", state.get("last_failure"))
        self.assertEqual(state["model_calls"], 4)
        observed = [
            e["data"] for e in runtime.store.events() if e["kind"] == "resolved_source_observation"
        ]
        self.assertEqual(len(observed), 1)
        self.assertFalse(observed[0]["model_read_action"])
        self.assertTrue(observed[0]["presented"])
        self.assertEqual(observed[0]["source"]["covered_line_count"], 3)

    def test_ambiguity_and_protected_paths_never_auto_select(self):
        (self.ws / "src/other.py").write_text("class Gadget:pass\n", encoding="utf-8")
        index = self.index()
        state = {}
        packet = recover_source(index, self.workspace, "src/Gadget.py", state, count_tokens=count)
        self.assertEqual(packet["resolution_status"], "ambiguous")
        self.assertNotIn("supplemental_source", packet)
        self.assertEqual(len(packet["candidates"]), 2)
        packet = recover_source(
            index,
            self.workspace,
            "src/Gadget.py",
            {},
            count_tokens=count,
            protected_paths=["src/library.py", "src/other.py"],
        )
        self.assertEqual(packet["candidates"], [])
        self.assertNotIn("supplemental_source", packet)
        self.assertNotIn("library.py", json.dumps(packet))

    def test_stale_and_omitted_packets_do_not_claim_read_coverage(self):
        index = self.index()
        state = dict(model_calls=1)
        packet = recover_source(index, self.workspace, "src/Gadget.py", state, count_tokens=count)
        self.assertIn("supplemental_source", packet)
        recovery_context(state, self.workspace)
        events = []
        runtime = SimpleNamespace(
            store=SimpleNamespace(event=lambda *a: events.append(a)), task_memory=False
        )
        delivered_recovery(
            runtime, state, self.workspace, index, [dict(role="user", content="packet did not fit")]
        )
        self.assertNotIn("read_coverage", state)
        self.assertFalse(events[-1][1]["delivered"])
        (self.ws / "src/library.py").write_text("class Gadget:pass\n", encoding="utf-8")
        text = recovery_context(state, self.workspace)
        self.assertEqual(json.loads(text[len(RECOVERY_LABEL) :])["resolution_status"], "stale")
        self.assertNotIn("supplemental_source", json.loads(text[len(RECOVERY_LABEL) :]))
        delivered_recovery(runtime, state, self.workspace, index, [dict(role="user", content=text)])
        self.assertNotIn("read_coverage", state)

    def test_huge_line_and_missing_tokenizer_are_locations_only(self):
        (self.ws / "src/library.py").write_text(
            'class Gadget: payload="' + ("x" * 20000) + '"\n', encoding="utf-8"
        )
        index = self.index()
        for counter in (count, None):
            packet = recover_source(
                index, self.workspace, "src/Gadget.py", {}, count_tokens=counter
            )
            self.assertNotIn("supplemental_source", packet)
            self.assertIn("source_omitted", packet)
            self.assertLessEqual(
                count(RECOVERY_LABEL + json.dumps(packet, ensure_ascii=False)), 768
            )

    def test_cache_guard_and_original_read_scope(self):
        index = self.index()
        state = {}
        with patch.object(self.workspace, "read", wraps=self.workspace.read) as read:
            first = recover_source(
                index, self.workspace, "src/Gadget.py", state, count_tokens=count
            )
            second = recover_source(
                index, self.workspace, "src/Gadget.py", state, count_tokens=count
            )
            self.assertEqual(first, second)
            self.assertEqual(read.call_count, 1)
        self.assertIsNone(
            recover_source(index, self.workspace, "src/Gadget.py", {}, guard=lambda: False)
        )
        for path in ("../outside.py", ".git/Gadget.py"):
            with self.assertRaises(ValueError):
                recover_source(index, self.workspace, path, {}, count_tokens=count)
        with self.assertRaises(ValueError):
            self.workspace.edit("src/other.py", None, content="no")

    def test_atomic_budget_preserves_mandatory_and_no_trigger(self):
        assembler = ContextAssembler(
            TokenEstimator(len),
            ModelProfile(context_limit=256, output_reserve=32, protocol_overhead=16),
        )
        text, report = assembler.assemble(
            "ORIGINAL ERROR", dict(source_recovery="s" * 230, history="current feedback")
        )
        self.assertIn("ORIGINAL ERROR", text)
        self.assertNotIn("s" * 230, text)
        self.assertIn("source_recovery", {p["name"] for p in report["dropped"]})
        self.assertIn("history", {p["name"] for p in report["selected"]})
        self.index()
        with patch(
            "infinicoder.memory.inspection_navigation.recover_source",
            side_effect=AssertionError("unexpected recovery"),
        ):
            runtime = self.runtime(
                lambda *_: action("read", path="src/library.py"), max_model_calls=1
            )
            state = runtime.execute(self.spec())
            self.assertNotIn("source_recovery_cache", state)

    def test_off_does_not_call_recovery_or_change_source(self):
        with patch(
            "infinicoder.memory.inspection_navigation.recover_source",
            side_effect=AssertionError("OFF consumer ran"),
        ):
            runtime = self.runtime(
                lambda *_: action("read", path="src/Gadget.py"),
                source_recovery=False,
                max_model_calls=1,
            )
            state = runtime.execute(self.spec())
            self.assertEqual(state["changed_files"], [])
            self.assertFalse(any("source_recovery" in e["kind"] for e in runtime.store.events()))


if __name__ == "__main__":
    unittest.main()
