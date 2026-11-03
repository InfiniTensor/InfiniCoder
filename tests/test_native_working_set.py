import hashlib, json, tempfile, time, unittest
from pathlib import Path
from types import SimpleNamespace
from infinicoder import (
    SourceAttachment,
    NativeWorkingSet,
    ReadOnlyTaskSources,
    TaskSpec,
    ModelProfile,
    CoreBudget,
    CoreRequest,
)


def count(messages, remaining):
    return (len(json.dumps(messages, ensure_ascii=False)) + 3) // 4


class WorkingSetTests(unittest.TestCase):
    def fixture(self, root, limit=4096):
        workspace = root / "workspace"
        workspace.mkdir()
        state = root / "state"
        state.mkdir()
        files = {
            "maths.py": "def scale(x):\n    return x*2\n" + "# source attachment padding\n" * 900,
            "api.py": "from maths import scale\ndef serve(x):\n    return scale(x)\n",
        }
        for name, text in files.items():
            (workspace / name).write_text(text)
        literal = json.dumps(files)
        task = (
            "Preserve API serve(x). Fix scale without changing caller.\nFILES:\n"
            + literal
            + "\nOnly maths.py is editable."
        )
        sources = ReadOnlyTaskSources(workspace, tuple(files))
        context = NativeWorkingSet(
            count,
            tokenizer_identity="owned deterministic engineering counter",
            implementation_paths=("maths.py",),
            source_capability=sources,
            attachment=SourceAttachment(literal, files),
        )
        req = CoreRequest(
            TaskSpec(task),
            workspace,
            state,
            ModelProfile(limit, 1024),
            CoreBudget(4, 30, 5),
            context_adapter=context,
        )
        packets = []
        model = SimpleNamespace(
            _prepare_messages_for_api=lambda m: m,
            query=lambda m, **kw: packets.append(m) or dict(role="assistant", content="ordinary"),
        )
        env = SimpleNamespace(
            implementation_workspace_binding=lambda: dict(
                root=str(workspace.resolve()), model_root="/workspace"
            ),
            execute=lambda a: dict(returncode=0, output="ordinary"),
        )
        return context, req, model, env, packets, files

    def test_long_initial_attachment_and_recent_output_are_reversible(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, r, m, e, packets, files = self.fixture(Path(tmp))
            messages = [
                dict(role="system", content="Pinned SDK rules"),
                dict(role="user", content=r.spec.task),
                dict(role="assistant", content="cat maths.py"),
                dict(role="user", content="line\n" * 10000 + "\nERROR public failure"),
                dict(role="assistant", content="inspect next"),
                dict(role="user", content="public feedback"),
            ]
            original = json.dumps(messages)
            with c.bind(r, m, e, time.monotonic() + 30):
                m.query(messages)
                self.assertLessEqual(count(packets[-1], 1), r.model_profile.input_budget())
                text = json.dumps(packets[-1])
                self.assertIn("Only maths.py is editable.", text)
                self.assertIn("Preserve API serve(x)", text)
                self.assertNotIn("# source attachment padding", text)
                self.assertEqual(json.dumps(messages), original)
                event = json.loads(c.tool(["event", "3", "0", "500"]))
                self.assertEqual(event["text"], messages[3]["content"][:500])
                initial = json.loads(c.tool(["initial", "maths.py", "1", "2"]))
                self.assertEqual(initial["exact_lines"][1]["text"], "    return x*2")
                self.assertTrue(initial["historical"])
                self.assertFalse(initial["complete_file"])

    def test_live_edits_refresh_interfaces_and_leave_initial_archive_historical(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, r, m, e, packets, files = self.fixture(Path(tmp), 8192)
            messages = [
                dict(role="system", content="rules"),
                dict(role="user", content=r.spec.task),
            ]
            with c.bind(r, m, e, time.monotonic() + 30):
                first = json.loads(c.tool(["source", "maths.py", "1", "2"]))
                (r.workspace / "maths.py").write_text(
                    "def scale(x, divisor=1):\n    return x/divisor\n"
                )
                m.query(messages)
                last = c.records[-1]
                self.assertIn("maths.py", last["invalidated"])
                second = json.loads(c.tool(["source", "maths.py", "1", "2"]))
                self.assertNotEqual(first["sha256"], second["sha256"])
                self.assertIn("divisor", json.dumps(packets[-1]))
                historical = json.loads(c.tool(["initial", "maths.py", "1", "2"]))
                self.assertEqual(
                    historical["sha256"], hashlib.sha256(files["maths.py"].encode()).hexdigest()
                )

    def test_constraints_cannot_be_implicitly_decomposed(self):
        for literal, task in [("{}", "{} {}"), ('{"a.py":"pass"}', "unrelated")]:
            with self.assertRaises(ValueError):
                SourceAttachment(literal, json.loads(literal)).validate(task)

    def test_unknown_namespace_is_rejected_before_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, r, m, e, packets, files = self.fixture(Path(tmp))
            e.implementation_workspace_binding = lambda: dict(
                root=str(Path(tmp)), model_root="/workspace"
            )
            with self.assertRaises(Exception):
                with c.bind(r, m, e, time.monotonic() + 30):
                    m.query([])
            self.assertFalse(packets)

    def test_latest_large_action_and_feedback_are_one_user_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, r, m, e, packets, files = self.fixture(Path(tmp))
            messages = [
                dict(role="system", content="rules"),
                dict(role="user", content=r.spec.task),
                dict(role="assistant", content="latest_read"),
                dict(role="user", content="output\n" * 9000 + "OWN_FAILURE_TOKEN"),
            ]
            with c.bind(r, m, e, time.monotonic() + 30):
                m.query(messages)
                receipts = [
                    json.loads(v["content"])
                    for v in packets[-1]
                    if v["role"] == "user"
                    and v["content"].startswith("{")
                    and json.loads(v["content"]).get("layer")
                    == "historical_action_feedback_receipt"
                ]
                self.assertEqual(len(receipts), 1)
                receipt = receipts[0]
                self.assertEqual(receipt["canonical_event_ids"], [2, 3])
                self.assertEqual(
                    [q["original_role"] for q in receipt["quotes"]], ["assistant", "user"]
                )
                self.assertIn("latest_read", receipt["quotes"][0]["head"]["text"])
                self.assertIn("OWN_FAILURE_TOKEN", receipt["quotes"][1]["tail"]["text"])
                for quote in receipt["quotes"]:
                    text = messages[quote["event_id"]]["content"]
                    for part in (quote["head"], quote["tail"]):
                        self.assertTrue(0 <= part["start"] <= part["end"] <= len(text))
                        self.assertEqual(part["text"], text[part["start"] : part["end"]])
                self.assertFalse(any(v["role"] == "assistant" for v in packets[-1]))

    def test_history_length_does_not_create_unbounded_exact_count_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, r, m, e, packets, files = self.fixture(Path(tmp))
            calls = []
            c.count_messages = lambda messages, remaining: (
                calls.append(len(messages)) or count(messages, remaining)
            )
            messages = [
                dict(role="system", content="rules"),
                dict(role="user", content=r.spec.task),
            ]
            for i in range(150):
                messages.extend(
                    [
                        dict(role="assistant", content="read " + str(i) + "x" * 600),
                        dict(role="user", content="feedback " + str(i) + "y" * 600),
                    ]
                )
            with c.bind(r, m, e, time.monotonic() + 30):
                m.query(messages)
                self.assertLessEqual(len(calls), 9)
                self.assertLessEqual(c.records[-1]["exact_count_queries"], 9)
                selected = set(c.records[-1]["selected"])
                for i in range(2, len(messages), 2):
                    self.assertEqual(i in selected, (i + 1) in selected)


if __name__ == "__main__":
    unittest.main()
