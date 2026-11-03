"""Executable Python counterexamples and real versioned memory consumers."""

import hashlib, json, sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace

from infinicoder import CodeIndex, Runtime, TaskSpec, Check, ModelResponse
from infinicoder.analysis.dataflow import python_use_def
from infinicoder.memory.long_memory import HierarchicalMemory


def defs(source, name, line):
    return {
        e["definition_line"]
        for e in python_use_def(source)["edges"]
        if e["name"] == name and e["use_line"] == line
    }


def execute(source, *args):
    namespace = {}
    exec(source, namespace)
    return namespace["f"](*args)


class ExpressionMemory(unittest.TestCase):
    def test_walrus_value_is_bound_before_later_use(self):
        source = "def f(values):\n if (item := values.pop()):\n  return item\n return item\n"
        self.assertEqual(execute(source, [9]), 9)
        self.assertEqual(execute(source, [0]), 0)
        for line in (3, 4):
            self.assertEqual(defs(source, "item", line), {2})

    def test_short_circuit_preserves_possible_unbound_definition(self):
        source = "def f(flag):\n flag and (item := 9)\n return item\n"
        self.assertEqual(execute(source, True), 9)
        with self.assertRaises(UnboundLocalError):
            execute(source, False)
        self.assertEqual(defs(source, "item", 3), {0, 2})

    def test_conditional_expression_merges_assignments_and_prior_value(self):
        source = "def f(flag):\n item=3\n (item:=9) if flag else 0\n return item\n"
        self.assertEqual([execute(source, f) for f in (False, True)], [3, 9])
        self.assertEqual(defs(source, "item", 4), {2, 3})

    def test_chained_comparison_optional_comparator_keeps_prior(self):
        source = "def f(first):\n item=3\n first < 1 < (item:=9)\n return item\n"
        self.assertEqual([execute(source, f) for f in (0, 2)], [9, 3])
        self.assertEqual(defs(source, "item", 4), {2, 3})

    def test_dictionary_keys_and_values_follow_python_evaluation_order(self):
        source = "def f():\n return {\n  (item:=2):\n  item,\n  (item:=3):\n  item\n }\n"
        self.assertEqual(execute(source), {2: 2, 3: 3})
        self.assertEqual(defs(source, "item", 4), {3})
        self.assertEqual(defs(source, "item", 6), {5})

    def test_lazy_generator_does_not_execute_assignment_at_creation(self):
        source = "def f():\n item=3\n gen=((item:=9) for _ in [0])\n return item\n"
        self.assertEqual(execute(source), 3)
        self.assertEqual(defs(source, "item", 4), {2})

    def test_comprehension_iteration_target_does_not_escape(self):
        source = (
            "def f(values):\n item=3\n result=[item+1 for item in values]\n return item,result\n"
        )
        self.assertEqual(execute(source, [7]), (3, [8]))
        self.assertEqual(defs(source, "item", 4), {2})
        edges = [
            e for e in python_use_def(source)["edges"] if e["name"] == "item" and e["use_line"] == 3
        ]
        self.assertEqual({e["definition_line"] for e in edges}, {3})
        self.assertTrue(all("<comprehension:" in e["scope"] for e in edges))

    def test_comprehension_filter_assignment_precedes_element_and_is_optional(self):
        source = "def f(values):\n item=3\n result=[item for value in values if (item:=value)]\n return item,result\n"
        self.assertEqual(execute(source, [7, 0]), (0, [7]))
        self.assertEqual(execute(source, []), (3, []))
        self.assertEqual(defs(source, "item", 4), {2, 3})
        self.assertEqual(defs(source, "item", 3), {3})

    def test_module_symbols_outside_file_page_still_have_source_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ws = root / "ws"
            ws.mkdir()
            for name in ("a.py", "b.py"):
                (ws / name).write_text("# empty\n")
            file = ws / "z.py"
            file.write_text("def chosen():return 1\n")
            with CodeIndex(ws, root / "index.sqlite") as index:
                index.refresh()
                packet = HierarchicalMemory(index).expand(limit=2)
                self.assertTrue(packet["symbols"])
                for symbol in packet["symbols"]:
                    self.assertEqual(
                        symbol["source_sha256"],
                        hashlib.sha256((ws / symbol["path"]).read_bytes()).hexdigest(),
                    )
                file.write_text("def chosen():return 2\n")
                index.refresh()
                changed = HierarchicalMemory(index).expand(limit=2)
                self.assertNotEqual(
                    packet["symbols"][0]["source_sha256"], changed["symbols"][0]["source_sha256"]
                )

    def test_analysis_cache_is_invalidated_and_real_tool_feedback_is_consumed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ws = root / "ws"
            ws.mkdir()
            path = ws / "a.py"
            path.write_text(
                "def f(values):\n if (item := values.pop()):\n  return item\n return item\n"
            )
            checker = root / "check.py"
            checker.write_text(
                'import runpy,sys\nf=runpy.run_path(sys.argv[1]+"/a.py")["f"]\nassert f([2])==9 and f([0])==7\n'
            )
            spec = TaskSpec(
                "Change f to return the popped value plus7,including zero.",
                allowed_paths=["a.py"],
                checks=[
                    Check(
                        "external",
                        [sys.executable, str(checker), "{workspace}"],
                        protected_paths=[str(checker)],
                    )
                ],
            )

            class Client:
                supports_structured_output = True
                config = SimpleNamespace(max_output_tokens=512)

                def __init__(self):
                    self.calls = 0

                def generate_structured(self, messages, schema, timeout):
                    self.calls += 1
                    if self.calls == 1:
                        value = dict(tool="use_def", arguments=dict(path="a.py", name="item"))
                    elif self.calls == 2:
                        feedback = json.loads(messages[-1]["content"])["observation"]
                        self.before = feedback
                        assert {
                            e["definition_line"] for e in feedback["edges"] if e["use_line"] == 3
                        } == {2}
                        value = dict(
                            tool="edit",
                            arguments=dict(
                                path="a.py",
                                expected_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
                                old=path.read_bytes().decode("utf-8"),
                                new="def f(values):\n item=values.pop()\n return item+7\n",
                            ),
                        )
                    else:
                        value = dict(tool="verify" if self.calls == 3 else "finish", arguments={})
                    return ModelResponse(
                        json.dumps(value),
                        usage=dict(prompt_tokens=3, completion_tokens=2, total_tokens=5),
                    )

            client = Client()
            runtime = Runtime(
                ws,
                client,
                root / "state",
                automatic_workers=False,
                profile="general",
                domain_compiler=False,
                max_model_calls=5,
                time_limit=30,
            )
            state = runtime.execute(spec)
            self.assertEqual(
                state["status"],
                "succeeded",
                [
                    (e["kind"], e["data"].get("error"))
                    for e in runtime.store.events()
                    if e["kind"] in ("action_result", "model_error")
                ],
            )
            self.assertEqual(client.calls, 4)
            with CodeIndex(ws, root / "state/context.sqlite") as index:
                index.refresh()
                facts = index.use_def("a.py", name="item")
                self.assertEqual(
                    {e["definition_line"] for e in facts["edges"] if e["use_line"] == 3}, {2}
                )
                self.assertNotEqual(client.before["source_sha256"], facts["source_sha256"])


if __name__ == "__main__":
    unittest.main()
