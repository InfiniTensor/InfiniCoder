"""Owned readonly-tool wrapper controls;no model/benchmark/GPU claims."""

import json, tempfile, time, unittest
from pathlib import Path
import test_native_working_set as working_fixture


class WorksetOutputTests(unittest.TestCase):
    def test_line_wrapper_variants_preserve_quoted_source_arguments(self):
        from infinicoder.memory.source.workset_output import parse_workset_output

        for tail, n in [
            ("", ""),
            (" 2>&1", ""),
            (" | head", 10),
            (" 2>&1|head -n 20", 20),
            (" | head -n20", 20),
            (" | head -20", 20),
        ]:
            with self.subTest(tail=tail):
                value = parse_workset_output("ic-workset source 'file with spaces.py' 1 20" + tail)
                self.assertEqual(
                    value.argv, ("ic-workset", "source", "file with spaces.py", "1", "20")
                )
                self.assertEqual(value.head_lines, n if n != "" else None)
        value = parse_workset_output("ic-workset find 'a | head -n 20'")
        self.assertEqual(value.argv, ("ic-workset", "find", "a | head -n 20"))
        self.assertIsNone(value.head_lines)

    def test_unknown_shell_and_ambiguous_redirection_are_rejected(self):
        from infinicoder.memory.source.workset_output import parse_workset_output

        for tail in (
            "; echo changed",
            " && cat x",
            " | cat",
            " > out",
            " | head -n 2 | head",
            " | head -c 20",
            " | head -n 0",
            " | head -1001",
            " | head x",
            " | head; echo x",
            " '2'>&1",
            " 2 >&1",
            " 2>& 1",
            " $(echo bad)",
            " `echo bad`",
        ):
            with self.subTest(tail=tail):
                with self.assertRaises(ValueError):
                    parse_workset_output("ic-workset files" + tail)

    def test_quoted_operator_parameters_are_not_reparsed_and_reset_the_tool_family(self):
        from infinicoder import NativeWorkingSet

        context = object.__new__(NativeWorkingSet)
        calls = []
        context._tool_family = "ic-plan"
        context.tool = lambda args: calls.append(args) or "owned readonly result"
        for query in ("'|'", "';'", "'a | head -n 20'"):
            self.assertEqual(
                context.execute_tool_command("ic-workset find " + query), "owned readonly result"
            )
        self.assertEqual(calls, [["find", "|"], ["find", ";"], ["find", "a | head -n 20"]])
        self.assertEqual(context._tool_family, "ic-workset")

    def test_actual_bound_tool_preserves_inner_failure_and_executes_valid_wrapped_read_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            context, request, model, environment, packets, _ = (
                working_fixture.WorkingSetTests().fixture(Path(tmp), 8192)
            )
            messages = [
                dict(role="system", content="Owned SDK format"),
                dict(role="user", content=request.spec.task),
            ]
            calls = []
            tool = context.tool

            def observe(args):
                calls.append(list(args))
                return tool(args)

            context.tool = observe
            with context.bind(request, model, environment, time.monotonic() + 30):
                model.query(messages)
                bad = environment.execute(dict(command="ic-workset event 1 0 20 2>&1 | head -20"))
                self.assertEqual(bad["returncode"], 2)
                self.assertIn("event id>=2", bad["output"])
                messages += [
                    dict(role="assistant", content="ordinary"),
                    dict(role="user", content=bad["output"]),
                ]
                model.query(messages)
                before = len(calls)
                good = environment.execute(
                    dict(command="ic-workset event 3 0 20 2>&1 | head -n 20")
                )
                self.assertEqual(good["returncode"], 0)
                self.assertEqual(len(calls), before + 1)
                view = json.loads(good["output"])
                self.assertEqual(view["role"], "user")
                self.assertEqual(view["text"], bad["output"][:20])
                before = len(calls)
                rejected = environment.execute(
                    dict(command="ic-workset files | head -20; echo changed")
                )
                self.assertEqual(rejected["returncode"], 2)
                self.assertEqual(len(calls), before)
                ordinary = environment.execute(dict(command="cat maths.py | head -n 1"))
                self.assertEqual(ordinary["output"], "ordinary")

    def test_truncated_display_is_marked_and_does_not_replay_underlying_query(self):
        from infinicoder import NativeWorkingSet

        context = object.__new__(NativeWorkingSet)
        calls = []
        context.tool = lambda args: calls.append(args) or "first\nsecond\n"
        context._output = lambda value: json.dumps(value)
        value = json.loads(context.execute_tool_command("ic-workset files | head -n 1"))
        self.assertTrue(value["truncated"])
        self.assertEqual(value["text"], "first\n")
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
