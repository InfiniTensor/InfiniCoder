"""One owned HTTP→observed callable→ordinary edit→check integration group."""

import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

from test_implementation_window import owned_http, action, count, bound, tool_names
from infinicoder import Runtime, TaskSpec, Check
from infinicoder.models.budget import TokenEstimator, ModelProfile
from infinicoder.models.model_client import CompatibleClient, ModelConfig
from infinicoder.analysis.task_compiler import compile_task
from infinicoder.execution.workspace_tools import Workspace
from infinicoder.analysis.symbol_tools import callable_ranges
from infinicoder.runtime import implementation_window as window


class CallableImplementationWindow(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.ws = self.root / "workspace"
        self.ws.mkdir()
        self.prefix_lines = 700
        self.source = (
            ("# unchanged support information\n" * self.prefix_lines)
            + '''def identity(fn):
    return fn

@identity
def transform(x):
    """Keep this public interface."""
    return -100

class Accumulator:
    offset = 7
    def combine(self, x):
        return x + self.offset

def unrelated(x):
    return x - 11
'''
        )
        self.path = self.ws / "large.py"
        self.path.write_bytes(self.source.encode())
        self.function = next(
            x for x in callable_ranges(self.source) if x["qualname"] == "transform"
        )
        self.method = next(
            x for x in callable_ranges(self.source) if x["qualname"] == "Accumulator.combine"
        )
        checker = self.root / "check.py"
        checker.write_text(
            "import runpy,sys\ne=runpy.run_path(sys.argv[1]+'/large.py')\nassert e['transform'](3)==10\nassert e['Accumulator']().combine(5)==12\nassert e['unrelated'](5)==-6\n",
            encoding="utf-8",
        )
        self.spec = TaskSpec(
            "Make transform return x+7;preserve other behavior.",
            allowed_paths=["large.py"],
            checks=[
                Check(
                    "external",
                    [sys.executable, str(checker), "{workspace}"],
                    protected_paths=[str(checker)],
                )
            ],
        )

    def runtime(self, url, **extra):
        opts = dict(
            profile="general",
            domain_compiler=False,
            automatic_workers=False,
            source_fallback=False,
            implementation_window=True,
            implementation_window_unit="python_callable",
            model_profile=ModelProfile(16384, 512, 256),
            estimator=TokenEstimator(count, bound),
            max_steps=16,
            max_model_calls=16,
            time_limit=45,
        )
        opts.update(extra)
        return Runtime(
            self.ws,
            CompatibleClient(
                ModelConfig(url, "owned-http", max_output_tokens=512, structured_outputs=True)
            ),
            self.root / "session",
            **opts,
        )

    def state(self, runtime):
        runtime._graph = compile_task(self.spec)
        return dict(
            milestone=0,
            model_calls=0,
            planning_required=False,
            clarifications=[],
            session_deadline_epoch=time.time() + 30,
        )

    def observe(self, runtime, state, row, start=None, number=None):
        start = start or row["start"]
        number = number or row["end"] - start + 1
        ws = Workspace(self.ws, self.spec.allowed_paths)
        result = ws.read("large.py", start, number)
        window.observe(
            runtime,
            self.spec,
            state,
            ws,
            action("read", path="large.py", start=start, count=number),
            result,
        )

    def packet(self, payload):
        for m in payload["messages"]:
            pos = m["content"].find(window.MARKER)
            if pos >= 0:
                return json.JSONDecoder().raw_decode(
                    m["content"][m["content"].index("\n", pos) + 1 :]
                )[0]
        self.fail("Callable source absent from actual HTTP request")

    def test_large_module_callable_HTTP_delivery_edit_verify_and_final(self):
        self.assertGreater(count(self.source), 1600)

        def respond(n, payload):
            if n <= 6:
                start = self.function["start"] if n % 2 else self.function["start"] + 2
                return action("read", path="large.py", start=start, count=2)
            if n == 7:
                p = self.packet(payload)
                s = p["source"]
                self.assertEqual(s["kind"], "python_callable")
                self.assertEqual(s["qualname"], "transform")
                self.assertEqual(
                    s["text"],
                    "".join(
                        self.source.splitlines(keepends=True)[
                            self.function["start"] - 1 : self.function["end"]
                        ]
                    ),
                )
                self.assertIn(self.spec.task, json.dumps(payload))
                self.assertEqual(p["latest_action_feedback"]["action"]["tool"], "read")
                self.assertIn(
                    "leave_implementation",
                    tool_names(payload["response_format"]["json_schema"]["schema"]),
                )
                return action(
                    "edit",
                    path="large.py",
                    expected_hash=s["sha256"],
                    old="return -100",
                    new="return x+7",
                )
            return action("verify" if n == 8 else "finish")

        with owned_http(respond) as (url, requests, errors):
            runtime = self.runtime(url)
            state = runtime.execute(self.spec)
        self.assertFalse(errors)
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(len(requests), 9)
        self.assertEqual(state["model_calls"], 9)
        self.assertEqual(
            self.path.read_bytes(), self.source.replace("return -100", "return x+7").encode()
        )
        record = state["implementation_window"]
        self.assertEqual(record["calls_used"], 1)
        self.assertEqual(record["delivered_requests"], 1)
        self.assertEqual(record["reason"], "source_changed_after_action")

    def test_pagination_method_original_indentation_class_interface_and_guard(self):
        runtime = self.runtime("http://127.0.0.1:1/v1")
        state = self.state(runtime)
        ws = Workspace(self.ws, self.spec.allowed_paths)
        for i in range(6):
            self.observe(runtime, state, self.method, start=self.method["start"] + i % 2, number=1)
            self.assertEqual(bool(state.get("implementation_window_attempted")), i == 5)
        window.select(runtime, self.spec, state, ws)
        text = window.packet(runtime, self.spec, state, ws)
        packet = json.loads(text.split("\n", 1)[1])
        focus = packet["source"]
        self.assertTrue(focus["text"].startswith("    def combine"))
        self.assertEqual(focus["owner"]["qualname"], "Accumulator")
        self.assertIn("class Accumulator:", focus["owner"]["class_header"])
        self.assertIn("offset = 7", focus["owner"]["static_field_declarations"])
        state["implementation_window"]["request"] = dict(source_delivered=True)
        window.guard(
            runtime,
            self.spec,
            state,
            ws,
            action(
                "edit",
                path="large.py",
                expected_hash=focus["sha256"],
                old="return x + self.offset",
                new="return x + self.offset + 1",
            ),
        )
        for old, new in [
            ("return x - 11", "return x - 12"),
            ("return x + self.offset", "return x + self.offset\n\ndef injected():\n    return 2"),
        ]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                window.guard(
                    runtime,
                    self.spec,
                    state,
                    ws,
                    action(
                        "edit", path="large.py", expected_hash=focus["sha256"], old=old, new=new
                    ),
                )
        window.guard(
            runtime,
            self.spec,
            state,
            ws,
            action("leave_implementation", reason="Need a class-level change."),
        )
        self.assertFalse(window.active(state))
        self.assertEqual(ws.path("large.py", write=True), self.path.resolve())

    def test_ambiguous_and_conditional_units_do_not_activate(self):
        runtime = self.runtime("http://127.0.0.1:1/v1")
        state = self.state(runtime)
        ws = Workspace(self.ws, self.spec.allowed_paths)
        for _ in range(3):
            window.observe(
                runtime,
                self.spec,
                state,
                ws,
                action("read"),
                ws.read("large.py", self.prefix_lines + 1, 100),
            )
        self.assertFalse(state.get("implementation_window_attempted"))
        self.assertEqual(callable_ranges("if True:\n    def f():\n        return 1\n"), [])
        self.assertEqual(callable_ranges("def f():\n    pass\ndef f():\n    pass\n"), [])

    def test_revision_and_requirement_change_invalidate_observed_rounds(self):
        runtime = self.runtime("http://127.0.0.1:1/v1")
        state = self.state(runtime)
        for _ in range(2):
            self.observe(runtime, state, self.function)
        self.path.write_bytes(self.path.read_bytes() + b"\n# changed version\n")
        self.observe(runtime, state, self.function)
        self.assertEqual(next(iter(state["implementation_window_reads"].values()))["rounds"], 1)
        state["clarifications"] = ["Updated requirement"]
        self.observe(runtime, state, self.function)
        self.assertEqual(next(iter(state["implementation_window_reads"].values()))["rounds"], 1)

    def test_policy_is_explicit_and_requires_existing_window(self):
        with self.assertRaises(ValueError):
            self.runtime("http://127.0.0.1:1/v1", implementation_window=False)
        runtime = self.runtime(
            "http://127.0.0.1:1/v1",
            implementation_window=False,
            implementation_window_unit="module",
        )
        self.assertFalse(runtime.implementation_window)
        self.assertEqual(runtime.implementation_window_unit, "module")

    def test_cancel_after_actual_delivery_cannot_commit_callable_edit(self):
        holder = {}

        def respond(n, payload):
            if n <= 3:
                return action("read", path="large.py", start=self.function["start"], count=4)
            if n == 4:
                source = self.packet(payload)["source"]
                holder["runtime"].store.request_control("cancel")
                return action(
                    "edit",
                    path="large.py",
                    expected_hash=source["sha256"],
                    old="return -100",
                    new="return x+7",
                )
            raise AssertionError("Cancelled execution requested another action")

        with owned_http(respond) as (url, requests, errors):
            runtime = self.runtime(url)
            holder["runtime"] = runtime
            state = runtime.execute(self.spec)
        self.assertFalse(errors)
        self.assertEqual(state["status"], "cancelled")
        self.assertEqual(self.path.read_bytes(), self.source.encode())
        self.assertFalse(state["changed_files"])
        self.assertEqual(state["implementation_window"]["delivered_requests"], 1)

    def test_CRLF_unicode_and_stale_response_keep_exact_original_source(self):
        text = self.source.replace("Keep this public interface.", "原始接口").replace("\n", "\r\n")
        self.path.write_bytes(text.encode())
        runtime = self.runtime("http://127.0.0.1:1/v1")
        state = self.state(runtime)
        for _ in range(3):
            self.observe(runtime, state, self.function)
        ws = Workspace(self.ws, self.spec.allowed_paths)
        window.select(runtime, self.spec, state, ws)
        packet = json.loads(window.packet(runtime, self.spec, state, ws).split("\n", 1)[1])
        self.assertIn("\r\n", packet["source"]["text"])
        self.assertIn("原始接口", packet["source"]["text"])
        state["implementation_window"]["request"] = dict(source_delivered=True)
        source = packet["source"]
        self.path.write_bytes(text.encode() + b"\r\n# concurrent edit\r\n")
        with self.assertRaises(ValueError):
            window.guard(
                runtime,
                self.spec,
                state,
                ws,
                action(
                    "edit",
                    path="large.py",
                    expected_hash=source["sha256"],
                    old="return -100",
                    new="return x+7",
                ),
            )
        self.assertEqual(state["implementation_window"]["status"], "invalidated")


if __name__ == "__main__":
    unittest.main()
