"""Notebook sources,hash guards,and original Runtime edit/verify consumers."""

import copy, hashlib, json, re, sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace

from infinicoder import NotebookWorkspace, Runtime, TaskSpec, Check, ModelResponse, CodeIndex
from infinicoder.execution.workspace_tools import Workspace
from infinicoder.models.budget import TokenEstimator


def count(text):
    return len(re.findall(r"\w+|[^\w\s]", text))


def action(tool, **args):
    return dict(tool=tool, arguments=args)


class NotebookTools(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.ws = self.root / "workspace"
        self.ws.mkdir()
        self.path = self.ws / "demo.ipynb"
        self.document = dict(
            nbformat=4,
            nbformat_minor=5,
            metadata=dict(untouched="中文", kernelspec=dict(name="python3", language="python")),
            cells=[
                dict(
                    id="docs",
                    cell_type="markdown",
                    metadata=dict(keep=1),
                    source=["Documentation only."],
                ),
                dict(
                    id="implementation",
                    cell_type="code",
                    metadata=dict(keep=2),
                    source=["def transform(x):\n", "    return -100\n"],
                    execution_count=9,
                    outputs=[
                        dict(
                            output_type="stream",
                            name="stdout",
                            text=["PASSED from an old execution"],
                        )
                    ],
                ),
                dict(
                    id="other",
                    cell_type="code",
                    metadata=dict(keep=3),
                    source=["def helper():return 5\n"],
                    execution_count=4,
                    outputs=[],
                ),
            ],
        )
        self.path.write_text(
            json.dumps(self.document, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        self.workspace = Workspace(self.ws, ["demo.ipynb"])
        self.book = NotebookWorkspace(self.workspace, count_tokens=count)

    def edit_args(self, **extra):
        value = self.book.read("demo.ipynb", 1)
        args = dict(
            path="demo.ipynb",
            cell=1,
            expected_hash=value["sha256"],
            expected_cell_hash=value["cell_sha256"],
            old="return -100",
            new="return x+7",
        )
        args.update(extra)
        return args

    def spec(self):
        checker = self.root / "check.py"
        checker.write_text(
            'import json,sys\nn=json.load(open(sys.argv[1]+"/demo.ipynb",encoding="utf-8"));env={}\nfor c in n["cells"]:\n if c["cell_type"]=="code":exec("".join(c["source"]) if isinstance(c["source"],list) else c["source"],env)\nassert [env["transform"](x) for x in (-2,0,9)]==[5,7,16]\nassert env["helper"]()==5\n',
            encoding="utf-8",
        )
        return TaskSpec(
            "Update transform(x) in demo.ipynb to return x+7,preserving the other cells and metadata.",
            allowed_paths=["demo.ipynb"],
            checks=[
                Check(
                    "external",
                    [sys.executable, str(checker), "{workspace}"],
                    protected_paths=[str(checker)],
                )
            ],
        )

    def runtime(self, respond, **extra):
        class Client:
            supports_structured_output = True
            config = SimpleNamespace(max_output_tokens=512)

            def __init__(self):
                self.calls = 0
                self.requests = []

            def generate_structured(self, messages, schema, timeout):
                self.calls += 1
                self.requests.append((messages, schema))
                value = respond(self.calls, messages, schema)
                return ModelResponse(
                    json.dumps(value),
                    usage=dict(prompt_tokens=3, completion_tokens=2, total_tokens=5),
                )

        options = dict(
            notebook_tools=True,
            profile="general",
            domain_compiler=False,
            estimator=TokenEstimator(count),
            max_model_calls=8,
            time_limit=30,
        )
        options.update(extra)
        return Runtime(self.ws, Client(), self.root / "state", **options)

    def test_catalog_and_source_ignore_saved_outputs_and_use_cell_coordinates(self):
        catalog = self.book.cells("demo.ipynb")
        self.assertEqual(catalog["total_cells"], 3)
        self.assertEqual(catalog["cells"][1]["declarations"][0]["name"], "transform")
        self.assertEqual(catalog["cells"][1]["declarations"][0]["line"], 1)
        value = self.book.read("demo.ipynb", 1)
        self.assertEqual(value["coordinate_kind"], "cell_local_lines")
        self.assertEqual(value["covered_line_count"], 2)
        self.assertNotIn("PASSED", json.dumps(value))
        self.assertNotIn("outputs", json.dumps(catalog))
        self.assertLessEqual(count(json.dumps(catalog, ensure_ascii=False)), 1600)

    def test_unique_cell_patch_preserves_metadata_and_clears_all_code_outputs(self):
        before = copy.deepcopy(self.document)
        receipt = self.book.edit(**self.edit_args())
        after = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(after["metadata"], before["metadata"])
        self.assertEqual(after["cells"][0], before["cells"][0])
        for i in (1, 2):
            self.assertEqual(after["cells"][i]["metadata"], before["cells"][i]["metadata"])
            self.assertEqual(after["cells"][i]["outputs"], [])
            self.assertIsNone(after["cells"][i]["execution_count"])
        self.assertEqual(after["cells"][2]["source"], before["cells"][2]["source"])
        self.assertEqual(receipt["notebook"]["cleared_code_outputs"], 2)
        self.assertEqual(receipt["after"], hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_whole_file_and_cell_hashes_both_guard_against_stale_edits(self):
        args = self.edit_args()
        stale = dict(args, expected_cell_hash="0" * 64)
        with self.assertRaises(ValueError):
            self.book.edit(**stale)
        self.path.write_bytes(self.path.read_bytes() + b"\n")
        with self.assertRaises(ValueError):
            self.book.edit(**args)

    def test_noop_ambiguous_and_out_of_scope_edits_are_rejected(self):
        for change in (dict(old="return", new="return"), dict(old="not present", new="x")):
            with self.assertRaises(ValueError):
                self.book.edit(**self.edit_args(**change))
        for name in ("../demo.ipynb", ".git/demo.ipynb"):
            with self.assertRaises(ValueError):
                self.book.read(name, 1)
        protected = NotebookWorkspace(self.workspace, protected_paths=["demo.ipynb"])
        with self.assertRaises(ValueError):
            protected.read("demo.ipynb", 1)
        other = self.ws / "other.ipynb"
        other.write_bytes(self.path.read_bytes())
        args = self.edit_args(path="other.ipynb")
        with self.assertRaises(ValueError):
            self.book.edit(**args)

    def test_cell_newlines_and_long_line_budget_are_explicit(self):
        d = copy.deepcopy(self.document)
        d["cells"][1]["source"] = "def transform(x):\r\n    return -100\r\n"
        self.path.write_text(json.dumps(d), encoding="utf-8")
        value = self.book.read("demo.ipynb", 1)
        self.book.edit(**self.edit_args(old=value["text"], new="def transform(x):\n    return x+7"))
        d["cells"][1]["source"] = 'payload="' + "a" * 20000 + '"\n'
        self.path.write_text(json.dumps(d), encoding="utf-8")
        limited = NotebookWorkspace(self.workspace, count_tokens=len, token_limit=600).read(
            "demo.ipynb", 1
        )
        self.assertEqual(limited["covered_line_count"], 0)
        self.assertEqual(limited["next_read_line"], 1)
        self.assertFalse(limited["source_complete"])

    def test_opaque_index_and_observed_refresh_never_invent_python_file_lines(self):
        with CodeIndex(self.ws, self.root / "index.sqlite") as index:
            index.refresh()
            self.assertIn("demo.ipynb", {f["path"] for f in index.files()})
            self.assertEqual(
                index.db.execute(
                    "SELECT COUNT(*) FROM symbols WHERE path=?", ("demo.ipynb",)
                ).fetchone()[0],
                0,
            )
            self.book.edit(**self.edit_args())

            def read_source(name):
                raw = (self.ws / name).read_bytes()
                return raw, raw.decode("utf-8-sig"), hashlib.sha256(raw).hexdigest()

            index.refresh_paths(["demo.ipynb"], read_source=read_source)
            self.assertEqual(
                index.db.execute("SELECT hash FROM files WHERE path=?", ("demo.ipynb",)).fetchone()[
                    0
                ],
                hashlib.sha256(self.path.read_bytes()).hexdigest(),
            )

    def test_real_runtime_cells_read_edit_verify_finish_and_automatic_route(self):
        def respond(n, messages, schema):
            if n == 1:
                return action("notebook_cells", path="demo.ipynb")
            if n == 2:
                return action("notebook_read", path="demo.ipynb", cell=1)
            if n == 3:
                result = json.loads(messages[-1]["content"])["observation"]
                self.assertIn("return -100", result["text"])
                return action(
                    "notebook_edit",
                    path="demo.ipynb",
                    cell=1,
                    expected_hash=result["sha256"],
                    expected_cell_hash=result["cell_sha256"],
                    old="return -100",
                    new="return x+7",
                )
            return action("verify" if n == 4 else "finish")

        runtime = self.runtime(respond)
        state = runtime.execute(self.spec())
        self.assertEqual(
            state["status"],
            "succeeded",
            [
                (e["kind"], e["data"].get("error"))
                for e in runtime.store.events()
                if e["kind"] == "action_result"
            ],
        )
        self.assertEqual(state["model_calls"], 5)
        self.assertIsNone(runtime.worker_path)
        self.assertEqual(state["changed_files"], ["demo.ipynb"])
        self.assertTrue(runtime.store.diff())
        self.assertFalse(state.get("read_coverage", {}).get("demo.ipynb"))
        self.assertEqual(sum(e["kind"] == "notebook_edit" for e in runtime.store.events()), 1)

    def test_revoked_tool_and_planning_stage_cannot_bypass_original_editor(self):
        from infinicoder.runtime.protocol import validate_dispatch, ActionProtocolError

        runtime = self.runtime(lambda *_: action("finish"))
        edit = action("notebook_edit", **self.edit_args())
        with self.assertRaises(ActionProtocolError):
            validate_dispatch(runtime, edit, stage="planning")

        def respond(*_):
            runtime.notebook_tools = False
            return edit

        runtime.client.generate_structured = lambda messages, schema, timeout: ModelResponse(
            json.dumps(respond()), usage=dict(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        )
        runtime.max_model_calls = 1
        before = self.path.read_bytes()
        state = runtime.execute(self.spec())
        self.assertEqual(self.path.read_bytes(), before)
        self.assertNotEqual(state["status"], "succeeded")

    def test_false_option_rejects_notebook_actions_and_preserves_original_route(self):
        runtime = self.runtime(
            lambda *_: action("notebook_read", path="demo.ipynb", cell=1),
            notebook_tools=False,
            max_model_calls=1,
            automatic_workers=False,
        )
        state = runtime.execute(self.spec())
        self.assertNotEqual(state["status"], "succeeded")
        self.assertFalse(
            any(
                e["kind"] == "action_started" and e["data"]["tool"] == "notebook_read"
                for e in runtime.store.events()
            )
        )

    def test_pause_and_new_runtime_resume_do_not_replay_committed_cell_edit(self):
        def respond(n, messages, schema):
            return (
                action("notebook_read", path="demo.ipynb", cell=1)
                if n == 1
                else action("notebook_edit", **self.edit_args())
            )

        def control(event):
            if event["kind"] == "notebook_edit":
                runtime.store.request_control("pause")

        runtime = self.runtime(respond, event_handler=control, tool_discovery=True)
        spec = self.spec()
        state = runtime.execute(spec)
        self.assertEqual(state["status"], "paused")
        self.assertEqual(state["model_calls"], 2)
        committed = self.path.read_bytes()
        resumed = self.runtime(
            lambda n, *_: action("verify" if n == 1 else "finish"), tool_discovery=True
        )
        state = resumed.execute(spec, resume=True)
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(state["model_calls"], 4)
        self.assertEqual(self.path.read_bytes(), committed)
        self.assertEqual(sum(e["kind"] == "notebook_edit" for e in resumed.store.events()), 1)

    def test_repeated_inspection_does_not_select_whole_notebook_worker(self):
        def respond(n, *_):
            if n <= 3:
                return action("notebook_read", path="demo.ipynb", cell=1)
            if n == 4:
                return action("notebook_edit", **self.edit_args())
            return action("verify" if n == 5 else "finish")

        runtime = self.runtime(respond)
        state = runtime.execute(self.spec())
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(state["model_calls"], 6)
        self.assertIsNone(state.get("fallback_path"))

    def test_model_phase_uses_cell_actions_without_inferred_whole_file_worker(self):
        def respond(n, *_):
            if n == 1:
                return action(
                    "plan",
                    units=[
                        dict(
                            goal="Implement transform(x) as x+7.",
                            write_set=["demo.ipynb"],
                            check_ids=["external"],
                            interfaces=[],
                            dependencies=[],
                        )
                    ],
                )
            if n == 2:
                return action("notebook_read", path="demo.ipynb", cell=1)
            if n == 3:
                return action("notebook_edit", **self.edit_args())
            return action("finish")

        runtime = self.runtime(respond)
        spec = self.spec()
        spec.allowed_paths.append("other.py")
        state = runtime.execute(spec)
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(state["model_calls"], 4)

    def test_explicit_notebook_worker_is_not_silently_overridden(self):
        runtime = self.runtime(lambda *_: action("finish"))
        spec = self.spec()
        spec.milestones = [
            dict(
                goal=spec.task,
                write_set=["demo.ipynb"],
                check_ids=["external"],
                worker_path="demo.ipynb",
            )
        ]
        with self.assertRaises(ValueError):
            runtime.execute(spec)


if __name__ == "__main__":
    unittest.main()
