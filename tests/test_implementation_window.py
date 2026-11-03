"""Owned HTTP transport and real editor/check consumers;zero target models."""

from contextlib import contextmanager
from copy import deepcopy
import json, re, sys, tempfile, threading, time, unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from infinicoder import Runtime, TaskSpec, Check, NotebookWorkspace
from infinicoder.models.budget import TokenEstimator, ModelProfile
from infinicoder.models.model_client import CompatibleClient, ModelConfig
from infinicoder.execution.workspace_tools import Workspace
from infinicoder.analysis.task_compiler import compile_task
from infinicoder.runtime import implementation_window as window


def count(text):
    return len(re.findall(r"\w+|[^\w\s]", text))


def bound(messages, schema, output):
    return (
        count(json.dumps(messages, ensure_ascii=False)) + count(json.dumps(schema)) + output + 256
    )


def action(name, **args):
    return dict(tool=name, arguments=args)


def tool_names(schema):
    result = set()

    def visit(v):
        if isinstance(v, dict):
            tool = v.get("properties", {}).get("tool", {})
            if isinstance(tool, dict) and isinstance(tool.get("const"), str):
                result.add(tool["const"])
            for x in v.values():
                visit(x)
        elif isinstance(v, list):
            for x in v:
                visit(x)

    visit(schema)
    return result


@contextmanager
def owned_http(respond):
    requests = []
    errors = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            try:
                reply = respond(len(requests), payload)
                body = json.dumps(
                    dict(
                        choices=[
                            dict(message=dict(content=json.dumps(reply)), finish_reason="stop")
                        ],
                        usage=dict(prompt_tokens=3, completion_tokens=2, total_tokens=5),
                    )
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except BaseException as exc:
                errors.append(exc)
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"owned control error")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:" + str(server.server_port) + "/v1", requests, errors
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class ImplementationWindow(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.ws = self.root / "workspace"
        self.ws.mkdir()
        self.document = dict(
            nbformat=4,
            nbformat_minor=5,
            metadata=dict(preserve="caller metadata"),
            cells=[
                dict(
                    cell_type="code",
                    metadata={},
                    source=["def transform(x):\n", "    return -100\n"],
                    execution_count=None,
                    outputs=[],
                ),
                dict(
                    cell_type="code",
                    metadata={},
                    source=["support = 17\n"],
                    execution_count=None,
                    outputs=[],
                ),
            ],
        )
        (self.ws / "demo.ipynb").write_text(json.dumps(self.document), encoding="utf-8")
        self.book = NotebookWorkspace(Workspace(self.ws, ["demo.ipynb"]), count_tokens=count)
        checker = self.root / "public-check.py"
        checker.write_text(
            'import json,sys\nenv={}\nn=json.load(open(sys.argv[1]+"/demo.ipynb",encoding="utf-8"))\nfor c in n["cells"]:\n if c["cell_type"]=="code":exec("".join(c["source"]) if isinstance(c["source"],list) else c["source"],env)\nassert [env["transform"](i) for i in (-2,0,9)]==[5,7,16]\nassert env["support"]==17\n',
            encoding="utf-8",
        )
        self.spec = TaskSpec(
            "Update transform(x) to return x+7 in demo.ipynb. Preserve support,other source,metadata and the original checks.",
            allowed_paths=["demo.ipynb"],
            checks=[
                Check(
                    "external",
                    [sys.executable, str(checker), "{workspace}"],
                    protected_paths=[str(checker)],
                )
            ],
        )

    def runtime(self, url, **extra):
        state_dir = extra.pop("state_dir", self.root / "session")
        options = dict(
            profile="general",
            domain_compiler=False,
            automatic_workers=False,
            source_fallback=False,
            notebook_tools=True,
            implementation_window=True,
            model_profile=ModelProfile(16384, 512, 256),
            estimator=TokenEstimator(count, bound),
            max_steps=12,
            max_model_calls=12,
            time_limit=40,
        )
        options.update(extra)
        return Runtime(
            self.ws,
            CompatibleClient(
                ModelConfig(
                    url,
                    "owned-scripted-http",
                    max_output_tokens=512,
                    temperature=0,
                    structured_outputs=True,
                )
            ),
            state_dir,
            **options,
        )

    def packet(self, payload):
        for message in payload["messages"]:
            text = message["content"]
            pos = text.find(window.MARKER)
            if pos >= 0:
                start = text.index("\n", pos) + 1
                return json.JSONDecoder().raw_decode(text[start:])[0]
        self.fail("Complete implementation packet absent from actual HTTP payload")

    def scripted_edit(self, n, payload):
        if n <= 3:
            return action("notebook_read", path="demo.ipynb", cell=0, count=(100, 4, 120)[n - 1])
        if n == 4:
            packet = self.packet(payload)
            source = packet["source"]
            self.assertEqual(source["text"], "".join(self.document["cells"][0]["source"]))
            self.assertIn(self.spec.task, json.dumps(payload, ensure_ascii=False))
            names = tool_names(payload["response_format"]["json_schema"]["schema"])
            self.assertIn("notebook_edit", names)
            self.assertIn("leave_implementation", names)
            self.assertNotIn("files", names)
            self.assertEqual(packet["latest_action_feedback"]["action"]["tool"], "notebook_read")
            return action(
                "notebook_edit",
                path="demo.ipynb",
                cell=0,
                expected_hash=source["sha256"],
                expected_cell_hash=source["cell_sha256"],
                old="return -100",
                new="return x+7",
            )
        return action("verify" if n == 5 else "finish")

    def test_actual_HTTP_complete_source_to_original_edit_public_and_final(self):
        with owned_http(self.scripted_edit) as (url, requests, errors):
            runtime = self.runtime(url)
            state = runtime.execute(self.spec)
        self.assertFalse(errors)
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(state["model_calls"], 6)
        self.assertEqual(len(requests), 6)
        record = state["implementation_window"]
        self.assertEqual(record["calls_used"], 1)
        self.assertEqual(record["delivered_requests"], 1)
        events = list(runtime.store.events())
        self.assertEqual(sum(e["kind"] == "implementation_window_requested" for e in events), 1)
        prepared = next(e["data"] for e in events if e["kind"] == "implementation_window_prepared")
        self.assertLessEqual(prepared["physical_bound"], 16384)
        self.assertFalse(prepared["source_delivered"])
        received = next(e["data"] for e in events if e["kind"] == "implementation_window_response")
        self.assertTrue(received["source_delivered"])
        self.assertEqual(received["status"], "response_received")
        self.assertTrue(state["changed_files"] == ["demo.ipynb"])
        self.assertTrue(runtime.store.diff())
        self.assertIn("files", tool_names(requests[4]["response_format"]["json_schema"]["schema"]))

    def control_state(self, runtime, spec=None):
        spec = spec or self.spec
        runtime._graph = compile_task(spec)
        return dict(
            milestone=0,
            model_calls=0,
            planning_required=False,
            clarifications=[],
            session_deadline_epoch=time.time() + 30,
        )

    def test_pagination_counts_complete_coverage_rounds_not_read_arguments(self):
        runtime = self.runtime("http://127.0.0.1:1/v1")
        state = self.control_state(runtime)
        ws = Workspace(self.ws, self.spec.allowed_paths)
        for i in range(6):
            start = 1 + i % 2
            result = self.book.read("demo.ipynb", 0, start=start, count=1)
            window.observe(
                runtime,
                self.spec,
                state,
                ws,
                action("notebook_read", path="demo.ipynb", cell=0, start=start, count=1),
                result,
            )
            self.assertEqual(bool(state.get("implementation_window_attempted")), i == 5)
        self.assertEqual(state["implementation_window"]["status"], "pending")

    def test_unrelated_source_change_resets_rounds_and_stage_gates_hold(self):
        runtime = self.runtime("http://127.0.0.1:1/v1")
        state = self.control_state(runtime)
        ws = Workspace(self.ws, self.spec.allowed_paths)
        for _ in range(2):
            window.observe(
                runtime,
                self.spec,
                state,
                ws,
                action("notebook_read"),
                self.book.read("demo.ipynb", 0),
            )
        data = deepcopy(self.document)
        data["cells"][1]["source"] = ["support=18\n"]
        (self.ws / "demo.ipynb").write_text(json.dumps(data))
        window.observe(
            runtime, self.spec, state, ws, action("notebook_read"), self.book.read("demo.ipynb", 0)
        )
        self.assertFalse(state.get("implementation_window_attempted"))
        self.assertEqual(next(iter(state["implementation_window_reads"].values()))["rounds"], 1)
        self.assertFalse(window.eligible(runtime, self.spec, state, ordinary=False))
        state["planning_required"] = True
        self.assertFalse(window.eligible(runtime, self.spec, state))
        state["planning_required"] = False
        runtime._explicit_worker_path = "demo.ipynb"
        self.assertFalse(window.eligible(runtime, self.spec, state))

    def test_three_requests_end_without_resetting_original_stagnation_or_quota(self):
        def respond(n, payload):
            if n <= 6:
                if n >= 4:
                    self.packet(payload)
                return action(
                    "notebook_read", path="demo.ipynb", cell=0, count=(100, 4, 120)[(n - 1) % 3]
                )
            return action("ask_user", question="Need a different implementation approach.")

        with owned_http(respond) as (url, requests, errors):
            runtime = self.runtime(url)
            state = runtime.execute(self.spec)
        self.assertFalse(errors)
        self.assertEqual(state["implementation_window"]["calls_used"], 3)
        self.assertEqual(
            state["implementation_window"]["reason"], "three_original_requests_consumed"
        )
        self.assertEqual(len(requests), 7)
        self.assertEqual(state["model_calls"], 7)
        self.assertGreaterEqual(max(state["no_progress"].values()), 2)
        self.assertEqual(
            sum(e["kind"] == "implementation_window_requested" for e in runtime.store.events()), 1
        )
        self.assertIn("files", tool_names(requests[-1]["response_format"]["json_schema"]["schema"]))

    def test_explicit_exit_preserves_broader_original_permissions(self):
        def respond(n, payload):
            if n <= 3:
                return action(
                    "notebook_read", path="demo.ipynb", cell=0, count=(100, 4, 120)[n - 1]
                )
            if n == 4:
                self.packet(payload)
                return action("leave_implementation", reason="Need another module before editing.")
            return action("ask_user", question="Which dependency is intended?")

        with owned_http(respond) as (url, requests, errors):
            runtime = self.runtime(url)
            state = runtime.execute(self.spec)
        self.assertFalse(errors)
        self.assertEqual(state["implementation_window"]["reason"], "model_requested_exit")
        self.assertIn("files", tool_names(requests[-1]["response_format"]["json_schema"]["schema"]))
        self.assertFalse(state["changed_files"])

    def test_source_drift_after_response_cannot_commit_a_stale_proposal(self):
        def respond(n, payload):
            if n == 4:
                proposed = self.scripted_edit(n, payload)
                data = deepcopy(self.document)
                data["cells"][1]["source"] = ["support=99\n"]
                (self.ws / "demo.ipynb").write_text(json.dumps(data))
                return proposed
            if n < 4:
                return self.scripted_edit(n, payload)
            return action("ask_user", question="Current source changed externally.")

        with owned_http(respond) as (url, requests, errors):
            runtime = self.runtime(url)
            state = runtime.execute(self.spec)
        self.assertFalse(errors)
        self.assertEqual(state["implementation_window"]["status"], "invalidated")
        self.assertFalse(state["changed_files"])
        self.assertIn("return -100", self.book.read("demo.ipynb", 0)["text"])

    def test_cancellation_after_delivery_does_not_dispatch_edit(self):
        holder = {}

        def respond(n, payload):
            proposed = self.scripted_edit(n, payload)
            if n == 4:
                holder["runtime"].store.request_control("cancel")
            return proposed

        with owned_http(respond) as (url, requests, errors):
            runtime = self.runtime(url)
            holder["runtime"] = runtime
            state = runtime.execute(self.spec)
        self.assertFalse(errors)
        self.assertEqual(state["status"], "cancelled")
        self.assertFalse(state["changed_files"])
        self.assertEqual(state["implementation_window"]["calls_used"], 1)
        self.assertEqual(state["implementation_window"]["delivered_requests"], 1)
        self.assertFalse(
            any(
                e["kind"] == "action_started" and e["data"]["tool"] == "notebook_edit"
                for e in runtime.store.events()
            )
        )

    def test_protected_checker_change_after_response_invalidates_edit_lease(self):
        checker = Path(self.spec.checks[0].protected_paths[0])

        def respond(n, payload):
            if n <= 3:
                return self.scripted_edit(n, payload)
            if n > 4:
                return action("ask_user", question="The original checker changed externally.")
            proposed = self.scripted_edit(n, payload)
            checker.write_text('raise RuntimeError("Changed owned oracle")\n')
            return proposed

        with owned_http(respond) as (url, requests, errors):
            runtime = self.runtime(url)
            state = runtime.execute(self.spec)
        self.assertFalse(errors)
        self.assertEqual(state["implementation_window"]["status"], "invalidated")
        self.assertFalse(state["changed_files"])
        self.assertEqual(len(requests), 5)
        self.assertIn("return -100", self.book.read("demo.ipynb", 0)["text"])

    def test_physical_overflow_declines_to_pure_ordinary_assembly_without_free_query(self):
        def capped(messages, schema, output):
            if any(window.MARKER in m["content"] for m in messages):
                return 20000
            return bound(messages, schema, output)

        def respond(n, payload):
            if n <= 3:
                return action(
                    "notebook_read", path="demo.ipynb", cell=0, count=(100, 4, 120)[n - 1]
                )
            self.assertFalse(any(window.MARKER in m["content"] for m in payload["messages"]))
            self.assertIn("files", tool_names(payload["response_format"]["json_schema"]["schema"]))
            return action("ask_user", question="A broader source view is required.")

        with owned_http(respond) as (url, requests, errors):
            runtime = self.runtime(url, estimator=TokenEstimator(count, capped))
            state = runtime.execute(self.spec)
        self.assertFalse(errors)
        self.assertEqual(len(requests), 4)
        self.assertEqual(state["model_calls"], 4)
        self.assertEqual(state["implementation_window"]["status"], "declined")
        self.assertEqual(state["implementation_window"]["calls_used"], 0)
        self.assertFalse(state["changed_files"])

    def test_OFF_and_nontrigger_HTTP_messages_and_schema_are_identical(self):
        def respond(n, payload):
            return (
                action("read", path="demo.ipynb", count=2)
                if n == 1
                else action("ask_user", question="Owned stopping point.")
            )

        bodies = []
        for enabled in (False, True):
            with owned_http(respond) as (url, requests, errors):
                runtime = self.runtime(
                    url,
                    implementation_window=enabled,
                    state_dir=self.root / ("enabled" if enabled else "disabled"),
                )
                runtime.execute(self.spec)
            self.assertFalse(errors)
            bodies.append([{k: v for k, v in p.items() if k != "model"} for p in requests])
        self.assertEqual(bodies[0], bodies[1])

    def test_committed_edit_pause_resume_keeps_window_consumed_without_replay(self):
        holder = {}

        def listener(event):
            if event["kind"] == "notebook_edit":
                holder["runtime"].store.request_control("pause")

        with owned_http(self.scripted_edit) as (url, requests, errors):
            runtime = self.runtime(url, event_handler=listener)
            holder["runtime"] = runtime
            paused = runtime.execute(self.spec)
            self.assertEqual(paused["status"], "paused")
            self.assertEqual(paused["model_calls"], 4)
            resumed = self.runtime(url)
            state = resumed.execute(self.spec, resume=True)
        self.assertFalse(errors)
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(len(requests), 6)
        self.assertEqual(state["implementation_window"]["calls_used"], 1)
        self.assertEqual(sum(e["kind"] == "notebook_edit" for e in resumed.store.events()), 1)

    def test_current_source_is_rechecked_immediately_before_request_dispatch(self):
        def listener(event):
            if event["kind"] == "implementation_window_prepared":
                data = deepcopy(self.document)
                data["cells"][1]["source"] = ["support=99\n"]
                (self.ws / "demo.ipynb").write_text(json.dumps(data))

        def respond(n, payload):
            return (
                self.scripted_edit(n, payload)
                if n <= 3
                else action("ask_user", question="Source changed before dispatch.")
            )

        with owned_http(respond) as (url, requests, errors):
            runtime = self.runtime(url, event_handler=listener)
            state = runtime.execute(self.spec)
        self.assertFalse(errors)
        self.assertEqual(state["implementation_window"]["status"], "invalidated")
        self.assertEqual(state["implementation_window"]["calls_used"], 0)
        self.assertEqual(state["implementation_window"]["delivered_requests"], 0)
        self.assertFalse(state["changed_files"])
        self.assertEqual(len(requests), state["model_calls"])

    def test_markdown_empty_and_missing_exact_counter_do_not_activate(self):
        data = deepcopy(self.document)
        data["cells"].extend(
            [
                dict(cell_type="markdown", source=["Documentation only."], metadata={}),
                dict(cell_type="code", source=[], metadata={}, outputs=[], execution_count=None),
            ]
        )
        (self.ws / "demo.ipynb").write_text(json.dumps(data))
        runtime = self.runtime("http://127.0.0.1:1/v1")
        state = self.control_state(runtime)
        ws = Workspace(self.ws, self.spec.allowed_paths)
        for cell in (2, 3):
            for _ in range(3):
                window.observe(
                    runtime,
                    self.spec,
                    state,
                    ws,
                    action("notebook_read"),
                    self.book.read("demo.ipynb", cell),
                )
        self.assertFalse(state.get("implementation_window_attempted"))
        runtime.assembler.estimator = TokenEstimator(count)
        self.assertFalse(window.eligible(runtime, self.spec, state))

    def test_plain_Python_complete_module_uses_the_same_original_editor(self):
        source = self.ws / "module.py"
        source.write_text("def transform(x):\n    return -100\n")
        check = self.root / "module-check.py"
        check.write_text(
            'import runpy,sys\nm=runpy.run_path(sys.argv[1]+"/module.py")\nassert m["transform"](2)==9\n'
        )
        spec = TaskSpec(
            "Make transform(x) return x+7 in module.py.",
            allowed_paths=["module.py"],
            checks=[Check("module", [sys.executable, str(check), "{workspace}"])],
        )

        def respond(n, payload):
            if n <= 3:
                return action("read", path="module.py", count=(10, 20, 30)[n - 1])
            if n == 4:
                value = self.packet(payload)["source"]
                self.assertEqual(value["kind"], "python_module")
                return action(
                    "edit",
                    path="module.py",
                    expected_hash=value["sha256"],
                    old="return -100",
                    new="return x+7",
                )
            return action("verify" if n == 5 else "finish")

        with owned_http(respond) as (url, requests, errors):
            runtime = self.runtime(url, profile="legacy", notebook_tools=False)
            state = runtime.execute(spec)
        self.assertFalse(errors)
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(state["implementation_window"]["calls_used"], 1)

    def test_parent_budget_and_physical_prepared_prefix_use_same_window(self):
        from infinicoder.collaboration.collaboration import (
            CollaborationRuntime,
            BranchContext,
            _AdvisedClient,
        )
        from infinicoder.collaboration.collaboration_budget import (
            CollaborationBudget,
            CollaborationLimits,
        )

        with owned_http(self.scripted_edit) as (url, requests, errors):
            owner = CollaborationBudget(
                CollaborationLimits(12, 40, 5),
                self.root / "owner.jsonl",
                request_token_bound=bound,
                tokenizer_identity="owned-fixture-counter",
                request_context_limit=16384,
            )
            owner.allocate({"A": 12})
            client = CompatibleClient(
                ModelConfig(
                    url, "owned-scripted-http", max_output_tokens=512, structured_outputs=True
                )
            )
            role = _AdvisedClient(
                client,
                owner,
                "A",
                advice=dict(strategy="Owned public context prefix"),
                workspace=self.ws,
                count_source_tokens=count,
            )
            runtime = CollaborationRuntime(
                self.ws,
                role,
                self.root / "physical-session",
                profile="general",
                domain_compiler=False,
                automatic_workers=False,
                source_fallback=False,
                notebook_tools=True,
                implementation_window=True,
                model_profile=ModelProfile(16384, 512, 256),
                max_model_calls=12,
                max_steps=12,
                time_limit=35,
            )
            # Existing Runtime verification consumes a source-revision producer;
            # this owned scan has no checkpoint/fork performance claim.
            scan = SimpleNamespace(
                scan=lambda _: dict(
                    owned_revision=Workspace(self.ws, self.spec.allowed_paths).revision()
                )
            )
            runtime.bind_collaboration(
                BranchContext(
                    "A",
                    self.ws,
                    runtime.state_dir,
                    self.spec,
                    role,
                    ModelProfile(16384, 512, 256),
                    12,
                    False,
                    False,
                    owner,
                    scan,
                )
            )
            runtime.physical_context_budget = True
            runtime.assembler.physical_alignment = True
            try:
                state = runtime.execute(self.spec)
            finally:
                owner.close()
        self.assertFalse(errors)
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(owner.snapshot()["physical_request_attempts"], 6)
        self.assertEqual(owner.snapshot()["known_tokens"], 30)
        self.assertTrue(
            any("Owned public context prefix" in m["content"] for m in requests[3]["messages"])
        )
        self.assertEqual(state["implementation_window"]["calls_used"], 1)

    def test_unknown_transport_does_not_claim_source_delivered(self):
        def respond(n, payload):
            if n <= 3:
                return self.scripted_edit(n, payload)
            if n == 4:
                raise RuntimeError("Deliberate owned HTTP failure")
            return (
                action("leave_implementation", reason="Service reply unavailable.")
                if n == 5
                else action("ask_user", question="Owned failure stopping point.")
            )

        with owned_http(respond) as (url, requests, errors):
            runtime = self.runtime(url)
            state = runtime.execute(self.spec)
        self.assertEqual(len(errors), 1)
        events = [
            e["data"]
            for e in runtime.store.events()
            if e["kind"] == "implementation_window_response"
        ]
        self.assertTrue(events)
        self.assertEqual(events[0]["status"], "transport_outcome_unknown")
        self.assertFalse(events[0]["source_delivered"])
        self.assertFalse(state["changed_files"])
        self.assertLessEqual(state["implementation_window"]["calls_used"], 3)

    def test_parent_physical_overflow_reassembles_ordinary_once_without_dispatch(self):
        from infinicoder.collaboration.collaboration import (
            CollaborationRuntime,
            BranchContext,
            _AdvisedClient,
        )
        from infinicoder.collaboration.collaboration_budget import (
            CollaborationBudget,
            CollaborationLimits,
        )

        def capped(messages, schema, output):
            return (
                20000
                if any(window.MARKER in m["content"] for m in messages)
                else bound(messages, schema, output)
            )

        def respond(n, payload):
            if n <= 3:
                return self.scripted_edit(n, payload)
            self.assertFalse(any(window.MARKER in m["content"] for m in payload["messages"]))
            return action("ask_user", question="Owned physical fallback stopping point.")

        with owned_http(respond) as (url, requests, errors):
            owner = CollaborationBudget(
                CollaborationLimits(12, 40, 5),
                self.root / "owner-overflow.jsonl",
                request_token_bound=capped,
                tokenizer_identity="owned-fixture-counter",
                request_context_limit=16384,
            )
            owner.allocate({"A": 12})
            client = CompatibleClient(
                ModelConfig(
                    url, "owned-scripted-http", max_output_tokens=512, structured_outputs=True
                )
            )
            role = _AdvisedClient(
                client,
                owner,
                "A",
                advice=dict(strategy="Owned physical prefix"),
                workspace=self.ws,
                count_source_tokens=count,
            )
            runtime = CollaborationRuntime(
                self.ws,
                role,
                self.root / "physical-overflow-session",
                profile="general",
                domain_compiler=False,
                automatic_workers=False,
                source_fallback=False,
                notebook_tools=True,
                implementation_window=True,
                model_profile=ModelProfile(16384, 512, 256),
                max_model_calls=12,
                max_steps=12,
                time_limit=35,
            )
            scan = SimpleNamespace(
                scan=lambda _: dict(
                    owned_revision=Workspace(self.ws, self.spec.allowed_paths).revision()
                )
            )
            runtime.bind_collaboration(
                BranchContext(
                    "A",
                    self.ws,
                    runtime.state_dir,
                    self.spec,
                    role,
                    ModelProfile(16384, 512, 256),
                    12,
                    False,
                    False,
                    owner,
                    scan,
                )
            )
            runtime.physical_context_budget = True
            runtime.assembler.physical_alignment = True
            try:
                state = runtime.execute(self.spec)
            finally:
                owner.close()
        self.assertFalse(errors)
        self.assertEqual(len(requests), 4)
        self.assertEqual(owner.snapshot()["physical_request_attempts"], 4)
        self.assertEqual(state["implementation_window"]["status"], "declined")
        self.assertFalse(state["changed_files"])
        self.assertEqual(state["implementation_window"]["calls_used"], 0)

    def test_palette_exit_and_overflow_restore_original_catalog(self):
        def respond(n, payload):
            names = tool_names(payload["response_format"]["json_schema"]["schema"])
            if n <= 3:
                return action(
                    "notebook_read", path="demo.ipynb", cell=0, count=(100, 4, 120)[n - 1]
                )
            if n == 4:
                self.packet(payload)
                self.assertNotIn("tool_catalog", names)
                return action("leave_implementation", reason="Need the original catalog again.")
            self.assertIn("tool_catalog", names)
            return action("ask_user", question="Owned catalog stopping point.")

        with owned_http(respond) as (url, requests, errors):
            runtime = self.runtime(url, tool_discovery=True)
            state = runtime.execute(self.spec)
        self.assertFalse(errors)
        self.assertEqual(state["implementation_window"]["reason"], "model_requested_exit")

        def capped(messages, schema, output):
            return (
                20000
                if any(window.MARKER in m["content"] for m in messages)
                else bound(messages, schema, output)
            )

        def fallback(n, payload):
            if n <= 3:
                return action(
                    "notebook_read", path="demo.ipynb", cell=0, count=(100, 4, 120)[n - 1]
                )
            self.assertIn(
                "tool_catalog", tool_names(payload["response_format"]["json_schema"]["schema"])
            )
            self.assertIn("tool_palette", payload["messages"][1]["content"])
            self.assertFalse(any(window.MARKER in m["content"] for m in payload["messages"]))
            return action("ask_user", question="Owned catalog fallback stopping point.")

        with owned_http(fallback) as (url, requests, errors):
            runtime = self.runtime(
                url,
                tool_discovery=True,
                estimator=TokenEstimator(count, capped),
                state_dir=self.root / "palette-overflow",
            )
            state = runtime.execute(self.spec)
        self.assertFalse(errors)
        self.assertEqual(state["implementation_window"]["status"], "declined")


if __name__ == "__main__":
    unittest.main()
