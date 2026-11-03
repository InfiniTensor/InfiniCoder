"""Owned controls exercise real HTTP,Runtime edits and sole-final selection.

No target-model inference or benchmark/gold candidate is used. The tiny
arithmetic fixture and scripted service exist solely for execution controls.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

from infinicoder.runtime.agent_execution import CoreBudget, CoreRequest, FinalAcceptance
from infinicoder.models.budget import ModelProfile
from infinicoder import (
    CollaborationPolicy,
    CollaborationRuntime,
    CollaborationLimits,
    run_collaboration,
)
from infinicoder.collaboration.collaboration_budget import BudgetedRoleClient, CollaborationBudget
from infinicoder.models.model_client import CompatibleClient, ModelConfig, ProviderError
from infinicoder.planning.task_contract import Check, TaskSpec
from infinicoder.execution.workspace_tools import ProcessCancelled, run_process


@contextmanager
def service(answer):
    requests = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                requests.append(body)
            content, usage = answer(body)
            raw = json.dumps(
                dict(
                    choices=[dict(message=dict(content=content), finish_reason="stop")], usage=usage
                )
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    client = CompatibleClient(
        ModelConfig(
            f"http://127.0.0.1:{server.server_port}/v1",
            "owned-scripted",
            max_output_tokens=256,
            temperature=0,
            structured_outputs=True,
        )
    )
    try:
        yield client, requests
    finally:
        server.shutdown()
        server.server_close()
        t.join()


USAGE = dict(prompt_tokens=10, completion_tokens=5, total_tokens=15)


def bound(messages, schema, output):
    return 8192


def owned_source_tokens(text):
    return len(text.encode("utf-8"))


def source_schema(value):
    if isinstance(value, dict):
        return "source" in value.get("properties", {}) or any(
            source_schema(v) for v in value.values()
        )
    return isinstance(value, list) and any(source_schema(v) for v in value)


class BudgetControls(unittest.TestCase):
    def test_resume_and_pause_reject_before_state_or_model(self):
        with (
            tempfile.TemporaryDirectory() as td,
            service(lambda body: ("{}", USAGE)) as (client, requests),
        ):
            root = Path(td)
            ws = root / "workspace"
            ws.mkdir()
            for flags in (dict(resume=True), dict(pause_required=True)):
                request = CoreRequest(
                    TaskSpec("owned preflight"),
                    ws,
                    root / "state",
                    ModelProfile(16384, 256, 256),
                    CoreBudget(3, 10, 2),
                    **flags,
                )
                with self.assertRaisesRegex(ValueError, "cannot resume/pause"):
                    run_collaboration(
                        request,
                        client=client,
                        runtime_factory=lambda c: None,
                        final_acceptance=lambda *a: None,
                        limits=CollaborationLimits(3, 10, 1),
                        request_token_bound=bound,
                        tokenizer_identity="owned",
                        exclusive_workspace=True,
                    )
                self.assertFalse((root / "state").exists())
            self.assertEqual(requests, [])

    def test_atomic_last_slot_plain_and_structured(self):
        with (
            tempfile.TemporaryDirectory() as td,
            service(lambda body: ("{}", USAGE)) as (client, requests),
        ):
            owner = CollaborationBudget(
                CollaborationLimits(3, 20, 1),
                Path(td) / "budget.jsonl",
                request_token_bound=bound,
                tokenizer_identity="owned synthetic counter",
            )
            owner.allocate({"A": 3})
            role = BudgetedRoleClient(client, owner, "A")
            role.generate([dict(role="user", content="plain")], 2)
            role.generate_structured([dict(role="user", content="schema")], {"type": "object"}, 2)
            barrier = threading.Barrier(2)

            def race():
                barrier.wait()
                try:
                    role.generate([dict(role="user", content="last")], 2)
                    return True
                except ProviderError:
                    return False

            with ThreadPoolExecutor(2) as pool:
                results = list(pool.map(lambda _: race(), range(2)))
            self.assertEqual(sorted(results), [False, True])
            self.assertEqual(len(requests), 3)
            self.assertIn("response_format", requests[1])
            self.assertEqual(owner.snapshot()["known_tokens"], 45)
            self.assertEqual(owner.snapshot()["inflight"], 0)
            owner.close()

    def test_unknown_usage_stops_without_refunding(self):
        with (
            tempfile.TemporaryDirectory() as td,
            service(lambda body: ("{}", {})) as (client, requests),
        ):
            owner = CollaborationBudget(
                CollaborationLimits(3, 20, 1),
                Path(td) / "budget.jsonl",
                request_token_bound=bound,
                tokenizer_identity="owned",
            )
            owner.allocate({"A": 3})
            role = BudgetedRoleClient(client, owner, "A")
            with self.assertRaises(ProviderError):
                role.generate([], 2)
            with self.assertRaises(ProviderError):
                role.generate([], 2)
            self.assertEqual(len(requests), 1)
            self.assertEqual(owner.snapshot()["unknown_usage_requests"], 1)
            self.assertEqual(owner.snapshot()["physical_request_attempts"], 1)
            owner.close()

    def test_late_paid_response_is_not_returned(self):
        started = threading.Event()
        release = threading.Event()

        def answer(body):
            started.set()
            release.wait(3)
            return ("late edit", USAGE)

        with tempfile.TemporaryDirectory() as td, service(answer) as (client, requests):
            owner = CollaborationBudget(
                CollaborationLimits(3, 20, 1),
                Path(td) / "budget.jsonl",
                request_token_bound=bound,
                tokenizer_identity="owned",
            )
            owner.allocate({"A": 3})
            role = BudgetedRoleClient(client, owner, "A")
            with ThreadPoolExecutor(1) as pool:
                f = pool.submit(role.generate, [], 4)
                self.assertTrue(started.wait(2))
                owner.stop("owned_cancel")
                release.set()
                with self.assertRaises(ProviderError):
                    f.result()
            self.assertEqual(owner.snapshot()["known_tokens"], 15)
            self.assertEqual(owner.snapshot()["inflight"], 0)
            owner.close()

    def test_process_cancel_stops_owned_group(self):
        event = threading.Event()
        timer = threading.Timer(0.15, event.set)
        timer.start()
        started = time.monotonic()
        try:
            with self.assertRaises(ProcessCancelled):
                run_process(
                    [sys.executable, "-B", "-c", "import time;time.sleep(20)"],
                    os.getcwd(),
                    5,
                    cancelled=event.is_set,
                )
        finally:
            timer.cancel()
        self.assertLess(time.monotonic() - started, 3)


@unittest.skipUnless(os.name == "posix", "complete source checkpoint forks require POSIX")
class RuntimeVerticalControl(unittest.TestCase):
    def _simple(self, hook=None, *, cancel_response=False):
        """One real source-worker path;hooks simulate external state changes."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ws = root / "base"
            ws.mkdir()
            before = "def add(a, b):\n    return a - b\n"
            correct = "def add(a, b):\n    return a + b\n"
            (ws / "solution.py").write_text(before)
            check = root / "check_owned.py"
            check.write_text(
                'import sys\nfrom pathlib import Path\nn={}\nexec((Path(sys.argv[1])/"solution.py").read_text(),n)\nassert n["add"](3,2)==5\n'
            )
            spec = TaskSpec(
                "Implement add in solution.py.",
                checks=[
                    Check(
                        "owned_public",
                        [sys.executable, "-B", str(check), "{workspace}"],
                        4,
                        protected_paths=[str(check)],
                    )
                ],
                allowed_paths=["solution.py"],
            )
            event = threading.Event()
            finals = []
            runtimes = []

            def answer(body):
                if cancel_response:
                    event.set()
                return json.dumps(dict(source=correct)), USAGE

            class HookRuntime(CollaborationRuntime):
                def execute(self, *args, **kwargs):
                    result = super().execute(*args, **kwargs)
                    if hook:
                        hook(self, ws, check)
                    return result

            with service(answer) as (client, requests):
                profile = ModelProfile(16384, 256, 256)
                req = CoreRequest(spec, ws, root / "session", profile, CoreBudget(6, 15, 4))

                def factory(c):
                    r = HookRuntime(
                        c.workspace,
                        c.client,
                        c.state_dir,
                        max_steps=c.call_quota,
                        max_model_calls=c.call_quota,
                        time_limit=10,
                        request_timeout=4,
                        model_profile=profile,
                        automatic_workers=c.automatic_workers,
                        source_fallback=c.source_fallback,
                        profile="general",
                        worker_format="json_source",
                        domain_compiler=False,
                    )
                    runtimes.append(r)
                    return r

                def final(path, spec, remaining):
                    finals.append(path)
                    return FinalAcceptance(
                        True, "owned final should run once only after valid selection"
                    )

                result = None
                error = None
                try:
                    result = run_collaboration(
                        req,
                        client=client,
                        runtime_factory=factory,
                        final_acceptance=final,
                        limits=CollaborationLimits(6, 15, 5),
                        policy=CollaborationPolicy("single"),
                        request_token_bound=bound,
                        tokenizer_identity="owned",
                        exclusive_workspace=True,
                        cancel_event=event,
                        source_token_counter=owned_source_tokens,
                    )
                except Exception as exc:
                    error = type(exc).__name__
                branch_unchanged = bool(
                    runtimes and (Path(runtimes[0].workspace) / "solution.py").read_text() == before
                )
                return result, error, len(finals), branch_unchanged, len(requests)

    def test_stale_public_pass_is_not_reused(self):
        def mutate(r, ws, checker):
            (Path(r.workspace) / "solution.py").write_text("def add(a,b): return 0\n")

        result, error, finals, _, _ = self._simple(mutate)
        self.assertIsNone(error)
        self.assertEqual(finals, 0)
        self.assertEqual(result.status, "public_verification_unavailable")
        self.assertEqual(result.public_observations[0].status, "unknown")

    def test_outside_scope_and_baseline_and_checker_changes_reject_handoff(self):
        hooks = [
            lambda r, ws, c: (Path(r.workspace) / "outside.txt").write_text("not authorized"),
            lambda r, ws, c: (ws / "solution.py").write_text("external concurrent change"),
            lambda r, ws, c: c.write_text("changed checker"),
        ]
        for hook in hooks:
            with self.subTest(hook=hook):
                _, error, finals, _, _ = self._simple(hook)
                self.assertEqual(error, "CheckpointConflict")
                self.assertEqual(finals, 0)

    def test_parent_cancel_before_response_cannot_edit(self):
        _, error, finals, unchanged, requests = self._simple(cancel_response=True)
        self.assertEqual(error, "CollaborationStopped")
        self.assertEqual(finals, 0)
        self.assertTrue(unchanged)
        self.assertEqual(requests, 1)

    def test_distinct_consumers_public_choice_and_single_final(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ws = root / "original"
            ws.mkdir()
            source = "def add(a, b):\n    return a - b\n"
            (ws / "solution.py").write_text(source)
            digest = hashlib.sha256(source.encode()).hexdigest()
            checker = root / "public_check.py"
            checker.write_text(
                'from pathlib import Path\nimport sys\nn={}\nexec((Path(sys.argv[1])/"solution.py").read_text(),n)\nassert n["add"](3,2)==5\n'
            )
            spec = TaskSpec(
                "Implement add in solution.py while preserving its signature.",
                checks=[
                    Check(
                        "public_add",
                        [sys.executable, "-B", str(checker), "{workspace}"],
                        5,
                        protected_paths=[str(checker)],
                    )
                ],
                allowed_paths=["solution.py"],
            )
            counter = {}
            lock = threading.Lock()

            def answer(body):
                messages = body["messages"]
                text = json.dumps(messages)
                schema = body.get("response_format", {}).get("json_schema", {}).get("schema", {})
                if "proposals" in schema.get("properties", {}):
                    proposals = []
                    for strategy, approach in [
                        (
                            "auto_source",
                            "Replace the complete arithmetic function from its interface.",
                        ),
                        (
                            "ordinary_actions",
                            "Inspect the arithmetic expression and perform a minimal operator edit.",
                        ),
                    ]:
                        proposals.append(
                            dict(
                                strategy=strategy,
                                approach=approach,
                                assumptions=[],
                                interfaces=["add(a,b)"],
                                write_paths=["solution.py"],
                                source_refs=[dict(path="solution.py", sha256=digest)],
                                verification=["public_add"],
                            )
                        )
                    return json.dumps(dict(proposals=proposals)), USAGE
                role = "B" if '"strategy": "ordinary_actions"' in text.replace('\\"', '"') else "A"
                with lock:
                    step = counter.get(role, 0)
                    counter[role] = step + 1
                if source_schema(schema):
                    return json.dumps(dict(source="def add(a, b):\n    return 0\n")), USAGE
                if role == "B" and step == 1:
                    return json.dumps(
                        dict(
                            tool="edit",
                            arguments=dict(
                                path="solution.py", expected_hash=digest, old="a - b", new="a + b"
                            ),
                        )
                    ), USAGE
                if role == "B" and step >= 2:
                    return json.dumps(dict(tool="finish", arguments={})), USAGE
                return json.dumps(dict(tool="read", arguments=dict(path="solution.py"))), USAGE

            final_calls = []
            runtimes = []

            def final(path, original, remaining):
                final_calls.append(path)
                self.assertEqual(asdict(original), asdict(spec))
                ns = {}
                exec((path / "solution.py").read_text(), ns)
                return FinalAcceptance(ns["add"](-2, 5) == 3, "owned independent arithmetic check")

            with service(answer) as (client, requests):
                profile = ModelProfile(16384, 256, 256)
                request = CoreRequest(spec, ws, root / "session", profile, CoreBudget(15, 35, 5))

                def factory(c):
                    r = CollaborationRuntime(
                        c.workspace,
                        c.client,
                        c.state_dir,
                        max_steps=c.call_quota,
                        max_model_calls=c.call_quota,
                        time_limit=30,
                        request_timeout=5,
                        model_profile=profile,
                        automatic_workers=c.automatic_workers,
                        source_fallback=c.source_fallback,
                        profile="general",
                        domain_compiler=False,
                        worker_format="json_source",
                    )
                    runtimes.append(r)
                    return r

                result = run_collaboration(
                    request,
                    client=client,
                    runtime_factory=factory,
                    final_acceptance=final,
                    limits=CollaborationLimits(15, 35, 5),
                    request_token_bound=bound,
                    tokenizer_identity="owned synthetic counter",
                    planning_context=source,
                    exclusive_workspace=True,
                    source_token_counter=owned_source_tokens,
                )
            self.assertTrue(result.success)
            self.assertEqual(result.selected_branch, "B")
            self.assertEqual(len(final_calls), 1)
            self.assertEqual((ws / "solution.py").read_text(), source)
            self.assertEqual(result.usage["physical_request_attempts"], len(requests))
            self.assertLessEqual(len(requests), 15)
            self.assertTrue(any(source_schema(r.get("response_format", {})) for r in requests))
            self.assertEqual(
                {o.branch_id: o.status for o in result.public_observations},
                {"A": "failed", "B": "passed"},
            )
            self.assertFalse(runtimes[1].source_fallback)
            states = [r.store.load() for r in runtimes]
            self.assertNotEqual(runtimes[0].state_dir, runtimes[1].state_dir)
            self.assertEqual(states[1]["status"], "succeeded")


if __name__ == "__main__":
    unittest.main()
