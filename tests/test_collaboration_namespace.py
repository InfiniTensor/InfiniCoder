"""Finite owned execution controls, not coding-quality benchmark tasks."""

import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from infinicoder import (
    CollaborationLimits,
    CollaborationPolicy,
    NamespaceCollaborationRuntime,
    run_collaboration,
)
from infinicoder.runtime.agent_execution import CoreBudget, CoreRequest, FinalAcceptance
from infinicoder.models.budget import ModelProfile
from infinicoder.planning.task_contract import Check, TaskSpec
from test_collaboration import USAGE, bound, service, source_schema, owned_source_tokens
import test_collaboration as owned_controls
from namespace_support import namespace_command, require_namespace


class NamespaceBindingControls(unittest.TestCase):
    def test_builder_capturing_original_workspace_rejects_branch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            original = root / "original"
            branch = root / "branch"
            original.mkdir()
            branch.mkdir()

            def wrong_builder(workspace, command):
                return [
                    "bwrap",
                    "--bind",
                    str(original),
                    "/workspace",
                    "--tmpfs",
                    "/tmp",
                    "--chdir",
                    "/workspace",
                    *command,
                ]

            with service(lambda body: ("{}", USAGE)) as (client, requests):
                runtime = NamespaceCollaborationRuntime(
                    branch,
                    client,
                    root / "state",
                    namespace_builder=wrong_builder,
                    domain_compiler=False,
                )
                with self.assertRaisesRegex(ValueError, "exact branch or snapshot"):
                    runtime._namespace(branch, ["true"])
                self.assertEqual(requests, [])


class NamespaceRuntimeVerticalControl(unittest.TestCase):
    def setUp(self):
        self.bwrap = require_namespace()

    def test_coding_deadline_preserves_one_final_after_quiescence(self):
        def expire_search(runtime, workspace, checker):
            # The candidate already has a current public pass. Force the
            # search clock across its boundary while the parent is waiting.
            runtime._collaboration.owner.coding_deadline = time.monotonic() - 1
            time.sleep(0.15)

        result, error, finals, _, requests = owned_controls.RuntimeVerticalControl()._simple(
            expire_search
        )
        self.assertIsNone(error)
        self.assertTrue(result.success)
        self.assertEqual(finals, 1)
        self.assertTrue(result.usage["coding_stopped"])
        self.assertFalse(result.usage["stopped"])

    def test_real_branch_tmp_public_snapshot_selection_and_unique_final(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            workspace = root / "baseline"
            workspace.mkdir()
            before = "def add(a, b):\n    return a - b\n"
            (workspace / "solution.py").write_text(before)
            digest = hashlib.sha256(before.encode()).hexdigest()
            public = root / "public"
            public.mkdir()
            checker = public / "check.py"
            checker.write_text(
                "from pathlib import Path\nimport sys\n"
                'assert sys.argv[1] == "/workspace"\n'
                'assert not Path("/tmp/ordinary-marker").exists()\n'
                'ns={}\nexec((Path(sys.argv[1])/"solution.py").read_text(),ns)\n'
                'assert ns["add"](3,2)==5\n'
            )

            def builder(source, argv):
                return namespace_command(self.bwrap, source, public, argv)

            spec = TaskSpec(
                "Repair add in solution.py and preserve its interface. "
                "Commands execute in /workspace; /tmp persists for ordinary commands.",
                checks=[
                    Check(
                        "public_add",
                        ["/usr/bin/python3", "-B", str(checker), "{workspace}"],
                        6,
                        protected_paths=[str(checker)],
                    )
                ],
                allowed_paths=["solution.py"],
            )
            counts = {}
            lock = threading.Lock()
            runtimes = []
            finals = []

            def answer(body):
                schema = body.get("response_format", {}).get("json_schema", {}).get("schema", {})
                if "proposals" in schema.get("properties", {}):
                    return json.dumps(
                        dict(
                            proposals=[
                                dict(
                                    strategy=strategy,
                                    approach=approach,
                                    assumptions=[],
                                    interfaces=["add(a,b)"],
                                    write_paths=["solution.py"],
                                    source_refs=[dict(path="solution.py", sha256=digest)],
                                    verification=["public_add"],
                                )
                                for strategy, approach in [
                                    ("auto_source", "Propose a complete arithmetic body."),
                                    (
                                        "ordinary_actions",
                                        "Read and apply a minimal arithmetic edit.",
                                    ),
                                ]
                            ]
                        )
                    ), USAGE
                text = json.dumps(body["messages"]).replace('\\"', '"')
                role = "B" if '"strategy": "ordinary_actions"' in text else "A"
                with lock:
                    step = counts.get(role, 0)
                    counts[role] = step + 1
                if source_schema(schema):
                    return json.dumps(dict(source="def add(a,b):\n    return 0\n")), USAGE
                actions = [
                    dict(tool="read", arguments=dict(path="solution.py")),
                    dict(
                        tool="command",
                        arguments=dict(
                            argv=[
                                "/usr/bin/python3",
                                "-c",
                                'from pathlib import Path;Path("/tmp/ordinary-marker").write_text("branch-private")',
                            ]
                        ),
                    ),
                    dict(
                        tool="command",
                        arguments=dict(
                            argv=[
                                "/usr/bin/python3",
                                "-c",
                                'from pathlib import Path;assert Path("/tmp/ordinary-marker").read_text()=="branch-private"',
                            ]
                        ),
                    ),
                    dict(
                        tool="edit",
                        arguments=dict(
                            path="solution.py", expected_hash=digest, old="a - b", new="a + b"
                        ),
                    ),
                    dict(tool="finish", arguments={}),
                ]
                return json.dumps(actions[min(step, len(actions) - 1)]), USAGE

            def final(source, original, remaining):
                finals.append(source)
                ns = {}
                exec((source / "solution.py").read_text(), ns)
                return FinalAcceptance(ns["add"](-2, 5) == 3, "owned independent arithmetic final")

            with service(answer) as (client, requests):
                profile = ModelProfile(16384, 256, 256)
                request = CoreRequest(
                    spec, workspace, root / "session", profile, CoreBudget(19, 55, 6)
                )

                def factory(c):
                    runtime = NamespaceCollaborationRuntime(
                        c.workspace,
                        c.client,
                        c.state_dir,
                        namespace_builder=builder,
                        max_steps=c.call_quota,
                        max_model_calls=c.call_quota,
                        time_limit=40,
                        request_timeout=6,
                        model_profile=profile,
                        allow_commands=True,
                        automatic_workers=c.automatic_workers,
                        source_fallback=c.source_fallback,
                        profile="general",
                        worker_format="json_source",
                        domain_compiler=False,
                    )
                    runtimes.append(runtime)
                    return runtime

                outcome = run_collaboration(
                    request,
                    client=client,
                    runtime_factory=factory,
                    final_acceptance=final,
                    limits=CollaborationLimits(19, 55, 12),
                    policy=CollaborationPolicy("heterogeneous"),
                    request_token_bound=bound,
                    tokenizer_identity="owned synthetic bound",
                    exclusive_workspace=True,
                    source_token_counter=owned_source_tokens,
                )
            if not outcome.success:
                details = "\n".join(
                    (runtime.state_dir / "events.jsonl").read_text()[-12000:]
                    for runtime in runtimes
                )
                self.fail(f"Collaboration did not pass: {outcome}\n{details}")
            self.assertEqual(outcome.selected_branch, "B")
            self.assertEqual(len(finals), 1)
            self.assertEqual((workspace / "solution.py").read_text(), before)
            self.assertEqual(outcome.usage["physical_request_attempts"], len(requests))
            events = []
            for runtime in runtimes:
                self.assertTrue(runtime._command_files is None or runtime._command_files.closed)
                events += [
                    json.loads(line)
                    for line in (runtime.state_dir / "events.jsonl").read_text().splitlines()
                ]
            execution = [
                row["data"] for row in events if row["kind"] == "collaboration_namespace_execution"
            ]
            self.assertTrue(
                any(
                    x["source_kind"] == "branch" and x["persistent_command_files"]
                    for x in execution
                )
            )
            self.assertTrue(
                any(
                    x["source_kind"] == "public_snapshot" and not x["persistent_command_files"]
                    for x in execution
                )
            )
            self.assertTrue(all(not x["host_fallback"] for x in execution))


if __name__ == "__main__":
    unittest.main()
