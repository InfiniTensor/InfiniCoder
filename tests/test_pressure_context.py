"""Owned full-Runtime controls;scripted HTTP,not target-model ability scores."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from infinicoder.models.budget import BudgetExceeded, ModelProfile, TokenEstimator
from infinicoder.collaboration.collaboration import CollaborationRuntime, _AdvisedClient
from infinicoder.collaboration.collaboration_budget import CollaborationBudget, CollaborationLimits
from infinicoder.models.model_client import CompatibleClient
from infinicoder.memory.pressure_context import (
    configure_pressure_evidence,
    snapshot,
    assemble,
    _source,
)
from infinicoder.routing.profiles import route_task
from infinicoder.memory.source.source_handoff import project_turn
from infinicoder.memory.source.source_sync import observe
from infinicoder.analysis.task_compiler import compile_task
from infinicoder.planning.task_contract import Check, TaskSpec
from infinicoder.execution.workspace_tools import Workspace
from test_collaboration import service, USAGE
import test_physical_context as physical


def record(action, result, step):
    return dict(
        proposal=json.dumps(action),
        action=action,
        observation=result,
        call=step,
        step=step,
        phase=0,
        transport=dict(
            version=2,
            origin="model",
            protocol="json_action",
            accepted=True,
            dispatch="entered",
            dispatched_action=action,
            finish_reason="stop",
            usage=USAGE,
            reply_sha256="owned",
            source_before=None,
            source_after=None,
        ),
    )


def stats(row):
    path = os.environ.get("IC_PRESSURE_STATS")
    if path:
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")


class EnvelopeControls(unittest.TestCase):
    def test_complete_failure_is_visible_once_without_mutating_the_caller(self):
        from infinicoder.models.budget import ContextAssembler
        from infinicoder.runtime.feedback import failure_packet

        with tempfile.TemporaryDirectory() as td:
            workspace = Workspace(Path(td), [])
            failure = dict(error="owned required failure " * 280)
            packet = failure_packet(failure)
            pair = project_turn(
                record(dict(tool="verify", arguments={}), dict(accepted=False, failures=packet), 1),
                "json_action",
            )
            core = dict(
                task="Keep every original requirement.",
                allowed_paths=["alpha.py"],
                last_failure=packet,
                amendments=[dict(source="user", description="Keep exact interfaces.")],
            )
            mandatory = json.dumps(core) + "\nORIGINAL REGISTRY"
            state = dict(
                last_failure=failure, changed_files=[], read_paths=[], source_working_set=[]
            )
            before = deepcopy(state)
            original_pair = deepcopy(pair)
            assembler = ContextAssembler(profile=ModelProfile(4096, 512, 256))
            frozen = snapshot(state, workspace, mandatory, "Owned system.", [], [pair], [], {})
            messages, report = assemble(assembler, frozen, force=True)
            delivered = json.JSONDecoder().raw_decode(messages[1]["content"])[0]
            self.assertEqual(
                {k: v for k, v in delivered.items() if k != "last_failure"},
                {k: v for k, v in core.items() if k != "last_failure"},
            )
            self.assertIn("complete_packet_ref", delivered["last_failure"])
            self.assertEqual(messages[-1], pair[-1])
            self.assertEqual(json.loads(messages[-1]["content"])["observation"]["failures"], packet)
            self.assertEqual(state, before)
            self.assertEqual(pair, original_pair)
            self.assertEqual(json.loads(frozen.failure_json), None)
            self.assertTrue(report["joint_evidence"]["current_failure_reference"])
            # No matching protected packet means the complete original core
            # value remains;it is still never appended a second time.
            frozen = snapshot(state, workspace, mandatory, "Owned system.", [], [], [], {})
            self.assertEqual(
                json.JSONDecoder().raw_decode(frozen.mandatory)[0]["last_failure"], packet
            )
            self.assertIsNone(json.loads(frozen.failure_json))
            # A reference can be larger than a short error. Keep that full
            # value while still avoiding a third copy of the same packet.
            small = dict(error="small")
            small_packet = failure_packet(small)
            small_pair = project_turn(
                record(
                    dict(tool="verify", arguments={}),
                    dict(accepted=False, failures=small_packet),
                    2,
                ),
                "json_action",
            )
            small_core = json.dumps(dict(task=core["task"], last_failure=small_packet))
            frozen = snapshot(
                dict(state, last_failure=small),
                workspace,
                small_core,
                "Owned system.",
                [],
                [small_pair],
                [],
                {},
            )
            self.assertEqual(frozen.mandatory, small_core)
            self.assertIsNone(json.loads(frozen.failure_json))
            self.assertIsNone(json.loads(frozen.failure_reference_json))

    def test_projection_changes_only_duplicate_source_text_and_keeps_session(self):
        from infinicoder.models.budget import ContextAssembler

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "alpha.py").write_text("def alpha(x):\n    return x + 1\n")
            workspace = Workspace(root, ["alpha.py"])
            source = workspace.read("alpha.py", 1, 100)
            source["outline"] = [dict(name="alpha", line=1)]
            action = dict(tool="read", arguments=dict(path="alpha.py"))
            pairs = [project_turn(record(action, source, i), "json_action") for i in (1, 2)]
            pairs.append(
                project_turn(
                    record(
                        dict(tool="verify", arguments={}),
                        dict(accepted=False, failures=[dict(error="owned failure")]),
                        3,
                    ),
                    "json_action",
                )
            )
            state = dict(
                source_working_set=[
                    dict(
                        path="alpha.py",
                        origin_sha256=source["sha256"],
                        start=1,
                        count=2,
                        prefix_chars=None,
                        whole_file=True,
                        last_call=2,
                        last_step=2,
                        observation_sha256=_source(pairs[0])[3],
                    )
                ],
                last_failure=dict(error="owned failure", private_receipt_path="NEVER_DISCLOSE"),
                changed_files=[],
                read_paths=["alpha.py"],
            )
            original = deepcopy(pairs)
            assembler = ContextAssembler(profile=ModelProfile(8192, 1024, 256))
            frozen = snapshot(
                state,
                workspace,
                "Complete owned caller contract.",
                "Owned system.",
                [],
                pairs,
                pairs[-1:],
                {},
            )
            messages, report = assemble(assembler, frozen)
            self.assertEqual(pairs, original)
            self.assertTrue(report["joint_evidence"]["active"])
            history = [json.loads(m["content"]) for m in messages[2:]]
            short = next(
                row
                for row in history
                if row.get("runtime_evidence") and "source_reference" in row.get("observation", {})
            )
            original_envelope = json.loads(pairs[0][-1]["content"])
            self.assertEqual(
                {k: v for k, v in short.items() if k != "observation"},
                {k: v for k, v in original_envelope.items() if k != "observation"},
            )
            self.assertEqual(short["observation"]["outline"], source["outline"])
            self.assertNotIn("text", short["observation"])
            self.assertNotIn("NEVER_DISCLOSE", json.dumps(messages))
            self.assertEqual(messages[-1], pairs[-1][-1])
            self.assertIsNone(
                assemble(
                    assembler,
                    snapshot(state, workspace, "contract", "system", [], pairs, pairs, {}),
                )
            )
            # Stale source is never called a visible current-version duplicate.
            (root / "alpha.py").write_text("def alpha(x):\n    return x + 2\n")
            changed = snapshot(state, workspace, "contract", "system", [], pairs, pairs[-1:], {})
            messages, report = assemble(assembler, changed)
            self.assertIn("observed_version_invalidated", messages[1]["content"])
            self.assertNotIn("source_reference", messages[3]["content"])
            self.assertIn("return x + 2", messages[1]["content"])


@unittest.skipUnless(physical.METADATA, "requires the two frozen serving tokenizers")
class PressureContextControls(unittest.TestCase):
    setUpClass = classmethod(physical.PhysicalContextControls.setUpClass.__func__)

    def fixture(self, root, client, count, bound, enabled=True, limit=8192):
        ws = root / "workspace"
        ws.mkdir()
        (ws / "alpha.py").write_text(
            "def adjust(x):\n    return x - 1\n"
            + "".join(f"# owned alpha source row {i}\n" for i in range(85))
        )
        (ws / "beta.py").write_text(
            "from alpha import adjust\ndef pipeline(x):\n    return adjust(x) * 3\n"
            + "".join(f"# owned beta source row {i}\n" for i in range(85))
        )
        checker = root / "public.py"
        checker.write_text(
            "import sys\nsys.path.insert(0,sys.argv[1])\nfrom beta import pipeline\n"
            "ok=all(pipeline(x)==(x+1)*2 for x in (-2,0,2,7))\n"
            'print("OWNED_PUBLIC_PASS" if ok else "OWNED_CURRENT_PUBLIC_FAIL")\nraise SystemExit(0 if ok else 1)\n'
        )
        spec = TaskSpec(
            "Repair the owned two-file pipeline using the public check. Preserve both public function interfaces.",
            allowed_paths=["alpha.py", "beta.py"],
            checks=[
                Check(
                    "public_owned",
                    [sys.executable, "-B", str(checker), "{workspace}"],
                    5,
                    protected_paths=[str(checker)],
                )
            ],
            milestones=[
                dict(
                    goal="Complete the cross-file implementation and its public check.",
                    write_set=["alpha.py", "beta.py"],
                    check_ids=["public_owned"],
                )
            ],
            amendments=[
                dict(
                    source="user",
                    description="Keep signed and zero inputs. Do not alter unrelated comments.",
                )
            ],
        )
        owner = CollaborationBudget(
            CollaborationLimits(12, 90, 5),
            root / "budget.jsonl",
            request_token_bound=bound,
            tokenizer_identity="frozen serving tokenizer",
            request_context_limit=limit,
        )
        owner.allocate({"A": 12})
        client = CompatibleClient(replace(client.config, max_output_tokens=1024))
        role = _AdvisedClient(client, owner, "A", workspace=ws, count_source_tokens=count)
        runtime = CollaborationRuntime(
            ws,
            role,
            root / "state",
            profile="general",
            model_profile=ModelProfile(limit, 1024, 256),
            max_model_calls=12,
            max_steps=12,
            time_limit=85,
            request_timeout=8,
            code_memory=False,
            development_workflows=False,
            domain_compiler=False,
            grounded_planning=False,
            automatic_workers=False,
            allow_commands=True,
            auto_verify_edits=False,
        )
        runtime.physical_context_budget = True
        runtime.assembler.physical_alignment = True
        runtime.assembler.estimator = TokenEstimator(count)
        runtime._route = route_task(spec, ws, "general")
        runtime._graph = compile_task(spec)
        runtime._worker_response_format = "json_source"
        runtime._worker_review_pending = None
        if enabled:
            configure_pressure_evidence(runtime)
        state = dict(
            steps=0,
            model_calls=0,
            milestone=0,
            notes=[],
            recent=[],
            changed_files=[],
            last_failure=None,
            progress_hint="",
            index_revision="owned",
        )
        workspace = Workspace(ws, spec.allowed_paths)
        return runtime, role, owner, spec, state, workspace

    def populate(self, runtime, state, workspace):
        class DB:
            def execute(self, *args):
                return SimpleNamespace(fetchone=lambda: None)

        index = SimpleNamespace(db=DB(), pack=lambda *a, **k: None)
        actions = []

        def add(action, result):
            step = len(actions) + 1
            actions.append(record(action, result, step))
            state.update(steps=step, model_calls=step)
            observe(state, action, result, index, workspace)

        add(
            dict(tool="read", arguments=dict(path="alpha.py", count=200)),
            workspace.read("alpha.py", 1, 200),
        )
        for _ in range(3):
            add(
                dict(tool="command", arguments=dict(argv=["owned trace"])),
                dict(exit_code=0, stdout_tail="unrelated_owned_trace " * 400),
            )
        add(
            dict(tool="read", arguments=dict(path="beta.py", count=200)),
            workspace.read("beta.py", 1, 200),
        )
        before = hashlib.sha256(workspace.path("alpha.py").read_bytes()).hexdigest()
        result = workspace.edit("alpha.py", before, old="return x - 1", new="return x + 1")
        add(
            dict(
                tool="edit",
                arguments=dict(
                    path="alpha.py", expected_hash=before, old="return x - 1", new="return x + 1"
                ),
            ),
            result,
        )
        add(
            dict(tool="verify", arguments={}),
            dict(accepted=False, failures=[dict(error="OWNED_CURRENT_PUBLIC_FAIL")]),
        )
        state.update(
            shell_loop_history=actions,
            changed_files=["alpha.py"],
            read_paths=["alpha.py", "beta.py"],
            last_failure=dict(error="OWNED_CURRENT_PUBLIC_FAIL"),
        )
        return index

    def test_no_pressure_is_identical_and_pressure_keeps_sources_and_feedback(self):
        for label, (count, bound) in self.counters.items():
            with (
                self.subTest(model=label),
                tempfile.TemporaryDirectory() as td,
                service(lambda body: ("{}", USAGE)) as (client, requests),
            ):
                runtime, role, owner, spec, state, workspace = self.fixture(
                    Path(td), client, count, bound
                )
                try:
                    index = SimpleNamespace(pack=lambda *a, **k: None)
                    runtime.pressure_evidence = False
                    off = runtime._context(spec, state, index)
                    configure_pressure_evidence(runtime)
                    on = runtime._context(spec, state, index)
                    self.assertEqual(off, on)
                    index = self.populate(runtime, state, workspace)
                    saved = deepcopy(state["shell_loop_history"])
                    runtime.pressure_evidence = False
                    off = runtime._context(spec, state, index)
                    configure_pressure_evidence(runtime)
                    on = runtime._context(spec, state, index)
                    self.assertIn("joint_evidence", state["budget_report"])
                    self.assertNotIn("CURRENT COMPLETE OBSERVED SOURCE REGIONS", json.dumps(off))
                    self.assertIn("return x + 1", on[1]["content"])
                    self.assertIn("return adjust(x) * 3", on[1]["content"])
                    self.assertEqual(state["shell_loop_history"], saved)
                    self.assertIn("after", json.dumps(on))
                    self.assertIn("OWNED_CURRENT_PUBLIC_FAIL", json.dumps(on))
                    schema = runtime._response_schema("actions", spec, state, workspace)
                    prepared, report = role.prepare_request(
                        on, schema, runtime.assembler, state["budget_report"]
                    )
                    self.assertLessEqual(report["physical_request"]["prepared_bound"], 8192)
                    stats(
                        dict(
                            kind="8K_context",
                            model=label,
                            physical=report["physical_request"],
                            joint=report["joint_evidence"],
                            no_pressure_messages_identical=True,
                        )
                    )
                    role.generate_prepared(prepared, 5)
                    self.assertEqual(len(requests), 1)
                    self.assertEqual(requests[0]["messages"], prepared.physical_messages)
                finally:
                    owner.close()

    def test_reassembly_is_pure_once_and_4k_protected_overflow_never_dispatches(self):
        for label, (count, bound) in self.counters.items():
            with (
                self.subTest(model=label),
                tempfile.TemporaryDirectory() as td,
                service(lambda body: ("{}", USAGE)) as (client, requests),
            ):
                runtime, role, owner, spec, state, workspace = self.fixture(
                    Path(td), client, count, bound
                )
                try:
                    index = self.populate(runtime, state, workspace)
                    messages = runtime._context(spec, state, index)
                    original = runtime.assembler.request_reassembly[1]
                    calls = []

                    def once():
                        calls.append(1)
                        return original()

                    runtime.assembler.bind_reassembly(messages, once)
                    # One local envelope recount,using frozen records only.
                    with patch(
                        "infinicoder.memory.pressure_context._read",
                        side_effect=AssertionError("source reread"),
                    ):
                        rebuilt, report = runtime.assembler.reassemble_request(messages, 512)
                    self.assertEqual(calls, [1])
                    self.assertEqual(
                        report["joint_evidence"]["source_rereads_during_reassembly"], 0
                    )
                    self.assertIn("OWNED_CURRENT_PUBLIC_FAIL", json.dumps(rebuilt))
                    self.assertEqual(requests, [])
                    runtime.assembler.profile = ModelProfile(4096, 1024, 256)
                    state["last_failure"] = dict(error="owned required failure " * 1800)
                    # Preserve the complete current feedback pair as well as its display packet.
                    state["shell_loop_history"][-1]["observation"]["failures"][0]["error"] = (
                        "owned required failure " * 1800
                    )
                    with self.assertRaises(BudgetExceeded):
                        runtime._context(spec, state, index)
                    self.assertEqual(owner.snapshot()["physical_request_attempts"], 0)
                    self.assertEqual(requests, [])
                finally:
                    owner.close()

    def test_full_runtime_read_edit_verify_chain(self):
        if os.name != "posix":
            self.skipTest("full checkpoint binding requires POSIX")
        from infinicoder.search.candidate_checkpoint import CandidateCheckpointStore
        from infinicoder.collaboration.collaboration import BranchContext

        for label, (count, bound) in self.counters.items():
            with self.subTest(model=label), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                position = []
                runtime = None
                workspace = None
                seen_pressure = []
                missing_failure = []

                def answer(body):
                    i = len(position)
                    position.append(i)
                    if i == 0:
                        action = dict(tool="read", arguments=dict(path="alpha.py", count=200))
                    elif i in (1, 2, 3):
                        action = dict(
                            tool="command",
                            arguments=dict(
                                argv=[sys.executable, "-c", 'print("unrelated_owned_trace "*400)']
                            ),
                        )
                    elif i == 4:
                        action = dict(tool="read", arguments=dict(path="beta.py", count=200))
                    elif i in (5, 7):
                        if i == 7:
                            text = json.dumps(body["messages"])
                            self.assertIn("return x + 1", text)
                            self.assertIn("return adjust(x) * 3", text)
                            missing_failure.append("OWNED_CURRENT_PUBLIC_FAIL" not in text)
                            seen_pressure.append(
                                runtime._active_state["budget_report"]["joint_evidence"]
                            )
                        path = "alpha.py" if i == 5 else "beta.py"
                        old, new = ("return x - 1", "return x + 1") if i == 5 else ("* 3", "* 2")
                        action = dict(
                            tool="edit",
                            arguments=dict(
                                path=path,
                                expected_hash=hashlib.sha256(
                                    workspace.path(path).read_bytes()
                                ).hexdigest(),
                                old=old,
                                new=new,
                            ),
                        )
                    elif i == 6:
                        action = dict(tool="verify", arguments={})
                    else:
                        action = dict(tool="finish", arguments={})
                    return json.dumps(action), USAGE

                with service(answer) as (client, requests):
                    runtime, role, owner, spec, state, workspace = self.fixture(
                        root, client, count, bound
                    )
                    try:
                        store = CandidateCheckpointStore(
                            workspace.root,
                            root / "checkpoints",
                            exclusive_workspace=True,
                            remaining=owner.remaining,
                        )
                        context = BranchContext(
                            "A",
                            workspace.root,
                            runtime.state_dir,
                            spec,
                            role,
                            runtime.assembler.profile,
                            12,
                            False,
                            False,
                            owner,
                            store,
                        )
                        runtime.bind_collaboration(context)
                        result = runtime.execute(spec)
                        trace = [
                            dict(
                                step=r.get("step"),
                                tool=(r.get("action") or {}).get("tool"),
                                error=(r.get("observation") or {}).get("error"),
                                accepted=(r.get("observation") or {}).get("accepted"),
                            )
                            for r in result.get("shell_loop_history", [])
                        ]
                        self.assertEqual(
                            result["status"],
                            "succeeded",
                            json.dumps(dict(reason=result.get("reason"), trace=trace)),
                        )
                        self.assertTrue(seen_pressure)
                        self.assertEqual(len(requests), 9)
                        self.assertEqual(len(seen_pressure[0]["whole_regions"]), 2)
                        self.assertEqual(missing_failure, [False])
                        self.assertEqual(owner.snapshot()["physical_request_attempts"], 9)
                        self.assertTrue(runtime.public_observation.status == "passed")
                        stats(
                            dict(
                                kind="full_Runtime_chain",
                                model=label,
                                scripted_HTTP_requests=len(requests),
                                public_status=runtime.public_observation.status,
                                pressure=seen_pressure,
                                target_model_inference_calls=0,
                            )
                        )
                        # Ordinary read remains the current-source consumer.
                        current = workspace.read("alpha.py", 1, 200)
                        self.assertEqual(
                            current["sha256"],
                            hashlib.sha256(workspace.path("alpha.py").read_bytes()).hexdigest(),
                        )
                        self.assertIn("return x + 1", current["text"])
                    finally:
                        owner.close()


if __name__ == "__main__":
    unittest.main()
