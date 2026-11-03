"""Owned HTTP plus serving tokenizers; no target-model inference or gold code."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from infinicoder.models.budget import BudgetExceeded, ModelProfile, TokenEstimator
from infinicoder.collaboration.collaboration import _AdvisedClient, CollaborationRuntime
from infinicoder.collaboration.collaboration_evidence import build_source_evidence
from infinicoder.collaboration.collaboration_budget import (
    CollaborationBudget,
    CollaborationLimits,
    CollaborationStopped,
)
from infinicoder.models.model_client import CompatibleClient, ProviderError
from infinicoder.routing.profiles import route_task
from infinicoder.analysis.task_compiler import compile_task
from infinicoder.planning.task_contract import Check, TaskSpec
from infinicoder.execution.workspace_tools import Workspace
from infinicoder.runtime.work_packet import WorkPacketBuilder
from test_collaboration import service, USAGE
from test_collaboration_evidence import manifest

METADATA = os.environ.get("IC_OWNED_TOKENIZER_METADATA")
SOURCE = "def add(a,b):\n    return a-b\n"
FAILURE = "Owned public observation: add(3,2) returned1;expected5."


@unittest.skipUnless(METADATA, "requires the two frozen serving tokenizers")
class PhysicalContextControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls.counters = {}
        for row in json.loads(Path(METADATA).read_bytes())["rows"]:
            for name, pin in row["tokenizer_files"].items():
                assert (
                    hashlib.sha256(
                        (Path(row["tokenizer_directory"]) / name).read_bytes()
                    ).hexdigest()
                    == pin
                )
            tokenizer = AutoTokenizer.from_pretrained(
                row["tokenizer_directory"], local_files_only=True, trust_remote_code=True
            )

            def count(text, tokenizer=tokenizer):
                return len(tokenizer.encode(text, add_special_tokens=False))

            def bound(messages, schema, output, count=count, tokenizer=tokenizer):
                chat = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                return (
                    count(chat)
                    + output
                    + (count(json.dumps(schema, ensure_ascii=False)) if schema else 0)
                    + 256
                )

            cls.counters[row["label"]] = (count, bound)

    def fixture(self, root, client, count, bound):
        ws = root / "workspace"
        ws.mkdir()
        (ws / "solution.py").write_text(SOURCE)
        spec = TaskSpec(
            "Repair add in solution.py, preserving its interface and unrelated behavior.",
            allowed_paths=["solution.py"],
            checks=[Check("public_owned", ["python", "owned.py", "{workspace}"], 2)],
            amendments=[dict(source="user", description="Retain signed and zero input behavior.")],
        )
        packet = build_source_evidence(
            spec,
            ws,
            manifest(ws),
            root / "source-index.sqlite",
            count_text=count,
            guard=lambda: None,
        )
        owner = CollaborationBudget(
            CollaborationLimits(6, 60, 2),
            root / "budget.jsonl",
            request_token_bound=bound,
            tokenizer_identity="frozen serving tokenizer",
            request_context_limit=16384,
        )
        owner.allocate({"A": 6})
        client = CompatibleClient(replace(client.config, max_output_tokens=4096))
        role = _AdvisedClient(
            client,
            owner,
            "A",
            advice=dict(strategy="auto_source", approach="Owned source repair."),
            public_context="Original caller scope and all final checks remain binding.",
            count_source_tokens=count,
            source_evidence=packet,
            workspace=ws,
        )
        runtime = CollaborationRuntime(
            ws,
            role,
            root / "state",
            profile="general",
            model_profile=ModelProfile(16384, 4096, 256),
            code_memory=False,
            development_workflows=False,
            grounded_planning=True,
            allow_commands=False,
        )
        runtime.physical_context_budget = True
        runtime.assembler.physical_alignment = True
        runtime.assembler.estimator = TokenEstimator(count)
        runtime.assembler.extra_fixed_context = json.dumps(
            role.prefix_messages(), ensure_ascii=False
        )
        runtime._route = route_task(spec, ws, "general")
        runtime._graph = compile_task(spec)
        runtime._worker_response_format = "json_source"
        runtime._worker_review_pending = None
        state = dict(
            steps=2,
            model_calls=1,
            milestone=0,
            notes=[],
            recent=[],
            changed_files=[],
            last_failure=dict(error=FAILURE),
            progress_hint="",
            index_revision="owned",
        )
        state["recent"] = [
            [
                dict(role="assistant", content=json.dumps(dict(tool="verify", arguments={}))),
                dict(role="user", content=json.dumps(dict(accepted=False, error=FAILURE))),
            ]
        ]
        return runtime, role, owner, spec, state, Workspace(ws, spec.allowed_paths)

    def messages(self, runtime, spec, state, workspace, kind, padding):
        if kind == "actions":
            index = SimpleNamespace(pack=lambda *a, **k: padding)
            return runtime._context(spec, state, index)
        unit = runtime._graph.unit(runtime._graph.order[0])
        builder = WorkPacketBuilder(runtime.assembler)
        messages, report = builder.build(
            spec,
            unit,
            state,
            workspace,
            "solution.py",
            SOURCE,
            hashlib.sha256(SOURCE.encode()).hexdigest(),
            [dict(id="owned-old-reference", text=padding)] if padding else [],
            "json_source",
        )
        state["budget_report"] = report
        return messages

    def test_initial_fit_is_identical_to_original_physical_envelope(self):
        for label, (count, bound) in self.counters.items():
            for kind in ("actions", "worker"):
                with (
                    self.subTest(model=label, kind=kind),
                    tempfile.TemporaryDirectory() as td,
                    service(lambda body: ("{}", USAGE)) as (client, requests),
                ):
                    runtime, role, owner, spec, state, workspace = self.fixture(
                        Path(td), client, count, bound
                    )
                    try:
                        messages = self.messages(runtime, spec, state, workspace, kind, "")
                        schema = runtime._response_schema(kind, spec, state, workspace)
                        before = deepcopy(messages)
                        prepared, report = role.prepare_request(
                            messages, schema, runtime.assembler, state["budget_report"]
                        )
                        self.assertFalse(report["physical_request"]["reassembled"])
                        self.assertEqual(prepared.messages, before)
                        role.generate_structured(messages, schema, 5)
                        role.generate_prepared(prepared, 5)
                        self.assertEqual(requests[0], requests[1])
                        self.assertEqual(owner.snapshot()["physical_request_attempts"], 2)
                    finally:
                        owner.close()

    def test_later_schema_overflow_reselects_once_and_preserves_current_evidence(self):
        for label, (count, bound) in self.counters.items():
            for kind in ("actions", "worker"):
                with (
                    self.subTest(model=label, kind=kind),
                    tempfile.TemporaryDirectory() as td,
                    service(lambda body: ("{}", USAGE)) as (client, requests),
                ):
                    runtime, role, owner, spec, state, workspace = self.fixture(
                        Path(td), client, count, bound
                    )
                    try:
                        # Calibrate only the owned optional old material to fill
                        # the old content selector. No benchmark input is used.
                        lo, hi = 0, 16000
                        while lo < hi:
                            mid = (lo + hi + 1) // 2
                            self.messages(
                                runtime, spec, state, workspace, kind, "old_reference " * mid
                            )
                            selected = {row["name"] for row in state["budget_report"]["selected"]}
                            if ("code_pack" if kind == "actions" else "notes") in selected:
                                lo = mid
                            else:
                                hi = mid - 1
                        messages = self.messages(
                            runtime, spec, state, workspace, kind, "old_reference " * lo
                        )
                        schema = runtime._response_schema(kind, spec, state, workspace)
                        physical = deepcopy(messages)
                        physical[1:1] = role.prefix_messages()
                        self.assertGreater(bound(physical, schema, 4096), 16384)
                        original_spec = deepcopy(spec)
                        original_state = deepcopy(state)
                        original_rebuild = runtime.assembler.request_reassembly[1]
                        rebuilds = []

                        def once():
                            rebuilds.append(1)
                            return original_rebuild()

                        runtime.assembler.bind_reassembly(messages, once)
                        prepared, report = role.prepare_request(
                            messages, schema, runtime.assembler, state["budget_report"]
                        )
                        self.assertTrue(report["physical_request"]["reassembled"])
                        self.assertEqual(len(rebuilds), 1)
                        self.assertLessEqual(report["physical_request"]["prepared_bound"], 16384)
                        text = json.dumps(prepared.messages)
                        self.assertIn(spec.task, text)
                        self.assertIn(FAILURE, text)
                        self.assertIn("solution.py", text)
                        self.assertIn("verify", text)
                        self.assertIn("Retain signed and zero input behavior.", text)
                        if kind == "worker":
                            self.assertEqual(
                                json.JSONDecoder().raw_decode(prepared.messages[1]["content"])[0][
                                    "source"
                                ],
                                SOURCE,
                            )
                        self.assertEqual(spec, original_spec)
                        self.assertEqual(state, original_state)
                        self.assertEqual(runtime.assembler.extra_request_tokens, 0)
                        role.generate_prepared(prepared, 5)
                        self.assertEqual(len(requests), 1)
                        self.assertEqual(requests[0]["messages"], prepared.physical_messages)
                        self.assertEqual(
                            requests[0]["response_format"]["json_schema"]["schema"], schema
                        )
                        self.assertEqual(owner.snapshot()["physical_request_attempts"], 1)
                    finally:
                        owner.close()

    def test_mandatory_overflow_cannot_drop_failure_or_dispatch(self):
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
                    assembler = runtime.assembler
                    system = "Owned mandatory policy."
                    mandatory = (
                        "Original task,permission,source and failure. " + "required " * 10500
                    )
                    text, report = assembler.assemble(mandatory, {}, fixed_overhead_text=system)
                    messages = [
                        dict(role="system", content=system),
                        dict(role="user", content=text),
                    ]

                    def rebuild():
                        text, report = assembler.assemble(mandatory, {}, fixed_overhead_text=system)
                        return [
                            dict(role="system", content=system),
                            dict(role="user", content=text),
                        ], report

                    assembler.bind_reassembly(messages, rebuild)
                    schema = runtime._response_schema("actions", spec, state, workspace)
                    with self.assertRaises(BudgetExceeded):
                        role.prepare_request(messages, schema, assembler, report)
                    self.assertEqual(requests, [])
                    self.assertEqual(owner.snapshot()["physical_request_attempts"], 0)
                    self.assertEqual(assembler.extra_request_tokens, 0)
                finally:
                    owner.close()

    def test_prepared_source_drift_cancel_and_expired_count_never_dispatch(self):
        count, bound = self.counters["mini"]
        for mode in (
            "source_drift",
            "unselected_source",
            "schema_drift",
            "cancel",
            "count_deadline",
        ):
            with (
                self.subTest(mode=mode),
                tempfile.TemporaryDirectory() as td,
                service(lambda body: ("{}", USAGE)) as (client, requests),
            ):
                runtime, role, owner, spec, state, workspace = self.fixture(
                    Path(td), client, count, bound
                )
                try:
                    messages = self.messages(runtime, spec, state, workspace, "actions", "")
                    schema = runtime._response_schema("actions", spec, state, workspace)
                    if mode == "count_deadline":

                        def expired(*args):
                            result = bound(*args)
                            owner.coding_deadline = 0
                            return result

                        owner.request_token_bound = expired
                        with self.assertRaises(CollaborationStopped):
                            role.prepare_request(
                                messages, schema, runtime.assembler, state["budget_report"]
                            )
                    else:
                        if mode == "unselected_source":
                            workspace.path("unselected.py").write_text("unchanged=1\n")
                            _, prepared = runtime._prepare_model_request(
                                messages,
                                spec,
                                state,
                                workspace,
                                clarification=False,
                                planning=False,
                                worker=False,
                                command=False,
                                native_plan=False,
                                source_focus=False,
                                diagnostic=False,
                                design=False,
                                problem_worker=False,
                                source_revision=workspace.revision(),
                            )
                            workspace.path("unselected.py").write_text("changed=2\n")
                        else:
                            prepared, _ = role.prepare_request(
                                messages, schema, runtime.assembler, state["budget_report"]
                            )
                        if mode == "source_drift":
                            workspace.path("solution.py").write_text(
                                "def add(a,b):\n    return a+b\n"
                            )
                            with self.assertRaises(ProviderError):
                                role.generate_prepared(prepared, 5)
                        elif mode == "schema_drift":
                            prepared.schema["owned_changed_schema"] = True
                            with self.assertRaises(ProviderError):
                                role.generate_prepared(prepared, 5)
                        elif mode == "unselected_source":
                            with self.assertRaises(ProviderError):
                                role.generate_prepared(prepared, 5)
                        else:
                            owner.stop("owned_cancel")
                            with self.assertRaises(CollaborationStopped):
                                role.generate_prepared(prepared, 5)
                    self.assertEqual(requests, [])
                    self.assertEqual(owner.snapshot()["physical_request_attempts"], 0)
                finally:
                    owner.close()


if __name__ == "__main__":
    unittest.main()
