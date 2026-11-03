"""Actionable lifecycle observations through the existing Runtime Check path."""

import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

from infinicoder import (
    CheckpointEndpoint,
    CheckpointProcedureContract,
    checkpoint_equivalence,
    checkpoint_check_report,
    CheckpointRunEndpoint,
    checkpoint_run_equivalence,
    Runtime,
    TaskSpec,
    Check,
    CompatibleClient,
    ModelConfig,
)
from infinicoder.runtime.feedback import failure_packet
from test_implementation_window import owned_http, action


class ScalarValues:
    @staticmethod
    def is_tensor(value):
        return False


PIN = {"owner.py": "a" * 64}
CASES = [
    dict(
        id="public",
        prefix=[dict(args=[1.0], kwargs={}), dict(args=[3.0], kwargs={})],
        suffix=[dict(args=[2.0], kwargs={}), dict(args=[-1.0], kwargs={})],
    )
]


def factory(
    *,
    omit=False,
    alias=False,
    wrong_source=False,
    wrong_count=False,
    save_transform=None,
    versions=None,
):
    instances = []

    def create(case):
        state = dict(model=1.0, optimizer={0: dict(momentum=0.0)}, progress=0)
        calls = dict(step=0, save=0, load=0)
        label = "owned-" + str(len(instances))
        instances.append(label)

        def step(value):
            if not wrong_count:
                calls["step"] += 1
            state["optimizer"][0]["momentum"] = state["optimizer"][0]["momentum"] * 0.5 + value
            state["model"] -= 0.1 * state["optimizer"][0]["momentum"]
            state["progress"] += 1
            return state["model"]

        def save():
            calls["save"] += 1
            if save_transform is not None:
                return save_transform(state)
            value = state if alias else copy.deepcopy(state)
            if omit:
                value.pop("optimizer")
            return value

        def load(value):
            calls["load"] += 1
            state.update(copy.deepcopy(value))

        def observe():
            return state

        def binding():
            return dict(
                instance_id=label,
                source_versions={"owner.py": "b" * 64} if wrong_source else (versions or PIN),
                calls=calls,
            )

        return CheckpointEndpoint(step, save, load, observe, binding)

    return create


class CheckpointProcedure(unittest.TestCase):
    def contract(self):
        return CheckpointProcedureContract(
            PIN, ("model", "optimizer", "progress"), "Owned public state lifecycle", atol=1e-12
        )

    def test_missing_optimizer_is_localized_and_reaches_original_failure_feedback(self):
        report = checkpoint_equivalence(self.contract(), CASES, factory(omit=True), ScalarValues)
        self.assertEqual(report["status"], "counterexample")
        self.assertEqual(report["stage"], "restore")
        self.assertEqual(report["component"], "optimizer")
        self.assertIn("momentum", report["difference"]["path"])
        formatted = checkpoint_check_report(report)
        packet = failure_packet([dict(check_id="public", exit_code=1, report=formatted)])
        self.assertEqual(packet[0]["detail"]["component"], "optimizer")
        self.assertEqual(formatted["passed"], 0)
        self.assertFalse(formatted["acceptance_proof"])

    def test_full_restore_matches_every_step_and_stable_payload_is_required(self):
        report = checkpoint_equivalence(self.contract(), CASES, factory(), ScalarValues)
        self.assertEqual(report["status"], "observed_values_match")
        self.assertEqual(report["step_calls"], 6)
        self.assertEqual(checkpoint_check_report(report)["passed"], 1)
        aliased = checkpoint_equivalence(self.contract(), CASES, factory(alias=True), ScalarValues)
        self.assertEqual(aliased["status"], "counterexample")
        self.assertEqual(aliased["stage"], "restore")

    def test_wrong_source_and_uncalled_entry_cannot_certify_equivalence(self):
        for constructor in (factory(wrong_source=True), factory(wrong_count=True)):
            with self.subTest(constructor=constructor):
                report = checkpoint_equivalence(self.contract(), CASES, constructor, ScalarValues)
                self.assertEqual(report["status"], "unknown")
                self.assertEqual(checkpoint_check_report(report)["error"], 1)

    def test_input_call_budget_and_distinct_instance_are_checked(self):
        called = []
        over = copy.deepcopy(CASES)
        over[0]["prefix"] *= 1
        over[0]["prefix"].append(dict(args=[4], kwargs={}))
        over[0]["suffix"].append(dict(args=[5], kwargs={}))
        report = checkpoint_equivalence(
            self.contract(), over, lambda case: called.append(case), ScalarValues
        )
        self.assertEqual(report["status"], "unknown")
        self.assertFalse(called)
        endpoint = factory()(CASES[0])
        reused = checkpoint_equivalence(self.contract(), CASES, lambda _: endpoint, ScalarValues)
        self.assertEqual(reused["status"], "unknown")

    def test_modified_contract_and_missing_components_do_not_pass(self):
        contract = self.contract()
        contract.source_versions["owner.py"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "contract mutated"):
            checkpoint_equivalence(contract, CASES, factory(), ScalarValues)
        creator = factory()

        def missing(case):
            endpoint = creator(case)
            endpoint.observe = lambda: dict(model=1)
            return endpoint

        self.assertEqual(
            checkpoint_equivalence(self.contract(), CASES, missing, ScalarValues)["status"],
            "unknown",
        )

    def test_real_run_boundaries_support_lazy_resume_and_localize_data_order(self):
        for replayed in (False, True):
            counts = dict(continuous=0, interrupted=0, resume=0)

            def continuous():
                counts["continuous"] += 1
                return dict(data_order=[0, 1, 2, 3])

            def interrupted():
                counts["interrupted"] += 1
                return b"owned stable checkpoint"

            def resume(value):
                counts["resume"] += 1
                self.assertEqual(value, b"owned stable checkpoint")
                return dict(data_order=[0, 1, 0, 1] if replayed else [0, 1, 2, 3])

            endpoint = CheckpointRunEndpoint(
                continuous, interrupted, resume, lambda: dict(source_versions=PIN, calls=counts)
            )
            contract = CheckpointProcedureContract(
                PIN, ("data_order",), "Owned deferred framework resume"
            )
            report = checkpoint_run_equivalence(contract, endpoint, ScalarValues)
            self.assertEqual(report["run_calls"], 3)
            self.assertEqual(
                report["status"], "counterexample" if replayed else "observed_values_match"
            )
            if replayed:
                self.assertEqual(report["component"], "data_order")

    def test_large_discrete_cursor_is_exact_despite_float_tolerance(self):
        counts = dict(continuous=0, interrupted=0, resume=0)

        def continuous():
            counts["continuous"] += 1
            return dict(data_position=2**30)

        def interrupted():
            counts["interrupted"] += 1
            return b"checkpoint"

        def resume(saved):
            counts["resume"] += 1
            return dict(data_position=2**30 + 1)

        endpoint = CheckpointRunEndpoint(
            continuous, interrupted, resume, lambda: dict(source_versions=PIN, calls=counts)
        )
        contract = CheckpointProcedureContract(
            PIN, ("data_position",), "Owned large integer cursor", atol=1e-3, rtol=1e-3
        )
        report = checkpoint_run_equivalence(contract, endpoint, ScalarValues)
        self.assertEqual(report["status"], "counterexample")
        self.assertEqual(report["difference"]["current"] - report["difference"]["reference"], 1)

    def test_actual_http_check_counterexample_edit_and_original_final(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            target = workspace / "owner.py"
            source = 'import copy\ndef save(state):\n    result=copy.deepcopy(state)\n    result.pop("optimizer",None)\n    return result\n'
            target.write_bytes(source.encode())
            checker = root / "public.py"
            # This is an owned control, not a repaired real issue or grader.
            checker.write_text("""import copy,hashlib,json,runpy,sys
from pathlib import Path
from infinicoder import CheckpointProcedureContract,checkpoint_equivalence,checkpoint_check_report
from test_checkpoint_procedure import CASES,PIN,ScalarValues,factory
root=Path(sys.argv[1]);path=root/'owner.py';module=runpy.run_path(str(path))
assert Path(module['save'].__code__.co_filename).resolve()==path.resolve()
versions={'owner.py':hashlib.sha256(path.read_bytes()).hexdigest()}
report=checkpoint_equivalence(CheckpointProcedureContract(versions,('model','optimizer','progress'),'Owned public state lifecycle',atol=1e-12),CASES,factory(save_transform=module['save'],versions=versions),ScalarValues)
result=checkpoint_check_report(report)
print(json.dumps(result))
sys.exit(0 if result['status']=='passed' else 1)
""")
            # Make the owned adapter importable to its isolated subprocess.
            import os

            prior = os.environ.get("PYTHONPATH")
            parts = [str(Path(__file__).parent)]
            if prior:
                parts.append(prior)
            os.environ["PYTHONPATH"] = os.pathsep.join(parts)
            self.addCleanup(
                lambda: (
                    os.environ.pop("PYTHONPATH", None)
                    if prior is None
                    else os.environ.__setitem__("PYTHONPATH", prior)
                )
            )
            spec = TaskSpec(
                "Repair checkpoint save/load so optimizer state and the resumed next step agree with uninterrupted execution.",
                allowed_paths=["owner.py"],
                checks=[
                    Check(
                        "checkpoint",
                        [sys.executable, str(checker), "{workspace}"],
                        protected_paths=[str(checker)],
                    )
                ],
                target_facts={
                    "checkpoint_equivalence": dict(
                        check_id="checkpoint", components=["model", "optimizer", "progress"]
                    )
                },
            )

            def respond(n, payload):
                visible = "\n".join(message["content"] for message in payload["messages"])
                self.assertIn(spec.task, visible)
                if n == 1:
                    self.assertIn("checkpoint_equivalence_v1", visible)
                    return action(
                        "plan",
                        units=[
                            dict(
                                goal=spec.task,
                                write_set=["owner.py"],
                                check_ids=["checkpoint"],
                                interfaces=[],
                                dependencies=[],
                            )
                        ],
                    )
                if n == 2:
                    return action("verify")
                if n == 3:
                    self.assertIn("optimizer", visible)
                    self.assertIn("counterexample", visible)
                    return action(
                        "edit",
                        path="owner.py",
                        expected_hash=hashlib.sha256(source.encode()).hexdigest(),
                        old='result.pop("optimizer",None)',
                        new="pass",
                    )
                return action("finish")

            with owned_http(respond) as (url, requests, errors):
                runtime = Runtime(
                    workspace,
                    CompatibleClient(
                        ModelConfig(
                            url,
                            "owned-http",
                            max_output_tokens=512,
                            temperature=0,
                            structured_outputs=True,
                        )
                    ),
                    root / "session",
                    profile="framework",
                    domain_compiler=False,
                    automatic_workers=False,
                    source_fallback=False,
                    code_memory=True,
                    development_workflows=True,
                    max_model_calls=5,
                    max_steps=6,
                    time_limit=30,
                )
                state = runtime.execute(spec)
            action_errors = [
                event.get("data", {}).get("error")
                for event in runtime.store.events()
                if event["kind"] == "action_result" and event.get("data", {}).get("error")
            ]
            self.assertFalse(errors, errors)
            self.assertEqual(state["status"], "succeeded", action_errors)
            self.assertEqual(len(requests), 4)
            self.assertIn("    pass\n", target.read_text())


if __name__ == "__main__":
    unittest.main()
