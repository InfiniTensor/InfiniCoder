"""Effect truth,repeat identity and the real observer-finally contract."""

import copy
import hashlib
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import time
import unittest

from infinicoder.memory.action_effect_memory import (
    EffectSequence,
    VersionLabels,
    make_effect_record,
    relation,
    render_effect,
)
from infinicoder.memory.native_episode_memory import NativeEpisodeMemory
from infinicoder.runtime.native_run import CoreDeadline


NS = dict(environment="owned-unit-workspace", root="/workspace", binding_identity="unit-only")
A = dict(kind="file", sha256="a" * 64)
B = dict(kind="file", sha256="b" * 64)


def record(number, before=A, after=A, *, feedback="", command="ordinary command", exact=True):
    event = dict(
        id=number,
        message_ids=[2 + number * 2, 3 + number * 2],
        command=command,
        observer_complete=True,
        error=None,
        returncode=0,
    )
    evidence = dict(
        complete=True,
        raw_sha256=hashlib.sha256(feedback.encode()).hexdigest(),
        message_sha256="c" * 64,
    )
    return make_effect_record(
        event,
        NS,
        ("state.py",),
        {"state.py": before},
        {"state.py": after},
        evidence,
        linked_exactly=exact,
    )


class EffectContracts(unittest.TestCase):
    def test_public_api_rejects_stacked_mechanisms_before_loading_a_model(self):
        from infinicoder.app.native_api import create_cpu_request
        from infinicoder.planning.task_contract import TaskSpec

        options = dict(
            model_config=None,
            tokenizer_path="unused",
            check_script="unused",
            profile="cpu-plain",
            implementation_paths=("state.py",),
            action_effect_memory=True,
        )
        for flag in (
            "revision_task_checkpoint",
            "task_state_consolidation",
            "behavior_plan",
            "contract_review",
            "source_question_actions",
            "observed_interface_map",
        ):
            with (
                self.subTest(flag=flag),
                self.assertRaisesRegex(ValueError, "action_effect_memory"),
            ):
                create_cpu_request(
                    TaskSpec("Keep interfaces."),
                    "unused",
                    "unused",
                    **dict(options, **{flag: True}),
                )

    def test_packaged_cli_exposes_the_explicit_default_off_switch(self):
        from infinicoder.app.native_cli import main

        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as stopped:
            main(["--help"])
        self.assertEqual(stopped.exception.code, 0)
        self.assertIn("--action-effect-memory", output.getvalue())

    def test_empty_successful_stdout_is_not_an_edit_or_missing_source_proof(self):
        self.assertEqual(record(0).payload["relation"], "unchanged")
        self.assertEqual(record(1, A, B).payload["relation"], "changed")
        self.assertEqual(record(2, None, B).payload["relation"], "unknown")
        self.assertEqual(record(3, A, A, exact=False).payload["relation"], "unknown")
        self.assertEqual(relation(dict(kind="missing"), B), "changed")
        self.assertEqual(relation(dict(kind="missing"), dict(kind="missing")), "unknown")

    def test_adjacent_repeat_requires_same_command_feedback_versions_and_link(self):
        sequence = EffectSequence()
        self.assertEqual(sequence.add(record(0, A, B)), 0)
        self.assertEqual(sequence.add(record(1, B, B)), 1)
        self.assertEqual(sequence.add(record(2, B, B)), 2)
        self.assertEqual(sequence.add(record(3, B, B, feedback="new observation")), 1)
        self.assertEqual(sequence.add(record(4, B, B, exact=False)), 0)
        self.assertEqual(sequence.add(record(5, B, B)), 1)
        with self.assertRaises(ValueError):
            sequence.add(record(5, B, B))

    def test_view_has_resolvable_full_versions_and_no_correctness_or_write_claim(self):
        labels = VersionLabels(NS)
        effect = record(0, A, B)
        text, ref, payload = render_effect(
            effect, 0, {"state.py": B}, {"state.py": dict(status="not_observed")}, labels
        )
        self.assertIn("changed v1->v2", text)
        self.assertIn("current=v2", text)
        self.assertEqual(payload["version_labels"]["v1"]["state"], A)
        self.assertEqual(payload["version_labels"]["v2"]["state"], B)
        self.assertIn(ref, text)
        self.assertNotIn("PASSED", text)
        changed_namespace = VersionLabels(dict(NS, binding_identity="other"))
        with self.assertRaises(ValueError):
            render_effect(effect, 0, {"state.py": B}, {}, changed_namespace)

    def test_completion_hook_sees_after_and_cleanup_cannot_mask_original_deadline(self):
        class Observer(NativeEpisodeMemory):
            def __init__(self):
                super().__init__(
                    lambda messages, timeout: 1,
                    tokenizer_identity="logic-only",
                    implementation_paths=("state.py",),
                )
                self.cancel = False
                self.hooks = []

            def _snapshot(self):
                if self.cancel:
                    raise TimeoutError("cleanup observation expired")
                return super()._snapshot()

            def execution_returned(self, event, result):
                self.hooks.append(("returned", event["after"]))

            def execution_completed(self, event):
                self.hooks.append(("completed", copy.deepcopy(event["after"])))

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            workspace = root / "workspace"
            workspace.mkdir()
            state = root / "state"
            state.mkdir()
            file = workspace / "state.py"
            file.write_text("before", encoding="utf8")
            observer = Observer()

            def execute(action):
                if action["command"] == "cancel":
                    observer.cancel = True
                    raise CoreDeadline("original task cancellation")
                file.write_text("after", encoding="utf8")
                return dict(returncode=0, output="")

            model = SimpleNamespace(
                query=lambda messages: None, _prepare_messages_for_api=lambda messages: messages
            )
            environment = SimpleNamespace(execute=execute)
            request = SimpleNamespace(
                workspace=workspace,
                state_dir=state,
                model_profile=SimpleNamespace(input_budget=lambda: 4096),
            )
            with observer.bind(request, model, environment, time.monotonic() + 30):
                environment.execute(dict(command="write"))
                self.assertEqual(observer.hooks[0], ("returned", None))
                self.assertEqual(
                    observer.hooks[1][1]["state.py"]["sha256"], hashlib.sha256(b"after").hexdigest()
                )
                with self.assertRaisesRegex(CoreDeadline, "original task cancellation"):
                    environment.execute(dict(command="cancel"))


if __name__ == "__main__":
    unittest.main()
