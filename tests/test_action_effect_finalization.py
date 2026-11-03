"""Display-record cleanup preserves original errors;no SDK or candidate execution."""

from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from infinicoder.runtime.native_run import CoreDeadline
from infinicoder.memory.native_action_effect_memory import NativeActionEffectMemory
from infinicoder.runtime.native_plain import NativePlainObserver


class FinalizationContracts(unittest.TestCase):
    def exercise(self, *, failure=None, invalid_info=False, entry_failure=False, delivered=False):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state = root / "state"
            state.mkdir()
            workspace = root / "workspace"
            workspace.mkdir()
            info_path = state / "native-memory-info.json"
            query = lambda messages: None
            model = SimpleNamespace(query=query)
            request = SimpleNamespace(
                state_dir=state,
                workspace=workspace,
                spec=SimpleNamespace(task="Owned lifecycle fixture"),
                budget=SimpleNamespace(wall_seconds=300),
                transport_adapter=SimpleNamespace(request_observations=lambda: []),
            )

            @contextmanager
            def plain(observer, bound, model, environment, deadline):
                observer.state = state
                observer.workspace = workspace
                observer.deadline = deadline
                info_path.write_text(
                    "{invalid" if invalid_info else json.dumps(dict(retained_messages=2)),
                    encoding="utf8",
                )
                if entry_failure:
                    raise failure
                yield observer

            observer = NativeActionEffectMemory(
                lambda messages, timeout: 1, tokenizer_identity="unit-lifecycle-only"
            )
            with patch.object(NativePlainObserver, "bind", plain):
                if failure is None:
                    with observer.bind(request, model, None, 100):
                        if delivered:
                            observer.report["deliveries"].append(dict(model_response_received=True))
                else:
                    with self.assertRaises(type(failure)) as caught:
                        with observer.bind(request, model, None, 100):
                            if delivered:
                                observer.report["deliveries"].append(
                                    dict(model_response_received=True)
                                )
                            raise failure
                    self.assertIs(caught.exception, failure)
            self.assertIs(model.query, query)
            report = (
                json.loads((state / "native-action-effects.json").read_bytes())
                if (state / "native-action-effects.json").exists()
                else {}
            )
            if invalid_info:
                self.assertEqual(report["info_finalization_error"], "JSONDecodeError")
                self.assertEqual(info_path.read_text(encoding="utf8"), "{invalid")
            else:
                info = json.loads(info_path.read_bytes())
                self.assertEqual(info["retained_messages"], 2)
                self.assertTrue(info["action_effect_memory"])
                self.assertFalse(info["canonical_history_changed"])
                self.assertEqual(info["actual_model_projection"], delivered)
                self.assertEqual(info["auxiliary_model_calls"], 0)
            self.assertFalse(info_path.with_suffix(".next").exists())

    def test_normal_exit_keeps_effect_state(self):
        self.exercise(delivered=True)

    def test_sdk_error_and_base_exception_cancel_keep_original_identity(self):
        for failure in (
            RuntimeError("Owned context failure"),
            CoreDeadline("Owned original deadline"),
        ):
            with self.subTest(error=type(failure).__name__):
                self.exercise(failure=failure, delivered=True)

    def test_invalid_display_record_does_not_mask_primary_error(self):
        self.exercise(failure=CoreDeadline("Owned primary deadline"), invalid_info=True)

    def test_parent_entry_error_still_finalizes_an_existing_display_record(self):
        self.exercise(failure=CoreDeadline("Owned bind entry deadline"), entry_failure=True)


if __name__ == "__main__":
    unittest.main()
