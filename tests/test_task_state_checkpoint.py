"""Pure checkpoint contracts;scripted SDK controls separately prove delivery."""

import copy
import unittest

from infinicoder.memory.task_state_checkpoint import (
    accept_memo,
    checkpoint_view,
    prepare_checkpoint,
)


def history():
    messages = [
        dict(role="system", content="Original action protocol"),
        dict(role="user", content="Keep public facade;repair cross-file state."),
    ]
    events = []
    for i in range(6):
        start = len(messages)
        messages.extend(
            [
                dict(role="assistant", content="read source " + str(i)),
                dict(role="user", content="recorded observation " + str(i)),
            ]
        )
        if i == 2:
            messages.append(
                dict(role="user", content="Additional format feedback;retain whole block.")
            )
        events.append(
            dict(
                id=i,
                message_ids=[start, start + 1],
                returncode=0,
                changed=[],
                error=None,
                observer_complete=True,
            )
        )
    return messages, events


class CheckpointContractTests(unittest.TestCase):
    def plan(self, messages, events):
        count = lambda ms: len(str(ms)) // 4
        return prepare_checkpoint(
            messages,
            events,
            "Exact public task",
            (),
            dict(
                count=count,
                input_limit=20000,
                mandatory={0, 1},
                observation_refs={e["id"]: str(e["id"]) * 64 for e in events},
                receipts=[],
                source_snapshot={"impl.py": dict(kind="file", sha256="a" * 64)},
                namespace="b" * 64,
            ),
        )

    def test_fixed_cut_keeps_task_all_future_feedback_and_provenance(self):
        messages, events = history()
        original = copy.deepcopy(messages)
        plan = self.plan(messages, events)
        memo = accept_memo(
            dict(
                content="Use event:2;observation:" + "0" * 64 + ";event:999 is uncertain.",
                extra=dict(response=dict(choices=[dict(finish_reason="stop")])),
            ),
            plan.evidence,
        )
        self.assertIn("event:2", memo.cited_refs)
        self.assertIn("event:999", memo.unresolved_refs)
        new = messages + [
            dict(role="assistant", content="ordinary edit"),
            dict(role="user", content="actual tool feedback"),
            dict(role="user", content="additional error"),
        ]
        view, meta = checkpoint_view(
            plan, memo, new, {"impl.py": dict(kind="file", sha256="c" * 64)}
        )
        self.assertEqual(view[:2], messages[:2])
        self.assertEqual(view[-3:], new[-3:])
        self.assertEqual(messages, original)
        self.assertTrue(meta["changes_after_checkpoint"])
        self.assertTrue(meta["memo_unverified"])
        again, info = checkpoint_view(
            plan, memo, new + [dict(role="user", content="another actual error")], {}
        )
        self.assertEqual(info["selected_ids"][:-1], meta["selected_ids"])
        self.assertEqual(again[-1]["content"], "another actual error")
        changed = copy.deepcopy(new)
        changed[2]["content"] = "rewritten past"
        with self.assertRaises(ValueError):
            checkpoint_view(plan, memo, changed, {})

    def test_unfinished_or_empty_memo_is_never_partially_accepted(self):
        messages, events = history()
        plan = self.plan(messages, events)
        for text, finish in (("partial", "length"), ("", "stop")):
            with self.assertRaises(ValueError):
                accept_memo(
                    dict(
                        content=text,
                        extra=dict(response=dict(choices=[dict(finish_reason=finish)])),
                    ),
                    plan.evidence,
                )
        value = plan.evidence.records
        value.clear()
        self.assertTrue(plan.evidence.records)


if __name__ == "__main__":
    unittest.main()
