"""G1:logical-query outcomes and budgets,without model/candidate execution."""

import unittest
from types import SimpleNamespace

from infinicoder import CoreBudget, ModelConfig
from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.models.native_delivery_policy import (
    DeliveryPolicy,
    NativeDeliveryExhausted,
    QueryDelivery,
)
from infinicoder.models.native_transport import NativeCompatibleTransport


class Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value


class QueryDeliveryTests(unittest.TestCase):
    def make(self, *, phase=2000, task=5000, retries=1):
        clock = Clock()
        controller = QueryDelivery(
            DeliveryPolicy(max_confirmed_not_dispatched_retries=retries),
            task,
            clock=clock,
            phase_deadline=lambda: phase,
        )
        return clock, controller

    def test_absolute_deadline_is_shared_by_retry_and_backoff(self):
        clock, controller = self.make()
        lease = controller.start()
        controller.begin_attempt(lease, "same-complete-request", 1)
        clock.value = 220
        controller.not_dispatched(lease, "trusted counter failed before forwarding")
        self.assertEqual(lease["deadline_monotonic"], 700)
        clock.value = 699
        controller.begin_attempt(lease, "same-complete-request", 2)
        self.assertEqual(controller.remaining(lease), 1)
        clock.value = 701
        with self.assertRaises(NativeDeliveryExhausted):
            controller.remaining(lease)
        self.assertFalse(controller.accept(lease))

    def test_unknown_dispatch_cannot_be_repeated_or_revived_by_late_response(self):
        _, controller = self.make()
        lease = controller.start()
        controller.begin_attempt(lease, "same", 1)
        controller.dispatched(lease)
        controller.unknown(lease, "read timeout after forwarding")
        self.assertFalse(controller.accept(lease))
        with self.assertRaises(NativeDeliveryExhausted):
            controller.begin_attempt(lease, "same", 2)
        usage = dict(prompt_tokens=5, completion_tokens=8, total_tokens=13)
        self.assertFalse(controller.delivered(lease, usage))
        self.assertEqual(lease["outcome"], "unknown")
        self.assertEqual(lease["late_usage"], [usage])
        self.assertFalse(lease["usage_unknown"])

    def test_only_one_confirmed_not_dispatched_retry(self):
        _, controller = self.make()
        lease = controller.start()
        controller.begin_attempt(lease, "same", 1)
        controller.not_dispatched(lease, "counter unavailable")
        self.assertTrue(controller.accept(lease))
        controller.begin_attempt(lease, "same", 2)
        controller.not_dispatched(lease, "counter unavailable again")
        self.assertEqual(lease["outcome"], "not_dispatched_exhausted")
        self.assertFalse(controller.accept(lease))
        self.assertFalse(any(a["dispatched"] for a in lease["attempts"]))

    def test_new_real_query_can_have_identical_messages(self):
        _, controller = self.make()
        first = controller.start()
        controller.begin_attempt(first, "same", 1)
        controller.dispatched(first)
        controller.unknown(first, "timeout")
        controller.finish(first)
        second = controller.start()
        controller.begin_attempt(second, "same", 2)
        self.assertNotEqual(first["query_id"], second["query_id"])
        self.assertTrue(controller.accept(second))
        controller.dispatched(second)
        self.assertTrue(controller.delivered(second, dict(total_tokens=1)))
        controller.finish(second)
        self.assertEqual(controller.snapshot()["queries"][1]["outcome"], "delivered")

    def test_task_and_phase_deadlines_remain_stricter(self):
        clock, controller = self.make(task=500, phase=250)
        lease = controller.start()
        self.assertEqual(lease["deadline_monotonic"], 250)
        clock.value = 249
        self.assertEqual(controller.remaining(lease), 1)
        clock.value = 250
        with self.assertRaises(NativeDeliveryExhausted):
            controller.remaining(lease)

    def test_changed_body_or_parallel_attempt_is_rejected(self):
        _, controller = self.make()
        lease = controller.start()
        controller.begin_attempt(lease, "A", 1)
        with self.assertRaises(NativeDeliveryExhausted):
            controller.begin_attempt(lease, "A", 2)
        controller.not_dispatched(lease, "not sent")
        with self.assertRaises(NativeDeliveryExhausted):
            controller.begin_attempt(lease, "B", 3)
        self.assertEqual(lease["outcome"], "rejected")

    def test_incompatible_wait_layer_rejected_before_generation(self):
        class Counter:
            identity = "interface-only-no-token-count-in-this-test"

            def __call__(self, *args):
                raise AssertionError("must not count or generate")

        transport = NativeCompatibleTransport(
            ModelConfig("http://127.0.0.1:1/v1", "synthetic"),
            Counter(),
            delivery_policy=DeliveryPolicy(),
        )
        with self.assertRaises(UnsupportedCore):
            transport.validate(SimpleNamespace(budget=CoreBudget(request_timeout=120)))
        transport.validate(SimpleNamespace(budget=CoreBudget(request_timeout=600)))

    def test_task_deadline_origin_uses_the_original_core_exception(self):
        class OriginalTaskDeadline(KeyboardInterrupt):
            pass

        def expired():
            raise OriginalTaskDeadline("original core task limit")

        clock, controller = self.make(task=200)
        controller.task_expired = expired
        lease = controller.start()
        self.assertEqual(lease["deadline_source"], "task")
        clock.value = 200
        with self.assertRaises(OriginalTaskDeadline):
            controller.remaining(lease)
        self.assertEqual(lease["outcome"], "task_expired")
        with self.assertRaises(OriginalTaskDeadline):
            controller.start()


if __name__ == "__main__":
    unittest.main()
