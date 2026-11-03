"""Caller-owned acceptance for the example task."""

import unittest

from solution import stable_unique


class StableUniqueTests(unittest.TestCase):
    def test_preserves_order_and_input(self):
        items = [3, 1, 3, 2, 1]
        self.assertEqual(stable_unique(items), [3, 1, 2])
        self.assertEqual(items, [3, 1, 3, 2, 1])

    def test_empty_and_generator_inputs(self):
        self.assertEqual(stable_unique([]), [])
        self.assertEqual(stable_unique(iter(["a", "b", "a"])), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
