"""Source-faithfulness controls;token counts here are scripted,not admission."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from infinicoder.analysis.code_context import CodeIndex
from infinicoder.memory.observed_interface_map import ObservedSource, build_interface_map


class ObservedInterfaceMapTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sources = {
            "api.py": "from provider import load\ndef run(path):\n    result = load(path)\n    return consume(result)\n",
            "provider.py": "def load(path):\n    return {'value': path}\n",
        }
        for path, text in self.sources.items():
            (self.root / path).write_bytes(text.encode("utf8"))
        self.index = CodeIndex(self.root, self.root / "index.sqlite")
        self.addCleanup(self.index.close)
        self.index.refresh_paths(
            self.sources,
            read_source=lambda p: (
                (self.root / p).read_bytes(),
                (self.root / p).read_text(encoding="utf8"),
                self.version(p),
            ),
        )

    def version(self, path):
        return hashlib.sha256((self.root / path).read_bytes()).hexdigest()

    def build(self, observed=None, counter=lambda text: 100):
        return build_interface_map(
            self.index,
            observed or [ObservedSource(p, t, "actual-http:1") for p, t in self.sources.items()],
            public_contract_ref="actual-http:1:user",
            namespace=dict(environment="fixture", root=str(self.root)),
            current_version=self.version,
            count_text=counter,
            remaining=lambda: 5,
        )

    def test_cross_file_result_use_and_version_invalidation(self):
        card = self.build()
        self.assertIsNotNone(card)
        relation = card.payload["relations"][0]
        self.assertEqual(relation["use"]["kind"], "result_passed_to_call")
        self.assertEqual(relation["use"]["expression"], "consume")
        self.assertEqual(
            {card.payload["files"][s["file"]]["path"] for s in relation["support"]},
            {"api.py", "provider.py"},
        )
        for support in relation["support"]:
            first, last = support["lines"]
            path = card.payload["files"][support["file"]]["path"]
            self.assertEqual(
                support["text"], "\n".join(self.sources[path].splitlines()[first - 1 : last])
            )
        self.assertTrue(card.current(self.version))
        (self.root / "provider.py").write_bytes(
            (self.sources["provider.py"] + "# changed\n").encode("utf8")
        )
        self.assertFalse(card.current(self.version))
        self.assertIsNone(self.build())
        (self.root / "provider.py").unlink()
        self.assertFalse(card.current(self.version))
        self.assertIsNone(self.build())

    def test_partial_feedback_does_not_certify_complete_definition(self):
        observed = [
            ObservedSource("api.py", self.sources["api.py"], "actual-http:1", ((1, 2),)),
            ObservedSource("provider.py", self.sources["provider.py"], "actual-http:1"),
        ]
        self.assertIsNone(self.build(observed))

    def test_unseen_import_cannot_support_cross_file_binding(self):
        observed = [
            ObservedSource("api.py", self.sources["api.py"], "actual-http:1", ((2, 4),)),
            ObservedSource("provider.py", self.sources["provider.py"], "actual-http:1"),
        ]
        self.assertIsNone(self.build(observed))

    def test_dynamic_receiver_does_not_bind_same_named_function(self):
        text = "from provider import load\ndef run(receiver):\n    result = receiver.load(1)\n    return consume(result)\n"
        (self.root / "api.py").write_bytes(text.encode("utf8"))
        self.sources["api.py"] = text
        self.index.refresh_paths(
            ["api.py"],
            read_source=lambda p: (
                (self.root / p).read_bytes(),
                (self.root / p).read_text(encoding="utf8"),
                self.version(p),
            ),
        )
        self.assertIsNone(self.build())

    def test_over_budget_omits_whole_map(self):
        counts = []
        self.assertIsNone(self.build(counter=lambda text: counts.append(len(text)) or 1000))
        self.assertEqual(len(counts), 2)


if __name__ == "__main__":
    unittest.main()
