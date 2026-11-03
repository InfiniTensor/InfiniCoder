"""Complete source units and scoped flow through the original Runtime consumer."""

import hashlib, json, sys, tempfile, unittest
from pathlib import Path

from infinicoder import Runtime, CodeIndex, TaskSpec, Check
from infinicoder.models.budget import TokenEstimator, ModelProfile
from infinicoder.models.model_client import CompatibleClient, ModelConfig
from test_implementation_window import owned_http, count, bound, action


def rows(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class CompleteSourceMemory(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.ws = self.root / "workspace"
        self.ws.mkdir()
        self.source = (
            "def identity(fn):\n    return fn\n\n@identity\ndef target(value):\n"
            '    """The public contract includes this documentation."""\n'
            + "".join("    # implementation step " + str(i) + "\n" for i in range(45))
            + "    # TAIL_SENTINEL_REQUIRES_COMPLETE_UNIT\n    return value - 99\n"
        )
        (self.ws / "core.py").write_bytes(self.source.encode("utf-8"))

    def index(self):
        index = CodeIndex(self.ws, self.root / "index.sqlite")
        self.addCleanup(index.close)
        index.refresh()
        return index

    def test_complete_definition_includes_decorator_docstring_and_tail(self):
        index = self.index()
        text, report = index.pack_tokens(
            "core.target", 12000, TokenEstimator(len), focus_paths=["core.py"]
        )
        row = next(r for r in rows(text) if r.get("symbol") == "core.py::target" and "excerpt" in r)
        self.assertTrue(row["source_unit_complete"])
        self.assertTrue(row["implementation_excerpt_complete"])
        self.assertTrue(row["excerpt"].startswith("@identity\ndef target"))
        self.assertIn("public contract includes", row["excerpt"])
        self.assertIn("TAIL_SENTINEL", row["excerpt"])
        self.assertEqual(row["source_sha256"], hashlib.sha256(self.source.encode()).hexdigest())
        self.assertLessEqual(len(text), 12000)
        self.assertEqual(report["tokens"], len(text))
        self.assertIn("unknown", row["dependency_closure"])

    def test_long_definition_falls_back_with_explicit_versioned_read_address(self):
        (self.ws / "core.py").write_text(
            self.source.replace("# implementation step ", "# " + ("longer " * 40)), encoding="utf-8"
        )
        index = self.index()
        text, _ = index.pack_tokens("core.target", 3500, TokenEstimator(len))
        row = next(r for r in rows(text) if r.get("symbol") == "core.py::target")
        self.assertFalse(row["source_unit_complete"])
        self.assertIn("source_retrieval", row)
        address = row["source_retrieval"]
        self.assertEqual(address["arguments"]["start"], 4)
        self.assertEqual(
            address["source_sha256"], hashlib.sha256((self.ws / "core.py").read_bytes()).hexdigest()
        )
        self.assertLessEqual(len(text), 3500)

    def test_cross_file_dependency_is_expanded_and_never_served_after_source_drift(self):
        (self.ws / "api.py").write_text(
            "from core import target\ndef transform(value):\n    return target(value)\n"
        )
        index = self.index()
        text, _ = index.pack_tokens("api.transform", 18000, TokenEstimator(len))
        original = next(
            r
            for r in rows(text)
            if r.get("symbol") == "core.py::target" and r.get("source_unit_complete")
        )
        file = self.ws / "core.py"
        file.write_text(self.source.replace("value - 99", "value + 7"))
        stale, _ = index.pack_tokens("api.transform", 18000, TokenEstimator(len))
        self.assertFalse(any(r.get("symbol") == "core.py::target" for r in rows(stale)))
        index.refresh()
        fresh, _ = index.pack_tokens("api.transform", 18000, TokenEstimator(len))
        updated = next(
            r
            for r in rows(fresh)
            if r.get("symbol") == "core.py::target" and r.get("source_unit_complete")
        )
        self.assertNotEqual(original["source_sha256"], updated["source_sha256"])
        self.assertIn("value + 7", updated["excerpt"])
        file.rename(self.ws / "renamed.py")
        index.refresh()
        deleted, _ = index.pack_tokens("api.transform", 18000, TokenEstimator(len))
        self.assertFalse(any(r.get("symbol") == "core.py::target" for r in rows(deleted)))

    def test_nested_function_dataflow_is_not_attributed_to_outer_function(self):
        (self.ws / "flow.py").write_text(
            "def outer(flag):\n    def inner(child):\n        return child + 1\n    return flag\n"
        )
        index = self.index()
        text, _ = index.pack_tokens(
            "flow.outer", 14000, TokenEstimator(len), focus_paths=["flow.py"]
        )
        flow = next(r for r in rows(text) if r.get("use_def_of") == "flow.py::outer")
        self.assertTrue(flow["edges"])
        self.assertTrue(all(e["scope"] == "<module>.outer" for e in flow["edges"]))
        self.assertFalse(any(e["name"] == "child" for e in flow["edges"]))

    def test_duplicate_definitions_do_not_claim_one_complete_runtime_binding(self):
        (self.ws / "ambiguous.py").write_text("def pick():return 1\ndef pick():return 2\n")
        index = self.index()
        text, _ = index.pack_tokens(
            "pick", 12000, TokenEstimator(len), focus_paths=["ambiguous.py"]
        )
        self.assertFalse(
            any(
                r.get("symbol") == "ambiguous.py::pick" and r.get("source_unit_complete")
                for r in rows(text)
            )
        )

    def test_tiny_budget_and_already_present_source_do_not_duplicate_or_overflow(self):
        index = self.index()
        text, report = index.pack_tokens("core.target", 0, TokenEstimator(len))
        self.assertEqual(text, "")
        self.assertEqual(report["tokens"], 0)
        text, _ = index.pack_tokens(
            "core.target", 12000, TokenEstimator(len), exclude_paths=["core.py"]
        )
        self.assertFalse(any(r.get("path") == "core.py" and "excerpt" in r for r in rows(text)))

    def test_actual_http_source_tail_reaches_original_edit_public_check_and_final(self):
        checker = self.root / "public.py"
        checker.write_text(
            'import runpy,sys\nf=runpy.run_path(sys.argv[1]+"/core.py")["target"]\nassert [f(x) for x in (-2,0,9)]==[5,7,16]\n'
        )
        spec = TaskSpec(
            "Repair core.target(value) in core.py to return value+7. Preserve its public decorator and documentation.",
            allowed_paths=["core.py"],
            checks=[
                Check(
                    "public",
                    [sys.executable, str(checker), "{workspace}"],
                    protected_paths=[str(checker)],
                )
            ],
        )

        def respond(n, payload):
            visible = "\n".join(m["content"] for m in payload["messages"])
            if n == 1:
                self.assertIn("TAIL_SENTINEL_REQUIRES_COMPLETE_UNIT", visible)
                self.assertIn("source_unit_complete", visible)
                self.assertIn(spec.task, visible)
                expected = hashlib.sha256(self.source.encode()).hexdigest()
                self.assertIn(expected, visible)
                return action(
                    "edit",
                    path="core.py",
                    expected_hash=expected,
                    old="return value - 99",
                    new="return value + 7",
                )
            return action("verify" if n == 2 else "finish")

        with owned_http(respond) as (url, requests, errors):
            runtime = Runtime(
                self.ws,
                CompatibleClient(
                    ModelConfig(
                        url,
                        "owned-http",
                        max_output_tokens=512,
                        temperature=0,
                        structured_outputs=True,
                    )
                ),
                self.root / "session",
                profile="general",
                domain_compiler=False,
                automatic_workers=False,
                source_fallback=False,
                code_memory=True,
                development_workflows=False,
                estimator=TokenEstimator(count, bound),
                model_profile=ModelProfile(16384, 512, 256),
                max_model_calls=4,
                max_steps=5,
                time_limit=30,
            )
            result = runtime.execute(spec)
        self.assertFalse(errors, errors)
        self.assertEqual(result["status"], "succeeded", result.get("last_failure"))
        self.assertEqual(len(requests), 3)
        self.assertEqual(result["model_calls"], 3)
        self.assertIn("return value + 7", (self.ws / "core.py").read_text())


if __name__ == "__main__":
    unittest.main()
