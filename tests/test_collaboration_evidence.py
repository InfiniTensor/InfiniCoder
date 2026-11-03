"""Owned source and transport controls; no target models or benchmark answers."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from infinicoder.models.budget import BudgetExceeded, ContextAssembler, ModelProfile, TokenEstimator
from infinicoder.search.candidate_checkpoint import CheckpointConflict
from infinicoder.collaboration.collaboration import _AdvisedClient, _plan
from infinicoder.collaboration.collaboration_evidence import (
    SourceEvidencePolicy,
    build_source_evidence,
    current_source_evidence,
)
from infinicoder.models.model_client import CompatibleClient, ModelConfig
from infinicoder.planning.task_contract import TaskSpec


def count(text):
    # A synthetic conservative counter for owned controls, not a model tokenizer.
    return len(text.encode("utf-8"))


def manifest(root):
    return dict(
        root_mode=0o755,
        entries={
            p.relative_to(root).as_posix(): dict(
                kind="file",
                bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            )
            for p in root.rglob("*")
            if p.is_file()
        },
    )


class SourcePacketControls(unittest.TestCase):
    def build(self, root, spec, *, guard=lambda: None, policy=None):
        return build_source_evidence(
            spec,
            root / "workspace",
            manifest(root / "workspace"),
            root / "index.sqlite",
            count_text=count,
            guard=guard,
            policy=policy,
        )

    def test_real_large_directory_selects_public_anchor_not_first_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ws = root / "workspace"
            ws.mkdir()
            for i in range(2544):
                file = ws / "vendor" / ("a%04d.py" % i)
                file.parent.mkdir(exist_ok=True)
                file.write_text("def utility():\n    return 0\n")
            wanted = ws / "zruntime" / "ownership.py"
            wanted.parent.mkdir()
            wanted.write_text("def preserve_shared_state(x):\n    return x\n")
            original = TaskSpec(
                "Repair preserve_shared_state in zruntime/ownership.py; retain identity and errors.",
                allowed_paths=["zruntime/"],
                amendments=["Do not alter persistence semantics."],
            )
            before = asdict(original)
            packet = self.build(root, original)
            self.assertEqual(asdict(original), before)
            self.assertEqual(packet["total_files"], 2545)
            self.assertIn("zruntime/ownership.py", [r["path"] for r in packet["records"]])
            self.assertNotIn("vendor/a0000.py", [r["path"] for r in packet["records"]])
            self.assertLessEqual(
                count(json.dumps(packet, ensure_ascii=False, separators=(",", ":"))), 2048
            )
            self.assertLessEqual(len(packet["records"]), 12)
            from infinicoder.execution.workspace_tools import Workspace

            self.assertIn(
                "utility", Workspace(ws, original.allowed_paths).read("vendor/a0000.py")["text"]
            )

    def test_edit_delete_and_rename_withdraw_current_facts(self):
        for mutation in ("edit", "delete", "rename"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                ws = root / "workspace"
                ws.mkdir()
                file = ws / "shared.py"
                file.write_text("def shared(): return 1\n")
                packet = self.build(root, TaskSpec("Repair shared.py."))
                if mutation == "edit":
                    file.write_text("def shared(): return 2\n")
                elif mutation == "delete":
                    file.unlink()
                else:
                    file.rename(ws / "renamed.py")
                current = current_source_evidence(packet, ws, count_text=count)
                self.assertEqual(current["records"], [])
                self.assertEqual(current["withdrawn_records"], 1)
                self.assertEqual(len(packet["records"]), 1)

    def test_explicit_module_name_outranks_incidental_example_identifiers(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ws = root / "workspace"
            ws.mkdir()
            (ws / "package").mkdir()
            (ws / "docs").mkdir()
            (ws / "package/settings.py").write_text(
                "import logging\nlogger=logging.getLogger(__name__)\n"
            )
            for i in range(20):
                (ws / ("docs/example%d.py" % i)).write_text(
                    "jpeg_ls = True\npillow_support = True\n"
                )
            spec = TaskSpec(
                "Do not configure the application logger in the `settings` module. "
                "Duplicate messages mention jpeg_ls and pillow_support."
            )
            packet = self.build(root, spec)
            self.assertEqual(packet["records"][0]["path"], "package/settings.py")
            self.assertEqual(
                packet["records"][0]["selection"], "quoted_module_candidate;binding_unknown"
            )

    def test_race_aborts_and_partial_scan_is_explicit(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ws = root / "workspace"
            ws.mkdir()
            file = ws / "code.py"
            file.write_text("def relevant(): return 1\n")
            base = manifest(ws)
            file.write_text("def relevant(): return 2\n")
            with self.assertRaises(CheckpointConflict):
                build_source_evidence(
                    TaskSpec("Repair code.py"),
                    ws,
                    base,
                    root / "i.sqlite",
                    count_text=count,
                    guard=lambda: None,
                )
            packet = self.build(
                root, TaskSpec("Repair code.py"), policy=SourceEvidencePolicy(scan_bytes=1)
            )
            self.assertEqual(packet["scanned_files"], 0)
            self.assertEqual(packet["omitted_files"], 1)
            self.assertEqual(packet["records"], [])

    def test_same_initial_packet_plain_and_advised_and_no_permission_change(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ws = root / "workspace"
            ws.mkdir()
            (ws / "code.py").write_text("x=1\n")
            (ws / "unmentioned.txt").write_text("still readable\n")
            spec = TaskSpec("Repair code.py", allowed_paths=["code.py"])
            packet = self.build(root, spec)
            client = CompatibleClient(ModelConfig("http://127.0.0.1:1/v1", "owned-never-called"))
            controls = []
            for role, advice in [
                ("A", None),
                ("B", None),
                ("A", dict(strategy="auto_source")),
                ("B", dict(strategy="ordinary_actions")),
            ]:
                wrapped = _AdvisedClient(
                    client,
                    object(),
                    role,
                    advice=advice,
                    source_evidence=packet,
                    workspace=ws,
                    public_context="Caller context remains verbatim.",
                    count_source_tokens=count,
                )
                first = wrapped.prefix_messages()[0]["content"].split("\n", 1)[1]
                controls.append(json.loads(first))
            self.assertTrue(all(c == controls[0] for c in controls))
            self.assertEqual(controls[0]["public_context"], "Caller context remains verbatim.")
            from infinicoder.execution.workspace_tools import Workspace

            self.assertEqual(
                Workspace(ws, spec.allowed_paths).read("unmentioned.txt")["text"], "still readable"
            )
            with self.assertRaisesRegex(ValueError, "edit scope"):
                Workspace(ws, spec.allowed_paths).path("unmentioned.txt", write=True)

    def test_planner_cannot_cite_an_unsupplied_baseline_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ws = root / "workspace"
            ws.mkdir()
            (ws / "code.py").write_text("x=1\n")
            (ws / "unseen.py").write_text("y=2\n")
            spec = TaskSpec("Repair code.py")
            base = manifest(ws)
            packet = self.build(root, spec)
            self.assertNotIn("unseen.py", [r["path"] for r in packet["records"]])
            proposals = [
                dict(
                    strategy=strategy,
                    approach=approach,
                    assumptions=[],
                    interfaces=[],
                    write_paths=["code.py"],
                    source_refs=[
                        dict(path="unseen.py", sha256=base["entries"]["unseen.py"]["sha256"])
                    ],
                    verification=[],
                )
                for strategy, approach in [
                    ("auto_source", "Whole source"),
                    ("ordinary_actions", "Minimal edit"),
                ]
            ]
            fake = SimpleNamespace(
                owner=SimpleNamespace(remaining=lambda: 5),
                generate_structured=lambda *a, **k: SimpleNamespace(
                    text=json.dumps(dict(proposals=proposals))
                ),
            )
            with self.assertRaisesRegex(ValueError, "not supplied"):
                _plan(fake, spec, base, "", packet)

    def test_fixed_physical_prefix_is_reserved_without_task_truncation(self):
        assembler = ContextAssembler(TokenEstimator(count), ModelProfile(1024, 128, 64))
        assembler.extra_fixed_context = "s" * 600
        with self.assertRaises(BudgetExceeded):
            assembler.assemble("caller requirement " * 30, {}, fixed_overhead_text="policy")
        text, _ = assembler.assemble(
            "Caller requirement remains unchanged.", {}, fixed_overhead_text="policy"
        )
        self.assertIn("Caller requirement remains unchanged.", text)


if __name__ == "__main__":
    unittest.main()
