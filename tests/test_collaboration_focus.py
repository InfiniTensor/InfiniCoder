"""One owned C++ vertical group for actual focus, scope, compile and selection."""

from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import unittest

from infinicoder import (
    CollaborationLimits,
    CollaborationPolicy,
    create_focus_collaboration_runtime,
    run_collaboration,
)
from infinicoder.runtime.agent_execution import CoreBudget, CoreRequest, FinalAcceptance
from infinicoder.models.budget import ModelProfile
from infinicoder.analysis.local_edit import select
from infinicoder.planning.task_contract import Check, TaskSpec
from infinicoder.execution.workspace_tools import Workspace, run_process
from test_collaboration import USAGE, bound, owned_source_tokens, service, source_schema
from namespace_support import namespace_command, require_namespace

BWRAP = os.environ.get("IC_OWNED_BWRAP") or shutil.which("bwrap")
READY = os.name == "posix" and BWRAP and shutil.which("g++") and importlib.util.find_spec("clang")
SOURCE = "int combine(int a,int b) { return a-b; }\n"
CORRECT = "int combine(int a,int b) { return a+b; }"


def project(root):
    ws = root / "baseline"
    (ws / "engine").mkdir(parents=True)
    (ws / "engine/math.cc").write_text(SOURCE)
    (ws / "engine/other.cc").write_text("int untouched() { return 7; }\n")
    (ws / "compile_commands.json").write_text(
        json.dumps(
            [
                dict(
                    directory=".",
                    file="engine/math.cc",
                    arguments=["g++", "-std=c++17", "-c", "engine/math.cc", "-o", "math.o"],
                )
            ]
        )
    )
    return ws


@unittest.skipUnless(READY, "requires existing Linux bwrap, g++ and Clang bindings")
class GroundedFocusVertical(unittest.TestCase):
    def setUp(self):
        require_namespace()

    def test_actual_read_focus_worker_compile_ordinary_and_unique_final(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ws = project(root)
            digest = hashlib.sha256(SOURCE.encode()).hexdigest()
            public = root / "public"
            public.mkdir()
            checker = public / "check.py"
            checker.write_text(
                "from pathlib import Path\nimport subprocess,sys\n"
                'ws=Path(sys.argv[1]);test=Path("/tmp/focus-main.cc")\n'
                'test.write_text("int combine(int,int); int main(){return combine(3,2)==5?0:1;}\\n")\n'
                'subprocess.run(["g++","-std=c++17",str(ws/"engine/math.cc"),str(test),"-o","/tmp/focus-check"],check=True)\n'
                'subprocess.run(["/tmp/focus-check"],check=True)\n'
            )

            def builder(source, argv):
                return namespace_command(BWRAP, source, public, argv)

            spec = TaskSpec(
                "Repair combine in engine/math.cc while preserving other engine behavior and its signature.",
                allowed_paths=["engine/"],
                checks=[
                    Check(
                        "public_cpp",
                        ["/usr/bin/python3", "-B", str(checker), "{workspace}"],
                        10,
                        protected_paths=[str(checker)],
                    )
                ],
            )
            counts = {}
            lock = threading.Lock()
            worker_requests = []

            def answer(body):
                schema = body.get("response_format", {}).get("json_schema", {}).get("schema", {})
                text = json.dumps(body["messages"]).replace('\\"', '"')
                if "proposals" in schema.get("properties", {}):
                    return json.dumps(
                        dict(
                            proposals=[
                                dict(
                                    strategy=strategy,
                                    approach=approach,
                                    assumptions=[],
                                    interfaces=["combine(int,int)"],
                                    write_paths=["engine/math.cc"],
                                    source_refs=[dict(path="engine/math.cc", sha256=digest)],
                                    verification=["public_cpp"],
                                )
                                for strategy, approach in [
                                    (
                                        "auto_source",
                                        "Inspect then repair the selected function body.",
                                    ),
                                    (
                                        "ordinary_actions",
                                        "Inspect then use a minimal expression edit.",
                                    ),
                                ]
                            ]
                        )
                    ), USAGE
                ordinary = '"strategy": "ordinary_actions"' in text
                role = "ordinary" if ordinary else "source"
                with lock:
                    step = counts.get(role, 0)
                    counts[role] = step + 1
                if source_schema(schema):
                    worker_requests.append(body)
                    return json.dumps(dict(source=CORRECT)), USAGE
                actions = [dict(tool="read", arguments=dict(path="engine/math.cc"))]
                if ordinary:
                    actions += [
                        dict(
                            tool="plan",
                            arguments=dict(
                                units=[
                                    dict(
                                        goal="Repair the arithmetic expression.",
                                        write_set=["engine/math.cc"],
                                        check_ids=["public_cpp"],
                                        interfaces=["combine(int,int)"],
                                        dependencies=[],
                                    )
                                ]
                            ),
                        ),
                        dict(
                            tool="edit",
                            arguments=dict(
                                path="engine/math.cc", expected_hash=digest, old="a-b", new="a+b"
                            ),
                        ),
                    ]
                else:
                    actions += [
                        dict(
                            tool="focus_edit",
                            arguments=dict(
                                path="engine/math.cc",
                                symbol="combine",
                                expected_hash=digest,
                                reason="The public arithmetic behavior requires addition.",
                            ),
                        )
                    ]
                actions += [dict(tool="finish", arguments={})]
                return json.dumps(actions[min(step, len(actions) - 1)]), USAGE

            finals = []
            runtimes = []

            def final(path, original, remaining):
                self.assertEqual(asdict(original), asdict(spec))
                self.assertEqual(
                    (path / "engine/other.cc").read_text(), "int untouched() { return 7; }\n"
                )
                finals.append(path)
                code = (
                    "from pathlib import Path;import subprocess;"
                    'p=Path("/tmp/independent.cc");'
                    'p.write_text("int combine(int,int);int main(){return combine(-3,2)==-1 && combine(0,8)==8 ? 0:1;}\\n");'
                    'subprocess.run(["g++","-std=c++17","/workspace/engine/math.cc",str(p),"-o","/tmp/independent"],check=True);'
                    'subprocess.run(["/tmp/independent"],check=True)'
                )
                result = run_process(
                    builder(path, ["/usr/bin/python3", "-B", "-c", code]),
                    str(path),
                    min(remaining, 12),
                )
                return FinalAcceptance(
                    result.exit_code == 0, "owned independent compiled C++ cases"
                )

            with service(answer) as (client, requests):
                profile = ModelProfile(16384, 256, 256)
                req = CoreRequest(spec, ws, root / "state", profile, CoreBudget(19, 65, 8))

                def factory(c):
                    runtime = create_focus_collaboration_runtime(
                        c, namespace_builder=builder, request_timeout=8
                    )
                    runtimes.append(runtime)
                    return runtime

                outcome = run_collaboration(
                    req,
                    client=client,
                    runtime_factory=factory,
                    final_acceptance=final,
                    limits=CollaborationLimits(19, 65, 15),
                    policy=CollaborationPolicy("heterogeneous"),
                    request_token_bound=bound,
                    tokenizer_identity="owned synthetic",
                    source_token_counter=owned_source_tokens,
                    exclusive_workspace=True,
                )
            events = [
                [
                    json.loads(line)
                    for line in (r.state_dir / "events.jsonl").read_text().splitlines()
                ]
                for r in runtimes
            ]
            self.assertTrue(outcome.success)
            self.assertEqual(len(finals), 1)
            self.assertEqual((ws / "engine/math.cc").read_text(), SOURCE)
            self.assertEqual(len(worker_requests), 1)
            self.assertTrue(
                any(
                    e["kind"] == "edit_focus_selected" and e["data"]["language"] == "native"
                    for e in events[0]
                )
            )
            self.assertFalse(any(e["kind"] == "edit_focus_selected" for e in events[1]))
            self.assertTrue(runtimes[0].grounded_planning)
            self.assertFalse(runtimes[1].grounded_planning)
            self.assertEqual(
                {o.status for o in outcome.public_observations},
                {"passed"},
                {r.collaboration_strategy: r.store.load().get("last_failure") for r in runtimes},
            )
            self.assertEqual(outcome.usage["physical_request_attempts"], len(requests))

    def test_current_scope_hash_compile_and_ambiguity_admission(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ws = project(root)
            workspace = Workspace(ws, ["engine/"])
            raw = (ws / "engine/math.cc").read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            inspected = [
                dict(path="engine/math.cc", sha256=digest, spans=[[1, 1]], full_file_visible=True)
            ]
            unit = type(
                "Unit",
                (),
                dict(kind="implement", worker_path=None, write_set=["engine/"], id="owned"),
            )()
            spec = TaskSpec("Repair combine", allowed_paths=["engine/"])
            args = dict(
                path="engine/math.cc",
                symbol="combine",
                expected_hash=digest,
                reason="Owned localization control.",
            )
            self.assertEqual(
                select(spec, args, workspace, inspected, unit, 0)["language"], "native"
            )
            with self.assertRaisesRegex(ValueError, "changed"):
                select(spec, dict(args, expected_hash="0" * 64), workspace, inspected, unit, 0)
            outside = Workspace(ws, ["another/"])
            with self.assertRaisesRegex(ValueError, "edit scope"):
                select(spec, args, outside, inspected, unit, 0)
            unit.write_set = ["engine/other.cc"]
            with self.assertRaisesRegex(ValueError, "edit scope"):
                select(spec, args, workspace, inspected, unit, 0)
            unit.write_set = ["engine/"]
            (ws / "compile_commands.json").unlink()
            with self.assertRaisesRegex(ValueError, "Clang"):
                select(spec, args, workspace, inspected, unit, 0)
            (ws / "compile_commands.json").write_text(
                json.dumps(
                    [
                        dict(
                            directory=".",
                            file="engine/math.cc",
                            arguments=["g++", "-std=c++17", "-c", "engine/math.cc"],
                        )
                    ]
                )
            )
            source = SOURCE + "double combine(double a,double b) { return a-b; }\n"
            (ws / "engine/math.cc").write_text(source)
            digest = hashlib.sha256(source.encode()).hexdigest()
            with self.assertRaisesRegex(ValueError, "Clang"):
                select(
                    spec,
                    dict(args, expected_hash=digest),
                    workspace,
                    [
                        dict(
                            path="engine/math.cc",
                            sha256=digest,
                            spans=[[1, 2]],
                            full_file_visible=True,
                        )
                    ],
                    unit,
                    0,
                )


if __name__ == "__main__":
    unittest.main()
