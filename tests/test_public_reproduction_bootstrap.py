"""Common public observation in both arms and actual optional frame evidence."""

import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

from infinicoder.runtime.agent_execution import CallerPublicOperation, CoreBudget, CoreRequest
from infinicoder.models.budget import ModelProfile
from infinicoder.verification.native_public_reproduction import NativePublicReproductionBootstrap
from infinicoder.verification.public_replay import (
    PublicReplayCapability,
    PublicReplaySpec,
    PublicTraceRecorder,
    digest,
)
from infinicoder.verification.public_reproduction import (
    PublicDevelopmentReproduction,
    public_source_directory,
)
from infinicoder.planning.task_contract import TaskSpec

SOURCE = b"def scale(value, divisor):\n    return value / divisor\n"
ENTRY = b"from impl import scale\nassert scale(12, 0)==0\n"


class PublicBootstrapControls(unittest.TestCase):
    def case(self, root, *, enabled=True, passed=False):
        workspace = root / "workspace"
        workspace.mkdir()
        state = root / "state"
        state.mkdir()
        source = (
            SOURCE
            if not passed
            else SOURCE.replace(b"value / divisor", b"0 if divisor==0 else value / divisor")
        )
        (workspace / "impl.py").write_bytes(source)
        (workspace / "repro.py").write_bytes(ENTRY)
        spec = PublicReplaySpec(
            ("python", "repro.py"),
            ("impl.py", "repro.py"),
            {"repro.py": digest(ENTRY)},
            "Owned public zero-divisor requirement",
            final_reserve_seconds=1,
        )
        calls = []
        packets = []

        def execute(tracing, deadline):
            calls.append(tracing)
            namespace = {}
            exec(
                compile(
                    (workspace / "impl.py").read_bytes(),
                    str((workspace / "impl.py").resolve()),
                    "exec",
                ),
                namespace,
            )
            recorder = PublicTraceRecorder(workspace, spec.source_paths)
            error = None
            try:
                if tracing:
                    with recorder:
                        namespace["scale"](12, 0)
                else:
                    namespace["scale"](12, 0)
            except ZeroDivisionError:
                error = "ZeroDivisionError"
            trace = recorder.report()
            trace.update(
                instrumented=tracing,
                exit_code=int(error is not None),
                exception_type=error,
                public_argv=spec.argv,
                source_after={p: digest((workspace / p).read_bytes()) for p in spec.source_paths},
            )
            return dict(
                trace=trace,
                process=dict(
                    output=error or "public relation passed", timed_out=False, truncated=False
                ),
            )

        def cap(tracing):
            operation = CallerPublicOperation(
                "owned-trace" if tracing else "owned-ordinary",
                workspace,
                dict(worker="owned-public-function"),
                spec.public_origin,
                dict(public=True),
                30,
                dict(instrumentation="bounded_python_frames" if tracing else "none"),
                spec.source_paths,
                (),
                1024,
                lambda deadline: execute(tracing, deadline),
            )
            return PublicReplayCapability(spec, operation)

        reproduction = PublicDevelopmentReproduction(cap(False), cap(True))
        context = NativePublicReproductionBootstrap(
            lambda m, t: len(json.dumps(m)) // 4,
            tokenizer_identity="owned-test-counter",
            reproduction=reproduction,
            diagnostics=enabled,
        )
        request = CoreRequest(
            TaskSpec("Owned contract:scale with zero divisor returns0;run public repro.py."),
            workspace,
            state,
            ModelProfile(16384, 4096),
            CoreBudget(4, 30, 5),
            transport_adapter=SimpleNamespace(delivery_policy=None),
            context_adapter=context,
        )
        model = SimpleNamespace(
            query=lambda messages, **kwargs: (
                packets.append(messages) or dict(role="assistant", content="ordinary")
            ),
            _prepare_messages_for_api=lambda m: m,
        )

        def edit(action, **kwargs):
            (workspace / "impl.py").write_bytes(source + b"# ordinary legal edit\n")
            return dict(returncode=0, output="edited", exception_info="")

        environment = SimpleNamespace(
            execute=edit,
            implementation_workspace_binding=lambda: dict(
                root=str(workspace.resolve()), model_root="/workspace"
            ),
        )
        return SimpleNamespace(**locals())

    def run_case(self, c, *, edit=False):
        messages = [
            dict(role="system", content="Owned system"),
            dict(role="user", content=c.request.spec.task),
        ]
        with c.context.bind(c.request, c.model, c.environment, time.monotonic() + 30):
            answer = c.model.query(messages)
            if edit:
                result = c.environment.execute(dict(command="owned edit"))
                c.model.query([*messages, answer, dict(role="user", content=result["output"])])

    def test_both_arms_receive_same_public_report_only_on_receives_dynamic_trace(self):
        common = []
        for enabled in (False, True):
            with tempfile.TemporaryDirectory() as folder:
                c = self.case(Path(folder), enabled=enabled)
                self.run_case(c)
                self.assertEqual(c.calls, [False, True] if enabled else [False])
                packet = c.packets[0]
                reports = [
                    m["content"]
                    for m in packet
                    if m["content"].startswith("PUBLIC DEVELOPMENT REPRODUCTION")
                ]
                traces = [
                    m["content"]
                    for m in packet
                    if m["content"].startswith("PUBLIC FAILURE EXECUTION SLICE")
                ]
                self.assertEqual(len(reports), 1)
                self.assertEqual(len(traces), int(enabled))
                common.append(reports[0])
                self.assertEqual(packet[1]["content"], c.request.spec.task)
                self.assertEqual(c.context.bootstrap_report["status"], "observed_nonzero_exit")
        self.assertEqual(common[0], common[1])

    def test_public_success_has_no_diagnostic_and_no_claim_of_final_acceptance(self):
        with tempfile.TemporaryDirectory() as folder:
            c = self.case(Path(folder), passed=True)
            self.run_case(c)
            self.assertEqual(c.calls, [False])
            self.assertIsNone(c.context.report)
            self.assertEqual(c.context.bootstrap_report["status"], "observed_exit_zero")
            self.assertIn("not final acceptance", c.packets[0][-1]["content"])

    def test_edit_keeps_original_feedback_and_marks_public_observation_historical(self):
        with tempfile.TemporaryDirectory() as folder:
            c = self.case(Path(folder))
            self.run_case(c, edit=True)
            self.assertEqual(c.calls, [False, True])
            self.assertEqual(c.context.bootstrap_report["source_versions"], "historical")
            self.assertFalse(
                any(m["content"].startswith("PUBLIC FAILURE EXECUTION SLICE") for m in c.packets[1])
            )
            self.assertIn("edited", [m["content"] for m in c.packets[1]])

    def test_exact_historical_output_tool_never_reruns_public_program(self):
        with tempfile.TemporaryDirectory() as folder:
            c = self.case(Path(folder))
            self.run_case(c)
            value = json.loads(c.context.tool(["output", "0", "200"]))
            self.assertEqual(value["text"], "ZeroDivisionError")
            self.assertFalse(value["commands_replayed"])
            self.assertEqual(c.calls, [False, True])

    def test_public_mount_keeps_reproduction_bytes_out_of_candidate_export(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            workspace = root / "workspace"
            workspace.mkdir()
            public = root / "public"
            public.mkdir()
            path = public / "repro.py"
            path.write_bytes(ENTRY)
            spec = PublicReplaySpec(
                ("python", "__ic_public__/repro.py"),
                ("impl.py", "__ic_public__/repro.py"),
                {"__ic_public__/repro.py": digest(ENTRY)},
                "Owned separate public source",
            )
            source, mapping = public_source_directory(
                spec, workspace, {"__ic_public__/repro.py": str(path)}
            )
            self.assertEqual(source, public.resolve())
            self.assertEqual(mapping["__ic_public__/repro.py"], path)
            self.assertEqual(list((workspace / "__ic_public__").iterdir()), [])
            (public / "unclaimed.py").write_bytes(b"# undeclared\n")
            with self.assertRaisesRegex(ValueError, "exactly"):
                public_source_directory(spec, workspace, {"__ic_public__/repro.py": str(path)})


if __name__ == "__main__":
    unittest.main()
