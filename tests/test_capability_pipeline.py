"""Capability interactions and real source evidence;no neural/GPU score."""

from contextlib import nullcontext, redirect_stdout, redirect_stderr
from dataclasses import replace
import hashlib, io, json, tempfile, time, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from infinicoder.runtime.agent_execution import CoreRequest, CoreBudget
from infinicoder.models.budget import ModelProfile
from infinicoder.planning.task_contract import TaskSpec
from infinicoder.memory.source.evidence_source import ReadOnlyTaskSources
from infinicoder.runtime.native_plain import NativePlainObserver
from infinicoder.runtime.extensions.native_capability_pipeline import (
    CapabilityPipelinePolicy,
    NativeCapabilityPipeline,
    NativeRouteObserver,
)
from infinicoder.app.development_api import (
    configure_development_request,
    create_development_request,
)
from infinicoder.optimization.native_measured_optimization import (
    PublicPerformancePolicy,
    PublicPerformanceSample,
)
from infinicoder.verification.public_performance_report import PublicPerformanceReport
from infinicoder.optimization.optimization import Measurement
import test_public_reproduction_bootstrap as bootstrap_controls
import test_native_artifact_architecture as architecture_controls


def count(messages, remaining):
    return len(json.dumps(messages)) // 4


class CapabilityInteractions(unittest.TestCase):
    def case(self, root, *, files=None, task="Implement a small integer helper."):
        workspace = root / "workspace"
        workspace.mkdir()
        state = root / "state"
        state.mkdir()
        files = files or {"impl.py": "def scale(x):\n    return x*2\n"}
        for path, source in files.items():
            (workspace / path).write_bytes(source.encode())
        observer = NativePlainObserver(
            count, tokenizer_identity="owned-unit-counter", implementation_paths=tuple(files)
        )
        req = CoreRequest(
            TaskSpec(task, required_artifacts=list(files)),
            workspace,
            state,
            ModelProfile(8192, 1024),
            CoreBudget(30, 300, 30),
            context_adapter=observer,
            transport_adapter=SimpleNamespace(delivery_policy=None),
        )
        packets = []
        model = SimpleNamespace(
            _prepare_messages_for_api=lambda m: m,
            query=lambda m, **kw: packets.append(m) or dict(role="assistant", content="ordinary"),
        )

        def execute(action):
            (workspace / "impl.py").write_bytes((action["command"] + "\n").encode())
            return dict(returncode=0, output="ordinary edit")

        env = SimpleNamespace(
            execute=execute,
            implementation_workspace_binding=lambda: dict(
                root=str(workspace.resolve()), model_root="/workspace"
            ),
        )
        return SimpleNamespace(**locals())

    def test_lightweight_route_has_exact_input_and_does_not_replace_contract_or_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self.case(Path(tmp))
            r = configure_development_request(
                c.req, source_capability=ReadOnlyTaskSources(c.workspace, tuple(c.files))
            )
            self.assertIsInstance(r.context_adapter, NativeRouteObserver)
            for field in (
                "spec",
                "budget",
                "model_profile",
                "execution_adapter",
                "transport_adapter",
                "acceptance_adapter",
            ):
                self.assertIs(getattr(r, field), getattr(c.req, field))
            h = [dict(role="system", content="original"), dict(role="user", content=r.spec.task)]
            with r.context_adapter.bind(r, c.model, c.env, time.monotonic() + 300):
                c.model.query(h)
            self.assertEqual(c.packets, [h])
            self.assertIsNone(r.planning_adapter)

    def test_source_words_are_not_domain_routing_instructions(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self.case(
                Path(tmp),
                task="Update README documentation.\nFILES:\n"
                + json.dumps({"note": "Implement Triton CUDA kernels and distributed scheduler"}),
            )
            r = configure_development_request(
                c.req, source_capability=ReadOnlyTaskSources(c.workspace, tuple(c.files))
            )
            self.assertEqual(r.context_adapter.selection["profile"], "general")
            self.assertFalse(r.context_adapter.selection["modules"]["domain_workflow"]["active"])
            self.assertIs(r.spec, c.req.spec)

    def test_public_trace_and_code_memory_share_one_journal_without_binding_a_second_observer(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = bootstrap_controls.PublicBootstrapControls().case(Path(tmp))
            r = configure_development_request(
                c.request,
                source_capability=ReadOnlyTaskSources(c.workspace, ("impl.py", "repro.py")),
            )
            p = r.context_adapter
            self.assertIsInstance(p, NativeCapabilityPipeline)
            h = [dict(role="system", content="original"), dict(role="user", content=r.spec.task)]
            with patch(
                "infinicoder.verification.native_public_reproduction.NativePublicReproductionBootstrap.bind",
                side_effect=AssertionError("second observer binding forbidden"),
            ):
                with p.bind(r, c.model, c.environment, time.monotonic() + 300):
                    c.model.query(h)
                    self.assertFalse(hasattr(p.public_facet, "index"))
                    self.assertEqual(
                        p.index.db.execute("SELECT COUNT(*) FROM native_history").fetchone()[0], 2
                    )
                    row = p.index.db.execute(
                        "SELECT messages FROM native_pipeline_projections ORDER BY id DESC LIMIT 1"
                    ).fetchone()
                    self.assertEqual(json.loads(row["messages"]), c.packets[0])
                    c.environment.execute(dict(command="ordinary edit"))
                    c.model.query(
                        [
                            *h,
                            dict(role="assistant", content="ordinary"),
                            dict(role="user", content="edited"),
                        ]
                    )
            self.assertEqual(c.calls, [False, True])
            self.assertEqual(len(c.packets), 2)
            self.assertTrue(
                any(m["content"].startswith("PUBLIC FAILURE EXECUTION SLICE") for m in c.packets[0])
            )
            self.assertFalse(
                any(m["content"].startswith("PUBLIC FAILURE EXECUTION SLICE") for m in c.packets[1])
            )
            self.assertEqual(len(list(c.state.glob("*.sqlite"))), 1)

    def test_single_shared_planner_call_and_live_unit_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = {
                "impl.py": "def scale(x):\n    return x*2\n",
                "api.py": "from impl import scale\ndef serve(x):\n    return scale(x)\n",
            }
            c = self.case(
                Path(tmp),
                files=files,
                task="Implement cross-module framework interfaces across impl.py and api.py.",
            )
            limits = []
            transport = SimpleNamespace(
                delivery_policy=None, limit_request=lambda **kw: limits.append(kw) or nullcontext()
            )
            r = configure_development_request(
                replace(c.req, transport_adapter=transport),
                source_capability=ReadOnlyTaskSources(c.workspace, tuple(files)),
                artifact_write_paths=tuple(files),
                policy=CapabilityPipelinePolicy(profile="framework"),
            )
            p = r.context_adapter
            unit = architecture_controls.unit
            value = dict(units=[unit("core", "impl.py"), unit("api", "api.py", ("core",))])
            calls = []

            def query(messages, **kw):
                calls.append(messages)
                return dict(
                    content=json.dumps(value),
                    extra=dict(
                        response=dict(
                            choices=[
                                dict(finish_reason="stop", message=dict(content=json.dumps(value)))
                            ]
                        )
                    ),
                )

            with (
                patch(
                    "infinicoder.planning.native_artifact_architecture.planning_alarm",
                    lambda _: nullcontext(),
                ),
            ):
                with p.bind(r, c.model, c.env, time.monotonic() + 300):
                    p.prepare_plan(
                        r,
                        c.model,
                        c.env,
                        query=query,
                        public_task=r.spec.task,
                        initial_messages=lambda: [],
                    )
                    self.assertEqual(p.planning_query_calls, 1)
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(limits[0]["max_physical_calls"], 1)
                    p.parse_tool_command("ic-plan unit api")
                    view = json.loads(p.tool(["unit", "api"]))
                    self.assertEqual(view["current_sources"][0]["path"], "api.py")
                    c.model.query(
                        [
                            dict(role="system", content="original"),
                            dict(role="user", content=r.spec.task),
                        ]
                    )
                    self.assertTrue(
                        any(
                            m["content"].startswith("{")
                            and json.loads(m["content"]).get("layer")
                            == "domain_development_workflow"
                            for m in c.packets[-1]
                        )
                    )
            self.assertEqual(len(c.packets), 1)

    def test_memory_and_measured_best_restore_share_actual_sdk_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self.case(
                Path(tmp),
                files={"impl.py": "baseline\n", "api.py": "fixed\n"},
                task="Optimize a cross-module system;preserve api.py.",
            )
            from infinicoder.runtime.agent_execution import CallerPublicOperation

            measured = []

            def measure(deadline):
                text = (c.workspace / "impl.py").read_text().strip()
                measured.append(text)
                pins = {
                    p: hashlib.sha256((c.workspace / p).read_bytes()).hexdigest() for p in c.files
                }
                return PublicPerformanceSample(
                    pins,
                    True,
                    2,
                    "owned-synthetic-pipeline",
                    Measurement((5 if text == "fast" else 10,) * 5),
                    3,
                    "owned-fixture",
                )

            op = CallerPublicOperation(
                "owned",
                c.workspace,
                dict(checker="synthetic"),
                "owned fixture",
                dict(checks=2),
                10,
                dict(device="synthetic"),
                tuple(c.files),
                (),
                512,
                measure,
            )
            r = configure_development_request(
                c.req,
                source_capability=ReadOnlyTaskSources(c.workspace, tuple(c.files)),
                performance_policy=PublicPerformancePolicy(
                    op, ("impl.py",), True, final_reserve_seconds=5
                ),
            )
            p = r.context_adapter
            h = [dict(role="system", content="original"), dict(role="user", content=r.spec.task)]
            with p.bind(r, c.model, c.env, time.monotonic() + 300):
                c.model.query(h)
                c.env.execute(dict(command="fast"))
                c.model.query(h)
                self.assertEqual(p.performance.view()["speedup_vs_first_correct"], 2)
                c.env.execute(dict(command="unmeasured"))
            self.assertEqual((c.workspace / "impl.py").read_bytes(), b"fast\n")
            self.assertEqual(measured, ["baseline", "fast"])
            self.assertEqual(len(c.packets), 2)

    def test_missing_public_measurement_authority_rejected_before_tokenizer_loading(self):
        with patch(
            "infinicoder.models.native_transport.LocalChatCounter",
            side_effect=AssertionError("should not load a tokenizer"),
        ):
            with self.assertRaises(ValueError):
                create_development_request(
                    TaskSpec("Optimize."),
                    ".",
                    "state",
                    public_performance=dict(workload_identity="fixed", rollback_paths=["impl.py"]),
                )

    def test_standard_decoder_rejects_unbound_or_incomplete_measurements(self):
        from infinicoder.verification.candidate_public_check import (
            CandidatePublicCheck,
            PublicCandidateVerdict,
        )
        from infinicoder.runtime.agent_execution import CallerPublicOperation

        with tempfile.TemporaryDirectory() as tmp:
            op = CallerPublicOperation(
                "owned",
                Path(tmp),
                dict(script_sha256="f" * 64),
                "owned",
                dict(checks=1),
                10,
                dict(device="synthetic"),
                ("impl.py",),
                (),
                512,
                lambda d: None,
            )
            check = CandidatePublicCheck(
                op, lambda raw: PublicCandidateVerdict("public_pass"), "owned"
            )
            contract = PublicPerformanceReport("fixed", ("impl.py",))
            decode, _ = contract.decoder(check)
            sample = dict(
                workload_identity="fixed",
                source_hashes={"impl.py": "a" * 64},
                correct=True,
                independent_checks=2,
                samples_ms=[1] * 5,
                warmup_count=3,
            )

            def report(value):
                return dict(
                    independent_final_used=False,
                    public_script_sha256="f" * 64,
                    process=dict(
                        output=json.dumps(dict(performance_sample=value)),
                        exit_code=0,
                        timed_out=False,
                        truncated=False,
                    ),
                )

            self.assertTrue(decode(report(sample)).correct)
            with self.assertRaises(ValueError):
                decode(report(dict(sample, warmup_count=0)))
            with self.assertRaises(ValueError):
                decode(dict(report(sample), public_script_sha256="e" * 64))
            with self.assertRaises(ValueError):
                decode(report(dict(sample, memory_observed=True)))

    def test_cpu_factory_accepts_shared_reproduction_and_separate_public_measurement_checker(self):
        from infinicoder.models.model_client import ModelConfig

        class Counter:
            identity = "owned-no-model-counter"

            def __call__(self, messages, remaining):
                return count(messages, remaining)

        with tempfile.TemporaryDirectory() as tmp:
            c = bootstrap_controls.PublicBootstrapControls().case(Path(tmp))
            public = c.root / "public-check.py"
            public.write_text("print('1 passed')\n")
            final = c.root / "final-check.py"
            final.write_text("print('1 passed')\n")
            with patch(
                "infinicoder.models.native_transport.LocalChatCounter", return_value=Counter()
            ):
                r = create_development_request(
                    c.request.spec,
                    c.workspace,
                    c.state,
                    model_config=ModelConfig("http://127.0.0.1:9/v1", "owned"),
                    tokenizer_path=c.root / "unused-tokenizer",
                    check_script=final,
                    public_check_script=public,
                    public_check_coverage=dict(scope="owned"),
                    public_check_paths=("impl.py",),
                    public_reproduction=c.spec,
                    source_read_paths=("impl.py", "repro.py"),
                )
            self.assertIsNotNone(r.acceptance_adapter.public_check)
            self.assertEqual(r.context_adapter.public_facet.reproduction.diagnostic.spec, c.spec)
            self.assertEqual(c.calls, [])

    def test_cpu_factory_preserves_final_source_exclusion_before_constructing_any_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final = root / "owned-final.py"
            # Deliberately nonexistent:only normalized path metadata is needed.
            with patch(
                "infinicoder.app.native_api.create_cpu_request",
                side_effect=AssertionError("must reject before constructing a request"),
            ):
                with self.assertRaises(ValueError):
                    create_development_request(
                        TaskSpec("Owned task"),
                        root,
                        root / "state",
                        check_script=final,
                        public_reproduction=dict(
                            argv=["python", "__ic_public__/repro.py"],
                            source_paths=["__ic_public__/repro.py"],
                            public_input_hashes={"__ic_public__/repro.py": "a" * 64},
                            public_origin="owned negative admission",
                        ),
                        public_reproduction_sources={
                            "__ic_public__/repro.py": str(root / "." / "owned-final.py")
                        },
                    )
            self.assertFalse(final.exists())

    def test_disabled_pipeline_diagnostics_does_not_swallow_invalid_legacy_boolean(self):
        with patch(
            "infinicoder.app.native_api.create_cpu_request",
            side_effect=AssertionError("must reject before constructing a request"),
        ):
            with self.assertRaises(ValueError):
                create_development_request(
                    TaskSpec("Owned task"),
                    ".",
                    "state",
                    pipeline_policy=dict(diagnostics=False),
                    public_diagnostics=1,
                    public_reproduction=dict(explicit="owned declaration"),
                )

    def test_readonly_public_identity_cannot_be_a_writable_artifact_or_shadow_the_source_reader(
        self,
    ):
        from infinicoder.verification.public_reproduction import PublicDevelopmentReproduction

        with tempfile.TemporaryDirectory() as tmp:
            c = bootstrap_controls.PublicBootstrapControls().case(Path(tmp))
            backing = c.root / "readonly-input" / "repro.py"
            reproduction = PublicDevelopmentReproduction(
                replace(c.reproduction.ordinary, readonly_sources={"repro.py": str(backing)}),
                replace(c.reproduction.diagnostic, readonly_sources={"repro.py": str(backing)}),
            )
            with self.assertRaisesRegex(ValueError, "readonly"):
                configure_development_request(
                    c.request,
                    source_capability=ReadOnlyTaskSources(c.workspace, ("impl.py",)),
                    reproduction=reproduction,
                    artifact_write_paths=("repro.py",),
                )
            with self.assertRaisesRegex(ValueError, "readonly"):
                configure_development_request(
                    c.request,
                    source_capability=ReadOnlyTaskSources(c.workspace, ("impl.py", "repro.py")),
                    reproduction=reproduction,
                )
            self.assertFalse(backing.exists())

    def test_capability_metadata_failure_does_not_mask_the_original_cancellation(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self.case(Path(tmp), files={"impl.py": "x=1\n", "api.py": "x=2\n"})
            r = configure_development_request(
                c.req, source_capability=ReadOnlyTaskSources(c.workspace, tuple(c.files))
            )
            failure = KeyboardInterrupt("owned original cancellation")
            write = Path.write_text

            def blocked(path, *args, **kwargs):
                if path.name == "native-capabilities.json":
                    raise OSError("owned optional metadata failure")
                return write(path, *args, **kwargs)

            with patch.object(Path, "write_text", blocked):
                with self.assertRaises(KeyboardInterrupt) as result:
                    with r.context_adapter.bind(r, c.model, c.env, time.monotonic() + 300):
                        raise failure
            self.assertIs(result.exception, failure)

    def test_cli_stdout_remains_one_json_when_a_dependency_prints_progress(self):
        from infinicoder.runtime.agent_execution import CoreOutcome, FinalAcceptance
        from infinicoder.app.native_cli import main

        with tempfile.TemporaryDirectory() as tmp:
            c = self.case(Path(tmp))
            task = c.root / "task.json"
            from dataclasses import asdict

            task.write_text(json.dumps(asdict(c.req.spec)))

            def construct(*args, **kwargs):
                print("owned dependency startup")
                return c.req

            def run(request):
                print("owned SDK progress")
                return CoreOutcome(
                    "owned",
                    "Submitted",
                    str(c.workspace),
                    "owned-trajectory",
                    "owned-ledger",
                    {},
                    FinalAcceptance(True, "owned-final"),
                    0,
                )

            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                patch("infinicoder.app.native_api.create_cpu_request", construct),
                patch("infinicoder.app.native_api.run_native", run),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                result = main(
                    [
                        "--workspace",
                        str(c.workspace),
                        "--state-dir",
                        str(c.state),
                        "--spec-json",
                        str(task),
                        "--check-script",
                        str(c.root / "owned-final.py"),
                        "--base-url",
                        "http://127.0.0.1:9/v1",
                        "--model",
                        "owned",
                        "--tokenizer",
                        str(c.root / "unused"),
                    ]
                )
            self.assertEqual(result, 0)
            self.assertTrue(json.loads(stdout.getvalue())["success"])
            self.assertIn("owned dependency startup", stderr.getvalue())
            self.assertIn("owned SDK progress", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
