from contextlib import nullcontext
import json, tempfile, time, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from infinicoder import (
    NativeArtifactArchitecture,
    ReadOnlyTaskSources,
    CoreBudget,
    CoreRequest,
    TaskSpec,
    ModelProfile,
    SourceAttachment,
)
from infinicoder.planning.native_artifact_architecture import artifact_graph, _plan_packet_identity


def unit(name, path, requires=()):
    return dict(
        id=name,
        goal="Implement public contract " + name,
        paths=[path],
        source_refs=[path],
        interfaces=["input x;output scalar"],
        requires=list(requires),
        verify=["run public integration test"],
    )


def count(messages, remaining):
    return len(json.dumps(messages)) // 4


class ArtifactArchitectureTests(unittest.TestCase):
    def case(self, root, response, context_policy="reversible"):
        workspace = root / "workspace"
        workspace.mkdir()
        state = root / "state"
        state.mkdir()
        files = {
            "core.py": "def scale(x):\n    return x*2\n",
            "api.py": "from core import scale\ndef serve(x):\n    return scale(x)\n",
        }
        for name, text in files.items():
            (workspace / name).write_bytes(text.encode())
        literal = json.dumps(files)
        task = "Implement public scale and serve interfaces.\nFILES:\n" + literal
        c = NativeArtifactArchitecture(
            count,
            tokenizer_identity="owned engineering counter",
            source_capability=ReadOnlyTaskSources(workspace, tuple(files)),
            attachment=SourceAttachment(literal, files),
            implementation_paths=tuple(files),
            artifact_write_paths=tuple(files),
            context_policy=context_policy,
        )
        c.request_limits = []

        def request_limit(**kw):
            c.request_limits.append(kw)
            return nullcontext()

        req = CoreRequest(
            TaskSpec(task),
            workspace,
            state,
            ModelProfile(8192, 1024),
            CoreBudget(30, 300, 30),
            transport_adapter=SimpleNamespace(limit_request=request_limit),
            context_adapter=c,
            planning_adapter=c,
        )
        model = SimpleNamespace(
            _prepare_messages_for_api=lambda m: m,
            query=lambda m, **kw: dict(role="assistant", content="ordinary"),
        )

        def execute(action):
            name = action["command"]
            (workspace / name).write_bytes((files[name] + "# implementation changed\n").encode())
            return dict(returncode=0, output="edited")

        env = SimpleNamespace(
            implementation_workspace_binding=lambda: dict(
                root=str(workspace.resolve()), model_root="/workspace"
            ),
            execute=execute,
        )
        queries = []

        def query(messages, **kw):
            queries.append(messages)
            return dict(
                role="assistant",
                content=response,
                extra=dict(
                    response=dict(
                        choices=[dict(finish_reason="stop", message=dict(content=response))]
                    )
                ),
            )

        initial = [
            dict(role="system", content="ordinary SDK rules"),
            dict(role="user", content="SDK_INSTANCE_PREFIX\n" + task + "\nSDK_FORMAT_RULES"),
        ]
        return c, req, model, env, query, queries, initial

    def test_one_plan_then_real_artifact_changes_and_dependency_invalidation(self):
        value = dict(units=[unit("core", "core.py"), unit("api", "api.py", ("core",))])
        with tempfile.TemporaryDirectory() as tmp:
            c, r, m, e, q, queries, initial = self.case(Path(tmp), json.dumps(value))
            with patch(
                "infinicoder.planning.native_artifact_architecture.planning_alarm",
                lambda _: nullcontext(),
            ):
                with c.bind(r, m, e, time.monotonic() + 300):
                    c.prepare_plan(
                        r, m, e, query=q, public_task=r.spec.task, initial_messages=lambda: initial
                    )
                    self.assertEqual(c.graph.order, ["core", "api"])
                    self.assertEqual(len(queries), 1)
                    self.assertIn("core.py", c.plan_source_versions)
                    c.parse_tool_command("ic-plan checkpoint core")
                    with self.assertRaises(ValueError):
                        c.tool(["checkpoint", "core"])
                    m.query(initial)
                    e.execute(dict(command="core.py"))
                    c.parse_tool_command("ic-plan checkpoint core")
                    c.tool(["checkpoint", "core"])
                    e.execute(dict(command="api.py"))
                    c.parse_tool_command("ic-plan checkpoint api")
                    c.tool(["checkpoint", "api"])
                    self.assertEqual(c.provisional, {"core", "api"})
                    self.assertEqual(c.changed_units, {"core", "api"})
                    # A later upstream source edit reopens its dependent stage.
                    (r.workspace / "core.py").write_bytes(b"def scale(x):\n    return x*3\n")
                    e.execute(dict(command="core.py"))
                    self.assertFalse(c.provisional)
                    self.assertFalse(c.plan_report.get("independent_acceptance", False))
                    c.parse_tool_command("ic-plan checkpoint api")
                    with self.assertRaises(ValueError):
                        c.tool(["checkpoint", "api"])
                    self.assertEqual(c.planning_query_calls, 1)

    def test_planner_uses_exact_public_task_and_one_physical_call(self):
        value = dict(units=[unit("core", "core.py"), unit("api", "api.py", ("core",))])
        with tempfile.TemporaryDirectory() as tmp:
            c, r, m, e, q, queries, initial = self.case(Path(tmp), json.dumps(value))
            with patch(
                "infinicoder.planning.native_artifact_architecture.planning_alarm",
                lambda _: nullcontext(),
            ):
                with c.bind(r, m, e, time.monotonic() + 300):
                    c.prepare_plan(
                        r, m, e, query=q, public_task=r.spec.task, initial_messages=lambda: initial
                    )
                    self.assertNotIn("SDK_INSTANCE_PREFIX", queries[0][1]["content"])
                    self.assertNotIn("SDK_FORMAT_RULES", queries[0][1]["content"])
                    self.assertEqual(c.request_limits[0]["max_physical_calls"], 1)
                    c.parse_tool_command("ic-plan focus api")
                    c.tool(["focus", "api"])
                    view = c._unit_view("api")
                    self.assertEqual(view["current_sources"][0]["path"], "api.py")
                    self.assertIn("return scale", json.dumps(view))

    def test_invalid_plan_has_one_query_and_preserves_ordinary_coding(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, r, m, e, q, queries, initial = self.case(Path(tmp), "not a JSON graph")
            with patch(
                "infinicoder.planning.native_artifact_architecture.planning_alarm",
                lambda _: nullcontext(),
            ):
                with c.bind(r, m, e, time.monotonic() + 300):
                    c.prepare_plan(
                        r, m, e, query=q, public_task=r.spec.task, initial_messages=lambda: initial
                    )
                    self.assertIsNone(c.graph)
                    self.assertEqual(c.plan_report["status"], "ordinary_fallback")
                    self.assertEqual(len(queries), 1)
                    m.query(initial)

    def test_graph_rejects_cycles_and_retains_unknown_source_references(self):
        a = unit("a", "a.py", ("b",))
        b = unit("b", "b.py", ("a",))
        with self.assertRaises(ValueError):
            artifact_graph(dict(units=[a, b]), set())
        a["requires"] = []
        a["source_refs"] = ["unknown.py::method"]
        graph, rows = artifact_graph(dict(units=[a]), set())
        self.assertIn("unknown", rows["a"]["source_status"]["unknown.py::method"])
        self.assertEqual(graph.order, ["a"])

    def test_packet_identity_only_ignores_inactive_sdk_assistant_fields(self):
        base = [dict(role="assistant", content="ordinary action", provider_specific_fields={})]
        prepared = [dict(base[0], tool_calls=None, function_call=None)]
        left, right = _plan_packet_identity(base), _plan_packet_identity(prepared)
        self.assertNotEqual(
            left["prepared_api_messages_sha256"], right["prepared_api_messages_sha256"]
        )
        self.assertEqual(left["comparable_messages_sha256"], right["comparable_messages_sha256"])
        for change in (
            dict(content="changed"),
            dict(tool_calls=[{"id": "actual"}]),
            dict(provider_specific_fields={"changed": True}),
        ):
            self.assertNotEqual(
                left["comparable_messages_sha256"],
                _plan_packet_identity([dict(base[0], **change)])["comparable_messages_sha256"],
            )

    def test_plain_planner_preserves_source_task_and_latest_feedback(self):
        value = dict(units=[unit("core", "core.py"), unit("api", "api.py", ("core",))])
        with tempfile.TemporaryDirectory() as tmp:
            c, r, m, e, q, queries, initial = self.case(Path(tmp), json.dumps(value), "plain")
            with patch(
                "infinicoder.planning.native_artifact_architecture.planning_alarm",
                lambda _: nullcontext(),
            ):
                with c.bind(r, m, e, time.monotonic() + 300):
                    c.prepare_plan(
                        r, m, e, query=q, public_task=r.spec.task, initial_messages=lambda: initial
                    )
                    self.assertIn(r.spec.task, queries[0][1]["content"])
                    self.assertNotIn("initial_source_archive", queries[0][1]["content"])
                    history = initial + [
                        dict(role="assistant", content="ordinary action"),
                        dict(role="user", content="real failed integration feedback"),
                    ]
                    projected = c.project(history)
                    self.assertEqual(projected[:2], initial)
                    self.assertEqual(projected[3:], history[2:])
                    self.assertEqual(projected[-1], history[-1])
                    self.assertEqual(
                        json.loads(projected[2]["content"])["layer"],
                        "implementation_artifact_graph",
                    )
                    prepared = next(x for x in c.plan_events if x["kind"] == "prepared_plan")
                    self.assertTrue(prepared["included_source_ranges"])
                    self.assertEqual(
                        prepared["dispatch_status"], "unproven;actual_HTTP_ledger_required"
                    )
                    self.assertEqual(c.records, [])
                    self.assertNotIn("working_set_directory", json.dumps(projected))
                    self.assertEqual(c.command_names, ("ic-plan",))
                    answer = e.execute(dict(command="ic-plan focus api"))
                    self.assertEqual(answer["returncode"], 0)
                    self.assertEqual(c.active_unit, "api")
                    self.assertEqual(c.planning_query_calls, 1)
            info = json.loads((r.state_dir / "native-memory-info.json").read_text())
            self.assertEqual(info["context_policy"], "plain")
            self.assertFalse(info["automatic_compression"])
            self.assertEqual(info["tool_families"], ["ic-plan"])

    def test_plain_invalid_plan_keeps_exact_original_coding_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, r, m, e, q, queries, initial = self.case(Path(tmp), "invalid plan", "plain")
            with patch(
                "infinicoder.planning.native_artifact_architecture.planning_alarm",
                lambda _: nullcontext(),
            ):
                with c.bind(r, m, e, time.monotonic() + 300):
                    c.prepare_plan(
                        r, m, e, query=q, public_task=r.spec.task, initial_messages=lambda: initial
                    )
                    self.assertIsNone(c.graph)
                    self.assertEqual(len(queries), 1)
                    self.assertEqual(c.project(initial), initial)
                    self.assertEqual(c.records, [])

    def test_plain_plan_does_not_replace_history_when_packet_cannot_fit(self):
        value = dict(units=[unit("core", "core.py")])
        with tempfile.TemporaryDirectory() as tmp:
            c, r, m, e, q, queries, initial = self.case(Path(tmp), json.dumps(value), "plain")
            with patch(
                "infinicoder.planning.native_artifact_architecture.planning_alarm",
                lambda _: nullcontext(),
            ):
                with c.bind(r, m, e, time.monotonic() + 300):
                    c.prepare_plan(
                        r, m, e, query=q, public_task=r.spec.task, initial_messages=lambda: initial
                    )
                    c._count = lambda messages: (
                        c.limit + 1
                        if "implementation_artifact_graph" in json.dumps(messages)
                        else 100
                    )
                    self.assertEqual(c.project(initial), initial)
                    self.assertFalse(any(x["kind"] == "prepared_plan" for x in c.plan_events))

    def test_controller_read_is_not_counted_as_source_in_the_prepared_packet(self):
        value = dict(units=[unit("core", "core.py")])
        with tempfile.TemporaryDirectory() as tmp:
            c, r, m, e, q, queries, initial = self.case(Path(tmp), json.dumps(value), "plain")
            with patch(
                "infinicoder.planning.native_artifact_architecture.planning_alarm",
                lambda _: nullcontext(),
            ):
                with c.bind(r, m, e, time.monotonic() + 300):
                    c.prepare_plan(
                        r, m, e, query=q, public_task=r.spec.task, initial_messages=lambda: initial
                    )

                    def tokens(messages):
                        for message in messages:
                            try:
                                note = json.loads(message["content"])
                            except ValueError:
                                continue
                            if (
                                isinstance(note, dict)
                                and note.get("layer") == "implementation_artifact_graph"
                            ):
                                return c.limit + 1 if note["unit_view"]["current_sources"] else 100
                        return 100

                    c._count = tokens
                    projected = c.project(initial)
                    prepared = next(x for x in c.plan_events if x["kind"] == "prepared_plan")
                    self.assertTrue(prepared["controller_observed_source_versions"])
                    self.assertEqual(prepared["included_source_ranges"], [])
                    self.assertIn("not_delivered", prepared["source_display_status"])
                    self.assertIn(
                        "not_delivered",
                        json.loads(projected[2]["content"])["unit_view"]["source_view_status"],
                    )

    def test_plain_policy_is_wired_through_public_factory_and_rejected_on_other_profiles(self):
        from infinicoder import create_cpu_request, ModelConfig

        class Counter:
            identity = "owned-factory-counter"

            def __call__(self, messages, remaining):
                return count(messages, remaining)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c, r, m, e, q, queries, initial = self.case(root, "unused", "plain")
            final = root / "owned-final.py"
            final.write_text("print('1 passed')\n")
            with patch(
                "infinicoder.models.native_transport.LocalChatCounter", return_value=Counter()
            ):
                request = create_cpu_request(
                    r.spec,
                    r.workspace,
                    root / "factory-state",
                    model_config=ModelConfig("http://127.0.0.1:9/v1", "owned"),
                    tokenizer_path=root / "unused-tokenizer",
                    check_script=final,
                    profile="cpu-architecture",
                    architecture_context="plain",
                    architecture_write_paths=("core.py", "api.py"),
                    implementation_paths=("core.py", "api.py"),
                    source_attachment=c.attachment,
                )
            self.assertIs(request.context_adapter, request.planning_adapter)
            self.assertEqual(request.context_adapter.context_policy, "plain")
            self.assertEqual(request.spec, r.spec)
            with patch(
                "infinicoder.models.native_transport.LocalChatCounter",
                side_effect=AssertionError("must reject before tokenizer"),
            ):
                with self.assertRaises(ValueError):
                    create_cpu_request(
                        r.spec,
                        r.workspace,
                        root / "unused-state",
                        model_config=None,
                        tokenizer_path=None,
                        check_script=final,
                        profile="cpu-plain",
                        architecture_context="plain",
                    )

    def test_plain_metadata_failure_preserves_original_cancellation(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, r, m, e, q, queries, initial = self.case(Path(tmp), "unused", "plain")
            original = KeyboardInterrupt("owned user cancellation")
            write = Path.write_text

            def blocked(path, text, *args, **kwargs):
                if path.name == "native-memory-info.json" and '"tool_families"' in text:
                    raise OSError("owned optional metadata failure")
                return write(path, text, *args, **kwargs)

            with patch.object(Path, "write_text", blocked):
                with self.assertRaises(KeyboardInterrupt) as result:
                    with c.bind(r, m, e, time.monotonic() + 300):
                        raise original
            self.assertIs(result.exception, original)


if __name__ == "__main__":
    unittest.main()
