"""Conditional capabilities sharing one action loop, source journal and query owner."""

from contextlib import contextmanager, ExitStack
from dataclasses import dataclass, asdict, replace
import re

from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.routing.development_workflow import workflow_packet
from infinicoder.planning.native_artifact_architecture import NativeArtifactArchitecture
from infinicoder.runtime.native_plain import NativePlainObserver
from infinicoder.memory.source.native_reversible_working_set import NativeWorkingSet
from infinicoder.optimization.native_measured_optimization import PublicPerformanceController
from infinicoder.memory.native_memory import _json
from infinicoder.routing.profiles import route_task
from infinicoder.verification.public_execution_facet import PublicExecutionFacet
from infinicoder.verification.public_replay import digest


@dataclass(frozen=True)
class CapabilityPipelinePolicy:
    profile: str = "auto"
    memory: str = "auto"
    planning: str = "auto"
    diagnostics: bool = True
    knowledge: bool = True

    def __post_init__(self):
        if self.profile not in ("auto", "general", "operator", "framework"):
            raise ValueError("explicit supported semantic profile required")
        if any(getattr(self, k) not in ("auto", "on", "off") for k in ("memory", "planning")):
            raise ValueError("memory/planning mode must be auto,on,off")
        if type(self.diagnostics) is not bool or type(self.knowledge) is not bool:
            raise ValueError("explicit diagnostic/knowledge switches required")


def capability_selection(
    request,
    readable_paths,
    policy,
    *,
    attachment=None,
    artifact_write_paths=(),
    reproduction=None,
    performance=False,
    provider=False,
):
    # Classification alone uses the instruction section. The request and its
    # physical task retain every original source and trailing constraint.
    text = re.split(r"REFERENCE:\n|FILES:\n", request.spec.task, maxsplit=1)[0]
    docs = bool(
        re.search(r"\b(readme|documentation|docstring|typo)\b|文档|注释", text, re.I)
    ) and not re.search(r"\b(implement|fix|debug|crash)\b|修复|实现", text, re.I)
    route = route_task(
        replace(request.spec, task=text),
        request.workspace,
        "general" if docs and policy.profile == "auto" else policy.profile,
    )
    multi = len(readable_paths) > 1
    memory = (
        policy.memory == "on"
        or policy.memory == "auto"
        and (multi or attachment is not None and len(attachment.literal.encode()) > 4096)
    )
    complex_task = multi and (
        route.profile == "framework"
        or bool(
            re.search(
                r"\b(across|integrat\w*|repository|architect\w*|multi.?file|cross.?module)\b|跨模块|多文件|复杂系统",
                text,
                re.I,
            )
        )
    )
    plan_requested = policy.planning == "on" or policy.planning == "auto" and complex_task
    plan = (
        plan_requested
        and bool(artifact_write_paths)
        and request.budget.max_model_calls >= 15
        and request.budget.wall_seconds >= 300
    )
    modules = dict(
        memory=dict(
            active=memory,
            reason="explicit_mode_or_multiple_readable_sources"
            if memory
            else "lightweight_or_disabled",
        ),
        planning=dict(
            active=plan,
            reason="explicit_artifacts_and_original_budget"
            if plan
            else "disabled_or_lightweight_or_missing_artifact_authority_or_budget",
        ),
        public_reproduction=dict(
            active=reproduction is not None,
            reason="caller_frozen_public_capability" if reproduction else "not_supplied",
        ),
        diagnostics=dict(
            active=reproduction is not None and policy.diagnostics,
            reason="caller_public_reproduction_and_switch"
            if reproduction and policy.diagnostics
            else "no_public_reproduction_or_disabled",
        ),
        domain_workflow=dict(
            active=policy.knowledge and route.profile in ("operator", "framework"),
            reason="semantic_profile"
            if policy.knowledge and route.profile in ("operator", "framework")
            else "not_a_domain_task_or_disabled",
        ),
        performance=dict(
            active=bool(performance),
            reason="caller_external_measurement_policy"
            if performance
            else "no_external_measurement;timing_unverified",
        ),
        implementation_provider=dict(
            active=bool(
                provider
                and route.profile == "operator"
                and not memory
                and not plan
                and reproduction is None
            ),
            reason="eligible_flat_operator_bundle"
            if provider
            and route.profile == "operator"
            and not memory
            and not plan
            and reproduction is None
            else "not_supplied_or_context_bundle_selected_or_profile_override",
        ),
    )
    return dict(
        profile=route.profile,
        reasons=route.reasons,
        modules=modules,
        policy=asdict(policy),
        source_coverage="Declared readable/watched paths only;classification grants no source or write permissions",
        classifier="public instruction semantics,explicit capabilities and bounded build metadata;no benchmark IDs",
        default_promoted=False,
        capability_improvement_proven=False,
    )


class NativeRouteObserver(NativePlainObserver):
    identity = "native_lightweight_capability_route_v1"

    def __init__(self, *args, selection, **kwargs):
        super().__init__(*args, **kwargs)
        self.selection = selection

    @contextmanager
    def bind(self, *args, **kwargs):
        with super().bind(*args, **kwargs):
            error = write_capability_metadata(
                self.state / "native-capabilities.json", self.selection
            )
            if error:
                self._save_event("capability_metadata_unavailable", dict(error_type=error))
            yield self


class NativeCapabilityPipeline(NativeArtifactArchitecture):
    identity = "native_single_owner_capability_pipeline_v1"

    def __init__(
        self,
        *args,
        selection,
        reproduction=None,
        performance_policy=None,
        performance_report=None,
        exclusive_workspace=False,
        public_check=None,
        **kwargs,
    ):
        super().__init__(
            *args, repository_task=selection["modules"]["planning"]["active"], **kwargs
        )
        self.selection = selection
        self.public_facet = (
            PublicExecutionFacet(reproduction, diagnostics=selection["policy"]["diagnostics"])
            if reproduction
            else None
        )
        self.performance_report, self.public_check = performance_report, public_check
        self.performance = None
        self._performance_policy = performance_policy
        self._exclusive = exclusive_workspace
        self.capability_records = []
        self.bound_once = False
        self.command_names = tuple(
            (
                ["ic-workset"]
                if selection["modules"]["memory"]["active"] or self.repository_task
                else []
            )
            + (["ic-plan"] if self.repository_task else [])
            + (["ic-public-reproduction"] if self.public_facet else [])
        )
        self.expose_tools = bool(self.command_names)
        self.HELP = " ".join(
            (
                [NativeWorkingSet.HELP]
                if selection["modules"]["memory"]["active"] or self.repository_task
                else []
            )
            + (
                ["ic-plan status;unit UNIT;focus UNIT;checkpoint UNIT;revise JSON"]
                if self.repository_task
                else []
            )
            + ([self.public_facet.HELP] if self.public_facet else [])
        )

    def validate(self, request):
        super().validate(request)
        if (
            request.repair_adapter is not None
            or request.candidate_search is not None
            or request.implementation_proposal is not None
        ):
            raise UnsupportedCore(
                "pipeline uses one ordinary coding loop;other coding/search owners remain explicit profiles"
            )
        if getattr(request.transport_adapter, "delivery_policy", None) is not None:
            raise UnsupportedCore("pipeline shares the ordinary transport budget")
        if len(self.paths) > 32:
            raise ValueError(
                "at most32 watched sources;use readable source paths for the larger repository"
            )
        if self.public_facet:
            self.public_facet.validate_component(request)

    def _snapshot(self):
        snapshot = super()._snapshot()
        if self.public_facet:
            for relative in self.public_facet.capability.readonly_sources or {}:
                path = self.public_facet.capability.source_path(relative)
                try:
                    if (
                        any(p.is_symlink() for p in (path, *path.parents))
                        or not path.is_file()
                        or path.stat().st_size > 1024**2
                    ):
                        raise ValueError("invalid frozen public source")
                    snapshot[relative] = dict(kind="file", sha256=digest(path.read_bytes()))
                except (ValueError, OSError) as error:
                    snapshot[relative] = dict(kind="unknown", reason=type(error).__name__)
        return snapshot

    @contextmanager
    def bind(self, request, model, environment, deadline):
        if self.bound_once:
            raise UnsupportedCore("capability pipeline is one-shot")
        self.bound_once = True
        self.request = request
        with super().bind(request, model, environment, deadline):
            self.index.db.execute(
                "CREATE TABLE native_pipeline_projections(id INTEGER PRIMARY KEY,messages TEXT,accounting TEXT)"
            )
            self.index.db.commit()
            try:
                with ExitStack() as stack:
                    if self.public_facet:
                        stack.enter_context(
                            self.public_facet.bind_component(self, request, environment, deadline)
                        )
                    policy = self._performance_policy
                    if self.performance_report:
                        decode, identity = self.performance_report.decoder(self.public_check)
                        original = self.public_check.operation

                        def measure(cutoff):
                            if (
                                self.public_operation is None
                                or self.public_operation.operation is not original
                            ):
                                raise UnsupportedCore("original public measurement is not bound")
                            raw = self.public_operation.invoke(cutoff)
                            text = _json(raw)
                            if len(text.encode()) > 8 * 1024**2:
                                raise ValueError("bounded public measurement report required")
                            folder = self.state / "public-performance-reports"
                            folder.mkdir(exist_ok=True)
                            path = folder / (str(len(list(folder.glob("*.json")))) + ".json")
                            path.write_text(text, encoding="utf8")
                            return replace(decode(raw), evidence_ref=str(path))

                        operation = replace(
                            original,
                            operation_id=original.operation_id + ".decoded-performance",
                            implementation_identity=dict(
                                original=original.implementation_identity, decoder=identity
                            ),
                            executor=measure,
                        )
                        policy = self.performance_report.policy(operation, self._exclusive)
                    if policy:
                        self.performance = PublicPerformanceController(policy, self.sources)
                        stack.enter_context(self.performance.bind(request, self, deadline))
                    yield self
            finally:
                try:
                    if self.graph is not None:
                        self._invalidate_artifacts()
                        self._save_plan()
                except (ValueError, OSError, TimeoutError) as error:
                    self.capability_records.append(
                        dict(kind="final_plan_state_unknown", error_type=type(error).__name__)
                    )
                error = write_capability_metadata(
                    self.state / "native-capabilities.json",
                    dict(
                        self.selection,
                        records=self.capability_records,
                        planning_query_calls=self.planning_query_calls,
                        runtime_observations=dict(
                            working_projections=len(self.records),
                            plan_status=self.plan_report["status"],
                            public_bootstrap_status=(self.public_facet.bootstrap_report or {}).get(
                                "status", "not_run"
                            )
                            if self.public_facet
                            else "not_configured",
                            diagnostic_status=(self.public_facet.report or {}).get(
                                "status", "not_run"
                            )
                            if self.public_facet
                            else "not_configured",
                            measured_versions=len(self.performance.measured)
                            if self.performance
                            else 0,
                        ),
                        one_query_owner=True,
                        one_source_journal=True,
                        extra_unbudgeted_model_calls=0,
                    ),
                )
                if error:
                    self._save_event("capability_metadata_unavailable", dict(error_type=error))

    def before_query(self, messages):
        if self.public_facet:
            self.public_facet.before_query(messages)
        if self.performance:
            self.performance.before_query()

    def execution_completed(self, event):
        super().execution_completed(event)
        if self.public_facet:
            self.public_facet.execution_completed(event)
        if self.performance and set(event.get("changed", ())).intersection(
            self.performance.policy.operation.observed_paths
        ):
            self.performance.source_changed()

    def prepare_plan(self, *args, **kwargs):
        if self.repository_task and self.public_facet:
            self.public_facet.before_query([])
        return super().prepare_plan(*args, **kwargs)

    def planning_messages(self, task, catalog):
        packet = super().planning_messages(task, catalog)
        return self.public_facet.append_view(packet, packet) if self.public_facet else packet

    def _append(self, packet, value, kind, max_tokens=None):
        extra = dict(role="user", content=_json(value))
        fits = (max_tokens is None or self._count([extra]) <= max_tokens) and self._count(
            [*packet, extra]
        ) <= self.limit
        self.capability_records.append(
            dict(
                kind=kind,
                prepared=fits,
                delivery_status="physical_HTTP_ledger_required",
                canonical_messages=len(self.seen),
            )
        )
        return [*packet, extra] if fits else packet

    def project(self, messages):
        packet = (
            NativeWorkingSet.project(self, messages)
            if self.selection["modules"]["memory"]["active"]
            else NativePlainObserver.project(self, messages)
        )
        packet = self.render_plan(packet, messages)
        if self.public_facet:
            packet = self.public_facet.append_view(packet, messages)
        if self.selection["modules"]["domain_workflow"]["active"]:
            changed = [p for e in self.executions for p in e.get("changed", ())]
            last = self.executions[-1] if self.executions else None
            failure = (
                dict(execution_id=last["id"])
                if last
                and last.get("returncode") not in (None, 0)
                and last.get("after") == self._snapshot()
                else None
            )
            state = dict(
                changed_files=changed,
                last_failure=failure,
                source_inspections=self.working_paths,
                optimization=dict(contract=True) if self.performance else {},
            )
            workflow = workflow_packet(self.selection["profile"], self.request.spec, state)
            compact = {
                k: workflow[k]
                for k in (
                    "id",
                    "stage",
                    "stage_basis",
                    "source_revision",
                    "acceptance",
                    "performance_rule",
                    "branches",
                )
            }
            compact.update(
                next_actions=workflow["next_actions"][:3],
                specialized=workflow["specialized"][:2],
                omitted_guidance_items=max(0, len(workflow["next_actions"]) - 3)
                + max(0, len(workflow["specialized"]) - 2),
            )
            packet = self._append(
                packet, dict(layer="domain_development_workflow", **compact), "domain_workflow", 420
            )
        if self.performance:
            packet = self._append(packet, self.performance.view(), "performance_feedback")
        tokens = self._count(packet)
        if tokens > self.limit:
            raise ValueError("composed capability input exceeds the actual shared context")
        with self.index.db:
            self.index.db.execute(
                "INSERT INTO native_pipeline_projections(messages,accounting) VALUES(?,?)",
                (
                    _json(packet),
                    _json(
                        dict(
                            prepared=True,
                            physical_HTTP_ledger_required=True,
                            input_tokens=tokens,
                            input_limit=self.limit,
                            canonical_messages=len(self.seen),
                        )
                    ),
                ),
            )
        return packet

    def tool(self, args):
        if getattr(self, "_tool_family", None) == "ic-public-reproduction":
            return self.public_facet.tool(args)
        return super().tool(args)


def write_capability_metadata(path, value):
    """Display persistence cannot replace an original cancellation or SDK error."""
    try:
        path.write_text(_json(value), encoding="utf8")
    except (OSError, ValueError, TypeError) as error:
        return type(error).__name__
    return None
