"""One budgeted artifact graph proposal and observed modular implementation."""

import ast, copy, json, re, time
from contextlib import contextmanager
from pathlib import PurePosixPath

from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.memory.source.native_reversible_working_set import NativeWorkingSet
from infinicoder.runtime.native_plain import NativePlainObserver
from infinicoder.memory.native_memory import _json, _sha
from infinicoder.planning.native_source_planning import planning_alarm, PlanningDeadline
from infinicoder.analysis.task_compiler import WorkGraph, WorkUnit, _topological_order

PLAN_SYSTEM = """Propose a compact implementation graph for this public programming task.
Return ONE JSON object, optionally in a json fence. At most3 implementation units:
{"units":[{"id":"core","goal":"concrete requirement","paths":["relative.py"],
"source_refs":["exact catalog path or symbol id"],"interfaces":["input/output/state contract"],
"requires":[],"verify":["public test or integration procedure"]}]}.
Separate module responsibilities and dependencies. Quote only catalog source_refs;
new interfaces and new files are proposals,never existing facts. Verification steps
are suggestions,not executed tests. Do not output code,bash actions,or private checks.
The ordinary coding agent will implement and integrate these units within the SAME budget."""


def _plan_packet_identity(messages):
    # The pinned SDK prepares inactive assistant function/tool fields as null;
    # its HTTP serializer omits them. Preserve every other field and any real
    # function/tool payload. A comparable hash is still not a dispatch receipt.
    comparable = [
        {
            k: v
            for k, v in row.items()
            if not (
                row.get("role") == "assistant"
                and k in ("tool_calls", "function_call")
                and v is None
            )
        }
        for row in messages
    ]
    return dict(
        prepared_api_messages_sha256=_sha(_json(messages)),
        comparable_messages_sha256=_sha(_json(comparable)),
        comparison_normalization="Omit only null assistant tool_calls/function_call;all other fields retained",
    )


def artifact_graph(value, known_sources):
    if (
        not isinstance(value, dict)
        or set(value) != {"units"}
        or not isinstance(value["units"], list)
        or not 1 <= len(value["units"]) <= 3
    ):
        raise ValueError("one graph with1..3 units required")
    units = []
    details = {}
    for row in value["units"]:
        if not isinstance(row, dict) or set(row) != set(
            ("id", "goal", "paths", "source_refs", "interfaces", "requires", "verify")
        ):
            raise ValueError("explicit artifact/interface/dependency/verification fields required")
        name = row["id"]
        if (
            not isinstance(name, str)
            or not re.fullmatch("[a-z][a-z0-9_-]{0,31}", name)
            or name in details
        ):
            raise ValueError("unique unit ids required")
        if not isinstance(row["goal"], str) or not 0 < len(row["goal"]) <= 800:
            raise ValueError("bounded unit goal required")
        for field in ("paths", "source_refs", "interfaces", "requires", "verify"):
            if (
                not isinstance(row[field], list)
                or len(row[field]) > 16
                or any(not isinstance(x, str) or not 0 < len(x) <= 500 for x in row[field])
            ):
                raise ValueError("bounded explicit unit lists required")
        if not row["paths"] or not row["interfaces"] or not row["verify"]:
            raise ValueError("unit artifacts,interfaces and validation required")
        for path in row["paths"]:
            p = PurePosixPath(path)
            if (
                p.is_absolute()
                or ".." in p.parts
                or "\\" in path
                or ":" in path
                or str(p) != path
                or any(x in (".git", ".infinicoder") for x in p.parts)
            ):
                raise ValueError("normalized proposed artifact paths required")
        unit = WorkUnit(name, row["goal"], dependencies=row["requires"], write_set=row["paths"])
        unit.validate()
        units.append(unit)
        details[name] = dict(
            copy.deepcopy(row),
            source_status={
                ref: "observed_catalog" if ref in known_sources else "unknown;not_source_fact"
                for ref in row["source_refs"]
            },
            interfaces_authority="model_proposed_contract;not_verified_behavior",
        )
    return WorkGraph(units, _topological_order(units), {}), details


class NativeArtifactArchitecture(NativeWorkingSet):
    identity = "native_artifact_architecture_v2_versioned_consumers"
    report_filename = "native-artifact-plan.json"
    command_names = ("ic-workset", "ic-plan")
    HELP = (
        NativeWorkingSet.HELP
        + " ic-plan status;ic-plan unit UNIT;ic-plan focus UNIT;ic-plan checkpoint UNIT;ic-plan revise JSON. Checkpoints are versioned provisional artifacts;final acceptance remains external."
    )

    def __init__(
        self,
        *args,
        repository_task=True,
        artifact_write_paths=(),
        context_policy="reversible",
        **kwargs,
    ):
        if context_policy not in ("reversible", "plain"):
            raise ValueError("architecture context_policy must be reversible or plain")
        super().__init__(*args, **kwargs)
        self.repository_task = repository_task
        self.context_policy = context_policy
        if context_policy == "plain":
            self.command_names = ("ic-plan",)
            self.HELP = (
                "ic-plan status;ic-plan unit UNIT;ic-plan focus UNIT;ic-plan checkpoint UNIT;"
                "ic-plan revise JSON. One local graph revision. Checkpoints are provisional;"
                "read and edit through ordinary tools;final acceptance remains external."
            )
        self.artifact_write_paths = tuple(artifact_write_paths)
        self.planning_query_calls = 0
        self.plan_report = dict(status="not_started")
        self.graph = None
        self.units = {}
        self.plan_events = []
        self.provisional = set()
        self.changed_units = set()
        self.local_replans = 0
        self.active_unit = None
        self.unit_state = {}
        self.execution_epoch = 0
        self.validation_events = []

    @contextmanager
    def bind(self, request, model, environment, deadline):
        interrupted = False
        try:
            with super().bind(request, model, environment, deadline) as bound:
                yield bound
        except BaseException:
            interrupted = True
            raise
        finally:
            if self.context_policy == "plain" and getattr(self, "state", None) is not None:
                try:
                    path = self.state / "native-memory-info.json"
                    if path.exists():
                        info = json.loads(path.read_text(encoding="utf8"))
                        info.update(
                            context_policy="plain",
                            automatic_compression=False,
                            model_tools_installed=True,
                            tool_families=["ic-plan"],
                            projection="Original SDK history plus an admitted bounded architecture packet;no working-set projection",
                        )
                        path.write_text(_json(info), encoding="utf8")
                except (OSError, ValueError):
                    if not interrupted:
                        raise

    def task_view(self, task):
        # Planning can be evaluated independently of source externalization.
        return task if self.context_policy == "plain" else super().task_view(task)

    def validate(self, request):
        super().validate(request)
        if request.planning_adapter is not self or request.context_adapter is not self:
            raise UnsupportedCore("artifact planner/context share one adapter instance")
        if type(self.repository_task) is not bool:
            raise ValueError("explicit planner applicability required")
        for path in self.artifact_write_paths:
            p = PurePosixPath(path)
            if p.is_absolute() or ".." in p.parts or "\\" in path or ":" in path or str(p) != path:
                raise ValueError("explicit normalized artifact write authorization required")
            self.sources.workspace.path(path)

    def _save_plan(self):
        (self.state / "native-artifact-plan.json").write_text(
            _json(
                dict(
                    report=self.plan_report,
                    context_policy=self.context_policy,
                    units=self.units,
                    order=self.graph.order if self.graph else [],
                    active_unit=self.active_unit,
                    provisional=sorted(self.provisional),
                    changed_units=sorted(self.changed_units),
                    local_replans=self.local_replans,
                    unit_state=self.unit_state,
                    validation_events=self.validation_events,
                    events=self.plan_events,
                    independent_acceptance=False,
                    edit_permissions_changed=False,
                )
            ),
            encoding="utf8",
        )

    def _catalog(self, public_task):
        anchors = set(re.findall(r"[A-Za-z_]\w*", public_task.lower()))
        ordered = sorted(
            self.sources.paths,
            key=lambda p: (-len(anchors & set(re.findall(r"[A-Za-z_]\w*", p.lower()))), p),
        )
        self._refresh(ordered[:8])
        rows = self._interfaces()
        known = set()
        self.catalog_bindings = {}
        for row in rows:
            known.add(row["path"])
            known.update(s["id"] for s in row["interfaces"])
            for ref in (row["path"], *[s["id"] for s in row["interfaces"]]):
                self.catalog_bindings[ref] = dict(
                    path=row["path"], sha256=row["sha256"], namespace=self._namespace()
                )
        return rows, known

    def _activate_graph(self, graph, units):
        self.graph, self.units = graph, units
        self.active_unit = graph.order[0]
        watched = [
            p
            for u in graph.units
            for p in u.write_set
            if p in self.artifact_write_paths and p in self.sources.paths
        ]
        self.paths = tuple(dict.fromkeys((*self.paths, *watched)))[:32]
        self.known = self._snapshot()
        self.unit_state = {}
        for name, row in units.items():
            row["source_bindings"] = {
                ref: copy.deepcopy(self.catalog_bindings.get(ref, dict(status="unknown")))
                for ref in row["source_refs"]
            }
            row["artifact_permissions"] = {
                p: "caller_authorized_observed"
                if p in self.artifact_write_paths and p in self.paths
                else "advisory;no_observed_write_authorization"
                for p in row["paths"]
            }
            self.unit_state[name] = dict(
                status="pending",
                last_change_epoch=0,
                required_epoch=self.execution_epoch,
                required_public_check_id=len(self.checks),
                checkpoint=None,
            )

    @staticmethod
    def _receipt_view(receipt):
        if receipt is None:
            return None
        return dict(
            epoch=receipt["epoch"],
            source_versions=receipt["source_versions"],
            dependency_receipt_hashes=receipt["dependency_receipt_hashes"],
            status=receipt["status"],
            public_check_ids=[r["check_id"] for r in receipt["public_check_observations"]],
            ordinary_execution_ids=[
                r["execution_id"] for r in receipt["ordinary_execution_observations"]
            ],
            receipt_sha256=_sha(_json(receipt)),
            independent_acceptance=False,
        )

    def _state_view(self):
        return {
            name: dict(state, checkpoint=self._receipt_view(state["checkpoint"]))
            for name, state in self.unit_state.items()
        }

    def _choose_active(self):
        ready = [
            n
            for n in self.graph.order
            if self.unit_state[n]["checkpoint"] is None
            and all(self.unit_state[d]["checkpoint"] is not None for d in self.units[n]["requires"])
        ]
        self.active_unit = (
            ready
            or [n for n in self.graph.order if self.unit_state[n]["checkpoint"] is None]
            or [self.graph.order[-1]]
        )[0]

    def _invalidate_artifacts(self):
        current = self._snapshot()
        stale = set()
        for name, state in self.unit_state.items():
            receipt = state["checkpoint"]
            if receipt and any(
                current.get(p, {}).get("sha256") != pin
                for p, pin in receipt["source_versions"].items()
            ):
                stale.add(name)
        if not stale:
            return
        direct = set(stale)
        self.execution_epoch += 1
        for name in self.graph.order:
            if stale.intersection(self.units[name]["requires"]):
                stale.add(name)
        for name in stale:
            self.unit_state[name].update(
                checkpoint=None,
                required_epoch=self.execution_epoch,
                required_public_check_id=len(self.checks),
                status="unattributed_or_unknown_source;needs_observation",
            )
            if name in direct:
                self.unit_state[name]["last_change_epoch"] = 0
        self.provisional.difference_update(stale)
        self._choose_active()
        self.plan_events.append(
            dict(
                kind="artifact_version_invalidated",
                units=sorted(stale),
                origin="source_snapshot;not_attributed_to_agent",
            )
        )

    def _unit_view(self, name):
        row = self.units[name]
        paths = list(row["paths"]) + [ref.split("::", 1)[0] for ref in row["source_refs"]]
        upstream = {}
        for dep in row["requires"]:
            receipt = self.unit_state[dep]["checkpoint"]
            upstream[dep] = self._receipt_view(receipt)
            if receipt:
                paths += list(receipt["source_versions"])
        paths = list(dict.fromkeys(p for p in paths if p in self.sources.paths))[:8]
        self._refresh(paths)
        records = []
        for path in paths[:3]:
            file = self.index.db.execute(
                "SELECT hash,source,error FROM files WHERE path=?", (path,)
            ).fetchone()
            if file is None:
                records.append(dict(path=path, status="missing_or_unknown;new_artifact_proposal"))
                continue
            try:
                raw, text, pin = self.sources.read(path)
            except (ValueError, OSError):
                records.append(dict(path=path, status="current_source_unknown"))
                continue
            if pin != file["hash"]:
                records.append(dict(path=path, status="changed_during_view"))
                continue
            symbols = [
                dict(s)
                for s in self.index.db.execute(
                    "SELECT id,signature,line,end FROM symbols WHERE path=? ORDER BY line LIMIT 3",
                    (path,),
                )
            ]
            chosen = next(
                (s for s in symbols if s["id"] in row["source_refs"]),
                symbols[0] if symbols else None,
            )
            lines = text.splitlines()
            start = chosen["line"] if chosen else 1
            end = min(chosen["end"] if chosen else len(lines), start + 19)
            records.append(
                dict(
                    path=path,
                    sha256=pin,
                    interfaces=symbols,
                    source_range=dict(
                        start=start,
                        end=end,
                        exact_lines=[
                            dict(line=i + 1, text=lines[i]) for i in range(start - 1, end)
                        ],
                        complete_definition=bool(
                            chosen and start == chosen["line"] and end == chosen["end"]
                        ),
                        complete_file=start == 1 and end == len(lines),
                    ),
                )
            )
        return dict(
            active_unit=name,
            current_sources=records,
            upstream_artifact_receipts=upstream,
            status=self.unit_state[name]["status"],
            proposed_interfaces=row["interfaces"],
            authority="Current ranged source and observed artifacts;model interfaces remain hypotheses",
        )

    def prepare_plan(self, request, model, environment, *, query, public_task, initial_messages):
        if self.plan_report["status"] != "not_started":
            raise UnsupportedCore("architecture proposal cannot repeat")
        self.plan_report = dict(status="not_applicable", planning_query_calls=0)
        if (
            not self.repository_task
            or request.budget.max_model_calls < 15
            or request.budget.wall_seconds < 300
        ):
            self.plan_report["reason"] = "lightweight_or_insufficient_original_budget"
            self._save_plan()
            return
        if not callable(getattr(request.transport_adapter, "limit_request", None)):
            self.plan_report["reason"] = "shared_transport_subrequest_limits_unavailable"
            self._save_plan()
            return
        cutoff = min(
            self.deadline - 5, time.monotonic() + min(60, request.budget.wall_seconds * 0.15)
        )
        try:
            with planning_alarm(cutoff):
                catalog, known = self._catalog(public_task)
                self.known_plan_sources = known
                task = self.task_view(public_task)
                messages = self.planning_messages(task, catalog)
                tokens = self._count(messages)
                if tokens > self.limit:
                    self.plan_report.update(
                        reason="planner_input_exceeds_shared_context", input_tokens=tokens
                    )
                    return
                folder = self.state / "artifact-plan"
                folder.mkdir()
                (folder / "input.json").write_text(_json(messages), encoding="utf8")
                self.planning_query_calls = 1
                self.plan_report.update(
                    status="planning_query_started", planning_query_calls=1, input_tokens=tokens
                )
                self._save_plan()
                from infinicoder.runtime.native_protocol import FormatError

                with request.transport_adapter.limit_request(
                    output_tokens=min(1024, request.model_profile.output_reserve),
                    absolute_deadline=cutoff,
                    max_physical_calls=1,
                ):
                    try:
                        response = query(
                            messages,
                            max_tokens=min(1024, request.model_profile.output_reserve),
                            timeout=max(0.001, cutoff - time.monotonic()),
                            num_retries=0,
                        )
                    except FormatError as error:
                        response = error.messages[0]
                (folder / "response.json").write_text(_json(response), encoding="utf8")
                choice = (response.get("extra", {}).get("response", {}).get("choices") or [{}])[0]
                text = choice.get("message", {}).get("content", response.get("content", ""))
                if choice.get("finish_reason") != "stop":
                    raise ValueError("incomplete architecture response")
                fence = re.fullmatch(r"\s*```(?:json)?\s*\n(.*?)\n```\s*", text, re.S)
                value = json.loads(fence.group(1) if fence else text)
                graph, units = artifact_graph(value, known)
                self._activate_graph(graph, units)
                self.plan_source_versions = {r["path"]: r["sha256"] for r in catalog}
                self.plan_report.update(
                    status="prepared",
                    plan_sha256=_sha(_json(value)),
                    source_versions=self.plan_source_versions,
                )
        except (PlanningDeadline, TimeoutError):
            self.graph = None
            self.units = {}
            self.unit_state = {}
            self.plan_report.update(status="ordinary_fallback", reason="planning_deadline;no_retry")
        except (ValueError, KeyError, TypeError, OSError) as error:
            self.graph = None
            self.units = {}
            self.unit_state = {}
            self.plan_report.update(
                status="ordinary_fallback",
                reason=type(error).__name__ + ";one_invalid_plan_no_retry",
            )
        finally:
            self._save_plan()

    def planning_messages(self, task, catalog):
        return [
            dict(role="system", content=PLAN_SYSTEM),
            dict(role="user", content=task + "\nCURRENT SOURCE INTERFACES\n" + _json(catalog)),
        ]

    def execution_completed(self, event):
        super().execution_completed(event)
        if self.graph is None:
            return
        self.execution_epoch += 1
        changed = set(event.get("changed", []))
        affected = [u.id for u in self.graph.units if changed.intersection(u.write_set)]
        self.changed_units.update(affected)
        if (
            not changed
            and event.get("returncode") is not None
            and not str(event.get("command", "")).lstrip().startswith(("ic-plan", "ic-workset"))
        ):
            self.validation_events.append(
                dict(
                    epoch=self.execution_epoch,
                    execution_id=event["id"],
                    command_sha256=_sha(event.get("command", "")),
                    returncode=event["returncode"],
                    sources=event["after"],
                    authority="ordinary source-preserving execution observation;test semantics and correctness unverified",
                )
            )
            self.validation_events = self.validation_events[-8:]
        if affected:
            self.plan_events.append(
                dict(
                    kind="observed_source_change",
                    execution_id=event["id"],
                    units=affected,
                    paths=sorted(changed),
                    coverage="declared observed source paths only;shared-path unit attribution ambiguous",
                    public_validation=False,
                )
            )
            # A later edit reopens provisional stages and their dependents.
            stale = set(affected)
            for name in self.graph.order:
                if stale.intersection(self.units[name]["requires"]):
                    stale.add(name)
            self.provisional.difference_update(stale)
            for name in stale:
                state = self.unit_state[name]
                state.update(
                    required_epoch=self.execution_epoch,
                    checkpoint=None,
                    required_public_check_id=min(
                        [c["id"] for c in self.checks if c.get("execution_id") == event["id"]]
                        + [len(self.checks)]
                    ),
                    status="changed_unverified" if name in affected else "needs_integration",
                )
                if name in affected:
                    state["last_change_epoch"] = self.execution_epoch
            self._choose_active()
        self._save_plan()

    def _checkpoint(self, name):
        row = self.units[name]
        state = self.unit_state[name]
        if any(
            value != "caller_authorized_observed" for value in row["artifact_permissions"].values()
        ):
            raise ValueError("advisory unit has no complete observed write authorization")
        if state["last_change_epoch"] <= 0:
            raise ValueError("checkpoint requires a real observed artifact change")
        self._refresh(row["paths"])
        pins = {}
        syntax = {}
        declarations = {}
        for path in row["paths"]:
            raw, text, pin = self.sources.read(path)
            pins[path] = pin
            if path.endswith((".py", ".pyi")):
                ast.parse(text)
                syntax[path] = "Python AST parsed;not behavioral validation"
            else:
                syntax[path] = "native build not observed"
            declarations[path] = [
                dict(s)
                for s in self.index.db.execute(
                    "SELECT id,signature,line,end FROM symbols WHERE path=? ORDER BY line LIMIT 8",
                    (path,),
                )
            ]
        ordinary = [
            v
            for v in self.validation_events
            if v["epoch"] >= state["required_epoch"]
            and all(v["sources"].get(p, {}).get("sha256") == pin for p, pin in pins.items())
        ]
        current = self._snapshot()
        checks = [
            dict(
                check_id=c["id"],
                judge_identity=c["judge_identity"],
                report_sha256=_sha(_json(c["report"])),
                authority="associated public observation;not final acceptance",
            )
            for c in self.checks
            if c["id"] >= state["required_public_check_id"]
            and c.get("finished")
            and c.get("origin") == "controller_public_operation"
            and c.get("before") == c.get("after") == current
        ]
        if state["last_change_epoch"] < state["required_epoch"] and not ordinary and not checks:
            raise ValueError(
                "dependency reopened this artifact;new source change or fresh associated observation required"
            )
        dependencies = {
            dep: copy.deepcopy(self.unit_state[dep]["checkpoint"]) for dep in row["requires"]
        }
        receipt = dict(
            epoch=self.execution_epoch,
            source_versions=pins,
            declarations=declarations,
            syntax_observations=syntax,
            dependency_receipt_hashes={
                d: _sha(_json(v)) if v else None for d, v in dependencies.items()
            },
            ordinary_execution_observations=ordinary[-2:],
            public_check_observations=checks[-2:],
            status="changed_with_public_observation"
            if checks
            else "changed_with_ordinary_observation"
            if ordinary
            else "changed_unverified",
            interface_semantics="proposed contract remains unverified",
            independent_acceptance=False,
        )
        state.update(checkpoint=receipt, status=receipt["status"])
        self.provisional.add(name)
        self._choose_active()
        return receipt

    def project(self, messages):
        packet = (
            NativePlainObserver.project(self, messages)
            if self.context_policy == "plain"
            else super().project(messages)
        )
        return self.render_plan(packet, messages)

    def render_plan(self, packet, messages):
        """Add one bounded plan without rewriting canonical history."""
        if self.graph is None:
            return packet
        self._invalidate_artifacts()
        unit_view = self._unit_view(self.active_unit)
        # Only freshly observed paths can be called current. Old SQLite rows
        # remain historical even when they happen to retain the same hash.
        observed = {
            r["path"]: r.get("sha256") for r in unit_view["current_sources"] if r.get("sha256")
        }
        note = dict(
            layer="implementation_artifact_graph",
            active=self.active_unit,
            order=self.graph.order,
            units=self.units,
            unit_view=unit_view,
            unit_state=self._state_view(),
            source_reference_versions={
                p: "current" if observed.get(p) == pin else "not_reobserved_or_changed"
                for p, pin in self.plan_source_versions.items()
            },
            changed_units=sorted(self.changed_units),
            provisional=sorted(self.provisional),
            authority="Model plan and checkpoints are not correctness;implement artifacts,integrate interfaces,then complete independent final acceptance.",
            tools="ic-plan status;ic-plan focus UNIT;ic-plan checkpoint UNIT;one local ic-plan revise JSON allowed",
        )
        extra = dict(role="user", content=_json(note))

        def compose():
            # In the independent planner mode the real latest action/feedback
            # remains last. The existing reversible profile keeps its contract.
            return (
                [*packet[:2], extra, *packet[2:]]
                if self.context_policy == "plain"
                else [*packet, extra]
            )

        if self._count(compose()) > self.limit:
            # Preserve the active artifact/dependency contract;omit complete
            # source records with explicit retrieval instead of clipping code.
            note["unit_view"]["current_sources"] = []
            note["unit_view"]["source_view_status"] = (
                "not_delivered;use ic-plan unit UNIT or ordinary source tools"
                if self.context_policy == "plain"
                else "not_delivered;use ic-plan unit UNIT or ic-workset source PATH START END"
            )
            extra = dict(role="user", content=_json(note))
        if self._count(compose()) <= self.limit:
            packet = compose()
            included = [
                dict(
                    path=r["path"],
                    sha256=r["sha256"],
                    start=r["source_range"]["start"],
                    end=r["source_range"]["end"],
                )
                for r in note["unit_view"]["current_sources"]
                if r.get("sha256") and "source_range" in r
            ]
            self.plan_events.append(
                dict(
                    kind="prepared_plan",
                    canonical_messages=len(messages),
                    plan_sha256=self.plan_report["plan_sha256"],
                    active=self.active_unit,
                    **_plan_packet_identity(self.prepare(packet)),
                    controller_observed_source_versions=observed,
                    included_source_ranges=included,
                    source_display_status="included_in_prepared_packet"
                    if included
                    else "not_delivered;no_source_ranges_in_packet",
                    dispatch_status="unproven;actual_HTTP_ledger_required",
                )
            )
        self._save_plan()
        return packet

    def parse_tool_command(self, command):
        args = super().parse_tool_command(command)
        self._tool_family = args[0]
        return args

    def tool(self, args):
        if getattr(self, "_tool_family", "ic-workset") != "ic-plan":
            return super().tool(args)
        if self.graph is None:
            raise ValueError(
                "No admitted architecture graph;ordinary implementation remains available"
            )
        if args == ["status"]:
            return self._output(
                dict(
                    order=self.graph.order,
                    active=self.active_unit,
                    units=self.units,
                    unit_state=self._state_view(),
                    provisional=sorted(self.provisional),
                )
            )
        if len(args) == 2 and args[0] == "unit":
            self.graph.unit(args[1])
            return self._output(self._unit_view(args[1]))
        if len(args) == 2 and args[0] in ("focus", "checkpoint"):
            name = args[1]
            self.graph.unit(name)
            if args[0] == "focus":
                self.active_unit = name
            else:
                self._checkpoint(name)
            self.plan_events.append(dict(kind=args[0], unit=name, independent_acceptance=False))
            self._save_plan()
            return self._output(
                dict(
                    unit=name,
                    status=self.unit_state[name]["status"]
                    if args[0] == "checkpoint"
                    else "selected",
                    unresolved_dependencies=[
                        x for x in self.units[name]["requires"] if x not in self.provisional
                    ],
                )
            )
        if len(args) == 2 and args[0] == "revise":
            if self.local_replans:
                raise ValueError("one local graph revision already used")
            value = json.loads(args[1])
            graph, units = artifact_graph(value, self.known_plan_sources)
            self.plan_events.append(
                dict(
                    kind="local_plan_revision",
                    old_units=copy.deepcopy(self.units),
                    new_units=copy.deepcopy(units),
                    extra_model_query=False,
                    ordinary_SDK_action=True,
                )
            )
            self._activate_graph(graph, units)
            self.local_replans = 1
            self.provisional.clear()
            self.changed_units.clear()
            self.plan_report["plan_sha256"] = _sha(_json(value))
            self._save_plan()
            return self._output(dict(status="revised;reopened", order=graph.order))
        raise ValueError("ic-plan status|unit UNIT|focus UNIT|checkpoint UNIT|revise JSON")
