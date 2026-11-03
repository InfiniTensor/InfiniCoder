"""One ordinary SDK workflow:public scenario→edit→same scenario→actual feedback."""

from contextlib import contextmanager
import copy
import json
from pathlib import Path
import time

from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.planning.behavior_plan import (
    BehaviorPlanPolicy,
    GUIDANCE,
    MODEL_EVIDENCE,
    digest,
    executable_script,
    parse_script,
    read_script,
    route,
)
from infinicoder.runtime.native_plain import NativePlainObserver

HEADER = "CONTROLLER BEHAVIOR SCENARIO OBSERVATION (diagnostic hypothesis;not acceptance)\n"


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class NativeBehaviorPlan(NativePlainObserver):
    identity = "executable_behavior_plan_v1"

    def __init__(
        self,
        count_messages,
        *,
        tokenizer_identity,
        implementation_paths=(),
        policy=None,
        diagnostic_factory=None,
    ):
        super().__init__(
            count_messages,
            tokenizer_identity=tokenizer_identity,
            implementation_paths=implementation_paths,
        )
        self.policy = policy or BehaviorPlanPolicy()
        self.diagnostic_factory = diagnostic_factory
        self.public_task = None
        self.bound = False
        if not isinstance(self.policy, BehaviorPlanPolicy):
            raise TypeError("typed behavior plan policy required")

    def validate(self, request):
        super().validate(request)
        if request.context_adapter is not self or any(
            getattr(request, k) is not None
            for k in (
                "planning_adapter",
                "implementation_proposal",
                "repair_adapter",
                "candidate_search",
            )
        ):
            raise UnsupportedCore("behavior plan requires a separate neutral single SDK run")
        if getattr(request.transport_adapter, "delivery_policy", None) is not None:
            raise UnsupportedCore("behavior plan uses the common legacy transport")
        if not callable(getattr(request.transport_adapter, "request_observations", None)):
            raise UnsupportedCore("actual request observations required")

    @contextmanager
    def bind_public_task(self, task):
        if self.public_task is not None or not isinstance(task, str) or not task.strip():
            raise UnsupportedCore("one exact Core public task required")
        self.public_task = task
        try:
            yield
        finally:
            self.public_task = None

    def _check_task(self):
        if time.monotonic() >= self.deadline:
            from infinicoder.runtime.native_run import CoreDeadline

            raise CoreDeadline("shared task deadline exhausted during behavior workflow")

    def _save(self):
        self.report.update(
            scenario_wall_s=self.spent,
            script_versions=len(self.versions),
            executions=len(self.runs),
            extra_model_calls=0,
            run_records=self.runs,
            script_version_hashes=self.versions,
        )
        path = self.state / "native-behavior-plan.json"
        tmp = path.with_suffix(".next")
        tmp.write_text(dump(self.report), encoding="utf8")
        tmp.replace(path)

    def _record(self, kind, **data):
        value = dict(kind=kind, **data)
        self.report["records"].append(value)
        with self.index.db:
            self.index.db.execute(
                "INSERT INTO native_behavior_plans(event) VALUES(?)", (dump(value),)
            )

    def _artifact(self, name, value):
        path = self.evidence / name
        tmp = path.with_suffix(".next")
        tmp.write_text(dump(value), encoding="utf8")
        tmp.replace(path)
        path.chmod(0o400)

    @contextmanager
    def bind(self, request, model, environment, deadline):
        if self.bound:
            raise UnsupportedCore("behavior plan cannot repeat or resume")
        self.bound = True
        self.request = request
        self.diagnostic = None
        self.spent = 0.0
        self.versions = []
        self.runs = []
        self.executed = set()
        self.origin = None
        self.feedback = None
        self.projected = None
        self.error_sent = False
        self.guidance_pending = False
        self.guidance_delivered = False
        with super().bind(request, model, environment, deadline):
            self.index.db.execute(
                "CREATE TABLE native_behavior_plans(id INTEGER PRIMARY KEY,event TEXT)"
            )
            self.index.db.commit()
            self.report = dict(
                identity=self.identity,
                status="not_applicable",
                records=[],
                deliveries=[],
                canonical_history_changed=False,
                history_crop=False,
                private_final_used=False,
                observed_paths=list(self.paths),
                unobserved_dependencies="unknown",
            )
            self.evidence = self.state / "behavior-plan" / "public-evidence"
            self.evidence.mkdir(parents=True)
            capability = (
                callable(self.diagnostic_factory)
                and callable(getattr(environment, "bind_behavior_plan_evidence", None))
                and hasattr(request.execution_adapter, "scratch")
            )
            self.route = route(
                self.public_task or "", observed_paths=self.paths, cpu_capability=capability
            )
            self.report["route"] = self.route
            if self.route["applicable"] and self.public_task is not None:
                try:
                    self.scratch = Path(request.execution_adapter.scratch)
                    if read_script(self.scratch) is not None:
                        raise ValueError(
                            "scenario must be generated during the current ordinary task"
                        )
                    self.diagnostic = self.diagnostic_factory(request, deadline)
                    self.environment_identity = digest(
                        dump(
                            dict(
                                identity=self.diagnostic.identity,
                                argv=self.diagnostic._argv(["/bin/true"]),
                            )
                        )
                    )
                    self.guidance_pending = True
                    self.report["status"] = "waiting_for_ordinary_scenario"
                except (OSError, ValueError, UnsupportedCore) as error:
                    self.report.update(
                        status="not_applicable;CPU scenario unavailable",
                        error=type(error).__name__ + ": " + str(error)[:300],
                    )
                    self.diagnostic = None
            query = model.query

            def observed_query(messages, **kwargs):
                self.projected = None
                returned = False
                try:
                    result = query(messages, **kwargs)
                    returned = True
                    return result
                finally:
                    projected = self.projected
                    if projected is not None:
                        rows = [
                            r
                            for r in request.transport_adapter.request_observations()
                            if r["attempt"] > projected["attempt_marker"]
                        ]
                        sent = any(r["dispatch_attempted"] for r in rows)
                        known_not_sent = bool(rows) and all(
                            not r["dispatch_attempted"] for r in rows
                        )
                        projected.update(
                            status="actual_HTTP_dispatched"
                            if sent
                            else "known_not_dispatched"
                            if known_not_sent
                            else "unknown_not_replayed",
                            query_returned=returned,
                            requests=[
                                dict(
                                    attempt=r["attempt"],
                                    status=r["status"],
                                    dispatch_attempted=r["dispatch_attempted"],
                                )
                                for r in rows
                            ],
                        )
                        if projected["kind"] == "guidance":
                            if sent or not known_not_sent:
                                self.guidance_pending = False
                                self.guidance_delivered = sent
                        elif (
                            self.feedback is not None
                            and self.feedback["id"] == projected["feedback_id"]
                            and (sent or not known_not_sent)
                        ):
                            self.feedback["deliveries"] += 1
                            if self.feedback.get("run_id") is not None:
                                self.runs[self.feedback["run_id"]]["delivered_queries"] += int(sent)
                        self.report["deliveries"].append(projected)
                        self._save()

            model.query = observed_query
            binding = (
                environment.bind_behavior_plan_evidence(self.evidence)
                if self.diagnostic is not None
                else _nothing()
            )
            try:
                with binding:
                    yield self
            finally:
                model.query = query
                self._save()

    @contextmanager
    def action_context(self, event, action, args, kwargs):
        before = after = None
        if self.diagnostic is not None:
            try:
                before = read_script(self.scratch)
            except (OSError, ValueError, UnicodeError):
                pass
        try:
            yield
        finally:
            if self.diagnostic is not None:
                try:
                    after = read_script(self.scratch)
                except (OSError, ValueError, UnicodeError):
                    pass
                if after is not None and (before is None or before["sha256"] != after["sha256"]):
                    self.origin = dict(execution_id=event["id"], sha256=after["sha256"])

    def _feedback_error(self, error, source):
        if self.error_sent:
            return
        self.error_sent = True
        self.report["status"] = "scenario_unavailable;ordinary_continues"
        self.feedback = dict(
            id="error",
            run_id=None,
            deliveries=0,
            source=copy.deepcopy(source),
            script_sha=None,
            observation=dict(
                status="scenario_unavailable;ordinary_continues",
                error=type(error).__name__ + ": " + str(error)[:220],
                authority="scenario_error;not_candidate_failure",
            ),
        )
        self._record("scenario_error", observation=self.feedback["observation"])

    def before_query(self, messages):
        if self.diagnostic is None or not self.guidance_delivered:
            return
        self._check_task()
        current = self._snapshot()
        if self.feedback is not None and self.feedback["source"] != current:
            if self.feedback["run_id"] is not None:
                old = self.runs[self.feedback["run_id"]]
                old["current"] = False
                old["currency"] = "historical_after_observed_source_change"
                self._artifact("run-" + str(old["id"]) + ".json", old)
            self._record("result_historical", feedback_id=self.feedback["id"])
            self.feedback = None
        if len(self.runs) >= self.policy.max_runs:
            return
        started = time.monotonic()
        try:
            value = read_script(self.scratch)
            if value is None:
                return
            if self.origin is None or self.origin["sha256"] != value["sha256"]:
                raise ValueError("scenario was not observed in a real ordinary task action")
            event = self.executions[self.origin["execution_id"]]
            if (
                event.get("message_ids") is None
                or not event["observer_complete"]
                or event["error"] is not None
            ):
                return
            if value["sha256"] not in self.versions:
                if len(self.versions) >= self.policy.max_script_versions:
                    return
                self.versions.append(value["sha256"])
                (self.evidence / ("plan-" + str(len(self.versions) - 1) + ".py")).write_text(
                    value["text"], encoding="utf8"
                )
            plan = parse_script(value, self.public_task)
            if (
                self.feedback is not None
                and self.feedback["script_sha"] is not None
                and self.feedback["script_sha"] != value["sha256"]
            ):
                old = self.runs[self.feedback["run_id"]]
                old["current"] = False
                old["currency"] = "historical_after_script_change"
                self._artifact("run-" + str(old["id"]) + ".json", old)
                self.feedback = None
            if (
                any(v["kind"] not in ("file", "missing") for v in current.values())
                or self.unattributed
            ):
                raise ValueError(
                    "current observed implementation version is unknown or unattributed"
                )
            if self.runs and not self.runs[-1]["delivered_queries"]:
                return
            key = digest(
                dump(
                    dict(
                        script=value["sha256"],
                        sources=current,
                        environment=self.environment_identity,
                    )
                )
            )
            if key in self.executed:
                return
            remaining = self.deadline - time.monotonic()
            allocation = (
                min(self.policy.total_seconds, self.request.budget.wall_seconds * 0.1) - self.spent
            )
            reserve = self.request.budget.request_timeout
            if (
                allocation <= 0
                or remaining <= reserve + 1
                or self.request.transport_adapter.remaining_physical_calls() < 1
            ):
                return
            seconds = min(self.policy.run_seconds, allocation, remaining - reserve)
            if seconds <= 0:
                return
            self.executed.add(key)
            result = self.diagnostic.run(
                executable_script(plan), absolute_deadline=time.monotonic() + seconds
            )
            self._check_task()
            after = self._snapshot()
            number = len(self.runs)
            after_script = read_script(self.scratch)
            record = dict(
                id=number,
                key=key,
                script_sha256=value["sha256"],
                public_basis=plan["public_basis"],
                observed_sources=current,
                after_observed_sources=after,
                environment_identity=self.environment_identity,
                unobserved_dependencies="unknown",
                current=after == current
                and after_script is not None
                and after_script["sha256"] == value["sha256"],
                result=result,
                delivered_queries=0,
                authority="model_scene_and_expectations_unverified;execution_is_not_acceptance",
            )
            self.runs.append(record)
            self.report["status"] = "scenario_executed;ordinary_coding_continues"
            self._record("scenario_execution", **copy.deepcopy(record))
            self._artifact("run-" + str(number) + ".json", record)
            if record["current"]:
                self.feedback = dict(
                    id=key,
                    run_id=number,
                    deliveries=0,
                    source=current,
                    script_sha=value["sha256"],
                    observation=record,
                )
        except Exception as error:
            self._check_task()
            self._feedback_error(error, current)
        finally:
            self.spent += time.monotonic() - started
            self._save()

    def project(self, messages):
        original = super().project(messages)
        if self.diagnostic is None:
            return original
        current = self._snapshot()
        content = None
        kind = None
        if self.guidance_pending and not self.executions:
            content = GUIDANCE
            kind = "guidance"
        elif (
            self.feedback is not None
            and self.feedback["deliveries"] < 2
            and self.feedback["source"] == current
        ):
            try:
                version = read_script(self.scratch)
            except (OSError, ValueError, UnicodeError):
                version = None
            if self.feedback["script_sha"] is not None and (
                version is None or version["sha256"] != self.feedback["script_sha"]
            ):
                return original
            observation = self.feedback["observation"]
            run_id = self.feedback["run_id"]
            if run_id is None:
                summary = observation
            else:
                process = observation["result"]["process"]
                output = process.get("output", "")
                summary = dict(
                    run_id=run_id,
                    public_basis=observation["public_basis"],
                    script_sha256=observation["script_sha256"],
                    observed_source_version=digest(dump(current)),
                    currency="current_for_observed_paths;unobserved_dependencies_unknown",
                    exit_code=process["exit_code"],
                    timed_out=process["timed_out"],
                    truncated=process["truncated"],
                    output_preview=output[:240],
                    preview_truncated=len(output) > 240,
                    full_observation=MODEL_EVIDENCE + "/run-" + str(run_id) + ".json",
                    authority=observation["authority"],
                )
            content = HEADER + dump(summary)
            kind = "observation"
        if content is None:
            return original
        prepared = [dict(m) for m in self.prepare(original)]
        prepared.insert(
            2 if kind == "guidance" else len(prepared), dict(role="user", content=content)
        )
        if (
            self._count([dict(role="user", content=content)]) > self.policy.context_tokens
            or self._count(prepared) > self.limit
        ):
            if kind == "guidance":
                self.guidance_pending = False
            self._record(
                "context_not_added",
                kind_of_context=kind,
                reason="complete original messages and bounded addition do not fit",
            )
            return original
        observations = self.request.transport_adapter.request_observations()
        self.projected = dict(
            kind=kind,
            attempt_marker=max((r["attempt"] for r in observations), default=0),
            input_sha256=digest(dump(prepared)),
            canonical_history_changed=False,
            feedback_id=self.feedback["id"] if kind == "observation" else None,
        )
        return prepared


@contextmanager
def _nothing():
    yield
