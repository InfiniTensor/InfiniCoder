"""Actual action effects for the next ordinary Native decision;default off."""

from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import time

from infinicoder.memory.action_effect_memory import (
    EffectSequence,
    VersionLabels,
    archive_effect,
    make_effect_record,
    relation,
    render_effect,
    source_state,
)
from infinicoder.memory.action_source_memory import _namespace, read_snapshot
from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.runtime.native_plain import NativePlainObserver
from infinicoder.memory.source.observed_source_reads import literal_read
from infinicoder.memory.task_history_archive import TaskHistoryArchive, MODEL_ROOT
from infinicoder.planning.task_source_plan import MappedWorkspaceSourcePlanReader
from infinicoder.memory.task_state_checkpoint import digest, dump


class EffectBudgetExpired(TimeoutError):
    pass


class NativeActionEffectMemory(NativePlainObserver):
    identity = "verified_action_effect_memory_v1"

    def __init__(
        self,
        count_messages,
        *,
        tokenizer_identity,
        implementation_paths=(),
        reader_factory=None,
        workspace_aliases=("/workspace",),
    ):
        super().__init__(
            count_messages,
            tokenizer_identity=tokenizer_identity,
            implementation_paths=implementation_paths,
        )
        if reader_factory is not None and not callable(reader_factory):
            raise ValueError("explicit source reader factory required")
        self.reader_factory = reader_factory
        self.workspace_aliases = tuple(workspace_aliases)
        self.bound_once = False

    def validate(self, request):
        super().validate(request)
        if request.context_adapter is not self:
            raise UnsupportedCore("action effects must own the plain context")
        if any(
            getattr(request, name) is not None
            for name in (
                "planning_adapter",
                "repair_adapter",
                "candidate_search",
                "implementation_proposal",
            )
        ):
            raise UnsupportedCore("action effects are a separate ordinary single-run profile")
        if getattr(request.transport_adapter, "delivery_policy", None) is not None:
            raise UnsupportedCore("action effects use the unchanged common legacy transport")
        if not callable(getattr(request.transport_adapter, "request_observations", None)):
            raise UnsupportedCore("actual physical request observations required")

    def _remaining(self):
        left = super()._remaining()
        if getattr(self, "source_deadline", None) is not None:
            left = min(left, self.source_deadline - time.monotonic())
        if left <= 0:
            raise EffectBudgetExpired("action-effect source processing budget exhausted")
        return left

    @contextmanager
    def _source_stage(self):
        if self.source_deadline is not None:
            raise ValueError("nested source stage")
        start = time.monotonic()
        allowance = min(2, (self.deadline - start) / 20, self.source_allowance - self.source_spent)
        if allowance <= 0:
            raise EffectBudgetExpired("no remaining action-effect source allocation")
        self.source_deadline = start + allowance
        try:
            yield
        finally:
            self.source_spent += time.monotonic() - start
            self.source_deadline = None

    def _save(self):
        self.report["source_processing_seconds"] = self.source_spent
        path = self.state / "native-action-effects.json"
        temporary = path.with_suffix(".next")
        try:
            temporary.write_text(dump(self.report), encoding="utf8")
            temporary.replace(path)
        except OSError as error:
            self.current_view = None
            self.prepared_for = None
            self.report["persistence_error"] = type(error).__name__

    def _unknown(self, phase, error):
        self.report["records"].append(
            dict(kind="effect_observation_unknown", phase=phase, error=type(error).__name__)
        )
        self._save_event(
            "effect_observation_unknown", dict(phase=phase, error=type(error).__name__)
        )

    def _reader(self):
        if self.reader is not None:
            if _namespace(self.reader.namespace) != self.source_namespace:
                raise ValueError("actual source namespace changed")
            return self.reader
        if self.reader_factory is None:
            raise ValueError("N_A;no caller-bound source capability")
        self.reader = self.reader_factory(
            self.request,
            self.index,
            self.environment,
            count_output=lambda text: self._count([dict(role="user", content=text)]),
            remaining=self._remaining,
        )
        self.source_namespace = _namespace(self.reader.namespace)
        self.labels = VersionLabels(self.reader.namespace)
        self.mapped = isinstance(self.reader, MappedWorkspaceSourcePlanReader)
        return self.reader

    def _probe(self, path):
        self._remaining()
        try:
            return dict(kind="file", sha256=self._reader().current_version(path))
        except FileNotFoundError:
            return dict(kind="missing")
        except (OSError, ValueError, UnicodeError):
            return dict(kind="unknown")

    def _finalize_info(self):
        state = getattr(self, "state", None)
        if state is None:
            return
        path = state / "native-memory-info.json"
        try:
            if not path.exists():
                return
            info = json.loads(path.read_bytes())
            if not isinstance(info, dict):
                raise ValueError("native memory info must be an object")
            info.update(
                projection="Original canonical history plus admitted controller action effects",
                canonical_history_changed=False,
                actual_model_projection=any(
                    d["model_response_received"] for d in self.report["deliveries"]
                ),
                action_effect_memory=True,
                automatic_compression=False,
                auxiliary_model_calls=0,
            )
            temporary = path.with_suffix(".next")
            temporary.write_text(dump(info), encoding="utf8")
            temporary.replace(path)
        except (OSError, ValueError, TypeError) as error:
            # The journal is already closed. A display-record failure must not
            # replace the original SDK error or task cancellation.
            self.report["info_finalization_error"] = type(error).__name__
            self._save()

    @contextmanager
    def bind(self, request, model, environment, deadline):
        if self.bound_once:
            raise UnsupportedCore("action effects cannot resume an existing journal")
        self.bound_once = True
        self.request = request
        self.environment = environment
        self.reader = None
        self.source_namespace = None
        self.labels = None
        self.mapped = False
        self.source_deadline = None
        self.source_spent = 0.0
        self.source_allowance = min(10, request.budget.wall_seconds * 0.02)
        self.active_paths = []
        self.returned = {}
        self.pending = {}
        self.completed_ids = set()
        self.sequence = EffectSequence()
        self.latest = None
        self.repetitions = 0
        self.current_view = None
        self.prepared_for = None
        self.pending_projection = None
        self.last_effect_ref = None
        self.archive = self.archive_context = None
        self.report = dict(
            identity=self.identity,
            status="dormant",
            records=[],
            deliveries=[],
            canonical_history_changed=False,
            original_task_unchanged=True,
            private_final_used=False,
            auxiliary_model_calls=0,
            automatic_checks=0,
            source_processing_budget_seconds=self.source_allowance,
        )
        try:
            with super().bind(request, model, environment, deadline):
                self.namespace = digest(
                    dict(
                        phase="plain",
                        state=str(self.state.resolve()),
                        workspace=str(self.workspace),
                        task=request.spec.task,
                    )
                )
                query = model.query

                def ordinary_query(messages, **kwargs):
                    self.pending_projection = None
                    try:
                        return query(messages, **kwargs)
                    finally:
                        if self.pending_projection is not None:
                            requests = [
                                r
                                for r in request.transport_adapter.request_observations()
                                if r["attempt"] > self.pending_projection["attempt_marker"]
                            ]
                            delivery = dict(
                                self.pending_projection,
                                HTTP_dispatched=any(r["dispatch_attempted"] for r in requests),
                                model_response_received=any(
                                    r["dispatch_attempted"] and r["status"] == "response_received"
                                    for r in requests
                                ),
                                requests=[
                                    dict(
                                        attempt=r["attempt"],
                                        dispatch_attempted=r["dispatch_attempted"],
                                        status=r["status"],
                                    )
                                    for r in requests
                                ],
                            )
                            self.report["deliveries"].append(delivery)
                            self._save_event("effect_delivery", delivery)
                            self.report["actual_HTTP_views"] = sum(
                                d["HTTP_dispatched"] for d in self.report["deliveries"]
                            )
                            self.report["actual_response_views"] = sum(
                                d["model_response_received"] for d in self.report["deliveries"]
                            )
                            self._save()

                model.query = ordinary_query
                try:
                    yield self
                finally:
                    model.query = query
                    try:
                        if self.archive is not None:
                            try:
                                self.archive.sync()
                            except (TimeoutError, OSError, ValueError) as error:
                                self._unknown("final_archive_sync", error)
                    finally:
                        if self.archive_context is not None:
                            self.archive_context.__exit__(None, None, None)
                        self._save()
        finally:
            self._finalize_info()

    @contextmanager
    def action_context(self, event, action, args, kwargs):
        event["effect_before"] = {}
        try:
            with self._source_stage():
                self._reader()
                # Existing CPU hashes are reused. Mapped workspaces require
                # the actual namespace probe for the two observed active paths.
                for path in self.active_paths:
                    if self.mapped or path not in self.paths:
                        event["effect_before"][path] = self._probe(path)
        except Exception as error:
            self._unknown("before_action", error)
        yield

    def execution_returned(self, event, result):
        output = result.get("output")
        complete = (
            isinstance(output, str)
            and type(result.get("returncode")) is int
            and not result.get("exception_info")
            and not result.get("truncated")
            and not result.get("timed_out")
        )
        self.returned[event["id"]] = dict(
            complete=complete,
            raw_sha256=hashlib.sha256(output.encode()).hexdigest()
            if isinstance(output, str)
            else None,
            output=output if complete and len(output.encode()) <= 64 * 1024 else None,
        )

    def execution_completed(self, event):
        returned = self.returned.pop(
            event["id"], dict(complete=False, raw_sha256=None, output=None)
        )
        if event.get("error") is not None:
            self.pending[event["id"]] = dict(
                paths=[],
                before={},
                after={},
                feedback=dict(complete=False),
                confirmed_read_path=None,
                namespace=json.loads(self.source_namespace) if self.source_namespace else None,
            )
            return
        try:
            with self._source_stage():
                reader = self._reader()
                before = {}
                after = {}
                confirmed = None
                changed = (
                    [
                        p
                        for p in event.get("changed", ())
                        if relation(event["before"].get(p), event["after"].get(p)) == "changed"
                    ]
                    if not self.mapped
                    else []
                )
                requested = (
                    literal_read(event.get("command", ""), self.workspace_aliases)
                    if returned["complete"]
                    else None
                )
                # A path proposed in a command is not admitted before the real
                # action and its exact returned bytes have been observed.
                if requested and len(requested) == 1 and returned["output"] is not None:
                    path, first, last = requested[0]
                    try:
                        snapshot = read_snapshot(reader, path, self._remaining)
                        fragment = b"".join(
                            snapshot.raw.splitlines(keepends=True)[first - 1 : last]
                        ).decode("utf8")
                        if fragment == returned["output"]:
                            confirmed = path
                            after[path] = dict(kind="file", sha256=snapshot.version)
                    except (OSError, ValueError, UnicodeError):
                        pass
                selected = list(
                    dict.fromkeys(changed + ([confirmed] if confirmed else []) + self.active_paths)
                )[:2]
                for path in selected:
                    if self.mapped or path not in self.paths:
                        before[path] = source_state(event.get("effect_before", {}).get(path))
                        if path not in after:
                            after[path] = self._probe(path)
                    else:
                        before[path] = source_state(event["before"].get(path))
                        if path not in after:
                            after[path] = source_state(event["after"].get(path))
                self.pending[event["id"]] = dict(
                    paths=selected,
                    before=before,
                    after=after,
                    feedback={k: v for k, v in returned.items() if k != "output"},
                    confirmed_read_path=confirmed,
                    namespace=copy.deepcopy(reader.namespace),
                )
                self._save_event(
                    "action_effect_pending",
                    dict(
                        execution_id=event["id"],
                        paths=selected,
                        before=before,
                        after=after,
                        namespace=reader.namespace,
                        feedback_sha256=returned["raw_sha256"],
                    ),
                )
        except Exception as error:
            self._unknown("completed_action", error)
            self.pending[event["id"]] = dict(
                paths=[],
                before={},
                after={},
                feedback=dict(complete=False),
                confirmed_read_path=None,
                namespace=json.loads(self.source_namespace) if self.source_namespace else None,
            )

    def _sync(self, messages):
        prepared = super()._sync(messages)
        if not any(
            e["id"] not in self.completed_ids and e.get("message_ids") and e["id"] in self.pending
            for e in self.executions
        ):
            return prepared
        started = time.monotonic()
        allowance = min(
            2, (self.deadline - started) / 20, self.source_allowance - self.source_spent
        )
        if allowance <= 0:
            self.latest = None
            self.current_view = None
            self.prepared_for = None
            return prepared
        self.source_deadline = started + allowance
        try:
            for event in self.executions:
                if (
                    event["id"] in self.completed_ids
                    or not event.get("message_ids")
                    or event["id"] not in self.pending
                ):
                    continue
                self.completed_ids.add(event["id"])
                pending = self.pending.pop(event["id"])
                self._remaining()
                self.current_view = None
                self.prepared_for = None
                if not pending["namespace"] or self.source_spent >= self.source_allowance:
                    self.latest = None
                    continue
                ids = event["message_ids"]
                raw = messages[ids[-1]]
                extra = raw.get("extra", {})
                output = extra.get("raw_output")
                exact = (
                    isinstance(output, str)
                    and hashlib.sha256(output.encode()).hexdigest()
                    == pending["feedback"].get("raw_sha256")
                    and extra.get("returncode") == event.get("returncode")
                )
                feedback = dict(pending["feedback"], message_sha256=digest(prepared[ids[-1]]))
                record = make_effect_record(
                    event,
                    pending["namespace"],
                    pending["paths"],
                    pending["before"],
                    pending["after"],
                    feedback,
                    linked_exactly=exact,
                )
                self.repetitions = self.sequence.add(record)
                self.latest = record
                stored = dict(
                    effect=record.payload,
                    same_observation_run=self.repetitions,
                    previous_effect_ref=self.last_effect_ref,
                )
                ref = "source-view:" + digest(stored)[:32]
                archive_effect(self.reader, ref, stored)
                self.last_effect_ref = ref
                self._remaining()
                if exact:
                    fresh = [
                        r["path"] for r in record.payload["paths"] if r["relation"] == "changed"
                    ]
                    if pending["confirmed_read_path"]:
                        fresh.append(pending["confirmed_read_path"])
                    self.active_paths = list(dict.fromkeys(fresh + self.active_paths))[:2]
                self._save_event(
                    "action_effect_linked",
                    dict(
                        effect=record.payload, same_observation_run=self.repetitions, effect_ref=ref
                    ),
                )
                self.report["records"].append(
                    dict(
                        kind="effect_linked",
                        execution_id=record.execution_id,
                        relation=record.payload["relation"],
                        same_observation_run=self.repetitions,
                        feedback_exactly_linked=exact,
                    )
                )
            self._flush_events()
        except Exception as error:
            self.latest = None
            self.current_view = None
            self.prepared_for = None
            self._unknown("link_effect", error)
        finally:
            self.source_spent += time.monotonic() - started
            self.source_deadline = None
        return prepared

    def _checks(self, paths, current):
        result = {p: dict(status="not_observed") for p in paths}
        for check in self.checks:
            eid = check.get("execution_id")
            if (
                eid is None
                or not check.get("finished")
                or not check.get("same_execution")
                or not self.executions[eid].get("message_ids")
            ):
                continue
            for path in paths:
                before = source_state(check["before"].get(path))
                after = source_state(check["after"].get(path))
                now = source_state(current.get(path))
                state = (
                    "current"
                    if before["kind"] == after["kind"] == now["kind"] == "file"
                    and before == after == now
                    else "stale"
                    if after["kind"] == now["kind"] == "file" and after != now
                    else "unknown"
                )
                result[path] = dict(
                    status=state,
                    check_id=check["id"],
                    scope=check["scope"],
                    coverage=check["coverage"],
                    before=before,
                    after=after,
                    interpretation="Version relation only;coverage is caller-declared,not proof of full task correctness",
                )
        return result

    def _activate_archive(self):
        if self.archive is not None:
            return
        if not callable(getattr(self.environment, "bind_history_archive", None)):
            raise ValueError("N_A;no actual readonly history binding")
        directory = self.state / "plain"
        directory.mkdir(exist_ok=True)
        export = directory / "history-export"
        archive = TaskHistoryArchive(
            self.index.db,
            export,
            remaining=self._remaining,
            phase="plain",
            namespace=self.namespace,
        )
        context = self.environment.bind_history_archive(
            export, phase="plain", namespace=self.namespace
        )
        declared = context.__enter__()
        if (
            not isinstance(declared, dict)
            or declared.get("phase") != "plain"
            or declared.get("namespace") != self.namespace
            or declared.get("model_root") != MODEL_ROOT
            or Path(declared.get("directory", "")).resolve() != export.resolve()
        ):
            context.__exit__(None, None, None)
            raise ValueError("readonly history namespace differs")
        self.archive, self.archive_context = archive, context

    def before_query(self, messages):
        self.current_view = None
        self.prepared_for = None
        self.report["status"] = "dormant_current_event"
        record = self.latest
        if record is None or not (record.payload["relation"] == "changed" or self.repetitions >= 2):
            return
        try:
            with self._source_stage():
                self._reader()
                prepared = self.prepare(messages)
                paths = [r["path"] for r in record.payload["paths"]]
                current = {p: self._probe(p) for p in paths}
                checks = self._checks(paths, current)
                text, ref, payload = render_effect(
                    record,
                    self.repetitions,
                    current,
                    checks,
                    self.labels,
                    effect_record_ref=self.last_effect_ref,
                )
                source_tokens = self._count([dict(role="user", content=text)])
                limit = min(256, self.limit // 12)
                if source_tokens > limit:
                    self.report["status"] = "capacity_N_A;effect view exceeds source budget"
                    return
                projected = prepared + [dict(role="user", content=text)]
                tokens = self._count(projected)
                if tokens > self.limit:
                    self.report["status"] = (
                        "capacity_N_A;complete original input and effect do not fit"
                    )
                    return
                archive_effect(self.reader, ref, payload)
                self._activate_archive()
                self.archive.sync()
                self.current_view = dict(role="user", content=text)
                self.prepared_for = digest(prepared)
                self.view_info = dict(
                    effect_ref=ref,
                    effect_execution_id=record.execution_id,
                    same_observation_run=self.repetitions,
                    effect_tokens=source_tokens,
                    effect_token_limit=limit,
                    projected_tokens=tokens,
                    input_limit=self.limit,
                )
                self.report["status"] = "prepared"
                self.report["last_view"] = dict(self.view_info)
        except Exception as error:
            self.current_view = None
            self.prepared_for = None
            self.report["status"] = "source_N_A;ordinary_input_unchanged"
            self._unknown("before_query", error)
        finally:
            self._save()

    def project(self, messages):
        prepared = self._sync(messages)
        if self.current_view is None or digest(prepared) != self.prepared_for:
            return super().project(messages)
        projected = prepared + [copy.deepcopy(self.current_view)]
        accounting = dict(
            self.view_info,
            canonical_history_changed=False,
            model_input_sha256=digest(projected),
            canonical_input_sha256=digest(prepared),
        )
        with self.index.db:
            self.index.db.execute(
                "INSERT INTO native_projections(messages,accounting) VALUES(?,?)",
                (dump(projected), dump(accounting)),
            )
        marker = max(
            (r["attempt"] for r in self.request.transport_adapter.request_observations()), default=0
        )
        self.pending_projection = dict(accounting, attempt_marker=marker)
        return projected
