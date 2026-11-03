"""Once,under the original budget,organize actual task history for the same SDK."""

from contextlib import contextmanager
import copy
import json
from pathlib import Path
import time

from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.memory.native_memory import MemoryCapacityError
from infinicoder.runtime.native_plain import NativePlainObserver
from infinicoder.memory.task_evidence_context import TaskEvidenceContext
from infinicoder.memory.task_history_archive import TaskHistoryArchive, MODEL_ROOT
from infinicoder.memory.task_state_checkpoint import (
    UnverifiedTaskMemo,
    accept_memo,
    checkpoint_view,
    digest,
    dump,
    prepare_checkpoint,
)


class NativeTaskStateConsolidation(NativePlainObserver):
    identity = "task_state_consolidation_v1"

    def __init__(self, count_messages, *, tokenizer_identity, implementation_paths=()):
        super().__init__(
            count_messages,
            tokenizer_identity=tokenizer_identity,
            implementation_paths=implementation_paths,
        )
        self.auxiliary = None
        self.bound_once = False

    def validate(self, request):
        super().validate(request)
        if request.context_adapter is not self:
            raise UnsupportedCore("task state must own the plain context")
        if any(
            getattr(request, name) is not None
            for name in (
                "planning_adapter",
                "repair_adapter",
                "candidate_search",
                "implementation_proposal",
            )
        ):
            raise UnsupportedCore(
                "task-state consolidation is a separate ordinary single-run profile"
            )
        transport = request.transport_adapter
        if getattr(transport, "delivery_policy", None) is not None:
            raise UnsupportedCore("task state uses the common legacy transport")
        if any(
            not callable(getattr(transport, name, None))
            for name in ("limit_request", "remaining_physical_calls", "request_observations")
        ):
            raise UnsupportedCore("task state requires the shared physical call ledger")

    @contextmanager
    def bind_auxiliary_calls(self, budget):
        if self.auxiliary is not None:
            raise UnsupportedCore("one attached auxiliary budget only")
        self.auxiliary = budget
        try:
            yield
        finally:
            self._save()

    def _check_task(self):
        if time.monotonic() >= self.deadline:
            from infinicoder.runtime.native_run import CoreDeadline

            raise CoreDeadline("original task deadline exhausted during state consolidation")

    def _count(self, messages):
        self._remaining()
        key = digest(self.prepare(messages))
        if key not in self.token_cache:
            if len(self.token_cache) >= 256:
                self.token_cache.pop(next(iter(self.token_cache)))
            self.token_cache[key] = super()._count(messages)
        return self.token_cache[key]

    def _save(self):
        self.report["auxiliary_model_calls"] = self.auxiliary.call_units if self.auxiliary else 0
        path = self.state / "native-task-state-consolidation.json"
        temporary = path.with_suffix(".next")
        temporary.write_text(dump(self.report), encoding="utf8")
        temporary.replace(path)

    def _record(self, kind, **data):
        self.report["records"].append(dict(kind=kind, **copy.deepcopy(data)))
        self._save_event(kind, copy.deepcopy(data))
        self._flush_events()

    def _close_archive(self):
        if self.archive_context is not None:
            context = self.archive_context
            self.archive_context = None
            context.__exit__(None, None, None)

    def _disable(self, reason):
        self.used = True
        self.checkpoint = None
        self.memo = None
        self.report["status"] = "disabled;ordinary_input_unchanged"
        self._record("checkpoint_disabled", reason=reason)
        self._close_archive()
        self._save()

    @contextmanager
    def bind(self, request, model, environment, deadline):
        if self.bound_once:
            raise UnsupportedCore("task state cannot resume or repeat a previous journal")
        try:
            with self._bind_memory(request, model, environment, deadline):
                yield self
        finally:
            state = getattr(self, "state", None)
            if state is not None:
                path = state / "native-memory-info.json"
                if path.is_file():
                    try:
                        info = json.loads(path.read_bytes())
                        active = getattr(self, "checkpoint", None) is not None
                        info.update(
                            projection="Canonical preserved;fixed checkpoint model view"
                            if active
                            else "Original SDK model input unchanged",
                            actual_model_projection=active,
                            canonical_history_changed=False,
                            memo_authority="model_generated_unverified",
                        )
                        path.write_text(dump(info), encoding="utf8")
                    except (OSError, ValueError):
                        pass

    @contextmanager
    def _bind_memory(self, request, model, environment, deadline):
        if self.bound_once:
            raise UnsupportedCore("task state cannot resume or repeat a previous journal")
        self.bound_once = True
        self.request = request
        self.environment = environment
        self.used = False
        self.checkpoint = self.memo = self.archive = self.archive_context = None
        self.token_cache = {}
        self.pending_projection = None
        with super().bind(request, model, environment, deadline):
            self.task_state = TaskEvidenceContext(self.index, remaining=self._remaining)
            self.namespace = digest(
                dict(
                    phase="plain",
                    state=str(self.state.resolve()),
                    workspace=str(self.workspace),
                    task=request.spec.task,
                )
            )
            self.report = dict(
                identity=self.identity,
                status="dormant",
                records=[],
                deliveries=[],
                canonical_history_changed=False,
                original_task_unchanged=True,
                private_final_used=False,
                planner_enabled=False,
                automatic_checks=0,
                max_consolidations=1,
            )
            if not callable(getattr(environment, "bind_history_archive", None)):
                self._disable("N_A;no actual task-private readonly history capability")
            query = model.query

            def ordinary_query(messages, **kwargs):
                self.pending_projection = None
                try:
                    return query(messages, **kwargs)
                finally:
                    if self.pending_projection is not None:
                        rows = [
                            r
                            for r in request.transport_adapter.request_observations()
                            if r["attempt"] > self.pending_projection["attempt_marker"]
                        ]
                        delivery = dict(
                            self.pending_projection,
                            HTTP_dispatched=any(r["dispatch_attempted"] for r in rows),
                            requests=[
                                dict(
                                    attempt=r["attempt"],
                                    status=r["status"],
                                    dispatch_attempted=r["dispatch_attempted"],
                                )
                                for r in rows
                            ],
                        )
                        self.report["deliveries"].append(delivery)
                        self._record("checkpoint_delivery", **delivery)
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
                            self.report["archive_final_sync_error"] = type(error).__name__
                finally:
                    self._close_archive()
                    self._save()

    def _activate_archive(self):
        directory = self.state / "plain"
        if directory.exists() and any(p.name != "history-export" for p in directory.iterdir()):
            raise UnsupportedCore("predeclared history parent contains unrelated state")
        directory.mkdir(exist_ok=True)
        export = directory / "history-export"
        self.archive = TaskHistoryArchive(
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
            raise UnsupportedCore("history namespace differs from actual readonly binding")
        self.archive_context = context
        self.archive.sync()

    def _selection(self, prepared, references):
        mandatory, episode = self._episode_selection(prepared)
        if episode["incomplete"]:
            raise MemoryCapacityError("unattributed current edit/check evidence cannot be omitted")
        prefix, blocks = self._history_blocks(prepared)
        current = self._snapshot()
        protected = set(prefix) | {i for block in blocks[-2:] for i in block}
        linked = {}
        for event in self.executions:
            ids = event.get("message_ids")
            block = next((b for b in blocks if ids and set(ids).issubset(b)), None)
            if block is not None:
                linked[tuple(block)] = event
        # Unlinked format/length/error blocks stay whole. They cannot acquire a
        # fabricated action or source origin just because a memo mentions them.
        for block in blocks:
            event = linked.get(tuple(block))
            if (
                event is None
                or not event.get("observer_complete")
                or event.get("error") is not None
            ):
                protected.update(block)
        for check in self.checks:
            eid = check.get("execution_id")
            if eid is None:
                continue
            ids = self.executions[eid].get("message_ids")
            if not ids:
                raise MemoryCapacityError(
                    "actual public check is not linked to a complete feedback block"
                )
            protected.update(i for block in blocks if set(ids) & set(block) for i in block)
        mandatory.update(protected)
        receipts = []
        for block, event in ((list(b), e) for b, e in linked.items()):
            changed = event.get("changed", [])
            if not changed:
                continue
            mandatory.update(block)
            if (
                set(block) & protected
                or event.get("checks")
                or not event.get("observer_complete")
                or event.get("error") is not None
                or type(event.get("returncode")) is not int
            ):
                continue
            if any(
                p in self.unattributed
                or event["after"][p] != current[p]
                or self.known[p] != current[p]
                or current[p].get("kind") == "unknown"
                for p in changed
            ):
                continue
            ref = references["executions"].get(event["id"])
            if not ref:
                continue
            receipts.append(
                dict(
                    kind="observed_edit_receipt",
                    execution_id=event["id"],
                    message_ids=block,
                    message_sha256={str(i): digest(prepared[i]) for i in block},
                    observation_ref=ref,
                    changed_paths=changed,
                    before={p: event["before"][p] for p in changed},
                    after={p: event["after"][p] for p in changed},
                    returncode=event["returncode"],
                    authority="Historical action and current declared versions only;not source text,permission or correctness",
                )
            )
            mandatory.difference_update(block)
        return mandatory, receipts, current

    def before_query(self, messages):
        if self.used or self.auxiliary is None:
            return
        self._check_task()
        prepared = self.prepare(messages)
        _, blocks = self._history_blocks(prepared)
        linked = [
            e
            for e in self.executions
            if e.get("message_ids") and e["observer_complete"] and e["error"] is None
        ]
        if len(linked) < 4:
            return
        original_tokens = self._count(prepared)
        if original_tokens < self.limit * 0.65:
            return
        self.used = True
        remaining = self.deadline - time.monotonic()
        steps = self.auxiliary.agent.config.step_limit - self.auxiliary.agent.n_calls
        if (
            original_tokens > self.limit
            or remaining < 60
            or steps < 4
            or self.request.transport_adapter.remaining_physical_calls() < 4
        ):
            self._disable(
                "N_A;past original capacity or insufficient original wall/step/physical budget"
            )
            return
        try:
            self.report["status"] = "preparing"
            references = self.task_state.capture_observations(
                prepared, blocks, copy.deepcopy(self.executions), copy.deepcopy(self.checks)
            )
            mandatory, receipts, current = self._selection(prepared, references)
            output_tokens = 384 if self.request.model_profile.context_limit <= 4096 else 768
            plan = prepare_checkpoint(
                prepared,
                linked,
                self.auxiliary.public_task,
                (),
                dict(
                    count=self._count,
                    input_limit=self.limit,
                    mandatory=mandatory,
                    observation_refs=references["executions"],
                    receipts=receipts,
                    source_snapshot=current,
                    namespace=self.namespace,
                ),
            )
            if plan is None:
                self._disable("N_A;complete auxiliary task and records do not fit")
                return
            provisional = UnverifiedTaskMemo("", digest(plan.evidence.records), (), ())
            view, _ = checkpoint_view(plan, provisional, prepared, current)
            if self._count(view) + output_tokens * 2 + 64 > self.limit:
                self._disable("N_A;mandatory view and memo allowance do not fit")
                return
            self._activate_archive()
            self.report["status"] = "auxiliary_requested"
            (self.state / "task-state-auxiliary-input.json").write_text(
                dump(plan.auxiliary_messages), encoding="utf8"
            )
            (self.state / "task-state-evidence-index.json").write_text(
                plan.evidence.records_json, encoding="utf8"
            )
            self._record(
                "checkpoint_auxiliary_requested",
                cut=plan.evidence.cut,
                namespace=self.namespace,
                output_tokens=output_tokens,
            )
            response = self.auxiliary.query_once(
                plan.auxiliary_messages,
                output_tokens=output_tokens,
                absolute_deadline=min(
                    self.deadline,
                    time.monotonic() + min(self.request.budget.request_timeout, 90, remaining / 4),
                ),
            )
            (self.state / "task-state-auxiliary-response.json").write_text(
                dump(response), encoding="utf8"
            )
            memo = accept_memo(response, plan.evidence)
            view, accounting = checkpoint_view(plan, memo, prepared, self._snapshot())
            tokens = self._count(view)
            if tokens > self.limit or tokens > original_tokens * 0.8:
                self._disable(
                    "complete memo view does not fit or does not reduce original input by20%"
                )
                return
            self.checkpoint, self.memo = plan, memo
            self.report.update(
                status="active",
                cut=plan.evidence.cut,
                namespace=self.namespace,
                activation_original_tokens=original_tokens,
                activation_projected_tokens=tokens,
                memo_checkpoint_sha256=memo.checkpoint_sha256,
                memo_unverified=True,
            )
            self._record(
                "checkpoint_activated",
                **accounting,
                original_tokens=original_tokens,
                projected_tokens=tokens,
            )
            self._save()
        except Exception as error:
            self._check_task()
            self._disable(type(error).__name__ + ": " + str(error)[:400])

    def project(self, messages):
        if self.checkpoint is None:
            return super().project(messages)
        prepared = self._sync(messages)
        self.archive.sync()
        projected, accounting = checkpoint_view(
            self.checkpoint, self.memo, prepared, self._snapshot()
        )
        tokens = self._count(projected)
        accounting.update(
            projected_tokens=tokens,
            input_limit=self.limit,
            checkpoint_cut=self.checkpoint.evidence.cut,
        )
        with self.index.db:
            self.index.db.execute(
                "INSERT INTO native_projections(messages,accounting) VALUES(?,?)",
                (dump(projected), dump(accounting)),
            )
        marker = max(
            (r["attempt"] for r in self.request.transport_adapter.request_observations()), default=0
        )
        self.pending_projection = dict(
            attempt_marker=marker,
            model_input_sha256=digest(projected),
            canonical_input_sha256=digest(prepared),
            checkpoint_cut=self.checkpoint.evidence.cut,
            memo_checkpoint_sha256=self.memo.checkpoint_sha256,
            projected_tokens=tokens,
        )
        # A later oversized view follows the original transport capacity exit;
        # it is never trimmed again or followed by a second summarization.
        return projected
