"""One shared-budget checkpoint grounded in the current selected implementation."""

from contextlib import contextmanager
import copy
import hashlib
from pathlib import Path
import time

from infinicoder.memory.native_task_state_consolidation import NativeTaskStateConsolidation
from infinicoder.memory.native_memory import MemoryCapacityError
from infinicoder.memory.source.observed_source_reads import literal_read
from infinicoder.memory.task_checkpoint_sources import (
    SourceStateBundle,
    build_checkpoint_sources,
    refresh_checkpoint_sources,
    validate_checkpoint_sources,
)
from infinicoder.memory.task_state_checkpoint import (
    UnverifiedTaskMemo,
    accept_memo,
    checkpoint_view,
    digest,
    dump,
    prepare_checkpoint,
)


class NativeRevisionTaskCheckpoint(NativeTaskStateConsolidation):
    identity = "revision_grounded_task_checkpoint_v1"

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
            raise ValueError("explicit namespace reader factory required")
        self.reader_factory = reader_factory
        self.workspace_aliases = tuple(workspace_aliases)

    @contextmanager
    def bind(self, request, model, environment, deadline):
        self.source_reader = None
        self.source_bundle = None
        self.source_cutoff = None
        self.source_refs = set()
        with super().bind(request, model, environment, deadline):
            self.report.update(
                source_status="dormant",
                source_records=[],
                source_deliveries_are="HTTP markers only;reading is not implementation success",
            )
            yield self

    def _remaining(self):
        left = super()._remaining()
        if self.source_cutoff is not None:
            left = min(left, self.source_cutoff - time.monotonic())
        if left <= 0:
            raise TimeoutError("source checkpoint read/count allocation exhausted")
        return left

    def execution_returned(self, event, result):
        try:
            self._observe_returned_source(event, result)
        except (ValueError, UnicodeError):
            self._save_event(
                "checkpoint_source_observation_unknown", dict(execution_id=event["id"])
            )

    def _observe_returned_source(self, event, result):
        requested = literal_read(event.get("command", ""), self.workspace_aliases)
        output = result.get("output")
        if (
            not requested
            or len(requested) != 1
            or type(result.get("returncode")) is not int
            or result["returncode"] != 0
            or result.get("exception_info")
            or result.get("truncated")
            or result.get("timed_out")
            or not isinstance(output, str)
            or not output
            or len(output.encode()) > 64 * 1024
        ):
            return
        path, first, last = requested[0]
        event["source_read"] = dict(
            path=path,
            lines=[first, last],
            output=output,
            output_sha256=hashlib.sha256(output.encode()).hexdigest(),
            status="runtime_output_waiting_for_SDK_link",
            authority="Actual command output;source identity not yet established",
            whole_file=first == 1 and last is None,
        )

    def _sync(self, messages):
        prepared = super()._sync(messages)
        for event in self.executions:
            observed = event.get("source_read")
            ids = event.get("message_ids")
            if (
                not observed
                or observed["status"] != "runtime_output_waiting_for_SDK_link"
                or not ids
            ):
                continue
            raw = messages[ids[-1]]
            extra = raw.get("extra", {})
            content = prepared[ids[-1]].get("content", "")
            if (
                extra.get("raw_output") == observed["output"]
                and extra.get("returncode") == 0
                and content.count(observed["output"]) == 1
            ):
                observed.update(
                    status="linked_exact_output",
                    linked_message_ids=list(ids),
                    prepared_feedback_sha256=digest(prepared[ids[-1]]),
                )
                self._save_event(
                    "checkpoint_source_read_linked",
                    dict(
                        execution_id=event["id"],
                        path=observed["path"],
                        message_ids=list(ids),
                        output_sha256=observed["output_sha256"],
                    ),
                )
            else:
                observed["status"] = "unattributed_feedback;not_source_selection"
        self._flush_events()
        return prepared

    def _budget(self):
        return dict(
            remaining=self._remaining, count=self._count, source_tokens=min(1536, self.limit // 6)
        )

    def _start_source_read(self):
        self.source_cutoff = time.monotonic() + min(10, super()._remaining() / 10)

    def _save_sources(self, accounting):
        if self.source_bundle is None:
            return
        payload = validate_checkpoint_sources(self.source_bundle)
        for row in payload["sources"]:
            if row["status"] != "current_at_read" or row["ref"] in self.source_refs:
                continue
            view = dict(
                read_event_ref=row["ref"],
                view="checkpoint_current_definition",
                source_files={row["path"]: row["whole_file_sha256"]},
                records=[copy.deepcopy(row)],
                authority=row["authority"],
                origin="controller_fresh_read",
            )
            self.source_reader._record_view(
                dict(op="checkpoint_source", path=row["path"], selection=row["selection"]), view
            )
            self.source_refs.add(row["ref"])
        self.report["source_status"] = (
            "current"
            if any(r["status"] == "current_at_read" for r in payload["sources"])
            else "unknown"
        )
        self.report["source_records"].append(
            dict(
                bundle_sha256=digest(payload),
                accounting=accounting,
                paths=list(self.source_bundle.paths),
            )
        )
        (self.state / "native-checkpoint-source-state.json").write_text(
            self.source_bundle.payload_json, encoding="utf8"
        )
        self._record(
            "checkpoint_source_state", bundle_sha256=digest(payload), accounting=accounting
        )

    def _refresh_sources(self):
        self._start_source_read()
        try:
            self.source_bundle, accounting = refresh_checkpoint_sources(
                self.source_bundle, self.source_reader, self._budget()
            )
            if accounting["status"] != "unchanged":
                self._save_sources(accounting)
        except Exception as error:
            self._check_task()
            payload = self.source_bundle.payload
            for row in payload["sources"]:
                symbol = row.get("definition", {}).get("symbol") or row.get("selected_symbol")
                ref = row.get("ref") or row.get("historical_ref")
                for key in ("definition", "definition_sha256", "whole_file_sha256", "ref"):
                    row.pop(key, None)
                row.update(
                    status="unknown",
                    current_at_read=False,
                    selected_symbol=symbol,
                    historical_ref=ref,
                    reason=type(error).__name__,
                )
            self.source_bundle = SourceStateBundle(dump(payload))
            self._save_sources(dict(status="invalidated", reason=type(error).__name__))
        finally:
            self.source_cutoff = None
        if self._count([dict(role="user", content=self.source_bundle.payload_json)]) > min(
            1536, self.limit // 6
        ):
            raise MemoryCapacityError(
                "current source unknown markers exceed source budget;old body is not restored"
            )

    def before_query(self, messages):
        if self.checkpoint is not None:
            if self.source_bundle is not None:
                self._refresh_sources()
            return
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
        if self.reader_factory is None:
            self.report["source_status"] = "N_A;no_bound_reader;original188_semantics"
            return super().before_query(messages)
        self.used = True
        try:
            self.report["status"] = "preparing"
            references = self.task_state.capture_observations(
                prepared, blocks, copy.deepcopy(self.executions), copy.deepcopy(self.checks)
            )
            mandatory, receipts, current = self._selection(prepared, references)
            self._start_source_read()
            try:
                if self.source_reader is None:
                    self.source_reader = self.reader_factory(
                        self.request,
                        self.index,
                        self.environment,
                        count_output=lambda text: self._count([dict(role="user", content=text)]),
                        remaining=self._remaining,
                    )
                self.source_bundle, source_accounting = build_checkpoint_sources(
                    self.auxiliary.public_task, linked, self.source_reader, current, self._budget()
                )
                if self.source_bundle is None:
                    self.report["source_status"] = source_accounting["status"]
                    self._record("checkpoint_source_N_A", accounting=source_accounting)
                    self._disable("N_A;no complete source state fits")
                    return
                self._save_sources(source_accounting)
                plan = prepare_checkpoint(
                    prepared,
                    linked,
                    self.auxiliary.public_task,
                    (),
                    dict(
                        count=self._count,
                        input_limit=self.limit,
                        source_state=self.source_bundle,
                        mandatory=mandatory,
                        observation_refs=references["executions"],
                        receipts=receipts,
                        source_snapshot=current,
                        namespace=self.namespace,
                    ),
                )
                if plan is None:
                    self._disable("N_A;complete source/auxiliary history do not fit")
                    return
                provisional = UnverifiedTaskMemo("", digest(plan.evidence.records), (), ())
                view, _ = checkpoint_view(
                    plan, provisional, prepared, current, current_sources=self.source_bundle
                )
                empty_tokens = self._count(view)
                output_tokens = 384 if self.request.model_profile.context_limit <= 4096 else 768
                self.report["preflight_counts"] = dict(
                    original_tokens=original_tokens,
                    necessary_view_tokens=empty_tokens,
                    input_limit=self.limit,
                    reduction_limit=original_tokens * 0.8,
                )
                if (
                    empty_tokens > original_tokens * 0.8
                    or empty_tokens + output_tokens * 2 + 64 > self.limit
                ):
                    self._disable(
                        "N_A;necessary source view cannot fit or reduce input20% before memo"
                    )
                    return
            finally:
                self.source_cutoff = None
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
                    time.monotonic()
                    + min(
                        self.request.budget.request_timeout,
                        90,
                        (self.deadline - time.monotonic()) / 4,
                    ),
                ),
            )
            (self.state / "task-state-auxiliary-response.json").write_text(
                dump(response), encoding="utf8"
            )
            memo = accept_memo(response, plan.evidence)
            # The auxiliary call can outlast a source revision. Revalidate the
            # selected slots before its first actual ordinary consumer.
            self._refresh_sources()
            view, accounting = checkpoint_view(
                plan, memo, prepared, self._snapshot(), current_sources=self.source_bundle
            )
            tokens = self._count(view)
            self.report["final_counts"] = dict(
                original_tokens=original_tokens,
                projected_tokens=tokens,
                input_limit=self.limit,
                fits=tokens <= self.limit,
                reduces20pct=tokens <= original_tokens * 0.8,
            )
            if tokens > self.limit or tokens > original_tokens * 0.8:
                self._disable("complete source/memo view does not fit or reduce input20%")
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
        except Exception as error:
            self._check_task()
            self._disable(type(error).__name__ + ": " + str(error)[:400])

    def project(self, messages):
        if self.checkpoint is None or self.source_bundle is None:
            return super().project(messages)
        prepared = self._sync(messages)
        self.archive.sync()
        projected, accounting = checkpoint_view(
            self.checkpoint,
            self.memo,
            prepared,
            self._snapshot(),
            current_sources=self.source_bundle,
        )
        tokens = self._count(projected)
        accounting.update(
            projected_tokens=tokens,
            input_limit=self.limit,
            checkpoint_cut=self.checkpoint.evidence.cut,
            source_bundle_sha256=digest(self.source_bundle.payload),
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
            source_bundle_sha256=accounting["source_bundle_sha256"],
        )
        return projected


def cpu_checkpoint_reader(request, index, environment, *, paths, count_output, remaining):
    """Explicit packaged CPU binding;reading never changes TaskSpec write scope."""
    from infinicoder.execution.native_cpu import CpuExecution
    from infinicoder.memory.source.evidence_source import ReadOnlyTaskSources
    from infinicoder.planning.task_source_plan import WorkspaceSourcePlanReader

    execution = request.execution_adapter
    workspace = Path(request.workspace).resolve()
    if not isinstance(execution, CpuExecution) or execution.checked != workspace:
        raise ValueError("packaged verified live CPU workspace required")
    sources = ReadOnlyTaskSources(workspace, paths)
    sources.remaining = remaining
    reader = WorkspaceSourcePlanReader(
        request, index, count_output=count_output, remaining=remaining
    )
    reader.session._raw_read = sources.read
    reader.namespace.update(
        environment="packaged_verified_CPU_workspace_bind",
        binding_identity=digest(
            dict(execution=execution.identity, workspace=str(workspace), paths=list(paths))
        ),
        read_coverage="explicit caller paths only;not editing permission",
    )
    from infinicoder.execution.checkpoint_native_parser import definitions

    reader.checkpoint_native_definitions = lambda path, source, left: definitions(
        request, path, source, reader, left
    )
    return reader
