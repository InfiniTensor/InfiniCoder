"""One source plan plus a versioned working set under actual capacity pressure."""

from contextlib import contextmanager, nullcontext
import copy
import json
from pathlib import Path

from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.memory.native_memory import MemoryCapacityError
from infinicoder.planning.native_source_planning import NativeSourcePlanning
from infinicoder.memory.task_evidence_context import TaskEvidenceContext, sha
from infinicoder.memory.task_history_archive import TaskHistoryArchive, MODEL_ROOT
from infinicoder.planning.task_source_plan import SourcePlanPacket, parse_draft, dump


class NativeTaskEvidenceContext(NativeSourcePlanning):
    identity = "native_task_evidence_context_v1"

    def __init__(self, *args, working_set=True, **kwargs):
        super().__init__(*args, **kwargs)
        if type(working_set) is not bool:
            raise ValueError("explicit working-set boolean required")
        self.working_set = working_set
        self.history_binding = None
        self.archive = None
        self.context_records = []

    @contextmanager
    def bind(self, request, model, environment, deadline):
        with super().bind(request, model, environment, deadline):
            self.task_state = TaskEvidenceContext(self.index, remaining=self._remaining)
            binder = getattr(environment, "bind_history_archive", None)
            binding = (
                binder(self.state / "history-export") if callable(binder) else nullcontext(None)
            )
            with binding as declared:
                if declared is not None:
                    if (
                        not isinstance(declared, dict)
                        or not declared.get("identity")
                        or declared.get("model_root") != MODEL_ROOT
                        or not declared.get("directory")
                    ):
                        raise UnsupportedCore("identified actual readonly history mapping required")
                    directory = Path(declared["directory"]).resolve()
                    if directory.is_relative_to(self.workspace) or self.workspace.is_relative_to(
                        directory
                    ):
                        raise UnsupportedCore(
                            "history export must be separate from candidate source"
                        )
                    self.archive = TaskHistoryArchive(
                        self.index.db, directory, remaining=self._remaining
                    )
                    self.history_binding = copy.deepcopy(declared)
                try:
                    yield self
                finally:
                    if self.archive is not None:
                        try:
                            self.archive.sync()
                        except (TimeoutError, OSError, ValueError) as error:
                            self.context_records.append(
                                dict(
                                    status="post_run_archive_incomplete",
                                    error_type=type(error).__name__,
                                )
                            )
                    self._save_context_report()
        info = self.state / "native-memory-info.json"
        if info.exists():
            value = json.loads(info.read_bytes())
            value.update(
                identity=self.identity,
                automatic_compression=self.working_set,
                projection="Full source-planning input while it fits;versioned evidence projection under capacity pressure",
                canonical_history_changed=False,
                history_access=self.history_binding,
            )
            info.write_bytes((dump(value) + "\n").encode())

    def _save_context_report(self):
        value = dict(
            identity=self.identity,
            working_set_enabled=self.working_set,
            history_mapping=self.history_binding,
            context_records=self.context_records,
            canonical_history_changed=False,
            extra_model_calls=0,
            resume_supported=False,
            source_coverage="Queried/explicit paths only;other sources unknown",
            capability_improvement_proven=False,
        )
        (self.state / "native-task-evidence-context.json").write_bytes(
            (dump(value) + "\n").encode()
        )

    def prepare_plan(self, *args, **kwargs):
        super().prepare_plan(*args, **kwargs)
        packet = self.packet
        if packet is None and self.reader is not None and self.report.get("planning_response_ref"):
            # Preserve returned unverified prose as a hypothesis record, even
            # when the common planner correctly declined to advertise it.
            path = Path(self.report["planning_response_ref"])
            response = json.loads(path.read_bytes())
            choice = (response.get("extra", {}).get("response", {}).get("choices") or [{}])[0]
            text = choice.get("message", {}).get("content", response.get("content", ""))
            packet = SourcePlanPacket(
                self.reader.namespace,
                parse_draft(
                    text,
                    complete=choice.get("finish_reason") == "stop",
                    workspace_aliases=getattr(self.reader, "workspace_aliases", ()),
                ),
                str(path),
            )
        self.task_state.capture_plan(packet)
        # Observation coverage is not edit permission or repository closure.
        paths = self.task_state.observed_paths()
        original = tuple(self.paths)
        self.paths = (
            original
            + tuple(path for path in paths if path not in original)[: max(0, 16 - len(original))]
        )
        self.known = self._snapshot()
        if self.archive is not None:
            self.archive.sync()

    def _episode(self, prepared, references):
        mandatory, original = self._episode_selection(prepared)
        if original["incomplete"]:
            raise MemoryCapacityError(
                "Unlinked edit/check/source observations cannot be safely compacted"
            )
        check = copy.deepcopy(original["original_check"])
        if check is not None:
            if not self.paths:
                check["association"] = "unknown;no_observed_source_coverage"
            elif any(v.get("kind") == "unknown" for v in self._snapshot().values()):
                check["association"] = "unknown_source_identity"
            check = {
                k: check[k]
                for k in (
                    "message_ids",
                    "association",
                    "judge_identity",
                    "scope",
                    "coverage",
                    "report_sha256",
                    "error",
                    "origin",
                )
                if k in check
            }
            check["observation_ref"] = references["checks"].get(self.checks[-1]["id"])
        return mandatory, dict(
            observed_paths=list(self.paths),
            edits=original["edits"],
            original_check=check,
            candidate_coverage="Observed sources only;unobserved repository changes unknown",
            verification_authority="Checker callbacks retain version binding;self-tests are observations",
        )

    def project(self, messages):
        ordinary = super().project(messages)
        prepared = self.prepare(messages)
        prefix, blocks = self._history_blocks(prepared)
        with self.lock:
            references = self.task_state.capture_observations(
                prepared, blocks, copy.deepcopy(self.executions), copy.deepcopy(self.checks)
            )
        if self.archive is not None:
            self.archive.sync()
        original_tokens = self._count(prepared)
        record = dict(
            original_tokens=original_tokens,
            input_limit=self.limit,
            canonical_history_changed=False,
            working_set_enabled=self.working_set,
        )
        try:
            if not self.working_set or original_tokens <= self.limit:
                # The same repaired planner prepares the full input in both
                # arms. Internal bookkeeping is not counted as model exposure.
                record.update(
                    compacted=False,
                    status="ordinary_source_planning_input_unchanged",
                    model_input_sha256=sha(self.prepare(ordinary)),
                    history_retrievable=self.archive is not None,
                )
                return ordinary
            if self.archive is None:
                raise MemoryCapacityError("No actual readonly history access;compression disabled")
            with self.lock:
                mandatory, episode = self._episode(prepared, references)
            # This maintenance is charged to the same task wall budget. At most
            # two changed observed excerpts are reread;there is no new planner.
            sources = self.task_state.source_state(self.reader, refresh=True)
            if self.checks and episode["original_check"] is not None:
                checked = self.checks[-1].get("after") or {}
                if any(
                    path in checked and checked[path].get("sha256") != digest
                    for path, digest in sources["current_versions"].items()
                ):
                    episode["original_check"]["association"] = "stale;source_changed_since_check"
            self.archive.sync()
            projected, projection = self.task_state.project(
                prepared,
                mandatory=mandatory,
                blocks=blocks,
                episode=episode,
                sources=sources,
                archive=dict(command="python -B " + MODEL_ROOT + "/read.py"),
                count=self._count,
                limit=self.limit,
            )
            record.update(projection)
            with self.index.db:
                self.index.db.execute(
                    "INSERT INTO native_projections(messages,accounting) VALUES(?,?)",
                    (dump(projected), dump(record)),
                )
            return projected
        except MemoryCapacityError as error:
            record.update(
                compacted=False, status="capacity_rejected;no_generation", error=str(error)
            )
            raise
        finally:
            self.context_records.append(record)
            self._save_context_report()
