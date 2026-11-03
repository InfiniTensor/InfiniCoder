"""Default-off per-phase history projection around the original SDK lifecycle.

Old editing actions remain retrievable as exact events. A deterministic version
receipt may replace their repeated input only when their real observed effects
still identify the current declared files. This layer makes no new source or
correctness claims and performs no planning or generation.
"""

from contextlib import contextmanager
import copy
from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
import time

from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.memory.native_memory import MemoryCapacityError
from infinicoder.runtime.native_plain import NativePlainObserver
from infinicoder.memory.task_evidence_context import TaskEvidenceContext, sha
from infinicoder.memory.task_history_archive import TaskHistoryArchive, MODEL_ROOT
from infinicoder.planning.task_source_plan import dump


@dataclass(frozen=True)
class PhaseHistoryBinding:
    root_state: Path
    phase_state: Path
    phase_id: str
    workspace: Path
    original_task_sha256: str
    phase_task_sha256: str
    observed_paths: tuple
    deadline: float
    counter: object
    tokenizer_identity: str

    def __post_init__(self):
        if self.phase_id not in ("plain", "A", "B"):
            raise UnsupportedCore("identified plain/A/B phase required")
        for path in (self.root_state, self.phase_state, self.workspace):
            if not isinstance(path, Path) or not path.is_absolute():
                raise UnsupportedCore("absolute caller-owned phase paths required")
            if any(p.is_symlink() for p in (path, *path.parents)):
                raise UnsupportedCore("phase paths cannot contain symlinks")
        if self.phase_state != self.root_state / self.phase_id:
            raise UnsupportedCore("phase state must be the exact original state child")
        if self.phase_state.is_relative_to(self.workspace) or self.workspace.is_relative_to(
            self.root_state
        ):
            raise UnsupportedCore("phase state must be separate from candidate source")
        for digest in (self.original_task_sha256, self.phase_task_sha256):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise UnsupportedCore("original and actual phase task identities required")
        if (
            not isinstance(self.observed_paths, tuple)
            or len(set(self.observed_paths)) != len(self.observed_paths)
            or not callable(self.counter)
            or not isinstance(self.tokenizer_identity, str)
            or not self.tokenizer_identity
        ):
            raise UnsupportedCore("explicit observations and actual tokenizer required")
        if (
            isinstance(self.deadline, bool)
            or not isinstance(self.deadline, (float, int))
            or not math.isfinite(self.deadline)
            or self.deadline <= time.monotonic()
        ):
            raise UnsupportedCore("finite remaining original phase deadline required")

    @property
    def namespace(self):
        return sha(
            dict(
                root=str(self.root_state),
                phase=self.phase_id,
                workspace=str(self.workspace),
                original_task=self.original_task_sha256,
                phase_task=self.phase_task_sha256,
            )
        )

    def manifest(self):
        return dict(
            phase=self.phase_id,
            namespace=self.namespace,
            original_task_sha256=self.original_task_sha256,
            phase_task_sha256=self.phase_task_sha256,
            observed_paths=self.observed_paths,
            coverage="Caller-declared observations only;not read evidence or write permission",
        )


class NativePhaseHistoryContext(NativePlainObserver):
    identity = "phase_history_budget_v1"

    def __init__(self, binding):
        if not isinstance(binding, PhaseHistoryBinding):
            raise UnsupportedCore("caller-issued phase history binding required")
        self.binding = binding
        super().__init__(
            binding.counter,
            tokenizer_identity=binding.tokenizer_identity,
            implementation_paths=binding.observed_paths,
        )
        self._used = False
        self.archive = self.history_context = None
        self.records, self.count_cache = [], {}

    def _count(self, messages):
        self._remaining()
        key = (self.tokenizer_identity, sha(self.prepare(messages)))
        if key not in self.count_cache:
            count = super()._count(messages)
            if len(self.count_cache) >= 128:
                self.count_cache.pop(next(iter(self.count_cache)))
            self.count_cache[key] = count
        self._remaining()
        return self.count_cache[key]

    @contextmanager
    def bind_phase(self, request, model, environment):
        binding = self.binding
        if (
            self._used
            or Path(request.state_dir).resolve() != binding.root_state
            or Path(request.workspace).resolve() != binding.workspace
            or hashlib.sha256(request.spec.task.encode()).hexdigest()
            != binding.original_task_sha256
        ):
            raise UnsupportedCore("phase history is one-shot and belongs to its original request")
        if not callable(getattr(environment, "bind_history_archive", None)):
            raise UnsupportedCore("actual readonly phase history mapping required")
        self._used = True
        self.environment = environment
        phase_request = replace(request, state_dir=binding.phase_state)
        with super().bind(phase_request, model, environment, binding.deadline):
            self.task_state = TaskEvidenceContext(self.index, remaining=self._remaining)
            try:
                yield self
            finally:
                # The archive/mount is activated only on actual input pressure.
                # Hooks and SQLite are released by the inherited lifecycle.
                try:
                    if self.archive is not None:
                        try:
                            self.archive.sync()
                        except (TimeoutError, OSError, ValueError) as error:
                            self.records.append(
                                dict(
                                    status="post_run_archive_incomplete", error=type(error).__name__
                                )
                            )
                finally:
                    if self.history_context is not None:
                        self.history_context.__exit__(None, None, None)
                        self.history_context = None
                    self._save()

    def _save(self):
        value = dict(
            identity=self.identity,
            phase=self.binding.manifest(),
            context_records=self.records,
            archive_activated=self.archive is not None,
            canonical_history_changed=False,
            extra_model_calls=0,
            planner_enabled=False,
            resume_supported=False,
            capability_improvement_proven=False,
        )
        (self.state / "native-phase-history.json").write_bytes((dump(value) + "\n").encode())

    def _activate_archive(self):
        if self.archive is None:
            directory = self.state / "history-export"
            self.archive = TaskHistoryArchive(
                self.index.db,
                directory,
                remaining=self._remaining,
                phase=self.binding.phase_id,
                namespace=self.binding.namespace,
            )
            context = self.environment.bind_history_archive(
                directory, phase=self.binding.phase_id, namespace=self.binding.namespace
            )
            declared = context.__enter__()
            if (
                not isinstance(declared, dict)
                or declared.get("model_root") != MODEL_ROOT
                or declared.get("phase") != self.binding.phase_id
                or declared.get("namespace") != self.binding.namespace
                or Path(declared.get("directory", "")).resolve() != directory.resolve()
            ):
                context.__exit__(None, None, None)
                raise UnsupportedCore("phase history mount differs from caller identity")
            self.history_context = context
        self.archive.sync()

    def _selection(self, prepared, references):
        mandatory, episode = self._episode_selection(prepared)
        if episode["incomplete"]:
            raise MemoryCapacityError(
                "Unlinked or unknown current edit/check evidence cannot be projected"
            )
        prefix, blocks = self._history_blocks(prepared)
        current = self._snapshot()
        protected = set(prefix) | (set(blocks[-1]) if blocks else set())
        # Every actual trusted SDK check keeps its complete original block.
        for check in self.checks:
            if check.get("execution_id") is not None:
                ids = self.executions[check["execution_id"]].get("message_ids")
                if not ids:
                    raise MemoryCapacityError(
                        "Original check has no complete actual feedback block"
                    )
                for block in blocks:
                    if set(ids) & set(block):
                        protected.update(block)
                        mandatory.update(block)
        receipts = []
        for event in self.executions:
            ids = event.get("message_ids")
            block = next((block for block in blocks if ids and set(ids).issubset(block)), None)
            if (
                block is None
                or set(block) & protected
                or not set(block).issubset(mandatory)
                or not event.get("observer_complete")
                or event.get("error") is not None
                or type(event.get("returncode")) is not int
                or event.get("checks")
            ):
                continue
            changed = event.get("changed", [])
            if not changed or any(
                path in self.unattributed
                or event["after"][path] != current[path]
                or current[path].get("kind") == "unknown"
                or self.known[path] != current[path]
                for path in changed
            ):
                continue
            observation_ref = references["executions"].get(event["id"])
            if (
                not observation_ref
                or ("observations", observation_ref) not in self.archive.exported
                or any(("events", str(i)) not in self.archive.exported for i in block)
            ):
                continue
            receipt = dict(
                kind="current_edit_receipt",
                phase=self.binding.phase_id,
                namespace=self.binding.namespace,
                execution_id=event["id"],
                message_ids=list(block),
                message_sha256={str(i): sha(prepared[i]) for i in block},
                archive_refs=[
                    dict(
                        kind="event",
                        ref=str(i),
                        phase=self.binding.phase_id,
                        namespace=self.binding.namespace,
                    )
                    for i in block
                ],
                observation_ref=observation_ref,
                command_sha256=hashlib.sha256(event["command"].encode()).hexdigest(),
                returncode=event["returncode"],
                error=event["error"],
                changed_paths=list(changed),
                before={p: event["before"][p] for p in changed},
                after={p: event["after"][p] for p in changed},
                current={p: current[p] for p in changed},
                authority="Actual historical edit linked to current declared path versions only;not source text,interface validity,whole repository coverage or independent acceptance",
                current_source_body="Unknown unless separately read at this version;read current files using ordinary tools",
            )
            receipts.append(receipt)
            mandatory.difference_update(block)
        episode.update(
            phase=self.binding.manifest(),
            source_body_coverage="No source reader;source facts unknown",
            verification_authority="Original checker callbacks retain binding;self-tests stay observations",
        )
        return mandatory, episode, receipts

    def project(self, messages):
        prepared = self._sync(messages)
        if (
            len(prepared) < 2
            or [m.get("role") for m in prepared[:2]] != ["system", "user"]
            or any(not isinstance(m.get("content"), str) for m in prepared)
        ):
            raise UnsupportedCore("phase history requires original pinned SDK text roles")
        original_tokens = self._count(prepared)
        record = dict(
            original_tokens=original_tokens,
            input_limit=self.limit,
            phase=self.binding.phase_id,
            canonical_history_changed=False,
        )
        try:
            if original_tokens <= self.limit:
                record.update(
                    status="original_input_unchanged",
                    compacted=False,
                    model_input_sha256=sha(prepared),
                )
                return messages
            _, blocks = self._history_blocks(prepared)
            with self.lock:
                references = self.task_state.capture_observations(
                    prepared, blocks, copy.deepcopy(self.executions), copy.deepcopy(self.checks)
                )
            self._activate_archive()
            with self.lock:
                mandatory, episode, receipts = self._selection(prepared, references)
            sources = self.task_state.source_state(None)
            projected, accounting = self.task_state.project(
                prepared,
                mandatory=mandatory,
                blocks=blocks,
                episode=episode,
                sources=sources,
                archive=dict(command="python -B " + MODEL_ROOT + "/read.py"),
                count=self._count,
                limit=self.limit,
                edit_receipts=receipts,
            )
            record.update(accounting, edit_receipt_count=len(receipts))
            with self.index.db:
                self.index.db.execute(
                    "INSERT INTO native_projections(messages,accounting) VALUES(?,?)",
                    (dump(projected), dump(record)),
                )
            return projected
        except (MemoryCapacityError, TimeoutError, OSError, ValueError) as error:
            record.update(
                status="capacity_or_evidence_rejected;no_generation", error=type(error).__name__
            )
            raise
        finally:
            self.records.append(record)
            self._save()
