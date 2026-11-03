"""Keep real candidate/check episodes in a bounded native model input.

This selects original messages; it neither interprets stdout as acceptance nor
executes checks. A trusted caller may notify an original evaluator's invocation.
Hashes cover only explicitly observed implementation paths, never a whole repo.
"""

from contextlib import contextmanager
import copy
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
import threading
import time

from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.memory.native_memory import (
    NativeRepositoryMemory,
    MemoryCapacityError,
    _json,
    _sha,
)


class NativeEpisodeMemory(NativeRepositoryMemory):
    identity = "native_episode_memory_v2_sdk_feedback"

    def __init__(self, count_messages, *, tokenizer_identity, implementation_paths, **kwargs):
        super().__init__(count_messages, tokenizer_identity=tokenizer_identity, **kwargs)
        self.paths = tuple(implementation_paths)

    def validate(self, request):
        super().validate(request)
        if not self.paths or len(set(self.paths)) != len(self.paths):
            raise UnsupportedCore("explicit distinct implementation paths required")
        for value in self.paths:
            if (
                not isinstance(value, str)
                or not value
                or "\\" in value
                or ":" in value
                or "\0" in value
            ):
                raise UnsupportedCore("implementation paths must be relative POSIX paths")
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts or str(path) != value:
                raise UnsupportedCore("implementation path escapes or is not normalized")

    def _snapshot(self):
        result = {}
        for relative in self.paths:
            path = self.workspace / relative
            try:
                # Do not follow either a source symlink or a linked parent.
                parents = [path, *path.parents]
                if any(p.is_symlink() for p in parents if p != self.workspace.parent):
                    result[relative] = dict(kind="unknown", reason="symlink")
                elif not path.resolve().is_relative_to(self.workspace):
                    result[relative] = dict(kind="unknown", reason="outside workspace")
                elif not path.exists():
                    result[relative] = dict(kind="missing")
                elif path.is_file():
                    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                    with os.fdopen(fd, "rb") as stream:
                        before = os.fstat(stream.fileno())
                        if not stat.S_ISREG(before.st_mode) or before.st_size > 16 * 1024 * 1024:
                            result[relative] = dict(
                                kind="unknown",
                                reason="not regular or exceeds16MiB observation bound",
                            )
                            continue
                        digest = hashlib.sha256()
                        while True:
                            self._remaining()
                            chunk = stream.read(1024 * 1024)
                            if not chunk:
                                break
                            digest.update(chunk)
                        after = os.fstat(stream.fileno())
                    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
                    if (
                        identity(before) != identity(after)
                        or before.st_ctime_ns != after.st_ctime_ns
                        or identity(after) != identity(path.lstat())
                    ):
                        result[relative] = dict(kind="unknown", reason="changed while hashing")
                    else:
                        result[relative] = dict(kind="file", sha256=digest.hexdigest())
                else:
                    result[relative] = dict(kind="unknown", reason="not a regular file")
            except OSError as error:
                result[relative] = dict(kind="unknown", reason=type(error).__name__)
        return result

    def _save_event(self, kind, event):
        # Check callbacks can run on the original endpoint's server thread.
        # SQLite remains owned by the agent thread; synchronize it at _sync.
        with self.lock:
            self.events.append(dict(kind=kind, **copy.deepcopy(event)))

    @contextmanager
    def action_context(self, event, action, args, kwargs):
        """Optional caller tools share the real action's lifetime;default no-op."""
        yield

    def decision_observed(self, pending):
        """A real model response returned; preparation alone is not a decision."""
        pass

    def execution_returned(self, event, result):
        """Optional observation only;does not replace or execute an SDK action."""
        pass

    def execution_completed(self, event):
        """Optional observation after finally has completed both source snapshots."""
        pass

    @contextmanager
    def bind(self, request, model, environment, deadline):
        self.deadline = deadline
        self.workspace = Path(request.workspace).resolve()
        self.lock = threading.RLock()
        self.controller_local = threading.local()
        self.events, self.executions, self.checks, self.origins = [], [], [], {}
        self.pending_response = self.active = None
        self.unattributed = set()
        self.known = self._snapshot()
        with super().bind(request, model, environment, deadline):
            self.index.db.execute(
                "CREATE TABLE native_episodes(id INTEGER PRIMARY KEY, event TEXT)"
            )
            query, execute = model.query, environment.execute

            def observed_query(messages, **kwargs):
                self.pending_response = None
                result = query(messages, **kwargs)
                prepared = self.prepare([result])
                self.pending_response = dict(
                    message_id=len(self.seen), message=copy.deepcopy(prepared[0])
                )
                self.decision_observed(self.pending_response)
                return result

            def observed_execute(action, *args, **kwargs):
                with self.lock:
                    pending = self.pending_response
                    event = dict(
                        id=len(self.executions),
                        before=self._snapshot(),
                        after=None,
                        response=copy.deepcopy(pending),
                        command=action.get("command"),
                        message_ids=None,
                        checks=[],
                        error=None,
                        observer_complete=True,
                    )
                    self.executions.append(event)
                    self.active = event
                try:
                    with self.action_context(event, action, args, kwargs):
                        result = execute(action, *args, **kwargs)
                    event["returncode"] = result.get("returncode")
                    self.execution_returned(event, result)
                    return result
                except BaseException as error:
                    event["error"] = type(error).__name__
                    raise
                finally:
                    with self.lock:
                        try:
                            event["after"] = self._snapshot()
                        except Exception as observation_error:
                            if event["error"] is None:
                                raise
                            # Preserve an original cancellation/deadline/tool exception
                            # when cleanup observation itself exhausts its budget.
                            event["after"] = {
                                p: dict(kind="unknown", reason=type(observation_error).__name__)
                                for p in self.paths
                            }
                            event["observer_complete"] = False
                        event["changed"] = [
                            p for p in self.paths if event["before"][p] != event["after"][p]
                        ]
                        # Changes outside this observed command have no invented origin.
                        self.unattributed.update(
                            p for p in self.paths if self.known[p] != event["before"][p]
                        )
                        for p in event["changed"]:
                            self.origins[p] = event
                            self.unattributed.discard(p)
                        self.known = event["after"]
                        self.active = None
                        self._save_event("execution", event)
                    self.execution_completed(event)

            model.query, environment.execute = observed_query, observed_execute
            try:
                yield self
            finally:
                model.query, environment.execute = query, execute
                self._flush_events()

    def begin_check(self, *, judge_identity, scope, coverage):
        """Trusted original-evaluator callback; never called from model output."""
        if not judge_identity or not scope or not coverage:
            raise ValueError("original checker identity, scope and coverage required")
        with self.lock:
            active = self.active
            event = dict(
                id=len(self.checks),
                execution_id=active["id"] if active is not None else None,
                before=self._snapshot(),
                after=None,
                judge_identity=judge_identity,
                scope=scope,
                coverage=coverage,
                report=None,
                error=None,
                finished=False,
                origin="sdk_execution",
                started_ns=time.monotonic_ns(),
            )
            dispatch = getattr(self.controller_local, "dispatch", None)
            if dispatch is not None:
                event.update(
                    origin="controller_public_operation",
                    execution_id=None,
                    controller_dispatch=copy.deepcopy(dispatch),
                )
            self.checks.append(event)
            if active is not None:
                active["checks"].append(event["id"])
            return event["id"]

    def end_check(self, token, *, report=None, error=None):
        with self.lock:
            event = self.checks[token]
            if event["finished"]:
                raise ValueError("duplicate original check notification")
            event.update(
                after=self._snapshot(),
                report=copy.deepcopy(report),
                error=type(error).__name__ if error is not None else None,
                finished=True,
                finished_ns=time.monotonic_ns(),
            )
            # A detached/background check is not linked to a later command.
            event["same_execution"] = (
                self.active is not None and self.active["id"] == event["execution_id"]
            )
            if event.get("origin") == "controller_public_operation":
                event["same_controller_dispatch"] = (
                    getattr(self.controller_local, "dispatch", None) == event["controller_dispatch"]
                )
            self._save_event("original_check", event)

    @contextmanager
    def controller_public_dispatch(self, dispatch):
        """A main-thread operation has explicit provenance, never SDK messages."""
        if self.active is not None or getattr(self.controller_local, "dispatch", None) is not None:
            raise UnsupportedCore(
                "controller public operation cannot nest an SDK action or another dispatch"
            )
        self.controller_local.dispatch = copy.deepcopy(dispatch)
        try:
            yield
        finally:
            self.controller_local.dispatch = None

    def record_controller_return(self, dispatch, *, before, after, report, error=None):
        """Caller capability result when it has no evaluator-notification hook.

        This records the actual capability invocation, not an invented callback.
        A caller which already notified begin/end retains its original record.
        """
        with self.lock:
            existing = [c for c in self.checks if c.get("controller_dispatch") == dispatch]
            if existing:
                return existing[-1]
            event = dict(
                id=len(self.checks),
                execution_id=None,
                origin="controller_public_operation",
                controller_dispatch=copy.deepcopy(dispatch),
                same_controller_dispatch=True,
                result_source="actual_caller_capability_return;not_evaluator_callback",
                before=copy.deepcopy(before),
                after=copy.deepcopy(after),
                judge_identity=copy.deepcopy(dispatch["implementation_identity"]),
                scope=dispatch["scope"],
                coverage=copy.deepcopy(dispatch["coverage"]),
                report=copy.deepcopy(report),
                error=error,
                finished=True,
                same_execution=False,
                finished_ns=time.monotonic_ns(),
            )
            self.checks.append(event)
            self._save_event("original_check", event)
            return event

    def observer_failed(self, error):
        """Caller keeps the original result; an observation failure cannot pass it."""
        with self.lock:
            self._save_event("observer_error", dict(error=type(error).__name__))
            if self.active is not None:
                self.active["observer_complete"] = False

    def _flush_events(self):
        with self.lock, self.index.db:
            saved = self.index.db.execute("SELECT COUNT(*) FROM native_episodes").fetchone()[0]
            for i in range(saved, len(self.events)):
                self.index.db.execute(
                    "INSERT INTO native_episodes VALUES(?,?)", (i, _json(self.events[i]))
                )

    def _sync(self, messages):
        prepared = super()._sync(messages)
        with self.lock:
            for event in self.executions:
                pending = event["response"]
                if event["message_ids"] is not None or pending is None:
                    continue
                i = pending["message_id"]
                # The pinned SDK has one execution per assistant/user pair.
                siblings = [
                    e
                    for e in self.executions
                    if e["response"] is not None and e["response"]["message_id"] == i
                ]
                if (
                    len(siblings) == 1
                    and i + 1 < len(prepared)
                    and prepared[i] == pending["message"]
                    and prepared[i]["role"] == "assistant"
                    and prepared[i + 1]["role"] == "user"
                ):
                    event["message_ids"] = [i, i + 1]
                    self._save_event(
                        "linked_execution",
                        dict(execution_id=event["id"], message_ids=event["message_ids"]),
                    )
        self._flush_events()
        return prepared

    @staticmethod
    def _history_blocks(prepared):
        """Keep SDK user-only errors without inventing rejected assistant text.

        An action and all following observations/errors form one indivisible
        block. Initial rejected-output feedback stays with the original task.
        """
        prefix = [0, 1]
        blocks = []
        i = 2
        while i < len(prepared) and prepared[i]["role"] == "user":
            prefix.append(i)
            i += 1
        while i < len(prepared):
            if prepared[i]["role"] != "assistant":
                raise UnsupportedCore("episode selector requires the pinned text SDK roles")
            start = i
            i += 1
            while i < len(prepared) and prepared[i]["role"] == "user":
                i += 1
            if i == start + 1:
                raise UnsupportedCore("assistant action has no actual SDK feedback")
            blocks.append(list(range(start, i)))
        return prefix, blocks

    def _episode_selection(self, prepared):
        current = self._snapshot()
        prefix, blocks = self._history_blocks(prepared)
        block_for_id = {i: block for block in blocks for i in block}
        mandatory = set(prefix)
        if blocks:
            mandatory.update(blocks[-1])
        missing, edits = [], []
        for path in self.paths:
            if (
                current[path].get("kind") == "unknown"
                or path in self.unattributed
                or current[path] != self.known[path]
            ):
                missing.append(path + ":unobserved_or_unknown_change")
            event = self.origins.get(path)
            if event is not None:
                ids = event["message_ids"]
                if ids is None or event["after"][path] != current[path]:
                    missing.append(path + ":unlinked_edit")
                else:
                    mandatory.update(i for number in ids for i in block_for_id[number])
                    edits.append(dict(path=path, message_ids=ids, source_identity=current[path]))
        evidence = None
        if self.checks:
            check = self.checks[-1]
            event = (
                self.executions[check["execution_id"]]
                if check["execution_id"] is not None
                else None
            )
            ids = event["message_ids"] if event is not None else None
            status = "unknown"
            controller = (
                check.get("origin") == "controller_public_operation"
                and check.get("same_controller_dispatch") is True
            )
            if check["finished"] and (
                controller or check.get("same_execution") and ids and event["observer_complete"]
            ):
                status = (
                    "current_observed_paths"
                    if check["before"] == check["after"] == current
                    and not any(v.get("kind") == "unknown" for v in current.values())
                    else "stale_or_changed_during_check"
                )
            if ids:
                mandatory.update(i for number in ids for i in block_for_id[number])
            elif not controller:
                missing.append("original_check:unlinked_action")
            evidence = self._check_evidence(check, event, ids, current, status)
        return mandatory, dict(
            observed_paths=list(self.paths),
            edits=edits,
            original_check=evidence,
            incomplete=missing,
            limits="Path hashes cover these targets only. Check coverage is explicit; historical claims are not acceptance. Final independent verification remains required.",
        )

    def _check_evidence(self, check, event, ids, current, status):
        # Extension point shares this exact snapshot with a source-workset
        # renderer. The original episode profile retains its prior metadata.
        return dict(
            message_ids=ids,
            association=status,
            judge_identity=check["judge_identity"],
            scope=check["scope"],
            coverage=check["coverage"],
            before=check["before"],
            after=check["after"],
            report_sha256=_sha(_json(check["report"])),
            error=check["error"],
            origin=check.get("origin", "sdk_execution"),
            controller_dispatch=check.get("controller_dispatch"),
        )

    def project(self, messages):
        prepared = self._sync(messages)
        if (
            len(prepared) < 2
            or [m.get("role") for m in prepared[:2]] != ["system", "user"]
            or any(not isinstance(m.get("content"), str) for m in prepared)
        ):
            raise UnsupportedCore("episode memory requires the pinned text system/task history")
        full = [dict(m) for m in prepared]
        full[1]["content"] += "\n\nMEMORY TOOLS\n" + self.HELP
        original_tokens = self._count(full)
        compacted = original_tokens > self.limit
        selected = set(range(len(full)))
        metadata = None
        if compacted:
            selected, metadata = self._episode_selection(prepared)
            # No silent deletion of a candidate origin or original check.
            if metadata["incomplete"]:
                projected = full
            else:

                def assemble(ids):
                    view = [dict(full[i]) for i in sorted(ids)]
                    omitted = [i for i in range(2, len(full)) if i not in ids]
                    note = dict(
                        episode=metadata,
                        archived_count=len(omitted),
                        recent_archived=[dict(id=i, sha256=self.seen[i]) for i in omitted[-4:]],
                        policy="Selected original messages retain their roles/order/text;ic-memory event ID recovers exact archived text. No commands replayed.",
                    )
                    view[1]["content"] += "\n\nHISTORICAL EPISODE INDEX\n" + _json(note)
                    return view

                projected = assemble(selected)
                if self._count(projected) <= self.limit:
                    for block in reversed(self._history_blocks(prepared)[1]):
                        pair = set(block)
                        if pair.issubset(selected):
                            continue
                        candidate = selected | pair
                        trial = assemble(candidate)
                        if self._count(trial) > self.limit:
                            break
                        selected, projected = candidate, trial
        else:
            projected = full
        final_tokens = self._count(projected)
        rejected = final_tokens > self.limit or bool(metadata and metadata["incomplete"])
        record = dict(
            original_messages=len(prepared),
            original_tokens=original_tokens,
            projected_tokens=final_tokens,
            input_limit=self.limit,
            compacted=compacted,
            selected=sorted(selected),
            omitted=[i for i in range(len(prepared)) if i not in selected],
            episode=metadata,
            original_task_sha256=_sha(prepared[1]["content"]),
            tokenizer_identity=self.tokenizer_identity,
            status="capacity_rejected;episode_incomplete"
            if rejected
            else "ready_for_native_query;HTTP ledger determines actual forwarding",
        )
        with self.index.db:
            self.index.db.execute(
                "INSERT INTO native_projections(messages,accounting) VALUES(?,?)",
                (_json(projected), _json(record)),
            )
        with (self.state / "native-memory-context.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(_json(record) + "\n")
        if rejected:
            raise MemoryCapacityError(
                "episode_incomplete: exact task/latest/edit/check episodes do not fit or cannot be reliably linked;no generation forwarded"
            )
        return projected
