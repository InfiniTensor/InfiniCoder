"""Conditional evidence execution in one original native agent loop.

Canonical SDK history is never edited. Operations run only before the next
ordinary query, under the existing deadline; projection cannot dispatch them.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import copy
import json
import math
import os
from pathlib import Path
import time

from infinicoder.domains.operator.candidate_observation import (
    HistoricalSources,
    _body_audit,
    _fragment,
)
from infinicoder.memory.evidence_route import _digest, _known_revision, choose_evidence_operation
from infinicoder.memory.source.evidence_source import literal_source_read
from infinicoder.memory.native_memory import _json
from infinicoder.runtime.native_plain import NativePlainObserver
from infinicoder.runtime.shell_control import classify_shell_control


@dataclass(frozen=True)
class PostEditPublicBudget:
    """Read-only access to the original caller search/call/cancellation owner.

    remaining_search_seconds must already exclude the caller's final reserve.
    The strategy never opens a second budget or calls a model itself.
    """

    remaining_search_seconds: object
    remaining_model_calls: object
    cancelled: object

    def __post_init__(self):
        if not all(
            callable(getattr(self, name))
            for name in ("remaining_search_seconds", "remaining_model_calls", "cancelled")
        ):
            raise ValueError("original caller budget callbacks required")

    def available(self):
        if self.cancelled():
            raise KeyboardInterrupt("caller cancelled before public feedback")
        seconds, calls = self.remaining_search_seconds(), self.remaining_model_calls()
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds < 0
            or type(calls) is not int
            or calls < 0
        ):
            raise ValueError("original budget callbacks returned unknown availability")
        return seconds, calls


class NativeEvidenceRoute(NativePlainObserver):
    identity = "plain_native_conditional_evidence_route_v1"
    HEADER = "\n\nCONTROLLER EVIDENCE (observed data;not SDK actions or final acceptance)\n"

    def __init__(
        self,
        *args,
        policy="evidence_route_v1",
        source_capability=None,
        body_write_scopes=(),
        workspace_aliases=("/workspace",),
        post_edit_budget=None,
        task_working_memory=False,
        **kwargs,
    ):
        if policy not in (None, "evidence_route_v1", "post_edit_public_v1"):
            raise ValueError("unknown evidence route policy")
        if post_edit_budget is not None and not isinstance(post_edit_budget, PostEditPublicBudget):
            raise TypeError("typed original post-edit budget binding required")
        if policy == "post_edit_public_v1" and post_edit_budget is None:
            raise ValueError("post-edit public feedback needs the original budget owner")
        if post_edit_budget is not None and policy not in (None, "post_edit_public_v1"):
            raise ValueError("post-edit feedback is separate from cyclic evidence routing")
        self.post_edit_budget = post_edit_budget
        if post_edit_budget is not None:
            self.identity = "native_post_edit_public_v1;" + (
                "on" if policy == "post_edit_public_v1" else "off"
            )
        self.policy = policy
        if type(task_working_memory) is not bool or (
            task_working_memory and policy not in (None, "post_edit_public_v1")
        ):
            raise ValueError("task working memory supports plain or post-edit execution")
        self.task_working_memory = task_working_memory
        self.working_memory = None
        if task_working_memory:
            self.identity += ";pressure_task_memory_v1"
            self.expose_tools = True
            self.command_names = ("ic-task-memory",)
        self.source_capability = source_capability
        self.body_write_scopes = tuple(body_write_scopes)
        self.workspace_aliases = tuple(workspace_aliases)
        if any(
            not isinstance(p, str)
            or not p.startswith("/")
            or p.endswith("/")
            or ".." in p.split("/")
            for p in self.workspace_aliases
        ):
            raise ValueError("workspace aliases must be explicit absolute roots")
        super().__init__(*args, **kwargs)

    def validate(self, request):
        super().validate(request)
        if self.post_edit_budget is not None:
            self.validate_scoped(request)
        from infinicoder.memory.source.source_workset import public_target_present

        for target in self.body_write_scopes:
            if (
                not isinstance(target, str)
                or "::" not in target
                or target.split("::", 1)[0] not in self.paths
                or not public_target_present(request.spec.task, target)
            ):
                raise ValueError(
                    "body audit requires a caller-explicit public body scope in observed paths"
                )
        if self.source_capability is not None:
            self.source_capability.validate(request.workspace, self.paths)

    def validate_scoped(self, request):
        from infinicoder.runtime.agent_execution import UnsupportedCore
        from infinicoder.execution.native_scoped_execution import ScopedNativeExecution

        if (
            type(self) is not NativeEvidenceRoute
            or (
                self.post_edit_budget is None
                and not (self.task_working_memory and self.policy is None)
            )
            or self.policy not in (None, "post_edit_public_v1")
            or self.source_capability is not None
            or self.body_write_scopes
            or not self.paths
            or len(self.paths) > 64
            or type(request.execution_adapter) is not ScopedNativeExecution
            or any(
                getattr(request, k) is not None
                for k in (
                    "planning_adapter",
                    "repair_adapter",
                    "candidate_search",
                    "implementation_proposal",
                )
            )
        ):
            raise UnsupportedCore(
                "scoped evidence supports only the explicit bounded post-edit policy"
            )
        if not request.spec.allowed_paths or any(
            not p.endswith("/") for p in request.spec.allowed_paths
        ):
            raise UnsupportedCore("post-edit feedback requires the original directory scope")
        if any(
            not any(path.startswith(prefix) for prefix in request.spec.allowed_paths)
            for path in self.paths
        ):
            raise UnsupportedCore(
                "observed implementation paths must stay inside the original write scope"
            )

    @contextmanager
    def bind_public_operation(self, operation):
        if self.post_edit_budget is not None:
            from infinicoder.runtime.agent_execution import UnsupportedCore

            manifest = operation.manifest if operation is not None else {}
            if (
                set(manifest.get("observed_paths", ())) != set(self.paths)
                or manifest.get("source_mutation_allowed") is not False
                or manifest.get("workspace") != str(self.workspace)
            ):
                raise UnsupportedCore(
                    "post-edit feedback needs the same declared read-only public operation"
                )
        with super().bind_public_operation(operation):
            yield

    @contextmanager
    def bind(self, request, model, environment, deadline):
        self.request = request
        self.window = []
        self.route_seen = set()
        self.need_seen = set()
        self.dispatched = set()
        self.results = []
        self.operation_count = 0
        self.post_edit_seen = set()
        self.post_edit_last_status = None
        self.post_edit_wall = 0.0
        self.historical = None
        self.report_deliveries = []
        with super().bind(request, model, environment, deadline):
            if self.task_working_memory:
                from infinicoder.memory.task_working_memory import TaskWorkingMemory

                self.working_memory = TaskWorkingMemory(self)
            if self.policy and self.source_capability is not None and self.body_write_scopes:
                initial = tuple(dict.fromkeys(t.split("::", 1)[0] for t in self.body_write_scopes))
                self.historical = HistoricalSources(
                    self.state / "routing-historical-source",
                    self._read,
                    initial,
                    remaining=self._remaining,
                )
            try:
                yield self
            finally:
                self._save_event(
                    "evidence_route_summary",
                    dict(
                        origin="controller",
                        policy=self.policy,
                        operations=self.operation_count,
                        results=len(self.results),
                        SDK_history_fabricated=False,
                    ),
                )
                self._flush_events()

    def _read(self, path, **kwargs):
        self._remaining()
        value = self.source_capability.read(path, **kwargs)
        self._remaining()
        return value

    def _need(self, prepared, current):
        _, blocks = self._history_blocks(prepared)
        by_start = {block[0]: block for block in blocks}
        selected = None
        for event in self.executions:
            if event["id"] in self.route_seen:
                continue
            self.route_seen.add(event["id"])
            ids = event.get("message_ids")
            block = by_start.get(ids[0]) if ids else None
            feedback = [prepared[i] for i in block[1:]] if block else []
            eligible = (
                feedback
                and event.get("observer_complete") is True
                and event.get("error") is None
                and event.get("returncode") not in (124, 137)
                and not self.unattributed
                and _known_revision(current)
                and event["before"] == event["after"] == current == self.known
                and classify_shell_control(event.get("command"))["status"] == "foreground_syntax"
            )
            if not eligible:
                self.window = []
                continue
            item = dict(
                execution_id=event["id"],
                message_ids=ids,
                revision=copy.deepcopy(current),
                command=event["command"],
                feedback=copy.deepcopy(feedback),
            )
            if self.window and self.window[-1]["revision"] != current:
                self.window = []
            self.window.append(item)
            self.window = self.window[-6:]
            signal = None
            pattern = None
            if len(self.window) >= 3 and all(x["feedback"] == feedback for x in self.window[-3:]):
                signal = "three_same_observations"
                pattern = [feedback]
            else:
                for period in (2, 3):
                    values = [(x["command"], x["feedback"]) for x in self.window[-2 * period :]]
                    if len(values) == 2 * period and values[:period] == values[period:]:
                        signal = "two_complete_cycles"
                        pattern = values[:period]
                        break
            if signal is None:
                continue
            fingerprint = _digest(dict(revision=current, observations=pattern))
            if fingerprint in self.need_seen:
                continue
            self.need_seen.add(fingerprint)
            selected = dict(
                id=fingerprint,
                origin="linked_actual_feedback",
                observer_complete=True,
                signal=signal,
                revision=copy.deepcopy(current),
                command=event["command"],
                feedback=feedback,
                trigger_refs=[
                    dict(execution_id=x["execution_id"], message_ids=x["message_ids"])
                    for x in self.window
                ],
            )
        return selected

    def _current_public_result(self, manifest, current):
        for check in sorted(self.checks, key=lambda c: c.get("finished_ns", 0), reverse=True):
            if (
                not check["finished"]
                or check["error"] is not None
                or not isinstance(check["report"], dict)
                or check["judge_identity"] != manifest["implementation_identity"]
                or check["scope"] != manifest["public_scope"]
                or check["coverage"] != manifest["public_coverage"]
                or check["before"] != check["after"]
                or check["after"] != current
            ):
                continue
            if check.get("origin") == "controller_public_operation":
                if check.get("same_controller_dispatch"):
                    return True
            elif check.get("same_execution") and check["execution_id"] is not None:
                event = self.executions[check["execution_id"]]
                if event["message_ids"] and event["observer_complete"] and event["error"] is None:
                    return True
        return False

    def _source_offers(self, need, current):
        if self.source_capability is None:
            return [], {}
        offers = []
        payloads = {}
        cutoff = min(self.deadline, time.monotonic() + 5)

        def read(path):
            if time.monotonic() >= cutoff:
                raise TimeoutError("source offer preparation allowance")
            raw, source, sha = self._read(path, max_bytes=1024**2)
            if path in current and current[path] != {"kind": "file", "sha256": sha}:
                raise ValueError("source changed during offer preparation")
            return raw, source, sha

        common = dict(
            origin="caller_capability", source_revision=current, busy=False, timeout_seconds=5
        )
        if self.historical is not None:
            for target in self.body_write_scopes:
                path, symbol = target.split("::", 1)
                captured = self.historical.entries.get(path, {})
                if captured.get("status") != "complete" or captured[
                    "captured_revision"
                ] == current.get(path, {}).get("sha256"):
                    continue
                _, source, sha = read(path)
                baseline = self.historical.historical_source(path, sha)
                try:
                    record = _body_audit(path, symbol, baseline, source, sha)
                except (SyntaxError, ValueError, RecursionError):
                    record = None
                if record and (record["outside_ast_changed"] or record["outside_bytes_changed"]):
                    identity = _digest(
                        dict(
                            capability=self.source_capability.identity,
                            body=target,
                            operation="scope_audit",
                        )
                    )
                    payload = dict(records=[record], source_files={path: sha})
                    payloads[identity] = payload
                    offers.append(
                        dict(
                            common,
                            kind="scope_audit",
                            identity=identity,
                            explicit_body_permission=True,
                            complete_historical_source=True,
                            scope_change_observed=True,
                            result_token_bound=self._count(
                                [dict(role="user", content=_json(payload))]
                            ),
                        )
                    )
        literal = literal_source_read(need["command"], self.workspace_aliases)
        if literal:
            path = literal[0]
            if path in self.source_capability.paths:
                _, source, sha = read(path)
                start, end = literal[1:]
                if start > len(source.splitlines()):
                    record = dict(
                        kind="literal_read_bounds",
                        path=path,
                        current_sha256=sha,
                        actual_file_lines=len(source.splitlines()),
                        requested_start=start,
                        requested_end=end,
                        begins_past_eof=True,
                        interpretation="Exact literal read bounds;no semantic correctness finding",
                    )
                    identity = _digest(
                        dict(
                            capability=self.source_capability.identity,
                            path=path,
                            operation="literal_bounds",
                        )
                    )
                    payload = dict(
                        records=[record, _fragment(path, source, sha, 1)], source_files={path: sha}
                    )
                    payloads[identity] = payload
                    offers.append(
                        dict(
                            common,
                            kind="literal_bounds",
                            identity=identity,
                            exact_literal_read=True,
                            read_starts_past_eof=True,
                            result_token_bound=self._count(
                                [dict(role="user", content=_json(payload))]
                            ),
                        )
                    )
        return offers, payloads

    def _packet(self, record, current):
        value = copy.deepcopy(record)
        support = (
            self._support_current(record.get("result"))
            if record["operation"] != "public_validation"
            else True
        )
        value["version_association"] = (
            "current_declared_paths_only"
            if record["before"] == record["after"] == current
            and _known_revision(current)
            and support
            else "stale_or_unknown;not_current_evidence"
        )
        if record["operation"] != "public_validation":
            value["supporting_source_versions_current"] = support
        value["limits"] = (
            "Caller public coverage or source facts only. This is not an SDK action or final acceptance;ordinary coding and the original task remain available."
        )
        return value

    def _support_current(self, payload):
        if (
            not isinstance(payload, dict)
            or not payload.get("source_files")
            or self.source_capability is None
        ):
            return False
        try:
            return all(
                self._read(path, max_bytes=1024**2)[2] == sha
                for path, sha in payload["source_files"].items()
            )
        except (OSError, ValueError, UnicodeError):
            return False

    def _view(self, prepared, records, current):
        view = copy.deepcopy(prepared)
        if records:
            view[1]["content"] += self.HEADER + _json([self._packet(r, current) for r in records])
        return view

    def before_query(self, messages):
        if self.policy == "post_edit_public_v1":
            return self._before_post_edit(messages)
        if not self.policy or self.operation_count >= 2:
            return
        prepared = self._sync(messages)
        current = self._snapshot()
        need = self._need(prepared, current)
        if need is None:
            return
        self._save_event("evidence_need", need)
        try:
            offers, payloads = self._source_offers(need, current)
            operation = self.public_operation
            manifest = operation.manifest if operation is not None else None
            if (
                manifest is not None
                and manifest["observed_paths"]
                and set(manifest["observed_paths"]) == set(self.paths)
                and _known_revision(current)
            ):
                offers.append(
                    dict(
                        kind="public_validation",
                        origin="caller_capability",
                        source_revision=current,
                        identity=operation.contract_sha256,
                        coverage=manifest["public_coverage"],
                        scope=manifest["public_scope"],
                        busy=any(not c["finished"] for c in self.checks),
                        source_mutation_allowed=manifest["source_mutation_allowed"],
                        current_public_check=self._current_public_result(manifest, current),
                        result_token_bound=manifest["result_token_bound"],
                        timeout_seconds=manifest["timeout_seconds"],
                    )
                )
            if not offers:
                self._save_event(
                    "evidence_route_noop",
                    dict(need=need["id"], reason="no_caller_operation_or_relevant_source_offer"),
                )
                return
            reserve = min(30, self.request.budget.wall_seconds / 6)
            # Count the exact unchanged SDK history plus retained observations
            # and metadata before reserving the caller's whole result bound.
            placeholder = dict(
                operation="pending",
                dispatch_key="f" * 64,
                trigger_refs=need["trigger_refs"],
                before=current,
                after=current,
                origin="controller",
                result=None,
                error=None,
                scope=manifest["public_scope"] if manifest else "caller source capability",
                coverage=manifest["public_coverage"] if manifest else "source facts",
                implementation_identity=manifest["implementation_identity"] if manifest else {},
                wall_s=0.0,
                operation_result_ref={"journal": "native-memory.sqlite", "event_sequence": 999999},
            )
            room = (
                self.limit
                - self._count(self._view(prepared, self.results + [placeholder], current))
                - 64
            )
            decision = choose_evidence_operation(
                need=need,
                offers=offers,
                dispatched=self.dispatched,
                remaining_seconds=self._remaining(),
                decision_reserve=reserve,
                available_result_tokens=room,
                operations_used=self.operation_count,
            )
            self._save_event("evidence_route_decision", dict(need=need["id"], decision=decision))
            if decision["operation"] is None:
                return
            if current != self._snapshot():
                self._save_event(
                    "evidence_route_noop",
                    dict(need=need["id"], reason="source_changed_before_dispatch"),
                )
                return
            if decision["operation"] != "public_validation" and not self._support_current(
                payloads[decision["identity"]]
            ):
                self._save_event(
                    "evidence_route_noop",
                    dict(need=need["id"], reason="supporting_source_changed_before_dispatch"),
                )
                return
            self._dispatch(decision, payloads, manifest, current)
        except (OSError, ValueError, TimeoutError, UnicodeError) as error:
            self._save_event(
                "evidence_route_unknown",
                dict(need=need["id"], reason=type(error).__name__ + ": " + str(error)),
            )
        finally:
            self._flush_events()

    def _complete_post_edit_coverage(self, current):
        """Undeclared new files are unknown,not silently covered by old hashes."""
        declared = set(self.paths)
        for prefix in self.request.spec.allowed_paths:
            root = self.workspace / prefix.rstrip("/")
            if root.is_symlink() or not root.is_dir():
                return False
            for base, dirs, files in os.walk(root, followlinks=False):
                if any((Path(base) / name).is_symlink() for name in (*dirs, *files)):
                    return False
                for name in files:
                    path = Path(base) / name
                    if (
                        not path.is_file()
                        or path.relative_to(self.workspace).as_posix() not in declared
                    ):
                        return False
        return _known_revision(current)

    @staticmethod
    def _post_status(record):
        value = record.get("result")
        if record.get("error") or not isinstance(value, dict) or value.get("error"):
            return "unknown"
        process = value.get("process") or {}
        if process.get("timed_out") or process.get("truncated"):
            return "unknown"
        if value.get("source_currency") not in (None, "current_on_declared_paths"):
            return "unknown"
        summary = value.get("summary")
        if not isinstance(summary, dict):
            return "unknown"
        status = summary.get("status")
        if status in ("passed", "failed"):
            return status
        # The existing CPU decoder normalizes structured reports to counts,
        # removing their original status string. Preserve that public format.
        if (
            summary.get("format") == "exit-code-only"
            or summary.get("collection_error")
            or summary.get("zero_tests")
            or summary.get("error", 0)
        ):
            return "unknown"
        passed, failed = summary.get("passed"), summary.get("failed")
        if type(passed) is not int or type(failed) is not int or min(passed, failed) < 0:
            return "unknown"
        if failed > 0 and process.get("exit_code") == 1:
            return "failed"
        if passed > 0 and failed == 0 and process.get("exit_code") == 0:
            return "passed"
        return "unknown"

    def _post_note(self, record, current):
        fresh = (
            record["before"] == record["after"] == current
            and _known_revision(current)
            and self._complete_post_edit_coverage(current)
        )
        result = record.get("result") or {}
        value = dict(
            kind="post_edit_public_v1",
            public_status=self._post_status(record) if fresh else "unknown",
            source_association="current_declared_paths" if fresh else "stale_or_unknown",
            tested_sources=record["after"],
            coverage=record["coverage"],
            report_ref=record["operation_result_ref"],
            read_only_report={
                k: result[k] for k in ("report_id", "record_sha256", "read") if k in result
            },
            authority="Actual caller public feedback;not final acceptance or permission to edit outside the task.",
        )

        def encode():
            return self.HEADER + _json(value)

        if self._count([dict(role="user", content=encode())]) > 768:
            return None
        summary = result.get("summary") if fresh else None
        if isinstance(summary, dict):
            value["summary"] = {}
            omitted = []
            for name in sorted(
                summary, key=lambda k: (k not in ("failures", "diagnostics", "details"), k)
            ):
                value["summary"][name] = copy.deepcopy(summary[name])
                if self._count([dict(role="user", content=encode())]) > 736:
                    value["summary"].pop(name)
                    omitted.append(name)
            if omitted:
                value["summary_fields_omitted"] = omitted
                if self._count([dict(role="user", content=encode())]) > 768:
                    value.pop("summary_fields_omitted")
        return encode()

    def _before_post_edit(self, messages):
        if self.operation_count >= 2:
            return
        prepared = self._sync(messages)
        current = self._snapshot()
        if (
            self.operation_count
            and self.post_edit_last_status != "failed"
            or self.unattributed
            or current != self.known
            or not self._complete_post_edit_coverage(current)
        ):
            return
        candidate = None
        for event in self.executions:
            if event["id"] in self.post_edit_seen:
                continue
            self.post_edit_seen.add(event["id"])
            if (
                event.get("changed")
                and event.get("message_ids")
                and event.get("observer_complete") is True
                and event.get("error") is None
                and event.get("returncode") not in (124, 137)
                and _known_revision(event["before"])
                and event["after"] == current
            ):
                candidate = event
        if candidate is None:
            return
        operation = self.public_operation
        manifest = operation.manifest if operation is not None else None
        if manifest is None:
            return
        if self._current_public_result(manifest, current):
            self._save_event(
                "post_edit_public_reused", dict(execution_id=candidate["id"], revision=current)
            )
            return
        try:
            seconds, calls = self.post_edit_budget.available()
            timeout = min(30.0, manifest["timeout_seconds"], max(0.0, 60.0 - self.post_edit_wall))
            if calls < 1 or timeout <= 0 or seconds < timeout + self.request.budget.request_timeout:
                self._save_event(
                    "post_edit_public_skipped",
                    dict(reason="original_search_or_repair_reserve", execution_id=candidate["id"]),
                )
                return
            if self._count(prepared) + 800 > self.limit:
                self._save_event(
                    "post_edit_public_skipped",
                    dict(
                        reason="complete_history_and_result_do_not_fit",
                        execution_id=candidate["id"],
                    ),
                )
                return
            placeholder = dict(
                before=current,
                after=current,
                coverage=manifest["public_coverage"],
                result={},
                error=None,
                operation_result_ref=dict(journal="native-memory.sqlite", event_sequence=999999),
            )
            if self._post_note(placeholder, current) is None:
                return
            if current != self._snapshot():
                return
            self.post_edit_budget.available()
            decision = dict(
                operation="public_validation",
                identity=operation.contract_sha256,
                dispatch_key=_digest(
                    dict(policy=self.policy, revision=current, operation=operation.contract_sha256)
                ),
                trigger_refs=[
                    dict(execution_id=candidate["id"], message_ids=candidate["message_ids"])
                ],
                timeout_seconds=timeout,
                result_token_bound=manifest["result_token_bound"],
            )
            before_count = len(self.results)
            self._dispatch(decision, {}, manifest, current)
            if len(self.results) > before_count:
                record = self.results[-1]
                self.post_edit_wall += record["wall_s"]
                self.post_edit_last_status = (
                    self._post_status(record)
                    if record["before"] == record["after"] == current
                    else "unknown"
                )
            self.post_edit_budget.available()
        except (OSError, ValueError, TimeoutError) as error:
            self._save_event(
                "post_edit_public_unknown",
                dict(execution_id=candidate["id"], error_class=type(error).__name__),
            )
        finally:
            self._flush_events()

    def _dispatch(self, decision, payloads, manifest, before):
        self.dispatched.add(decision["dispatch_key"])
        self.operation_count += 1
        dispatch = dict(
            id=decision["dispatch_key"],
            trigger_refs=decision["trigger_refs"],
            implementation_identity=manifest["implementation_identity"]
            if decision["operation"] == "public_validation"
            else {"source_capability": self.source_capability.identity},
            scope=manifest["public_scope"]
            if decision["operation"] == "public_validation"
            else "caller-declared source facts",
            coverage=manifest["public_coverage"]
            if decision["operation"] == "public_validation"
            else ["read-only source observation"],
        )
        self._save_event(
            "evidence_operation_dispatched",
            dict(origin="controller", decision=decision, dispatch=dispatch),
        )
        started = time.monotonic()
        cutoff = min(self.deadline, started + decision["timeout_seconds"])
        result = None
        error = None
        raised = None
        try:
            if decision["operation"] == "public_validation":
                with self.controller_public_dispatch(dispatch):
                    result = self.public_operation.invoke(cutoff)
                if not isinstance(result, dict):
                    raise ValueError(
                        "caller public executor must return structured observation data"
                    )
            else:
                result = payloads[decision["identity"]]
        except BaseException as exc:
            error = type(exc).__name__ + ": " + str(exc)
            raised = exc
        try:
            after = self._snapshot()
        except (OSError, ValueError, TimeoutError):
            after = {
                p: dict(kind="unknown", reason="post_operation_source_unavailable")
                for p in self.paths
            }
        if decision["operation"] == "public_validation":
            self.record_controller_return(
                dispatch, before=before, after=after, report=result, error=error
            )
        record = dict(
            origin="controller",
            operation=decision["operation"],
            dispatch_key=decision["dispatch_key"],
            trigger_refs=decision["trigger_refs"],
            before=before,
            after=after,
            result=result,
            error=error,
            scope=dispatch["scope"],
            coverage=dispatch["coverage"],
            implementation_identity=dispatch["implementation_identity"],
            wall_s=time.monotonic() - started,
            acceptance_proof=False,
            operation_result_ref=dict(
                journal="native-memory.sqlite", event_sequence=len(self.events)
            ),
        )
        try:
            actual = self._count([dict(role="user", content=_json(result))])
            record["result_bound_respected"] = actual <= decision["result_token_bound"]
            record["actual_result_tokens"] = actual
        except (OSError, ValueError, TimeoutError):
            record["result_bound_respected"] = False
        self.results.append(record)
        self._save_event("evidence_operation_result", record)
        if raised is not None and not isinstance(raised, Exception):
            raise raised

    def project(self, messages):
        # Pure preparation: no execute/invoke/source reads beyond the normal
        # observer snapshot. Repeated calls never dispatch an operation.
        super().project(messages)
        if self.working_memory is not None:
            note = ""
            if self.policy == "post_edit_public_v1" and self.results:
                note = self._post_note(self.results[-1], self._snapshot()) or ""
            return self.working_memory.project(messages, note)
        if not self.policy or not self.results:
            return messages
        if self.policy == "post_edit_public_v1":
            prepared = self.prepare(messages)
            note = self._post_note(self.results[-1], self._snapshot())
            if note is None:
                return messages
            view = copy.deepcopy(prepared)
            view[1]["content"] += note
            if self._count(view) > self.limit:
                self._save_event(
                    "post_edit_public_projection_omitted", dict(reason="original_history_retained")
                )
                return messages
            self._save_event(
                "post_edit_public_projection_prepared",
                dict(
                    input_sha256=_digest(view),
                    report_ref=self.results[-1]["operation_result_ref"],
                    physical_forwarding_proven=False,
                ),
            )
            return view
        prepared = self.prepare(messages)
        current = self._snapshot()
        view = self._view(prepared, self.results, current)
        if (
            any(not r["result_bound_respected"] for r in self.results)
            or self._count(view) > self.limit
        ):
            # Never evict original task/feedback/history to force module output
            # into capacity. Retain the full original result in the journal and
            # explicitly mark non-delivery instead of claiming it was consumed.
            self._save_event(
                "evidence_projection_unavailable",
                dict(
                    reason="result_or_later_history_exceeds_capacity",
                    result_refs=[r["operation_result_ref"] for r in self.results],
                    SDK_input_preserved=True,
                ),
            )
            self._flush_events()
            return messages
        with self.index.db:
            self.index.db.execute(
                "INSERT INTO native_projections(messages,accounting) VALUES(?,?)",
                (
                    _json(view),
                    _json(
                        dict(
                            status="controller_evidence_projection;HTTP_forwarding_unproven",
                            operations=self.operation_count,
                            original_history_unchanged=True,
                        )
                    ),
                ),
            )
        return view

    def tool(self, args):
        if self.working_memory is not None:
            return self.working_memory.tool(args)
        return super().tool(args)

    def observe_public_report(self, action, result):
        """Record a trusted report result; later match the actual SDK output."""
        with self.lock:
            if self.working_memory is not None and (action == "run" or self.active is not None):
                self.report_deliveries.append(
                    dict(
                        execution_id=self.active["id"] if self.active is not None else None,
                        action=action,
                        report_id=result["report_id"],
                        record_sha256=result["record_sha256"],
                        client_output=json.dumps(
                            dict(ok=True, result=result), ensure_ascii=False, sort_keys=True
                        ),
                    )
                )

    def parse_tool_command(self, command):
        if not self.task_working_memory:
            return super().parse_tool_command(command)
        import shlex

        args = shlex.split(command)
        if any(value in (";", "&&", "||", "|", ">", "<") for value in args):
            raise ValueError("ic-task-memory requires a standalone command")
        return args
