"""Once: real edit→independent review→public observation→original SDK revision."""

from contextlib import contextmanager
import copy
import json
import signal
import sys
import threading
import time

from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.planning.contract_review import (
    ContractReviewPolicy,
    REVIEW_SYSTEM,
    changed_source,
    digest,
    dump,
    lightweight,
    parse_review,
)
from infinicoder.runtime.native_plain import NativePlainObserver

HEADER = "CONTROLLER CONTRACT REVIEW OBSERVATION (not an assistant action or final acceptance)\n"


class ReviewPreparationExpired(Exception):
    pass


class NativeContractReview(NativePlainObserver):
    identity = "native_contract_review_v1"

    def __init__(
        self,
        count_messages,
        *,
        tokenizer_identity,
        implementation_paths=(),
        policy=None,
        reader_factory=None,
        diagnostic_factory=None,
    ):
        super().__init__(
            count_messages,
            tokenizer_identity=tokenizer_identity,
            implementation_paths=implementation_paths,
        )
        self.policy = policy or ContractReviewPolicy()
        if not isinstance(self.policy, ContractReviewPolicy):
            raise TypeError("typed contract review policy required")
        self.reader_factory, self.diagnostic_factory = reader_factory, diagnostic_factory
        self.auxiliary = None
        self.reader = None
        self.diagnostic = None
        self.used = False
        self.bound_once = False
        self.cutoff = None

    def validate(self, request):
        super().validate(request)
        if request.context_adapter is not self:
            raise UnsupportedCore("contract review must own the plain context")
        if any(
            getattr(request, n) is not None
            for n in (
                "planning_adapter",
                "repair_adapter",
                "candidate_search",
                "implementation_proposal",
            )
        ):
            raise UnsupportedCore("contract review is a separate single-run capability")
        transport = request.transport_adapter
        if getattr(transport, "delivery_policy", None) is not None:
            raise UnsupportedCore("contract review uses the common legacy transport")
        if any(
            not callable(getattr(transport, n, None))
            for n in ("limit_request", "remaining_physical_calls", "request_observations")
        ):
            raise UnsupportedCore("contract review requires the current shared physical transport")

    def _check_task(self):
        if time.monotonic() >= self.deadline:
            from infinicoder.runtime.native_run import CoreDeadline

            raise CoreDeadline("shared deadline exhausted during contract review")

    def _source_remaining(self):
        self._check_task()
        left = self.deadline - time.monotonic()
        if self.cutoff is not None:
            left = min(left, self.cutoff - time.monotonic())
        if left <= 0:
            raise ReviewPreparationExpired("contract review preparation exhausted")
        return left

    @contextmanager
    def _preparation(self):
        self._check_task()
        started = time.monotonic()
        self.cutoff = min(self.deadline, started + self.policy.preparation_seconds)
        old_timer = signal.getitimer(signal.ITIMER_REAL) if sys.platform == "linux" else (0, 0)
        altered = (
            sys.platform == "linux"
            and threading.current_thread() is threading.main_thread()
            and (not old_timer[0] or self.cutoff - started < old_timer[0])
        )
        if altered:
            old_handler = signal.getsignal(signal.SIGALRM)

            def expire(signum, frame):
                self._check_task()
                raise ReviewPreparationExpired("five-second review preparation expired")

            signal.signal(signal.SIGALRM, expire)
            signal.setitimer(signal.ITIMER_REAL, max(0.001, self.cutoff - time.monotonic()))
        try:
            yield
        finally:
            self.cutoff = None
            if altered:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)
                if old_timer[0]:
                    signal.setitimer(
                        signal.ITIMER_REAL,
                        max(0.001, old_timer[0] - (time.monotonic() - started)),
                        old_timer[1],
                    )

    def _save_review(self):
        self.report["auxiliary_model_calls"] = self.auxiliary.call_units if self.auxiliary else 0
        path = self.state / "native-contract-review.json"
        temporary = path.with_suffix(".next")
        temporary.write_text(json.dumps(self.report, ensure_ascii=False, indent=2), encoding="utf8")
        temporary.replace(path)

    def _note(self, kind, **data):
        self.report["records"].append(dict(kind=kind, **data))

    def _read(self, path):
        self._source_remaining()
        raw, text, sha = self.reader.session.read(path, max_bytes=131072)
        if digest(text) != sha or self.reader.current_version(path) != sha:
            raise ValueError("review source changed while reading")
        return text

    @contextmanager
    def bind_auxiliary_calls(self, budget):
        if self.auxiliary is not None:
            raise UnsupportedCore("review budget cannot nest")
        self.auxiliary = budget
        try:
            yield
        finally:
            self._save_review()

    @contextmanager
    def bind(self, request, model, environment, deadline):
        if self.bound_once:
            raise UnsupportedCore("contract review cannot resume or repeat")
        self.bound_once = True
        self.request = request
        self.command_outputs = {}
        self.initial_sources = {}
        self.feedback = None
        self.projected = None
        with super().bind(request, model, environment, deadline):
            self.report = dict(
                identity=self.identity,
                status="waiting_for_first_real_edit",
                records=[],
                deliveries=[],
                authority="one independent model hypothesis and real public observation;not final acceptance",
                canonical_history_changed=False,
                history_crop=False,
                automatic_edit=False,
                private_final_used=False,
            )
            self.index.db.execute(
                "CREATE TABLE native_contract_reviews(id INTEGER PRIMARY KEY,event TEXT)"
            )
            self.index.db.commit()
            try:
                if (
                    self.policy.task_kind == "lightweight"
                    or not self.paths
                    or len(self.paths) > 32
                    or self.reader_factory is None
                ):
                    self.used = True
                    self.report["status"] = "N/A;lightweight or no bounded caller source capability"
                else:
                    try:
                        with self._preparation():
                            self.reader = self.reader_factory(
                                request,
                                self.index,
                                environment,
                                count_output=lambda text: self._count(
                                    [dict(role="user", content=text)]
                                ),
                                remaining=self._source_remaining,
                            )
                            if not getattr(self.reader, "namespace", {}).get("root") or not getattr(
                                self.reader, "session", None
                            ):
                                raise ValueError("identified source namespace required")
                            for path in self.paths:
                                try:
                                    self.initial_sources[path] = self._read(path)
                                except FileNotFoundError:
                                    self.initial_sources[path] = ""
                            if self.policy.task_kind == "auto" and lightweight(
                                self.initial_sources
                            ):
                                self.used = True
                                self.report["status"] = "N/A;small dependency-free single function"
                            if not self.used and self.diagnostic_factory is not None:
                                try:
                                    self.diagnostic = self.diagnostic_factory(request, deadline)
                                except (OSError, ValueError) as error:
                                    self._note(
                                        "readonly_diagnostic_unavailable",
                                        error=type(error).__name__ + ": " + str(error)[:300],
                                    )
                    except (
                        ReviewPreparationExpired,
                        TimeoutError,
                        OSError,
                        ValueError,
                        UnicodeError,
                        RecursionError,
                    ) as error:
                        self._check_task()
                        self.used = True
                        self.report.update(
                            status="N/A;source or diagnostic unavailable",
                            error=type(error).__name__ + ": " + str(error)[:300],
                        )
                execute, query = environment.execute, model.query

                def observed_execute(action, *args, **kwargs):
                    ordinal = len(self.executions)
                    result = execute(action, *args, **kwargs)
                    if len(self.executions) == ordinal + 1:
                        output = result.get("output")
                        self.command_outputs[ordinal] = dict(
                            output=output
                            if isinstance(output, str) and len(output) <= 65536
                            else None,
                            returncode=result.get("returncode"),
                        )
                    return result

                def ordinary_query(messages, **kwargs):
                    self.projected = None
                    try:
                        return query(messages, **kwargs)
                    finally:
                        if self.projected is not None:
                            rows = request.transport_adapter.request_observations()
                            after = [
                                r
                                for r in rows or ()
                                if r["attempt"] > self.projected["attempt_marker"]
                            ]
                            sent = any(r["dispatch_attempted"] for r in after)
                            known_not_sent = bool(after) and all(
                                not r["dispatch_attempted"] for r in after
                            )
                            if sent or not known_not_sent:
                                self.feedback = None
                            self.report["deliveries"].append(
                                dict(
                                    self.projected,
                                    status="actual_HTTP_dispatched"
                                    if sent
                                    else "known_not_dispatched"
                                    if known_not_sent
                                    else "unknown;not_replayed",
                                )
                            )
                            self._save_review()

                environment.execute, model.query = observed_execute, ordinary_query
                try:
                    yield self
                finally:
                    environment.execute, model.query = execute, query
            finally:
                self._save_review()
                if self.reader is not None and callable(getattr(self.reader, "close", None)):
                    self.reader.close()

    def before_query(self, messages):
        if self.used or self.auxiliary is None:
            return
        for event in self.executions:
            changed = event.get("changed", ())
            if (
                not changed
                or not event.get("message_ids")
                or not event["observer_complete"]
                or event["error"] is not None
            ):
                continue
            observed = self.command_outputs.get(event["id"], {})
            prepared = self.prepare(messages)
            feedback_id = event["message_ids"][1]
            output = observed.get("output")
            if output is None or output not in prepared[feedback_id].get("content", ""):
                continue
            current = self._snapshot()
            if any(
                current[p] != event["after"][p]
                or p in self.unattributed
                or current[p].get("kind") != "file"
                for p in changed
            ):
                continue
            self.used = True
            if (
                self.deadline - time.monotonic() < 60
                or self.request.transport_adapter.remaining_physical_calls() < 3
            ):
                self.report["status"] = "N/A;less than60s or three remaining physical calls"
                self._save_review()
                return
            if self.public_operation is None and self.diagnostic is None:
                self.report["status"] = "N/A;no declared executable observation capability"
                self._save_review()
                return
            try:
                self._review(event, prepared, observed, current)
            except Exception as error:
                self._check_task()
                self.report.update(
                    status="review_unavailable;ordinary_continues",
                    error=type(error).__name__ + ": " + str(error)[:500],
                )
                self._save_review()
            return

    def _review(self, event, prepared, observation, current):
        self.folder = self.state / "contract-review"
        self.folder.mkdir(exist_ok=True)
        sources = {}
        changed = []
        with self._preparation():
            for path in event["changed"]:
                before = self.initial_sources.get(path)
                if before is None or event["before"][path].get("sha256") not in (
                    None,
                    digest(before),
                ):
                    raise ValueError("before-source provenance unavailable")
                after = self._read(path)
                if digest(after) != current[path]["sha256"]:
                    raise ValueError("review does not target the current edited version")
                sources[path] = after
                changed.append(changed_source(path, before, after))
            # Core owns the exact original public binding. The ordinary SDK's
            # instance template also contains bash-action formatting examples;
            # those are not this independent review's output instructions.
            task = self.auxiliary.public_task
            if not isinstance(task, str) or not task.strip():
                raise ValueError("original Core public task binding unavailable")
            payload = dict(
                original_public_task=task,
                namespace=self.reader.namespace,
                changes=changed,
                last_actual_observation=observation,
                capabilities=dict(
                    existing_public_operation=self.public_operation.manifest
                    if self.public_operation is not None
                    else None,
                    cpu_python=dict(
                        identity=self.diagnostic.identity,
                        source_readonly=True,
                        input_applicability="unverified;preserve original public input forms",
                    )
                    if self.diagnostic
                    else None,
                ),
                authority="Current source and actual observation;concerns and expectations remain hypotheses",
            )
            review_messages = [
                dict(role="system", content=REVIEW_SYSTEM),
                dict(role="user", content=dump(payload)),
            ]
            tokens = self._count(review_messages)
            limit = self.limit * 2 // 3
            if tokens > limit:
                self.report.update(
                    status="N/A;complete contract and changed units do not fit",
                    review_input_tokens=tokens,
                    review_input_limit=limit,
                )
                self._save_review()
                return
        versions = {p: digest(s) for p, s in sources.items()}
        self.report.update(
            status="review_started",
            trigger_execution_id=event["id"],
            source_versions=versions,
            review_input_tokens=tokens,
            review_input_limit=limit,
        )
        (self.folder / "input.json").write_text(dump(review_messages), encoding="utf8")
        self._save_review()
        cutoff = min(
            self.deadline,
            time.monotonic()
            + min(self.policy.review_seconds, (self.deadline - time.monotonic()) / 4),
        )
        from infinicoder.runtime.native_protocol import FormatError

        try:
            response = self.auxiliary.query_once(
                review_messages,
                output_tokens=min(2048, self.request.model_profile.output_reserve),
                absolute_deadline=cutoff,
            )
        except FormatError as error:
            response = error.messages[0]
        (self.folder / "response.json").write_text(dump(response), encoding="utf8")
        choice = (response.get("extra", {}).get("response", {}).get("choices") or [{}])[0]
        decision = parse_review(
            choice.get("message", {}).get("content", response.get("content", "")),
            complete=choice.get("finish_reason") == "stop",
            public_task=task,
            sources=sources,
            public_operation=self.public_operation is not None,
            cpu_probe=self.diagnostic is not None,
        )
        self.report["decision"] = decision
        self._check_task()
        if decision == {"status": "none"}:
            self.report["status"] = "review_none;ordinary_continues"
            self._save_review()
            return
        if any(self.reader.current_version(p) != h for p, h in versions.items()):
            raise ValueError("sources changed during review;no probe executed")
        cutoff = min(
            self.deadline,
            time.monotonic()
            + min(self.policy.probe_seconds, (self.deadline - time.monotonic()) / 4),
        )
        if decision["probe"]["kind"] == "existing_public_operation":
            identity = self.public_operation.manifest["implementation_identity"]
            prior = next(
                (
                    c
                    for c in reversed(self.checks)
                    if c["finished"]
                    and c["error"] is None
                    and c["before"] == c["after"] == current
                    and c["judge_identity"] == identity
                ),
                None,
            )
            if prior is not None:
                result = dict(
                    authority="existing_real_public_check",
                    reused=True,
                    report=prior["report"],
                    scope=prior["scope"],
                    coverage=prior["coverage"],
                )
            else:
                dispatch = dict(
                    id="contract-review:0",
                    implementation_identity=identity,
                    scope=self.public_operation.manifest["public_scope"],
                    coverage=self.public_operation.manifest["public_coverage"],
                )
                with self.controller_public_dispatch(dispatch):
                    report = self.public_operation.invoke(cutoff)
                self.record_controller_return(
                    dispatch, before=current, after=self._snapshot(), report=report
                )
                result = dict(
                    authority="caller_public_operation_observation;not_final",
                    reused=False,
                    report=report,
                )
        else:
            result = self.diagnostic.run(decision["probe"]["script"], absolute_deadline=cutoff)
        self._check_task()
        currency = (
            "current_on_reviewed_paths"
            if all(self.reader.current_version(p) == h for p, h in versions.items())
            else "changed_during_observation;historical_only"
        )
        (self.folder / "probe-result.json").write_text(dump(result), encoding="utf8")
        raw = dump(result)
        visible = (
            result
            if len(raw) <= 6000
            else dict(
                status="bounded_observation_preview",
                preview=raw[:6000],
                truncated=True,
                full_result_sha256=digest(raw),
            )
        )
        self.feedback = dict(
            origin="controller_contract_review",
            reviewed_source_versions=versions,
            source_currency=currency,
            public_basis=decision["public_basis"],
            concern=decision["concern"],
            expected_observation=decision["expected_observation"],
            actual_observation=visible,
            authority="Concern,expected values and generated test applicability are unverified. Execution output is an observation,not a final verdict. Inspect whether the test matches the public contract before revising.",
        )
        self.report.update(
            status="one_actual_observation_available",
            probe_executed=True,
            result=result,
            source_currency=currency,
        )
        with self.index.db:
            self.index.db.execute(
                "INSERT INTO native_contract_reviews VALUES(?,?)", (0, dump(self.feedback))
            )
        self._save_review()

    def project(self, messages):
        original = super().project(messages)
        if self.feedback is None:
            return original
        self._check_task()
        value = copy.deepcopy(self.feedback)
        try:
            if any(
                self.reader.current_version(p) != h
                for p, h in value["reviewed_source_versions"].items()
            ):
                value["source_currency"] = "historical;source_changed_after_observation"
        except (OSError, ValueError, UnicodeError):
            value["source_currency"] = "historical;current_source_unknown"
        projected = copy.deepcopy(original) + [dict(role="user", content=HEADER + dump(value))]
        try:
            fits = self._count(projected) <= self.limit
        except (TimeoutError, OSError, ValueError):
            self._check_task()
            fits = False
        if not fits:
            self.report["deliveries"].append(
                dict(status="omitted;complete original input preserved")
            )
            self.feedback = None
            self._save_review()
            return original
        receipts = self.request.transport_adapter.request_observations()
        self.projected = dict(
            attempt_marker=max((r["attempt"] for r in receipts or ()), default=0),
            input_sha256=digest(dump(self.prepare(projected))),
        )
        return projected
