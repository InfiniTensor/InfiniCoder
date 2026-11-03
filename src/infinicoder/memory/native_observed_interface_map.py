"""Default-off,transient representation of source already delivered to the SDK."""

from contextlib import contextmanager
import copy
import hashlib
import json
import re
import signal
import sys
import threading
import time

from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.runtime.native_plain import NativePlainObserver
from infinicoder.memory.observed_interface_map import (
    ObservedSource,
    build_interface_map,
    dump,
    RENDERER,
)
from infinicoder.memory.source.observed_source_reads import ReadObservationWorkset
from infinicoder.memory.task_evidence_context import TaskEvidenceContext

HEADER = "OBSERVED INTERFACE MAP (controller representation,not execution output)\n"


class MapMaintenanceExpired(Exception):
    pass


class NativeObservedInterfaceMap(NativePlainObserver):
    identity = "native_observed_interface_map_v1"
    _call_seconds = 5
    _task_seconds = 10

    def __init__(
        self,
        count_messages,
        *,
        tokenizer_identity,
        implementation_paths=(),
        reader_factory=None,
        public_sources=None,
        scope_quotes=(),
    ):
        super().__init__(
            count_messages,
            tokenizer_identity=tokenizer_identity,
            implementation_paths=implementation_paths,
        )
        if reader_factory is not None and not callable(reader_factory):
            raise ValueError("explicit source namespace factory required")
        self.reader_factory = reader_factory
        self.public_sources = dict(public_sources or {})
        self.scope_quotes = tuple(scope_quotes)
        if len(self.public_sources) > 32:
            raise ValueError("at most32 explicitly supplied public FILES sources")
        for path, text in self.public_sources.items():
            ObservedSource(path, text, "caller-declared;not-yet-delivered")
        if len(self.scope_quotes) > 2 or any(
            not isinstance(q, str) or not q for q in self.scope_quotes
        ):
            raise ValueError("at most2 exact public scope quotes")
        self.reader = None
        self._used = False

    def validate(self, request):
        super().validate(request)
        if request.context_adapter is not self:
            raise UnsupportedCore("observed map must own its context adapter")
        if any(
            getattr(request, name, None) is not None
            for name in (
                "planning_adapter",
                "repair_adapter",
                "candidate_search",
                "implementation_proposal",
            )
        ):
            raise UnsupportedCore("observed map is a separate plain single-run variable")
        if any(q not in request.spec.task for q in self.scope_quotes):
            raise ValueError("scope quotes must be exact original public task text")

    def _source_remaining(self):
        self._check_task()
        left = super()._remaining()
        if self.cutoff is not None:
            left = min(left, self.cutoff - time.monotonic())
        if left <= 0:
            raise MapMaintenanceExpired("observed-map maintenance exhausted")
        return left

    def _check_task(self):
        if time.monotonic() >= self.deadline:
            from infinicoder.runtime.native_run import CoreDeadline

            raise CoreDeadline("original task deadline expired during source representation")

    @contextmanager
    def _maintenance(self):
        self._check_task()
        started = time.monotonic()
        budget = min(
            self._call_seconds - (self.query_spent if self.in_query else 0),
            self._task_seconds - self.spent,
        )
        if budget <= 0:
            raise MapMaintenanceExpired("task observed-map maintenance exhausted")
        self.cutoff = min(self.deadline, started + budget)
        alarm_left = self.cutoff - time.monotonic()
        if alarm_left <= 0:
            self._check_task()
            raise MapMaintenanceExpired("optional maintenance deadline reached")
        altered = False
        old_handler = old_timer = None
        # Keep an earlier Core deadline untouched. A shorter optional alarm is
        # restored with elapsed time deducted,so it cannot renew task time.
        if sys.platform == "linux" and threading.current_thread() is threading.main_thread():
            old_timer = signal.getitimer(signal.ITIMER_REAL)
            if not old_timer[0] or alarm_left < old_timer[0]:
                old_handler = signal.getsignal(signal.SIGALRM)

                def expire(signum, frame):
                    self._check_task()
                    raise MapMaintenanceExpired("five-second observed-map deadline")

                signal.signal(signal.SIGALRM, expire)
                signal.setitimer(signal.ITIMER_REAL, alarm_left)
                altered = True
        raised = None
        try:
            yield
        except BaseException as error:
            raised = error
            raise
        finally:
            elapsed = time.monotonic() - started
            self.spent += elapsed
            self.cutoff = None
            if self.in_query:
                self.query_spent += elapsed
            if altered:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)
                if old_timer[0]:
                    signal.setitimer(
                        signal.ITIMER_REAL, max(0.001, old_timer[0] - elapsed), old_timer[1]
                    )
            if not isinstance(raised, KeyboardInterrupt):
                self._check_task()

    def _note(self, kind, **value):
        self.records.append(dict(kind=kind, query=self.query_number, **value))

    def _optional(self, function, *args):
        try:
            with self._maintenance():
                return function(*args)
        except (
            MapMaintenanceExpired,
            TimeoutError,
            OSError,
            ValueError,
            KeyError,
            UnicodeError,
            RecursionError,
        ) as error:
            self._check_task()
            self._note(
                "optional_observation_unavailable",
                operation=function.__name__,
                error_type=type(error).__name__,
            )
            return None

    @contextmanager
    def bind(self, request, model, environment, deadline):
        if self._used:
            raise UnsupportedCore("observed maps are task-local and cannot resume")
        self._used = True
        self.cutoff = None
        self.spent = 0.0
        self.query_number = 0
        self.records = []
        self.maps = []
        self.observed = {}
        self.in_query = False
        self.query_spent = 0.0
        self.active_map = None
        self.map_slots = {}
        self.map_versions = set()
        self.attempted_sets = set()
        self.staged = []
        self.current_projection = None
        self.map_attempts = {}
        self.map_confirmed = {}
        self.map_unknown = {}
        self.map_predispatch = {}
        self.reader = None
        self.reads = None
        self.linked = set()
        self.request = request
        self.public_done = False
        with super().bind(request, model, environment, deadline):

            def open_reader():
                candidate = self.reader_factory(
                    request,
                    self.index,
                    environment,
                    count_output=lambda text: self._count([dict(role="user", content=text)]),
                    remaining=self._source_remaining,
                )
                if (
                    not getattr(candidate, "identity", None)
                    or not getattr(candidate, "namespace", {}).get("root")
                    or not getattr(candidate, "namespace", {}).get("environment")
                    or not callable(getattr(getattr(candidate, "session", None), "read", None))
                    or any(
                        not callable(getattr(candidate, name, None))
                        for name in ("current_version", "_record_view")
                    )
                ):
                    raise ValueError("identified actual source namespace required")
                self.task_state = TaskEvidenceContext(self.index, remaining=self._source_remaining)
                namespace = hashlib.sha256(
                    dump(dict(task=request.spec.task, namespace=candidate.namespace)).encode()
                ).hexdigest()
                reads = ReadObservationWorkset(
                    candidate,
                    self.task_state,
                    namespace_id=namespace,
                    remaining=self._source_remaining,
                )
                self.reader, self.reads = candidate, reads

            if self.reader_factory is not None:
                self._optional(open_reader)
            query, execute = model.query, environment.execute

            def actual_query(messages, **kwargs):
                self.query_number += 1
                self.staged = []
                self.current_projection = None
                self.in_query = True
                self.query_spent = 0.0
                returned = False
                probe = getattr(
                    getattr(request, "transport_adapter", None), "request_observations", None
                )

                def receipts():
                    try:
                        rows = probe() if callable(probe) else None
                        return (
                            rows
                            if rows is not None
                            and all(
                                isinstance(r, dict) and type(r.get("attempt")) is int for r in rows
                            )
                            else None
                        )
                    except Exception:
                        return None  # optional receipt failure is unknown,not a new agent failure

                before = receipts()
                marker = (
                    max((r["attempt"] for r in before), default=0) if before is not None else None
                )
                try:
                    result = query(messages, **kwargs)
                    returned = True
                    return result
                finally:
                    if self.current_projection is not None:
                        number = self.current_projection
                        after = receipts()
                        physical = (
                            [r for r in after if r["attempt"] > marker]
                            if after is not None and marker is not None
                            else []
                        )
                        self.map_attempts[number] += 1
                        if returned or any(r["status"] == "response_received" for r in physical):
                            outcome = "confirmed_response"
                            self.map_confirmed[number] += 1
                            self.map_slots[number] += 1
                        elif physical and all(
                            not r["dispatch_attempted"]
                            and r["status"]
                            in (
                                "context_rejected_before_dispatch",
                                "invalid_request",
                                "not_dispatched",
                            )
                            for r in physical
                        ):
                            outcome = "confirmed_not_dispatched"
                            self.map_predispatch[number] += 1
                        else:
                            outcome = "unknown;conservative_exposure_reservation"
                            self.map_unknown[number] += 1
                            self.map_slots[number] += 1
                        self._note(
                            "ordinary_query_outcome",
                            map=number,
                            outcome=outcome,
                            prepared_attempts=self.map_attempts[number],
                            reserved_logical_inputs=self.map_slots[number],
                            confirmed=self.map_confirmed[number],
                            unknown=self.map_unknown[number],
                            physical_receipts=physical,
                        )
                    self.in_query = False
                    self._save()

            def actual_read(action, *args, **kwargs):
                token = (
                    self._optional(self.reads.begin_read, action)
                    if self.reads is not None
                    else None
                )
                self._check_task()
                ordinal = len(self.executions)
                if "timeout" in kwargs:
                    kwargs = dict(
                        kwargs,
                        timeout=min(
                            kwargs["timeout"], max(0.001, self.deadline - time.monotonic())
                        ),
                    )
                result = execute(action, *args, **kwargs)
                if self.reads is not None and len(self.executions) == ordinal + 1:
                    self._optional(self.reads.finish_read, token, result, self.executions[ordinal])
                return result

            model.query, environment.execute = actual_query, actual_read
            try:
                yield self
            finally:
                model.query, environment.execute = query, execute
                close = getattr(self.reader, "close", None)
                if callable(close):
                    close()
                self._save()
        info = self.state / "native-memory-info.json"
        if info.exists():
            value = json.loads(info.read_bytes())
            value.update(
                identity=self.identity,
                automatic_source="representation_of_delivered_sources",
                projection="Canonical SDK history unchanged;at most one transient observed relation card",
                automatic_compression=False,
                extra_model_calls=0,
            )
            info.write_text(dump(value), encoding="utf8")

    def _save(self):
        value = dict(
            identity=self.identity,
            renderer=RENDERER,
            extra_model_calls=0,
            canonical_history_changed=False,
            history_compression=False,
            planner=False,
            resume_supported=False,
            source_namespace=getattr(self.reader, "namespace", None),
            maintenance_wall_s=self.spent,
            maps=self.maps,
            records=self.records,
            lifetime=dict(
                prepared_attempts=self.map_attempts,
                confirmed_responses=self.map_confirmed,
                unknown_reserved=self.map_unknown,
                confirmed_not_dispatched=self.map_predispatch,
                reserved_logical_inputs=self.map_slots,
            ),
            observed_sources={
                p: dict(
                    sha=s.source_sha256,
                    observed_lines=s.observed_lines,
                    message_ref=s.delivered_message_ref,
                )
                for p, s in self.observed.items()
            },
            limits=dict(
                query_maintenance_seconds=self._call_seconds,
                task_seconds=self._task_seconds,
                map_versions=2,
                max_reserved_logical_inputs_per_version=3,
                tokens=[768, 384],
            ),
            actual_delivery_authority="Physical HTTP payloads;successful SDK response confirms staged source delivery",
            capability_gain_proven=False,
        )
        path = self.state / "native-observed-interface-map.json"
        temporary = path.with_suffix(".next")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf8")
        temporary.replace(path)

    def _stage_public(self, prepared):
        if (
            self.public_done
            or not self.public_sources
            or len(prepared) < 2
            or prepared[1].get("role") != "user"
        ):
            return
        content = prepared[1].get("content", "")
        for match in re.finditer(r"FILES:\s*", content):
            try:
                values, _ = json.JSONDecoder().raw_decode(content[match.end() :])
            except (ValueError, TypeError):
                continue
            if isinstance(values, dict) and all(
                values.get(p) == text for p, text in self.public_sources.items()
            ):
                for path, text in self.public_sources.items():
                    self.staged.append(
                        dict(
                            path=path,
                            sha=hashlib.sha256(text.encode()).hexdigest(),
                            lines=(),
                            message=1,
                            kind="explicit_public_FILES",
                        )
                    )
                return

    def _stage_reads(self, messages, prepared):
        if self.reads is None:
            return
        for event in self.executions:
            ids = event.get("message_ids")
            if ids and event["id"] in self.reads.pending:
                self.reads.link_result(event, messages[ids[1]], prepared[ids[1]])
            record = self.reads.records.get(event["id"])
            if (
                record is None
                or record["status"] != "linked_exact_source_observation"
                or event["id"] in self.linked
            ):
                continue
            for item in record["source_slices"]:
                self.staged.append(
                    dict(
                        path=item["path"],
                        sha=item["source_sha256"],
                        lines=(tuple(item["observed_lines"]),),
                        message=ids[1],
                        kind="exact_ordinary_read",
                        execution_id=event["id"],
                    )
                )

    def _confirm_sources(self):
        self.reads._namespace()
        if any(item["kind"] == "explicit_public_FILES" for item in self.staged):
            self.public_done = True
        for item in self.staged:
            self._source_remaining()
            path = item["path"]
            if "execution_id" in item:
                self.linked.add(item["execution_id"])
            try:
                raw, text, version = self.reader.session.read(path, max_bytes=1_000_000)
                if version != item["sha"] or self.reader.current_version(path) != version:
                    continue
            except (OSError, ValueError, KeyError, UnicodeError):
                continue
            # This definition representation is UTF-8 text;BOM or differing raw
            # bytes are not silently normalized into a different source hash.
            if hashlib.sha256(text.encode()).hexdigest() != version:
                continue
            source = ObservedSource(
                path,
                text,
                "query:" + str(self.query_number) + ":message:" + str(item["message"]),
                item["lines"],
            )
            old = self.observed.get(path)
            if (
                old is None
                or old.source_sha256 != source.source_sha256
                or not all(
                    old.covers(a, b)
                    for a, b in (source.observed_lines or ((1, len(text.splitlines())),))
                )
            ):
                self.observed[path] = source
                if len(self.observed) > 32:
                    self.observed.pop(next(iter(self.observed)))
            if "execution_id" in item:
                self.linked.add(item["execution_id"])
            self._note(
                "source_delivery_confirmed",
                path=path,
                sha=version,
                message=item["message"],
                channel=item["kind"],
            )

    def decision_observed(self, pending):
        # NativeEpisodeMemory calls this only after an actual SDK query has
        # returned. Prepared but unsent/truncated feedback is not promoted.
        if self.reader is not None:
            self._optional(self._confirm_sources)
        if self.current_projection is not None:
            self._note("map_response_returned", map=self.current_projection)

    def _current_sources(self):
        values = []
        for source in self.observed.values():
            self._source_remaining()
            try:
                current = self.reader.current_version(source.path)
            except (OSError, ValueError, UnicodeError):
                continue
            if current == source.source_sha256:
                values.append(source)
        return values

    def _make_map(self):
        self.reads._namespace()
        if self.active_map is not None:
            if (
                self.active_map.current(self.reader.current_version)
                and self.map_slots[len(self.maps) - 1] < 3
            ):
                return
            self._note(
                "map_withdrawn",
                map=len(self.maps) - 1,
                reason="changed/unknown source or three-query lifetime exhausted",
            )
            self.active_map = None
        if len(self.maps) >= 2:
            return
        sources = self._current_sources()
        if len(sources) < 2:
            return
        previous = self.maps[-1]["source_versions"] if self.maps else {}
        if previous and not any(
            s.path in previous and previous[s.path] != s.source_sha256 for s in sources
        ):
            return
        key = (
            RENDERER,
            self.tokenizer_identity,
            tuple(sorted((s.path, s.source_sha256, s.observed_lines) for s in sources)),
        )
        if key in self.attempted_sets:
            return
        self.attempted_sets.add(key)
        by_path = {s.path: s for s in sources}

        def exact(path):
            raw, text, version = self.reader.session.read(path, max_bytes=1_000_000)
            if (
                version != by_path[path].source_sha256
                or self.reader.current_version(path) != version
            ):
                raise ValueError("observed source changed during indexing")
            return raw, text, version

        self.index.refresh_paths(by_path, read_source=exact)
        card = build_interface_map(
            self.index,
            sources,
            public_contract_ref="canonical-message:1",
            namespace=self.reader.namespace,
            current_version=self.reader.current_version,
            count_text=lambda text: self._count([dict(role="user", content=HEADER + text)]),
            remaining=self._source_remaining,
            scope_quotes=self.scope_quotes,
        )
        if card is None:
            return
        if previous and not any(
            p in previous and previous[p] != v for p, v in card.source_versions.items()
        ):
            return
        version = tuple(sorted(card.source_versions.items()))
        if version in self.map_versions:
            return
        self.map_versions.add(version)
        self.active_map = card
        number = len(self.maps)
        self.map_slots[number] = 0
        self.map_attempts[number] = self.map_confirmed[number] = self.map_unknown[
            number
        ] = self.map_predispatch[number] = 0
        self.maps.append(
            dict(
                payload=card.payload,
                tokens=card.tokens,
                source_versions=card.source_versions,
                formed_at_query=self.query_number,
            )
        )
        self._note("map_formed", map=number, tokens=card.tokens)

    def before_query(self, messages):
        if self.reader is None:
            return
        prepared = self.prepare(messages)
        self._optional(self._stage_public, prepared)
        self._optional(self._stage_reads, messages, prepared)
        self._optional(self._make_map)

    def project(self, messages):
        super().project(messages)
        if self.reader is None or self.active_map is None:
            return messages

        def represent():
            self.reads._namespace()
            if not self.active_map.current(self.reader.current_version):
                self.active_map = None
                return messages
            number = len(self.maps) - 1
            if self.map_slots[number] >= 3:
                return messages
            projected = copy.deepcopy(messages)
            projected.insert(2, dict(role="user", content=HEADER + self.active_map.text))
            if self._count(projected) > self.limit:
                return messages
            self.current_projection = number
            record = dict(
                map=number,
                original_sha=hashlib.sha256(dump(self.prepare(messages)).encode()).hexdigest(),
                projected_sha=hashlib.sha256(dump(self.prepare(projected)).encode()).hexdigest(),
                query=self.query_number,
                status="observed_map_prepared;physical_HTTP_proves_forwarding",
                canonical_history_changed=False,
            )
            self._note("map_input_prepared", **{k: v for k, v in record.items() if k != "query"})
            with self.index.db:
                self.index.db.execute(
                    "INSERT INTO native_projections(messages,accounting) VALUES(?,?)",
                    (dump(self.prepare(projected)), dump(record)),
                )
            return projected

        result = self._optional(represent)
        return messages if result is None else result
