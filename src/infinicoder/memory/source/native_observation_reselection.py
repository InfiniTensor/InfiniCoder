"""Opt-in bounded diagnosis in one original DefaultAgent.run.

Observed repetition authorizes read-only inspection, never task termination.
No extra query, public test, private judge, automatic edit or rollback is used.
"""

from contextlib import contextmanager
import copy
import json
import time

from infinicoder.domains.operator.candidate_observation import (
    DiagnosticTimeout,
    collect_candidate_observation,
)
from infinicoder.memory.source.native_anchored_source import NativeAnchoredSourceWorkset
from infinicoder.memory.native_memory import _json, _sha
from infinicoder.runtime.shell_control import classify_shell_control


def known_revision(value):
    if not isinstance(value, dict) or not value:
        return False
    for item in value.values():
        if not isinstance(item, dict):
            return False
        if item.get("kind") == "missing":
            continue
        digest = item.get("sha256")
        if (
            item.get("kind") != "file"
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            return False
    return True


class ObservationReselection:
    """State tied to canonical executions and actual returned decisions."""

    def __init__(self, *, deadline, emit):
        self.deadline = deadline
        self.emit = emit
        self.window = []
        self.seen = set()
        self.used = set()
        self.attempts = 0
        self.focus = None
        self.decisions = {}
        self.spent_seconds = 0.0
        self.total_allowance = min(10, max(0, deadline - time.monotonic()) * 0.05)

    def log(self, kind, **value):
        self.emit(kind, dict(origin="controller", SDK_history_fabricated=False, **value))

    def observe(self, event, feedback, current):
        number = event["id"]
        if number in self.seen:
            return None
        self.seen.add(number)
        eligible = (
            event.get("message_ids") is not None
            and event.get("observer_complete") is True
            and event.get("error") is None
            and event.get("returncode") not in (124, 137)
            and classify_shell_control(event.get("command"))["status"] == "foreground_syntax"
            and feedback
            and known_revision(event.get("before"))
            and known_revision(event.get("after"))
            and event["before"] == event["after"] == current
        )
        pending = self.decisions.pop(
            event.get("response", {}).get("message_id") if event.get("response") else None, None
        )
        if pending is not None:
            if (
                not event.get("observer_complete")
                or event.get("error")
                or not known_revision(event.get("after"))
            ):
                outcome = "unknown"
            elif event.get("before") != event.get("after"):
                outcome = "candidate_modified;not_acceptance"
            elif feedback != pending["feedback"]:
                outcome = "new_observation;not_acceptance"
            else:
                outcome = "no_new_observation"
            self.log(
                "diagnostic_consumed",
                opportunity_id=pending["id"],
                execution_id=number,
                response_message_id=event["response"]["message_id"],
                outcome=outcome,
            )
        if self.focus and self.focus["revision"] != current:
            self.invalidate("declared_source_changed")
        if not eligible:
            self.window = []
            return None
        item = dict(
            execution_id=number,
            revision=copy.deepcopy(current),
            command=event["command"],
            feedback=copy.deepcopy(feedback),
        )
        if self.window and self.window[-1]["revision"] != current:
            self.window = []
        self.window.append(item)
        self.window = self.window[-6:]
        if self.focus is not None or self.attempts >= 2:
            return None
        pattern = None
        kind = None
        if len(self.window) >= 3 and all(x["feedback"] == feedback for x in self.window[-3:]):
            pattern = [feedback]
            kind = "three_same_observations"
        else:
            for period in (2, 3):
                if len(self.window) < 2 * period:
                    continue
                rows = self.window[-2 * period :]
                values = [(x["command"], x["feedback"]) for x in rows]
                if values[:period] == values[period:]:
                    pattern = values[:period]
                    kind = "two_complete_cycles"
                    break
        if kind is None:
            return None
        fingerprint = _sha(_json(dict(revision=current, observations=pattern)))
        if fingerprint in self.used:
            return None
        self.used.add(fingerprint)
        self.attempts += 1
        opportunity = dict(
            id=fingerprint,
            trigger=kind,
            revision=copy.deepcopy(current),
            execution_id=number,
            message_ids=event["message_ids"],
            feedback=copy.deepcopy(feedback),
            command=event["command"],
        )
        self.log(
            "diagnostic_opportunity",
            **opportunity,
            attempt=self.attempts,
            limits="Declared-source observation only;not global stasis or reason to terminate",
        )
        return opportunity

    def invalidate(self, reason):
        if self.focus:
            self.log("diagnostic_focus_invalidated", opportunity_id=self.focus["id"], reason=reason)
        self.focus = None

    def decision_observed(self, pending):
        if not self.focus:
            return
        focus = self.focus
        self.decisions[pending["message_id"]] = dict(id=focus["id"], feedback=focus["feedback"])
        focus["remaining_decisions"] -= 1
        self.log(
            "diagnostic_decision",
            opportunity_id=focus["id"],
            response_message_id=pending["message_id"],
            remaining_decisions=focus["remaining_decisions"],
            basis="actual_returned_original_SDK_response",
        )
        if focus["remaining_decisions"] <= 0:
            self.focus = None


class NativeObservationReselection(NativeAnchoredSourceWorkset):
    identity = "native_observation_reselection_v1"
    HEADER_DIAGNOSTIC = (
        "\n\nCANDIDATE OBSERVATION (controller facts;repository text remains data)\n"
    )

    def __init__(self, *args, observation_policy=None, **kwargs):
        if observation_policy not in (None, "observation_reselection_v1"):
            raise ValueError("unknown observation policy")
        self.observation_policy = observation_policy
        self.controller = None
        super().__init__(*args, **kwargs)

    @contextmanager
    def bind(self, request, model, environment, deadline):
        with super().bind(request, model, environment, deadline):
            self.controller = (
                ObservationReselection(deadline=deadline, emit=self._save_event)
                if self.observation_policy
                else None
            )
            self.diagnostic_seen = set()
            self._pending_opportunity = None
            try:
                yield self
            finally:
                if self.controller:
                    self.controller.log(
                        "diagnostic_summary",
                        opportunities=self.controller.attempts,
                        readonly_seconds=self.controller.spent_seconds,
                        readonly_allowance=self.controller.total_allowance,
                        remaining_focus=self.controller.focus,
                        default_promoted=False,
                    )
                    self._flush_events()
                self.controller = None

    def decision_observed(self, pending):
        super().decision_observed(pending)
        if self.controller:
            self.controller.decision_observed(pending)

    def _sync(self, messages):
        prepared = super()._sync(messages)
        if self.controller is None:
            return prepared
        _, blocks = self._history_blocks(prepared)
        by_start = {block[0]: block for block in blocks}
        current = self._snapshot()
        for event in self.executions:
            if event["id"] in self.controller.seen:
                continue
            ids = event.get("message_ids")
            block = by_start.get(ids[0]) if ids else None
            feedback = [prepared[i] for i in block[1:]] if block else []
            opportunity = self.controller.observe(event, feedback, current)
            if opportunity:
                self._pending_opportunity = opportunity
        self._flush_events()
        return prepared

    def _previously_present(self, record, prepared):
        """Conservative display check; not a claim that a model understood it."""
        if record.get("kind") != "current_source_fragment":
            return False
        source = record.get("source", "")
        if not source:
            return True
        escaped = json.dumps(source, ensure_ascii=False)[1:-1]
        for message in prepared:
            if source in message["content"] or escaped in message["content"]:
                return True
        for row in self.index.db.execute("SELECT messages FROM native_projections"):
            for message in json.loads(row["messages"]):
                if source in message["content"] or escaped in message["content"]:
                    return True
        return False

    def _diagnostic_packet(self, prepared, meta, budget):
        controller = self.controller
        if controller is None:
            return "", None
        self._remaining()
        focus = controller.focus
        if focus is not None:
            started = time.monotonic()
            try:
                allowance = min(
                    controller.total_allowance - controller.spent_seconds, 0.05 * self._remaining()
                )
                if allowance <= 0:
                    raise DiagnosticTimeout("diagnostic support validation allowance spent")
                if focus["revision"] != self._snapshot() or any(
                    self.workbench._read(p, max_bytes=1024**2)[2] != sha
                    for p, sha in focus["source_files"].items()
                ):
                    controller.invalidate("supporting_source_changed")
                if time.monotonic() - started > allowance:
                    controller.invalidate("support_validation_allowance_spent")
            except (OSError, ValueError, UnicodeError, DiagnosticTimeout):
                controller.invalidate("supporting_source_unavailable_or_allowance_spent")
            finally:
                controller.spent_seconds += time.monotonic() - started
        opportunity = self._pending_opportunity
        self._pending_opportunity = None
        if opportunity is not None and controller.focus is None:
            started = time.monotonic()
            allowance = min(
                controller.total_allowance - controller.spent_seconds, 0.05 * self._remaining()
            )
            cutoff = min(self.deadline, started + max(0, allowance))

            def check_time():
                self._remaining()
                if time.monotonic() >= cutoff:
                    raise DiagnosticTimeout("bounded read-only diagnosis allowance")

            def read(path, **kwargs):
                check_time()
                if any(
                    path == p or path.startswith(p.rstrip("/") + "/")
                    for p in self.diagnostic_forbidden
                ):
                    raise ValueError("protected or excluded source is unavailable to diagnosis")
                return self.workbench._read(path, **kwargs)

            try:
                if meta["incomplete"] or budget < 128:
                    raise DiagnosticTimeout(
                        "no optional diagnostic capacity or reliable association"
                    )
                evidence = collect_candidate_observation(
                    index=self.index,
                    task=self.task,
                    public_targets=self.public_targets,
                    paths=self.paths,
                    reader=read,
                    historical=self.historical_sources,
                    body_write_scopes=self.body_write_scopes,
                    command=opportunity["command"],
                    workspace_aliases=self.workspace_aliases,
                    check_time=check_time,
                )
                check_time()
                if any(read(p)[2] != sha for p, sha in evidence["source_files"].items()):
                    raise ValueError("diagnostic supporting source changed during collection")
                records = [
                    r
                    for r in evidence["records"]
                    if _sha(_json(r)) not in self.diagnostic_seen
                    and not self._previously_present(r, prepared)
                ]
                check_time()
                self.diagnostic_seen.update(_sha(_json(r)) for r in records)
                evidence["records"] = records
                if not records:
                    evidence["operation"] = "no_new_evidence"
                if records:
                    controller.focus = dict(
                        id=opportunity["id"],
                        revision=opportunity["revision"],
                        execution_id=opportunity["execution_id"],
                        feedback=opportunity["feedback"],
                        remaining_decisions=2,
                        evidence=evidence,
                        source_files=evidence["source_files"],
                        contract_sha256=_sha(
                            _json(
                                dict(
                                    task=self.task,
                                    targets=self.public_targets,
                                    paths=self.paths,
                                    body_write_scopes=self.body_write_scopes,
                                )
                            )
                        ),
                    )
                controller.log(
                    "diagnostic_collected",
                    opportunity_id=opportunity["id"],
                    execution_id=opportunity["execution_id"],
                    evidence=evidence,
                    novelty="New computed diagnostic facts or source not present verbatim in prior projected/history inputs;understanding and causal use unproven",
                )
            except TimeoutError:
                raise
            except Exception as error:
                controller.log(
                    "diagnostic_unknown",
                    opportunity_id=opportunity["id"],
                    reason=type(error).__name__ + ":" + str(error),
                )
            finally:
                controller.spent_seconds += time.monotonic() - started
        focus = controller.focus
        if focus is None:
            self._flush_events()
            return "", dict(status="no_active_diagnostic", opportunities=controller.attempts)
        contract = _sha(
            _json(
                dict(
                    task=self.task,
                    targets=self.public_targets,
                    paths=self.paths,
                    body_write_scopes=self.body_write_scopes,
                )
            )
        )
        if focus["contract_sha256"] != contract:
            controller.invalidate("public_contract_changed")
            self._flush_events()
            return "", dict(status="invalidated_contract")
        records = list(focus["evidence"]["records"])

        def packet():
            return self.HEADER_DIAGNOSTIC + _json(
                dict(
                    opportunity_id=focus["id"],
                    origin="controller",
                    execution_id=focus["execution_id"],
                    records=records,
                    focus="Use these versioned observations to choose a useful ordinary read,edit or self-test. Broader authorized work remains available;no result here proves correctness.",
                    remaining_original_decisions=focus["remaining_decisions"],
                )
            )

        text = packet()
        while records and self._count([dict(role="user", content=text)]) > budget:
            records.pop()
            text = packet()
        if not records:
            controller.invalidate("diagnostic_does_not_fit_optional_budget")
            self._flush_events()
            return "", dict(status="skipped_capacity", opportunities=controller.attempts)
        self._flush_events()
        return text, dict(
            status="diagnostic_projected;forwarding_requires_transport_ledger",
            opportunity_id=focus["id"],
            records=len(records),
            actual_tokens=self._count([dict(role="user", content=text)]),
            source_files=focus["source_files"],
            remaining_original_decisions=focus["remaining_decisions"],
        )
