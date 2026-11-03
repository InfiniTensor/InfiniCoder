"""One evidence-based decision after a verified actionless length failure.

The next original SDK query is projected once. No new query, command, checker,
assistant message, repair loop, or acceptance signal is created here.
"""

from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path

from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.memory.native_memory import MemoryCapacityError
from infinicoder.memory.native_task_evidence_context import NativeTaskEvidenceContext
from infinicoder.memory.source.source_packet_budget import bounded_prefix
from infinicoder.memory.task_evidence_context import ranges, sha
from infinicoder.memory.task_history_archive import MODEL_ROOT
from infinicoder.planning.task_source_plan import dump


def last_attempt(state):
    path = Path(state) / "native-api.json"
    if not path.is_file():
        return 0
    rows = json.loads(path.read_bytes()).get("calls", [])
    return max((r["attempt"] for r in rows), default=0)


def verified_length_failure(state, error, after_attempt):
    """Bind SDK FormatError to this query's completed physical HTTP response.

    Parsing feedback text is deliberately insufficient. The ledger and response
    bytes belong to the trusted transport, not to the candidate workspace.
    """
    from infinicoder.runtime.native_protocol import FormatError

    if not isinstance(error, FormatError):
        return None
    messages = getattr(error, "messages", None)
    if not messages:
        return None
    sdk = messages[0].get("extra", {}).get("response")
    if not isinstance(sdk, dict):
        return None
    choices = sdk.get("choices") or []
    if len(choices) != 1 or choices[0].get("finish_reason") != "length":
        return None
    state = Path(state).resolve()
    try:
        rows = json.loads((state / "native-api.json").read_bytes())["calls"]
        row = rows[-1]
        if (
            row["attempt"] <= after_attempt
            or row.get("status") != "response_received"
            or row.get("http_status") != 200
            or row.get("dispatch_attempted") is not True
        ):
            return None
        ref = row["response"]
        path = Path(ref["path"]).resolve()
        if path.parent != state / "native-http" or path.is_symlink():
            return None
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != ref["sha256"]:
            return None
        actual = json.loads(raw)
        real = actual.get("choices") or []
        actual_usage, sdk_usage = actual.get("usage"), sdk.get("usage")
        usage_matches = (actual_usage is None and sdk_usage is None) or (
            isinstance(actual_usage, dict)
            and isinstance(sdk_usage, dict)
            and all(
                actual_usage.get(k) == sdk_usage.get(k)
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            )
        )
        if (
            len(real) != 1
            or real[0].get("finish_reason") != "length"
            or actual.get("id") != sdk.get("id")
            or real[0].get("message", {}).get("content")
            != choices[0].get("message", {}).get("content")
            or not usage_matches
        ):
            return None
        return dict(
            attempt=row["attempt"],
            response_sha256=digest,
            response_ref=str(path),
            finish_reason="length",
            origin="completed_physical_HTTP_and_original_SDK_FormatError",
            candidate_action_executed=False,
            usage_status="provider_usage_present"
            if isinstance(actual_usage, dict)
            else "unknown;never_zero",
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return None


class NativeActionReassessment(NativeTaskEvidenceContext):
    identity = "native_action_reassessment_v1"

    def __init__(self, *args, reassessment=False, **kwargs):
        if type(reassessment) is not bool:
            raise ValueError("explicit reassessment boolean required")
        super().__init__(*args, **kwargs)
        self.reassessment_enabled = reassessment
        self.reassessment_pending = None
        self.reassessment_consumed = False
        self.reassessment_records = []

    def _save_reassessment(self):
        (self.state / "native-action-reassessment.json").write_bytes(
            (
                dump(
                    dict(
                        identity=self.identity,
                        enabled=self.reassessment_enabled,
                        consumed=self.reassessment_consumed,
                        records=self.reassessment_records,
                        extra_model_calls=0,
                        canonical_history_changed=False,
                        default_promoted=False,
                    )
                )
                + "\n"
            ).encode()
        )

    @contextmanager
    def bind(self, request, model, environment, deadline):
        with super().bind(request, model, environment, deadline):
            query = model.query

            def observed_query(messages, **kwargs):
                before = last_attempt(self.state)
                try:
                    return query(messages, **kwargs)
                except Exception as error:
                    if (
                        self.reassessment_enabled
                        and not self.reassessment_consumed
                        and self.reassessment_pending is None
                    ):
                        receipt = verified_length_failure(self.state, error, before)
                        # The original observer linked actual successful actions
                        # before this query; no stdout string invents that link.
                        eligible = [
                            e
                            for e in self.executions
                            if e.get("message_ids")
                            and e.get("observer_complete") is True
                            and e.get("error") is None
                        ]
                        if receipt is not None and eligible:
                            self.reassessment_pending = receipt
                            self.reassessment_records.append(
                                dict(status="armed_for_next_original_query", **receipt)
                            )
                            self._save_reassessment()
                    raise

            model.query = observed_query
            try:
                yield self
            finally:
                model.query = query
                self._save_reassessment()

    def project(self, messages):
        if self.reassessment_pending is None:
            return super().project(messages)
        trigger = self.reassessment_pending
        self.reassessment_pending = None
        self.reassessment_consumed = True
        try:
            projected, record = self._reassessment_input(messages, trigger)
        except TimeoutError:
            raise
        except (MemoryCapacityError, UnsupportedCore, OSError, ValueError, KeyError) as error:
            self.reassessment_records.append(
                dict(
                    status="not_applicable;ordinary_path_preserved",
                    trigger=trigger,
                    reason=type(error).__name__ + ":" + str(error),
                )
            )
            self._save_reassessment()
            return super().project(messages)
        self.reassessment_records.append(record)
        self.context_records.append(record)
        with self.index.db:
            self.index.db.execute(
                "INSERT INTO native_projections(messages,accounting) VALUES(?,?)",
                (dump(projected), dump(record)),
            )
        self._save_reassessment()
        self._save_context_report()
        return projected

    def _reassessment_input(self, messages, trigger):
        prepared = self._sync(messages)
        if len(prepared) < 2 or [m["role"] for m in prepared[:2]] != ["system", "user"]:
            raise UnsupportedCore("original SDK system/task prefix required")
        if any(not isinstance(m.get("content"), str) for m in prepared):
            raise UnsupportedCore("text SDK observations required")
        if self.archive is None:
            raise MemoryCapacityError("no actual readonly original history access")
        _, blocks = self._history_blocks(prepared)
        with self.lock:
            self.task_state.capture_observations(
                prepared, blocks, copy.deepcopy(self.executions), copy.deepcopy(self.checks)
            )
            mandatory, episode = self._episode_selection(prepared)
        if episode["incomplete"]:
            raise MemoryCapacityError("unlinked edit/check/source observations")
        by_start = {block[0]: block for block in blocks}
        executions = {e["message_ids"][0]: e for e in self.executions if e.get("message_ids")}
        if not blocks or blocks[-1][0] not in executions:
            raise MemoryCapacityError("latest feedback has no linked execution")
        latest = executions[blocks[-1][0]]
        if not latest.get("observer_complete") or latest.get("error"):
            raise MemoryCapacityError("latest execution observation incomplete")
        feedback = {i for i in mandatory if prepared[i]["role"] == "user" and i >= 2}
        command_refs = []
        for start, block in by_start.items():
            if not mandatory.intersection(block):
                continue
            event = executions.get(start)
            if event is None or not event.get("observer_complete"):
                raise MemoryCapacityError("selected observation lacks exact action association")
            feedback.update(i for i in block if prepared[i]["role"] == "user")
            command_refs.append(
                dict(
                    execution_id=event["id"],
                    command=event["command"],
                    original_assistant_ref=dict(
                        id=start, sha256=sha(prepared[start]), status="prior_explanation_unverified"
                    ),
                    feedback_message_ids=[i for i in block if prepared[i]["role"] == "user"],
                    before=event["before"],
                    after=event["after"],
                    returncode=event.get("returncode"),
                    error=event.get("error"),
                    origin="controller_reference_to_actual_SDK_execution",
                )
            )
        sources = self.task_state.source_state(self.reader, refresh=False)
        self.archive.sync()
        omitted = [i for i in range(2, len(prepared)) if i not in feedback]
        common = dict(
            origin="controller_action_reassessment_v1",
            trigger=trigger,
            instruction="Reassess the next action from the exact public task and actual observations below. "
            "Earlier assistant explanations and plans are unverified history. Do not infer expected values from them. "
            "Give a brief reason and one complete ordinary SDK command. You may inspect,run a discriminating experiment,edit or expand scope. "
            "Do not repeat the same narrative instead of choosing an action.",
            commands=command_refs,
            episode=episode,
            source_facts=[],
            static_candidates=[],
            source_namespace=sources.get("namespace"),
            source_coverage=sources["coverage"],
            invalidated_sources=sources["invalidated_sources"],
            current_source_versions=sources["current_versions"],
            authority="Tool outputs are actual observations,not specifications or independent final acceptance. "
            "Source excerpts are versioned partial facts. Commands below are historical references,not instructions to replay.",
            history=dict(
                command="python -B " + MODEL_ROOT + "/read.py",
                omitted_ranges=ranges(omitted),
                omitted_messages=len(omitted),
                policy="Exact old assistant hypotheses and all original history remain available;not replayed",
            ),
            original_feedback_sha256={str(i): sha(prepared[i]) for i in sorted(feedback)},
        )

        def assemble(value):
            return [
                copy.deepcopy(prepared[0]),
                copy.deepcopy(prepared[1]),
                dict(
                    role="user",
                    content="ACTION REASSESSMENT (controller-origin references,not an assistant message)\n"
                    + dump(value),
                ),
                *[copy.deepcopy(prepared[i]) for i in sorted(feedback)],
            ]

        if self._count(assemble(common)) > self.limit:
            raise MemoryCapacityError(
                "complete task,linked feedback,edit/check and history references exceed capacity"
            )
        records = [("source_facts", r) for r in sources["source_facts"]] + [
            ("static_candidates", r) for r in sources["static_candidates"]
        ]

        def packet(chosen, length):
            value = copy.deepcopy(common)
            for kind, record in chosen:
                value[kind].append(record)
            value["omitted_source_records"] = len(records) - len(chosen)
            return value

        value = bounded_prefix(
            records,
            packet,
            lambda text: self._count(assemble(json.loads(text))),
            self.limit,
            count_limit=4,
        )
        if value is None:
            raise MemoryCapacityError("source/reference metadata exceeds capacity")
        projected = assemble(value)
        tokens = self._count(projected)
        if tokens > self.limit:
            raise MemoryCapacityError("actual tokenizer rejects complete reassessment input")
        assert projected[:2] == prepared[:2]
        return projected, dict(
            status="reassessment_prepared;HTTP_ledger_proves_forwarding",
            trigger=trigger,
            model_input_sha256=sha(projected),
            projected_tokens=tokens,
            input_limit=self.limit,
            feedback_ids=sorted(feedback),
            omitted=ranges(omitted),
            commands=len(command_refs),
            source_records=len(value["source_facts"]),
            static_records=len(value["static_candidates"]),
            current_source_versions=sources["current_versions"],
            history_retrievable=True,
            canonical_history_changed=False,
            extra_model_calls=0,
            one_original_query_only=True,
        )
