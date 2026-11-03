"""Verified source transitions from actual SDK actions,not proposed intent."""

from dataclasses import dataclass
import hashlib
import json
import re

from infinicoder.memory.action_source_memory import _namespace
from infinicoder.memory.task_checkpoint_sources import _relative
from infinicoder.memory.task_state_checkpoint import digest, dump
from infinicoder.memory.task_history_archive import MODEL_ROOT


def source_state(value):
    if isinstance(value, dict):
        if (
            value.get("kind") == "file"
            and isinstance(value.get("sha256"), str)
            and re.fullmatch("[0-9a-f]{64}", value["sha256"])
        ):
            return dict(kind="file", sha256=value["sha256"])
        if value.get("kind") == "missing":
            return dict(kind="missing")
    return dict(kind="unknown")


def relation(before, after):
    before, after = source_state(before), source_state(after)
    if "unknown" in (before["kind"], after["kind"]):
        return "unknown"
    if before == after:
        return "unchanged" if before["kind"] == "file" else "unknown"
    return "changed"


@dataclass(frozen=True)
class ActionEffectRecord:
    payload_json: str

    @property
    def payload(self):
        return json.loads(self.payload_json)

    @property
    def execution_id(self):
        return self.payload["execution_id"]


def make_effect_record(event, namespace, paths, before, after, feedback, *, linked_exactly):
    """The producer supplies actual stable observations and original SDK feedback.

    A successful shell status or empty stdout never establishes a file change.
    Stable absence can establish creation/deletion,not an unchanged source body.
    """
    namespace = json.loads(_namespace(namespace))
    paths = tuple(dict.fromkeys(paths))
    if len(paths) > 2 or any(not _relative(p) for p in paths):
        raise ValueError("at most two real relative source paths")
    command = event.get("command")
    ids = event.get("message_ids")
    if not isinstance(command, str) or type(event.get("id")) is not int:
        raise ValueError("actual action identity required")
    exact = bool(
        linked_exactly
        and ids
        and event.get("observer_complete")
        and event.get("error") is None
        and type(event.get("returncode")) is int
        and feedback.get("complete") is True
    )
    rows = []
    for path in paths:
        first, last = source_state(before.get(path)), source_state(after.get(path))
        status = relation(first, last) if exact else "unknown"
        rows.append(dict(path=path, before=first, after=last, relation=status))
    status = (
        "changed"
        if any(r["relation"] == "changed" for r in rows)
        else "unchanged"
        if rows and all(r["relation"] == "unchanged" for r in rows)
        else "unknown"
    )
    payload = dict(
        execution_id=event["id"],
        message_ids=list(ids or ()),
        namespace=namespace,
        command_sha256=hashlib.sha256(command.encode()).hexdigest(),
        feedback_sha256=feedback.get("raw_sha256"),
        SDK_feedback_sha256=feedback.get("message_sha256"),
        feedback_exactly_linked=exact,
        returncode=event.get("returncode"),
        error=event.get("error"),
        paths=rows,
        relation=status,
        coverage="Only listed observed source paths;all other files,background work and external side effects unknown",
        authority="Actual source versions around an action;not proof of correctness,absence of writes,or permission to edit",
    )
    return ActionEffectRecord(dump(payload))


class EffectSequence:
    """Count adjacent identical known observations without suppressing actions."""

    def __init__(self):
        self.previous_id = None
        self.previous_signature = None
        self.count = 0

    def add(self, record):
        value = record.payload
        if self.previous_id is not None and record.execution_id <= self.previous_id:
            raise ValueError("action effect already consumed or out of order")
        signature = None
        if value["relation"] == "unchanged" and value["feedback_exactly_linked"]:
            signature = digest(
                {
                    k: value[k]
                    for k in (
                        "namespace",
                        "command_sha256",
                        "feedback_sha256",
                        "returncode",
                        "error",
                        "paths",
                    )
                }
            )
        if signature is not None:
            self.count = (
                self.count + 1
                if self.previous_id == record.execution_id - 1
                and self.previous_signature == signature
                else 1
            )
        else:
            self.count = 0
        self.previous_id, self.previous_signature = record.execution_id, signature
        return self.count


class VersionLabels:
    """Task-local unique names;full identities are saved in every referenced record."""

    def __init__(self, namespace):
        self.namespace = _namespace(namespace)
        self.identities = {}

    def name(self, path, state):
        state = source_state(state)
        if state["kind"] == "unknown":
            return "unknown"
        key = dump(dict(namespace=json.loads(self.namespace), path=path, state=state))
        if key not in self.identities:
            self.identities[key] = "v" + str(len(self.identities) + 1)
        return self.identities[key]

    def referenced(self, labels):
        return {name: json.loads(key) for key, name in self.identities.items() if name in labels}


def render_effect(record, repetitions, current, checks, labels, *, effect_record_ref=None):
    """Whole factual view;caller performs real source/full-input token admission."""
    value = record.payload
    if _namespace(value["namespace"]) != labels.namespace:
        raise ValueError("version labels belong to another namespace")
    fields = []
    used = set()
    lines = [
        "CONTROLLER ACTION EFFECTS. Observations only;original task alone controls edits.",
        f"event={value['execution_id']} returncode={value['returncode']} same_observation_run={repetitions}",
    ]
    for row in value["paths"]:
        path = row["path"]
        before = labels.name(path, row["before"])
        after = labels.name(path, row["after"])
        now = labels.name(path, current.get(path))
        used.update((before, after, now))
        check = checks.get(path, dict(status="not_observed"))
        lines.append(
            f"{path}: {row['relation']} {before}->{after}; current={now}; check_version={check['status']}"
        )
        fields.append(dict(path=path, before=before, after=after, current=now, check=check))
    payload = dict(
        effect=value,
        same_observation_run=repetitions,
        current_source={p: source_state(s) for p, s in current.items()},
        display=fields,
        version_labels=labels.referenced(used),
        action_effect_record_ref=effect_record_ref,
        interpretation="Repeated observations concern listed source only;idempotent writes,reads and polls remain allowed",
    )
    ref = "source-view:" + digest(payload)[:32]
    lines += [
        "Other side effects and unobserved files: unknown.",
        f"Full versions/coverage: python3 {MODEL_ROOT}/read.py source {ref}",
    ]
    return "\n".join(lines), ref, payload


def archive_effect(reader, ref, payload):
    """Append to the existing source table consumed by readonly TaskHistoryArchive."""
    if _namespace(reader.namespace) != _namespace(payload["effect"]["namespace"]):
        raise ValueError("effect namespace changed")
    existing = reader.index.db.execute(
        "SELECT record FROM source_plan_views WHERE ref=?", (ref,)
    ).fetchone()
    if existing:
        if json.loads(existing[0])["view"]["records"] != [payload]:
            raise ValueError("effect reference collision")
        return
    files = {
        row["path"]: row["after"]["sha256"]
        for row in payload["effect"]["paths"]
        if row["after"]["kind"] == "file"
    }
    reader._record_view(
        dict(op="action_effect", execution_id=payload["effect"]["execution_id"]),
        dict(
            read_event_ref=ref,
            view="actual_action_effect",
            source_files=files,
            records=[payload],
            origin="controller_actual_linked_execution",
            authority=payload["effect"]["authority"],
        ),
    )
