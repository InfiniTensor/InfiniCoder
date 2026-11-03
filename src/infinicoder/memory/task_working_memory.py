"""Pressure-triggered selection of exact history with reversible archive access."""

import copy
import json

from infinicoder.memory.native_memory import MemoryCapacityError, _json


class TaskWorkingMemory:
    """One task's working context; the observer remains the canonical history owner."""

    HELP = (
        "Older actions and observations are archived, not replayed. "
        "Use standalone ic-task-memory event ID START END to retrieve exact historical "
        "characters (ID>=2; range<=4000), or ic-task-memory history QUERY to find event IDs. "
        "Historical source and agent hypotheses are not current facts. Read current files "
        "before editing. Keep the original requirements and resolve the current failure."
    )

    def __init__(self, observer):
        self.observer = observer
        self.initial = observer._snapshot()
        self.active = False
        self.records = []

    def project(self, messages, evidence_note=""):
        owner = self.observer
        prepared = owner.prepare(messages)
        full = copy.deepcopy(prepared)
        full[1]["content"] += evidence_note
        original_tokens = owner._count(full)
        if not self.active and original_tokens <= int(owner.limit * 0.9):
            return full if evidence_note else messages

        prefix, blocks = owner._history_blocks(prepared)
        current = owner._snapshot()
        changed = {p: v for p, v in current.items() if v != self.initial.get(p)}
        # Protect the latest complete action, plus the latest source-edit action
        # for every observed file. A receipt is not a substitute for the edit.
        mandatory = set(prefix)
        if blocks:
            mandatory.update(blocks[-1])
        for path in changed:
            event = owner.origins.get(path)
            if (
                event
                and event.get("message_ids")
                and event.get("after", {}).get(path) == current[path]
            ):
                linked = set(event["message_ids"])
                for block in blocks:
                    if linked.intersection(block):
                        mandatory.update(block)
        # A failed public check is useful only with its real associated output.
        # Preserve the latest completed check's SDK block, including stale ones
        # as historical evidence; current-source status is separately labelled.
        for check in reversed(owner.checks):
            if check.get("finished") and check.get("execution_id") is not None:
                event = owner.executions[check["execution_id"]]
                linked = set(event.get("message_ids") or ())
                for block in blocks:
                    if linked.intersection(block):
                        mandatory.update(block)
                break

        # Only caller-issued report bytes that actually appear in SDK feedback
        # establish a delivery. Agent claims or swallowed command output do not.
        delivered = []
        for item in owner.report_deliveries:
            if item["execution_id"] is None:
                continue
            event = owner.executions[item["execution_id"]]
            ids = event.get("message_ids") or ()
            if any(
                prepared[i]["role"] == "user" and item["client_output"] in prepared[i]["content"]
                for i in ids
            ):
                delivered.append((item, ids))
        delivered_reports = {(item["report_id"], item["record_sha256"]) for item, ids in delivered}
        runs = [
            item
            for item in owner.report_deliveries
            if item["action"] == "run"
            and (item["report_id"], item["record_sha256"]) in delivered_reports
        ]
        if runs:
            latest = (runs[-1]["report_id"], runs[-1]["record_sha256"])
            for item, ids in delivered:
                if (item["report_id"], item["record_sha256"]) == latest:
                    for block in blocks:
                        if set(ids).intersection(block):
                            mandatory.update(block)

        optional = [b for b in blocks if not mandatory.intersection(b)]

        def assemble(keep):
            selected = mandatory.union(i for b in keep for i in b)
            omitted = [i for i in range(2, len(prepared)) if i not in selected]
            state = dict(
                kind="task_working_memory",
                changed_observed_sources=changed,
                changes_are_unverified=True,
                source_coverage="caller-declared paths only; other paths unknown",
                historical_events=len(prepared) - 2,
                omitted_events=omitted,
                retrieval=self.HELP,
            )
            view = [copy.deepcopy(full[i]) for i in sorted(selected)]
            view[1]["content"] += "\n\nTASK WORKING MEMORY\n" + _json(state)
            return view, sorted(selected)

        view, selected = assemble([])
        minimum = owner._count(view)
        if minimum > owner.limit:
            raise MemoryCapacityError(
                "original task, current edits and latest complete feedback exceed input capacity"
            )
        # Bounded exact-tokenizer search over complete recent action blocks.
        # Always return a measured packet, even with non-additive tokenizers.
        low, high = 0, len(optional)
        best, selected_best, best_tokens = view, selected, minimum
        target = max(minimum, int(owner.limit * 0.8))
        for _ in range(7):
            if low >= high:
                break
            n = (low + high + 1) // 2
            trial, ids = assemble(optional[-n:])
            tokens = owner._count(trial)
            if tokens <= target:
                low = n
                best, selected_best, best_tokens = trial, ids, tokens
            else:
                high = n - 1
        self.active = True
        record = dict(
            original_tokens=original_tokens,
            projected_tokens=best_tokens,
            input_limit=owner.limit,
            selected_events=selected_best,
            protected_events=sorted(mandatory),
            original_task_preserved=True,
            canonical_history_changed=False,
        )
        self.records.append(record)
        owner._save_event("task_working_memory", record)
        with owner.index.db:
            owner.index.db.execute(
                "INSERT INTO native_projections(messages,accounting) VALUES(?,?)",
                (_json(best), _json(record)),
            )
        return best

    def tool(self, args):
        owner = self.observer
        if len(args) == 4 and args[0] == "event":
            number, start, end = map(int, args[1:])
            if number < 2 or not 0 <= start < end <= start + 4000:
                raise ValueError("event ID>=2 and character range<=4000 required")
            row = owner.index.db.execute(
                "SELECT sha,message FROM native_history WHERE id=?", (number,)
            ).fetchone()
            if row is None:
                raise ValueError("unknown archived event")
            message = json.loads(row["message"])
            value = dict(
                id=number,
                sha256=row["sha"],
                role=message["role"],
                historical=True,
                text=message["content"][start:end],
                characters=len(message["content"]),
                start=start,
                end=min(end, len(message["content"])),
            )
        elif len(args) == 2 and args[0] == "history" and args[1].strip():
            # Parameterized literal search, not SQL LIKE wildcards or a shell.
            rows = owner.index.db.execute(
                "SELECT id,sha,message FROM native_history WHERE id>=2 "
                "AND instr(lower(message),lower(?))>0 ORDER BY id DESC LIMIT 8",
                (args[1],),
            ).fetchall()
            value = dict(
                historical=True,
                matches=[
                    dict(
                        id=r["id"],
                        sha256=r["sha"],
                        role=json.loads(r["message"])["role"],
                        preview=json.loads(r["message"])["content"][:160],
                    )
                    for r in rows
                ],
            )
        else:
            raise ValueError(self.HELP)
        text = _json(value)
        if owner._count([dict(role="user", content=text)]) > min(1400, owner.limit // 3):
            raise ValueError("archive view exceeds budget; request a narrower range")
        return text
