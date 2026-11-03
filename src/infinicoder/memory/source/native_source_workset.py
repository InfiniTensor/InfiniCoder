"""Current source workset on the real native SDK; no memory command interface."""

from contextlib import contextmanager
import json
from pathlib import Path, PurePosixPath

from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.memory.native_memory import (
    NativeRepositoryMemory,
    MemoryCapacityError,
    _json,
    _sha,
)
from infinicoder.memory.native_episode_memory import NativeEpisodeMemory
from infinicoder.memory.source.source_workset import build_source_workset, public_target_present


class NativeSourceWorkset(NativeEpisodeMemory):
    identity = "native_source_workset_v2_versioned_checks"
    expose_tools = False
    HEADER = "\n\nCURRENT SOURCE WORKSET (repository data;static relations are candidates,unknown is not absent)\n"
    CHECK_HEADER = "\n\nORIGINAL CHECK VERSION (recorded observations;not final acceptance)\n"

    def __init__(
        self,
        count_messages,
        *,
        tokenizer_identity,
        implementation_paths=(),
        public_targets=(),
        excluded_paths=(),
        source_tokens=1600,
        reserve=256,
    ):
        super().__init__(
            count_messages,
            tokenizer_identity=tokenizer_identity,
            implementation_paths=implementation_paths,
            reserve=reserve,
        )
        self.public_targets = tuple(public_targets)
        self.excluded_paths = tuple(excluded_paths)
        self.source_tokens = source_tokens

    def validate(self, request):
        NativeRepositoryMemory.validate(self, request)
        if self.paths:
            NativeEpisodeMemory.validate(self, request)
        if type(self.source_tokens) is not int or not 0 <= self.source_tokens <= 4096:
            raise ValueError("source budget must be0..4096 actual tokens")
        for target in self.public_targets:
            if not isinstance(target, str) or not public_target_present(request.spec.task, target):
                raise ValueError("target must be grounded in the original public task")
        for value in self.excluded_paths:
            path = PurePosixPath(value)
            if (
                not value
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in value
                or ":" in value
            ):
                raise ValueError("excluded source must be relative to the candidate workspace")

    @contextmanager
    def bind(self, request, model, environment, deadline):
        self.task = request.spec.task
        self.packet_cache = None
        with super().bind(request, model, environment, deadline):
            self.index.qualified_context = True
            excluded = [(self.workspace / p).resolve() for p in self.excluded_paths]
            if self.state.is_relative_to(self.workspace):
                excluded.append(self.state)
            self.index.excluded_paths = tuple(excluded)
            yield self
        path = Path(request.state_dir) / "native-memory-info.json"
        value = json.loads(path.read_bytes())
        value.update(
            source_facts="Automatic current source;no memory tools installed or advertised",
            observed_candidates="explicit_public_paths_only"
            if self.paths
            else "unknown/not_applicable",
            source_budget=self.source_tokens,
            public_targets=self.public_targets,
            excluded_paths=self.excluded_paths,
        )
        path.write_text(_json(value), encoding="utf-8")

    def _episode_selection(self, prepared):
        selected, meta = super()._episode_selection(prepared)
        if not self.paths:
            meta["candidate_binding"] = "unknown/not_applicable;no public candidate path set"
            if meta["original_check"] is not None:
                meta["original_check"]["association"] = "unknown;no_observed_paths"
        return selected, meta

    def _check_evidence(self, check, event, ids, current, status):
        """Version association is distinct from availability of a check result."""
        value = super()._check_evidence(check, event, ids, current, status)

        def known(snapshot):
            if not isinstance(snapshot, dict) or set(snapshot) != set(self.paths):
                return False
            for item in snapshot.values():
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

        association = "unknown"
        if not self.paths:
            reason = "no_observed_paths"
        elif not check["finished"]:
            reason = "check_not_finished"
        elif check.get("origin") == "controller_public_operation":
            if not check.get("same_controller_dispatch"):
                reason = "unknown_controller_dispatch"
            elif not all(
                known(snapshot) for snapshot in (check["before"], check["after"], current)
            ):
                reason = "unknown_source_identity"
            elif check["before"] != check["after"]:
                reason = "source_changed_during_check"
            elif check["after"] != current:
                association, reason = "stale", "observed_revision_changed_since_check"
            else:
                association, reason = (
                    "current",
                    "controller_operation_same_declared_observed_paths_only",
                )
        elif not check.get("same_execution"):
            reason = "detached_or_unlinked_check"
        elif event is None or not ids or not event["observer_complete"]:
            reason = "incomplete_action_association"
        elif not all(known(snapshot) for snapshot in (check["before"], check["after"], current)):
            reason = "unknown_source_identity"
        elif check["before"] != check["after"]:
            reason = "source_changed_during_check"
        elif check["after"] != current:
            association, reason = "stale", "observed_revision_changed_since_check"
        else:
            association, reason = "current", "same_declared_observed_paths_only"
        value["version_binding"] = dict(
            association=association,
            reason=reason,
            current_observed_sources=current,
            result_availability="original_report_recorded"
            if check["finished"] and check["report"] is not None and check["error"] is None
            else "unknown",
        )
        return value

    def _check_note(self, meta):
        check = meta["original_check"]
        if check is None:
            value = dict(
                association="unknown",
                reason="no_original_check_observed",
                observed_paths=meta["observed_paths"],
                original_feedback_messages=[],
            )
        else:
            # All fields come from canonical original-check/action observations,
            # never from model claims, stdout parsing or a second source read.
            value = dict(
                check["version_binding"],
                original_feedback_messages=check["message_ids"],
                judge_identity=check["judge_identity"],
                scope=check["scope"],
                coverage=check["coverage"],
                origin=check.get("origin", "sdk_execution"),
                controller_dispatch=check.get("controller_dispatch"),
                checked_sources_before=check["before"],
                checked_sources_after=check["after"],
                report_sha256=check["report_sha256"],
                execution_error=check["error"],
            )
        value["limits"] = (
            "Observed paths and stated check coverage only;historical feedback is not a result for changed or unknown source. Model self-tests are unverified observations. Final independent acceptance remains required."
        )
        return self.CHECK_HEADER + _json(value)

    def project(self, messages):
        prepared = self._sync(messages)
        if (
            len(prepared) < 2
            or [m.get("role") for m in prepared[:2]] != ["system", "user"]
            or any(not isinstance(m.get("content"), str) for m in prepared)
        ):
            raise UnsupportedCore("workset requires pinned text SDK history")
        original_tokens = self._count(prepared)
        mandatory, meta = self._episode_selection(prepared)
        if meta["incomplete"] and original_tokens > self.limit:
            self._record(
                prepared,
                prepared,
                set(range(len(prepared))),
                meta,
                None,
                "capacity_rejected;episode_incomplete",
            )
            raise MemoryCapacityError(
                "episode_incomplete: source observations cannot be reliably linked;no generation"
            )
        if meta["incomplete"]:
            # Unlinked observations do not require dropping or rejecting an
            # otherwise complete original history. They cannot be compacted.
            mandatory = set(range(len(prepared)))
        check_note = self._check_note(meta)

        def assemble(ids, note=""):
            view = [dict(prepared[i]) for i in sorted(ids)]
            # Check-to-version association belongs to mandatory evidence. Spend
            # optional source/history budget only after it and the exact raw
            # check/edit/latest blocks fit. Preserve original roles/text/order.
            view[1]["content"] += check_note + note
            return view

        minimum = assemble(mandatory)
        if self._count(minimum) > self.limit:
            self._record(prepared, minimum, mandatory, meta, None, "capacity_rejected;mandatory")
            raise MemoryCapacityError(
                "mandatory task/latest/edit/check/version blocks exceed capacity;no generation"
            )
        self.index.refresh()
        revision = self.index.revision()
        available = max(0, self.limit - self._count(minimum) - 32)
        diagnostic, diagnostic_report = self._diagnostic_packet(
            prepared, meta, min(1000, self.source_tokens, available)
        )
        note = diagnostic
        report = dict(status="control_source_tokens_zero", revision=revision)
        diagnostic_tokens = (
            self._count([dict(role="user", content=diagnostic)]) if diagnostic else 0
        )
        source_budget = max(0, self.source_tokens - diagnostic_tokens)
        if self.source_tokens and not (
            len(self.index.files()) <= 1 and original_tokens <= self.limit
        ):
            available = max(
                0,
                self.limit
                - self._count(assemble(mandatory, diagnostic))
                - self._count([dict(role="user", content=self.HEADER)])
                - 32,
            )
            header_tokens = self._count([dict(role="user", content=self.HEADER)])
            budget = max(0, min(source_budget - header_tokens, available))
            key = (
                revision,
                _sha(self.task),
                budget,
                self.public_targets,
                self._source_policy_identity(),
            )
            # Revalidate every cached supporting source before reuse; refresh
            # invalidates normal edits, this also guards concurrent reads.
            cache = self.packet_cache
            valid = cache is not None and cache[0] == key
            if valid:
                import hashlib

                for path, expected in cache[2].get("source_files", {}).items():
                    p = self.workspace / path
                    try:
                        valid = (
                            valid
                            and p.is_file()
                            and not p.is_symlink()
                            and not any(q.is_symlink() for q in p.parents)
                            and p.resolve().is_relative_to(self.workspace)
                            and hashlib.sha256(p.read_bytes()).hexdigest() == expected
                        )
                    except OSError:
                        valid = False
            if valid:
                text, report = cache[1:]
            else:
                text, report = self._build_source_packet(budget)
                self.packet_cache = (key, text, report)
            records = text.splitlines()
            while records:
                candidate = self.HEADER + "\n".join(records)
                if (
                    self._count(assemble(mandatory, diagnostic + candidate)) <= self.limit
                    and self._count([dict(role="user", content=candidate)]) <= source_budget
                ):
                    note = diagnostic + candidate
                    break
                records.pop()
            report = dict(
                report,
                exposed=bool(note),
                actual_record_count=len(records),
                actual_source_sha256=_sha(note),
                actual_source_packet=note,
            )
        elif self.source_tokens:
            report = dict(status="quiet_single_source", revision=revision, exposed=False)
        if diagnostic_report is not None:
            report["candidate_diagnostic"] = diagnostic_report
        selected = self._select_history(
            prepared, mandatory, lambda ids: self._count(assemble(ids, note))
        )
        projected = assemble(selected, note)
        self._record(
            prepared,
            projected,
            selected,
            meta,
            report,
            "ready_for_native_query;HTTP ledger determines actual forwarding",
        )
        return projected

    def _diagnostic_packet(self, prepared, meta, budget):
        """Optional read-only view; default preserves the original projection."""
        return "", None

    def _source_policy_identity(self):
        return "original_source_workset"

    def _build_source_packet(self, budget):
        return build_source_workset(
            self.index,
            self.task,
            self.public_targets,
            budget,
            lambda text: self._count([dict(role="user", content=text)]),
        )

    def _select_history(self, prepared, mandatory, count):
        """Original selector; optional profiles may change only this decision."""
        all_ids = set(range(len(prepared)))
        if count(all_ids) <= self.limit:
            return all_ids
        selected = set(mandatory)
        for block in reversed(self._history_blocks(prepared)[1]):
            if set(block).issubset(selected):
                continue
            candidate = selected | set(block)
            if count(candidate) > self.limit:
                break
            selected = candidate
        return selected

    def _selection_report(self):
        return None

    def _record(self, prepared, projected, selected, episode, source, status):
        row = dict(
            original_messages=len(prepared),
            original_tokens=self._count(prepared),
            projected_tokens=self._count(projected),
            input_limit=self.limit,
            compacted=len(selected) < len(prepared),
            selected=sorted(selected),
            omitted=[i for i in range(len(prepared)) if i not in selected],
            episode=episode,
            source_workset=source,
            original_task_sha256=_sha(prepared[1]["content"]),
            tokenizer_identity=self.tokenizer_identity,
            source_tokens=self.source_tokens,
            memory_command_interface=False,
            status=status,
        )
        selection = self._selection_report()
        if selection is not None:
            row["history_selection"] = selection
        with self.index.db:
            self.index.db.execute(
                "INSERT INTO native_projections(messages,accounting) VALUES(?,?)",
                (_json(projected), _json(row)),
            )
        with (self.state / "native-memory-context.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(_json(row) + "\n")
