"""Select original whole history blocks by distinct observed evidence.

This changes model input, never canonical history or execution. Exact duplicate
observations do not prove no external progress, a cause, or correctness. Unknown
or unlinked actions remain separate. No summaries or instructions are generated.
"""

import hashlib
import json
import time


def _sha(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _known(snapshot):
    if not isinstance(snapshot, dict) or not snapshot:
        return False
    for value in snapshot.values():
        if not isinstance(value, dict):
            return False
        if value.get("kind") == "missing":
            continue
        digest = value.get("sha256")
        if (
            value.get("kind") != "file"
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            return False
    return True


def select_history(prepared, blocks, mandatory, executions, *, checks=(), views=(), count, limit):
    """Return message IDs and a caller-only audit; count uses the real renderer.

    Mandatory blocks are never collapsed. Under pressure, first cover different
    observations from newest to oldest, then fill retained representatives. A
    whole optional block that does not fit is skipped; it is never cut to fit.
    """
    started = time.monotonic()
    all_ids = set(range(len(prepared)))
    mandatory = set(mandatory)
    if not mandatory <= all_ids:
        raise ValueError("mandatory IDs outside canonical history")
    by_start = {}
    for event in executions:
        ids = event.get("message_ids")
        if isinstance(ids, list) and len(ids) == 2:
            by_start.setdefault(ids[0], []).append(event)
    groups = {}
    entries = []
    unknown_reasons = []
    for ordinal, block in enumerate(blocks):
        block = tuple(block)
        if (
            not block
            or not set(block) <= all_ids
            or prepared[block[0]].get("role") != "assistant"
            or len(block) < 2
            or any(prepared[i].get("role") != "user" for i in block[1:])
        ):
            raise ValueError("selector requires original complete assistant/user blocks")
        possible = by_start.get(block[0], [])
        event = possible[0] if len(possible) == 1 else None
        linked = (
            event is not None
            and event.get("message_ids") == list(block[:2])
            and event.get("observer_complete") is True
            and event.get("error") is None
        )
        observed_checks = [
            c for c in checks if event is not None and c.get("execution_id") == event.get("id")
        ]
        check_refs = []
        checks_known = True
        for check in observed_checks:
            if (
                not check.get("finished")
                or not check.get("same_execution")
                or check.get("error") is not None
                or check.get("report") is None
                or not check.get("judge_identity")
                or not check.get("scope")
                or not check.get("coverage")
                or not _known(check.get("before"))
                or not _known(check.get("after"))
            ):
                checks_known = False
            check_refs.append(
                {
                    key: check.get(key)
                    for key in (
                        "judge_identity",
                        "scope",
                        "coverage",
                        "before",
                        "after",
                        "report",
                        "error",
                    )
                }
            )
        if event is not None and set(event.get("checks", [])) != {c["id"] for c in observed_checks}:
            checks_known = False
        source_views = []
        for view in views:
            versions = view.get("source_files")
            if (
                linked
                and view.get("execution_id") == event["id"]
                and view.get("kind") == "workbench_source_view"
                and view.get("SDK_history_fabricated") is False
                and type(view.get("records")) is int
                and view["records"] > 0
                and isinstance(versions, dict)
                and versions
                and _known({p: dict(kind="file", sha256=d) for p, d in versions.items()})
            ):
                source_views.append(
                    {key: view.get(key) for key in ("view", "selector", "records", "source_files")}
                )
        category = "ordinary"
        if linked:
            changed = (
                _known(event.get("before"))
                and _known(event.get("after"))
                and event["before"] != event["after"]
            )
            if changed or (observed_checks and checks_known):
                category = "edit_check"
            elif source_views:
                category = "source_view"
        if not linked:
            reason = "unknown_or_ambiguous_execution_link"
        elif not isinstance(event.get("command"), str):
            reason = "unknown_command"
        elif not _known(event.get("before")) or not _known(event.get("after")):
            reason = "unknown_declared_source"
        elif event["before"] != event["after"]:
            reason = "observed_source_changed"
        elif not checks_known:
            reason = "unknown_original_check_association_or_versions"
        else:
            reason = None
        eligible = reason is None
        signature = None
        if eligible:
            # Include ALL original messages, not just stdout/command. A different
            # assistant hypothesis or an additional SDK error is distinct.
            signature = dict(
                block_sha256=_sha(
                    [{k: prepared[i][k] for k in ("role", "content")} for i in block]
                ),
                command_sha256=hashlib.sha256(event["command"].encode()).hexdigest(),
                observed_sources_sha256=_sha(event["before"]),
                original_checks_sha256=_sha(check_refs),
                source_views_sha256=_sha(source_views),
            )
            key = "observed:" + _sha(signature)
        else:
            key = "opaque:" + str(ordinal)
            unknown_reasons.append(dict(block=list(block), reason=reason))
        entry = dict(
            ids=block,
            key=key,
            ordinal=ordinal,
            category=category,
            execution_id=event.get("id") if eligible else None,
        )
        entries.append(entry)
        group = groups.setdefault(key, dict(signature=signature, members=[]))
        group["members"].append(entry)

    folded = set()
    protected_duplicates = []
    duplicate_groups = []
    for key, group in groups.items():
        members = group["members"]
        if group["signature"] is None or len(members) < 3:
            continue
        duplicate_groups.append(
            dict(
                group["signature"],
                execution_ids=[e["execution_id"] for e in members],
                original_blocks=[list(e["ids"]) for e in members],
            )
        )
        # First and latest representatives remain candidates. Intermediate
        # mandatory blocks remain in full even if their bytes are identical.
        for entry in members[1:-1]:
            ids = set(entry["ids"])
            if ids & mandatory:
                protected_duplicates.append(list(entry["ids"]))
            else:
                folded.update(ids)

    count_calls = 0
    cache = {}
    budget_exhausted = False

    def measure(ids):
        nonlocal count_calls, budget_exhausted
        key = tuple(sorted(ids))
        if key in cache:
            return cache[key]
        if count_calls == 24:
            budget_exhausted = True
            return None
        count_calls += 1
        cache[key] = count(ids)
        return cache[key]

    full_tokens = measure(all_ids)
    skipped = []
    considered = set()
    if full_tokens <= limit:
        selected = all_ids - folded
        mode = (
            "byte_preserved_full_history"
            if not folded
            else "collapse_exact_nonmandatory_intermediate_blocks"
        )
        # Deleting messages is not assumed token-monotone under every renderer.
        if folded and measure(selected) > limit:
            selected = all_ids
            folded = set()
            mode = "original_full_history_renderer_fallback"
    else:
        selected = set(mandatory)
        if measure(selected) > limit:
            raise ValueError("mandatory whole blocks exceed admitted input capacity")
        mode = "distinct_observation_coverage_then_representatives"
        covered = {e["key"] for e in entries if set(e["ids"]).issubset(selected)}
        newest = {key: group["members"][-1] for key, group in groups.items() if key not in covered}
        categories = ("edit_check", "source_view", "ordinary")
        queues = {
            category: sorted(
                (e for e in newest.values() if e["category"] == category),
                key=lambda e: e["ordinal"],
                reverse=True,
            )
            for category in categories
        }
        # Give each nonempty category its newest distinct group, then rotate.
        # Categories use trusted execution associations, never shell-name guesses.
        while any(queues.values()) and not budget_exhausted:
            for category in categories:
                if not queues[category]:
                    continue
                entry = queues[category].pop(0)
                trial = selected | set(entry["ids"])
                size = measure(trial)
                if size is None:
                    break
                considered.add(entry["ordinal"])
                if size > limit:
                    skipped.append(list(entry["ids"]))
                    continue
                selected = trial
                covered.add(entry["key"])
        # Only the oldest representative may supplement a represented group.
        # Never fill the input with its intermediate duplicate copies.
        earliest = sorted(
            (
                g["members"][0]
                for g in groups.values()
                if g["signature"] is not None and len(g["members"]) > 1
            ),
            key=lambda e: e["ordinal"],
            reverse=True,
        )
        for entry in earliest:
            if budget_exhausted:
                break
            ids = set(entry["ids"])
            if ids.issubset(selected):
                continue
            size = measure(selected | ids)
            if size is None:
                break
            considered.add(entry["ordinal"])
            if size <= limit:
                selected |= ids

    if not mandatory <= selected:
        raise AssertionError("mandatory evidence lost")
    # Every assistant and its complete original feedback move as a unit.
    if any(set(e["ids"]) & selected and not set(e["ids"]) <= selected for e in entries):
        raise AssertionError("partial original action block selected")
    represented = {e["key"] for e in entries if set(e["ids"]).issubset(selected)}
    report = dict(
        identity="distinct_original_observations_v1",
        mode=mode,
        full_input_tokens=full_tokens,
        input_limit=limit,
        exact_duplicate_groups=duplicate_groups,
        canonical_blocks=len(entries),
        distinct_block_groups=len(groups),
        represented_groups=len(represented),
        opaque_blocks=len(unknown_reasons),
        ungrouped_reasons=unknown_reasons,
        selected_categories={
            category: sum(e["category"] == category and set(e["ids"]) <= selected for e in entries)
            for category in ("edit_check", "source_view", "ordinary")
        },
        folded_intermediate_message_ids=sorted(folded),
        mandatory_duplicate_blocks_preserved=protected_duplicates,
        oversized_optional_blocks_skipped=skipped,
        selected=sorted(selected),
        mandatory=sorted(mandatory),
        selector_count_calls=count_calls,
        selector_count_limit=24,
        selector_count_budget_exhausted=budget_exhausted,
        not_considered_optional_blocks=[
            list(e["ids"])
            for e in entries
            if e["ordinal"] not in considered
            and not set(e["ids"]) <= selected
            and not set(e["ids"]) & folded
        ],
        selector_wall_s=time.monotonic() - started,
        canonical_history_changed=False,
        generated_summary=False,
        execution_suppressed=False,
        limits="Exact original block bytes and declared observed source versions only. Unobserved files,external progress,model reasoning and correctness remain unknown.",
    )
    return selected, report
