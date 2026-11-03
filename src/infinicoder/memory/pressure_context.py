"""Whole observed source regions compete with old history only under pressure.

No summary model,tool execution,index refresh,new protocol or inferred writes.
The existing physical owner remains responsible for the final request bound,
workspace revision,deadline and dispatch. Snapshot reassembly is purely local.
"""

from infinicoder._resources import package_root

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json

from infinicoder.models.budget import BudgetExceeded, TokenEstimator
from infinicoder.memory.source.source_sync import _read

LABEL = "\nCURRENT COMPLETE OBSERVED SOURCE REGIONS\n"
READS = ("read", "read_symbol", "read_function_body", "workpack")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def identity():
    root = package_root()
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "memory/pressure_context.py",
            "memory/source/source_sync.py",
            "runtime/engine.py",
            "models/budget.py",
        )
    }


def configure_pressure_evidence(runtime, *, enabled=True):
    """Configure the existing full physical Runtime,not a Native flat adapter."""
    from infinicoder.collaboration.collaboration import CollaborationRuntime

    if type(enabled) is not bool:
        raise ValueError("pressure evidence enablement must be boolean")
    if (
        not isinstance(runtime, CollaborationRuntime)
        or not getattr(runtime, "physical_context_budget", False)
        or runtime.shell_loop
        or runtime.source_sync
    ):
        raise ValueError(
            "requires the full physical Runtime,ordinary protocol,and no competing source_sync projection"
        )
    runtime.pressure_evidence = enabled
    return runtime


def _event(pair):
    """Read existing provenance envelopes,never infer a read from stdout."""
    try:
        envelope = json.loads(pair[-1]["content"])
        if not isinstance(envelope, dict):
            return None
        if envelope.get("runtime_evidence") is True:
            if envelope.get("dispatch") != "entered":
                return None
            action = envelope.get("dispatched_action")
            result = envelope.get("observation")
            shape = "provenance"
        elif "legacy_history" in envelope:
            legacy = envelope["legacy_history"]
            action = json.loads(legacy[0]["content"])
            result = json.loads(legacy[-1]["content"])
            shape = "legacy"
        elif len(pair) > 1 and pair[0]["role"] == "assistant":
            action = json.loads(pair[0]["content"])
            result = envelope
            shape = "direct"
        else:
            return None
        return (
            (action, result, shape)
            if isinstance(action, dict) and isinstance(result, dict)
            else None
        )
    except (ValueError, TypeError, KeyError, IndexError):
        return None


def _source(pair):
    try:
        event = _event(pair)
        if event is None:
            return None
        action, observation, shape = event
        if action.get("tool") not in READS or observation.get("error"):
            return None
        source = observation.get("source") if action["tool"] == "workpack" else observation
        if not isinstance(source, dict) or not {
            "path",
            "sha256",
            "start",
            "total_lines",
            "text",
        }.issubset(source):
            return None
        return (
            action,
            observation,
            source,
            hashlib.sha256(
                json.dumps(source, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest(),
            shape,
        )
    except (ValueError, TypeError, KeyError, IndexError):
        return None


def _protected(pairs, state):
    selected = {len(pairs) - 1} if pairs else set()
    for i in range(len(pairs) - 1, -1, -1):
        event = _event(pairs[i])
        if event is None:
            continue
        action, result, _ = event
        if (
            action.get("tool") in ("edit", "edit_symbol", "edit_function_body")
            and result.get("path")
            and result.get("after")
        ):
            selected.add(i)
            break
    if state.get("last_failure"):
        for i in range(len(pairs) - 1, -1, -1):
            event = _event(pairs[i])
            if event is None:
                continue
            result = event[1]
            if (
                result.get("error")
                or result.get("accepted") is False
                or result.get("passed") is False
            ):
                selected.add(i)
                break
    return sorted(selected)


def _views(state, workspace):
    cache = {}
    views = []
    references = []
    seen = set()
    changed = set(state.get("changed_files", [])[-3:])
    recent = set(state.get("read_paths", [])[-3:])
    entries = sorted(
        state.get("source_working_set", []),
        key=lambda e: (e["path"] in changed, e["path"] in recent, e["last_call"], e["last_step"]),
        reverse=True,
    )
    for position, entry in enumerate(entries):
        ref = dict(
            path=entry["path"],
            observed_sha256=entry["origin_sha256"],
            observation_sha256=entry["observation_sha256"],
            start=entry["start"],
            count=entry["count"],
            status="not_refreshed;read_current_source_for_expansion",
        )
        if position >= 8:
            references.append(ref)
            continue
        try:
            if entry["path"] not in cache:
                cache[entry["path"]] = _read(workspace, entry["path"])
            current, lines = cache[entry["path"]]
            ref["current_sha256"] = current
            ref["status"] = "current_source_body_omitted;read_current_source_for_expansion"
            if current != entry["origin_sha256"]:
                ref["status"] = "observed_version_invalidated;read_current_source_for_expansion"
            if entry.get("prefix_chars") is not None:
                raise ValueError("partial_line_requires_focused_read")
            if current != entry["origin_sha256"]:
                if not entry["whole_file"]:
                    raise ValueError("changed_partial_region_requires_focused_read")
                start, count = 1, len(lines)
            else:
                start, count = entry["start"], entry["count"]
            if not 1 <= start <= len(lines) + 1 or count < 0 or start - 1 + count > len(lines):
                raise ValueError("invalid_observed_range")
            text = "\n".join(lines[start - 1 : start - 1 + count])
            if len(text.encode()) > 262144:
                raise ValueError("complete_region_exceeds_256KiB_bound")
            key = (entry["path"], current, start, count)
            if key in seen:
                references.append(ref)
                continue
            seen.add(key)
            views.append(
                dict(
                    path=entry["path"],
                    sha256=current,
                    start=start,
                    end=start + count - 1,
                    text=text,
                    origin_read_sha256=entry["observation_sha256"],
                    observed_sha256=entry["origin_sha256"],
                    changed_since_observation=current != entry["origin_sha256"],
                    completeness="whole_observed_region"
                    if current == entry["origin_sha256"]
                    else "current_whole_file_refresh",
                    authority="current_source_only;behavior_and_dependency_completeness_unverified",
                )
            )
        except (OSError, ValueError, UnicodeError, KeyError, TypeError) as error:
            ref["status"] = "requires_current_read"
            ref["reason"] = type(error).__name__
        references.append(ref)
    return views, references


@dataclass(frozen=True)
class JointSnapshot:
    mandatory: str
    system: str
    dialogue_json: str
    pairs_json: str
    views_json: str
    references_json: str
    protected: tuple
    optional_json: str
    failure_json: str
    lost_observations: tuple
    failure_reference_json: str


def _current_failure(mandatory, pairs, protected, failure, estimator):
    """Deduplicate only an identical complete,protected display packet.

    The saved state,private acceptance receipt and full feedback message are
    never changed. A reference replaces only a duplicate model-facing value.
    """
    if not failure:
        return mandatory, None, None
    try:
        core, end = json.JSONDecoder().raw_decode(mandatory)
        in_core = isinstance(core, dict) and core.get("last_failure") == failure
    except (ValueError, TypeError):
        core = None
        in_core = False
    reference = None
    brief = None
    for i in reversed(protected):
        event = _event(pairs[i])
        if event is None:
            continue
        action, result, shape = event
        if shape not in ("direct", "provenance") or result.get("failures") != failure:
            continue
        reference = dict(
            kind="complete_current_failure_in_protected_action_feedback",
            tool=action.get("tool"),
            observation_message_sha256=hashlib.sha256(pairs[i][-1]["content"].encode()).hexdigest(),
            field_path=["observation", "failures"] if shape == "provenance" else ["failures"],
            original_session_and_full_feedback_unchanged=True,
        )
        envelope = json.loads(pairs[i][-1]["content"])
        # The existing provenance step is a visible message locator. Keep the
        # longer digest in the report,not another copy in the model contract.
        brief = dict(field_path=reference["field_path"])
        if shape == "provenance" and type(envelope.get("step")) is int:
            brief["history_step"] = envelope["step"]
        else:
            brief["message_sha256"] = reference["observation_message_sha256"]
        break
    if in_core:
        if reference is not None:
            core["last_failure"] = {"complete_packet_ref": brief}
            candidate = json.dumps(core, ensure_ascii=False) + mandatory[end:]
            if estimator.estimate(candidate) < estimator.estimate(mandatory):
                mandatory = candidate
            else:
                reference = None
        return mandatory, None, reference
    if reference is not None:
        candidate = mandatory + "\nCURRENT FAILURE REFERENCE\n" + _json(brief)
        original = mandatory + "\nCURRENT ORIGINAL FAILURE DATA\n" + _json(failure)
        if estimator.estimate(candidate) < estimator.estimate(original):
            return candidate, None, reference
    return mandatory, failure, None


def snapshot(
    state,
    workspace,
    mandatory,
    system,
    dialogue,
    raw_pairs,
    legacy_pairs,
    optional,
    *,
    estimator=None,
):
    from infinicoder.runtime.feedback import failure_packet

    views, refs = _views(state, workspace)
    visible = {item[3] for pair in legacy_pairs if (item := _source(pair)) is not None}
    raw = {item[3] for pair in raw_pairs if (item := _source(pair)) is not None}
    registered = {row["observation_sha256"] for row in refs}
    lost = tuple(sorted((raw & registered) - visible))
    protected = tuple(_protected(raw_pairs, state))
    mandatory, failure, reference = _current_failure(
        mandatory,
        raw_pairs,
        protected,
        failure_packet(state.get("last_failure")),
        estimator or TokenEstimator(),
    )
    return JointSnapshot(
        mandatory,
        system,
        _json(dialogue),
        _json(raw_pairs),
        _json(views),
        _json(refs),
        protected,
        _json(optional),
        _json(failure),
        lost,
        _json(reference),
    )


def assemble(assembler, frozen, *, force=False):
    """Select whole blocks from an immutable snapshot;no source access here."""
    if not isinstance(frozen, JointSnapshot):
        raise TypeError("JointSnapshot required")
    if not force and not frozen.lost_observations:
        return None
    views = json.loads(frozen.views_json)
    pairs = json.loads(frozen.pairs_json)
    dialogue = json.loads(frozen.dialogue_json)
    protected = set(frozen.protected)
    optional = json.loads(frozen.optional_json)
    complete_visible = set()
    current_versions = {
        row["origin_read_sha256"] for row in views if row["sha256"] == row["observed_sha256"]
    }
    for i in protected:
        item = _source(pairs[i])
        if item is not None and item[3] in current_versions:
            complete_visible.add(item[3])
    # A protected latest read is already present verbatim;do not duplicate it.
    candidates = [row for row in views if row["origin_read_sha256"] not in complete_visible]
    mandatory = frozen.mandatory
    failure = json.loads(frozen.failure_json)
    if failure:
        mandatory += "\nCURRENT ORIGINAL FAILURE DATA\n" + _json(failure)
    fixed = frozen.system + "".join(m["content"] for m in dialogue)
    fixed += "".join(m["content"] for i in sorted(protected) for m in pairs[i])
    assembler.assemble(
        mandatory, {}, fixed_overhead_text=fixed
    )  # Mandatory overflow is never optional.
    chosen = []
    queries = 0
    references = json.loads(frozen.references_json)

    def directory_reference(ref):
        # Path+source SHA+range already identify the omitted source fact. Do
        # not spend a second digest on an equal current version or copy the
        # observation-envelope digest into this compact model directory.
        result = {key: ref[key] for key in ("path", "observed_sha256", "start", "count", "status")}
        current = ref.get("current_sha256")
        if current == ref["observed_sha256"]:
            result["status"] = "current_source_text_omitted"
        elif current is not None:
            result["current_sha256"] = current
        if ref.get("reason"):
            result["reason"] = ref["reason"]
        return result

    def required_text(rows):
        covered = complete_visible | {
            row["origin_read_sha256"] for row in rows if row["sha256"] == row["observed_sha256"]
        }
        omitted = [
            directory_reference(ref)
            for ref in references
            if ref["observation_sha256"] not in covered
        ]
        source = LABEL + _json(dict(regions=rows)) if rows else ""
        directory = "\nOMITTED SOURCE REFERENCES\n" + _json(omitted) if omitted else ""
        return mandatory + source + directory, covered, bool(directory)

    def fits(required, history):
        nonlocal queries
        queries += 1
        if queries > 32:
            raise BudgetExceeded("bounded joint selection exceeded32 assembly trials")
        try:
            _, report = assembler.assemble(
                required, {"history": history or None}, fixed_overhead_text=fixed
            )
            return not history or "history" in {r["name"] for r in report["selected"]}
        except BudgetExceeded:
            return False

    for row in candidates:
        if fits(required_text([*chosen, row])[0], ""):
            chosen.append(row)
    required, covered, directory_visible = required_text(chosen)
    # The directory is explicit evidence of omission,never a prose summary.
    # If even this and protected feedback cannot fit,report the bound rather
    # than silently remove the caller contract or a pending edit receipt.
    assembler.assemble(required, {}, fixed_overhead_text=fixed)
    projected = deepcopy(pairs)
    for i, pair in enumerate(projected):
        item = _source(pair)
        if i in protected or item is None or item[3] not in covered:
            continue
        action, observation, source, observation_sha, shape = item
        if shape == "legacy":
            continue  # Do not rewrite unattributed historical wrappers.
        reference = dict(
            path=source["path"],
            observed_sha256=source["sha256"],
            start=source["start"],
            observation_sha256=observation_sha,
            status="same_version_complete_text_visible_in_this_request",
            original_event_retained_in_session=True,
            historical_behavior_not_revalidated=True,
        )
        # Change only the duplicate text;outlines,interfaces,errors and all
        # transport/provenance fields remain exactly the original values.
        new_source = deepcopy(source)
        new_source.pop("text")
        new_source["source_reference"] = reference
        new_result = deepcopy(observation)
        if action["tool"] == "workpack":
            new_result["source"] = new_source
        else:
            new_result = new_source
        if shape == "provenance":
            envelope = json.loads(pair[-1]["content"])
            envelope["observation"] = new_result
        else:
            envelope = new_result
        pair[-1]["content"] = json.dumps(envelope, ensure_ascii=False)
    selected = set(protected)
    for i in range(len(projected) - 1, -1, -1):
        if i in protected:
            continue
        trial = sorted(selected | {i})
        history = "\n".join(m["content"] for j in trial if j not in protected for m in projected[j])
        if fits(required, history):
            selected.add(i)
    history = "\n".join(m["content"] for i in sorted(selected - protected) for m in projected[i])
    parts = {key: value for key, value in optional.items() if key != "history"}
    parts["history"] = history or None
    text, report = assembler.assemble(required, parts, fixed_overhead_text=fixed)
    keep_history = "history" in {row["name"] for row in report["selected"]}
    delivered = selected if keep_history else protected
    messages = [dict(role="system", content=frozen.system), dict(role="user", content=text)]
    messages.extend(m for i in sorted(delivered) for m in projected[i])
    messages.extend(dialogue)
    report["joint_evidence"] = dict(
        active=True,
        trigger="physical_reassembly" if force else "registered_source_would_be_omitted",
        whole_regions=[
            {k: row[k] for k in ("path", "sha256", "start", "end", "origin_read_sha256")}
            for row in chosen
        ],
        protected_pairs=sorted(protected),
        selected_pairs=sorted(delivered),
        omitted_pairs=sorted(set(range(len(pairs))) - delivered),
        omission_references_visible=directory_visible,
        assembly_trials=queries,
        source_rereads_during_reassembly=0,
        historical_events_changed=False,
        extra_model_calls=0,
        current_failure_reference=json.loads(frozen.failure_reference_json),
    )
    return messages, report
