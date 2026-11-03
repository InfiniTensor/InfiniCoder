"""Pure evidence-operation selection; no source, shell, model or check runs.

Offers must be computed by trusted, version-bound producers. An available
operation is not automatically relevant, and a model claim is not an offer.
"""

import hashlib
import json
import math


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _known_revision(value):
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


def choose_evidence_operation(
    *,
    need,
    offers,
    dispatched=(),
    remaining_seconds,
    decision_reserve,
    available_result_tokens,
    operations_used,
):
    """Choose a single applicable public operation, or an explicit no-op.

    Need and offers describe observed facts, not planned/model-reported effects.
    The executor must revalidate the selected source and deadline at dispatch.
    Unknown costs, unknown provenance and incomplete source identities do not
    acquire execution authority here. No result is an acceptance certificate.
    """

    def noop(reason):
        return dict(operation=None, reason=reason, acceptance_proof=False)

    if not isinstance(need, dict) or need.get("origin") != "linked_actual_feedback":
        return noop("no_linked_evidence_need")
    if not _known_revision(need.get("revision")) or not need.get("trigger_refs"):
        return noop("unknown_source_or_trigger_identity")
    if need.get("observer_complete") is not True or need.get("signal") not in (
        "three_same_observations",
        "two_complete_cycles",
        "current_public_fault",
    ):
        return noop("incomplete_or_unsupported_need")
    if type(operations_used) is not int or operations_used < 0 or operations_used >= 2:
        return noop("automatic_operation_limit")
    if (
        any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
            for v in (remaining_seconds, decision_reserve)
        )
        or remaining_seconds <= decision_reserve
    ):
        return noop("no_shared_time_after_decision_reserve")
    if type(available_result_tokens) is not int or available_result_tokens <= 0:
        return noop("no_result_capacity")
    reasons = []
    for kind in ("scope_audit", "literal_bounds", "public_validation"):
        for offer in offers:
            if not isinstance(offer, dict) or offer.get("kind") != kind:
                continue
            if (
                offer.get("origin") != "caller_capability"
                or offer.get("source_revision") != need["revision"]
            ):
                reasons.append("stale_or_untrusted_offer")
                continue
            if not offer.get("identity") or offer.get("busy") is not False:
                reasons.append("missing_identity_or_execution_busy")
                continue
            if kind == "scope_audit" and not (
                offer.get("explicit_body_permission") is True
                and offer.get("complete_historical_source") is True
                and offer.get("scope_change_observed") is True
            ):
                reasons.append("scope_not_relevant")
                continue
            if kind == "literal_bounds" and not (
                offer.get("exact_literal_read") is True
                and offer.get("read_starts_past_eof") is True
            ):
                reasons.append("no_verified_literal_eof")
                continue
            if kind == "public_validation":
                if offer.get("current_public_check") is True:
                    reasons.append("already_current_public_result")
                    continue
                if (
                    offer.get("current_public_check") is not False
                    or offer.get("source_mutation_allowed") is not False
                ):
                    reasons.append("unknown_public_result_or_effect_contract")
                    continue
                if not offer.get("coverage") or not offer.get("scope"):
                    reasons.append("no_explicit_public_coverage")
                    continue
            bound = offer.get("result_token_bound")
            timeout = offer.get("timeout_seconds")
            if type(bound) is not int or bound <= 0 or bound > available_result_tokens:
                reasons.append("declared_result_does_not_fit")
                continue
            if (
                isinstance(timeout, bool)
                or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout)
                or timeout <= 0
            ):
                reasons.append("unknown_execution_bound")
                continue
            question = dict(
                scope_audit="outside_body_change",
                literal_bounds="literal_source_extent",
                public_validation="current_public_behavior",
            )[kind]
            key = _digest(
                dict(
                    operation=offer["identity"], source_revision=need["revision"], question=question
                )
            )
            if key in dispatched:
                reasons.append("already_dispatched_this_evidence_question")
                continue
            return dict(
                operation=kind,
                identity=offer["identity"],
                dispatch_key=key,
                question=question,
                trigger_refs=need["trigger_refs"],
                source_revision=need["revision"],
                timeout_seconds=min(timeout, 60, remaining_seconds - decision_reserve),
                result_token_bound=bound,
                reason="applicable_to_current_evidence_need",
                acceptance_proof=False,
            )
    return noop("no_applicable_operation:" + ",".join(dict.fromkeys(reasons)))
