"""A bounded alternative hypothesis in B's first ordinary SDK response."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re


class HypothesisNotEstablished(ValueError):
    pass


def instructions():
    return (
        "\nALTERNATIVE IMPLEMENTATION HYPOTHESIS\n"
        "A changed the source but failed the version-bound public check. Before your first "
        "ordinary command, write one line HYPOTHESIS: followed by a JSON object with exactly "
        "three nonempty string fields: previous_explanation (your inferred explanation of A), "
        "alternative_mechanism (a materially different implementation approach), and "
        "discriminating_check (a concrete public check that distinguishes them). "
        "Each field is at most 1000 characters. Then issue the usual single command in this "
        "same response. Do not request another planning turn. Use the actual public failure "
        "and readonly A source; do not invent private test results. These are hypotheses, "
        "not established facts. A reworded role or unchanged implementation is not evidence "
        "of diversity. Missing or malformed hypotheses stop this alternative without a retry."
    )


def parse(content):
    if not isinstance(content, str):
        raise HypothesisNotEstablished("first alternative response must contain text")
    # The descriptor must precede the first command fence, not live in code/output.
    prefix = content.split("```", 1)[0]
    lines = [
        line[len("HYPOTHESIS:") :].strip()
        for line in prefix.splitlines()
        if line.startswith("HYPOTHESIS:")
    ]
    if len(lines) != 1 or len(lines[0]) > 4096:
        raise HypothesisNotEstablished("one bounded hypothesis descriptor required")
    try:
        value = json.loads(lines[0])
    except (ValueError, RecursionError) as error:
        raise HypothesisNotEstablished("invalid hypothesis JSON") from error
    keys = {"previous_explanation", "alternative_mechanism", "discriminating_check"}
    if (
        not isinstance(value, dict)
        or set(value) != keys
        or any(not isinstance(v, str) or not v.strip() or len(v) > 1000 for v in value.values())
    ):
        raise HypothesisNotEstablished("three bounded hypothesis fields required")
    normalize = lambda text: re.sub(r"\W+", "", text.casefold())
    if normalize(value["previous_explanation"]) == normalize(value["alternative_mechanism"]):
        raise HypothesisNotEstablished("alternative repeats the previous explanation")
    return value


def evidence_binding(state):
    """Bind to controller-owned versions and verify the retained public raw report."""
    folder = Path(state) / "candidate-evidence"
    verdict = json.loads((folder / "public-A.json").read_bytes())
    changes = json.loads((folder / "changes.json").read_bytes())
    raw = (folder / "raw-public-A.json").read_bytes()
    if (
        verdict.get("status") != "public_fail"
        or not verdict.get("controller_origin")
        or verdict.get("independent_final_used") is not False
        or not changes.get("changed_entries")
        or verdict.get("tested_revision") != changes.get("A_revision")
        or verdict.get("retained_revision") != changes.get("A_revision")
        or verdict.get("raw_sha256") != hashlib.sha256(raw).hexdigest()
    ):
        raise HypothesisNotEstablished(
            "alternative requires unchanged version-bound public failure"
        )
    return dict(
        baseline_revision=changes["baseline_revision"],
        attempted_revision=changes["A_revision"],
        public_contract_sha256=verdict["contract_sha256"],
        public_raw_sha256=verdict["raw_sha256"],
    )


@contextmanager
def bind_first_response(model, state, record):
    binding = evidence_binding(state)
    original = model.query
    used = False
    record.update(status="pending", binding=binding, interpretation="model hypotheses, not facts")

    def query(*args, **kwargs):
        nonlocal used
        response = original(*args, **kwargs)
        if not used:
            used = True
            try:
                descriptor = parse(response.get("content") if isinstance(response, dict) else None)
            except HypothesisNotEstablished as error:
                record.update(status="not_established", reason=str(error))
                raise
            record.update(
                status="proposed",
                descriptor=descriptor,
                diversity="unconfirmed until source and external checks change",
            )
        return response

    model.query = query
    try:
        yield
    finally:
        model.query = original
