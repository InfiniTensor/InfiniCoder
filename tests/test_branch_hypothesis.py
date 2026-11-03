"""Hypotheses bind to real public evidence and never add a model request."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from infinicoder.search.branch_hypothesis import (
    bind_first_response,
    evidence_binding,
    HypothesisNotEstablished,
    parse,
)
from infinicoder.search.native_candidate_search import CandidateSearchPolicy


DESCRIPTOR = dict(
    previous_explanation="Discard negative values as invalid input",
    alternative_mechanism="Preserve signed contributions in the total",
    discriminating_check="Check total([-3, 2]) equals -1",
)


class BranchHypothesisTests(unittest.TestCase):
    def evidence(self, root):
        folder = root / "candidate-evidence"
        folder.mkdir()
        raw = b'{"status":"failed","case":"signed total"}'
        (folder / "raw-public-A.json").write_bytes(raw)
        (folder / "changes.json").write_text(
            json.dumps(
                dict(
                    baseline_revision="0" * 64, A_revision="1" * 64, changed_entries=["solution.py"]
                )
            )
        )
        (folder / "public-A.json").write_text(
            json.dumps(
                dict(
                    status="public_fail",
                    controller_origin=True,
                    independent_final_used=False,
                    tested_revision="1" * 64,
                    retained_revision="1" * 64,
                    contract_sha256="2" * 64,
                    raw_sha256=hashlib.sha256(raw).hexdigest(),
                )
            )
        )
        return folder

    def test_first_ordinary_response_is_recorded_without_additional_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.evidence(root)
            calls = []
            response = dict(
                content="HYPOTHESIS: "
                + json.dumps(DESCRIPTOR)
                + "\n```mswea_bash_command\necho inspect\n```"
            )

            def query(messages):
                calls.append(messages)
                return response if len(calls) == 1 else dict(content="ordinary continuation")

            model = SimpleNamespace(query=query)
            record = {}
            with bind_first_response(model, root, record):
                self.assertIs(model.query(["task"]), response)
                model.query(["continuation"])
            self.assertIs(model.query, query)
            self.assertEqual(len(calls), 2)
            self.assertEqual(record["status"], "proposed")
            self.assertEqual(record["binding"]["attempted_revision"], "1" * 64)
            self.assertEqual(record["descriptor"], DESCRIPTOR)

    def test_missing_repeated_or_stale_hypotheses_cannot_authorize_alternative(self):
        repeated = dict(DESCRIPTOR, alternative_mechanism=DESCRIPTOR["previous_explanation"])
        for text in (
            "ordinary action",
            "HYPOTHESIS: {}",
            "HYPOTHESIS: " + json.dumps(repeated),
            "```bash\necho HYPOTHESIS: " + json.dumps(DESCRIPTOR),
        ):
            with self.assertRaises(HypothesisNotEstablished):
                parse(text)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = self.evidence(root)
            (folder / "raw-public-A.json").write_bytes(b"different report")
            with self.assertRaises(HypothesisNotEstablished):
                evidence_binding(root)

    def test_default_search_contract_retains_original_identity(self):
        self.assertEqual(CandidateSearchPolicy().identity, "public_counterexample_branch_v1")
        self.assertEqual(
            CandidateSearchPolicy(hypothesis_branch=True).identity, "public_hypothesis_branch_v1"
        )


if __name__ == "__main__":
    unittest.main()
