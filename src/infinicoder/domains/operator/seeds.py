"""Applicability checks for reusable seeds. Unknown preconditions never pass."""

import json
from pathlib import Path


class SeedCatalog:
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.entries = []
        for manifest in sorted(self.root.glob("*/seed.json")):
            seed = json.loads(manifest.read_text(encoding="utf-8"))
            for key in ("id", "preconditions", "invariants", "evidence", "source"):
                if key not in seed:
                    raise ValueError(f"{manifest}: missing {key}")
            if not seed["preconditions"] or not seed["evidence"] or not seed["invariants"]:
                raise ValueError("Seed requires explicit preconditions, invariants and evidence")
            source = (manifest.parent / seed["source"]).resolve()
            if not source.is_relative_to(manifest.parent.resolve()) or not source.is_file():
                raise ValueError("Seed source must exist within its directory")
            seed["source_path"] = str(source)
            self.entries.append(seed)

    def select(self, facts):
        results = []
        for seed in self.entries:
            failures, unknown = [], []
            for condition in seed["preconditions"]:
                key, op, expected = condition["fact"], condition["op"], condition["value"]
                if op not in ("eq", "in", "ge", "divisible_by"):
                    raise ValueError("Unknown precondition operator")
                if key not in facts:
                    unknown.append(key)
                    continue
                actual = facts[key]
                try:
                    if op == "eq":
                        ok = actual == expected
                    elif op == "in":
                        ok = actual in expected
                    elif op == "ge":
                        ok = actual >= expected
                    else:
                        ok = expected > 0 and actual % expected == 0
                except (TypeError, ValueError):
                    ok = False
                if not ok:
                    failures.append(condition)
            results.append(
                {
                    "id": seed["id"],
                    "applicable": not failures and not unknown,
                    "mismatches": failures,
                    "unknown_facts": unknown,
                    "invariants": seed["invariants"],
                    "evidence": seed["evidence"],
                    "source_path": seed["source_path"],
                    "requires_fresh_verification": True,
                }
            )
        return results
