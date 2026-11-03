"""Explicit, composable domain obligations; no task-name or hardware-host dispatch."""

from infinicoder._resources import package_file

import json
from pathlib import Path


class CapabilityRegistry:
    def __init__(self, directory=None):
        self.directory = Path(directory or package_file("capabilities"))
        self.entries = {}
        for path in sorted(self.directory.glob("*.json")):
            item = json.loads(path.read_text(encoding="utf-8"))
            if item["id"] in self.entries:
                raise ValueError("Duplicate capability id")
            if not item.get("obligations") or not item.get("checks"):
                raise ValueError("Capability needs obligations and checks")
            self.entries[item["id"]] = item

    def compose(self, ids):
        missing = sorted(set(ids) - self.entries.keys())
        if missing:
            raise ValueError("Unknown capabilities: " + ", ".join(missing))
        return [self.entries[key] for key in dict.fromkeys(ids)]

    def context(self, ids, facts=None):
        return json.dumps(
            {
                "capabilities": self.compose(ids),
                "supplied_target_facts": facts or {},
                "rule": "Missing target/API facts are unknown. Inspect source or ask; do not invent them. "
                "Templates suggest structure, never prove correctness or performance.",
            },
            ensure_ascii=False,
        )
