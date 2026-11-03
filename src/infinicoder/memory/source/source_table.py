"""Lossless factoring of visible source provenance;no summarization or reads."""

import json
import re


GUIDANCE = """
Source rows use file_id references into source_files. Each file entry contains
the exact path and full SHA256 for that version;row line/end/text/kind retain
their original meaning and uncertainty. File IDs are local to this source plan,
not paths or edit authority. read_ids may use the currently visible file aliases
in reference_contract;state fields still require field IDs/names. Nothing here
is a tool-read receipt,dynamic binding proof or acceptance."""


def _key(path, version):
    if (
        not isinstance(path, str)
        or not path
        or not isinstance(version, str)
        or not re.fullmatch("[0-9a-f]{64}", version)
    ):
        raise ValueError("Source table requires an exact path and full SHA256")
    return json.dumps([path, version], ensure_ascii=False, separators=(",", ":"))


def register(plan):
    """Append IDs for new dependencies;existing plan-version IDs never shift."""
    ledger = plan.setdefault("source_file_ids", {})
    if len(set(ledger.values())) != len(ledger) or set(ledger.values()) != {
        "F" + str(i) for i in range(len(ledger))
    }:
        raise ValueError("Invalid source file ID ledger")
    scope = plan["scope"]
    inputs = [(plan["path"], scope["source_sha256"])]
    inputs.extend((r["path"], r["source_sha256"]) for r in plan["records"])
    for path, version in inputs:
        key = _key(path, version)
        if key not in ledger:
            ledger[key] = "F" + str(len(ledger))
    return ledger


def project(records, plan):
    ledger = register(plan)
    files = {}
    rows = []
    aliases = {}
    target = _key(plan["path"], plan["scope"]["source_sha256"])
    target_id = ledger[target]
    files[target_id] = dict(
        id=target_id, path=plan["path"], source_sha256=plan["scope"]["source_sha256"]
    )
    aliases[target_id] = ["T0"]
    for record in records:
        if "file_id" in record:
            raise ValueError("Cannot project an already factored source row")
        key = _key(record["path"], record["source_sha256"])
        if key not in ledger:
            raise ValueError("Visible source row is outside the current plan ledger")
        identity = ledger[key]
        files[identity] = dict(
            id=identity, path=record["path"], source_sha256=record["source_sha256"]
        )
        value = {k: v for k, v in record.items() if k not in ("path", "source_sha256")}
        value["file_id"] = identity
        rows.append(value)
        aliases.setdefault(identity, []).append(record["id"])
    table = sorted(files.values(), key=lambda r: int(r["id"][1:]))
    if restore(rows, table) != records:
        raise ValueError("Source provenance projection is not reversible")
    return rows, table, aliases


def restore(records, files):
    mapping = {}
    versions = set()
    for file in files:
        if (
            set(file) != {"id", "path", "source_sha256"}
            or not isinstance(file["id"], str)
            or not re.fullmatch(r"F\d+", file["id"])
        ):
            raise ValueError("Invalid source file entry")
        key = _key(file["path"], file["source_sha256"])
        if file["id"] in mapping or key in versions:
            raise ValueError("Duplicate source file identity")
        mapping[file["id"]] = file
        versions.add(key)
    rows = []
    for record in records:
        if (
            "path" in record
            or "source_sha256" in record
            or not isinstance(record.get("file_id"), str)
            or record.get("file_id") not in mapping
        ):
            raise ValueError("Invalid or mixed source row representation")
        value = dict(record)
        file = mapping[value.pop("file_id")]
        value.update(path=file["path"], source_sha256=file["source_sha256"])
        rows.append(value)
    return rows


def render(value, *, compact=False):
    """Canonical separator encoding;string contents/parsed values stay exact."""
    ordinary = json.dumps(value, ensure_ascii=False)
    if not compact:
        return ordinary
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if json.loads(encoded) != json.loads(ordinary):
        raise ValueError("Source packet JSON encoding changed its values")
    return encoded
