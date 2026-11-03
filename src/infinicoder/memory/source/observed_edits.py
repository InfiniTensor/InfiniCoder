"""Optional editing against a runtime-recorded read revision.

The model may omit a long hash; Workspace.edit still performs both original
compare-and-swap checks. Notes, command output and code indexes cannot mint reads.
"""

from infinicoder._resources import package_root

from copy import deepcopy
import hashlib
import re


GUIDANCE = """Existing-file edit: first read the file using the read tool,then
use edit {path,old,new} or edit {path,start,end,new}. Omit expected_hash (or use
null) to bind the edit to the most recent actual read of that path. The runtime
supplies that recorded revision,never the current unobserved file hash. Re-read
after changes or a stale-version error. Explicit hashes still require an exact
match. For a NEW file use edit {path,expected_hash:null,content}. Creating a file
does not authorize replacing an existing unread file. Read notes and this guard
do not establish correctness;all independent final checks remain mandatory."""


def identity():
    root = package_root()
    return hashlib.sha256(
        b"".join(
            (root / name).read_bytes()
            for name in (
                "memory/source/observed_edits.py",
                "execution/workspace_tools.py",
                "runtime/protocol.py",
            )
        )
    ).hexdigest()


def _path(workspace, path):
    return workspace.path(path).relative_to(workspace.root).as_posix()


def record(state, workspace, result):
    """Called only after successful runtime read/read_symbol; never on model notes."""
    digest = result.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
        raise ValueError("Actual read revision required")
    path = _path(workspace, result["path"])
    entries = state.setdefault("observed_edit_reads", {})
    entries.pop(path, None)
    evicted = None
    if len(entries) >= 128:
        evicted = next(iter(entries))
        entries.pop(evicted)
    entries[path] = dict(
        sha256=digest,
        start=result["start"],
        observed_lines=len(result["text"].splitlines()),
        origin="runtime_read",
    )
    receipt = dict(path=path, source_sha256=digest, origin="runtime_read", acceptance_proof=False)
    if evicted is not None:
        receipt["forgotten_path"] = evicted
    return receipt


def prepare(state, workspace, arguments):
    """Resolve omitted/null hashes from recorded observations, not current bytes."""
    args = deepcopy(arguments)
    path = _path(workspace, args.get("path"))
    if args.get("expected_hash") is not None:
        return args, None
    # Missing files retain the original create-only None precondition. If
    # another writer creates the path later, Workspace's atomic check rejects it.
    if not workspace.path(args["path"]).exists():
        args["expected_hash"] = None
        return args, None
    entry = state.get("observed_edit_reads", {}).get(path)
    if not entry or entry.get("origin") != "runtime_read":
        raise ValueError("Existing file has no runtime read revision;use read before editing")
    digest = entry.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
        raise ValueError("Invalid recorded read revision;read the file again")
    args["expected_hash"] = digest
    return args, dict(
        path=path, source_sha256=digest, origin="runtime_read", acceptance_proof=False
    )


def guidance(prompt):
    # Compact frontends otherwise instruct the model to copy a long hash.
    prompt = prompt.replace(
        "read returns a source\nhash;edit uses that exact current expected_hash and unique old/new or start/end/new.",
        "read records a source revision;edit may use unique old/new or start/end/new without copying its hash.",
    )
    return prompt + "\n" + GUIDANCE
