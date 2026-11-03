"""Read-only dispatch for a real source-action worker;no virtual shell replay."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import uuid

from infinicoder.memory.source.source_question_action import (
    answer_source_question,
    SourceAnswerCapacityError,
)
from infinicoder.planning.task_source_plan import dump


def question_file(path, scratch):
    prefix = "/tmp/task-scratch/"
    if not isinstance(path, str) or not path.startswith(prefix):
        raise ValueError("question file must be under the caller task scratch mapping")
    relative = path[len(prefix) :]
    parts = PurePosixPath(relative)
    if (
        not relative
        or parts.is_absolute()
        or ".." in parts.parts
        or str(parts) != relative
        or "\\" in relative
    ):
        raise ValueError("normalized task-scratch question path required")
    root = Path(scratch).resolve()
    file = root / relative
    if any(p.is_symlink() for p in (file, *file.parents)) or not file.resolve().is_relative_to(
        root
    ):
        raise ValueError("question payload cannot traverse symlinks or leave task scratch")
    fd = os.open(file, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > 8192:
            raise ValueError("regular question file at most8192 bytes required")
        raw = stream.read(8193)
        after = os.fstat(stream.fileno())
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if (
        len(raw) > 8192
        or identity(before) != identity(after)
        or identity(after) != identity(file.lstat())
    ):
        raise ValueError("question file changed during reading")
    return raw.decode("utf8")


def run_source_question_command(reader, args, *, scratch, deadline):
    if not args or args[0] not in ("ask", "view", "locate", "inspect"):
        raise ValueError("readonly source tools permit ask/view/locate/inspect;no edits or trials")
    command, *values = args
    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    if command == "ask":
        parser.add_argument("question", nargs="?")
        parser.add_argument("--anchor")
        parser.add_argument("--question-file")
    elif command == "view":
        parser.add_argument("ref")
        parser.add_argument("--historical", action="store_true")
    elif command == "locate":
        parser.add_argument("query")
        parser.add_argument("--path", action="append", default=[])
        parser.add_argument("--cursor")
    else:
        parser.add_argument("target")
        parser.add_argument("--start", type=int)
        parser.add_argument("--end", type=int)
        parser.add_argument(
            "--view",
            choices=("context", "source", "module", "relations", "flow"),
            default="context",
        )
        parser.add_argument("--cursor")
    try:
        options = vars(parser.parse_args(values))
    except (argparse.ArgumentError, SystemExit):
        raise ValueError("invalid bounded readonly source arguments") from None

    def fit(view):
        try:
            if reader.workbench.count_output(dump(view)) <= min(
                2048, reader.workbench.output_limit
            ):
                return view
        except SourceAnswerCapacityError:
            pass
        limited = dict(
            identity="source_question_action_v1",
            read_event_ref="source-view:" + uuid.uuid4().hex,
            status="budget_limited",
            records=[],
            source_files={},
            unresolved=[dict(status="complete_view_does_not_fit_current_input")],
        )
        return reader._record_view(
            dict(op="budget_limited", original_view_ref=view["read_event_ref"]), limited
        )

    if command == "ask":
        if bool(options["question"]) == bool(options["question_file"]):
            raise ValueError("one literal question or task-scratch question file required")
        question = (
            options["question"]
            if options["question"]
            else question_file(options["question_file"], scratch)
        )
        try:
            view = answer_source_question(
                question,
                options["anchor"],
                reader,
                lambda messages, remaining: reader.workbench.count_output(messages[0]["content"]),
                deadline,
                max_tokens=min(2048, reader.workbench.output_limit),
            )
        except ValueError as error:
            if (
                not isinstance(error, SourceAnswerCapacityError)
                and str(error) != "minimal failure metadata exceeds the actual answer budget"
            ):
                raise
            view = dict(
                identity="source_question_action_v1",
                read_event_ref="source-view:" + uuid.uuid4().hex,
                status="budget_limited",
                records=[],
                source_files={},
                unresolved=[dict(status="current_SDK_input_has_no_answer_capacity")],
            )
        return fit(
            reader._record_view(dict(op="ask", question=question, anchor=options["anchor"]), view)
        )
    if command == "view":
        ref = options["ref"]
        if not re.fullmatch(r"source-view:[0-9a-f]{32}", ref):
            raise ValueError("exact returned source-view reference required")
        row = reader.index.db.execute(
            "SELECT sha256,record FROM source_plan_views WHERE ref=?", (ref,)
        ).fetchone()
        if row is None or hashlib.sha256(row["record"].encode()).hexdigest() != row["sha256"]:
            raise ValueError("missing or inconsistent original source view")
        record = json.loads(row["record"])
        if record["namespace"] != reader.namespace:
            raise ValueError("source view belongs to another namespace")
        view = dict(record["view"])
        changed = []
        for path, digest in view["source_files"].items():
            try:
                current = reader.current_version(path)
            except (OSError, ValueError, UnicodeError):
                current = None
            if current != digest:
                changed.append(path)
        view.update(
            read_event_ref="source-view:" + uuid.uuid4().hex,
            original_view_ref=ref,
            changed_sources=changed,
        )
        if changed:
            view["original_status"] = view.get("status")
            view["status"] = (
                "historical_stale"
                if options["historical"]
                else "stale;ask_or_inspect_current_source"
            )
            if not options["historical"]:
                view["records"] = []
        view["historical_requested"] = options["historical"]
        return fit(
            reader._record_view(dict(op="view", ref=ref, historical=options["historical"]), view)
        )
    if command == "locate":
        options["path_globs"] = options.pop("path")
    return fit(reader.query(dict(op=command, **options)))
