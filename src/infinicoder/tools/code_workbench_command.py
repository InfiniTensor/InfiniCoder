"""One argument contract shared by the virtual and real shell tool adapters."""

import argparse
import json


def run_workbench_command(workbench, args):
    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    command, *values = args
    if command == "locate":
        parser.add_argument("query")
        parser.add_argument("--path", action="append", default=[])
        parser.add_argument("--cursor")
    elif command == "inspect":
        parser.add_argument("target")
        parser.add_argument("--start", type=int)
        parser.add_argument("--end", type=int)
        parser.add_argument("--view", default="context")
        parser.add_argument("--cursor")
    elif command == "edit-body":
        parser.add_argument("symbol")
        parser.add_argument("--expected", required=True)
        parser.add_argument("--body-file", required=True)
    elif command == "edit-range":
        parser.add_argument("path")
        parser.add_argument("--expected", required=True)
        parser.add_argument("--start", type=int, required=True)
        parser.add_argument("--end", type=int, required=True)
        parser.add_argument("--new-file", required=True)
    elif command == "trial":
        parser.add_argument("--edit-script", required=True)
        parser.add_argument("--probe-script", required=True)
    elif command == "trial-show":
        parser.add_argument("identity")
        parser.add_argument("--part", choices=("result", "diff", "logs"), default="result")
        parser.add_argument("--cursor")
    elif command == "trial-apply":
        parser.add_argument("identity")
    else:
        raise ValueError("Unknown ic-code action;use the tools declared by this caller/profile")
    try:
        options = parser.parse_args(values)
    except (argparse.ArgumentError, SystemExit):
        raise ValueError("Invalid bounded tool arguments") from None
    workbench.remaining()
    try:
        if command == "locate":
            result = workbench.locate(options.query, path_globs=options.path, cursor=options.cursor)
        elif command == "inspect":
            result = workbench.inspect(
                options.target,
                start=options.start,
                end=options.end,
                view=options.view,
                cursor=options.cursor,
            )
        elif command == "edit-body":
            result = workbench.edit_body(options.symbol, options.expected, options.body_file)
        elif command == "edit-range":
            result = workbench.edit_range(
                options.path, options.expected, options.start, options.end, options.new_file
            )
        elif command == "trial":
            result = workbench.trial(options.edit_script, options.probe_script)
        elif command == "trial-show":
            result = workbench.trial_show(
                options.identity, part=options.part, cursor=options.cursor
            )
        else:
            result = workbench.trial_apply(options.identity)
    except (OSError, SyntaxError, UnicodeError, RecursionError) as error:
        raise ValueError(
            type(error).__name__
            + ": "
            + str(error)[:400]
            + ";inspect current sources before continuing"
        ) from None
    return json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
