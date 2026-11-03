"""Frozen standalone client copied into a task namespace; stdlib only."""

import argparse
import json
import socket


def main():
    p = argparse.ArgumentParser(
        description="Run public development checks or read an existing report without rerunning"
    )
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("run")
    read = sub.add_parser("read")
    read.add_argument("report_id")
    read.add_argument("--offset", type=int, default=0)
    read.add_argument("--limit", type=int, default=2000)
    args = p.parse_args()
    request = vars(args)
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    # The server clamps execution to the shared task deadline; the ordinary
    # command remains under its original enclosing command/task timeout.
    client.connect("/tmp/ic-public-check.sock")
    with client, client.makefile("rwb") as stream:
        stream.write((json.dumps(request) + "\n").encode())
        stream.flush()
        raw = stream.readline(128 * 1024 + 1)
        if len(raw) > 128 * 1024:
            raise ValueError("public response exceeds client bound")
        response = json.loads(raw)
    print(json.dumps(response, ensure_ascii=False))
    if not response.get("ok"):
        return 2
    result = response["result"]
    if args.action == "read":
        return 0
    if result.get("error"):
        return 2
    process, summary = result.get("process", {}), result.get("summary", {})
    return (
        0
        if process.get("exit_code") == 0
        and not process.get("timed_out")
        and not process.get("truncated")
        and summary.get("passed", 0) > 0
        and summary.get("failed", 0) == 0
        and summary.get("error", 0) == 0
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
