"""Small shell client; no package imports, model client, or host-path API."""

import json
import os
import socket
import sys
import time

SOCKET = "/tmp/task-scratch/.ic-code.sock"
MAX_REQUEST = 16384
MAX_RESPONSE = 1024 * 1024


def main():
    try:
        lease = os.environ["IC_CODE_LEASE"]
        remaining = float(os.environ["IC_CODE_DEADLINE"]) - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("original action deadline expired")
        payload = json.dumps(dict(lease=lease, argv=sys.argv[1:])).encode() + b"\n"
        if len(payload) > MAX_REQUEST:
            raise ValueError("arguments exceed16KiB;write payload to task scratch")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(remaining)
            client.connect(SOCKET)
            client.sendall(payload)
            with client.makefile("rb") as stream:
                raw = stream.readline(MAX_RESPONSE + 1)
            if len(raw) > MAX_RESPONSE or not raw.endswith(b"\n"):
                raise ValueError("incomplete or oversized tool response;outcome unknown")
            response = json.loads(raw)
        if type(response.get("returncode")) is not int or not isinstance(
            response.get("output"), str
        ):
            raise ValueError("invalid tool response;outcome unknown")
        print(response["output"])
        return response["returncode"]
    except (KeyError, OSError, ValueError, TimeoutError) as error:
        print(
            json.dumps(
                dict(
                    error=type(error).__name__ + ": " + str(error),
                    tool_outcome="not_confirmed;do not automatically retry writes",
                    fallback="Ordinary shell reading and editing remain available.",
                )
            )
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
