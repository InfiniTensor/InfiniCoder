"""Opt-in complete XML compatibility without changing the shell workflow prompt.

Only two existing,strict grammars are accepted. No prose stripping,tag repair,
batch execution,action inference or candidate execution is used for parsing.
"""

from infinicoder._resources import package_root

import hashlib


def identity():
    root = package_root()
    return hashlib.sha256(
        b"".join(
            (root / name).read_bytes()
            for name in (
                "runtime/compatible_tools.py",
                "tools/native_tools.py",
                "tools/tool_interface.py",
                "runtime/protocol.py",
                "runtime/shell_loop.py",
                "runtime/engine.py",
            )
        )
    ).hexdigest()


def decode(text, available, schema, *, finish_reason):
    from infinicoder.runtime.shell_loop import decode as shell_decode
    from infinicoder.tools.native_tools import decode as xml_decode

    if finish_reason not in ("stop", "end_turn", "tool_calls"):
        raise ValueError("Incomplete compatible reply;no action dispatched")
    try:
        return shell_decode(text, available), "shell"
    except ValueError as shell_error:
        try:
            return xml_decode(text, schema, finish_reason=finish_reason), "xml"
        except ValueError as xml_error:
            raise ValueError(
                "No complete supported action. Shell: "
                + str(shell_error)
                + "; XML: "
                + str(xml_error)
            ) from None
