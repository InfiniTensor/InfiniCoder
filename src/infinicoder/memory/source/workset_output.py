"""A bounded readonly output wrapper; never a shell or a subprocess executor."""

from dataclasses import dataclass
import shlex


@dataclass(frozen=True)
class WorksetOutputRequest:
    argv: tuple[str, ...]
    head_lines: int | None = None
    merge_stderr: bool = False


def _tokens(command):
    """Keep operator provenance so quoted query text is never a pipeline."""
    if not isinstance(command, str) or len(command) > 16384:
        raise ValueError("ic-workset command exceeds its bounded input")
    result = []
    start = None
    quote = None
    i = 0

    def word(end):
        nonlocal start
        if start is not None:
            values = shlex.split(command[start:end], posix=True)
            if len(values) != 1:
                raise ValueError("ic-workset requires one quoted argument per word")
            result.append((values[0], False, command[start:end], start, end))
            start = None

    while i < len(command):
        char = command[i]
        if quote is not None:
            if quote == '"' and char in ("$", "`"):
                raise ValueError("ic-workset does not expand shell expressions")
            if char == quote:
                quote = None
            elif char == "\\" and quote == '"':
                i += 1
                if i >= len(command):
                    raise ValueError("ic-workset has an unfinished escape")
            i += 1
            continue
        if char in "\r\n":
            raise ValueError("ic-workset requires one command line")
        if char.isspace():
            word(i)
            i += 1
            continue
        if char in ("$", "`"):
            raise ValueError("ic-workset does not expand shell expressions")
        if char in ("'", '"'):
            if start is None:
                start = i
            quote = char
            i += 1
            continue
        if char == "\\":
            if start is None:
                start = i
            i += 2
            if i > len(command):
                raise ValueError("ic-workset has an unfinished escape")
            continue
        if char in "|&;<>":
            word(i)
            end = i + 1
            while end < len(command) and command[end] in "|&;<>":
                end += 1
            result.append((command[i:end], True, command[i:end], i, end))
            i = end
            continue
        if start is None:
            start = i
        i += 1
    if quote is not None:
        raise ValueError("ic-workset has an unfinished quote")
    word(len(command))
    return result


def parse_workset_output(command):
    tokens = _tokens(command)
    pipes = [i for i, (value, operator, *_) in enumerate(tokens) if operator and value == "|"]
    if len(pipes) > 1:
        raise ValueError("ic-workset supports only one readonly head output filter")
    head = None
    if pipes:
        at = pipes[0]
        suffix = tokens[at + 1 :]
        tokens = tokens[:at]
        if any(operator for _, operator, *_ in suffix):
            raise ValueError("ic-workset output filter cannot contain shell operators")
        args = [value for value, *_ in suffix]
        if args == ["head"]:
            head = 10
        elif (
            len(args) == 3
            and args[:2] == ["head", "-n"]
            and args[2].isascii()
            and args[2].isdigit()
        ):
            head = int(args[2])
        elif len(args) == 2 and args[0] == "head":
            value = (
                args[1][2:]
                if args[1].startswith("-n")
                else args[1][1:]
                if args[1].startswith("-")
                else ""
            )
            if value.isascii() and value.isdigit():
                head = int(value)
        if head is None or not 1 <= head <= 1000:
            raise ValueError(
                "ic-workset supports head with1..1000 lines only;no files or byte filters"
            )
    merged = False
    redirect = tokens[-3:]
    if (
        len(redirect) == 3
        and [token[:3] for token in redirect]
        == [("2", False, "2"), (">&", True, ">&"), ("1", False, "1")]
        and redirect[0][4] == redirect[1][3]
        and redirect[1][4] == redirect[2][3]
    ):
        tokens = tokens[:-3]
        merged = True
    if any(operator for _, operator, *_ in tokens):
        raise ValueError(
            "ic-workset supports a single query,optional2>&1 and head;no other shell operations"
        )
    args = tuple(value for value, *_ in tokens)
    if not 2 <= len(args) <= 32 or args[0] != "ic-workset":
        raise ValueError("one complete ic-workset SUBCOMMAND invocation required")
    return WorksetOutputRequest(args, head, merged)
