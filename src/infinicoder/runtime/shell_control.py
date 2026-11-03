"""Conservative shell *syntax* classification for read-only diagnostics.

This never executes or rewrites a command. Foreground syntax does not prove
that an external program cannot fork, or authorize stopping an agent task.
Unsupported shell constructs remain unknown, rather than guessed safe.
"""

import re


_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
_COMPOUND = frozenset(
    (
        "if",
        "then",
        "elif",
        "else",
        "fi",
        "for",
        "while",
        "until",
        "do",
        "done",
        "case",
        "esac",
        "select",
        "function",
        "coproc",
        "[[",
        "]]",
    )
)
_SHELLS = frozenset(("sh", "bash", "dash", "zsh", "ksh", "fish"))


def classify_shell_control(command):
    """Return status, reason and the offending operator's original position.

    Supported: ordinary words, quotes/escapes/comments, foreground lists and
    pipes, and ordinary redirections. Substitutions, heredocs, compound syntax
    and explicit nested shell executors are deliberately unsupported. This is
    an opportunity filter, not a shell validator or a process safety boundary.
    """

    def result(status, reason, position=None, operator=None):
        return dict(status=status, reason=reason, position=position, operator=operator)

    if not isinstance(command, str) or not command.strip():
        return result("unknown", "missing_command")
    words = []
    word = []
    word_start = None
    command_start = True
    quote = None
    quote_start = None
    background = None
    i = 0

    def finish_word():
        nonlocal word_start, command_start
        if word_start is not None:
            value = "".join(word)
            words.append((value, word_start, command_start))
            if not (command_start and _ASSIGNMENT.match(value)):
                command_start = False
            word.clear()
            word_start = None

    while i < len(command):
        c = command[i]
        if quote == "'":
            if c == "'":
                quote = None
            else:
                word.append(c)
            i += 1
            continue
        if c == "\\":
            if i + 1 == len(command):
                return result("unknown", "dangling_escape", i, c)
            if word_start is None:
                word_start = i
            if command[i + 1] != "\n":
                word.append(command[i + 1])
            i += 2
            continue
        if c == "`" or (c == "$" and command[i : i + 2] in ("$(", "${", "$'", '$"')):
            return result("unknown", "unsupported_substitution", i, command[i : i + 2])
        if quote == '"':
            if c == '"':
                quote = None
            else:
                word.append(c)
            i += 1
            continue
        if c in ("'", '"'):
            if word_start is None:
                word_start = i
            quote = c
            quote_start = i
            i += 1
            continue
        if c == "#" and word_start is None:
            end = command.find("\n", i)
            i = len(command) if end < 0 else end
            continue
        if c in "(){}" or command.startswith("<<", i):
            return result("unknown", "unsupported_compound_or_heredoc", i, command[i : i + 2])
        if c.isspace():
            finish_word()
            if c == "\n":
                command_start = True
            i += 1
            continue
        if c in "&|;<>":
            finish_word()
            if command.startswith((";&", ";;&"), i):
                return result("unknown", "unsupported_case_control", i, command[i : i + 3])
            operator = next(
                (
                    op
                    for op in ("&>>", "&&", "||", "|&", ">&", "<&", "&>", ">>", "<>", ">|")
                    if command.startswith(op, i)
                ),
                c,
            )
            if operator == "&" and background is None:
                background = i
            if operator in ("&", "&&", "||", "|", "|&", ";"):
                command_start = True
            i += len(operator)
            continue
        if word_start is None:
            word_start = i
        word.append(c)
        i += 1
    if quote is not None:
        return result("unknown", "unclosed_quote", quote_start, quote)
    finish_word()
    for value, position, first in words:
        # Conservatively reject explicit shell executor words even after
        # wrappers such as env/timeout. Literal arguments can also be unknown;
        # no command is blocked or modified because of that uncertainty.
        if value.rsplit("/", 1)[-1] in _SHELLS or value in ("eval", "source"):
            return result("unknown", "unsupported_shell_executor_word", position, value)
        if first and (value in _COMPOUND or value in (".", "trap")):
            return result("unknown", "unsupported_compound_command", position, value)
    if background is not None:
        return result("explicit_background", "asynchronous_list_operator", background, "&")
    if not words:
        return result("unknown", "no_command_word")
    return result("foreground_syntax", "supported_foreground_syntax")
