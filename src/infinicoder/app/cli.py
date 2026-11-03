"""User-facing Runtime CLI; advanced options remain available explicitly."""

import json
import sys


def main(argv=None):
    from infinicoder.app.config import RuntimeConfig
    from infinicoder.runtime.engine import main as runtime_main

    try:
        argv = list(sys.argv[1:] if argv is None else argv)
        if argv and argv[0] in ("inspect", "solve"):
            from infinicoder.app.solve_cli import main as solve_main

            return solve_main(argv)
        if not any(arg == "--context-limit" or arg.startswith("--context-limit=") for arg in argv):
            argv += ["--context-limit", str(RuntimeConfig().context_limit)]
        return runtime_main(argv, bind_config=True)
    except (ValueError, OSError, ImportError) as exc:
        print(
            json.dumps(
                {"status": "error", "error_type": type(exc).__name__, "reason": str(exc)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    except KeyboardInterrupt:
        return 130
