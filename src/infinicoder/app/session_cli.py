"""Read session status/events/diffs and request cooperative pause/cancel."""

import argparse
import json
from infinicoder.session.session import SessionStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=["status", "events", "diff", "pause", "cancel", "migrate"]
    )
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--after", type=int, default=0)
    args = parser.parse_args()
    store = SessionStore(args.state_dir)
    if args.action in ("pause", "cancel"):
        store.request_control(args.action)
        value = {"requested": args.action, "effective": "at next model/action boundary"}
    elif args.action == "migrate":
        value = store.migrate()
    elif args.action == "diff":
        print(store.diff())
        return
    elif args.action == "events":
        value = store.events(args.after)
    else:
        value = store.load()
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
