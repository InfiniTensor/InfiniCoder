"""Bounded independent exploration of a declared finite resource/state design.

Transition deltas and states are MODEL CLAIMS,not observed implementation facts.
Missing transitions are unknown,never implicitly treated as safe no-ops.
"""

from collections import deque
import re


def check(arguments, required_events=None):
    if (
        not isinstance(arguments, dict)
        or set(arguments) - {"initial", "initial_units", "terminals", "transitions", "depth"}
        or not {"initial", "terminals", "transitions"} <= set(arguments)
    ):
        raise ValueError(
            "lifecycle design requires initial,terminals,transitions and optional initial_units,depth"
        )

    def name(value):
        if not isinstance(value, str) or not re.fullmatch("[A-Za-z][A-Za-z0-9_]{0,31}", value):
            raise ValueError("bounded state/event identifier required")
        return value

    initial = name(arguments["initial"])
    terminals = arguments["terminals"]
    transitions = arguments["transitions"]
    depth = arguments.get("depth", 6)
    if not isinstance(terminals, list) or not 1 <= len(terminals) <= 16:
        raise ValueError("1..16 unique terminal states required")
    checked = [name(v) for v in terminals]
    if len(set(checked)) != len(checked):
        raise ValueError("unique terminal states required")
    terminals = set(checked)
    initial_units = arguments.get("initial_units", 0)
    if type(initial_units) is not int or not 0 <= initial_units <= 8:
        raise ValueError("initial_units must be integer in[0,8]")
    if (
        not isinstance(transitions, list)
        or not 1 <= len(transitions) <= 64
        or type(depth) is not int
        or not 1 <= depth <= 12
    ):
        raise ValueError("1..64 transitions and depth1..12 required")
    outgoing = {}
    states = {initial, *terminals}
    seen_edges = set()
    for item in transitions:
        if not isinstance(item, dict) or set(item) != {"from", "event", "to", "delta"}:
            raise ValueError("transition needs from,event,to,delta")
        source, event, target = name(item["from"]), name(item["event"]), name(item["to"])
        delta = item["delta"]
        if type(delta) is not int or not -8 <= delta <= 8:
            raise ValueError("resource delta must be integer in[-8,8]")
        key = (source, event)
        if key in seen_edges:
            raise ValueError("deterministic finite design needs one transition per state/event")
        seen_edges.add(key)
        states.update((source, target))
        outgoing.setdefault(source, []).append(dict(item))
    if len(states) > 16:
        raise ValueError("at most16 states supported")
    required_events = required_events or {}
    if (
        not isinstance(required_events, dict)
        or len(required_events) > 16
        or any(not isinstance(v, list) or len(v) > 32 for v in required_events.values())
    ):
        raise ValueError("caller required_events must map states to bounded event lists")
    for state, events in required_events.items():
        name(state)
        [name(v) for v in events]
    base = dict(
        status="bounded_design_checked",
        scope="declared resource model only;not code execution or concurrency proof",
        max_depth=depth,
        initial_units=initial_units,
        limitations=[
            "Initial units and transition deltas are design assumptions;implementation must be independently instrumented/tested.",
            "Only declared transitions are explored;missing events are unknown,no implicit no-op. Interleavings beyond the explicit model and depth are unverified.",
        ],
    )
    queue = deque([(initial, initial_units, [])])
    seen = set()
    terminal_states = set()
    reached = set()
    missing = {}
    cutoff = False
    explored = 0
    while queue:
        state, held, path = queue.popleft()
        key = (state, held, len(path))
        if key in seen:
            continue
        seen.add(key)
        reached.add(state)
        explored += 1
        if explored > 4096:
            return dict(
                base,
                status="inconclusive",
                reason="state budget exceeded;no partial pass",
                explored=explored,
            )
        if held < 0:
            return dict(
                base,
                status="counterexample",
                counterexample=dict(
                    kind="release_without_ownership", state=state, held=held, events=path
                ),
                explored=explored,
            )
        if state in terminals:
            terminal_states.add(state)
            if held:
                return dict(
                    base,
                    status="counterexample",
                    counterexample=dict(
                        kind="terminal_resource_leak", state=state, held=held, events=path
                    ),
                    explored=explored,
                )
        available = {r["event"] for r in outgoing.get(state, [])}
        absent = set(required_events.get(state, [])) - available
        if absent:
            missing[state] = sorted(absent)
        if len(path) == depth:
            if outgoing.get(state):
                cutoff = True
            continue
        for item in outgoing.get(state, []):
            queue.append((item["to"], held + item["delta"], path + [item["event"]]))
            if len(queue) > 8192:
                return dict(
                    base,
                    status="inconclusive",
                    reason="pending-state budget exceeded;no partial pass",
                    explored=explored,
                )
    unknown = set(required_events) - reached
    if unknown:
        return dict(
            base,
            status="inconclusive",
            reason="caller-required states not reached",
            unreached_required_states=sorted(unknown),
            explored=explored,
        )
    if missing:
        return dict(
            base,
            status="inconclusive",
            reason="caller-required events lack explicit transitions",
            missing_events=missing,
            explored=explored,
        )
    if not terminal_states:
        return dict(
            base,
            status="inconclusive",
            reason="no terminal reached within bound",
            explored=explored,
        )
    unreachable = terminals - terminal_states
    if unreachable:
        return dict(
            base,
            status="inconclusive",
            reason="some declared terminal paths untested",
            unreached_terminals=sorted(unreachable),
            explored=explored,
        )
    return dict(
        base,
        explored=explored,
        terminal_states_checked=sorted(terminal_states),
        reachable_states=sorted(reached),
        depth_cutoff=cutoff,
        complete_implementation_coverage=False,
    )
