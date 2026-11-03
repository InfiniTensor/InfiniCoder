"""Bounded exact counting of whole source packets; bytes only select trials."""

import json


def bounded_prefix(records, packet, count, limit, *, count_limit=4):
    """Return an exactly measured complete prefix, with at most four counts.

    Empty metadata is measured first. A byte heuristic only picks a trial; it
    never authorizes delivery. If refinement runs out, the measured smaller
    packet is returned rather than counting every added record.
    """
    if type(count_limit) is not int or not 2 <= count_limit <= 4:
        raise ValueError("two to four whole-packet counts required")
    empty = packet([], 0)
    if count(json.dumps(empty, ensure_ascii=False, sort_keys=True, separators=(",", ":"))) > limit:
        return None
    best = empty
    size = len(json.dumps(empty, ensure_ascii=False).encode())
    chosen = 0
    for record in records:
        cost = len(json.dumps(record, ensure_ascii=False).encode())
        if chosen and size + cost > limit * 3:
            break
        size += cost
        chosen += 1
    if not chosen:
        return best
    low, high = 0, len(records) + 1
    for _ in range(count_limit - 1):
        candidate = packet(records[:chosen], chosen)
        tokens = count(
            json.dumps(candidate, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        if tokens <= limit:
            best = candidate
            low = chosen
            if chosen == len(records):
                break
            chosen = (chosen + high) // 2
        else:
            high = chosen
            chosen = (low + chosen) // 2
        if chosen <= low or chosen >= high:
            break
    return best
