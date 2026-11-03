"""Explicit teaching cards. They are not validated seeds or acceptance evidence."""

from infinicoder._resources import package_file


CARDS = {
    "triton": "triton.md",
    "framework": "framework.md",
    "python": "python.md",
    "native": "native.md",
    "operator": "operator.md",
    "system": "system.md",
    "optimizer": "optimizer.md",
}


def teaching_card(card):
    if card not in CARDS:
        raise ValueError("Unknown teaching card")
    return {
        "id": card,
        "evidence": False,
        "text": package_file("knowledge").joinpath(CARDS[card]).read_text(encoding="utf-8"),
    }
