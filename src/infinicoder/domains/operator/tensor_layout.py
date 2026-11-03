"""Rank-independent address obligations derived from caller-declared layouts.

Symbols remain text, never evaluated. This module knows no operator or task ids.
"""

from dataclasses import dataclass
import re


def dimension(value):
    if type(value) is int and value >= 0:
        return str(value)
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", value):
        return value
    raise ValueError("Layout dimensions/strides must be nonnegative ints or identifiers")


@dataclass(frozen=True)
class TensorLayout:
    shape: tuple
    strides: tuple

    def __post_init__(self):
        if (
            not isinstance(self.shape, (list, tuple))
            or not isinstance(self.strides, (list, tuple))
            or len(self.shape) != len(self.strides)
        ):
            raise ValueError("Layout requires equally ranked shape and strides")
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "strides", tuple(self.strides))
        if len(self.shape) > 16:
            raise ValueError("Layout rank exceeds bound")
        for value in (*self.shape, *self.strides):
            dimension(value)

    def minimum_span(self):
        if any(type(v) is not int for v in (*self.shape, *self.strides)):
            raise ValueError("Numerical span requires concrete dimensions and strides")
        return (
            0 if 0 in self.shape else 1 + sum((n - 1) * s for n, s in zip(self.shape, self.strides))
        )

    def obligations(self):
        shape, strides = [dimension(v) for v in self.shape], [dimension(v) for v in self.strides]
        indices = ["i" + str(i) for i in range(len(shape))]
        address = (
            " + ".join(i if s == "1" else i + " * " + s for i, s in zip(indices, strides)) or "0"
        )
        empty = " or ".join(n + " == 0" for n in shape)
        span = "1" + "".join(" + (" + n + " - 1) * " + s for n, s in zip(shape, strides))
        return {
            "indices": indices,
            "valid_indices": [f"0 <= {i} < {n}" for i, n in zip(indices, shape)],
            "element_address": address,
            "minimum_accessible_elements": "0 if " + empty + " else " + span if empty else "1",
            "rule": "Use THIS buffer address, not another buffer stride. Do not require trailing padding beyond the last logical element.",
        }


def derive_layouts(facts):
    result = {}
    tensors = facts.get("tensors", {})
    if not isinstance(tensors, dict) or len(tensors) > 16:
        raise ValueError("tensors must be a bounded layout mapping")
    for name, spec in tensors.items():
        if not isinstance(name, str) or not isinstance(spec, dict):
            raise ValueError("Invalid tensor declaration")
        layout = TensorLayout(spec.get("shape"), spec.get("element_strides"))
        result[name] = layout.obligations()
    return result
