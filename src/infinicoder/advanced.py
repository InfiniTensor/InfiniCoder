"""Compatibility access to advanced adapters and experimental capabilities.

These names retain their existing contracts. Their presence is not a claim
that every combination is supported or that it improves model quality.
"""

from importlib import import_module


def __getattr__(name):
    package = import_module("infinicoder")
    if name not in package._EXPORTS:
        raise AttributeError(name)
    return getattr(package, name)


def __dir__():
    return sorted(import_module("infinicoder")._EXPORTS)
