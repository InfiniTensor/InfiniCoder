"""Opt-in, bounded Python call observations for independent checkers.

Observations explain a failure; they never set acceptance or expected values.
Install only inside an isolated checker process. No oracle-specific function names.
"""

import functools
import inspect
from pathlib import Path
import sys
import builtins
import threading
from contextlib import contextmanager


def describe(value, depth=0):
    kind = type(value).__name__
    if value is None or isinstance(value, (bool, int, float)):
        return {"type": kind, "value": value}
    if isinstance(value, str):
        return {"type": kind, "value": value[:300], "length": len(value)}
    if depth < 2 and isinstance(value, dict):
        return {
            "type": kind,
            "length": len(value),
            "items": {str(k)[:100]: describe(v, depth + 1) for k, v in list(value.items())[:12]},
        }
    if depth < 2 and isinstance(value, (list, tuple)):
        return {
            "type": kind,
            "length": len(value),
            "items": [describe(v, depth + 1) for v in value[:8]],
        }
    return {"type": kind}


class PythonObservationRecorder:
    def __init__(self, line_limit=100000):
        if isinstance(line_limit, bool) or not isinstance(line_limit, int) or line_limit < 1:
            raise ValueError("line_limit must be a positive int")
        self.line_limit, self.last = line_limit, None

    @contextmanager
    def imports(self, workspace):
        """Observe normally imported workspace modules in a single checker process."""
        root, seen = Path(workspace).resolve(), set()
        original = builtins.__import__
        guard = threading.local()

        def observe_import(*args, **kwargs):
            if getattr(guard, "active", False):
                return original(*args, **kwargs)
            guard.active = True
            try:
                result = original(*args, **kwargs)
                for module in list(sys.modules.values()):
                    filename = getattr(module, "__file__", None)
                    if not filename or id(module) in seen:
                        continue
                    if getattr(getattr(module, "__spec__", None), "_initializing", False):
                        continue
                    seen.add(id(module))
                    path = Path(filename).resolve()
                    if path.suffix == ".py" and path.is_relative_to(root):
                        self.instrument(module, path)
                return result
            finally:
                guard.active = False

        builtins.__import__ = observe_import
        try:
            yield
        finally:
            builtins.__import__ = original

    def instrument(self, module, source_path):
        source_path = str(Path(source_path).resolve())
        for name, fn in list(vars(module).items()):
            if inspect.isfunction(fn) and fn.__module__ == module.__name__:
                setattr(module, name, self.wrap(fn, source_path))
        return module

    def wrap(self, fn, source_path):
        @functools.wraps(fn)
        def invoke(*args, **kwargs):
            observation = {
                "function": fn.__name__,
                "arguments": describe(args),
                "keywords": describe(kwargs),
            }
            try:
                observation["parameters"] = {
                    k: describe(v)
                    for k, v in inspect.signature(fn).bind(*args, **kwargs).arguments.items()
                }
            except (ValueError, TypeError):
                pass
            count, prior = 0, sys.gettrace()

            def trace(frame, event, arg):
                nonlocal count
                if event == "line" and frame.f_code.co_filename == source_path:
                    count += 1
                    if count > self.line_limit:
                        raise RuntimeError(
                            "candidate exceeded Python line-event budget; check loop progress"
                        )
                return trace

            try:
                sys.settrace(trace)
                result = fn(*args, **kwargs)
                observation["returned"] = describe(result)
                return result
            except Exception as error:
                observation["exception"] = {
                    "type": type(error).__name__,
                    "message": str(error)[:500],
                }
                tb = error.__traceback__
                while tb:
                    if tb.tb_frame.f_code.co_filename == source_path:
                        observation["location"] = {
                            "path": Path(source_path).name,
                            "line": tb.tb_lineno,
                            "locals": {
                                k: describe(v)
                                for k, v in sorted(
                                    tb.tb_frame.f_locals.items(),
                                    key=lambda pair: isinstance(pair[1], (list, tuple, dict)),
                                )[:20]
                            },
                        }
                    tb = tb.tb_next
                raise
            finally:
                sys.settrace(prior)
                observation["line_events"] = count
                self.last = observation

        return invoke


def render_observation(observation):
    """Flatten typed values for small-model consumption without deriving a fix."""
    import json

    lines = ["Observed call: " + str(observation.get("function", "?"))]

    def emit(path, value):
        if not isinstance(value, dict):
            return
        suffix = " = " + json.dumps(value["value"], ensure_ascii=False) if "value" in value else ""
        if "length" in value:
            suffix += " (length " + str(value["length"]) + ")"
        lines.append(path + ": " + str(value.get("type", "?")) + suffix)
        if isinstance(value.get("items"), dict):
            for key, child in value["items"].items():
                emit(path + "." + key, child)

    for name, value in observation.get("parameters", {}).items():
        emit(name, value)
    if observation.get("returned"):
        emit("RETURN", observation["returned"])
    if observation.get("exception"):
        lines.append("EXCEPTION: " + json.dumps(observation["exception"]))
    location = observation.get("location") or {}
    if location:
        lines.append("LOCATION: " + str(location.get("path")) + ":" + str(location.get("line")))
        for name, value in location.get("locals", {}).items():
            emit("local." + name, value)
    return "\n".join(lines)[:3500]


def compact_observation(observation):
    """Keep type/size/scalar evidence before shedding large example buffers."""

    def compact(value):
        if not isinstance(value, dict):
            return value
        result = {k: value[k] for k in ("type", "value", "length") if k in value}
        if isinstance(value.get("items"), dict):
            result["items"] = {k: compact(v) for k, v in value["items"].items()}
        return result

    result = {
        k: observation[k] for k in ("function", "exception", "line_events") if k in observation
    }
    result["parameters"] = {k: compact(v) for k, v in observation.get("parameters", {}).items()}
    if observation.get("returned"):
        result["returned"] = compact(observation["returned"])
    if observation.get("location"):
        location = observation["location"]
        result["location"] = {k: location[k] for k in ("path", "line") if k in location}
        result["location"]["locals"] = {
            k: compact(v) for k, v in location.get("locals", {}).items()
        }
    return result
