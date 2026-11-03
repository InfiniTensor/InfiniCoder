"""Caller-owned save/load/continue observations, consumed by ordinary Check.

Run inside the caller's existing isolated public checker. This module chooses
no device, imports no candidate and repairs no source. A finite equivalence
observation is neither an independent final nor a performance measurement.
"""

from infinicoder._resources import package_root

from dataclasses import dataclass
import copy
import hashlib
import json
import math
import re

from infinicoder.verification.public_transition_probe import (
    capture_value,
    semantic_values,
    ObservationUnavailable,
)
from infinicoder.verification.reference_compare import difference, safe_path


@dataclass(frozen=True)
class CheckpointProcedureContract:
    source_versions: dict
    components: tuple
    public_origin: str
    atol: float = 0.0
    rtol: float = 0.0

    def __post_init__(self):
        versions = self.source_versions
        if not isinstance(versions, dict) or not 1 <= len(versions) <= 16:
            raise ValueError("one to16 explicit source versions required")
        for name, pin in versions.items():
            safe_path(name)
            if not isinstance(pin, str) or not re.fullmatch("[0-9a-f]{64}", pin):
                raise ValueError("exact source SHA256 required")
        if not isinstance(self.components, (list, tuple)):
            raise ValueError("state components must be a list or tuple")
        components = tuple(self.components)
        if (
            not 1 <= len(components) <= 8
            or len(set(components)) != len(components)
            or any(
                not isinstance(x, str) or not re.fullmatch("[a-z][a-z0-9_]{0,47}", x)
                for x in components
            )
        ):
            raise ValueError("explicit distinct state components required")
        if not isinstance(self.public_origin, str) or not 1 <= len(self.public_origin) <= 1024:
            raise ValueError("public case provenance required")
        for value in (self.atol, self.rtol):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("finite nonnegative tolerances required")
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "source_versions", copy.deepcopy(versions))
        object.__setattr__(self, "_frozen", self._encoded())

    def _encoded(self):
        return json.dumps(
            dict(
                source_versions=self.source_versions,
                components=self.components,
                public_origin=self.public_origin,
                atol=self.atol,
                rtol=self.rtol,
                implementation_identity=identity(),
            ),
            sort_keys=True,
            ensure_ascii=False,
        )

    @property
    def identity(self):
        if self._encoded() != self._frozen:
            raise ValueError("checkpoint procedure contract mutated")
        return hashlib.sha256(self._frozen.encode()).hexdigest()


@dataclass
class CheckpointEndpoint:
    step: object
    save: object
    load: object
    observe: object
    capture_binding: object

    def __post_init__(self):
        if not all(
            callable(getattr(self, name))
            for name in ("step", "save", "load", "observe", "capture_binding")
        ):
            raise ValueError("actual step/save/load/state and caller binding hooks required")


def _literal_keys(value, depth=0):
    # Optimizer state dictionaries use integer parameter IDs. Prefix both
    # string and integer keys so 0 and "0" can never be conflated.
    if depth > 8:
        raise ObservationUnavailable("checkpoint state nesting exceeds8")
    if isinstance(value, dict):
        if len(value) > 64 or any(type(k) not in (str, int) for k in value):
            raise ObservationUnavailable("bounded string/integer state keys required")
        return {
            ("s:" + k if type(k) is str else "i:" + str(k)): _literal_keys(v, depth + 1)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        if len(value) > 64:
            raise ObservationUnavailable("checkpoint state sequence exceeds64")
        items = [_literal_keys(v, depth + 1) for v in value]
        return tuple(items) if isinstance(value, tuple) else items
    return value


def _state_difference(expected, actual, atol, rtol, path):
    # Cursor/rank/scheduler integers are discrete even when floating tensors
    # have an approximate comparison policy. Relative tolerance must not hide
    # a one-step error at a large training/data position.
    def discrete(left, right, where):
        if type(left) in (int, bool):
            if type(right) is not type(left) or left != right:
                return dict(path=where, reference=left, current=right)
        elif isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) & set(right)):
                found = discrete(left[key], right[key], where + "." + key)
                if found:
                    return found
        elif isinstance(left, list) and isinstance(right, list):
            for index, (a, b) in enumerate(zip(left, right)):
                found = discrete(a, b, where + "[" + str(index) + "]")
                if found:
                    return found
        return None

    return discrete(expected, actual, path) or difference(expected, actual, atol, rtol, path=path)


def checkpoint_equivalence(contract, cases, endpoint_factory, torch):
    """Compare split state and every resumed step with uninterrupted execution.

    cases: [{id, prefix:[{args,kwargs}], suffix:[{args,kwargs}]}]. At most two
    public cases and eight total step invocations; each phase has1..3 steps.
    The caller bounds process time and captures independently instrumented
    step/save/load counts and actual loaded source versions for each instance.
    save() must produce a stable checkpoint, as a persisted checkpoint would.
    """
    if not isinstance(contract, CheckpointProcedureContract) or not callable(endpoint_factory):
        raise ValueError("explicit contract and caller endpoint factory required")
    identity = contract.identity
    report = dict(
        kind="checkpoint_equivalence_v1",
        contract_sha256=identity,
        public_origin=contract.public_origin,
        source_versions=copy.deepcopy(contract.source_versions),
        components=list(contract.components),
        authority="bounded public observation;not independent final acceptance",
        step_calls=0,
        calls=[],
        scope="declared states and cases only;determinism and isolation supplied by caller",
    )
    stage = "setup"

    def verdict(status, **detail):
        return dict(report, status=status, stage=stage, **detail)

    def binding(endpoint):
        value = endpoint.capture_binding()
        if (
            not isinstance(value, dict)
            or value.get("source_versions") != contract.source_versions
            or not isinstance(value.get("instance_id"), str)
            or not value["instance_id"]
            or set(value.get("calls", {})) != {"step", "save", "load"}
            or any(type(n) is not int or n < 0 for n in value["calls"].values())
        ):
            raise ObservationUnavailable("actual instance/source/call binding unavailable")
        return copy.deepcopy(value)

    def invoke(endpoint, operation, *args, **kwargs):
        before = binding(endpoint)
        expected = dict(before["calls"])
        expected[operation] += 1
        if operation == "step":
            if report["step_calls"] >= 8:
                raise ObservationUnavailable("checkpoint step budget exhausted")
            report["step_calls"] += 1
        try:
            return getattr(endpoint, operation)(*args, **kwargs)
        finally:
            after = binding(endpoint)
            if after["instance_id"] != before["instance_id"] or after["calls"] != expected:
                raise ObservationUnavailable("declared operation not called exactly once")
            report["calls"].append(
                dict(instance_id=after["instance_id"], operation=operation, calls=after["calls"])
            )

    def capture(value):
        return capture_value(
            _literal_keys(value), torch, max_tensor_elements=8192, max_values=65536
        )

    def step(endpoint, arguments):
        value = copy.deepcopy(arguments)
        return invoke(endpoint, "step", *value["args"], **value["kwargs"])

    def state(endpoint):
        observed = endpoint.observe()
        if not isinstance(observed, dict) or set(observed) != set(contract.components):
            raise ObservationUnavailable("state coverage differs from declared components")
        return {name: capture(observed[name]) for name in contract.components}

    def mismatch(expected, actual):
        for component in contract.components:
            found = _state_difference(
                semantic_values(expected[component]),
                semantic_values(actual[component]),
                contract.atol,
                contract.rtol,
                component,
            )
            if found:
                return dict(component=component, difference=found)
        return None

    try:
        if not isinstance(cases, list) or not 1 <= len(cases) <= 2:
            raise ObservationUnavailable("one or two public cases required")
        ids = set()
        total = 0
        for case in cases:
            if (
                not isinstance(case, dict)
                or set(case) != {"id", "prefix", "suffix"}
                or not isinstance(case["id"], str)
                or not case["id"]
                or case["id"] in ids
            ):
                raise ObservationUnavailable(
                    "distinct public case IDs and explicit prefix/suffix required"
                )
            ids.add(case["id"])
            for phase in ("prefix", "suffix"):
                if not isinstance(case[phase], list) or not 1 <= len(case[phase]) <= 3:
                    raise ObservationUnavailable("one to3 steps per phase required")
                for arguments in case[phase]:
                    if (
                        not isinstance(arguments, dict)
                        or set(arguments) != {"args", "kwargs"}
                        or not isinstance(arguments["args"], (list, tuple))
                        or not isinstance(arguments["kwargs"], dict)
                        or any(not isinstance(k, str) for k in arguments["kwargs"])
                    ):
                        raise ObservationUnavailable("explicit step args/kwargs required")
                    capture(arguments)
            total += len(case["prefix"]) + 2 * len(case["suffix"])
        if total > 8:
            raise ObservationUnavailable("declared cases exceed eight total step calls")
        for case in cases:
            report["case_id"] = case["id"]
            stage = "initialize"
            continuous = endpoint_factory(case)
            if not isinstance(continuous, CheckpointEndpoint) or any(
                binding(continuous)["calls"].values()
            ):
                raise ObservationUnavailable("fresh continuous endpoint required")
            stage = "prefix"
            for arguments in case["prefix"]:
                step(continuous, arguments)
            stage = "save"
            saved = invoke(continuous, "save")
            split = state(continuous)
            stage = "uninterrupted"
            expected = []
            for arguments in case["suffix"]:
                output = step(continuous, arguments)
                expected.append((capture(output), state(continuous)))
            stage = "restore"
            resumed = endpoint_factory(case)
            if (
                not isinstance(resumed, CheckpointEndpoint)
                or any(binding(resumed)["calls"].values())
                or binding(resumed)["instance_id"] == binding(continuous)["instance_id"]
            ):
                raise ObservationUnavailable("a distinct fresh restored endpoint required")
            invoke(resumed, "load", saved)
            found = mismatch(split, state(resumed))
            if found:
                return verdict("counterexample", **found)
            for index, arguments in enumerate(case["suffix"]):
                stage = "continue"
                output = capture(step(resumed, arguments))
                found = _state_difference(
                    semantic_values(expected[index][0]),
                    semantic_values(output),
                    contract.atol,
                    contract.rtol,
                    "output",
                )
                if found:
                    return verdict(
                        "counterexample", component="output", step=index, difference=found
                    )
                found = mismatch(expected[index][1], state(resumed))
                if found:
                    return verdict("counterexample", step=index, **found)
        assert contract.identity == identity
        stage = "complete"
        return verdict("observed_values_match")
    except ObservationUnavailable as error:
        return verdict("unknown", reason=str(error)[:300])
    except Exception as error:
        return verdict(
            "candidate_error", exception=dict(type=type(error).__name__, message=str(error)[:400])
        )


def checkpoint_check_report(observation):
    """Ordinary Check JSON; feed its counterexample through existing feedback."""
    if not isinstance(observation, dict) or observation.get("kind") != "checkpoint_equivalence_v1":
        raise ValueError("actual checkpoint procedure observation required")
    ok = observation.get("status") == "observed_values_match"
    unknown = observation.get("status") == "unknown"
    detail = {key: value for key, value in observation.items() if key != "calls"}
    return dict(
        status="passed" if ok else "failed",
        passed=int(ok),
        failed=int(not ok and not unknown),
        error=int(unknown),
        case_detail=detail,
        acceptance_proof=False,
    )


@dataclass
class CheckpointRunEndpoint:
    """Original public framework runs, including lazy restoration boundaries.

    continuous() returns declared uninterrupted observations. interrupted()
    performs the original stop/save workflow and returns its checkpoint.
    resume(checkpoint) returns the same public observations after continuation.
    The caller binds common seeds,inputs and the exact comparison boundary.
    """

    continuous: object
    interrupted: object
    resume: object
    capture_binding: object

    def __post_init__(self):
        if not all(
            callable(getattr(self, name))
            for name in ("continuous", "interrupted", "resume", "capture_binding")
        ):
            raise ValueError("actual continuous/interrupted/resume runs and binding required")


def checkpoint_run_equivalence(contract, endpoint, torch):
    """Three real high-level runs;original task/process time limits still apply.

    Does not replace framework loops or assume eager restoration. Underlying
    training-step counts and resources remain those of the original public task.
    """
    if not isinstance(contract, CheckpointProcedureContract) or not isinstance(
        endpoint, CheckpointRunEndpoint
    ):
        raise ValueError("explicit contract and caller run endpoint required")
    pin = contract.identity
    stage = "setup"
    calls = []
    base = dict(
        kind="checkpoint_equivalence_v1",
        procedure="public_run_boundaries",
        contract_sha256=pin,
        public_origin=contract.public_origin,
        source_versions=copy.deepcopy(contract.source_versions),
        components=list(contract.components),
        authority="bounded public observation;not independent final acceptance",
        comparison_boundary="caller-declared public continuation endpoint",
        calls=calls,
    )

    def binding():
        value = endpoint.capture_binding()
        if (
            not isinstance(value, dict)
            or value.get("source_versions") != contract.source_versions
            or set(value.get("calls", {})) != {"continuous", "interrupted", "resume"}
            or any(type(x) is not int or x < 0 for x in value["calls"].values())
        ):
            raise ObservationUnavailable("actual public run source/call binding unavailable")
        return copy.deepcopy(value)

    def invoke(name, *args):
        before = binding()
        expected = dict(before["calls"])
        expected[name] += 1
        try:
            return getattr(endpoint, name)(*args)
        finally:
            after = binding()
            if after["calls"] != expected:
                raise ObservationUnavailable("declared public run not invoked exactly once")
            calls.append(dict(operation=name, calls=after["calls"]))

    def capture(value):
        if not isinstance(value, dict) or set(value) != set(contract.components):
            raise ObservationUnavailable("public run state coverage differs from declaration")
        return {
            key: capture_value(
                _literal_keys(value[key]), torch, max_tensor_elements=8192, max_values=65536
            )
            for key in contract.components
        }

    def verdict(status, **detail):
        return dict(base, status=status, stage=stage, run_calls=len(calls), **detail)

    try:
        if any(binding()["calls"].values()):
            raise ObservationUnavailable("fresh public run adapter required")
        stage = "continuous"
        expected = capture(invoke("continuous"))
        stage = "interrupted"
        saved = invoke("interrupted")
        stage = "resume"
        actual = capture(invoke("resume", saved))
        if contract.identity != pin:
            raise ObservationUnavailable("public run contract changed")
        for component in contract.components:
            found = _state_difference(
                semantic_values(expected[component]),
                semantic_values(actual[component]),
                contract.atol,
                contract.rtol,
                component,
            )
            if found:
                return verdict("counterexample", component=component, difference=found)
        stage = "complete"
        return verdict("observed_values_match")
    except ObservationUnavailable as error:
        return verdict("unknown", reason=str(error)[:300])
    except Exception as error:
        return verdict(
            "unknown" if stage == "continuous" else "candidate_error",
            exception=dict(type=type(error).__name__, message=str(error)[:400]),
        )


def identity():
    root = package_root()
    return hashlib.sha256(
        b"".join(
            (root / name).read_bytes()
            for name in (
                "verification/checkpoint_procedure.py",
                "verification/public_transition_probe.py",
                "verification/reference_compare.py",
            )
        )
    ).hexdigest()
