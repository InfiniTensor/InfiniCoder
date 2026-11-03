"""Bounded public tensor/state experiments,never final acceptance.

This module is used inside a caller's isolated diagnostic runner. It does not
import a candidate,choose a private oracle,grant a device,or execute a shell.
Adapters bind actual callables and public inputs;their reports are compared by
the caller outside candidate stdout. Source and runtime isolation remain the
caller's explicit responsibility.
"""

from dataclasses import dataclass, field
import copy
import hashlib
import json
import math
import re

from infinicoder.verification.reference_compare import difference, safe_path


def encoded(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")
    )


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


class ObservationUnavailable(ValueError):
    pass


@dataclass(frozen=True)
class PublicTransitionContract:
    public_reference_identity: dict
    candidate_entry_binding: dict
    public_input_schema: dict
    state_schema: dict
    case_recipe_id: str
    comparison_policy: dict
    execution_operation: dict
    coverage: dict
    observed_paths: tuple
    budget: dict = field(
        default_factory=lambda: dict(
            max_runs=2, seconds_per_run=30, total_seconds=60, max_actual_calls=8, max_state_steps=3
        )
    )
    _original: str = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        for name in (
            "public_reference_identity",
            "candidate_entry_binding",
            "public_input_schema",
            "state_schema",
            "comparison_policy",
            "execution_operation",
            "coverage",
            "budget",
        ):
            value = getattr(self, name)
            if not isinstance(value, dict):
                raise ValueError("explicit data mapping required: " + name)
            object.__setattr__(self, name, copy.deepcopy(value))
        object.__setattr__(self, "observed_paths", tuple(self.observed_paths))
        reference = self.public_reference_identity
        binding = self.candidate_entry_binding
        if (
            reference.get("public") is not True
            or not reference.get("id")
            or not reference.get("provenance")
        ):
            raise ValueError("caller-authorized public reference identity and provenance required")
        if not re.fullmatch("[0-9a-f]{64}", reference.get("artifact_sha256", "")):
            raise ValueError("pinned public reference artifact required")
        if set(binding) != {"path", "qualname", "adapter_id"}:
            raise ValueError("exact candidate path,qualname and adapter_id required")
        safe_path(binding["path"])
        if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*){0,3}", binding["qualname"]):
            raise ValueError("bounded explicit candidate callable required")
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,79}", binding["adapter_id"]):
            raise ValueError("identified public adapter required")
        if not 1 <= len(self.observed_paths) <= 64 or len(set(self.observed_paths)) != len(
            self.observed_paths
        ):
            raise ValueError("one to64 distinct observed paths required")
        for path in self.observed_paths:
            safe_path(path)
        if binding["path"] not in self.observed_paths:
            raise ValueError("candidate entry must be in the observed source set")
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,79}", self.case_recipe_id):
            raise ValueError("identified bounded recipe required")
        if not self.public_input_schema or not self.coverage or not self.execution_operation:
            raise ValueError(
                "explicit public input domain,coverage and execution identity required"
            )
        for name in ("atol", "rtol"):
            value = self.comparison_policy.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("finite nonnegative " + name + " required")
        if (
            set(self.comparison_policy) - {"atol", "rtol", "compare_output_layout"}
            or type(self.comparison_policy.get("compare_output_layout", False)) is not bool
        ):
            raise ValueError("explicit supported numerical/layout comparison policy required")
        if self.budget != dict(
            max_runs=2, seconds_per_run=30, total_seconds=60, max_actual_calls=8, max_state_steps=3
        ):
            raise ValueError("first-version probe limits are fixed;new limits need a new contract")
        raw = encoded(self._data())
        if len(raw.encode()) > 32768:
            raise ValueError("transition contract exceeds32KiB")
        object.__setattr__(self, "_original", raw)

    def _data(self):
        return {
            name: copy.deepcopy(getattr(self, name))
            for name in (
                "public_reference_identity",
                "candidate_entry_binding",
                "public_input_schema",
                "state_schema",
                "case_recipe_id",
                "comparison_policy",
                "execution_operation",
                "coverage",
                "observed_paths",
                "budget",
            )
        }

    def manifest(self):
        raw = encoded(self._data())
        if raw != self._original:
            raise ValueError("public transition contract mutated")
        return json.loads(raw)

    @property
    def identity(self):
        return digest(self.manifest())


def capture_value(value, torch, *, max_tensor_elements=256, max_values=2048):
    """Capture actual logical values;layout metadata is observed separately.

    The runner has already selected its allowed device. No device inventory or
    model query occurs here. Sparse/quantized and nonfinite values are explicit
    unsupported observations rather than approximate passing comparisons.
    """
    remaining = [max_values]

    def capture(item, depth):
        if depth > 8:
            raise ObservationUnavailable("nested observation depth exceeds8")
        if torch.is_tensor(item):
            if item.layout != torch.strided or item.is_quantized:
                raise ObservationUnavailable("sparse or quantized tensor snapshot unsupported")
            count = item.numel()
            if count > max_tensor_elements or count > remaining[0]:
                raise ObservationUnavailable("tensor observation element bound")
            remaining[0] -= count
            data = item.detach().cpu().reshape(-1)
            values = (
                [[float(x.real), float(x.imag)] for x in data.tolist()]
                if item.is_complex()
                else data.tolist()
            )
            if any(
                not math.isfinite(float(x))
                for row in values
                for x in (row if isinstance(row, list) else [row])
            ):
                raise ObservationUnavailable(
                    "nonfinite public-reference observations require an explicit future policy"
                )
            return dict(
                kind="tensor",
                dtype=str(item.dtype),
                shape=list(item.shape),
                values=values,
                element_strides=list(item.stride()),
                storage_offset=item.storage_offset(),
                device=str(item.device),
                complex_values=item.is_complex(),
            )
        if type(item) in (int, float, bool) or item is None or isinstance(item, str):
            remaining[0] -= 1
            if remaining[0] < 0 or isinstance(item, str) and len(item) > 512:
                raise ObservationUnavailable("scalar observation bound")
            if type(item) is float and not math.isfinite(item):
                raise ObservationUnavailable("nonfinite scalar")
            return item
        if isinstance(item, (tuple, list)):
            if len(item) > 64:
                raise ObservationUnavailable("sequence observation bound")
            return dict(
                kind="tuple" if isinstance(item, tuple) else "list",
                items=[capture(x, depth + 1) for x in item],
            )
        if (
            isinstance(item, dict)
            and len(item) <= 64
            and all(isinstance(k, str) and len(k) <= 128 for k in item)
        ):
            return {k: capture(v, depth + 1) for k, v in sorted(item.items())}
        raise ObservationUnavailable("unsupported observation type: " + type(item).__name__)

    return capture(value, 0)


def semantic_values(value, *, compare_layout=False):
    if isinstance(value, dict):
        if value.get("kind") == "tensor":
            keys = ("kind", "dtype", "shape", "values", "complex_values")
            if compare_layout:
                keys += ("element_strides", "storage_offset")
            return {k: copy.deepcopy(value[k]) for k in keys}
        return {k: semantic_values(v, compare_layout=compare_layout) for k, v in value.items()}
    if isinstance(value, list):
        return [semantic_values(v, compare_layout=compare_layout) for v in value]
    return value


@dataclass
class TransitionEndpoint:
    """Adapter-issued actual callable and capture hook,never model metadata.

    capture_binding() must expose an independent target-call count and the
    currently loaded source binding. For a framework public step wrapper this
    is the selected internal implementation hook,not merely Optimizer.step.
    The adapter is trusted application code and is qualified with wrong-entry
    controls before use;this is not a hostile-Python isolation guarantee.
    """

    invoke: object
    read_state: object
    capture_binding: object

    def __post_init__(self):
        if not all(callable(f) for f in (self.invoke, self.read_state, self.capture_binding)):
            raise ValueError(
                "actual invocation,state reader and independent callable capture required"
            )


def _input_bindings(value, torch):
    if isinstance(value, torch.Tensor):
        return (
            "tensor",
            value,
            value.untyped_storage(),
            (
                tuple(value.shape),
                tuple(value.stride()),
                value.storage_offset(),
                str(value.dtype),
                str(value.device),
                value.requires_grad,
                value.data_ptr(),
                value._version,
                value.untyped_storage().nbytes(),
                value.untyped_storage()._cdata,
            ),
        )
    if isinstance(value, dict):
        return ("dict", {k: _input_bindings(v, torch) for k, v in value.items()})
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, [_input_bindings(v, torch) for v in value])
    return ("scalar", type(value), value)


def _input_binding_change(value, old, torch, path="input"):
    kind = old[0]
    if kind == "tensor":
        if not isinstance(value, torch.Tensor) or value is not old[1]:
            return path + ".tensor_object_replaced"
        current = (
            tuple(value.shape),
            tuple(value.stride()),
            value.storage_offset(),
            str(value.dtype),
            str(value.device),
            value.requires_grad,
            value.data_ptr(),
            value._version,
            value.untyped_storage().nbytes(),
            value.untyped_storage()._cdata,
        )
        if current != old[3]:
            return path + ".tensor_metadata_or_version_changed"
        return None
    if kind == "dict":
        if not isinstance(value, dict) or set(value) != set(old[1]):
            return path + ".dictionary_keys_changed"
        pairs = ((v, old[1][k], path + "." + str(k)) for k, v in value.items())
    elif kind in ("tuple", "list"):
        if type(value).__name__ != kind or len(value) != len(old[1]):
            return path + ".sequence_changed"
        pairs = ((v, o, path + "[" + str(i) + "]") for i, (v, o) in enumerate(zip(value, old[1])))
    else:
        return None if type(value) is old[1] and value == old[2] else path + ".scalar_changed"
    for current, binding, where in pairs:
        changed = _input_binding_change(current, binding, torch, where)
        if changed:
            return changed
    return None


def run_role(contract, role, cases, endpoint_factory, torch):
    """Run finite actual calls in the caller's isolated candidate/ref process.

    cases and endpoint_factory are frozen adapter code,not a model-authored
    expected-value program. At most4 calls per role means<=8 paired calls.
    Each case owns fresh tensors/state;the recorded inputs catch role drift.
    """
    if not isinstance(contract, PublicTransitionContract) or role not in ("reference", "candidate"):
        raise ValueError("identified contract and explicit execution role required")
    data = contract.manifest()
    rows = []
    bindings = []
    result = dict(
        contract_sha256=contract.identity,
        role=role,
        calls=rows,
        bindings=bindings,
        setup_error=None,
        environment_error=None,
        observation_error=None,
        coverage=data["coverage"],
    )
    try:
        if not isinstance(cases, list) or not 1 <= len(cases) <= 4:
            raise ObservationUnavailable("one to4 bounded public cases required")
        for case in cases:
            if set(case) != {"case_id", "steps", "initial_state"} or not isinstance(
                case["case_id"], str
            ):
                raise ObservationUnavailable("explicit case_id,steps,initial_state required")
            steps = case["steps"]
            if not isinstance(steps, list) or not 1 <= len(steps) <= 3:
                raise ObservationUnavailable("one to3 state steps required")
            endpoint = endpoint_factory(case)
            if not isinstance(endpoint, TransitionEndpoint):
                raise TypeError("adapter must return TransitionEndpoint")
            initial = capture_value(endpoint.read_state(), torch)
            retained = []
            for step, arguments in enumerate(steps):
                if len(rows) >= 4:
                    raise ObservationUnavailable("paired call budget exceeds8 actual target calls")
                if set(arguments) != {"args", "kwargs"}:
                    raise ObservationUnavailable("explicit args/kwargs required")
                before = capture_value(arguments, torch)
                binding_before = copy.deepcopy(endpoint.capture_binding())
                input_binding = (
                    _input_bindings(arguments, torch)
                    if data["state_schema"].get("inputs_immutable") is True
                    else None
                )
                if step == 0 and binding_before.get("actual_target_calls") != 0:
                    raise ObservationUnavailable(
                        "adapter called the target before the controlled experiment"
                    )
                row = dict(
                    case_id=case["case_id"],
                    step=step,
                    initial_state=initial,
                    input_before=before,
                    state_before=capture_value(endpoint.read_state(), torch),
                    binding_before=binding_before,
                )
                rows.append(row)
                try:
                    output = endpoint.invoke(*arguments["args"], **arguments["kwargs"])
                    row["output"] = capture_value(output, torch)
                except ObservationUnavailable:
                    raise
                except Exception as error:
                    row["candidate_exception"] = dict(
                        type=type(error).__name__, message=str(error)[:400]
                    )
                if data["state_schema"].get("retain_returned_values_between_steps") is True:
                    row["retained_outputs_after"] = capture_value(retained, torch)
                    if not row.get("candidate_exception"):
                        retained.append(output)
                row.update(
                    input_after=capture_value(arguments, torch),
                    state_after=capture_value(endpoint.read_state(), torch),
                    binding_after=copy.deepcopy(endpoint.capture_binding()),
                )
                if input_binding is not None:
                    row["input_binding_violation"] = _input_binding_change(
                        arguments, input_binding, torch
                    )
                before_count = binding_before.get("actual_target_calls")
                after_count = row["binding_after"].get("actual_target_calls")
                if (
                    type(before_count) is not int
                    or type(after_count) is not int
                    or after_count - before_count != 1
                ):
                    raise ObservationUnavailable(
                        "actual declared target was not called exactly once"
                    )
                bindings.append(copy.deepcopy(row["binding_after"]))
                if row.get("candidate_exception"):
                    return result
    except ObservationUnavailable as error:
        result["observation_error"] = str(error)
    except Exception as error:
        result["setup_error"] = dict(type=type(error).__name__, message=str(error)[:400])
    return result


def compare_transition_reports(contract, reference, candidate, source_versions):
    """Decode trusted runner observations,never stdout PASSED or model claims."""
    data = contract.manifest()
    binding = data["candidate_entry_binding"]
    policy = data["comparison_policy"]
    base = dict(
        contract_sha256=contract.identity,
        coverage=data["coverage"],
        source_versions=copy.deepcopy(source_versions),
        authority="finite public diagnostic;not independent final acceptance or general equivalence",
    )

    def verdict(status, **detail):
        return dict(base, status=status, **detail)

    for role, report in (("reference", reference), ("candidate", candidate)):
        if (
            not isinstance(report, dict)
            or report.get("contract_sha256") != contract.identity
            or report.get("role") != role
        ):
            return verdict("unknown", reason="role_or_contract_binding_missing")
        if report.get("environment_error"):
            return verdict("env_error", role=role, reason=report["environment_error"])
        if report.get("setup_error") or report.get("observation_error"):
            return verdict(
                "unknown",
                role=role,
                reason="setup_or_observation_failure",
                detail=report.get("setup_error") or report.get("observation_error"),
            )
    left, right = reference.get("calls"), candidate.get("calls")
    if (
        not isinstance(left, list)
        or not isinstance(right, list)
        or not left
        or not right
        or len(left) + len(right) > 8
    ):
        return verdict("unknown", reason="missing_or_excess_actual_calls")
    base.update(
        actual_calls=len(left) + len(right), reference_calls=len(left), candidate_calls=len(right)
    )
    mandatory = {
        "case_id",
        "step",
        "initial_state",
        "input_before",
        "input_after",
        "state_before",
        "state_after",
        "binding_before",
        "binding_after",
    }
    retained = data["state_schema"].get("retain_returned_values_between_steps") is True
    if retained:
        mandatory.add("retained_outputs_after")
    immutable_inputs = data["state_schema"].get("inputs_immutable") is True
    if immutable_inputs:
        mandatory.add("input_binding_violation")
    for row in left + right:
        if (
            not isinstance(row, dict)
            or not mandatory.issubset(row)
            or not ("output" in row or "candidate_exception" in row)
        ):
            return verdict("unknown", reason="incomplete_call_capture")
        before = row["binding_before"].get("actual_target_calls")
        after = row["binding_after"].get("actual_target_calls")
        if type(before) is not int or type(after) is not int or after - before != 1:
            return verdict("unknown", reason="actual_target_call_capture_invalid")
    version = source_versions.get(binding["path"], {})
    if version.get("kind") != "file":
        return verdict("unknown", reason="current_candidate_source_unknown")
    for r in right:
        observed = r.get("binding_after", {})
        if (
            observed.get("path") != binding["path"]
            or observed.get("qualname") != binding["qualname"]
            or observed.get("source_sha256") != version["sha256"]
            or observed.get("role") != "candidate"
        ):
            return verdict("unknown", reason="candidate_loaded_wrong_source_or_entry")
    for r in left:
        observed = r.get("binding_after", {})
        if (
            observed.get("reference_artifact_sha256")
            != data["public_reference_identity"]["artifact_sha256"]
            or observed.get("role") != "reference"
        ):
            return verdict("unknown", reason="public_reference_binding_unproven")
    for i, (a, b) in enumerate(zip(left, right)):
        if a.get("candidate_exception"):
            return verdict(
                "env_error", reason="public_reference_failed", detail=a["candidate_exception"]
            )
        if (a.get("case_id"), a.get("step")) != (b.get("case_id"), b.get("step")) or encoded(
            a.get("input_before")
        ) != encoded(b.get("input_before")):
            return verdict("unknown", reason="role_input_or_sequence_mismatch", call=i)
        initial = (
            difference(
                semantic_values(a["state_before"]),
                semantic_values(b["state_before"]),
                policy["atol"],
                policy["rtol"],
            )
            if a["step"] == 0
            else None
        )
        if initial:
            return verdict("unknown", reason="initial_states_differ", call=i, difference=initial)
        detail = dict(
            call=i,
            case_id=a["case_id"],
            step=a["step"],
            input=a["input_before"],
            initial_state=a["initial_state"],
        )
        if b.get("candidate_exception"):
            return verdict("candidate_error", **detail, exception=b["candidate_exception"])
        if immutable_inputs and a.get("input_binding_violation"):
            return verdict(
                "unknown",
                reason="public_reference_changed_input_binding",
                detail=a["input_binding_violation"],
            )
        if immutable_inputs and b.get("input_binding_violation"):
            return verdict(
                "counterexample",
                **detail,
                difference=dict(
                    path=b["input_binding_violation"],
                    reference="input binding preserved",
                    current="input binding changed",
                ),
            )
        for component in ("output", "state_after", "input_after") + (
            ("retained_outputs_after",) if retained else ()
        ):
            found = difference(
                semantic_values(
                    a[component],
                    compare_layout=policy.get("compare_output_layout", False)
                    if component == "output"
                    else False,
                ),
                semantic_values(
                    b[component],
                    compare_layout=policy.get("compare_output_layout", False)
                    if component == "output"
                    else False,
                ),
                policy["atol"],
                policy["rtol"],
                path=component,
            )
            if found:
                expected, current = found["reference"], found["current"]
                if type(expected) in (int, float) and type(current) in (int, float):
                    found["absolute_error"] = abs(expected - current)
                    found["relative_error"] = (
                        abs(expected - current) / abs(expected) if expected else None
                    )
                return verdict("counterexample", **detail, difference=found)
    if len(left) != len(right):
        return verdict("unknown", reason="incomplete_role_execution")
    return verdict(
        "observed_values_match",
        executed_steps=len(left),
        meaning="Only the declared public cases matched;independent final and performance checks remain required",
    )
