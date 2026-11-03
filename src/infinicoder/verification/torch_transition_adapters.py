"""Public-library adapters for bounded unary tensors and Adam state steps.

Only an isolated diagnostic runner calls this module. Public references are
caller-frozen artifacts;candidate modules are never imported by a controller.
No benchmark grader,hidden case factory or reference patch is used.
"""

import ast
import builtins
import copy
import hashlib
import importlib
import importlib.util
import inspect
import os
from pathlib import Path
import sys

from infinicoder.verification.public_transition_probe import (
    ObservationUnavailable,
    TransitionEndpoint,
)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ObservationUnavailable("callable module loader unavailable")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return module


def _tensor(torch, values, device, stride=1, offset=0):
    source = torch.tensor(values, dtype=torch.float32, device=device)
    storage = torch.zeros(
        offset + max(0, len(values) - 1) * stride + 1, dtype=torch.float32, device=device
    )
    view = storage.as_strided((len(values),), (stride,), offset)
    view.copy_(source)
    return view


def _header(node):
    node = ast.FunctionDef(
        name=node.name,
        args=node.args,
        body=[ast.Pass()],
        decorator_list=node.decorator_list,
        returns=node.returns,
        type_comment=node.type_comment,
    )
    return ast.dump(node, include_attributes=False)


def _optimizer_namespace(node, module):
    """Retain the original task's bounded-function admission in public probes."""
    forbidden = {
        "open",
        "eval",
        "exec",
        "compile",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "__import__",
        "breakpoint",
        "save",
        "load",
        "from_file",
        "write",
        "read",
        "system",
        "popen",
        "fork",
        "spawn",
        "exit",
        "quit",
    }
    for item in ast.walk(node):
        if isinstance(item, (ast.Import, ast.ImportFrom)):
            modules = (
                [a.name for a in item.names]
                if isinstance(item, ast.Import)
                else [item.module or ""]
            )
            if any(
                n.split(".")[0]
                not in {"torch", "math", "typing", "collections", "functools", "contextlib"}
                for n in modules
            ):
                raise ObservationUnavailable("optimizer import outside bounded public executor")
            if any(a.name.split(".")[-1] in forbidden for a in item.names):
                raise ObservationUnavailable("optimizer file/process/reflection API")
        if isinstance(item, (ast.Global, ast.Nonlocal)):
            raise ObservationUnavailable("optimizer global mutation outside bounded executor")
        name = (
            item.id
            if isinstance(item, ast.Name)
            else item.attr
            if isinstance(item, ast.Attribute)
            else None
        )
        if name is not None and (name in forbidden or name.startswith("__")):
            raise ObservationUnavailable("optimizer file/process/reflection API")
    namespace = {
        k: v
        for k, v in module.__dict__.items()
        if k not in {"os", "sys", "inspect", "subprocess", "pickle", "importlib", "warnings"}
    }
    namespace["__builtins__"] = {
        k: v for k, v in builtins.__dict__.items() if k not in forbidden and not k.startswith("__")
    }
    namespace["__builtins__"]["__import__"] = builtins.__import__
    node = copy.deepcopy(node)
    node.decorator_list = []
    node.returns = None
    for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
        arg.annotation = None
    return node, namespace


def prepare_adapter(contract, payload, torch, role):
    """Return fresh controlled cases and an independently captured endpoint."""
    data = contract.manifest()
    binding = data["candidate_entry_binding"]
    schema = data["public_input_schema"]
    root = Path(payload["workspace"]).resolve()
    candidate = (root / binding["path"]).resolve()
    if not candidate.is_relative_to(root) or candidate.is_symlink() or not candidate.is_file():
        raise ObservationUnavailable("candidate entry is not a regular workspace source")
    expected = payload["source_versions"].get(binding["path"], {})
    if expected.get("kind") != "file" or file_sha(candidate) != expected["sha256"]:
        raise ObservationUnavailable("candidate source differs from the caller snapshot")
    reference = Path(payload["public_reference_path"]).resolve()
    if (
        reference.is_relative_to(root)
        or file_sha(reference) != data["public_reference_identity"]["artifact_sha256"]
    ):
        raise ObservationUnavailable("public reference artifact binding failed")
    if str(torch.__version__) != schema["torch_runtime"]["version"]:
        raise ObservationUnavailable("public Torch runtime version differs")
    device = schema.get("device", "cpu")
    if device not in ("cpu", "cuda:0"):
        raise ObservationUnavailable("only explicit CPU or one mapped CUDA device is supported")
    actual_device_uuid = None
    if device == "cuda:0":
        permitted = data["execution_operation"].get("cuda_visible_devices")
        if (
            not isinstance(permitted, str)
            or not permitted.startswith("GPU-")
            or os.environ.get("CUDA_VISIBLE_DEVICES") != permitted
        ):
            raise ObservationUnavailable("explicit caller CUDA UUID mapping unavailable")
        properties = torch.cuda.get_device_properties(torch.device("cuda:0"))
        value = str(properties.uuid)
        actual_device_uuid = value if value.startswith("GPU-") else "GPU-" + value
        if actual_device_uuid.lower() != permitted.lower():
            raise ObservationUnavailable("actual CUDA UUID differs from caller binding")
    artifact_sha = data["public_reference_identity"]["artifact_sha256"]

    def proof(count, target):
        actual = getattr(target, "__func__", target)
        code = getattr(actual, "__code__", None)
        source = (
            Path(code.co_filename)
            if code is not None and Path(code.co_filename).is_file()
            else reference
        )
        return dict(
            role=role,
            path=binding["path"] if role == "candidate" else str(source),
            qualname=binding["qualname"]
            if role == "candidate"
            else getattr(actual, "__qualname__", str(type(actual))),
            source_sha256=file_sha(candidate) if role == "candidate" else file_sha(source),
            adapter_id=binding["adapter_id"],
            reference_artifact_sha256=artifact_sha,
            actual_target_calls=count,
            callable_code_file=code.co_filename if code is not None else None,
            torch_runtime=str(torch.__version__),
            actual_device_uuid=actual_device_uuid,
        )

    if binding["adapter_id"] == "torch_dictionary_tensor_v1":
        if (
            data["case_recipe_id"] != "dictionary_repeated_layouts_v1"
            or binding["qualname"] != "run"
            or schema.get("dtype") != "float32"
            or schema.get("shape") != [3, 7]
            or schema.get("tensor_key") != "x"
            or data["state_schema"].get("retain_returned_values_between_steps") is not True
        ):
            raise ObservationUnavailable(
                "dictionary recipe requires run(d),float32 x[3,7] and retained outputs"
            )
        cases = []
        for name, row_stride, column_stride, offset in (
            ("contiguous", 7, 1, 0),
            ("independent_stride_offset", 17, 2, 1),
        ):
            steps = []
            for step in range(2):
                values = torch.tensor(
                    [(i - 10) / 8 + step / 4 for i in range(21)], dtype=torch.float32, device=device
                ).reshape(3, 7)
                storage = torch.zeros(
                    offset + 2 * row_stride + 6 * column_stride + 1,
                    dtype=torch.float32,
                    device=device,
                )
                x = storage.as_strided((3, 7), (row_stride, column_stride), offset)
                x.copy_(values)
                steps.append(dict(args=[dict(x=x)], kwargs={}))
            cases.append(dict(case_id=name, initial_state={}, steps=steps))
        if role == "candidate":
            sys.path.insert(0, str(root))
        module = (
            _load(reference, "_ic_public_transition_reference")
            if role == "reference"
            else _load(candidate, "_ic_current_transition_candidate")
        )
        target = getattr(module, "reference" if role == "reference" else "run", None)
        if not callable(target) or inspect.isclass(target):
            raise ObservationUnavailable("dictionary entry must be an actual function")
        target_file = inspect.getsourcefile(target)
        if role == "candidate" and (
            target_file is None or Path(target_file).resolve() != candidate
        ):
            raise ObservationUnavailable("dictionary candidate resolved to another module")
        if (
            role == "reference"
            and target_file is not None
            and Path(target_file).resolve().is_relative_to(root)
        ):
            raise ObservationUnavailable(
                "dictionary public reference resolved into candidate source"
            )

        def factory(case):
            count = [0]

            def invoke(*args, **kwargs):
                if count[0] >= 4:
                    raise ObservationUnavailable("actual dictionary callable limit")
                count[0] += 1
                return target(*args, **kwargs)

            return TransitionEndpoint(invoke, lambda: {}, lambda: proof(count[0], target))

        return cases, factory
    if binding["adapter_id"] == "torch_unary_tensor_v1":
        if data["case_recipe_id"] != "unary_independent_layouts_v1":
            raise ObservationUnavailable("unsupported tensor recipe")
        if schema.get("dtype") != "float32" or schema.get("shape") != [17]:
            raise ObservationUnavailable("first unary recipe requires explicit float32 shape17")
        # Fixed public generic values/layouts,not any grader's case generator.
        cases = []
        for name, stride, offset in (("contiguous", 1, 0), ("independent_stride_offset", 2, 1)):
            cases.append(
                dict(
                    case_id=name,
                    initial_state={},
                    steps=[
                        dict(
                            args=[
                                _tensor(
                                    torch,
                                    [float(i - 8) / 7 for i in range(17)],
                                    device,
                                    stride,
                                    offset,
                                )
                            ],
                            kwargs={},
                        )
                    ],
                )
            )
        if role == "candidate":
            sys.path.insert(0, str(root))
        module = (
            _load(reference, "_ic_public_transition_reference")
            if role == "reference"
            else _load(candidate, "_ic_current_transition_candidate")
        )
        if role == "reference":
            target = getattr(module, "reference", None)
        else:
            target = module
            parts = binding["qualname"].split(".")
            for part in parts:
                if inspect.isclass(target):
                    target = target()
                target = getattr(target, part)
        if not callable(target) or inspect.isclass(target):
            raise ObservationUnavailable("declared tensor entry is not a function/method callable")
        target_file = inspect.getsourcefile(target)
        if role == "candidate" and (
            target_file is None or Path(target_file).resolve() != candidate
        ):
            raise ObservationUnavailable("candidate entry resolved to another module")
        if (
            role == "reference"
            and target_file is not None
            and Path(target_file).resolve().is_relative_to(root)
        ):
            raise ObservationUnavailable("public reference resolved into candidate source")

        def factory(case):
            count = [0]

            def invoke(*args, **kwargs):
                if count[0] >= 4:
                    raise ObservationUnavailable("actual callable limit")
                count[0] += 1
                return target(*args, **kwargs)

            return TransitionEndpoint(invoke, lambda: {}, lambda: proof(count[0], target))

        return cases, factory
    if (
        binding["adapter_id"] != "torch_adam_state_v1"
        or data["case_recipe_id"] != "optimizer_state_and_one_factor_v1"
    ):
        raise ObservationUnavailable("unsupported public tensor/state adapter")
    if (
        binding["qualname"] != "_single_tensor_adam"
        or schema.get("dtype") != "float32"
        or schema.get("shape") != [3]
    ):
        raise ObservationUnavailable(
            "explicit public Adam implementation and float32 shape3 required"
        )
    adam = importlib.import_module("torch.optim.adam")
    if file_sha(adam.__file__) != schema["torch_runtime"]["optimizer_module_sha256"]:
        raise ObservationUnavailable("installed public optimizer source differs")
    installed = getattr(adam, "_single_tensor_adam")
    target = installed
    if role == "candidate":
        tree = ast.parse(candidate.read_text(encoding="utf-8"))
        nodes = [
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == binding["qualname"]
        ]
        if len(nodes) != 1:
            raise ObservationUnavailable("candidate function is missing or ambiguous")
        original = ast.parse(inspect.getsource(installed)).body[0]
        if _header(nodes[0]) != _header(original):
            raise ObservationUnavailable("candidate callable interface differs")
        # Compile only this declared function against the installed public
        # module's original globals. The isolated role process owns the patch.
        node, namespace = _optimizer_namespace(nodes[0], adam)
        code = ast.Module(body=[node], type_ignores=[])
        exec(compile(ast.fix_missing_locations(code), str(candidate), "exec"), namespace)
        target = namespace[binding["qualname"]]
    public = _load(reference, "_ic_public_optimizer_constructor")
    if not callable(getattr(public, "make_optimizer", None)):
        raise ObservationUnavailable(
            "public reference must expose make_optimizer(parameters,options)"
        )
    options = dict(schema.get("options", {}))
    if set(options) - {"lr", "betas", "eps", "weight_decay", "amsgrad", "maximize"}:
        raise ObservationUnavailable("unsupported public optimizer option")
    if type(options.get("amsgrad", False)) is not bool:
        raise ObservationUnavailable("boolean AMSGrad option required")
    variants = [
        dict(
            case_id="repeated_state",
            initial_state=dict(existing=False),
            steps=[
                dict(
                    args=[
                        _tensor(
                            torch, [0.1 * (step + 1), -0.2 * (step + 1), 0.05 * (step + 1)], device
                        )
                    ],
                    kwargs={},
                )
                for step in range(3)
            ],
        ),
        dict(
            case_id="existing_state_one_factor",
            initial_state=dict(existing=True),
            steps=[dict(args=[_tensor(torch, [0.15, -0.3, 0.075], device, 2, 1)], kwargs={})],
        ),
    ]

    def factory(case):
        params = [torch.nn.Parameter(_tensor(torch, [1.0, -2.0, 0.5], device))]
        count = [0]

        def captured(*args, **kwargs):
            if count[0] >= 4:
                raise ObservationUnavailable("actual optimizer target limit")
            count[0] += 1
            return target(*args, **kwargs)

        adam._single_tensor_adam = captured
        selected = dict(options)
        if case["initial_state"]["existing"]:
            selected["amsgrad"] = not selected.get("amsgrad", False)
        optimizer = public.make_optimizer(params, selected)
        if not isinstance(optimizer, torch.optim.Adam):
            raise ObservationUnavailable("reference constructor returned another optimizer")
        if case["initial_state"]["existing"]:
            state = optimizer.state[params[0]]
            state.update(
                step=torch.tensor(2.0, device=device),
                exp_avg=torch.full_like(params[0], 0.03),
                exp_avg_sq=torch.full_like(params[0], 0.04),
            )
            if selected.get("amsgrad"):
                state["max_exp_avg_sq"] = torch.full_like(params[0], 0.05)

        def invoke(gradient):
            params[0].grad = gradient
            optimizer.step()
            return None

        def state():
            return dict(
                parameters=params,
                optimizer_state={str(i): dict(optimizer.state[p]) for i, p in enumerate(params)},
            )

        return TransitionEndpoint(invoke, state, lambda: proof(count[0], target))

    return variants, factory
