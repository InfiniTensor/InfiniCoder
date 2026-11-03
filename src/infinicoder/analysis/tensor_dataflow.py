"""Task-local static tensor-flow hypotheses with source versions and unknowns.

No candidate program is imported or executed. Library transfer rules are explicit
assumptions;their obligations never establish implementation correctness.
"""

from infinicoder._resources import package_file

import ast, hashlib, json, re
from dataclasses import asdict
from pathlib import Path

MAX_SOURCE = 32768
MAX_NODES = 2000


def identity():
    return hashlib.sha256(
        Path(__file__).read_bytes() + package_file("analysis/tensor_io.py").read_bytes()
    ).hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _shape(values):
    if not isinstance(values, (list, tuple)) or len(values) > 8:
        return None
    if any(
        not (
            type(v) is int
            and 0 <= v < 2**31
            or isinstance(v, str)
            and re.fullmatch("[A-Za-z][A-Za-z0-9_]{0,31}", v)
        )
        for v in values
    ):
        return None
    return tuple(values)


def declarations(spec, state):
    """Only explicit layouts or unambiguous typed JSON literals from caller text."""
    known = {}
    origins = {}
    conflicts = set()
    for name, value in spec.tensor_layouts.items():
        shape = _shape(value.get("shape"))
        if shape is not None:
            known[name] = shape
            origins[name] = dict(kind="caller_layout", field="tensor_layouts." + name)
    structured = set(known)
    if spec.amendments or state.get("clarifications"):
        return (
            known,
            origins,
            dict(reason="amended prose not reinterpreted;current structured layouts only"),
        )
    text = spec.task[:24000]
    decoder = json.JSONDecoder()
    starts = [m.start() for m in re.finditer(r"\{", text)][:32]
    pattern = (
        r"\s*(?:torch\.)?(?:float(?:16|32|64)?|bfloat16|int(?:8|16|32|64)|bool)\s*\[([^\]]*)\]\s*"
    )
    for start in starts:
        try:
            value, length = decoder.raw_decode(text[start:])
        except (ValueError, RecursionError):
            continue
        if not isinstance(value, dict) or len(value) > 16:
            continue
        for name, description in value.items():
            if (
                name in structured
                or not isinstance(name, str)
                or not re.fullmatch("[A-Za-z][A-Za-z0-9_]{0,31}", name)
                or not isinstance(description, str)
            ):
                continue
            match = re.fullmatch(pattern, description)
            if not match:
                continue
            words = [] if not match[1].strip() else [v.strip() for v in match[1].split(",")]
            shape = _shape([int(v) if v.isdigit() else v for v in words])
            if shape is None:
                continue
            if name in known and known[name] != shape:
                conflicts.add(name)
                continue
            known[name] = shape
            origins[name] = dict(
                kind="caller_text_literal",
                task_sha256=hashlib.sha256(spec.task.encode()).hexdigest(),
                span=[start, start + length],
                declaration=description,
            )
    for name in conflicts:
        known.pop(name, None)
        origins.pop(name, None)
    return (
        known,
        origins,
        dict(
            ambiguous_inputs=sorted(conflicts),
            scope="declared shapes only;symbol values and prose constraints remain unverified",
        ),
    )


class Flow:
    """Small conservative shape transfer domain;None means unknown rank."""

    def __init__(self, source, inputs, aliases):
        self.source, self.inputs, self.aliases = source, inputs, aliases
        self.env = {}
        self.dims = {}
        self.dict_args = set()
        self.trace = []
        self.obligations = []
        self.unknowns = set()
        self.returns = []
        self.steps = 0

    def text(self, node):
        return (ast.get_source_segment(self.source, node) or type(node).__name__)[:120]

    def dim(self, node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value if node.value >= 0 else None
        if isinstance(node, ast.Name):
            return self.dims.get(node.id)
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "shape"
        ):
            shape = self.expr(node.value.value)
            index = (
                node.slice.value
                if isinstance(node.slice, ast.Constant) and type(node.slice.value) is int
                else None
            )
            return (
                shape[index]
                if shape is not None and index is not None and -len(shape) <= index < len(shape)
                else None
            )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "size"
            and len(node.args) == 1
        ):
            shape = self.expr(node.func.value)
            index = (
                node.args[0].value
                if isinstance(node.args[0], ast.Constant) and type(node.args[0].value) is int
                else None
            )
            return (
                shape[index]
                if shape is not None and index is not None and -len(shape) <= index < len(shape)
                else None
            )
        return None

    def new_shape(self, args):
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, ast.Attribute) and arg.attr == "shape":
                return self.expr(arg.value)
            if isinstance(arg, (ast.Tuple, ast.List)):
                return tuple(self.dim(x) for x in arg.elts)
        return tuple(self.dim(x) for x in args) if args else None

    def broadcast(self, left, right):
        if left is None or right is None:
            return None
        result = []
        for i in range(1, max(len(left), len(right)) + 1):
            a = left[-i] if i <= len(left) else 1
            b = right[-i] if i <= len(right) else 1
            result.append(b if a == 1 else a if b == 1 or a == b else None)
        return tuple(reversed(result))

    def subscript(self, shape, index):
        if shape is None:
            return None
        indices = list(index.elts) if isinstance(index, ast.Tuple) else [index]
        if sum(isinstance(v, ast.Constant) and v.value is Ellipsis for v in indices) > 1:
            return None
        result = []
        offset = 0
        for i, part in enumerate(indices):
            if isinstance(part, ast.Constant) and part.value is Ellipsis:
                count = (
                    len(shape)
                    - offset
                    - sum(
                        not (isinstance(v, ast.Constant) and v.value is None)
                        for v in indices[i + 1 :]
                    )
                )
                if count < 0:
                    return None
                result.extend(shape[offset : offset + count])
                offset += count
                continue
            if isinstance(part, ast.Constant) and part.value is None:
                result.append(1)
                continue
            if offset >= len(shape):
                return None
            if isinstance(part, ast.Slice):
                result.append(
                    shape[offset] if part.lower is part.upper is part.step is None else None
                )
            elif isinstance(part, ast.Constant) and type(part.value) is int:
                pass
            elif isinstance(part, ast.Name) and self.env.get(part.id) == ():
                pass
            else:
                return None
            offset += 1
        return tuple(result) + shape[offset:]

    def expr(self, node, depth=0):
        if depth > 20:
            self.unknowns.add("expression depth")
            return None
        if isinstance(node, ast.Name):
            return self.env.get(node.id)
        if isinstance(node, ast.Constant):
            return () if isinstance(node.value, (int, float, bool)) else None
        if isinstance(node, ast.Subscript):
            if (
                isinstance(node.value, ast.Name)
                and node.value.id in self.dict_args
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
            ):
                return self.inputs.get(node.slice.value)
            return self.subscript(self.expr(node.value, depth + 1), node.slice)
        if isinstance(node, (ast.BinOp, ast.BoolOp)):
            values = [node.left, node.right] if isinstance(node, ast.BinOp) else node.values
            shape = ()
            for value in values:
                shape = self.broadcast(shape, self.expr(value, depth + 1))
            return shape
        if isinstance(node, ast.UnaryOp):
            return self.expr(node.operand, depth + 1)
        if isinstance(node, ast.Call):
            name = ""
            receiver = None
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
                receiver = self.expr(node.func.value, depth + 1)
                if (
                    isinstance(node.func.value, ast.Name)
                    and self.aliases.get(node.func.value.id) == "torch"
                ):
                    name = "torch." + name
            elif isinstance(node.func, ast.Name):
                name = self.aliases.get(node.func.id, node.func.id)
            if (
                name in ("new_empty", "new_zeros", "new_ones", "reshape", "view")
                and receiver is not None
            ):
                return self.new_shape(node.args)
            if name in ("torch.empty", "torch.zeros", "torch.ones"):
                return self.new_shape(node.args)
            if (
                name
                in (
                    "torch.empty_like",
                    "torch.zeros_like",
                    "torch.ones_like",
                    "torch.cumsum",
                    "torch.clone",
                )
                and node.args
            ):
                return self.expr(node.args[0], depth + 1)
            if (
                name in ("cumsum", "clone", "detach", "contiguous", "fill_", "zero_")
                and receiver is not None
            ):
                return receiver
            if name in ("torch.sum", "torch.mean", "torch.amax", "sum", "mean", "amax"):
                base = (
                    self.expr(node.args[0], depth + 1)
                    if name.startswith("torch.") and node.args
                    else receiver
                )
                args = node.args[1:] if name.startswith("torch.") else node.args
                dim = next(
                    (k.value for k in node.keywords if k.arg == "dim"), args[0] if args else None
                )
                keep = next(
                    (
                        k.value.value
                        for k in node.keywords
                        if k.arg == "keepdim" and isinstance(k.value, ast.Constant)
                    ),
                    False,
                )
                if dim is None:
                    return () if base is not None else None
                axis = (
                    dim.value
                    if isinstance(dim, ast.Constant) and type(dim.value) is int
                    else -dim.operand.value
                    if isinstance(dim, ast.UnaryOp)
                    and isinstance(dim.op, ast.USub)
                    and isinstance(dim.operand, ast.Constant)
                    and type(dim.operand.value) is int
                    else None
                )
                if base is not None and axis is not None and -len(base) <= axis < len(base):
                    axis %= len(base)
                    return (
                        tuple(1 if i == axis else v for i, v in enumerate(base))
                        if keep
                        else tuple(v for i, v in enumerate(base) if i != axis)
                    )
            self.unknowns.add("unsupported call " + name)
        return None

    def assign(self, target, value, node):
        shape = self.expr(value)
        if isinstance(target, ast.Name):
            dimension = self.dim(value)
            if shape is None and dimension is not None:
                shape = ()
            self.dict_args.discard(target.id)
            self.aliases.pop(target.id, None)
            self.env[target.id] = shape
            self.dims[target.id] = dimension
            if shape is not None:
                row = dict(line=node.lineno, name=target.id, shape=list(shape))
                if row not in self.trace and len(self.trace) < 64:
                    self.trace.append(row)
        elif isinstance(target, ast.Subscript):
            dest = self.expr(target)
            requirements = []
            if dest is None or shape is None:
                return
            extra = max(0, len(shape) - len(dest))
            for dimension in shape[:extra]:
                if dimension != 1 and dimension is not None:
                    requirements.append(str(dimension) + " == 1")
            tail = shape[extra:]
            for left, right in zip(reversed(tail), reversed(dest)):
                if left is not None and right is not None and left not in (1, right):
                    requirements.append(str(left) + " == 1 or " + str(left) + " == " + str(right))
            if requirements:
                row = dict(
                    line=node.lineno,
                    target=self.text(target),
                    target_shape=list(dest),
                    value=self.text(value),
                    value_shape=list(shape),
                    requires=requirements[:4],
                    kind="conditional_tensor_assignment",
                    reachability="unproven;loop/branch may be empty",
                    interpretation="Necessary shape obligations under tensor assignment semantics,not a runtime observation or proof of defect.",
                )
                if row not in self.obligations and len(self.obligations) < 12:
                    self.obligations.append(row)

    def statements(self, rows, depth=0):
        if depth > 10:
            self.unknowns.add("statement depth")
            return
        for row in rows:
            self.steps += 1
            if self.steps > 512:
                self.unknowns.add("statement budget")
                return
            if isinstance(row, ast.Assign):
                for target in row.targets:
                    self.assign(target, row.value, row)
            elif isinstance(row, ast.AnnAssign) and row.value is not None:
                self.assign(row.target, row.value, row)
            elif isinstance(row, ast.AugAssign):
                self.assign(
                    row.target,
                    ast.copy_location(ast.BinOp(left=row.target, op=row.op, right=row.value), row),
                    row,
                )
            elif isinstance(row, ast.For):
                if isinstance(row.target, ast.Name):
                    self.env[row.target.id] = ()
                for iteration in range(2):
                    before = dict(self.env)
                    self.statements(row.body, depth + 1)
                    if self.env == before:
                        break
                self.statements(row.orelse, depth + 1)
            elif isinstance(row, ast.If):
                before = dict(self.env)
                dims = dict(self.dims)
                self.statements(row.body, depth + 1)
                left = dict(self.env)
                self.env = dict(before)
                self.dims = dict(dims)
                self.statements(row.orelse, depth + 1)
                right = dict(self.env)
                self.env = {
                    key: left.get(key) if left.get(key) == right.get(key) else None
                    for key in left.keys() | right.keys()
                }
                self.dims = {}
            elif isinstance(row, ast.Return):
                shape = self.expr(row.value) if row.value is not None else None
                if shape is not None and list(shape) not in self.returns:
                    self.returns.append(list(shape))
            elif isinstance(row, ast.Expr) and isinstance(row.value, ast.Call):
                call = row.value
                self.expr(call)
                if (
                    isinstance(call.func, ast.Attribute)
                    and call.func.attr.endswith("_")
                    and call.func.attr not in ("fill_", "zero_")
                    and isinstance(call.func.value, ast.Name)
                ):
                    self.env[call.func.value.id] = None
                    self.unknowns.add("unmodeled in-place call " + call.func.attr)
            elif isinstance(row, ast.Import):
                self.aliases.update(
                    {a.asname or a.name: a.name for a in row.names if a.name == "torch"}
                )
            elif isinstance(row, ast.ImportFrom) and row.module == "torch":
                self.aliases.update({a.asname or a.name: "torch." + a.name for a in row.names})
            elif not isinstance(row, ast.Pass):
                self.unknowns.add("unsupported statement " + type(row).__name__)


def analyze(source, inputs):
    report = dict(
        scope="bounded static shape transfer;assumes tensor library semantics;not implementation or numerical correctness",
        functions=[],
        unknowns=[],
    )
    if len(source.encode()) > MAX_SOURCE:
        return dict(report, status="inconclusive", unknowns=["source bound"])
    try:
        tree = ast.parse(source)
    except (SyntaxError, RecursionError) as error:
        return dict(
            report, status="inconclusive", unknowns=[type(error).__name__ + ": " + str(error)[:160]]
        )
    if sum(1 for n in ast.walk(tree)) > MAX_NODES:
        return dict(report, status="inconclusive", unknowns=["AST bound"])
    aliases = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            aliases.update({a.asname or a.name: a.name for a in node.names if a.name == "torch"})
        if isinstance(node, ast.ImportFrom) and node.module == "torch":
            aliases.update({a.asname or a.name: "torch." + a.name for a in node.names})
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    aliases.pop(target.id, None)
    functions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))][:8]
    if not functions:
        return dict(report, status="inconclusive", unknowns=["no supported top-level function"])
    for function in functions:
        flow = Flow(source, inputs, dict(aliases))
        flow.dict_args = {a.arg for a in function.args.args}
        for arg in function.args.args:
            flow.aliases.pop(arg.arg, None)
            if arg.arg in inputs:
                flow.env[arg.arg] = inputs[arg.arg]
                flow.dict_args.discard(arg.arg)
        try:
            flow.statements(function.body)
        except (ValueError, TypeError, IndexError, RecursionError):
            flow.unknowns.add("unsupported transfer")
        report["functions"].append(
            dict(
                name=function.name,
                line=function.lineno,
                bindings=flow.trace,
                obligations=flow.obligations,
                return_shape_hypotheses=flow.returns,
                unknowns=sorted(flow.unknowns)[:8],
            )
        )
    report["status"] = "static_hypotheses"
    return report


def pack(spec, state, workspace, budget, estimate):
    inputs, origins, assumptions = declarations(spec, state)
    contract = digest(dict(spec=asdict(spec), clarifications=state.get("clarifications", [])))
    module = identity()
    records = state.setdefault("tensor_flow_records", [])
    expired = state.setdefault("tensor_flow_expired_ids", [])
    active = []
    for record in records:
        try:
            with workspace.path(record["path"]).open("rb") as stream:
                raw = stream.read(MAX_SOURCE + 1)
            current = (
                len(raw) <= MAX_SOURCE
                and hashlib.sha256(raw).hexdigest() == record["source_sha256"]
            )
        except (OSError, ValueError):
            current = False
        if not current or record["contract"] != contract or record["analysis_identity"] != module:
            if record["id"] not in expired:
                expired.append(record["id"])
        else:
            active.append(record)
    state["tensor_flow_records"] = records = active
    state["tensor_flow_expired_ids"] = expired[-16:]
    receipt = dict(
        included=False,
        status="no_declared_tensor_inputs",
        model_calls=0,
        acceptance_proof=False,
        analysis_identity=module,
        assumptions=assumptions,
        new_records=[],
    )
    if not inputs:
        return "", receipt
    observed = []
    seen = set()
    for row in reversed(state.get("shell_loop_history", [])):
        action = row.get("action") or {}
        observation = row.get("observation") or {}
        if (
            not isinstance(action, dict)
            or action.get("tool") not in ("read", "edit")
            or not isinstance(observation, dict)
        ):
            continue
        path = observation.get("path")
        version = observation.get("sha256") or observation.get("after")
        if (
            not isinstance(path, str)
            or path in seen
            or not isinstance(version, str)
            or not re.fullmatch("[0-9a-f]{64}", version)
        ):
            continue
        seen.add(path)
        try:
            file = workspace.path(path)
            if (
                file.suffix != ".py"
                or str(file.resolve()) in state.get("checker_files", {})
                or not file.is_file()
                or file.stat().st_size > MAX_SOURCE
            ):
                continue
            with file.open("rb") as stream:
                source = stream.read(MAX_SOURCE + 1)
            if len(source) > MAX_SOURCE:
                continue
            if hashlib.sha256(source).hexdigest() != version:
                continue
            source = source.decode("utf8")
        except (OSError, ValueError, UnicodeError):
            continue
        ident = digest(dict(path=path, version=version, contract=contract, module=module))
        record = next((r for r in records if r["id"] == ident), None)
        if record is None:
            record = dict(
                id=ident,
                path=path,
                source_sha256=version,
                contract=contract,
                analysis_identity=module,
                input_shape_hypotheses={k: list(v) for k, v in inputs.items()},
                input_origins=origins,
                assumptions=assumptions,
                report=analyze(source, inputs),
            )
            records.append(record)
            receipt["new_records"].append(record)
        observed.append(record)
        if len(observed) == 2:
            break
    state["tensor_flow_records"] = records[-8:]
    if not observed:
        return "", dict(receipt, status="no_current_observed_source")
    observed.sort(key=lambda r: not any(f["obligations"] for f in r["report"]["functions"]))
    selected = observed[0]
    functions = selected["report"]["functions"]
    packet = dict(
        kind="static_tensor_dataflow",
        path=selected["path"],
        source_sha256=selected["source_sha256"],
        analysis_id=selected["id"],
        input_shape_hypotheses=selected["input_shape_hypotheses"],
        analysis_status=selected["report"]["status"],
        obligations=[
            dict(
                function=f["name"],
                **{
                    k: o[k]
                    for k in ("line", "target", "target_shape", "value", "value_shape", "requires")
                },
            )
            for f in functions
            for o in f["obligations"]
        ][:2],
        bindings=[dict(function=f["name"], **b) for f in functions for b in f["bindings"]][-5:],
        unknowns=(selected["report"]["unknowns"] + [u for f in functions for u in f["unknowns"]])[
            :3
        ],
        assumptions="Shapes from current caller declarations;symbol values and reachability unknown. Static tensor transfer rules,not actual execution or acceptance.",
    )
    text = ""
    for keep in (5, 3, 1, 0):
        value = dict(packet, bindings=packet["bindings"][-keep:] if keep else [])
        trial = "\nSTATIC TENSOR DATA FLOW\n" + json.dumps(value, ensure_ascii=False)
        if estimate(trial) + 6 <= budget:
            text = trial
            break
    receipt.update(
        included=bool(text),
        status="included" if text else "omitted_for_context_budget",
        selected_id=selected["id"],
        context_tokens=estimate(text) + 6 if text else 0,
        budget=budget,
    )
    return text, receipt


def analyze_io_edges(source, *, path, source_sha256, entry_symbol=None):
    """Independent syntactic I/O view;does not change the shape transfer domain."""
    from infinicoder.analysis.tensor_io import analyze as io_analysis

    return io_analysis(
        source,
        path=path,
        source_sha256=source_sha256,
        entry_symbol=entry_symbol,
        max_source=MAX_SOURCE,
        max_nodes=MAX_NODES,
    )
