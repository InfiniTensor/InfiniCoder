"""Source-bound executable design hypotheses,kept out of code facts/acceptance.

Pure mathematical/finite-model checks only. Candidate code is never imported,
commands are never run,and a checked design cannot satisfy a caller Check.
"""

from infinicoder._resources import package_root

from copy import deepcopy
import hashlib
import json
import re
from infinicoder.memory.repair_memory import signature

TOOLS = {"operator": ("layout_probe",), "framework": ("lifecycle_probe",)}


def _summary(report):
    fields = (
        "status",
        "scope",
        "mode",
        "counterexample",
        "reason",
        "missing_events",
        "unreached_required_states",
        "missing_terminals",
        "active_accesses",
        "logical_elements",
        "thread_points",
        "minimum_span_elements",
        "full_write_coverage",
        "unique_writers",
        "terminal_states_checked",
        "max_depth",
        "initial_units",
        "complete_implementation_coverage",
    )
    result = {k: deepcopy(report[k]) for k in fields if k in report}
    if len(json.dumps(result, ensure_ascii=False)) > 1400:
        for key in ("missing_events", "unreached_required_states", "missing_terminals"):
            if key in result:
                result[key + "_detail_omitted"] = True
                result.pop(key)
        if len(json.dumps(result, ensure_ascii=False)) > 1400:
            result = {k: v for k, v in result.items() if k in ("status", "scope", "reason", "mode")}
            result["detail_omitted"] = True
        result["full_report_location"] = "session.design_hypotheses by design_id"
    return result


def identity():
    root = package_root()
    return hashlib.sha256(
        b"".join(
            (root / p).read_bytes()
            for p in (
                "planning/design_probes.py",
                "domains/operator/layout_probe.py",
                "domains/sysdev/lifecycle_probe.py",
                "domains/operator/tensor_layout.py",
                "runtime/engine.py",
                "runtime/protocol.py",
                "runtime/worker.py",
                "runtime/work_packet.py",
                "runtime/recovery.py",
                "analysis/task_compiler.py",
            )
        )
    ).hexdigest()


def schemas():
    def obj(properties, required):
        return dict(
            type="object", properties=properties, required=required, additionalProperties=False
        )

    text = dict(type="string", minLength=1, maxLength=512)
    ident = dict(type="string", pattern="^[A-Za-z][A-Za-z0-9_]{0,23}$")
    state_ident = dict(type="string", pattern="^[A-Za-z][A-Za-z0-9_]{0,31}$")
    number = dict(type="integer", minimum=0, maximum=(1 << 63) - 1)
    vector = dict(type="array", maxItems=4, items=number)
    source = dict(path=text, expected_hash=dict(type="string", pattern="^[0-9a-f]{64}$"))
    layout = dict(
        source,
        tensor=ident,
        shape=vector,
        element_strides=vector,
        axes=dict(
            type="array",
            maxItems=4,
            items=obj(dict(name=ident, extent=dict(anyOf=[number, text])), ["name", "extent"]),
        ),
        indices=dict(type="array", maxItems=4, items=text),
        element_offset=text,
        mask=text,
        mode=dict(type="string", enum=["read", "write"]),
    )
    transition = obj(
        {
            "from": state_ident,
            "event": state_ident,
            "to": state_ident,
            "delta": dict(type="integer", minimum=-8, maximum=8),
        },
        ["from", "event", "to", "delta"],
    )
    lifecycle = dict(
        source,
        initial=state_ident,
        initial_units=dict(type="integer", minimum=0, maximum=8),
        terminals=dict(type="array", minItems=1, maxItems=16, items=state_ident),
        transitions=dict(type="array", minItems=1, maxItems=64, items=transition),
        depth=dict(type="integer", minimum=1, maximum=12),
    )
    return dict(
        layout_probe=obj(
            layout,
            [
                "path",
                "expected_hash",
                "shape",
                "element_strides",
                "axes",
                "indices",
                "element_offset",
                "mode",
            ],
        ),
        lifecycle_probe=obj(
            lifecycle, ["path", "expected_hash", "initial", "terminals", "transitions"]
        ),
    )


def _source(state, workspace, args, observed):
    path, digest = args.get("path"), args.get("expected_hash")
    if (
        not isinstance(path, str)
        or len(path) > 512
        or not isinstance(digest, str)
        or not re.fullmatch("[0-9a-f]{64}", digest)
    ):
        raise ValueError("design probe needs bounded path and current expected_hash")
    file = workspace.path(path)
    if str(file.resolve()) in state.get("checker_files", {}):
        raise ValueError("acceptance checker is not a design target")
    if not file.is_file() or not 0 < file.stat().st_size <= 2_000_000:
        raise ValueError("design source must be a bounded existing file")
    if hashlib.sha256(file.read_bytes()).hexdigest() != digest:
        raise ValueError("design source changed;inspect the current version")
    relative = file.relative_to(workspace.root).as_posix()
    if not any(
        not r.get("source_omitted")
        and r.get("path") == relative
        and r.get("sha256", r.get("source_sha256")) == digest
        for r in observed
    ):
        raise ValueError("inspect current source before proposing its design")
    return dict(path=relative, sha256=digest)


def _layout_binding(spec, args):
    layouts = spec.tensor_layouts
    tensor = args.get("tensor")
    if tensor is not None and (
        not isinstance(tensor, str) or not re.fullmatch("[A-Za-z][A-Za-z0-9_]{0,23}", tensor)
    ):
        raise ValueError("bounded tensor name required")
    if not layouts:
        return dict(kind="model_case_assumption", tensor=tensor, caller_layout_verified=False)
    if tensor is None and len(layouts) == 1:
        tensor = next(iter(layouts))
    if tensor not in layouts:
        raise ValueError(
            "select a caller-declared tensor;do not bypass its layout with an invented case"
        )
    known = layouts[tensor]
    symbols = {}
    for key in ("shape", "element_strides"):
        values = args.get(key)
        if not isinstance(values, list) or len(values) != len(known[key]):
            raise ValueError("probe rank differs from caller layout")
        for declared, value in zip(known[key], values):
            if type(declared) is int:
                if declared != value:
                    raise ValueError("probe contradicts caller " + key)
            else:
                if declared in symbols and symbols[declared] != value:
                    raise ValueError("inconsistent symbolic layout instantiation")
                symbols[declared] = value
    return dict(
        kind="caller_layout" if not symbols else "caller_symbolic_instantiation_assumption",
        tensor=tensor,
        caller_layout_verified=not bool(symbols),
        symbol_values_are_assumptions=symbols,
    )


def run(tool, args, spec, state, workspace, profile, observed):
    if tool not in TOOLS.get(profile, ()):
        raise ValueError("design tool unavailable for this task domain")
    if not isinstance(args, dict):
        raise ValueError("design arguments must be an object")
    source = _source(state, workspace, args, observed)
    requirements = signature(spec)
    declaration = deepcopy(args)
    design = {k: v for k, v in declaration.items() if k not in ("path", "expected_hash", "tensor")}
    caller = spec.target_facts.get("design_requirements", {})
    if not isinstance(caller, dict):
        raise ValueError("caller design_requirements must be an object")
    if tool == "layout_probe":
        from infinicoder.domains.operator.layout_probe import check

        binding = _layout_binding(spec, declaration)
        report = check(design)
    else:
        from infinicoder.domains.sysdev.lifecycle_probe import check

        if "tensor" in declaration:
            raise ValueError("tensor is not a lifecycle field")
        contract = caller.get("lifecycle", {})
        if not isinstance(contract, dict):
            raise ValueError("caller lifecycle requirement must be an object")
        if "initial_units" in contract and (
            type(contract["initial_units"]) is not int or not 0 <= contract["initial_units"] <= 8
        ):
            raise ValueError("caller initial_units must be integer in[0,8]")
        if "minimum_depth" in contract and (
            type(contract["minimum_depth"]) is not int or not 1 <= contract["minimum_depth"] <= 12
        ):
            raise ValueError("caller minimum_depth must be integer in[1,12]")
        if (
            "initial_units" in contract
            and design.get("initial_units", 0) != contract["initial_units"]
        ):
            raise ValueError("design contradicts caller initial resource units")
        if "initial" in contract and design.get("initial") != contract["initial"]:
            raise ValueError("design contradicts caller initial state")
        if "minimum_depth" in contract and design.get("depth", 6) < contract["minimum_depth"]:
            raise ValueError("design depth weaker than caller requirement")
        report = check(design, required_events=contract.get("required_events"))
        required_terminals = contract.get("terminals", [])
        if (
            not isinstance(required_terminals, list)
            or len(required_terminals) > 16
            or any(
                not isinstance(v, str) or not re.fullmatch("[A-Za-z][A-Za-z0-9_]{0,31}", v)
                for v in required_terminals
            )
        ):
            raise ValueError("caller terminal requirement must be bounded state identifiers")
        if not set(required_terminals) <= set(design["terminals"]):
            report = dict(
                report,
                status="inconclusive",
                reason="caller terminal paths omitted",
                missing_terminals=sorted(set(required_terminals) - set(design["terminals"])),
            )
        unsupported = set(contract) - {
            "initial",
            "initial_units",
            "minimum_depth",
            "terminals",
            "required_events",
        }
        if unsupported and report["status"] == "bounded_design_checked":
            report = dict(
                report,
                status="inconclusive",
                reason="caller design requirements exceed this checker",
                unchecked_requirements=sorted(unsupported),
            )
        binding = dict(
            kind="caller_constrained_model" if contract else "model_resource_assumption",
            caller_requirements_present=bool(contract),
        )
    if (
        hashlib.sha256(workspace.path(source["path"]).read_bytes()).hexdigest() != source["sha256"]
        or signature(spec) != requirements
    ):
        raise ValueError(
            "source or caller requirements changed during design check;repeat against current evidence"
        )
    report = json.loads(json.dumps(report))
    token = hashlib.sha256(
        json.dumps([tool, declaration, source, requirements, identity()], sort_keys=True).encode()
    ).hexdigest()
    record = dict(
        id=token,
        tool=tool,
        source=source,
        requirements=requirements,
        analysis_identity=identity(),
        model_declaration=declaration,
        binding=binding,
        symbolic_observation=report,
        acceptance_proof=False,
        implementation_correspondence="unverified;source hash is a lease,not proof that code implements this design",
    )
    history = [r for r in state.get("design_hypotheses", []) if r["id"] != token] + [record]
    state["design_hypotheses"] = history[-6:]
    return dict(
        design_id=token,
        source=source,
        model_claims=True,
        binding=binding,
        symbolic_observation=_summary(report),
        acceptance_proof=False,
        implementation_correspondence=record["implementation_correspondence"],
        next="Use the concrete counterexample to revise design/code;run every original independent correctness/integration check. A design check cannot finish the task.",
    )


def current(spec, state, workspace):
    records = []
    expired = []
    for record in state.get("design_hypotheses", []):
        try:
            valid = (
                record["requirements"] == signature(spec)
                and record["analysis_identity"] == identity()
                and hashlib.sha256(
                    workspace.path(record["source"]["path"]).read_bytes()
                ).hexdigest()
                == record["source"]["sha256"]
            )
        except (OSError, ValueError):
            valid = False
        if not valid:
            expired.append(record["id"])
            continue
        compact = _summary(record["symbolic_observation"])
        records.append(
            dict(
                id=record["id"],
                tool=record["tool"],
                source=record["source"],
                binding=record["binding"],
                symbolic_observation=compact,
                model_claims=True,
                acceptance_proof=False,
            )
        )
    # Reuse is local to this session/source/contract,never code facts or a skill.
    return dict(
        current=records[-1:],
        expired_ids=expired[-6:],
        implementation_correctness="unverified;run original final checks",
    )


def context(profile, spec, state, workspace, base=None):
    if profile not in TOOLS:
        return base
    tool = (
        "layout_probe {path,expected_hash,tensor?,shape,element_strides,axes:[{name,extent}],indices:[expr],element_offset:expr,mask?:bool_expr,mode:read|write}. d0.. are dimensions,s0.. THIS tensor strides. extent supports integer expressions,ceildiv,min,max;no Python execution. Write mode checks complete coverage and unique writers,read mode only visited bounds. Use a small declared case;caller layouts cannot be contradicted."
        if profile == "operator"
        else "lifecycle_probe {path,expected_hash,initial,initial_units?:0,terminals:[state],transitions:[{from,event,to,delta}],depth?:6}. Model each object ownership including success,error,cancel and repeated cleanup. Tool explores declared transitions for underflow/leaks;missing paths are unknown. Deltas are assumptions,not observed code behavior."
    )
    base = base or {}
    return dict(
        id="executable-design-probes-v1",
        analysis_identity=identity(),
        source_revision=state.get("index_revision"),
        stage=base.get("stage", "investigate"),
        sequence=base.get("sequence", []),
        tool=tool,
        method="Inspect source;propose an index/ownership model when needed;use tool counterexamples to guide implementation;validate actual code independently. Do not submit a long form merely to fill a template.",
        design_evidence=current(spec, state, workspace),
        acceptance="All original caller checks remain mandatory. A model design or bounded symbolic result never certifies actual code,numerics,concurrency or performance.",
    )
