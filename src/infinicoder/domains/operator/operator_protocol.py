"""Budgeted operator-primitive work questions,not source facts or solved kernels."""

import hashlib, json, re
from pathlib import Path

CARDS = (
    (
        "scan",
        r"\b(prefix[ -]scan|exclusive[ -]scan|inclusive[ -]scan|cumulative sums?|cumsum|associative scan)\b|前缀和|前缀扫描|累积和",
        (
            "Bind the scan axis from the requirement;preserve all other dimensions and treat each independent slice separately. Do not infer rank or axis from the first input.",
            "Distinguish inclusive/exclusive and forward/reverse semantics. Before coding,work through a tiny asymmetric slice by hand,including its first and last output;use the identity for an empty prefix.",
            "Choose a scan algorithm,not a pointwise map or ordinary reduction. State accumulation dtype,output dtype and stride assumptions. Check singleton,tail and noncontiguous slices with public examples.",
        ),
    ),
    (
        "reduction",
        r"\b(reduction|reduce[ -]sum|reduce[ -]max|softmax|rmsnorm|layernorm)\b|归约|归一化",
        (
            "Bind reduced axes,kept dimensions and broadcasting before coding. Separate independent output slices from elements contributing to one output.",
            "Specify neutral element,accumulation precision,empty/tail behavior and numerical stability. Derive a tiny asymmetric example before choosing tiling or fusion.",
            "Check independent input/output strides and masking;only use a seed when its layout,dtype and semantic preconditions hold. Measure after correctness.",
        ),
    ),
    (
        "contraction",
        r"\b(gemm|matmul|matrix multiplication|batched matrix)\b|矩阵乘|张量收缩",
        (
            "Bind batch,free and contraction axes for each operand independently. Check broadcasting,transpose and output shape from the public contract.",
            "Track each operand stride,offset,dtype and accumulation precision separately;do not inherit one input layout for another.",
            "Choose tile sizes after a correct contraction mapping;test unequal dimensions,tails and layout variants before external timing.",
        ),
    ),
)


def identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def pack(spec, domain, budget, estimate):
    """Caller text selects reviewed questions;no answers,code or inferred facts."""
    report = dict(
        identity=identity(),
        status="not_selected",
        selected=[],
        included=False,
        model_calls=0,
        acceptance_proof=False,
        effects=["context_only"],
    )
    if domain != "operator":
        report["reason"] = "operator profile required"
        return "", report
    task = spec.task[:24000]
    matched = [(name, steps) for name, pattern, steps in CARDS if re.search(pattern, task, re.I)]
    report["matched"] = [name for name, steps in matched]
    if not matched:
        report["reason"] = "no reviewed primitive matches;generic workflow retained"
        return "", report
    selected = []
    text = ""
    for name, steps in matched[:2]:
        trial = selected + [dict(primitive=name, work_questions=steps)]
        view = dict(
            kind="operator_primitive_protocol",
            knowledge_identity=identity(),
            cards=trial,
            epistemic_status="reviewed generic method;task bindings and implementation remain unverified",
            next="Resolve these questions against the caller requirement and current source,then implement. Do not submit this guidance as a design action or acceptance proof.",
        )
        candidate = "\nOPERATOR PRIMITIVE WORK PROTOCOL\n" + json.dumps(view, ensure_ascii=False)
        if estimate(candidate) + 6 <= budget:
            selected = trial
            text = candidate
    report.update(
        selected=[r["primitive"] for r in selected],
        included=bool(text),
        status="included" if text else "omitted_for_context_budget",
        context_tokens=estimate(text) + 6 if text else 0,
        budget=budget,
    )
    return text, report
