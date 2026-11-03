"""Executable workflow selection, driven by controller evidence, not self-report.

Templates guide model decisions; they neither supply benchmark implementations
nor replace caller-owned checks or trusted performance measurements.
"""

from infinicoder._resources import package_file

import hashlib
import json
from pathlib import Path
import re


def identity():
    folder = package_file("workflows")
    return hashlib.sha256(
        Path(__file__).read_bytes()
        + b"".join(p.read_bytes() for p in sorted(folder.glob("*.json")))
    ).hexdigest()


def workflow_packet(profile, spec, state):
    if profile not in ("operator", "framework"):
        return None
    template = json.loads(
        package_file("workflows").joinpath(profile + ".json").read_text(encoding="utf-8")
    )
    text = spec.task.lower()
    branches = []
    if profile == "framework":
        rules = [
            (
                "training",
                r"\b(train\w*|optimizer|adamw?|sgd|gradient|backward|loss|momentum)\b|训练|梯度|优化器",
            ),
            (
                "checkpoint",
                r"\b(checkpoint\w*|state_dict|load_state_dict|resum\w*|restor\w*)\b|保存恢复|断点续训",
            ),
            (
                "inference",
                r"\b(infer\w*|serving|prefill|decode|kv.?cache|request|batching)\b|推理|调度",
            ),
            (
                "concurrency",
                r"\b(concurr\w*|thread\w*|lock|queue|cancel\w*|shutdown|distribut\w*|collective)\b|并发|分布式|取消",
            ),
        ]
        branches = [name for name, pattern in rules if re.search(pattern, text)]
    else:
        if re.search(r"\b(reduc\w*|softmax|norm\w*|argmin|argmax|sum|mean)\b|归约|规约", text):
            branches.append("reduction")
        if re.search(r"\b(gemm|matmul|convol\w*|attention)\b|矩阵|卷积|注意力", text):
            branches.append("contraction")
    # A successful independent correctness pass must belong to the current
    # source; a prior baseline/best record alone never certifies a new edit.
    failure = state.get("last_failure")
    changed = bool(state.get("changed_files"))
    optimization = state.get("optimization") or {}
    if failure:
        stage = "diagnose"
    elif changed:
        stage = "validate"
    elif (
        state.get("read_coverage")
        or state.get("source_inspections")
        or state.get("plan_milestones")
    ):
        stage = "implement"
    else:
        stage = "investigate"
    obligations = template["stages"][stage]
    specialized = [item for name in branches for item in template["branches"].get(name, [])]
    result = dict(
        id=template["id"],
        analysis_identity=identity(),
        stage=stage,
        stage_basis="controller source inspections,edits and independent failures;model notes do not advance stages",
        source_revision=state.get("index_revision"),
        sequence=template["sequence"],
        next_actions=obligations,
        specialized=specialized,
        acceptance=template["acceptance"],
        performance_feedback_available=bool(optimization.get("contract")),
        performance_rule="Only trusted independent measurements may select or retain an optimization. If unavailable,report performance as unverified.",
        branches=branches,
    )

    # A caller binds an actual public procedure, not a command guessed from
    # task prose or a private final. Reuse the ordinary verify tool.
    procedure = spec.target_facts.get("checkpoint_equivalence")
    if (
        profile == "framework"
        and isinstance(procedure, dict)
        and procedure.get("check_id") in {check.id for check in spec.checks}
        and isinstance(procedure.get("components"), list)
        and 1 <= len(procedure["components"]) <= 8
        and all(
            isinstance(x, str) and re.fullmatch("[a-z][a-z0-9_]{0,47}", x)
            for x in procedure["components"]
        )
    ):
        result["public_procedure"] = dict(
            kind="checkpoint_equivalence_v1",
            check_id=procedure["check_id"],
            components=list(procedure["components"]),
            coverage_authority="caller declaration;actual returned report binds tested source versions",
            invoke=dict(tool="verify", arguments={}),
            rule="Use the registered public check when verify is available in the current phase. Compare restored state and continued steps,then repair the reported producer. This is a public observation,not final acceptance;do not replace caller checks.",
        )

    if state.get("profiling"):
        result.update(
            profiling_workloads=[w["name"] for w in state["profiling"]["contract"]["workloads"]],
            profiling_rule="Use profile on a configured runnable workload to obtain source-bound diagnostic evidence. First establish correctness;form a bottleneck hypothesis from observations,then transform and reverify. Profiler durations are not optimization selection timings.",
        )
    return result
