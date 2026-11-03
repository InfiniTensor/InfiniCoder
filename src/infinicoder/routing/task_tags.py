"""Multi-axis, source-attributed routing hypotheses; never instructions or proof.

This bounded classifier uses no benchmark IDs, reference implementation, model
calls or repository code execution. Absence of a match means unknown, not false.
"""

import hashlib
import re
from pathlib import Path


DOMAIN_RULES = {
    "operator": r"\b(triton|cuda kernel|gemm|matmul|rmsnorm|rms.?norm|layernorm|softmax|exclusive scan|inclusive scan)\b|算子|归约核",
    "framework": r"\b(vllm|sglang|megatron|pytorch|infinitrain|infinilm|infinitensor|autograd|kv cache|gradient accumulation)\b|训练框架|推理框架|梯度累积",
    "system": r"\b(sqlite|concurrency|deadlock|transaction|thread pool|state machine|resource lifecycle)\b|并发|事务|死锁|状态机|资源回收",
}
INTENT_RULES = {
    "debug": r"\b(fix|debug|bug|incorrect|crash|leak|failing|failure|regression)\b|修复|调试|泄漏|报错|回退",
    "optimize": r"\b(optimize|optimise|speedup|throughput|latency|bottleneck)\b|加速|性能优化|降低延迟|提升吞吐",
    "implement": r"\b(implement|add|build|create)\b|实现|新增|编写",
    "refactor": r"\b(refactor|restructure)\b|重构",
    "documentation": r"\b(readme|documentation|docstring|typo)\b|文档|注释|错别字",
}
STRUCTURE_RULES = {
    "cross_module": r"\b(cross.files?|cross.modules?|across modules|multiple modules|call graph|data flow)\b|跨文件|跨模块|调用链|数据流",
    "state_lifecycle": r"\b(ownership|lifecycle|checkpoint|synchronization|synchronisation|kv cache)\b|生命周期|所有权|检查点|分布式同步",
}
PRIMITIVE_RULES = {
    "scan": r"\b(scan|cumsum|prefix sum)\b|前缀和|扫描",
    "reduction": r"\b(reduction|reduce|mean|variance|rmsnorm|rms.?norm|layernorm|softmax)\b|归约|均值|方差",
    "contraction": r"\b(gemm|matmul|einsum|matrix multiplication)\b|矩阵乘|张量收缩",
}


def extract(spec, inventory, framework_facts=(), *, context_limit, allow_commands=None):
    text = spec.task[:24000]
    task_sha = hashlib.sha256(spec.task.encode()).hexdigest()
    evidence = []
    axes = {}

    def matches(axis, rules):
        values = []
        for label, pattern in rules.items():
            found = list(re.finditer(pattern, text, re.I))[:4]
            if found:
                values.append(label)
                for match in found:
                    evidence.append(
                        dict(
                            axis=axis,
                            label=label,
                            kind="task_text_match",
                            task_sha256=task_sha,
                            span=list(match.span()),
                            text=match.group(),
                        )
                    )
        axes[axis] = values

    matches("domain", DOMAIN_RULES)
    matches("intent", INTENT_RULES)
    matches("structure", STRUCTURE_RULES)
    matches("primitive", PRIMITIVE_RULES)
    # A documentation-only request can mention several technologies without
    # asking to execute any of their development workflows.
    code_goal = re.search(
        r"\b(bug|crash|incorrect|leak|failing|regression)\b|崩溃|错误行为|泄漏|回退", text, re.I
    )
    substantive = set(axes["intent"]) & {"implement", "optimize", "refactor"}
    doc_target = re.search(
        r"\b(?:implement|add|optimize|optimise|refactor)\s+(?:the\s+)?(?:readme|documentation|docstrings?)\b|编写文档|新增文档",
        text,
        re.I,
    )
    docs_only = (
        "documentation" in axes["intent"] and not code_goal and (not substantive or doc_target)
    )
    defaults = []
    if docs_only:
        axes["domain"] = ["general"]
        axes["primitive"] = []
        axes["intent"] = ["documentation"]
    else:
        if "operator" in spec.capabilities:
            axes["domain"].append("operator")
            evidence.append(
                dict(
                    axis="domain", label="operator", kind="caller_capability", field="capabilities"
                )
            )
        if framework_facts:
            axes["domain"].append("framework")
            evidence.extend(
                dict(axis="domain", label="framework", kind="bounded_repository_metadata", fact=f)
                for f in framework_facts
            )
        if not axes["domain"]:
            axes["domain"] = ["general"]
            defaults.append("general domain fallback;specialty unknown")
        if not axes["intent"]:
            axes["intent"] = ["implement"]
            defaults.append("implementation fallback;intent not explicitly matched")
    # Filenames establish language candidates, never call/data dependencies.
    suffixes = {Path(p.rstrip("/")).suffix.lower() for p in spec.allowed_paths}
    if not suffixes or suffixes <= {""}:
        suffixes = set(inventory["suffix_counts"])
    axes["language"] = sorted(
        {
            ".py": "python",
            ".cu": "cuda",
            ".cuh": "cuda",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".c": "c",
            ".rs": "rust",
            ".go": "go",
            ".js": "javascript",
            ".ts": "typescript",
        }[suffix]
        for suffix in suffixes
        if suffix in {".py", ".cu", ".cuh", ".cpp", ".cc", ".c", ".rs", ".go", ".js", ".ts"}
    )
    evidence.append(
        dict(
            axis="language",
            kind="filename_metadata",
            suffixes=sorted(suffixes),
            interpretation="language candidates only;not source analysis or a modification plan",
        )
    )
    axes["context"] = ["short_context" if context_limit <= 8192 else "long_context"]
    if allow_commands is not None:
        axes["commands"] = ["permitted" if allow_commands else "unavailable"]
    from infinicoder.analysis.tensor_dataflow import declarations

    shapes, origins, shape_limits = declarations(spec, {})
    axes["inputs"] = ["declared_tensor_shapes"] if shapes else []
    for name, origin in origins.items():
        evidence.append(
            dict(
                axis="inputs",
                label="declared_tensor_shapes",
                input=name,
                origin=origin,
                interpretation="caller declaration;not observed runtime shape",
            )
        )
    axes = {axis: sorted(set(values)) for axis, values in axes.items()}
    unknowns = []
    unknowns.extend(defaults)
    if not axes["structure"]:
        unknowns.append(
            "semantic dependency structure unknown;write permissions and file count are insufficient"
        )
    if not shapes:
        unknowns.append("no unambiguous current caller tensor shapes")
    if inventory["scan_truncated"]:
        unknowns.append("repository filename inventory is incomplete")
    return dict(
        version=1,
        axes=axes,
        evidence=evidence,
        unknowns=unknowns,
        tensor_declaration_limits=shape_limits,
        task_sha256=task_sha,
        status="uncalibrated_multi_label_hypotheses",
        probabilities=None,
        model_calls=0,
        semantic_correctness_established=False,
    )
