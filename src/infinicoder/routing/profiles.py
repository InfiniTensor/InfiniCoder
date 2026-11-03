"""Task routing based on declared semantics and bounded repository metadata."""

from dataclasses import dataclass, asdict
from pathlib import Path
import ast
import hashlib
import json
import re

try:
    import tomllib
except ImportError:  # Python3.10 can use an installed compatible TOML parser.
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


@dataclass(frozen=True)
class Route:
    profile: str
    reasons: tuple
    cards: tuple = ()
    needs_plan: bool = False
    source_path: str = None

    def to_dict(self):
        return asdict(self)


_FRAMEWORKS = {
    "torch",
    "pytorch",
    "vllm",
    "sglang",
    "megatron-core",
    "megatron-lm",
    "infinitensor",
    "infinilm",
    "infinitrain",
}
_NATIVE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".cu", ".h", ".hpp", ".cuh"}


def repository_route_facts(root, allowed_paths):
    """Read bounded build metadata as data,without executing build commands.

    Inspect the root and ancestors of explicit scoped paths,never dependencies
    or a recursive tree scan. Package identity alone can select framework;
    having a torch dependency alone cannot. Ambiguous metadata stays unknown.
    """
    root = Path(root).resolve()
    folders = {root}
    explicit = [p for p in allowed_paths if not any(c in p for c in "*?[")]
    for name in explicit[:8]:
        try:
            path = (root / name).resolve()
        except (OSError, RuntimeError):
            continue
        if not path.is_relative_to(root):
            continue
        parent = path if path.is_dir() else path.parent
        for _ in range(8):
            if not parent.is_relative_to(root):
                break
            folders.add(parent)
            if parent == root:
                break
            parent = parent.parent
    facts = []
    frameworks = set()
    native = []
    remaining = 1048576

    def source(path, offset, kind, value, digest):
        return "{}:{} sha256={} {}={}".format(
            path.relative_to(root).as_posix(), offset, digest, kind, value
        )

    for folder in sorted(folders):
        for name in ("pyproject.toml", "setup.py", "CMakeLists.txt", "compile_commands.json"):
            path = folder / name
            try:
                if (
                    not path.resolve().is_relative_to(root)
                    or not path.is_file()
                    or path.stat().st_size > min(262144, remaining)
                ):
                    continue
                blob = path.read_bytes()
                remaining -= len(blob)
                if len(blob) > 262144 or remaining < 0:
                    continue
                text = blob.decode("utf-8")
                digest = hashlib.sha256(blob).hexdigest()
                identities = []
                if name == "pyproject.toml":
                    if tomllib is None:
                        continue
                    data = tomllib.loads(text)
                    for table in [data.get("project", {}), data.get("tool", {}).get("poetry", {})]:
                        if isinstance(table, dict) and isinstance(table.get("name"), str):
                            identities.append((table["name"], 1))
                elif name == "setup.py":
                    tree = ast.parse(text)
                    for node in tree.body:
                        if (
                            isinstance(node, ast.Expr)
                            and isinstance(node.value, ast.Call)
                            and ast.unparse(node.value.func) in ("setup", "setuptools.setup")
                        ):
                            identities.extend(
                                (k.value.value, k.value.lineno)
                                for k in node.value.keywords
                                if k.arg == "name"
                                and isinstance(k.value, ast.Constant)
                                and isinstance(k.value.value, str)
                            )
                elif name == "CMakeLists.txt":
                    blank = lambda m: "".join("\n" if c == "\n" else " " for c in m.group())
                    cleaned = re.sub(r"\[(=*)\[.*?\]\1\]", blank, text, flags=re.S)
                    # Quoted/interpolated CMake identities remain unknown.
                    # Removing strings also avoids declarations inside messages.
                    cleaned = re.sub(r'"(?:\\.|[^"\\])*"', blank, cleaned, flags=re.S)
                    cleaned = re.sub(r"(?m)#.*$", "", cleaned)
                    identities = [
                        (m.group(1), cleaned[: m.start()].count("\n") + 1)
                        for m in re.finditer(
                            r"(?im)^\s*project\s*\(\s*([A-Za-z0-9_.-]+)(?=\s|\))", cleaned
                        )
                    ]
                    native.append(source(path, 1, "build", "cmake", digest))
                else:
                    commands = json.loads(text)
                    if isinstance(commands, list):
                        languages = sorted(
                            {
                                Path(row.get("file", "")).suffix.lower()
                                for row in commands[:128]
                                if isinstance(row, dict) and isinstance(row.get("file"), str)
                            }
                            & _NATIVE_SUFFIXES
                        )
                        if languages:
                            native.append(
                                source(path, 1, "compiled_suffixes", ",".join(languages), digest)
                            )
                normalized = {re.sub(r"[-_.]+", "-", name).lower() for name, _ in identities}
                # Conflicting identities in the same file do not choose a domain.
                if len(normalized) == 1 and normalized <= _FRAMEWORKS:
                    identity, line = identities[0]
                    frameworks.add(next(iter(normalized)))
                    facts.append(source(path, line, "framework", identity, digest))
            except (
                OSError,
                RuntimeError,
                UnicodeError,
                ValueError,
                TypeError,
                AttributeError,
                SyntaxError,
                RecursionError,
            ):
                continue
    # Multiple independent package identities in scope remain ambiguous.
    framework = facts if len(frameworks) == 1 else []
    scoped_native = any(Path(p.rstrip("/")).suffix.lower() in _NATIVE_SUFFIXES for p in explicit)
    broad = not allowed_paths or any(
        any(c in p for c in "*?[") or p.endswith("/") or (root / p).is_dir() for p in allowed_paths
    )
    return tuple(framework), tuple(native if scoped_native or broad else ()), scoped_native


def route_task(spec, workspace, profile="auto"):
    if profile not in ("auto", "general", "operator", "framework", "legacy"):
        raise ValueError("unknown runtime profile: " + str(profile))
    text = spec.task.lower()
    root = Path(workspace)
    reasons = []
    automatic = profile == "auto"
    framework_facts = ()
    native_facts = ()
    scoped_native = False
    if profile == "auto":
        framework_facts, native_facts, scoped_native = repository_route_facts(
            root, spec.allowed_paths
        )
        if "operator" in spec.capabilities or re.search(
            r"\b(triton|cuda kernel|rmsnorm|layernorm|gemm|softmax)\b|算子", text
        ):
            profile = "operator"
            reasons.append("operator semantics or declared capability")
        elif re.search(
            r"\b(vllm|sglang|megatron|pytorch|infinitrain|infinilm|infinitensor|distributed|scheduler|autograd|kv cache|training framework|inference framework)\b|训练框架|推理框架|分布式",
            text,
        ):
            profile = "framework"
            reasons.append("framework or lifecycle semantics")
        elif framework_facts:
            profile = "framework"
            reasons.extend(framework_facts)
        else:
            profile = "general"
            reasons.append("general coding contract")
    else:
        reasons.append("caller-selected profile")
    explicit = [
        p for p in spec.allowed_paths if not p.endswith("/") and not any(c in p for c in "*?[")
    ]
    single = explicit[0] if len(explicit) == len(spec.allowed_paths) == 1 else None
    if single and (not (root / single).is_file() or (root / single).stat().st_size > 24000):
        single = None
    if (
        single
        and profile == "framework"
        and Path(single).suffix in (".cc", ".cpp", ".cxx", ".cu", ".h", ".hpp", ".cuh")
    ):
        single = None
        reasons.append("native framework changes use source inspection and focused edits")
    complex_task = (
        len(explicit) > 1
        or not spec.allowed_paths
        or any(
            any(c in p for c in "*?[") or p.endswith("/") or (root / p).is_dir()
            for p in spec.allowed_paths
        )
    )
    cards = {
        "operator": ("operator", "triton")
        if re.search(r"\btriton\b", text)
        else ("operator", "native"),
        "framework": ("system", "framework"),
        "general": (),
        "legacy": (),
    }[profile]
    if profile == "framework" and re.search(r"\b(adamw?|sgd|optimizer|momentum)\b|优化器", text):
        cards += ("optimizer",)
    if automatic and (scoped_native or native_facts):
        if "native" not in cards:
            cards += ("native",)
        if scoped_native:
            reasons.append("explicit modification scope contains native source suffixes")
        reasons.extend(native_facts)
    return Route(
        profile,
        tuple(reasons),
        cards,
        complex_task and not spec.milestones and profile != "legacy",
        single,
    )
