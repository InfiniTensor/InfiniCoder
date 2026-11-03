"""Canonical task contract boundary."""

from dataclasses import dataclass, field
from pathlib import Path


def _validate_string_list(value, field, allow_empty=False):
    if not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value):
        raise ValueError(f"{field} must be a list of non-empty strings")
    if not allow_empty and not value:
        raise ValueError(f"{field} must not be empty")
    return value


def _validate_edit_scope(path, field):
    # Scope matching is exact or a trailing-slash directory prefix, not glob.
    # Never silently widen a caller's pattern into a directory permission.
    if "*" in path or "?" in path:
        raise ValueError(
            f"{field}: wildcard paths (* or ?) are not supported; "
            "use an exact workspace-relative file or a directory prefix ending in /"
        )


def validate_task_spec(spec):
    """strict contract validation before any model/process runs."""
    if not isinstance(spec.task, str) or not spec.task.strip():
        raise ValueError("task: must be a non-empty string")
    if not isinstance(spec.checks, list):
        raise ValueError("checks: must be a list")
    seen = set()
    for i, c in enumerate(spec.checks):
        where = f"checks[{i}]"
        if not isinstance(c.id, str) or not c.id:
            raise ValueError(f"{where}.id: must be a non-empty string")
        if c.id in seen:
            raise ValueError(f"{where}.id: duplicate check id {c.id!r}")
        seen.add(c.id)
        # N03 core: argv must be a LIST of strings; a bare string like "python"
        # iterates into characters and must be rejected, not accepted.
        if not isinstance(c.argv, list):
            raise ValueError(
                f"{where}.argv: must be a list of strings, got {type(c.argv).__name__}: {c.argv!r}"
            )
        if not c.argv:
            raise ValueError(f"{where}.argv: must not be empty")
        if any(not isinstance(x, str) for x in c.argv):
            raise ValueError(
                f"{where}.argv: every element must be a string (empty strings are legal arguments)"
            )
        if (
            isinstance(c.timeout, bool)
            or not isinstance(c.timeout, (int, float))
            or not float(c.timeout) > 0
            or c.timeout != c.timeout
            or c.timeout == float("inf")
        ):
            raise ValueError(f"{where}.timeout: must be a finite positive number")
    _validate_string_list(spec.allowed_paths, "allowed_paths", allow_empty=True)
    for path in spec.allowed_paths:
        _validate_edit_scope(path, "allowed_paths")
    _validate_string_list(spec.capabilities, "capabilities", allow_empty=True)
    _validate_string_list(spec.required_artifacts, "required_artifacts", allow_empty=True)
    for path in spec.allowed_paths + spec.required_artifacts:
        raw = Path(path)
        if (
            raw.is_absolute()
            or ".." in raw.parts
            or not raw.parts
            or raw.parts[0] in (".git", ".infinicoder")
        ):
            raise ValueError("Contract paths must be safe workspace-relative paths")
    for check in spec.checks:
        _validate_string_list(check.output_paths, "check.output_paths", allow_empty=True)
        _validate_string_list(check.protected_paths, "check.protected_paths", allow_empty=True)
        for path in check.output_paths:
            raw = Path(path)
            if (
                raw.is_absolute()
                or ".." in raw.parts
                or not raw.parts
                or path in (".", "./")
                or raw.parts[0].startswith(".")
            ):
                raise ValueError("Check output_paths must be explicit relative paths")
        if check.report_path is not None:
            path = check.report_path
            if (
                not isinstance(path, str)
                or not path
                or Path(path).is_absolute()
                or ".." in Path(path).parts
            ):
                raise ValueError("Check report_path must be a safe relative file path")
            if not any(path.startswith(p.rstrip("/") + "/") for p in check.output_paths):
                raise ValueError("Check report_path must be inside a declared output directory")
    if not isinstance(spec.target_facts, dict):
        raise ValueError("target_facts: must be an object")
    from infinicoder.domains.operator.tensor_layout import derive_layouts

    derive_layouts({"tensors": spec.tensor_layouts})
    if not isinstance(spec.milestones, list):
        raise ValueError("milestones: must be a list")
    if not isinstance(spec.amendments, list):
        raise ValueError("amendments: must be a list")
    for i, a in enumerate(spec.amendments):
        where = f"amendments[{i}]"
        if not isinstance(a, dict):
            raise ValueError(f"{where}: must be an object")
        if not isinstance(a.get("source"), str) or not a.get("source", "").strip():
            raise ValueError(f"{where}.source: must be a non-empty string")
        if not isinstance(a.get("description"), str) or not a.get("description", "").strip():
            raise ValueError(f"{where}.description: must be a non-empty string")
    for i, m in enumerate(spec.milestones):
        where = f"milestones[{i}]"
        if not isinstance(m, dict):
            raise ValueError(f"{where}: must be an object")
        if set(m) - {"goal", "check_ids", "write_set", "worker_path", "provisional"}:
            raise ValueError(f"{where}: unknown fields")
        if "worker_path" in m:
            path = m["worker_path"]
            if (
                not isinstance(path, str)
                or not path
                or Path(path).is_absolute()
                or ".." in Path(path).parts
            ):
                raise ValueError("Milestone worker_path must be a safe relative path")
            scope = m.get("write_set", spec.allowed_paths)
            if scope and not any(
                path == a.rstrip("/") or a.endswith("/") and path.startswith(a) for a in scope
            ):
                raise ValueError("Milestone worker_path must be in its write_set")
        if "write_set" in m:
            _validate_string_list(m["write_set"], f"{where}.write_set")
            for path in m["write_set"]:
                _validate_edit_scope(path, f"{where}.write_set")
                raw = Path(path)
                if raw.is_absolute() or ".." in raw.parts or not raw.parts:
                    raise ValueError("Milestone write_set must be workspace-relative")
                if spec.allowed_paths and not any(
                    path == a.rstrip("/") or a.endswith("/") and path.startswith(a)
                    for a in spec.allowed_paths
                ):
                    raise ValueError("Milestone write_set must be within task allowed_paths")
        if not isinstance(m.get("goal"), str) or not m.get("goal", "").strip():
            raise ValueError(f"{where}.goal: must be a non-empty string")
        ids = m.get("check_ids")
        if not isinstance(ids, list):
            raise ValueError(f"{where}.check_ids: must be a list")
        if type(m.get("provisional", False)) is not bool:
            raise ValueError("provisional must be boolean")
        if m.get("provisional") and i == len(spec.milestones) - 1:
            raise ValueError("final phase cannot be provisional")
        # Callback-only contracts have no named check IDs. The runtime still
        # refuses them without an independent verifier and invokes it at final.
        _validate_string_list(
            ids, f"{where}.check_ids", allow_empty=m.get("provisional", False) or not spec.checks
        )
        unknown = set(ids) - seen
        if unknown:
            raise ValueError(f"{where}.check_ids: unknown check ids {sorted(unknown)}")


@dataclass
class Check:
    id: str
    argv: list
    timeout: float = 120
    output_paths: list = field(default_factory=list)
    protected_paths: list = field(default_factory=list)
    report_path: str = None


@dataclass
class TaskSpec:
    task: str
    checks: list = field(default_factory=list)
    allowed_paths: list = field(default_factory=list)
    capabilities: list = field(default_factory=list)
    target_facts: dict = field(default_factory=dict)
    required_artifacts: list = field(default_factory=list)
    milestones: list = field(default_factory=list)
    amendments: list = field(default_factory=list)
    tensor_layouts: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError(f"TaskSpec must be a mapping, got {type(value).__name__}")
        unknown = set(value) - {
            "task",
            "checks",
            "allowed_paths",
            "capabilities",
            "target_facts",
            "required_artifacts",
            "milestones",
            "amendments",
            "tensor_layouts",
        }
        if unknown:
            raise ValueError(f"TaskSpec unknown fields: {sorted(unknown)}")
        spec = cls(**value)
        spec.checks = [Check(**c) if isinstance(c, dict) else c for c in spec.checks]
        validate_task_spec(spec)
        return spec
