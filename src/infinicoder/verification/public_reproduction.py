"""An explicit common public bootstrap, separate from optional dynamic traces."""

from dataclasses import dataclass
from pathlib import Path

from infinicoder.verification.public_replay import PublicReplayCapability


def validate_public_input_sources(sources, final_script):
    """Path metadata only;never expose the independent final as public input."""
    if sources is not None and type(sources) is not dict:
        raise ValueError("public reproduction sources must be an explicit path mapping")
    if sources:
        if final_script is None:
            raise ValueError(
                "explicit final script identity required before binding public sources"
            )
        if any(Path(v).resolve() == Path(final_script).resolve() for v in sources.values()):
            raise ValueError("independent final script cannot be a public reproduction input")


@dataclass(frozen=True)
class PublicDevelopmentReproduction:
    ordinary: PublicReplayCapability
    diagnostic: PublicReplayCapability

    def __post_init__(self):
        if not all(isinstance(x, PublicReplayCapability) for x in (self.ordinary, self.diagnostic)):
            raise ValueError("two explicit caller-owned public reproduction capabilities required")
        if (
            self.ordinary.spec != self.diagnostic.spec
            or self.ordinary.model_root != self.diagnostic.model_root
            or Path(self.ordinary.operation.workspace).resolve()
            != Path(self.diagnostic.operation.workspace).resolve()
        ):
            raise ValueError(
                "ordinary and instrumented public reproduction must share inputs and workspace"
            )
        for cap, expected in ((self.ordinary, "none"), (self.diagnostic, "bounded_python_frames")):
            if cap.operation.environment_identity.get("instrumentation") != expected:
                raise ValueError("ordinary and diagnostic roles must declare their instrumentation")


def cpu_public_reproduction(spec, workspace, state_dir, execution, *, public_sources=None):
    from infinicoder.verification.public_replay_cpu import cpu_public_replay

    mapping = attach_cpu_public_sources(execution, spec, workspace, public_sources)
    return PublicDevelopmentReproduction(
        cpu_public_replay(
            spec,
            workspace,
            state_dir,
            execution,
            trace_enabled=False,
            independent_reproduction=True,
            readonly_sources=mapping,
        ),
        cpu_public_replay(
            spec,
            workspace,
            state_dir,
            execution,
            trace_enabled=True,
            independent_reproduction=True,
            readonly_sources=mapping,
        ),
    )


def public_source_directory(spec, workspace, sources):
    """One flat immutable attachment directory,not exported candidate files."""
    from infinicoder.verification.public_replay import PublicReplaySpec, digest

    if isinstance(spec, dict):
        spec = PublicReplaySpec(**spec)
    if not sources:
        return None, {}
    if type(sources) is not dict or not set(sources) <= set(spec.public_input_hashes):
        raise ValueError("public attachments require explicit input hashes")
    roots = set()
    mapping = {}
    for relative, value in sources.items():
        path = Path(value).absolute()
        if relative != "__ic_public__/" + path.name or any(
            p.is_symlink() for p in (path, *path.parents)
        ):
            raise ValueError(
                "public attachment names must preserve a regular source filename under __ic_public__"
            )
        if (
            not path.is_file()
            or path.stat().st_size > 1024**2
            or digest(path.read_bytes()) != spec.public_input_hashes[relative]
        ):
            raise ValueError("public attachment differs from its frozen source")
        roots.add(path.parent.resolve())
        mapping[relative] = path
    if len(roots) != 1:
        raise ValueError("one separate public attachment directory required")
    source = roots.pop()
    workspace = Path(workspace).resolve()
    if source.is_relative_to(workspace) or workspace.is_relative_to(source):
        raise ValueError("public sources must be outside candidate workspace")
    if any(p.is_symlink() or not p.is_file() for p in source.iterdir()):
        raise ValueError("flat public source directory required")
    if {p.resolve() for p in source.iterdir()} != {p.resolve() for p in mapping.values()}:
        raise ValueError("public mount must contain exactly the declared public inputs")
    target = workspace / "__ic_public__"
    if target.is_symlink():
        raise ValueError("public mount target cannot be a symlink")
    target.mkdir(exist_ok=True)
    if any(target.iterdir()):
        raise ValueError("public mount target must be empty;do not hide candidate files")
    return source, mapping


def attach_cpu_public_sources(execution, spec, workspace, sources):
    source, mapping = public_source_directory(spec, workspace, sources)
    if source is None:
        return mapping
    base = execution.sandbox_argv

    def builder(actual_workspace, command, *args):
        argv = list(base(actual_workspace, command, *args))
        argv[-len(command) : -len(command)] = ["--ro-bind", str(source), "/workspace/__ic_public__"]
        return argv

    execution.sandbox_argv = builder
    return mapping
