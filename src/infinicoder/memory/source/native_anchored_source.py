"""Task-anchored Native context;ordinary SDK loop and shell remain unchanged."""

from contextlib import contextmanager
import json
from pathlib import Path

from infinicoder.memory.source.anchored_source_workset import build_anchored_workset
from infinicoder.tools.native_shell_workbench import NativeShellCodeWorkbench


class NativeAnchoredSourceWorkset(NativeShellCodeWorkbench):
    identity = "native_anchored_source_v1"

    def __init__(
        self,
        *args,
        source_dependency_policy=None,
        workspace_aliases=(),
        body_write_scopes=(),
        **kwargs,
    ):
        if source_dependency_policy not in (None, "anchored_static_v1"):
            raise ValueError("unknown source dependency policy")
        if any(
            not isinstance(p, str)
            or not p.startswith("/")
            or p.endswith("/")
            or ".." in p.split("/")
            for p in workspace_aliases
        ):
            raise ValueError("workspace aliases must be explicit absolute container roots")
        self.source_dependency_policy = source_dependency_policy
        self.workspace_aliases = tuple(workspace_aliases)
        self.body_write_scopes = tuple(body_write_scopes)
        super().__init__(*args, **kwargs)

    def validate(self, request):
        super().validate(request)
        from infinicoder.memory.source.source_workset import public_target_present

        for target in self.body_write_scopes:
            if (
                not isinstance(target, str)
                or "::" not in target
                or target.split("::", 1)[0] not in self.paths
                or not public_target_present(request.spec.task, target)
            ):
                raise ValueError(
                    "body write scope must be caller-explicit,public and in the declared implementation files"
                )

    def _source_policy_identity(self):
        return ("public_roots_v1", self.source_dependency_policy, self.workspace_aliases)

    @contextmanager
    def bind(self, request, model, environment, deadline):
        try:
            with super().bind(request, model, environment, deadline):
                from infinicoder.domains.operator.candidate_observation import HistoricalSources

                forbidden = list(self.excluded_paths)
                for check in request.spec.checks:
                    for name in check.protected_paths:
                        p = Path(name)
                        p = p if p.is_absolute() else self.workspace / p
                        if p.resolve().is_relative_to(self.workspace):
                            forbidden.append(p.resolve().relative_to(self.workspace).as_posix())
                self.diagnostic_forbidden = tuple(forbidden)
                self.historical_sources = HistoricalSources(
                    self.state / "historical-source",
                    self.workbench._read,
                    self.paths,
                    forbidden=forbidden,
                    remaining=self._remaining,
                )
                yield self
        finally:
            path = Path(request.state_dir) / "native-memory-info.json"
            if path.is_file():
                value = json.loads(path.read_bytes())
                value.update(
                    source_policy="public_roots_v1",
                    source_dependency_policy=self.source_dependency_policy,
                    workspace_aliases=self.workspace_aliases,
                    historical_sources=str(
                        Path(request.state_dir) / "historical-source/manifest.json"
                    ),
                    source_facts="Automatic current public roots and optional bounded static dependency candidates;no runtime binding or correctness proof",
                )
                path.write_text(
                    json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf8"
                )

    def _build_source_packet(self, budget):
        return build_anchored_workset(
            self.index,
            self.task,
            self.public_targets,
            budget,
            lambda text: self._count([dict(role="user", content=text)]),
            dependency_policy=self.source_dependency_policy,
            workspace_aliases=self.workspace_aliases,
            remaining=self._remaining,
        )
