"""Caller-owned public feedback and final acceptance for the Runtime recipe."""

import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

from infinicoder.execution.native_cpu import CpuExecution, ScriptAcceptance
from infinicoder.execution.workspace_tools import Workspace


class RuntimeScriptChecks:
    """Reuse CPU snapshot checks without granting the model command execution.

    The public script can guide repairs. The independent script is executed only
    by finish(), after the model loop has ended. Both scripts are frozen before
    the first request and kept outside the model's workspace.
    """

    def __init__(self, spec, workspace, state_dir, public, final, python_environment, deadline):
        self.spec, self.workspace = spec, Path(workspace)
        self.folder = Path(state_dir) / "trusted-checks"
        self.deadline = deadline
        self.records = []
        self.finished = False
        self.source = Workspace(workspace, spec.allowed_paths)
        self.protected_revision = (
            self.source.revision(spec.allowed_paths) if spec.allowed_paths else None
        )
        self.execution = CpuExecution(
            Path(state_dir).with_name(Path(state_dir).name + "-check-sandbox"),
            python_environment=python_environment,
        )
        request = self._request(workspace, state_dir)
        self.execution.validate(request)
        self.public = ScriptAcceptance(public, self.execution, timeout=60)
        self.final = ScriptAcceptance(final, self.execution)
        for checker in (self.public, self.final):
            checker.validate(request)
        if self.public.script.resolve() == self.final.script.resolve():
            raise ValueError("public feedback and independent final require separate scripts")

    def _request(self, workspace, state):
        return SimpleNamespace(spec=self.spec, workspace=workspace, state_dir=state)

    def _run(self, checker, workspace, folder, deadline=None):
        folder.mkdir(parents=True, exist_ok=False)
        request = self._request(workspace, folder)
        with checker.public_command(request, time.monotonic() + checker.timeout):
            final = checker.verify(request, deadline=deadline)
        report = json.loads(Path(final.evidence_ref).read_bytes())
        if report["source_revision_before"] != report["source_revision_after"]:
            report["passed"] = False
            report["reason"] = "trusted check changed the source under test"
            Path(final.evidence_ref).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report, final.evidence_ref

    def verify_public(self, snapshot):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            return False, "task deadline expired before public check"
        self.public.timeout = min(60, remaining)
        report, ref = self._run(
            self.public, Path(snapshot), self.folder / f"public-{len(self.records)}", self.deadline
        )
        self.records.append(
            dict(passed=report["passed"], evidence_ref=ref, wall_s=report["wall_s"])
        )
        return report["passed"], json.dumps(
            dict(
                summary=report["summary"],
                output=report["process"]["output"],
                reason=report["reason"],
            )
        )

    def finish(self):
        if self.finished:
            raise ValueError("independent final acceptance already executed")
        self.finished = True
        before = self.source.revision(self.spec.allowed_paths) if self.spec.allowed_paths else None
        report, ref = self._run(self.final, self.workspace, self.folder / "final")
        after = self.source.revision(self.spec.allowed_paths) if self.spec.allowed_paths else None
        scope_preserved = self.protected_revision == before == after
        return dict(
            passed=report["passed"] and scope_preserved,
            checker_passed=report["passed"],
            scope_preserved=scope_preserved,
            scope_identity="Workspace.revision excluding authorized paths and documented generated metadata",
            evidence_ref=ref,
            summary=report["summary"],
        )

    def identity(self):
        return dict(
            public_sha256=hashlib.sha256(self.public.raw).hexdigest(),
            final_sha256=hashlib.sha256(self.final.raw).hexdigest(),
            python_environment=str(self.execution.runtime),
            execution=self.execution.identity,
            public_checks=len(self.records),
            public_wall_seconds=sum(r["wall_s"] for r in self.records),
            final_timeout_seconds=self.final.timeout,
        )
