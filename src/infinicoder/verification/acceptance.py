"""Acceptance executor: frozen source, checker identity and full receipts.

Independent from model routing/context. The process callback owns resource
accounting; this executor owns acceptance integrity and snapshot cleanup.
"""

import hashlib
import json
import time
from pathlib import Path
from infinicoder.models.budget import BudgetExceeded
from infinicoder.verification.check_identity import checker_files
from infinicoder.execution.workspace_tools import Workspace
from infinicoder.planning.diagnostics import compiler_diagnostics
from infinicoder.verification.verification import (
    Snapshot,
    environment_id,
    outcome_acceptable,
    parse_test_summary,
)


def _stdout_report(text):
    """Retain a bounded JSON observation without changing its acceptance gate."""
    if not isinstance(text, str) or len(text.encode("utf-8")) > 65536:
        return None
    try:
        report = json.loads(text)
    except (ValueError, TypeError):
        return None
    if isinstance(report, dict) and report.get("status") in ("passed", "failed"):
        return report
    return None


class VerificationExecutionError(Exception):
    """The trusted process callback raised; this is not a candidate verdict."""

    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__("Independent verification executor did not return a result")


class VerificationBudgetError(VerificationExecutionError):
    """A controller budget exception, kept separate from environment faults."""


class VerificationControlStop(VerificationExecutionError):
    def __init__(self, control, evidence):
        self.control = control
        super().__init__(evidence)


class AcceptanceRunner:
    def __init__(
        self,
        state_dir,
        store,
        state,
        process,
        execution_profile,
        verifier=None,
        diagnostic_namespaces=None,
    ):
        self.state_dir, self.store, self.state = state_dir, store, state
        self.process, self.execution_profile, self.verifier = process, execution_profile, verifier
        from infinicoder.planning.diagnostic_namespaces import normalize

        self.diagnostic_namespaces = normalize(diagnostic_namespaces)
        self._snapmarks = []
        # Only the synchronous process invocation may consume this lease.
        # Historical snapshot names are not execution permissions.
        self._active_execution = None

    def matches_active_execution(self, argv, cwd, *, check_id, check_argv):
        """A trusted adapter may admit exactly the currently invoked check.

        This grants no model tool or workspace permission. A prior snapshot,
        matching directory prefix, or live-workspace command is insufficient.
        This matcher currently supports commands without a {report} placeholder.
        """
        lease = self._active_execution
        if not lease or lease["check_id"] != check_id or lease["check_argv"] != list(check_argv):
            return False
        expected = self.state_dir / "verify" / ("snap-" + str(lease["snapshot_sequence"]))
        snapshot = Path(lease["snapshot"])
        try:
            if (
                snapshot != expected.absolute()
                or not snapshot.is_dir()
                or expected.resolve() != expected.absolute()
                or expected.parent.resolve() != expected.parent.absolute()
                or self.state_dir.resolve() != self.state_dir.absolute()
                or str(cwd) != lease["cwd"]
                or Path(cwd).absolute() != snapshot
            ):
                return False
            expanded = [str(snapshot) if a == "{workspace}" else a for a in check_argv]
            if list(argv) != lease["argv"] or list(argv) != expanded:
                return False
            if lease["checker_files"] != self.state.get("checker_files", {}):
                return False
            for name, digest in lease["checker_files"].items():
                path = Path(name)
                if (
                    not path.is_file()
                    or path.is_symlink()
                    or path.resolve() != path.absolute()
                    or hashlib.sha256(path.read_bytes()).hexdigest() != digest
                ):
                    return False
            return Workspace(snapshot).revision() == lease["execution_revision"]
        except (OSError, ValueError):
            return False

    def active_execution_timeout(self, argv, cwd, timeout, *, check_id, check_argv):
        """Recheck control and the original deadline after expensive admission."""
        if not self.matches_active_execution(argv, cwd, check_id=check_id, check_argv=check_argv):
            return None
        lease = self._active_execution
        receipt = dict(lease, executed_argv=list(argv), execution_outcome="not_started")
        control = self.store.take_control()
        if control:
            receipt["failure_origin"] = "verification_control"
            raise VerificationControlStop(control, [receipt])
        available = min(timeout, lease["deadline_monotonic"] - time.monotonic())
        if available <= 0:
            receipt.update(
                failure_origin="verification_budget",
                error_type="BudgetExceeded",
                error="Original verification deadline expired during binding",
            )
            raise VerificationBudgetError([receipt])
        return available

    def verify(self, spec, ws, check_ids, remaining, require_artifacts=True):
        """frozen-snapshot acceptance with structured reports.

        Checks run against a snapshot copy; live-workspace edits during
        verification cannot change the outcome, and check side effects
        cannot touch the user's files. Evidence binds snapshot revision,
        checker identity and environment."""
        from infinicoder.planning.diagnostic_namespaces import validate_checks

        validate_checks(self.diagnostic_namespaces, spec)
        self.last_process_outcomes = {}
        live_before = ws.revision()
        if remaining() <= 0:
            return False, [{"check_id": "deadline", "error": "No verification budget remains"}]
        if checker_files(spec, ws) != self.state.get("checker_files", {}):
            return False, [
                {
                    "check_id": "checker_identity",
                    "error": "Acceptance files changed since session start",
                }
            ]
        snap_seq = len(self._snapmarks)
        snap_dir = self.state_dir / "verify" / f"snap-{snap_seq}"
        output_paths = list(
            dict.fromkeys(p for c in spec.checks if c.id in check_ids for p in c.output_paths)
        )
        snapshot = Snapshot(ws.root, snap_dir, excluded_paths=output_paths)
        self._snapmarks.append(str(snap_dir))
        snap_rev = snapshot.revision()
        protected_revision = snapshot.revision(output_paths)
        env = environment_id()
        evidence = []
        try:
            for check in spec.checks:
                if check.id not in check_ids:
                    continue
                report_file = (
                    Workspace(snapshot.dir).path(check.report_path) if check.report_path else None
                )
                if report_file:
                    if report_file.exists():
                        report_file.unlink()  # A source-supplied/stale report is never evidence.
                    report_file.parent.mkdir(parents=True, exist_ok=True)
                executed_argv = [
                    str(snapshot.dir) if arg == "{workspace}" else arg for arg in check.argv
                ]
                executed_argv = [
                    str(report_file) if arg == "{report}" and report_file else arg
                    for arg in executed_argv
                ]
                timeout = min(check.timeout, self.execution_profile.max_timeout, remaining())
                if timeout <= 0:
                    return False, evidence + [
                        {
                            "check_id": "deadline",
                            "error": "No verification budget remains",
                            "execution_outcome": "not_started",
                        }
                    ]
                lease = dict(
                    check_id=check.id,
                    snapshot=str(snapshot.dir.resolve()),
                    cwd=str(snapshot.dir),
                    argv=list(executed_argv),
                    check_argv=list(check.argv),
                    checker_files=dict(self.state.get("checker_files", {})),
                    revision=snap_rev,
                    execution_revision=snapshot.revision(),
                    snapshot_sequence=snap_seq,
                )
                receipt = dict(
                    lease,
                    failure_origin="verification_executor",
                    execution_outcome="unknown",
                    timeout_s=timeout,
                    environment=env,
                )
                control = self.store.take_control()
                if control:
                    receipt.update(
                        failure_origin="verification_control", execution_outcome="not_started"
                    )
                    raise VerificationControlStop(control, evidence + [receipt])
                timeout = min(timeout, remaining())
                if timeout <= 0:
                    return False, evidence + [
                        {
                            "check_id": "deadline",
                            "error": "No verification budget remains",
                            "execution_outcome": "not_started",
                        }
                    ]
                receipt["timeout_s"] = timeout
                lease["deadline_monotonic"] = time.monotonic() + timeout
                started = time.monotonic()
                self._active_execution = lease
                try:
                    result = self.process(executed_argv, str(snapshot.dir), timeout)
                except VerificationExecutionError as error:
                    error.evidence = evidence + error.evidence
                    raise
                except Exception as error:
                    receipt.update(
                        error_type=type(error).__name__,
                        error=str(error)[:2000],
                        wall_s=time.monotonic() - started,
                        executed_argv=list(executed_argv),
                    )
                    for name, observe in (
                        ("snapshot_after", snapshot.revision),
                        ("observed_current_revision", ws.revision),
                    ):
                        try:
                            receipt[name] = observe()
                        except Exception as observation_error:
                            receipt[name + "_unavailable"] = type(observation_error).__name__
                    error_type = (
                        VerificationBudgetError
                        if isinstance(error, BudgetExceeded)
                        else VerificationExecutionError
                    )
                    if isinstance(error, BudgetExceeded):
                        receipt["failure_origin"] = "verification_budget"
                    raise error_type(evidence + [receipt]) from error
                finally:
                    self._active_execution = None
                self.last_process_outcomes[check.id] = dict(
                    completed_normally=(
                        type(result.timed_out) is bool
                        and not result.timed_out
                        and type(result.exit_code) is int
                        and 0 <= result.exit_code < 124
                    ),
                    exit_code=result.exit_code,
                    timed_out=result.timed_out,
                    truncated=result.truncated,
                )
                control = self.store.take_control()
                if control:
                    receipt.update(
                        failure_origin="verification_control",
                        execution_outcome="completed",
                        exit_code=result.exit_code,
                        wall_s=time.monotonic() - started,
                        executed_argv=list(executed_argv),
                    )
                    raise VerificationControlStop(control, evidence + [receipt])
                report, report_error = None, None
                if report_file:
                    try:
                        if (
                            report_file.is_symlink()
                            or not report_file.is_file()
                            or report_file.stat().st_size > 65536
                        ):
                            raise ValueError("Missing, linked or oversized report")
                        report = json.loads(report_file.read_text(encoding="utf-8"))
                        if not isinstance(report, dict) or report.get("status") not in (
                            "passed",
                            "failed",
                        ):
                            raise ValueError("Report must be an object with passed/failed status")
                    except (OSError, ValueError, UnicodeError) as e:
                        report_error = str(e)
                report_text = (
                    json.dumps(report, ensure_ascii=False) if report is not None else result.output
                )
                summary = parse_test_summary(report_text)
                ok_struct, why = outcome_acceptable(summary)
                diagnostic_text = report_text
                try:
                    detail = json.loads(report_text).get("case_detail")
                    if isinstance(detail, str):
                        diagnostic_text = detail
                except (ValueError, AttributeError):
                    pass
                item = {
                    "check_id": check.id,
                    "exit_code": result.exit_code,
                    "stdout_tail": (result.output or "")[-800:],
                    "summary": summary,
                    "revision": snap_rev,
                    "checker": hashlib.sha256(json.dumps(check.argv).encode()).hexdigest()[:16],
                    "environment": env,
                    "output_paths": check.output_paths,
                    "executed_argv": executed_argv,
                    "diagnostics": compiler_diagnostics(diagnostic_text, Workspace(snapshot.dir)),
                    "snapshot_after": snapshot.revision(),
                }
                item["checker_files"] = self.state.get("checker_files", {})
                prefix = (self.diagnostic_namespaces or {}).get(check.id)
                if prefix is not None:
                    from infinicoder.planning.diagnostic_namespaces import observe

                    item["diagnostics"] = observe(
                        diagnostic_text,
                        Workspace(snapshot.dir),
                        prefix,
                        check_id=check.id,
                        argv=check.argv,
                        checker_files=item["checker_files"],
                        snapshot_revision=snap_rev,
                        protected_revision=protected_revision,
                        output_paths=output_paths,
                    )
                if report is not None:
                    item["report"] = report
                elif not report_file:
                    # Most CLI checks return the same structured report on
                    # stdout rather than through {report}. An 800-character
                    # tail can lose its failure identity/numerical witness and
                    # cease to be JSON. Preserve the bounded returned report
                    # for feedback;summary/exit-code gates above are unchanged.
                    observed = _stdout_report(result.output)
                    if observed is not None:
                        item["report"] = observed
                        item["report_origin"] = {
                            "kind": "process_output",
                            "sha256": hashlib.sha256(result.output.encode("utf-8")).hexdigest(),
                        }
                if report_error:
                    item["error"] = "Invalid structured report: " + report_error
                outputs = {}
                for folder in check.output_paths:
                    target = Workspace(snapshot.dir).path(folder)
                    paths = target.rglob("*") if target.is_dir() else [target]
                    for path in paths:
                        if path.is_file() and not path.is_symlink():
                            with path.open("rb") as stream:
                                digest = hashlib.sha256()
                                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                                    digest.update(chunk)
                            outputs[path.relative_to(snapshot.dir).as_posix()] = digest.hexdigest()
                item["output_hashes"] = outputs
                evidence.append(item)
                if result.exit_code != 0 or report_error:
                    return False, evidence
                if not ok_struct:
                    item["rejected_because"] = why
                    return False, evidence
            # checks must be read-only observers; a check that edits
            # the code under test (fake success) is rejected.
            if snapshot.revision(output_paths) != protected_revision:
                evidence.append(
                    {
                        "check_id": "revision",
                        "error": "Checks mutated the verification snapshot; "
                        "acceptance must not rewrite the code under test.",
                    }
                )
                return False, evidence
            if self.verifier:
                # Custom verifier owns its own deadline contract; CLI uses bounded Check commands.
                ok, reason = self.verifier(str(snapshot.dir))
                evidence.append(
                    {
                        "check_id": "callback",
                        "ok": bool(ok),
                        "reason": str(reason),
                        "revision": snap_rev,
                    }
                )
                if not ok:
                    return False, evidence
            if snapshot.revision(output_paths) != protected_revision:
                evidence.append(
                    {"check_id": "revision", "error": "Custom verifier mutated protected source"}
                )
                return False, evidence
            for path in spec.required_artifacts if require_artifacts else []:
                if not Workspace(snapshot.dir).path(path).is_file():
                    evidence.append({"check_id": "artifact", "missing": path})
                    return False, evidence
            if remaining() <= 0:
                evidence.append(
                    {"check_id": "deadline", "error": "Deadline expired during verification"}
                )
                return False, evidence
            if ws.revision() != live_before:
                evidence.append(
                    {"check_id": "revision", "error": "Live workspace changed during verification."}
                )
                return False, evidence
            if not evidence or not check_ids and self.verifier is None:
                return False, evidence
            self.store.event("verification", {"accepted": True, "evidence": evidence})
            return True, evidence
        finally:
            snapshot.cleanup()
