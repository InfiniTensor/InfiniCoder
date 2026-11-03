"""Optional candidate experiment profile on the original source/suffix core."""

import time

from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.tools.native_shell_workbench import NativeShellCodeWorkbench


class NativeCandidateTrialWorkset(NativeShellCodeWorkbench):
    identity = "native_candidate_trial_v1;original_suffix;explicit_CPU_experiments"

    def validate(self, request):
        super().validate(request)
        if self.scratch is None or not getattr(request.execution_adapter, "snapshot_trials", False):
            raise UnsupportedCore(
                "candidate trials need explicit CPU snapshot execution and scratch mapping"
            )
        request.execution_adapter.snapshot_runner(
            request, time.monotonic() + request.budget.wall_seconds
        )

    def tool_help(self):
        return (
            super().tool_help()
            + """
Optional CPU comparison:two trials/task,including failures;shares your time budget.
Write both scripts under /tmp/task-scratch with heredocs.
  /tmp/ic-code-bin/ic-code trial --edit-script /tmp/task-scratch/edit.sh --probe-script /tmp/task-scratch/repro.sh
Edit changes only candidate;probe runs identically in base/candidate.
Use repository/self-written probes. Copies have no network,live checker or GPU.
  /tmp/ic-code-bin/ic-code trial-show ID --part result|diff|logs [--cursor CURSOR]
  /tmp/ic-code-bin/ic-code trial-apply ID
Observations are not acceptance. Apply checks saved versions,without replay.
Delete/rename/link/mode changes need ordinary edits. Original shell/checks remain.
"""
        )

    def shell_options(self, request, deadline):
        return dict(
            trial_runner=request.execution_adapter.snapshot_runner(request, deadline),
            trial_generated_paths=request.execution_adapter.trial_generated_paths,
            trial_output_limit=max(128, self.limit // 6),
        )
