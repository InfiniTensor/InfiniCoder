"""Evidence-based history selection independent of optional shell tool bindings."""

from infinicoder.memory.execution_evidence_selection import select_history
from infinicoder.memory.source.native_source_workset import NativeSourceWorkset


class EvidenceSelectionMixin:
    """Change only the optional whole-history decision of a source workset."""

    def project(self, messages):
        self.selection_audit = None
        return super().project(messages)

    def _select_history(self, prepared, mandatory, count):
        selected, self.selection_audit = select_history(
            prepared,
            self._history_blocks(prepared)[1],
            mandatory,
            self.executions,
            checks=self.checks,
            views=self.events,
            count=count,
            limit=self.limit,
        )
        return selected

    def _selection_report(self):
        return self.selection_audit


class NativeEvidenceSourceWorkset(EvidenceSelectionMixin, NativeSourceWorkset):
    """Explicit tool-free profile;does not adapt or change the execution layer.

    The caller must independently supply an admitted execution adapter. No shell
    worker,client mount,tool advertisement or automatic fallback is installed.
    Empty public implementation paths remain unknown rather than a source closure.
    """

    identity = "native_distinct_evidence_source_memory_v1;original_whole_blocks;no_shell_tools"
