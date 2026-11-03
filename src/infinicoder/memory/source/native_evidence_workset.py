"""Task memory selected by evidence source over the existing shell tools."""

from infinicoder.memory.source.native_evidence_context import EvidenceSelectionMixin
from infinicoder.tools.native_shell_workbench import NativeShellCodeWorkbench


class NativeEvidenceWorkset(EvidenceSelectionMixin, NativeShellCodeWorkbench):
    identity = "native_distinct_evidence_memory_v1;original_whole_blocks"
