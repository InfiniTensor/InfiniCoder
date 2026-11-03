"""Reuse the tested execution strategy without another SDK/journal owner."""

from contextlib import contextmanager
import copy

from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.verification.native_public_failure_slice import (
    NativePublicFailureSlice,
    validate_public_sources,
)
from infinicoder.verification.native_public_reproduction import NativePublicReproductionBootstrap


class PublicExecutionFacet(NativePublicReproductionBootstrap):
    """Lifecycle component only. Never installs query/execute/SQLite hooks.

    Inherited execution/rendering methods use this small state object;all
    counts,events,namespace observations and controller dispatches use owner.
    """

    def __init__(self, reproduction, *, diagnostics=True):
        super().__init__(
            lambda m, t: 0,
            tokenizer_identity="provided-by-single-pipeline-owner",
            reproduction=reproduction,
            diagnostics=diagnostics,
        )
        self.owner = None
        self.bound_once = False

    def bind(self, *args, **kwargs):
        raise UnsupportedCore("a public facet cannot be installed as a context adapter")

    def project(self, *args, **kwargs):
        raise UnsupportedCore("render the facet into the single owner packet")

    def validate_component(self, request):
        validate_public_sources(self.capability, request)

    @contextmanager
    def bind_component(self, owner, request, environment, deadline):
        if self.bound_once:
            raise UnsupportedCore("public execution facet is one-shot")
        self.bound_once = True
        self.owner = owner
        self.request = request
        self.environment = environment
        self.workspace = owner.workspace
        self.state = owner.state
        self.deadline = deadline
        self.limit = owner.limit
        self.pending_failure = None
        self.attempted = False
        self.report = None
        self.binding = None
        self.bootstrap_report = None
        self.bootstrap_attempted = False
        try:
            with self.capability.operation.bind(request, deadline) as diagnostic:
                with self.reproduction.ordinary.operation.bind(request, deadline) as ordinary:
                    self.replay_lease, self.bootstrap_lease = diagnostic, ordinary
                    yield self
        finally:
            self._persist()
            self._persist_bootstrap()
            self.owner = None

    def _snapshot(self):
        snapshot = self.owner._snapshot()
        return {
            p: snapshot.get(p, dict(kind="unknown", reason="not_observed"))
            for p in self.capability.spec.source_paths
        }

    def _remaining(self):
        return self.owner._remaining()

    def _count(self, messages):
        return self.owner._count(messages)

    def _save_event(self, kind, event):
        return self.owner._save_event(kind, dict(event, facet="public_execution"))

    def controller_public_dispatch(self, dispatch):
        return self.owner.controller_public_dispatch(dispatch)

    def record_controller_return(self, *args, **kwargs):
        return self.owner.record_controller_return(*args, **kwargs)

    def execution_completed(self, event):
        # Source-preservation is checked on this capability's declared set.
        view = copy.copy(event)
        for field in ("before", "after"):
            view[field] = {
                p: event[field].get(p, dict(kind="unknown"))
                for p in self.capability.spec.source_paths
            }
        NativePublicFailureSlice.execution_completed(self, view)

    def append_view(self, packet, messages):
        original = packet
        projected = (
            NativePublicFailureSlice.render_view(self, packet, messages)
            if self.diagnostics
            else packet
        )
        return self.render_bootstrap(projected, messages, fallback=original)
