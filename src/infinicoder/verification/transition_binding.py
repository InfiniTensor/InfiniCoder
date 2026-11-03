"""Add public transition tools to an existing neutral caller request.

Keep its actual namespace builder,task,budget,transport and final adapter.
The only extra ordinary mounts are readonly public client/socket/reference.
"""

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import stat

from infinicoder.runtime.agent_execution import CoreRequest, UnsupportedCore
from infinicoder.runtime.native_plain import NativePlainObserver
from infinicoder.execution.native_sandbox import NativeSandboxExecution
from infinicoder.verification.native_transition_probe import NativeTransitionProbe
from infinicoder.verification.public_transition_probe import PublicTransitionContract
from infinicoder.verification.transition_public_cpu import (
    CLIENT,
    SOCKET,
    REFERENCE,
    TransitionAcceptance,
    _public_reference_bytes,
)
from infinicoder.verification.transition_reports import PublicTransitionCapability


class TransitionNamespaceExecution(NativeSandboxExecution):
    """Preserve a caller's original bwrap builder,including protected overlays."""

    def __init__(self, original, state):
        if not isinstance(original, NativeSandboxExecution) or original.snapshot_trials:
            raise UnsupportedCore("single-run caller NativeSandboxExecution required")
        self.original = original
        self.state = Path(state).resolve()
        self.transition_mounts = None
        super().__init__(
            self._builder,
            original.scratch,
            env=original.env,
            snapshot_trials=original.snapshot_trials,
            trial_generated_paths=original.trial_generated_paths,
            task_environment=original.task_environment,
        )
        self.identity = original.identity + ";readonly_public_transition_mounts_v1"
        self.command_limit = original.command_limit

    def _builder(self, workspace, command):
        argv = list(self.original.sandbox_argv(workspace, command))
        if (
            not command
            or len(argv) <= len(command)
            or argv[-len(command) :] != list(command)
            or Path(argv[0]).name != "bwrap"
        ):
            raise UnsupportedCore("original bwrap builder with exact command suffix required")
        if self.transition_mounts is not None:
            client, socket, reference = self.transition_mounts
            argv[-len(command) : -len(command)] = [
                "--ro-bind",
                str(client),
                CLIENT,
                "--ro-bind",
                str(socket),
                SOCKET,
                "--ro-bind",
                str(reference),
                REFERENCE,
            ]
        return argv

    def validate(self, request):
        self.original.validate(request)
        super().validate(request)
        self._builder(request.workspace, ["/bin/true"])

    @contextmanager
    def bind_transition_service(self, client, socket, reference):
        client, socket, reference = map(lambda p: Path(p).absolute(), (client, socket, reference))
        if (
            self.transition_mounts is not None
            or client != self.state / "transition-client.py"
            or reference != self.state / "transition-reference.py"
            or not client.is_file()
            or not reference.is_file()
            or not stat.S_ISSOCK(socket.lstat().st_mode)
            or any(p.is_symlink() for v in (client, socket, reference) for p in (v, *v.parents))
        ):
            raise UnsupportedCore("exact task-owned readonly transition service required")
        self.transition_mounts = client, socket, reference
        try:
            yield
        finally:
            self.transition_mounts = None


class PublicTransitionMaterials:
    """Explicit public reference/capability without inferring an execution type."""

    def __init__(self, contract, reference, workspace, state, operation, exposed_runtime_roots):
        if not isinstance(contract, PublicTransitionContract):
            raise TypeError("explicit public contract required")
        self.contract = contract
        self.data = contract.manifest()
        self.reference = Path(reference).absolute()
        self.raw = None
        self.workspace = Path(workspace).resolve()
        self.state = Path(state).resolve()
        if not isinstance(exposed_runtime_roots, tuple) or not exposed_runtime_roots:
            raise ValueError("caller must declare original exposed runtime roots")
        self.runtime_roots = tuple(Path(p).resolve() for p in exposed_runtime_roots)
        if any(not Path(p).is_absolute() for p in exposed_runtime_roots):
            raise ValueError("absolute exposed runtime roots required")
        self.capability = PublicTransitionCapability(contract, operation)
        self._validate_reference()

    def _validate_reference(self):
        self.raw = _public_reference_bytes(
            self.reference, self.workspace, self.state, self.runtime_roots, self.data, self.raw
        )

    def validate(self, request):
        self._validate_reference()
        self.capability.validate(request)


def attach_public_transition(
    request,
    contract,
    reference_script,
    operation,
    *,
    automatic,
    exposed_runtime_roots,
    count_messages=None,
    tokenizer_identity=None,
):
    """Return a request with the same original task and independent acceptance.

    Both ON/OFF get the identical manual run/read material. Only ON consumes
    automatic edit feedback. Unsupported workflows are refused,never erased.
    Caller capabilities remain explicit;there is no model-selected runner.
    """
    if not isinstance(request, CoreRequest) or type(automatic) is not bool:
        raise TypeError("CoreRequest and explicit automatic boolean required")
    if (
        request.candidate_search is not None
        or request.repair_adapter is not None
        or request.planning_adapter is not None
        or request.context_adapter is not None
        and type(request.context_adapter) is not NativePlainObserver
    ):
        raise UnsupportedCore(
            "transition attachment requires the neutral single-run context;existing workflows are not cleared"
        )
    counter = (
        count_messages
        if count_messages is not None
        else getattr(request.transport_adapter, "counter", None)
    )
    identity = (
        tokenizer_identity if tokenizer_identity is not None else getattr(counter, "identity", None)
    )
    if not callable(counter) or not isinstance(identity, str) or not identity:
        raise ValueError("actual caller tokenizer counter and identity required")
    materials = PublicTransitionMaterials(
        contract,
        reference_script,
        request.workspace,
        request.state_dir,
        operation,
        exposed_runtime_roots,
    )
    paths = tuple(
        dict.fromkeys((*getattr(request.context_adapter, "paths", ()), *contract.observed_paths))
    )
    execution = TransitionNamespaceExecution(request.execution_adapter, request.state_dir)
    context = (
        NativeTransitionProbe(
            counter,
            tokenizer_identity=identity,
            implementation_paths=paths,
            capability=materials.capability,
        )
        if automatic
        else NativePlainObserver(counter, tokenizer_identity=identity, implementation_paths=paths)
    )
    acceptance = TransitionAcceptance(request.acceptance_adapter, materials, execution, context)
    return replace(
        request, execution_adapter=execution, acceptance_adapter=acceptance, context_adapter=context
    )
