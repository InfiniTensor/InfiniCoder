"""One existing implementation producer inside the ordinary Native SDK loop."""

from contextlib import contextmanager
import copy
from dataclasses import asdict
import json
from pathlib import Path
import time

from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.verification.candidate_public_check import (
    CandidatePublicCheck,
    PublicCandidateVerdict,
)
from infinicoder.domains.operator.convolution_provider import ConvolutionProviderContract, digest
from infinicoder.domains.operator.reduction_provider import ReductionProviderContract
from infinicoder.domains.operator.rmsnorm_provider import RMSNormProviderContract
from infinicoder.domains.operator.fusion_provider import FusionProviderContract
from infinicoder.runtime.native_plain import NativePlainObserver
from infinicoder.execution.workspace_tools import Workspace


class NativeImplementationProvider(NativePlainObserver):
    identity = "native_existing_convolution_implementation_v1"

    def __init__(
        self, count_messages, *, tokenizer_identity, contract, public_check, implementation_paths=()
    ):
        super().__init__(
            count_messages,
            tokenizer_identity=tokenizer_identity,
            implementation_paths=implementation_paths,
        )
        if type(contract) is FusionProviderContract:
            from infinicoder.domains.operator import fusion_provider as producer

            self.producer_family = "linear GELU fusion"
            self.producer_dispatch_kind = "source_bound_linear_gelu_fusion"
            self.identity = "native_linear_gelu_fusion_expert_v1"
        elif type(contract) is ConvolutionProviderContract:
            from infinicoder.domains.operator import convolution_provider as producer

            self.producer_family = "convolution"
            self.producer_dispatch_kind = "existing_convolution_provider"
        elif type(contract) is ReductionProviderContract:
            from infinicoder.domains.operator import reduction_provider as producer

            self.producer_family = "single-axis sum"
            self.producer_dispatch_kind = "single_axis_sum_provider"
            self.identity = "native_single_axis_sum_implementation_v1"
        elif type(contract) is RMSNormProviderContract:
            from infinicoder.domains.operator import rmsnorm_provider as producer

            self.producer_family = "RMSNorm inference"
            self.producer_dispatch_kind = "rmsnorm_inference_provider"
            self.identity = "native_rmsnorm_implementation_v1"
        else:
            raise ValueError("explicit supported implementation provider contract required")
        self.producer = producer
        self.contract_type = type(contract)
        if not isinstance(public_check, CandidatePublicCheck):
            raise ValueError("original caller public operation and decoder required")
        if contract.target not in public_check.operation.observed_paths:
            raise ValueError("public operation must observe the exact target")
        self.contract = contract
        self.public_check = public_check
        self.bound_once = False
        self.handoff_policy = None
        self.schedule_search = None

    def producer_identity(self):
        if type(self.contract) is not self.contract_type:
            raise ValueError("implementation producer contract type changed")
        return self.producer.identity()

    def public_reference(self):
        self.producer_identity()
        return self.producer._source_and_lowerer_task(self.request.spec, self.contract)[0]

    def _namespace(self):
        self._workspace()
        return self.binding

    def validate(self, request):
        super().validate(request)
        if request.context_adapter is not self or any(
            getattr(request, name) is not None
            for name in (
                "planning_adapter",
                "repair_adapter",
                "candidate_search",
                "implementation_proposal",
            )
        ):
            raise UnsupportedCore(
                "implementation provider is a separate ordinary single-run profile"
            )
        if getattr(request.transport_adapter, "delivery_policy", None) is not None:
            raise UnsupportedCore("implementation provider uses the unchanged common transport")
        if (
            Path(self.public_check.operation.workspace).resolve()
            != Path(request.workspace).resolve()
        ):
            raise UnsupportedCore(
                "implementation provider public check must belong to this live workspace"
            )

    def _save(self):
        if self.report is None:
            return
        path = self.state / "native-implementation-provider.json"
        temporary = path.with_suffix(".next")
        try:
            temporary.write_text(
                json.dumps(self.report, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf8",
            )
            temporary.replace(path)
        except (OSError, ValueError, TypeError) as error:
            self.report["persistence_error"] = type(error).__name__

    def _skip(self, reason):
        self._save_event("implementation_provider_not_applicable", dict(reason=reason))

    def _workspace(self):
        describe = getattr(self.environment, "implementation_workspace_binding", None)
        if not callable(describe):
            raise UnsupportedCore("no exact live source namespace capability")
        binding = describe()
        if (
            binding.get("root") != str(Path(self.request.workspace).resolve())
            or binding.get("model_root") != "/workspace"
        ):
            raise UnsupportedCore("caller source namespace is not the actual live workspace")
        if self.binding is not None and binding != self.binding:
            raise UnsupportedCore("live source namespace changed")
        self.binding = copy.deepcopy(binding)
        workspace = Workspace(self.request.workspace, (self.contract.target,))
        path = workspace.root / self.contract.target
        if any(
            p.is_symlink() or hasattr(p, "is_junction") and p.is_junction()
            for p in (path, *path.parents)
        ):
            raise UnsupportedCore("provider target and parents must not be links")
        if path.exists() and (not path.is_file() or path.stat().st_size > 1_000_000):
            raise UnsupportedCore("provider target must be a bounded regular source file")
        return workspace

    def _current(self):
        workspace = self._workspace()
        path = workspace.path(self.contract.target, write=True)
        return digest(path.read_bytes()) if path.exists() else None

    def _restore(self, admission):
        """Restore only our unchanged produced bytes,never a concurrent revision."""
        workspace = self._workspace()
        path = workspace.path(self.contract.target, write=True)
        if not path.exists() or digest(path.read_bytes()) != admission["candidate_sha256"]:
            return "skipped_current_version_changed"
        if admission["target_before"] is None:
            # Same ordinary hash-guard policy as Workspace.edit. This controller
            # has no authority to remove a different or concurrently edited file.
            if digest(path.read_bytes()) != admission["candidate_sha256"]:
                return "skipped_current_version_changed"
            path.unlink()
        else:
            workspace.edit(
                self.contract.target,
                admission["candidate_sha256"],
                old=admission["candidate_source"],
                new=admission["target_before"].decode("utf8"),
            )
        return "restored_original_starter"

    @contextmanager
    def bind(self, request, model, environment, deadline):
        if self.bound_once:
            raise UnsupportedCore("implementation provider cannot resume an existing run")
        self.bound_once = True
        self.request = request
        self.environment = environment
        self.attempted = False
        self.binding = None
        self.report = None
        try:
            with super().bind(request, model, environment, deadline):
                yield self
        finally:
            # A display failure must not replace an SDK error or cancellation.
            try:
                self._save()
                if hasattr(self, "capability_selection"):
                    (self.state / "native-capabilities.json").write_text(
                        json.dumps(
                            dict(
                                self.capability_selection,
                                one_query_owner=True,
                                one_source_journal=True,
                            ),
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf8",
                    )
                path = Path(request.state_dir) / "native-memory-info.json"
                if self.report is not None and path.exists():
                    info = json.loads(path.read_bytes())
                    info.update(
                        implementation_provider=True,
                        automatic_compression=False,
                        auxiliary_model_calls=0,
                        canonical_history_changed=False,
                        projection="Original canonical history plus implementation-module observation",
                        prepared_provider_views=len(self.report["views"]),
                    )
                    path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf8")
            except (OSError, ValueError, TypeError):
                pass

    def before_query(self, messages):
        if self.attempted:
            return
        self.attempted = True
        if self.contract.mode != "enabled":
            return
        start = time.monotonic()
        reserve = self.contract.final_reserve_seconds
        if self._remaining() <= reserve + 1:
            self._skip("insufficient caller final reserve")
            return
        try:
            operation = self.public_operation
            if operation is None or operation.operation is not self.public_check.operation:
                raise UnsupportedCore("original public operation is not bound")
            workspace = self._workspace()
            self.producer_identity()
            admission = self.producer.assess(self.request.spec, workspace, self.contract)
            if admission["status"] != "Applicable":
                self._skip(admission.get("reason", admission["status"]))
                return
            if time.monotonic() - start > 10 or self._remaining() <= reserve + 1:
                self._skip("static processing or final reserve bound")
                return
            # Count the real tokenizer before any write. No original task or
            # history is ever cropped to make space for the controller result.
            preview = self._view(
                admission["candidate_sha256"], "unknown", "stale_or_unverified", ""
            )
            # Do not use a repetitive padding string as an alleged worst-case
            # tokenizer count. Reserve the caller result bound plus a byte-sized
            # evidence margin;the actual SDK still checks every physical input.
            projected = self._count([*messages, dict(role="user", content=preview)])
            if projected + max(2048, operation.manifest["result_token_bound"]) + 128 > self.limit:
                self._skip("no actual tokenizer room for provider result")
                return
            if time.monotonic() - start > 10 or self._remaining() <= reserve + 1:
                self._skip("static processing or final reserve bound after tokenizer")
                return
            self.report = dict(
                identity=self.identity,
                contract=asdict(self.contract),
                status="admitted",
                namespace=self.binding,
                generated_sha256=admission["candidate_sha256"],
                original_sha256=admission["target_before_sha256"],
                facts=admission["facts"],
                implementation=admission["implementation"],
                preconditions=admission["preconditions"],
                producer="existing deterministic "
                + self.producer_family
                + " lowerer;not model reasoning",
                original_task_unchanged=True,
                auxiliary_model_calls=0,
                private_final_used=False,
                public_executions=0,
                views=[],
            )
            # Recheck mapping,permission,reference,producer and current bytes at
            # the write boundary;Workspace adds its own expected-hash guard.
            self._workspace()
            applied = self.producer.apply(self.request.spec, workspace, self.contract, admission)
            self.report.update(
                status="produced", write=applied, static_wall_seconds=time.monotonic() - start
            )
            self._save_event(
                "implementation_produced", dict(target=self.contract.target, **applied)
            )
        except (ValueError, TypeError, OSError, UnsupportedCore, TimeoutError) as error:
            if self.report is not None:
                try:
                    self._restore(admission)
                except (ValueError, OSError, UnsupportedCore):
                    pass
                self.report = None
            self._skip(type(error).__name__ + ": " + str(error))
            return
        dispatch = dict(
            kind=self.producer_dispatch_kind,
            implementation_identity=operation.manifest["implementation_identity"],
            scope=operation.manifest["public_scope"],
            coverage=operation.manifest["public_coverage"],
            target=self.contract.target,
            source_sha256=admission["candidate_sha256"],
        )
        before = {}
        started = time.monotonic()
        try:
            before = self._snapshot()
            if self._remaining() <= reserve:
                raise TimeoutError("caller final reserve exhausted before public execution")
            with self.controller_public_dispatch(dispatch):
                self.report["public_executions"] += 1
                result = operation.invoke(self.deadline - reserve)
            encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
            if len(encoded.encode()) > 4 * 1024**2:
                raise ValueError("public result exceeds controller record bound")
            self.report["public_result"] = result
            self.record_controller_return(
                dispatch, before=before, after=self._snapshot(), report=result
            )
            if self._current() != admission["candidate_sha256"]:
                raise ValueError("source-preserving public operation changed target")
            verdict = self.public_check.decoder(result)
            if not isinstance(verdict, PublicCandidateVerdict):
                raise ValueError("caller decoder returned no public verdict")
            self.report.update(
                public_status=verdict.status,
                public_evidence=verdict.counterexample,
                tested_sha256=admission["candidate_sha256"],
                status="public_checked",
            )
            if verdict.status in ("unknown", "env_error"):
                self.report["restoration"] = self._restore(admission)
                self.report["status"] = "public_contract_unavailable"
            # A definite functional FAIL stays available for ordinary model
            # repair. Performance below target never erases a correct source.
        except BaseException as error:
            self.report.update(
                status="public_operation_error",
                public_error=type(error).__name__ + ": " + str(error),
            )
            try:
                self.report["restoration"] = self._restore(admission)
            except (OSError, ValueError, UnsupportedCore) as restore_error:
                self.report["restore_error"] = type(restore_error).__name__
            try:
                self.record_controller_return(
                    dispatch,
                    before=before,
                    after=self._snapshot(),
                    report=None,
                    error=type(error).__name__,
                )
            except Exception as recording_error:
                self.report["observation_error"] = type(recording_error).__name__
            if not isinstance(error, Exception):
                raise
        finally:
            self.report["public_wall_seconds"] = time.monotonic() - started
            self.report["controller_wall_seconds"] = time.monotonic() - start
            self._save()

    def _view(self, current, status, freshness, evidence):
        data = dict(
            producer="existing " + self.producer_family + " implementation module",
            target="/workspace/" + self.contract.target,
            current_sha256=current,
            public_status=status,
            public_result_version=freshness,
            evidence=evidence.encode("utf8")[:2048].decode("utf8", errors="ignore"),
            controller_audit_ref="native-implementation-provider.json (outside candidate;not a workspace file)",
            authority="Public development evidence only. Ordinary coding continues;independent final still required.",
        )
        return "IMPLEMENTATION MODULE OBSERVATION\n" + json.dumps(
            data, ensure_ascii=False, separators=(",", ":")
        )

    def project(self, messages):
        original = super().project(messages)
        if self.report is None:
            return original
        try:
            current = self._current()
        except (ValueError, OSError, UnsupportedCore):
            current = None
        tested = self.report.get("tested_sha256")
        freshness = "current" if tested is not None and current == tested else "stale_or_unverified"
        status = (
            self.report.get("public_status", "unknown") if freshness == "current" else "unknown"
        )
        evidence = (
            self.report.get("public_evidence")
            or self.report.get("public_error")
            or self.report["status"]
        )
        if freshness != "current":
            evidence = "Previous public result does not verify this current source. " + evidence
        text = self._view(current, status, freshness, evidence)
        self.report["views"].append(
            dict(
                canonical_messages=len(messages),
                current_sha256=current,
                freshness=freshness,
                view_sha256=digest(text),
            )
        )
        self._save_event("implementation_view_prepared", self.report["views"][-1])
        self._save()
        return [*original, dict(role="user", content=text)]
