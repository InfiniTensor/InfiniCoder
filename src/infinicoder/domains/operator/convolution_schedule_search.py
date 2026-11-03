"""One owner's fixed schedule producer and existing public best-correct loop."""

import copy
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.verification.candidate_public_check import contract_hash
from infinicoder.domains.operator.convolution_provider import digest
from infinicoder.domains.operator.convolution_schedules import (
    propose_convolution_schedules,
    producer_identity,
)
from infinicoder.memory.source.evidence_source import ReadOnlyTaskSources
from infinicoder.optimization.native_measured_optimization import (
    PublicPerformanceController,
    PublicPerformancePolicy,
    PublicPerformanceSample,
)


@dataclass(frozen=True)
class ConvolutionScheduleSelection:
    target: str
    source_sha256: str
    schedule_name: str
    producer_pins: tuple
    public_check_sha256: str
    selection_policy_sha256: str
    public_report_ref: str
    public_report_sha256: str
    workload_identity: str


class ConvolutionScheduleSearch:
    """No SDK,query owner or final verifier;uses the original owner's journal."""

    state_filename = "native-convolution-schedules.json"
    selection_type = ConvolutionScheduleSelection

    def _producer_identity(self):
        return producer_identity()

    def _propose(self, owner):
        return propose_convolution_schedules(owner.request.spec, owner._workspace(), owner.contract)

    def __init__(self, check, policy, decoder, decoder_identity):
        if (
            policy.operation is not check.operation
            or not callable(decoder)
            or not isinstance(decoder_identity, str)
            or not decoder_identity.strip()
        ):
            raise ValueError(
                "original public operation and identified trusted performance decoder required"
            )
        self.check, self.policy, self.decoder, self.decoder_identity = (
            check,
            policy,
            decoder,
            decoder_identity,
        )
        self.attempted = False
        self.selected = None
        self.bundle = None
        self.performance = None
        self.manifest_sha256 = contract_hash(self.manifest())

    def manifest(self):
        return dict(
            check=self.check.manifest(),
            performance=dict(
                rollback_paths=self.policy.rollback_paths,
                exclusive_workspace=self.policy.exclusive_workspace,
                max_candidates=self.policy.max_candidates,
                final_reserve_seconds=self.policy.final_reserve_seconds,
                min_speedup=self.policy.min_speedup,
            ),
            decoder_identity=self.decoder_identity,
            producer=self._producer_identity(),
            selection_code=digest(Path(__file__).read_bytes()),
        )

    def _save(self, status):
        data = dict(
            status=status,
            policy_sha256=self.manifest_sha256,
            selected=asdict(self.selected) if self.selected else None,
            records=self.records,
            reused_baseline_measurements=self.reused,
            additional_public_executions=self.additional,
            extra_model_calls=0,
            private_final_used=False,
            authority="Original public workload only;one unchanged independent final still required",
        )
        path = self.owner.state / self.state_filename
        temporary = path.with_suffix(".next")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")
        temporary.replace(path)

    def _measure(self, deadline):
        owner = self.owner
        operation = owner.public_operation
        if operation is None or operation.operation is not self.check.operation:
            raise UnsupportedCore("original source-preserving public operation is not bound")
        pins = {p: self.sources.read(p)[2] for p in self.sources.paths}
        if not self.reused:
            report = owner.report
            if (
                report.get("tested_sha256") != pins[owner.contract.target]
                or report.get("public_status") != "public_pass"
            ):
                raise ValueError("baseline public evidence no longer binds the source")
            raw = copy.deepcopy(report["public_result"])
            self.reused = 1
            owner._save_event(
                "schedule_baseline_measurement_reused",
                dict(
                    source_hashes=pins,
                    additional_underlying_executions=0,
                    decoder_identity=self.decoder_identity,
                ),
            )
        else:
            self.additional += 1
            raw = operation.invoke(deadline)
        encoded = json.dumps(raw, ensure_ascii=False, allow_nan=False).encode("utf8")
        if len(encoded) > 4 * 1024**2:
            raise ValueError("public result exceeds original bound")
        verdict = self.check.decoder(raw)
        if verdict.status not in ("public_pass", "public_fail"):
            raise ValueError("unknown public evidence cannot select a schedule")
        sample = self.decoder(copy.deepcopy(raw))
        if not isinstance(sample, PublicPerformanceSample) or sample.correct != (
            verdict.status == "public_pass"
        ):
            raise ValueError("typed timing contradicts original public correctness")
        if sample.source_hashes != pins:
            raise ValueError("timing source hashes differ")
        path = self.owner.state / ("schedule-public-" + str(len(self.public_records)) + ".json")
        path.write_bytes(encoded)
        self.public_records[pins[owner.contract.target]] = dict(
            correct=sample.correct,
            path=str(path),
            sha256=digest(encoded),
            workload_identity=sample.workload_identity,
        )
        return sample

    def run(self, owner):
        if self.attempted:
            return
        self.attempted = True
        self.owner = owner
        self.records = []
        self.public_records = {}
        self.reused = 0
        self.additional = 0
        report = owner.report
        if (
            not report
            or report.get("public_status") != "public_pass"
            or owner._current() != report.get("generated_sha256")
            or contract_hash(self.manifest()) != self.manifest_sha256
        ):
            self._save("not_applicable_or_baseline_changed")
            return
        self.sources = ReadOnlyTaskSources(owner.request.workspace, (owner.contract.target,))
        self.sources.remaining = owner._remaining
        self.bundle = self._propose(owner)
        if self.bundle is None or self.bundle.variants[0].sha256 != owner._current():
            self._save("original_producer_binding_unavailable")
            return
        decoded = replace(
            self.policy.operation,
            operation_id=self.policy.operation.operation_id + ".fixed-schedule-performance",
            implementation_identity=dict(
                original=self.policy.operation.implementation_identity,
                decoder=self.decoder_identity,
                selection=self.manifest_sha256,
            ),
            executor=self._measure,
        )
        self.performance = PublicPerformanceController(
            replace(self.policy, operation=decoded), self.sources
        )
        with self.performance.bind(owner.request, owner, owner.deadline):
            self.performance.before_query()
            for variant in self.bundle.variants[1:]:
                if self.performance.disabled or self.performance.best is None:
                    break
                if owner._remaining() <= self.policy.final_reserve_seconds + 1:
                    self.records.append(dict(status="final_reserve;no_new_schedule"))
                    break
                if not self.bundle.current_for(owner.request.spec, owner.contract):
                    self.records.append(dict(status="producer_or_task_changed;no_new_schedule"))
                    break
                current = self.performance._snapshot()
                pins = self.performance._pins(current)
                if pins != self.performance.best["pins"]:
                    self.records.append(dict(status="unattributed_source_change;no_write"))
                    break
                target = owner.contract.target
                owner._workspace().edit(
                    target, pins[target], old=current[target]["text"], new=variant.source
                )
                owner._save_event(
                    "controller_schedule_produced",
                    dict(
                        schedule=variant.schedule.name,
                        target=target,
                        before_sha256=pins[target],
                        source_sha256=variant.sha256,
                        provenance="fixed_schedule_producer;not_an_SDK_edit",
                    ),
                )
                # This hook attributes an actual source version to the producer.
                # The controller has no model and does not invent an SDK edit.
                self.performance.source_changed()
                self.performance.before_query()
                self.records.append(
                    dict(
                        schedule=variant.schedule.name,
                        produced_sha256=variant.sha256,
                        current_sha256=owner._current(),
                        performance_disabled=self.performance.disabled,
                    )
                )
        best = self.performance.best
        if best is not None and self.performance.final_source_binding_verified:
            pin = best["pins"][owner.contract.target]
            variant = next((v for v in self.bundle.variants if v.sha256 == pin), None)
            public = self.public_records.get(pin)
            if variant is not None and public and public["correct"]:
                self.selected = self.selection_type(
                    owner.contract.target,
                    pin,
                    variant.schedule.name,
                    self.bundle.producer_pins,
                    contract_hash(self.check.manifest()),
                    self.manifest_sha256,
                    public["path"],
                    public["sha256"],
                    public["workload_identity"],
                )
        self._save("selected_public_correct" if self.selected else "public_selection_unavailable")

    def selection_for(self, owner):
        selected = self.selected
        if (
            selected is None
            or self.owner is not owner
            or self.bundle is None
            or not self.bundle.current_for(owner.request.spec, owner.contract)
            or contract_hash(self.manifest()) != self.manifest_sha256
            or owner._current() != selected.source_sha256
            or contract_hash(self.check.manifest()) != selected.public_check_sha256
            or digest(Path(selected.public_report_ref).read_bytes())
            != selected.public_report_sha256
        ):
            return None
        return selected


def with_convolution_schedule_search(
    request, *, performance_decoder, decoder_identity, exclusive_workspace
):
    """Attach to the same unused provider;explicit handoff and ownership required."""
    from infinicoder.domains.operator.native_implementation_provider import (
        NativeImplementationProvider,
    )
    from infinicoder.domains.operator.convolution_provider import ConvolutionProviderContract

    owner = request.context_adapter
    if (
        type(owner) is not NativeImplementationProvider
        or type(owner.contract) is not ConvolutionProviderContract
        or owner.bound_once
        or getattr(owner, "schedule_search", None) is not None
        or not getattr(owner.handoff_policy, "enabled", False)
    ):
        raise UnsupportedCore(
            "fixed schedule search requires one unused provider with explicit handoff policy"
        )
    target = owner.contract.target
    if owner.paths != (target,) or owner.public_check.operation.observed_paths != (target,):
        raise UnsupportedCore(
            "fixed schedule selection currently requires a single observed target"
        )
    policy = PublicPerformancePolicy(
        owner.public_check.operation,
        (target,),
        exclusive_workspace,
        2,
        max(60, owner.contract.final_reserve_seconds),
        1.02,
    )
    owner.schedule_search = ConvolutionScheduleSearch(
        owner.public_check, policy, performance_decoder, decoder_identity
    )
    return request
