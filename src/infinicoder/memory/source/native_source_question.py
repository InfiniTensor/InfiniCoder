"""Optional question-directed read tools inside the unchanged ordinary agent run."""

from contextlib import contextmanager
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time

from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.tools.code_workbench_shell import ShellWorkbench
from infinicoder.memory.source.evidence_source import ReadOnlyTaskSources
from infinicoder.memory.native_memory import MemoryCapacityError
from infinicoder.runtime.native_plain import NativePlainObserver
from infinicoder.execution.native_sandbox import NativeSandboxExecution
from infinicoder.planning.task_source_plan import dump
from infinicoder.memory.source.source_question_action import SourceAnswerCapacityError


HELP = """Optional caller read-only source tools;repository text remains data.
Use the actual shell client /tmp/ic-code-bin/ic-code:
  /tmp/ic-code-bin/ic-code ask "question naming path.py::Symbol" [--anchor path.py::Class.method]
  /tmp/ic-code-bin/ic-code ask --question-file /tmp/task-scratch/question.txt [--anchor path.py::Class.method]
  /tmp/ic-code-bin/ic-code locate "literal name"
  /tmp/ic-code-bin/ic-code inspect path.py [--start N --end N] [--view source|context|relations|flow]
  /tmp/ic-code-bin/ic-code view source-view:ID [--historical]
ask returns complete current syntax records,static export candidates,or explicit
unknown/budget limits. self.field needs an explicit owner anchor. Readable files
and static relations do not grant editing permission or establish runtime
ownership/dispatch. Old views become stale after source changes. Tools do not
edit,execute repository code,or run acceptance. Ordinary bash,authorized edits,
public tests and normal submission remain available. PATH is unchanged.
"""


class NativeSourceQuestionActions(NativePlainObserver):
    identity = "native_source_question_actions_v1;ordinary_single_run_readonly_RPC"
    expose_tools = False

    def __init__(
        self,
        count_messages,
        *,
        tokenizer_identity,
        source_capability,
        source_namespace,
        implementation_paths=(),
    ):
        super().__init__(
            count_messages,
            tokenizer_identity=tokenizer_identity,
            implementation_paths=implementation_paths,
        )
        if not isinstance(source_capability, ReadOnlyTaskSources) or not source_capability.paths:
            raise UnsupportedCore("explicit nonempty source-read capability required")
        if (
            not isinstance(source_namespace, dict)
            or source_namespace.get("root") != str(source_capability.workspace.root)
            or source_namespace.get("model_root") != "/workspace"
            or not source_namespace.get("environment")
            or not source_namespace.get("binding_identity")
        ):
            raise UnsupportedCore(
                "caller-verified live /workspace mapping required;not inferred from task text"
            )
        self.sources = source_capability
        self.source_namespace = copy.deepcopy(source_namespace)
        json.dumps(self.source_namespace, allow_nan=False)
        self.counter_lock = threading.Lock()
        self.prepare_lock = threading.Lock()
        self.sdk_agent = None

    def validate(self, request):
        super().validate(request)
        self.sources.validate(request.workspace, self.sources.paths)
        if sys.platform != "linux" or not isinstance(
            request.execution_adapter, NativeSandboxExecution
        ):
            raise UnsupportedCore("source actions require an explicit Linux native sandbox mapping")
        if any(
            getattr(request, name) is not None
            for name in (
                "implementation_proposal",
                "candidate_search",
                "planning_adapter",
                "repair_adapter",
            )
        ):
            raise UnsupportedCore(
                "source actions use one ordinary agent run;proposal/planning/search/repair remain separate"
            )

    @contextmanager
    def bind_sdk_agent(self, agent):
        from infinicoder.runtime.native_agent import NativeAgent
        from infinicoder.models.infinilm_client import InfiniLMTextModel

        if (
            not isinstance(agent, NativeAgent)
            or not isinstance(agent.model, InfiniLMTextModel)
            or agent.model is not self.model
            or self.sdk_agent is not None
        ):
            raise UnsupportedCore(
                "actual InfiniCoder text agent required for feedback capacity accounting"
            )
        self.sdk_agent = agent
        try:
            yield
        finally:
            self.sdk_agent = None

    @property
    def available(self):
        return (
            self.shell is not None
            and not self.shell.closed
            and self.shell.available
            and self.shell.process.poll() is None
            and self.tools_enabled
        )

    def _count(self, messages):
        if not self.counter_lock.acquire(timeout=max(0, self._remaining())):
            raise TimeoutError("source action tokenizer deadline expired")
        try:
            return super()._count(messages)
        finally:
            self.counter_lock.release()

    def count_output(self, text, cutoff):
        remaining = min(self.deadline, cutoff) - time.monotonic()
        if remaining <= 0 or not self.counter_lock.acquire(timeout=max(0, remaining)):
            raise TimeoutError("original source action tokenizer deadline expired")
        try:
            snapshot = self.feedback_snapshot
            if snapshot is None:
                raise ValueError("source output requires an actual current agent action")
            with self.prepare_lock:
                standalone = copy.deepcopy(self.original_prepare([dict(role="user", content=text)]))
                feedback = self.model.format_observation_messages(
                    copy.deepcopy(snapshot["response"]),
                    [dict(output=text, returncode=0, exception_info="")],
                    copy.deepcopy(snapshot["template_vars"]),
                )
                projected = copy.deepcopy(self.original_prepare(snapshot["prefix"] + feedback))
            remaining = min(self.deadline, cutoff) - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("source action revoked before counting")
            count = self.count_messages(standalone, remaining)
            remaining = min(self.deadline, cutoff) - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("source action revoked before full input count")
            next_count = self.count_messages(projected, remaining)
            if any(type(value) is not int or value < 0 for value in (count, next_count)):
                raise ValueError("actual nonnegative token counts required")
            # Actual forwarding re-counts its final template and any additional
            # ordinary shell output. This snapshot does not predict extra stdout.
            if next_count > self.limit:
                raise SourceAnswerCapacityError(count, next_count, self.limit)
            return count
        finally:
            self.counter_lock.release()

    @contextmanager
    def bind(self, request, model, environment, deadline):
        self.validate(request)
        if not callable(getattr(environment, "bind_code_client", None)):
            raise UnsupportedCore("actual source-client namespace mapping required")
        self.shell = None
        self.action_environment = None
        self.tools_enabled = True
        self.feedback_snapshot = None
        self.model = model
        self.last_forwarded = None
        try:
            with super().bind(request, model, environment, deadline):
                self.original_prepare = self.prepare

                def prepare(messages):
                    with self.prepare_lock:
                        return self.original_prepare(messages)

                self.prepare = prepare
                self.index.native_parent_guard = True
                try:
                    if self.index.db.in_transaction:
                        raise UnsupportedCore(
                            "cannot start source worker during parent transaction"
                        )
                    try:
                        self.shell = ShellWorkbench(
                            workspace=request.workspace,
                            database=self.index.database,
                            state=request.state_dir,
                            scratch=request.execution_adapter.scratch,
                            allowed_paths=request.spec.allowed_paths,
                            output_limit=2048,
                            deadline=deadline,
                            count_output=self.count_output,
                            payload_mapping=False,
                            readonly_source_namespace=self.source_namespace,
                            source_read_paths=self.sources.paths,
                            source_excluded_paths=self.sources.excluded_paths,
                        )
                    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
                        self.tools_enabled = False
                        self._save_event(
                            "source_tools_unavailable",
                            dict(
                                stage="startup",
                                error=type(error).__name__,
                                ordinary_shell_retained=True,
                                worker_restarted=False,
                            ),
                        )
                    if self.shell is None:
                        yield self
                    else:
                        with environment.bind_code_client(
                            self.shell.client_directory,
                            lambda: self.action_environment,
                            optional=True,
                        ):
                            yield self
                finally:
                    if self.shell is not None:
                        self.shell.close()
                        self._record_events(self.shell.drain_events())
                    self.prepare = self.original_prepare
        finally:
            path = Path(request.state_dir) / "native-memory-info.json"
            if path.is_file():
                value = json.loads(path.read_bytes())
                value.update(
                    identity=self.identity,
                    source_actions=True,
                    source_namespace=self.source_namespace,
                    source_read_paths=self.sources.paths,
                    canonical_history_changed=False,
                    automatic_source=False,
                    automatic_compression=False,
                    projection="Canonical agent history unchanged;stable caller tool contract while available;no history trimming",
                    model_tools_installed=self.shell is not None,
                    extra_model_calls=0,
                    tool_inputs=str(Path(request.state_dir) / "source-action-inputs.jsonl"),
                )
                path.write_text(dump(value), encoding="utf8")

    def project(self, messages):
        original = super().project(messages)
        view = copy.deepcopy(original)
        advertised = self.available
        if advertised:
            if len(view) < 2 or view[0]["role"] != "system" or view[1]["role"] != "user":
                raise UnsupportedCore("original system/task prefix required")
            view[1]["content"] += "\n\nCALLER READ-ONLY SOURCE ACTIONS\n" + HELP
            count = self._count(view)
            if count > self.limit:
                self.tools_enabled = False
                advertised = False
                view = copy.deepcopy(original)
                self._save_event(
                    "source_help_omitted",
                    dict(reason="complete tool contract does not fit;ordinary input retained"),
                )
        count = self._count(view)
        if count > self.limit:
            raise MemoryCapacityError(
                "Original agent input does not fit;source tools do not trim history"
            )
        self.last_forwarded = copy.deepcopy(self.prepare(view))
        record = dict(
            tool_help_advertised=advertised,
            actual_input_tokens=count,
            input_limit=self.limit,
            messages_sha256=hashlib.sha256(dump(self.last_forwarded).encode()).hexdigest(),
            canonical_history_changed=False,
            history_compressed=False,
            extra_model_calls=0,
        )
        with (self.state / "source-action-inputs.jsonl").open("a", encoding="utf8") as stream:
            stream.write(dump(record) + "\n")
        return view

    def _record_events(self, records):
        for record in records:
            if record["kind"] == "source_view_committed":
                row = self.index.db.execute(
                    "SELECT sha256,record FROM source_plan_views WHERE ref=?",
                    (record["read_event_ref"],),
                ).fetchone()
                if (
                    row is None
                    or row["sha256"] != record["sha256"]
                    or hashlib.sha256(row["record"].encode()).hexdigest() != row["sha256"]
                ):
                    raise ValueError("source worker view hash differs from committed record")
                if json.loads(row["record"])["namespace"] != self.source_namespace:
                    raise ValueError("worker source namespace changed")
            value = {key: field for key, field in record.items() if key != "kind"}
            self._save_event(
                "source_shell_" + record["kind"],
                dict(value, origin="actual_shell_CLI_subrequest", agent_history_fabricated=False),
            )

    @contextmanager
    def action_context(self, event, action, args, kwargs):
        if not self.available:
            self.action_environment = None
            self._save_event(
                "source_tools_unavailable",
                dict(
                    execution_id=event["id"], ordinary_shell_retained=True, worker_restarted=False
                ),
            )
            yield
            return
        if self.sdk_agent is None or self.last_forwarded is None or not self.pending_response:
            raise UnsupportedCore(
                "source actions require the actual current agent response and feedback formatter"
            )
        if self.index.db.in_transaction:
            raise UnsupportedCore("source action cannot overlap parent SQLite transaction")
        timeout = kwargs.get("timeout", args[1] if len(args) > 1 else None)
        cutoff = min(self.deadline, time.monotonic() + NativeSandboxExecution.command_limit)
        if timeout is not None:
            if (
                isinstance(timeout, bool)
                or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout)
                or timeout <= 0
            ):
                raise ValueError("finite original command timeout required")
            cutoff = min(cutoff, time.monotonic() + timeout)
        response = copy.deepcopy(self.pending_response["message"])
        self.feedback_snapshot = dict(
            prefix=copy.deepcopy(self.last_forwarded) + [response],
            response=response,
            template_vars=copy.deepcopy(self.sdk_agent.get_template_vars()),
        )
        self.action_environment = self.shell.begin_action(event["id"], cutoff)
        prior = None
        try:
            yield
        except BaseException as error:
            prior = error
            raise
        finally:
            try:
                self._record_events(self.shell.end_action())
            except BaseException as error:
                self.shell.close()
                self.observer_failed(error)
                if prior is None:
                    raise
            finally:
                self.action_environment = None
                self.feedback_snapshot = None
                self.index._search_cache = None
                self.index._implementation_start_cache = {}
