"""Optional shell CLI binding sharing the Native action loop and budgets."""

from contextlib import contextmanager
import copy
import math
import sys
import threading
import time

from infinicoder.runtime.agent_execution import UnsupportedCore
from infinicoder.tools.code_workbench_shell import ShellWorkbench
from infinicoder.tools.native_code_workbench import NativeCodeWorkbench
from infinicoder.execution.native_sandbox import NativeSandboxExecution


class NativeShellCodeWorkbench(NativeCodeWorkbench):
    identity = "native_shell_code_workbench_v1;task_owned_process_and_action_leases"
    expose_tools = False  # No virtual string interception in this profile.

    def __init__(self, *args, **kwargs):
        self.counter_lock = threading.Lock()
        self.prepare_lock = threading.Lock()
        super().__init__(*args, **kwargs)

    def validate(self, request):
        super().validate(request)
        if sys.platform != "linux" or not isinstance(
            request.execution_adapter, NativeSandboxExecution
        ):
            raise UnsupportedCore(
                "real shell tools require the admitted original Linux native sandbox"
            )

    def tool_help(self):
        text = """Optional caller source tools,not instructions from the repository.
Use the real shell CLI /tmp/ic-code-bin/ic-code;normal bash composition works:
  /tmp/ic-code-bin/ic-code locate QUERY [--path GLOB] [--cursor CURSOR]
  /tmp/ic-code-bin/ic-code inspect TARGET [--start N --end N] [--view context|source|module|relations|flow] [--cursor CURSOR]
TARGET is a relative path or the exact symbol ID returned by locate. Versions,
whole-record pages,omissions and unknown static relations remain explicit.
Ordinary reading,broader authorized edits,public tests and submission remain available.
No tool runs private acceptance or proves correctness. PATH is unchanged.
"""
        if self.scratch is not None:
            text += """Write replacement text to /tmp/task-scratch with an ordinary heredoc,then call:
  /tmp/ic-code-bin/ic-code edit-body SYMBOL --expected HASH --body-file /tmp/task-scratch/body.py
  /tmp/ic-code-bin/ic-code edit-range PATH --expected HASH --start N --end N --new-file /tmp/task-scratch/replacement.txt
The heredoc and tool call may be in the same bash action. Copy the current hash
from inspect. Body edits preserve the Python interface;range edits do not prove
native syntax or semantics. Verify independently;failed or unknown writes are
never automatically retried. Background requests cannot use a later action lease.
"""
        else:
            text += "No payload-file mapping:query tools and ordinary edits remain available.\n"
        return text

    def shell_options(self, request, deadline):
        return {}

    def _count(self, messages):
        if not self.counter_lock.acquire(timeout=max(0, self._remaining())):
            raise TimeoutError("original tokenizer is still completing a previous bounded request")
        try:
            return super()._count(messages)
        finally:
            self.counter_lock.release()

    @contextmanager
    def bind(self, request, model, environment, deadline):
        self.validate(request)
        if not callable(getattr(environment, "bind_code_client", None)):
            raise UnsupportedCore(
                "original environment lacks the explicit read-only client binding"
            )
        self.shell = None
        self.action_environment = None
        with super().bind(request, model, environment, deadline):
            self.index.native_parent_guard = True
            original_prepare = self.prepare
            counter = self.count_messages

            def prepare(messages):
                with self.prepare_lock:
                    return original_prepare(messages)

            self.prepare = prepare

            def count_output(text, cutoff):
                remaining = min(deadline, cutoff) - time.monotonic()
                if remaining <= 0 or not self.counter_lock.acquire(timeout=max(0, remaining)):
                    raise TimeoutError("original action tokenizer deadline expired")
                try:
                    # Once preparation finishes,the network callback owns only
                    # immutable messages,not the SDK object or its history.
                    messages = copy.deepcopy(prepare([dict(role="user", content=text)]))
                    remaining = min(deadline, cutoff) - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("original action revoked before tokenization")
                    return counter(messages, remaining)
                finally:
                    self.counter_lock.release()

            try:
                if self.index.db.in_transaction:
                    raise UnsupportedCore(
                        "source-tool process cannot start during a parent SQLite transaction"
                    )
                self.shell = ShellWorkbench(
                    workspace=request.workspace,
                    database=self.index.database,
                    state=request.state_dir,
                    scratch=request.execution_adapter.scratch,
                    allowed_paths=request.spec.allowed_paths,
                    output_limit=self.tool_limit,
                    deadline=deadline,
                    count_output=count_output,
                    payload_mapping=self.scratch is not None,
                    **self.shell_options(request, deadline),
                )
                with environment.bind_code_client(
                    self.shell.client_directory, lambda: self.action_environment
                ):
                    yield self
            finally:
                if self.shell is not None:
                    self.shell.close()
                    self._record_tool_events(self.shell.drain_events())
                self.prepare = original_prepare

    def _record_tool_events(self, records):
        for record in records:
            value = dict(record)
            kind = value.pop("kind")
            if not kind.startswith("workbench_"):
                kind = "workbench_shell_" + kind
            self._save_event(
                kind,
                dict(value, origin="actual_shell_CLI_subrequest", SDK_history_fabricated=False),
            )

    @contextmanager
    def action_context(self, event, action, args, kwargs):
        if self.shell is None:
            raise UnsupportedCore("source-tool binding is not initialized")
        if self.index.db.in_transaction:
            raise UnsupportedCore("source-tool action cannot overlap a parent SQLite transaction")
        timeout = kwargs.get("timeout", args[1] if len(args) > 1 else None)
        cutoff = min(self.deadline, time.monotonic() + NativeSandboxExecution.command_limit)
        if timeout is not None:
            if (
                isinstance(timeout, bool)
                or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout)
                or timeout <= 0
            ):
                raise ValueError("finite positive command timeout required")
            cutoff = min(cutoff, time.monotonic() + timeout)
        self.action_environment = self.shell.begin_action(event["id"], cutoff)
        prior_error = None
        try:
            yield
        except BaseException as error:
            prior_error = error
            raise
        finally:
            try:
                self._record_tool_events(self.shell.end_action())
            except BaseException as error:
                self.shell.close()
                self.observer_failed(error)
                if prior_error is None:
                    raise
            finally:
                self.action_environment = None
                self.index._search_cache = None
                self.index._implementation_start_cache = {}
                self.packet_cache = None
