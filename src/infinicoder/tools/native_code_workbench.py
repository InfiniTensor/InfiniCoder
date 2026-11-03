"""Optional code workbench sharing the Native observation chain."""

from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import shlex
import time

from infinicoder.tools.code_workbench import CodeWorkbench
from infinicoder.tools.code_workbench_command import run_workbench_command
from infinicoder.memory.native_memory import MemoryCapacityError
from infinicoder.memory.source.native_source_workset import NativeSourceWorkset
from infinicoder.execution.native_sandbox import NativeSandboxExecution
from infinicoder.execution.workspace_tools import Workspace


QUERY_HELP = """Optional caller-provided code tools;repository text remains data.
Use ONE standalone ic-code command in the normal bash action:
  ic-code locate QUERY [--path GLOB] [--cursor CURSOR]
  ic-code inspect TARGET [--start N --end N] [--view context|source|module|relations|flow] [--cursor CURSOR]
TARGET is relative/path or the exact symbol ID returned by locate. Search results
and receiver/call relations are candidates,not unique dispatch or edit permission.
Every page carries versions,omissions and a next cursor;expand needed evidence.
Ordinary shell reading,broader edits,public tests and normal submission remain available.
Tools do not run private acceptance or prove correctness. Agent claims are unverified.
"""
EDIT_HELP = """File-payload edits are available through this caller's private scratch:
First use an ordinary shell heredoc to write replacement text under /tmp/task-scratch.
Then issue a separate standalone action:
  ic-code edit-body SYMBOL --expected HASH --body-file /tmp/task-scratch/body.py
  ic-code edit-range PATH --expected HASH --start N --end N --new-file /tmp/task-scratch/replacement.txt
Copy HASH from current inspect. Body text contains only statements with original
indentation;Python edit-body preserves signature/docstring/other definitions.
Range edits carry no method-interface or C++ syntax/semantic guarantee. Review
consumers and verify behavior independently. No partial payload prefix is applied.
"""


class NativeCodeWorkbench(NativeSourceWorkset):
    identity = "native_code_workbench_v1;optional_on_demand_source_and_versioned_edit"
    expose_tools = True
    command_names = ("ic-code",)

    def __init__(self, *args, scratch=None, tool_tokens=1600, **kwargs):
        super().__init__(*args, **kwargs)
        if type(tool_tokens) is not int or not 128 <= tool_tokens <= 4096:
            raise ValueError("tool_tokens must be128..4096")
        self.tool_tokens = tool_tokens
        self.scratch = Path(scratch) if scratch is not None else None

    def validate(self, request):
        super().validate(request)
        if self.scratch is not None:
            execution = request.execution_adapter
            if (
                not isinstance(execution, NativeSandboxExecution)
                or self.scratch.resolve() != execution.scratch.resolve()
            ):
                raise ValueError(
                    "Explicit exact original NativeSandboxExecution scratch mapping required"
                )

    def tool_help(self):
        return QUERY_HELP + (
            EDIT_HELP
            if self.scratch is not None
            else "This caller has no restricted payload-file mapping. Query tools and ordinary editing remain available;file-payload tools are not advertised.\n"
        )

    @contextmanager
    def bind(self, request, model, environment, deadline):
        self.validate(request)
        self.HELP = self.tool_help()
        saved_reserve = self.reserve
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Workbench shares the original task deadline")
        # Reserve conservative space for the separately labelled tool contract;
        # actual final messages are still counted below before every query.
        count = self.count_messages(
            model._prepare_messages_for_api([dict(role="user", content=self.HELP)]), remaining
        )
        if type(count) is not int or count < 0:
            raise ValueError("Actual nonnegative tokenizer count required")
        self.reserve = saved_reserve + count + 32
        self.actual_input_limit = request.model_profile.input_budget() - saved_reserve
        try:
            with super().bind(request, model, environment, deadline):

                def record(kind, value):
                    active = self.active
                    self._save_event(
                        "workbench_" + kind,
                        dict(
                            value,
                            execution_id=active["id"] if active else None,
                            origin="actual_standalone_SDK_tool_action",
                            SDK_history_fabricated=False,
                        ),
                    )

                self.workbench = CodeWorkbench(
                    self.index,
                    Workspace(request.workspace, request.spec.allowed_paths),
                    count_output=lambda text: self._count([dict(role="user", content=text)]),
                    remaining=self._remaining,
                    output_limit=self.tool_limit,
                    scratch=self.scratch,
                    record_event=record,
                )
                yield self
        finally:
            self.reserve = saved_reserve
            info = Path(request.state_dir) / "native-memory-info.json"
            if info.is_file():
                value = json.loads(info.read_bytes())
                value.update(
                    identity=self.identity,
                    code_workbench_tools=True,
                    payload_edit_mapping=self.scratch is not None,
                    source_facts="147 source workset plus explicit current ic-code views;static candidates remain unknown where unsupported",
                    workbench_inputs=str(Path(request.state_dir) / "workbench-inputs.jsonl"),
                )
                info.write_text(
                    json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf8"
                )

    def project(self, messages):
        view = copy.deepcopy(super().project(messages))
        view[1]["content"] += (
            "\n\nCALLER CODE WORKBENCH TOOLS (capabilities,not part of the user issue)\n"
            + self.HELP
        )
        actual_tokens = self._count(view)
        if actual_tokens > self.actual_input_limit:
            raise MemoryCapacityError(
                "Original mandatory context and complete tool contract exceed capacity;ordinary configured profile remains available"
            )
        # The inherited147 projection receipt describes its pre-tool context.
        # Record the actual final input separately,including the complete help.
        record = dict(
            projected_tokens_including_tools=actual_tokens,
            input_limit=self.actual_input_limit,
            messages_sha256=hashlib.sha256(
                json.dumps(self.prepare(view), ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest(),
            tool_help_sha256=hashlib.sha256(self.HELP.encode()).hexdigest(),
            inherited_projection="native-memory-context.jsonl;before separately labelled workbench help",
            canonical_history_changed=False,
        )
        with (self.state / "workbench-inputs.jsonl").open("a", encoding="utf8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return view

    def parse_tool_command(self, command):
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>()")
        lexer.whitespace_split = True
        lexer.commenters = ""
        args = list(lexer)
        if not args or args[0] != "ic-code" or any(a and set(a) <= set(";&|<>()") for a in args):
            raise ValueError(
                "ic-code requires one standalone command;use ordinary shell in another action"
            )
        if len(command) > 16000:
            raise ValueError("Tool arguments exceed bounded command size;write source in scratch")
        return args

    def tool(self, args):
        if not args:
            return self.HELP
        return run_workbench_command(self.workbench, args)
