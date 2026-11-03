"""Silent canonical observation around the unchanged native SDK input.

No source packet, tool advertisement, history compression or automatic check
is installed. Subclasses may dispatch explicit operations at before_query;
project and tokenizer preparation remain free of execution side effects.
"""

from contextlib import contextmanager
import json

from infinicoder.memory.native_episode_memory import NativeEpisodeMemory
from infinicoder.memory.native_memory import NativeRepositoryMemory, _json


class NativePlainObserver(NativeEpisodeMemory):
    identity = "plain_native_observer_v1"
    expose_tools = False

    def __init__(self, count_messages, *, tokenizer_identity, implementation_paths=()):
        super().__init__(
            count_messages,
            tokenizer_identity=tokenizer_identity,
            implementation_paths=implementation_paths,
            reserve=0,
        )
        self.public_operation = None

    def validate(self, request):
        # Whole-repository tasks can have no caller-declared observation set.
        # That is unknown source coverage, not inferred permission or closure.
        if self.paths:
            NativeEpisodeMemory.validate(self, request)
        else:
            NativeRepositoryMemory.validate(self, request)

    @contextmanager
    def bind(self, request, model, environment, deadline):
        with super().bind(request, model, environment, deadline):
            query = model.query

            def boundary(messages, **kwargs):
                self._sync(messages)
                self.before_query(messages)
                return query(messages, **kwargs)

            model.query = boundary
            try:
                yield self
            finally:
                model.query = query
        path = self.state / "native-memory-info.json"
        value = json.loads(path.read_bytes())
        value.update(
            identity=self.identity,
            projection="Original SDK messages unchanged",
            automatic_source=False,
            automatic_compression=False,
            model_tools_installed=False,
            observed_paths=self.paths,
            source_coverage="Declared paths only;empty means unknown",
        )
        working = getattr(self, "working_memory", None)
        if working is not None:
            value.update(
                projection="Exact history selected only after input pressure",
                automatic_compression=working.active,
                model_tools_installed=True,
                task_working_memory_records=working.records,
            )
        path.write_text(_json(value), encoding="utf8")

    @contextmanager
    def bind_public_operation(self, operation):
        if self.public_operation is not None:
            raise ValueError("nested public operation binding")
        self.public_operation = operation
        try:
            yield self
        finally:
            self.public_operation = None

    def before_query(self, messages):
        """Execution extension point, separate from repeatable preparation."""

    def project(self, messages):
        prepared = self._sync(messages)
        # Record the observation without preparing a second input or spending
        # tokenizer requests. The original SDK performs its normal preparation.
        with self.index.db:
            self.index.db.execute(
                "INSERT INTO native_projections(messages,accounting) VALUES(?,?)",
                (
                    _json(prepared),
                    _json(
                        dict(
                            status="plain_native",
                            input_changed=False,
                            original_messages=len(prepared),
                            automatic_operations=0,
                        )
                    ),
                ),
            )
        return messages
