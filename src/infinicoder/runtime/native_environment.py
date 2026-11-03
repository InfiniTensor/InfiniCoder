"""Sandbox base: configuration and completion, with no host executor."""

# Portions adapted from mini-SWE-agent 2.4.6 (MIT); see THIRD_PARTY_NOTICES.
import platform
from typing import Any
from pydantic import BaseModel
from infinicoder.runtime.native_protocol import Submitted, recursive_merge


class NativeEnvironmentConfig(BaseModel):
    cwd: str = ""
    env: dict[str, str] = {}
    timeout: int = 30


class NativeEnvironment:
    def __init__(self, *, config_class: type = NativeEnvironmentConfig, **kwargs):
        """Execution is provided by the task sandbox subclass."""
        self.config = config_class(**kwargs)

    def execute(self, action, **kwargs):
        raise NotImplementedError("A caller-authorized sandbox is required")

    def _check_finished(self, output: dict):
        """Raises Submitted if the output indicates task completion."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if (
            lines
            and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
            and output["returncode"] == 0
        ):
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(self.config.model_dump(), platform.uname()._asdict(), kwargs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }
