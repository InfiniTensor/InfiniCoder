"""Caller-issued public correctness decoding, never inferred from model text."""

from dataclasses import dataclass
import hashlib
import json

from infinicoder.runtime.agent_execution import CallerPublicOperation


def contract_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class PublicCandidateVerdict:
    status: str
    counterexample: str = ""

    def __post_init__(self):
        if self.status not in ("public_pass", "public_fail", "unknown", "env_error"):
            raise ValueError("explicit public correctness classification required")
        if not isinstance(self.counterexample, str) or len(self.counterexample.encode()) > 8192:
            raise ValueError("bounded public counterexample required")
        if self.status == "public_fail" and not self.counterexample.strip():
            raise ValueError("a definite public failure needs its actual counterexample")


@dataclass(frozen=True)
class CandidatePublicCheck:
    """A trusted decoder of one exact, already-public caller operation.

    The decoder is application code, not a model response or a guessed command
    exit status. The controller supplies tested versions and retains raw bytes.
    A public pass covers only the operation's declared tests, never the final.
    """

    operation: CallerPublicOperation
    decoder: object
    decoder_identity: str

    def __post_init__(self):
        if (
            not isinstance(self.operation, CallerPublicOperation)
            or self.operation.source_mutation_allowed
            or not self.operation.observed_paths
        ):
            raise ValueError(
                "explicit source-preserving public operation with observed paths required"
            )
        if (
            not callable(self.decoder)
            or not isinstance(self.decoder_identity, str)
            or not self.decoder_identity.strip()
        ):
            raise ValueError("identified caller-owned public verdict decoder required")
        if len(json.dumps(self.manifest(), allow_nan=False).encode()) > 32768:
            raise ValueError("bounded public selection contract required")

    def manifest(self):
        return dict(
            operation=self.operation.manifest(),
            decoder_identity=self.decoder_identity,
            authority="caller-public correctness only;independent final is never a selector",
        )
