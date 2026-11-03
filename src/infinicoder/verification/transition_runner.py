"""One isolated candidate/reference role;never imported by the controller."""

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    parser.add_argument("--role", choices=("candidate", "reference"), required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.payload).read_bytes())
    # Public dependency imports must not resolve through candidate torch.py or
    # a workspace package. Candidate-local imports are added only afterwards.
    root = Path(payload["workspace"]).resolve()
    sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != root]
    from infinicoder.verification.public_transition_probe import (
        PublicTransitionContract,
        ObservationUnavailable,
        run_role,
    )

    contract = PublicTransitionContract(**payload["contract"])
    report = None
    with (
        open(os.devnull, "w") as sink,
        contextlib.redirect_stdout(sink),
        contextlib.redirect_stderr(sink),
    ):
        try:
            import torch
            from infinicoder.verification.torch_transition_adapters import prepare_adapter

            cases, factory = prepare_adapter(contract, payload, torch, args.role)
            report = run_role(contract, args.role, cases, factory, torch)
        except ObservationUnavailable as error:
            report = dict(
                contract_sha256=contract.identity,
                role=args.role,
                calls=[],
                bindings=[],
                observation_error=str(error),
            )
        except (SyntaxError, TypeError, AttributeError) as error:
            report = dict(
                contract_sha256=contract.identity,
                role=args.role,
                calls=[],
                bindings=[],
                setup_error=dict(type=type(error).__name__, message=str(error)[:600]),
            )
        except Exception as error:
            report = dict(
                contract_sha256=contract.identity,
                role=args.role,
                calls=[],
                bindings=[],
                environment_error=dict(type=type(error).__name__, message=str(error)[:600]),
            )
    body = json.dumps(report, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(body.encode()) > 262144:
        body = json.dumps(
            dict(
                contract_sha256=contract.identity,
                role=args.role,
                calls=[],
                observation_error="role report exceeds256KiB",
            )
        )
    print("IC_TRANSITION_REPORT:" + body)


if __name__ == "__main__":
    main()
