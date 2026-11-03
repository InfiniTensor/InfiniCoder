"""Versioned model hypotheses and caller-executed probes;never acceptance."""

from infinicoder._resources import package_root

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re

from infinicoder.memory.repair_memory import signature, view as repair_view
from infinicoder.planning.diagnostics import stable_failure
from infinicoder.verification.verification import Snapshot, environment_id


def identity():
    root = package_root()
    return hashlib.sha256(
        b"".join(
            (root / name).read_bytes()
            for name in (
                "planning/diagnostic_flow.py",
                "runtime/engine.py",
                "runtime/protocol.py",
                "memory/repair_memory.py",
            )
        )
    ).hexdigest()


def failure(state, spec, workspace):
    current = repair_view(state, spec, workspace)
    if not current or current.get("status") != "observed_failure_on_current_source":
        return None
    record = next(
        (
            r
            for r in state["repair_evidence"]["observations"]
            if r["observation_id"] == current["observation_id"]
        ),
        None,
    )
    checker = hashlib.sha256(
        json.dumps(state.get("checker_files", {}), sort_keys=True).encode()
    ).hexdigest()
    if record is None or record["checker_identity"] != checker:
        return None
    return record


def key(record):
    return hashlib.sha256(
        json.dumps(
            [
                record["requirements"],
                record["tested_revision"],
                record["checker_identity"],
                record["checks"],
                record["failed_checks"],
                stable_failure(record["failure"]),
            ],
            sort_keys=True,
        ).encode()
    ).hexdigest()


def request(state, spec, workspace):
    record = failure(state, spec, workspace)
    if not record:
        return None
    token = key(record)
    if token in state.get("diagnostic_attempted", []):
        return None
    due = dict(
        failure_id=record["observation_id"],
        key=token,
        source_revision=record["tested_revision"],
        reads=0,
        invalid=0,
    )
    previous = state.get("diagnostic_due")
    if previous and previous["key"] == token:
        return previous
    state["diagnostic_due"] = due
    return due


def pending(state, spec, workspace):
    due = state.get("diagnostic_due")
    if not due:
        return None
    record = failure(state, spec, workspace)
    if record is None or key(record) != due["key"]:
        state.pop("diagnostic_due", None)
        return None
    due["failure_id"] = record["observation_id"]
    return due


def completed(state, token):
    values = state.setdefault("diagnostic_attempted", [])
    if token not in values:
        values.append(token)
    del values[:-8]
    state.pop("diagnostic_due", None)


def schema():
    def obj(props, required):
        return dict(type="object", properties=props, required=required, additionalProperties=False)

    text = dict(type="string", minLength=1, maxLength=500)
    prediction = obj(
        dict(
            exit_code=dict(type=["integer", "null"], minimum=-255, maximum=255),
            contains=dict(type=["string", "null"], minLength=1, maxLength=300),
        ),
        ["exit_code", "contains"],
    )
    hypothesis = obj(
        dict(
            id=dict(type="string", pattern="^[A-Za-z][A-Za-z0-9_-]{0,23}$"),
            claim=text,
            prediction=prediction,
        ),
        ["id", "claim", "prediction"],
    )
    source = obj(
        dict(path=text, sha256=dict(type="string", pattern="^[0-9a-f]{64}$")), ["path", "sha256"]
    )
    return obj(
        dict(
            failure_id=dict(type="string", pattern="^[0-9a-f]{64}$"),
            sources=dict(type="array", minItems=1, maxItems=4, items=source),
            hypotheses=dict(type="array", minItems=2, maxItems=3, items=hypothesis),
            argv=dict(
                type="array", minItems=1, maxItems=64, items=dict(type="string", maxLength=8000)
            ),
            timeout=dict(type="number", exclusiveMinimum=0, maximum=120),
        ),
        ["failure_id", "sources", "hypotheses", "argv"],
    )


def prepare_inputs(workspace, args, reads):
    """Shared inspected-source and distinguishing-prediction checks;no failure invented."""
    sources = args["sources"]
    hypotheses = args["hypotheses"]
    argv = args["argv"]
    if not isinstance(sources, list) or not 1 <= len(sources) <= 4:
        raise ValueError("diagnose requires1..4 inspected source versions")
    seen = set()
    for source in sources:
        if not isinstance(source, dict) or set(source) != {"path", "sha256"}:
            raise ValueError("source needs path and sha256")
        path = source["path"]
        digest = source["sha256"]
        if (
            not isinstance(path, str)
            or len(path) > 500
            or not isinstance(digest, str)
            or not re.fullmatch("[0-9a-f]{64}", digest)
        ):
            raise ValueError("invalid source reference")
        if path in seen:
            raise ValueError("duplicate source reference")
        seen.add(path)
        if hashlib.sha256(workspace.path(path).read_bytes()).hexdigest() != digest:
            raise ValueError("diagnostic source changed;read its current version")
        observed = any(
            r.get("path") == path
            and r.get("source_sha256", r.get("sha256")) == digest
            and r.get("read_spans")
            for r in reads
        )
        if not observed:
            raise ValueError(
                "read the referenced source before diagnosing;model-authored hashes are not read evidence"
            )
    if not isinstance(hypotheses, list) or not 2 <= len(hypotheses) <= 3:
        raise ValueError("provide2..3 competing hypotheses")
    names = set()
    predictions = []
    for hypothesis in hypotheses:
        if not isinstance(hypothesis, dict) or set(hypothesis) != {"id", "claim", "prediction"}:
            raise ValueError("hypothesis needs id,claim,prediction")
        name = hypothesis["id"]
        claim = hypothesis["claim"]
        prediction = hypothesis["prediction"]
        if (
            not isinstance(name, str)
            or not re.fullmatch("[A-Za-z][A-Za-z0-9_-]{0,23}", name)
            or name in names
        ):
            raise ValueError("distinct bounded hypothesis IDs required")
        names.add(name)
        if not isinstance(claim, str) or not claim.strip() or len(claim) > 500:
            raise ValueError("bounded nonempty hypothesis claim required")
        if not isinstance(prediction, dict) or set(prediction) != {"exit_code", "contains"}:
            raise ValueError("prediction needs exit_code and contains;unused field is null")
        code = prediction["exit_code"]
        contains = prediction["contains"]
        if code is not None and (type(code) is not int or not -255 <= code <= 255):
            raise ValueError("invalid predicted exit code")
        if contains is not None and (
            not isinstance(contains, str) or not contains or len(contains) > 300
        ):
            raise ValueError("invalid literal output prediction")
        if code is None and contains is None:
            raise ValueError("hypothesis needs an observable prediction")
        predictions.append(json.dumps(prediction, sort_keys=True))
    if len(set(predictions)) < 2:
        raise ValueError("identical predictions cannot distinguish hypotheses")
    if (
        not isinstance(argv, list)
        or not 1 <= len(argv) <= 64
        or any(not isinstance(x, str) or len(x) > 8000 or "\x00" in x for x in argv)
        or not argv[0].strip()
        or sum(map(len, argv)) > 16000
    ):
        raise ValueError("bounded command argv required")
    timeout = args.get("timeout", 30)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or not 0 < timeout <= 120
    ):
        raise ValueError("diagnostic timeout must be finite and <=120 seconds")
    return deepcopy(sources), deepcopy(hypotheses), list(argv), timeout


def prepare(state, spec, workspace, args, reads):
    if (
        not isinstance(args, dict)
        or not {"failure_id", "sources", "hypotheses", "argv"} <= set(args)
        or set(args) - {"failure_id", "sources", "hypotheses", "argv", "timeout"}
    ):
        raise ValueError("diagnose needs failure_id,sources,hypotheses,argv and optional timeout")
    record = failure(state, spec, workspace)
    if not record or args["failure_id"] != record["observation_id"]:
        raise ValueError("diagnose must reference the current independent failure_id")
    sources, hypotheses, argv, timeout = prepare_inputs(workspace, args, reads)
    declaration = dict(
        requirements=signature(spec),
        failure_id=record["observation_id"],
        failure_key=key(record),
        revision=record["tested_revision"],
        checker_identity=record["checker_identity"],
        sources=deepcopy(sources),
        hypotheses=deepcopy(hypotheses),
        argv=list(argv),
        timeout=timeout,
        model_claims=True,
        acceptance_proof=False,
    )
    declaration["experiment_key"] = hashlib.sha256(
        json.dumps(
            [
                declaration["failure_key"],
                sorted((r["path"], r["sha256"]) for r in sources),
                argv,
                timeout,
            ],
            sort_keys=True,
        ).encode()
    ).hexdigest()
    declaration["declaration_id"] = hashlib.sha256(
        json.dumps(declaration, sort_keys=True).encode()
    ).hexdigest()
    if any(
        r["experiment_key"] == declaration["experiment_key"]
        for r in state.get("diagnostic_history", [])
    ):
        raise ValueError("identical diagnostic experiment already observed at this source version")
    return declaration


def probe(declaration, workspace, snapshot_dir, process, remaining, clamp_timeout):
    """A file snapshot plus caller executor;no stronger OS isolation is implied."""
    if remaining() <= 0:
        raise TimeoutError("no shared diagnostic budget remains")
    live = workspace.revision()
    if live != declaration["revision"]:
        raise ValueError("source changed before diagnostic experiment")
    snapshot = Snapshot(workspace.root, snapshot_dir)
    try:
        before = snapshot.revision()
        if before != live or workspace.revision() != live:
            raise ValueError("source changed while preparing diagnostic snapshot")
        if remaining() <= 0:
            raise TimeoutError("shared budget expired while copying diagnostic snapshot")

        def remap(arg):
            arg = arg.replace("{workspace}", str(snapshot.dir))
            # Resolve whole-path aliases,including Windows short names.
            try:
                path = Path(arg)
                if path.is_absolute():
                    return str(snapshot.dir / path.resolve().relative_to(workspace.root))
            except (OSError, ValueError):
                pass
            return arg.replace(str(workspace.root), str(snapshot.dir))

        argv = [remap(arg) for arg in declaration["argv"]]
        result = asdict(
            process(argv, snapshot.dir, min(clamp_timeout(declaration["timeout"]), remaining()))
        )
        unchanged = snapshot.revision() == before and workspace.revision() == live
        complete = (
            unchanged
            and not result.get("timed_out")
            and not result.get("truncated")
            and result["exit_code"] not in (124, 125)
        )
        comparisons = []
        for hypothesis in declaration["hypotheses"]:
            predicted = hypothesis["prediction"]
            matches = (
                predicted["exit_code"] is None or predicted["exit_code"] == result["exit_code"]
            ) and (predicted["contains"] is None or predicted["contains"] in result["output"])
            comparisons.append(
                dict(id=hypothesis["id"], prediction_consistent=bool(matches) if complete else None)
            )
        record = dict(
            declaration_id=declaration["declaration_id"],
            experiment_key=declaration["experiment_key"],
            failure_id=declaration["failure_id"],
            failure_key=declaration["failure_key"],
            requirements=declaration["requirements"],
            revision=live,
            checker_identity=declaration["checker_identity"],
            hypotheses=deepcopy(declaration["hypotheses"]),
            sources=deepcopy(declaration["sources"]),
            comparisons=comparisons,
            outcome="observed_prediction_comparison" if complete else "inconclusive",
            source_unchanged=unchanged,
            output_sha256=hashlib.sha256(result["output"].encode()).hexdigest(),
            exit_code=result["exit_code"],
            timed_out=result["timed_out"],
            truncated=result["truncated"],
            environment=environment_id(),
            provenance="model-designed experiment executed by caller;predictions are hypotheses,not causes or final acceptance",
            acceptance_proof=False,
        )
        return record, dict(
            executed_argv=argv,
            process=result,
            snapshot_before=before,
            snapshot_after=snapshot.revision(),
        )
    finally:
        snapshot.cleanup()


def save(state, record):
    history = state.setdefault("diagnostic_history", [])
    history.append(deepcopy(record))
    omitted = max(0, len(history) - 6)
    if omitted:
        state["diagnostic_history_omitted"] = state.get("diagnostic_history_omitted", 0) + omitted
    del history[:-6]
    completed(state, record["failure_key"])


def context(state, spec, workspace):
    due = pending(state, spec, workspace)
    rows = [r for r in state.get("diagnostic_history", []) if r["requirements"] == signature(spec)]
    value = dict(
        origin="model_hypotheses_and_caller_probe_observations;not acceptance",
        history_omitted=state.get("diagnostic_history_omitted", 0),
    )
    if due:
        value["requested"] = dict(
            failure_id=due["failure_id"],
            reads_remaining=max(0, 2 - due["reads"]),
            next="Use diagnose with2..3 hypotheses,different observable predictions and one bounded experiment;read current source if needed. Repeated verify on unchanged rejected code is not a discriminating experiment.",
        )
    if rows:
        row = rows[-1]
        value["last_probe"] = {
            k: deepcopy(row[k])
            for k in (
                "declaration_id",
                "revision",
                "hypotheses",
                "comparisons",
                "outcome",
                "output_sha256",
            )
        }
        checker = hashlib.sha256(
            json.dumps(state.get("checker_files", {}), sort_keys=True).encode()
        ).hexdigest()
        value["last_probe"]["current"] = (
            row["revision"] == workspace.revision() and row["checker_identity"] == checker
        )
        if len(json.dumps(value, ensure_ascii=False)) > 2400:
            for h in value["last_probe"]["hypotheses"]:
                h.pop("claim", None)
            value["last_probe"]["claims_omitted"] = True
        if len(json.dumps(value, ensure_ascii=False)) > 2400:
            value["last_probe"]["hypotheses"] = [
                dict(id=h["id"], predictions_omitted=True)
                for h in value["last_probe"]["hypotheses"]
            ]
    return value if due or rows else None


def observation(record, result):
    value = {
        k: deepcopy(record[k])
        for k in (
            "declaration_id",
            "failure_id",
            "comparisons",
            "outcome",
            "source_unchanged",
            "exit_code",
            "timed_out",
            "truncated",
            "output_sha256",
            "acceptance_proof",
        )
    }
    value["output"] = result["process"]["output"][:1400]
    value["output_display_truncated"] = len(result["process"]["output"]) > 1400
    value["next"] = (
        "Use this observation to repair or revisit a hypothesis;matching a model prediction never replaces the independent final checks."
    )
    while len(json.dumps(value, ensure_ascii=False)) > 4200:
        value["output"] = value["output"][: len(value["output"]) // 2]
        value["output_display_truncated"] = True
    return value
