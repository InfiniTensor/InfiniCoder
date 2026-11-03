"""Optional source evidence on the original candidate lifecycle, never a judge."""

from infinicoder._resources import package_root

from dataclasses import asdict
import hashlib
import json


def identity():
    root = package_root()
    return hashlib.sha256(
        b"".join(
            (root / name).read_bytes()
            for name in (
                "domains/operator/candidate_io.py",
                "analysis/tensor_io.py",
                "analysis/tensor_dataflow.py",
                "runtime/engine.py",
                "runtime/feedback.py",
                "runtime/shell_loop.py",
            )
        )
    ).hexdigest()


def target(runtime, worker, state=None):
    if runtime.candidate_io_policy is None:
        return None
    paths = (
        runtime.optimization.paths
        if runtime.optimization is not None
        else (worker,)
        if worker
        else ()
    )
    if not paths and runtime.conversation_policy and state:
        declaration = (
            state.get("continuous_conversation", {})
            .get("current_snapshot", {})
            .get("implementation_targets", {})
        )
        if declaration.get("status") == "known":
            paths = tuple(declaration.get("hashes", {}))
    return paths[0] if len(paths) == 1 else None


def capture(runtime, workspace, state, path):
    if runtime.candidate_io_policy is None or not path:
        return None
    try:
        p = workspace.path(path, write=True)
        if (
            p.suffix != ".py"
            or str(p.resolve()) in state.get("checker_files", {})
            or p.is_symlink()
        ):
            return None
        with p.open("rb") as f:
            data = f.read(32769)
        if len(data) > 32768:
            return dict(path=path, status="unknown_source_bound", sha256=None)
        return dict(path=path, status="observed", sha256=hashlib.sha256(data).hexdigest())
    except FileNotFoundError:
        return dict(path=path, status="missing", sha256=None)
    except (OSError, ValueError):
        return dict(path=path, status="unknown", sha256=None)


def observe(runtime, spec, state, workspace, before, result):
    if not before:
        return result
    after = capture(runtime, workspace, state, before["path"])
    if not after:
        return result
    packet = dict(
        origin="actual_candidate_source",
        path=before["path"],
        candidate_sha256=after["sha256"],
        before_sha256=before["sha256"],
        rules=identity(),
        call=state.get("model_calls"),
        step=state["steps"],
        phase=state["milestone"],
        requirements=hashlib.sha256(
            json.dumps(asdict(spec), sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        acceptance_proof=False,
        assumptions=["Static library binding assumptions;not runtime or numerical verification."],
        unknowns=[
            "Reachability,aliasing,other writes,mask coverage and complete output initialization are unproved."
        ],
    )
    if before["status"] not in ("observed", "missing"):
        packet.update(status="unknown", reason="pre-action source identity unknown")
    elif after["status"] != "observed":
        packet.update(status=after["status"])
    elif before["sha256"] == after["sha256"]:
        packet.update(status="unchanged", analysis_recomputed=False)
        prior = next(
            (
                h["observation"]["candidate_io_observation"]
                for h in reversed(state.get("shell_loop_history", []))
                if h.get("observation", {})
                .get("candidate_io_observation", {})
                .get("candidate_sha256")
                == after["sha256"]
                and h["observation"]["candidate_io_observation"].get("rules") == packet["rules"]
            ),
            None,
        )
        if prior:
            packet["prior_analysis_call"] = prior.get("prior_analysis_call", prior["call"])
        else:
            return result
    else:
        try:
            p = workspace.path(before["path"], write=True)
            with p.open("rb") as f:
                data = f.read(32769)
            if len(data) > 32768 or hashlib.sha256(data).hexdigest() != after["sha256"]:
                raise ValueError("source changed during observation")
            from infinicoder.analysis.tensor_dataflow import analyze_io_edges

            analysis = analyze_io_edges(
                data.decode("utf8"), path=before["path"], source_sha256=after["sha256"]
            )
            runtime.store.event("candidate_io_analysis", dict(packet, report=analysis))
            packet.update(
                status=analysis["status"],
                analysis_recomputed=True,
                analysis_event_seq=runtime.store._seq,
                source_ref=analysis["source_ref"],
                relations=analysis["relations"][:1],
                assumptions=analysis["assumptions"],
                unknowns=analysis["unknowns"],
            )
        except (OSError, ValueError, UnicodeError, RecursionError):
            packet.update(
                status="unknown", reason="candidate unavailable or changed before analysis"
            )
    if len(json.dumps(packet, ensure_ascii=False)) > 3500:
        packet.pop("relations", None)
        packet.update(
            status="omitted_source_report_bound",
            reason="Full analysis remains in original analysis event;no relation subset displayed",
        )
    result["candidate_io_observation"] = packet
    return result


def retained(runtime, workspace, state, result):
    packet = result.get("candidate_io_observation")
    if packet:
        current = capture(runtime, workspace, state, packet["path"])
        sha = current.get("sha256") if current else None
        packet.update(
            retained_sha256=sha,
            candidate_still_current=(sha == packet["candidate_sha256"])
            if sha and packet["candidate_sha256"]
            else None,
            scope="At this action end only;candidate_still_current is historical,not a guarantee after later actions. Not correctness or performance acceptance.",
        )
    return result
