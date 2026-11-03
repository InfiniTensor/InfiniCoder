"""Authenticated loopback transport to an explicit caller public role service."""

import copy, hashlib, json, re, time, urllib.parse, urllib.request, uuid
from pathlib import Path

from infinicoder.runtime.agent_execution import CallerPublicOperation
from infinicoder.verification.public_check_reports import observed_versions
from infinicoder.verification.public_transition_probe import compare_transition_reports, encoded
from infinicoder.models.http_transport import open_request


def remote_transition_operation(contract, workspace, *, endpoint, authorization, server_identity):
    """Build a bounded public operation,never infer an endpoint from model text.

    The application supplies an authenticated loopback/SSH-forwarded service.
    Credentials stay in this controller closure,not metadata or model inputs.
    The registered server binds its own fixed references and isolated runtime.
    """
    parsed = urllib.parse.urlsplit(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.path != "/transition"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.port is None
    ):
        raise ValueError("explicit loopback public transition endpoint required")
    if not isinstance(authorization, str) or not re.fullmatch("[0-9a-f]{64}", authorization):
        raise ValueError("private256-bit service capability required")
    if not isinstance(server_identity, str) or not re.fullmatch("[0-9a-f]{64}", server_identity):
        raise ValueError("pinned public server identity required")
    root = Path(workspace).resolve()
    data = contract.manifest()
    paths = tuple(data["observed_paths"])

    def execute(deadline):
        versions = observed_versions(root, paths, deadline)
        if any(v["kind"] != "file" for v in versions.values()):
            raise ValueError("candidate source identity unknown")
        files = {}
        total = 0
        for name in paths:
            if time.monotonic() >= deadline:
                raise TimeoutError("public remote source observation deadline")
            path = root / name
            if any(p.is_symlink() for p in (path, *path.parents)):
                raise ValueError("public source snapshot cannot follow links")
            size = path.stat().st_size
            total += size
            if size > 1024**2 or total > 1024**2:
                raise ValueError("bounded1MiB remote source set required")
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != versions[name]["sha256"]:
                raise ValueError("source changed while preparing public request")
            files[name] = raw.decode("utf-8")
        remaining = deadline - time.monotonic()
        seconds = min(28, remaining - 0.5)
        if seconds < 0.5:
            raise TimeoutError("public remote deadline exhausted")
        identifier = uuid.uuid4().hex
        payload = dict(
            request_id=identifier,
            contract_sha256=contract.identity,
            source_files=files,
            source_versions=versions,
            timeout_seconds=seconds,
        )
        raw = encoded(payload).encode()
        if len(raw) > 2 * 1024**2:
            raise ValueError("public request exceeds2MiB")
        request = urllib.request.Request(
            endpoint,
            data=raw,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + authorization,
            },
        )
        with open_request(request, timeout=max(0.1, deadline - time.monotonic())) as response:
            body = response.read(2 * 1024**2 + 1)
            if response.status != 200 or len(body) > 2 * 1024**2:
                raise ValueError("bounded public role response required")
        value = json.loads(body)
        if value.get("request_id") != identifier or value.get("server_identity") != server_identity:
            raise ValueError("public server identity/request binding mismatch")
        result = value["result"]
        roles = result.get("role_reports", {})
        report = compare_transition_reports(
            contract, roles.get("reference"), roles.get("candidate"), versions
        )
        permitted = data["execution_operation"].get("cuda_visible_devices")
        if permitted:
            for role in roles.values():
                for call in role.get("calls", []):
                    if (
                        call.get("binding_after", {}).get("actual_device_uuid", "").lower()
                        != permitted.lower()
                    ):
                        report.update(
                            status="unknown", reason="remote_actual_device_binding_missing"
                        )
        if observed_versions(root, paths, deadline) != versions:
            report.update(status="unknown", reason="live_sources_changed_during_remote_probe")
        result = copy.deepcopy(result)
        result["transition_report"] = report
        result["remote_request_id"] = identifier
        result["remote_server_identity"] = server_identity
        return result

    return CallerPublicOperation(
        data["execution_operation"]["id"],
        root,
        dict(
            transition_contract_sha256=contract.identity,
            public_server_identity=server_identity,
            transport_module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ),
        "explicit remote public numerical role service",
        data["coverage"],
        30,
        dict(
            transport="authenticated loopback/SSH forward",
            server_identity=server_identity,
            ordinary_shell_GPU=False,
            device=data["execution_operation"].get("cuda_visible_devices"),
            endpoint_scope="caller-provided loopback only",
        ),
        paths,
        ("bounded remote source snapshot and public diagnostic artifacts",),
        600,
        execute,
    )
