"""Caller-pinned single-GPU public diagnostics in isolated Docker roles.

This provider never calls a benchmark grader or changes ordinary shell access.
Its device,image,public package directory and shared lock are caller inputs.
"""

from infinicoder._resources import package_root

import fcntl, hashlib, json, os, re, shutil, stat, subprocess, time, uuid
from pathlib import Path
from contextlib import contextmanager

from infinicoder.runtime.agent_execution import CallerPublicOperation
from infinicoder.verification.public_check_reports import observed_versions
from infinicoder.verification.public_transition_probe import compare_transition_reports, encoded
from infinicoder.verification.transition_public_cpu import _public_reference_bytes
from infinicoder.verification.transition_reports import PublicTransitionCapability
from infinicoder.execution.workspace_tools import run_process


class PublicDockerTransitionOperation:
    def __init__(
        self,
        contract,
        reference_script,
        workspace,
        state,
        *,
        image_id,
        device_uuid,
        public_site,
        shared_lock,
        runtime_files,
    ):
        if not re.fullmatch("sha256:[0-9a-f]{64}", image_id) or not re.fullmatch(
            "GPU-[0-9a-fA-F-]{36}", device_uuid
        ):
            raise ValueError("pinned image SHA256 and explicit GPU UUID required")
        self.docker = shutil.which("docker")
        if self.docker is None:
            raise ValueError("existing Docker executable required;no installation fallback")
        self.contract = contract
        self.data = contract.manifest()
        self.reference = Path(reference_script).absolute()
        self.raw = None
        self.workspace = Path(workspace).resolve()
        self.state = Path(state).resolve()
        self.package = package_root().resolve()
        self.site = Path(public_site).resolve()
        self.lock = Path(shared_lock).absolute()
        self.image = image_id
        self.device = device_uuid
        schema = self.data["public_input_schema"]
        if (
            schema.get("device") != "cuda:0"
            or self.data["execution_operation"].get("cuda_visible_devices") != device_uuid
            or self.data["candidate_entry_binding"]["adapter_id"] != "torch_dictionary_tensor_v1"
        ):
            raise ValueError("explicit CUDA dictionary contract and same UUID required")
        for path in (
            self.site,
            self.lock,
            *self.workspace.parents,
            self.workspace,
            self.state,
            self.package,
        ):
            if any(c in str(path) for c in (",", "\0", "\n")):
                raise ValueError("mount/lock paths cannot contain delimiters")
        if (
            not self.site.is_dir()
            or len(self.site.parts) < 4
            or not (self.site / "torch/__init__.py").is_file()
        ):
            raise ValueError("explicit installed public Python site directory required")
        for private in (self.workspace, self.state):
            if private.is_relative_to(self.site) or self.site.is_relative_to(private):
                raise ValueError("runtime overlaps source/state")
        if (
            not self.lock.is_file()
            or any(p.is_symlink() for p in (self.lock, *self.lock.parents))
            or self.lock.stat().st_uid != os.getuid()
            or not stat.S_ISREG(self.lock.stat().st_mode)
        ):
            raise ValueError("existing task-owned shared GPU lock required")
        self.runtime_files = dict(runtime_files)
        if set(self.runtime_files) != {
            "torch/__init__.py",
            "torch/version.py",
            "triton/__init__.py",
        }:
            raise ValueError("explicit public Torch/Triton source identity required")
        if any(not re.fullmatch("[0-9a-f]{64}", p) for p in self.runtime_files.values()):
            raise ValueError("runtime SHA256 required")
        self.code = {
            n: hashlib.sha256((self.package / n).read_bytes()).hexdigest()
            for n in (
                "verification/public_transition_docker.py",
                "verification/public_transition_probe.py",
                "verification/torch_transition_adapters.py",
                "verification/transition_runner.py",
                "verification/reference_compare.py",
            )
        }
        self._validate(time.monotonic() + 10)
        isolation = dict(
            network="none",
            root="readonly",
            capabilities="drop_ALL",
            no_new_privileges=True,
            pids=128,
            memory="8g",
            cpus=4,
            tmpfs="512m",
        )
        operation = CallerPublicOperation(
            self.data["execution_operation"]["id"],
            self.workspace,
            dict(
                transition_contract_sha256=contract.identity,
                provider_modules=self.code,
                runtime_files=self.runtime_files,
            ),
            "finite public CUDA dictionary and returned-value diagnostics",
            self.data["coverage"],
            30,
            dict(
                image_id=image_id,
                cuda_visible_devices=device_uuid,
                public_site=str(self.site),
                shared_lock=str(self.lock),
                runtime_source_pins=self.runtime_files,
                isolation=isolation,
                ordinary_shell_GPU=False,
                dependency_closure="Existing declared public runtime;not a full binary dependency proof",
            ),
            tuple(self.data["observed_paths"]),
            ("task-owned source snapshots and two ephemeral isolated containers",),
            600,
            self.execute,
        )
        self.capability = PublicTransitionCapability(contract, operation)

    def _validate(self, deadline):
        self.raw = _public_reference_bytes(
            self.reference,
            self.workspace,
            self.state,
            [Path("/usr"), Path("/lib"), Path("/lib64"), self.site],
            self.data,
            self.raw,
        )
        for name, pin in self.code.items():
            if time.monotonic() >= deadline:
                raise TimeoutError("public Docker identity deadline")
            if hashlib.sha256((self.package / name).read_bytes()).hexdigest() != pin:
                raise ValueError("public diagnostic code changed")
        for name, pin in self.runtime_files.items():
            path = self.site / name
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024**2:
                raise ValueError("bounded public runtime source file required")
            if hashlib.sha256(path.read_bytes()).hexdigest() != pin:
                raise ValueError("public runtime identity changed")

    def validate(self, request):
        self._validate(time.monotonic() + 10)
        self.capability.validate(request)

    @contextmanager
    def _lease(self, deadline):
        fd = os.open(self.lock, os.O_RDWR | os.O_NOFOLLOW)
        with os.fdopen(fd, "r+") as lock:
            if os.fstat(lock.fileno()).st_ino != self.lock.stat().st_ino:
                raise ValueError("GPU lock replaced")
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError("shared GPU lock deadline")
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(min(0.05, max(0, deadline - time.monotonic())))
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _role(self, folder, workspace, payload, reference, role, deadline):
        name = "ic-public-transition-" + uuid.uuid4().hex
        argv = [
            self.docker,
            "run",
            "--pull",
            "never",
            "--name",
            name,
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "128",
            "--memory",
            "8g",
            "--cpus",
            "4",
            "--user",
            str(os.getuid()) + ":" + str(os.getgid()),
            "--gpus",
            "device=" + self.device,
            "--tmpfs",
            "/tmp:rw,nosuid,exec,size=512m",
        ]
        env = dict(
            HOME="/tmp",
            USER="infinicoder",
            LOGNAME="infinicoder",
            PYTHONPATH="/opt/python-site",
            PYTHONNOUSERSITE="1",
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            CUDA_VISIBLE_DEVICES=self.device,
            TORCHINDUCTOR_COMPILE_THREADS="2",
            TORCHINDUCTOR_CACHE_DIR="/tmp/inductor",
        )
        for key, value in env.items():
            argv += ["--env", key + "=" + value]
        mounts = [(Path(p), p, True) for p in ("/usr", "/lib", "/lib64")]
        mounts += [
            (self.site, "/opt/python-site", True),
            (workspace, "/workspace", False),
            (self.package, "/tmp/ic-transition-runtime/infinicoder", True),
            (payload, "/tmp/ic-transition-contract.json", True),
            (reference, "/tmp/ic-transition-reference.py", True),
        ]
        for source, target, readonly in mounts:
            argv += [
                "--mount",
                "type=bind,src="
                + str(source)
                + ",dst="
                + target
                + (",readonly" if readonly else ""),
            ]
        bootstrap = "import runpy,sys;sys.path[:0]=['/tmp/ic-transition-runtime','/opt/python-site'];runpy.run_module('infinicoder.verification.transition_runner',run_name='__main__')"
        argv += [
            "--workdir",
            "/workspace",
            self.image,
            "/usr/bin/python3",
            "-I",
            "-S",
            "-B",
            "-c",
            bootstrap,
            "--payload",
            "/tmp/ic-transition-contract.json",
            "--role",
            role,
        ]
        timeout = deadline - time.monotonic() - 2
        if timeout <= 0:
            raise TimeoutError("no role and cleanup allowance remains")
        process = None
        cleanup = {}
        try:
            process = run_process(
                argv, workspace, timeout=timeout, max_output=524288, spool_cap=1024**2
            )
        finally:
            # Clean only this exact generated name,never another container.
            remaining = deadline - time.monotonic()
            if remaining > 0:
                try:
                    subprocess.run(
                        [self.docker, "rm", "-f", name],
                        capture_output=True,
                        timeout=min(1, remaining),
                    )
                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        check = subprocess.run(
                            [self.docker, "ps", "-aq", "--filter", "name=^/" + name + "$"],
                            capture_output=True,
                            timeout=min(1, remaining),
                        )
                        cleanup = dict(
                            confirmed_absent=check.returncode == 0 and not check.stdout.strip()
                        )
                except subprocess.TimeoutExpired:
                    cleanup = dict(confirmed_absent=False, reason="cleanup timed out")
        value = dict(
            container_name=name, process=process.__dict__ if process else None, cleanup=cleanup
        )
        (folder / (role + "-process.json")).write_text(encoded(value) + "\n")
        lines = (
            [
                line[len("IC_TRANSITION_REPORT:") :]
                for line in process.output.splitlines()
                if line.startswith("IC_TRANSITION_REPORT:")
            ]
            if process
            else []
        )
        if (
            not process
            or process.exit_code != 0
            or process.timed_out
            or process.truncated
            or len(lines) != 1
            or not cleanup.get("confirmed_absent")
        ):
            return dict(
                contract_sha256=self.contract.identity,
                role=role,
                calls=[],
                environment_error="public isolated CUDA role failed or cleanup unconfirmed",
            ), process
        return json.loads(lines[0]), process

    def execute(self, deadline):
        self._validate(deadline)
        versions = observed_versions(self.workspace, tuple(self.data["observed_paths"]), deadline)
        if any(v["kind"] != "file" for v in versions.values()):
            raise ValueError("candidate source identity unknown")
        root = self.state / "transition-docker-runs"
        root.mkdir(exist_ok=True)
        folder = root / uuid.uuid4().hex
        folder.mkdir()
        workspace = folder / "workspace"
        workspace.mkdir()
        total = 0
        for name in self.data["observed_paths"]:
            if time.monotonic() >= deadline:
                raise TimeoutError("source snapshot deadline")
            source = self.workspace / name
            target = workspace / name
            if any(p.is_symlink() for p in (source, *source.parents)):
                raise ValueError("candidate snapshot cannot follow links")
            size = source.stat().st_size
            total += size
            if size > 1024**2 or total > 8 * 1024**2:
                raise ValueError("bounded candidate source snapshot required")
            raw = source.read_bytes()
            if hashlib.sha256(raw).hexdigest() != versions[name]["sha256"]:
                raise ValueError("candidate changed while copying")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        reference = folder / "public-reference.py"
        reference.write_bytes(self.raw)
        reference.chmod(0o400)
        payload = folder / "payload.json"
        payload.write_text(
            encoded(
                dict(
                    contract=self.data,
                    source_versions=versions,
                    workspace="/workspace",
                    public_reference_path="/tmp/ic-transition-reference.py",
                )
            )
            + "\n"
        )
        payload.chmod(0o400)
        roles = {}
        processes = []
        with self._lease(deadline):
            for role in ("reference", "candidate"):
                roles[role], process = self._role(
                    folder, workspace, payload, reference, role, deadline
                )
                processes.append(process)
                (folder / (role + "-report.json")).write_text(encoded(roles[role]) + "\n")
        result = compare_transition_reports(
            self.contract, roles["reference"], roles["candidate"], versions
        )
        if (
            observed_versions(self.workspace, tuple(self.data["observed_paths"]), deadline)
            != versions
        ):
            result.update(status="unknown", reason="live_sources_changed")
        (folder / "comparison.json").write_text(encoded(result) + "\n")
        summary = dict(
            passed=int(result["status"] == "observed_values_match"),
            failed=int(result["status"] in ("counterexample", "candidate_error")),
            error=int(result["status"] in ("unknown", "env_error")),
        )
        return dict(
            transition_report=result,
            summary=summary,
            role_reports=roles,
            artifacts=str(folder),
            independent_final_used=False,
            process=dict(
                exit_code=0 if summary["passed"] else 1,
                timed_out=any(p and p.timed_out for p in processes),
                truncated=any(p and p.truncated for p in processes),
            ),
        )
