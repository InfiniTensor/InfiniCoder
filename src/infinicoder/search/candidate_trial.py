"""Version-bound candidate experiments; execution belongs to the caller sandbox.

Observations here are model-authored experiments, never independent acceptance.
This module has no shell executor, model client, task recipe or search loop.
"""

from dataclasses import dataclass
from contextlib import contextmanager
import difflib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import stat
import time

from infinicoder.verification.verification import SNAPSHOT_IGNORE


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True).encode()


@dataclass(frozen=True)
class TrialLimits:
    trials: int = 2
    seconds: float = 180
    files: int = 4096
    bytes: int = 64 * 1024**2
    file_bytes: int = 8 * 1024**2
    changed_files: int = 32

    def __post_init__(self):
        if any(
            type(getattr(self, key)) is not int or getattr(self, key) <= 0
            for key in ("trials", "files", "bytes", "file_bytes", "changed_files")
        ):
            raise ValueError("positive bounded trial limits required")
        if isinstance(self.seconds, bool) or not 0 < self.seconds <= 180:
            raise ValueError("trial seconds must be positive and at most180")
        if self.trials > 2:
            raise ValueError("at most two candidate trials per task")


class CandidateTrials:
    """One task-owned store, with current-action revocation on every operation.

    runner(snapshot, script_path, cutoff, phase) must execute in the caller's
    explicitly admitted sandbox. A callable alone is not execution admission;
    the native adapter validates the runner before constructing this service.
    generated_paths is trusted caller configuration, never a model argument.
    """

    def __init__(
        self,
        workspace,
        directory,
        *,
        read_payload,
        runner,
        remaining,
        task_deadline,
        generated_paths=(),
        limits=None,
        record=None,
    ):
        self.workspace = workspace
        self.directory = Path(directory).resolve()
        if (
            self.directory.is_relative_to(workspace.root)
            or workspace.root.is_relative_to(self.directory)
            or any(path.is_symlink() for path in (self.directory, *self.directory.parents))
        ):
            raise ValueError("trial state must be separate and without linked parents")
        self.directory.mkdir(mode=0o700, exist_ok=False)
        self.read_payload, self.runner, self.remaining = read_payload, runner, remaining
        self.task_deadline, self.limits = task_deadline, limits or TrialLimits()
        self.record = record or (lambda kind, value: None)
        self.generated = tuple(self._relative(path) for path in generated_paths)
        self.used = 0
        self._write(
            self.directory / "owner.json",
            dict(
                workspace=str(workspace.root),
                max_trials=self.limits.trials,
                generated_paths=list(self.generated),
                authority="model_authored_experiment;not_public_or_final_acceptance",
                identity_scope="paths,regular bytes,modes,contained links;external state is unknown",
            ),
        )

    @staticmethod
    def _relative(value):
        if not isinstance(value, str):
            raise ValueError("relative path must be text")
        path = PurePosixPath(value)
        if (
            not isinstance(value, str)
            or not value
            or "\\" in value
            or "\0" in value
            or not path.parts
            or path.is_absolute()
            or ".." in path.parts
            or str(path) != value
            or any(part in (".git", ".infinicoder") for part in path.parts)
        ):
            raise ValueError("normalized relative source/output path required")
        return value

    @staticmethod
    def _write(path, value):
        temporary = path.with_suffix(path.suffix + ".next")
        temporary.write_bytes(encoded(value))
        temporary.replace(path)

    def _guard(self, cutoff=None):
        self.remaining()
        if time.monotonic() >= min(self.task_deadline, cutoff or self.task_deadline):
            raise TimeoutError("candidate trial shares the original task/action deadline")

    def _read(self, path, cutoff=None):
        self._guard(cutoff)
        if path.is_relative_to(self.workspace.root):
            with self._parent(path.relative_to(self.workspace.root).as_posix(), write=False) as (
                parent,
                name,
            ):
                raw, mode = self._at(parent, name)
            if raw is None:
                raise ValueError("source disappeared during trial read")
            self._guard(cutoff)
            return raw, mode
        if any(parent.is_symlink() for parent in (path, *path.parents)):
            raise ValueError("regular payload/source without linked components required")
        fd = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > self.limits.file_bytes:
                raise ValueError("source exceeds the explicit regular-file trial bound")
            chunks, size = [], 0
            while True:
                self._guard(cutoff)
                block = stream.read(min(65536, self.limits.file_bytes - size + 1))
                if not block:
                    break
                chunks.append(block)
                size += len(block)
                if size > self.limits.file_bytes:
                    raise ValueError("source grew beyond trial bound")
            after = os.fstat(stream.fileno())
        identity = lambda info: (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
        if identity(before) != identity(after) or identity(after) != identity(path.lstat()):
            raise ValueError("source changed during trial read")
        return b"".join(chunks), stat.S_IMODE(after.st_mode)

    def _scan(self, root, cutoff=None):
        self._guard(cutoff)
        result = {}
        total = 0
        for folder, directories, files in os.walk(root, followlinks=False):
            self._guard(cutoff)
            ignored = set(SNAPSHOT_IGNORE(folder, directories + files))
            for name in tuple(directories) + tuple(files):
                path = Path(folder, name)
                relative = path.relative_to(root).as_posix()
                if name in ignored or any(
                    relative == item or relative.startswith(item + "/") for item in self.generated
                ):
                    if name in directories:
                        directories.remove(name)
                    continue
                info = path.lstat()
                mode = stat.S_IMODE(info.st_mode)
                if stat.S_ISLNK(info.st_mode):
                    if name in directories:
                        directories.remove(name)
                    lexical = Path(os.path.abspath(path.parent / os.readlink(path)))
                    if not lexical.is_relative_to(root) or not path.resolve().is_relative_to(root):
                        raise ValueError("trial link leaves its copied source tree")
                    result[relative] = dict(
                        kind="link", target=lexical.relative_to(root).as_posix()
                    )
                elif stat.S_ISDIR(info.st_mode):
                    result[relative] = dict(kind="directory", mode=mode)
                elif stat.S_ISREG(info.st_mode):
                    raw, checked_mode = self._read(path, cutoff)
                    total += len(raw)
                    if total > self.limits.bytes:
                        raise ValueError("workspace exceeds caller trial byte bound")
                    result[relative] = dict(
                        kind="file", sha256=digest(raw), bytes=len(raw), mode=checked_mode
                    )
                else:
                    raise ValueError("special files are unavailable in source trials")
                if len(result) > self.limits.files:
                    raise ValueError("workspace exceeds caller trial entry bound")
        return result

    def _copy(self, source, target, manifest, cutoff):
        target.mkdir(mode=0o700)
        for name, fact in sorted(manifest.items(), key=lambda item: (item[0].count("/"), item[0])):
            self._guard(cutoff)
            path = target / name
            if fact["kind"] == "directory":
                path.mkdir(mode=fact["mode"])
                path.chmod(fact["mode"])
            elif fact["kind"] == "link":
                path.symlink_to(os.path.relpath(target / fact["target"], path.parent))
            else:
                raw, mode = self._read(source / name, cutoff)
                if digest(raw) != fact["sha256"] or mode != fact["mode"]:
                    raise ValueError("source changed while copying a trial")
                with path.open("xb") as stream:
                    stream.write(raw)
                path.chmod(mode)
                info = (source / name).stat()
                os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
        if self._scan(source, cutoff) != manifest or self._scan(target, cutoff) != manifest:
            raise ValueError("trial copy/source identities differ")

    def _load(self, identity):
        if (
            not isinstance(identity, str)
            or len(identity) != 32
            or any(c not in "0123456789abcdef" for c in identity)
        ):
            raise ValueError("use the exact task-owned trial ID")
        folder = self.directory / identity
        result = json.loads((folder / "result.json").read_bytes())
        if result.get("trial_id") != identity:
            raise ValueError("saved trial identity changed")
        return folder, result

    def trial(self, edit_script, probe_script):
        self._guard()
        if self.used >= self.limits.trials:
            return dict(
                status="quota_exhausted", trials_used=self.used, max_trials=self.limits.trials
            )
        self.used += 1
        identity = secrets.token_hex(16)
        folder = self.directory / identity
        folder.mkdir(mode=0o700)
        started = time.monotonic()
        cutoff = min(
            started + self.limits.seconds,
            started + self.remaining(),
            started + max(0, self.task_deadline - started) / 3,
        )
        result = dict(
            trial_id=identity,
            status="started",
            comparable=False,
            apply_available=False,
            trials_used=self.used,
            started_monotonic=started,
            cutoff=cutoff,
            steps=[],
            authority="model_authored_experiment;not_public_or_final_acceptance",
            source_identity_scope="copied paths/bytes/modes/links;external dependencies/state unknown",
        )
        self._write(folder / "result.json", result)
        try:
            edit, edit_proof = self.read_payload(edit_script)
            probe, probe_proof = self.read_payload(probe_script)
            result.update(edit_script=edit_proof, probe_script=probe_proof)
            (folder / "edit.sh").write_bytes(edit.encode())
            (folder / "probe.sh").write_bytes(probe.encode())
            baseline = self._scan(self.workspace.root, cutoff)
            self._write(folder / "baseline.json", baseline)
            self._copy(self.workspace.root, folder / "base", baseline, cutoff)
            self._copy(self.workspace.root, folder / "candidate", baseline, cutoff)
            result["baseline_sha256"] = digest(encoded(baseline))

            def run(name, snapshot, script):
                self._guard(cutoff)
                observation = self.runner(snapshot, script, cutoff, name)
                if (
                    not isinstance(observation, dict)
                    or observation.get("authority") != "sandbox_execution_observation"
                ):
                    raise ValueError("admitted snapshot runner observation required")
                if (
                    type(observation.get("returncode")) is not int
                    or type(observation.get("timed_out")) is not bool
                    or type(observation.get("truncated")) is not bool
                    or not isinstance(observation.get("output_sha256"), str)
                    or len(observation["output_sha256"]) != 64
                ):
                    raise ValueError("complete process observation metadata required")
                if observation.get("phase") not in (None, name):
                    raise ValueError("snapshot observation phase differs from its request")
                result["steps"].append(dict(observation, phase=name))
                self._guard(cutoff)
                return observation

            edit_result = run("edit", folder / "candidate", folder / "edit.sh")
            candidate = self._scan(folder / "candidate", cutoff)
            self._write(folder / "candidate.json", candidate)
            result["candidate_sha256"] = digest(encoded(candidate))
            changes = []
            for name in sorted(set(baseline) | set(candidate)):
                before, after = baseline.get(name), candidate.get(name)
                if before != after:
                    changes.append(dict(path=name, before=before, after=after))
            result["changes"] = changes
            payloads = folder / "payloads"
            payloads.mkdir()
            diff = []
            for change in changes:
                name, before, after = change["path"], change["before"], change["after"]
                if after and after["kind"] == "file":
                    raw, _ = self._read(folder / "candidate" / name, cutoff)
                    path = payloads / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(raw)
                if (before is None or before["kind"] == "file") and (
                    after is None or after["kind"] == "file"
                ):
                    old = self._read(folder / "base" / name, cutoff)[0] if before else b""
                    new = self._read(folder / "candidate" / name, cutoff)[0] if after else b""
                    try:
                        diff.extend(
                            difflib.unified_diff(
                                old.decode("utf8").splitlines(True),
                                new.decode("utf8").splitlines(True),
                                fromfile="base/" + name,
                                tofile="candidate/" + name,
                            )
                        )
                    except UnicodeError:
                        diff.append("Binary source change " + name + "\n")
                else:
                    diff.append("Source metadata change " + name + "\n")
            (folder / "diff.txt").write_text("".join(diff), encoding="utf8")
            if edit_result.get("returncode") != 0 or edit_result.get("timed_out"):
                result["status"] = "edit_failed"
            else:
                baseline_observation = run("base_probe", folder / "base", folder / "probe.sh")
                candidate_observation = run(
                    "candidate_probe", folder / "candidate", folder / "probe.sh"
                )
                base_unchanged = self._scan(folder / "base", cutoff) == baseline
                candidate_unchanged = self._scan(folder / "candidate", cutoff) == candidate
                live_unchanged = self._scan(self.workspace.root, cutoff) == baseline
                result.update(
                    base_probe_source_unchanged=base_unchanged,
                    candidate_probe_source_unchanged=candidate_unchanged,
                    live_source_unchanged=live_unchanged,
                    outputs_equal=baseline_observation.get("output_sha256")
                    == candidate_observation.get("output_sha256"),
                )
                result["comparable"] = bool(
                    base_unchanged
                    and candidate_unchanged
                    and live_unchanged
                    and all(
                        not step.get("timed_out")
                        and not step.get("truncated")
                        and not step.get("revoked")
                        for step in result["steps"]
                    )
                )
                result["status"] = (
                    "complete" if result["comparable"] else "changed_or_incomplete_observation"
                )
                file_changes = [
                    change for change in changes if (change["after"] or {}).get("kind") == "file"
                ]
                supported = all(
                    change["after"] is not None
                    and (
                        change["after"]["kind"] == "directory"
                        and change["before"] is None
                        and not change["after"]["mode"] & 0o7000
                        or change["after"]["kind"] == "file"
                        and not change["after"]["mode"] & 0o7000
                        and (
                            change["before"] is None
                            or change["before"]["kind"] == "file"
                            and change["before"]["mode"] == change["after"]["mode"]
                        )
                    )
                    for change in changes
                )
                result["apply_available"] = bool(
                    result["comparable"]
                    and supported
                    and file_changes
                    and len(file_changes) <= self.limits.changed_files
                )
                result["automatic_apply"] = False
            self._guard(cutoff)
        except (OSError, ValueError, TimeoutError) as error:
            result.update(
                status="unavailable",
                comparable=False,
                apply_available=False,
                error=type(error).__name__ + ": " + str(error),
            )
        finally:
            result["wall_s"] = time.monotonic() - started
            result["finished_monotonic"] = time.monotonic()
            if result["finished_monotonic"] >= cutoff:
                result.update(status="expired", comparable=False, apply_available=False)
            self._write(folder / "result.json", result)
            self.record("trial", result)
        return result

    def records(self, identity, part="result"):
        self._guard()
        folder, result = self._load(identity)
        if part == "diff":
            text = (
                (folder / "diff.txt").read_text(encoding="utf8")
                if (folder / "diff.txt").is_file()
                else ""
            )
            return [
                dict(kind="trial_diff", line=i + 1, text=line)
                for i, line in enumerate(text.splitlines())
            ]
        if part == "logs":
            records = []
            for step in result["steps"]:
                self._guard()
                path = Path(step["output_ref"])
                if path.parent != folder or path.name != step["phase"] + ".output.bin":
                    raise ValueError("log reference leaves this trial")
                raw, _ = self._read(path)
                if digest(raw) != step["output_sha256"]:
                    raise ValueError("recorded experiment log changed")
                if raw:
                    records.append(
                        dict(
                            kind="trial_log_source",
                            phase=step["phase"],
                            raw_output_sha256=step["output_sha256"],
                        )
                    )
                for i, line in enumerate(raw.decode("utf8", errors="replace").splitlines()):
                    for offset in range(0, max(1, len(line)), 128):
                        records.append(
                            dict(
                                kind="trial_log",
                                phase=step["phase"],
                                line=i + 1,
                                character_offset=offset,
                                text=line[offset : offset + 128],
                            )
                        )
            return records
        if part != "result":
            raise ValueError("trial part must be result,diff or logs")
        summary = ("trial_id", "status", "comparable", "apply_available", "trials_used")
        records = [dict(kind="trial_result", **{key: result[key] for key in summary})]

        def details(kind, values, **identity):
            for field, value in values.items():
                raw = json.dumps(value, ensure_ascii=False, sort_keys=True)
                if len(raw) <= 128:
                    records.append(dict(kind=kind, field=field, value=value, **identity))
                else:
                    # Fragments are explicitly JSON text,never partial facts.
                    # Cursors recover the complete bound value in order.
                    for offset in range(0, len(raw), 128):
                        records.append(
                            dict(
                                kind=kind,
                                field=field,
                                json_fragment=raw[offset : offset + 128],
                                character_offset=offset,
                                total_characters=len(raw),
                                **identity,
                            )
                        )

        details(
            "trial_detail",
            {
                key: value
                for key, value in result.items()
                if key not in (*summary, "changes", "steps")
            },
        )
        for ordinal, item in enumerate(result.get("changes", [])):
            details("trial_change_detail", item, change_index=ordinal)
        for step in result["steps"]:
            records.append(
                dict(
                    kind="trial_observation",
                    **{
                        key: step.get(key)
                        for key in ("phase", "returncode", "timed_out", "revoked", "truncated")
                    },
                )
            )
            details(
                "trial_observation_detail",
                {
                    key: value
                    for key, value in step.items()
                    if key not in ("phase", "returncode", "timed_out", "revoked", "truncated")
                },
                phase=step["phase"],
            )
        return records

    @contextmanager
    def _parent(self, relative, *, write=True):
        """Anchor POSIX writes to directory handles;never follow model links."""
        self._relative(relative)
        self.workspace.path(relative, write=write)
        if os.name != "posix":
            raise ValueError("candidate apply requires the admitted POSIX sandbox")
        flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
        current = os.open(self.workspace.root, flags)
        try:
            for part in PurePosixPath(relative).parts[:-1]:
                child = os.open(part, flags, dir_fd=current)
                os.close(current)
                current = child
            yield current, PurePosixPath(relative).name
        finally:
            os.close(current)

    def _at(self, parent, name, guard=True):
        if guard:
            self._guard()
        try:
            fd = os.open(
                name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | os.O_NONBLOCK, dir_fd=parent
            )
        except FileNotFoundError:
            return None, None
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > self.limits.file_bytes:
                raise ValueError("regular bounded live source required")
            raw = stream.read(self.limits.file_bytes + 1)
            after = os.fstat(stream.fileno())
        identity = lambda info: (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
        if (
            len(raw) > self.limits.file_bytes
            or identity(before) != identity(after)
            or identity(after) != identity(os.stat(name, dir_fd=parent, follow_symlinks=False))
        ):
            raise ValueError("live source changed while reading")
        return raw, stat.S_IMODE(after.st_mode)

    def _replace(self, relative, raw, mode, expected, *, guard=True, expected_mode=None):
        with self._parent(relative) as (parent, name):
            current, current_mode = self._at(parent, name, guard)
            if (
                digest(current) if current is not None else None
            ) != expected or current_mode != expected_mode:
                raise ValueError("live source conflict;candidate was not replayed")
            temporary = ".ic-trial-" + secrets.token_hex(16)
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=parent)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fchmod(stream.fileno(), mode)
                    os.fsync(stream.fileno())
                if guard:
                    self._guard()
                current, current_mode = self._at(parent, name, guard)
                if (
                    digest(current) if current is not None else None
                ) != expected or current_mode != expected_mode:
                    raise ValueError("live source changed during candidate apply")
                os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
            finally:
                try:
                    os.unlink(temporary, dir_fd=parent)
                except FileNotFoundError:
                    pass

    def apply(self, identity):
        self._guard()
        folder, result = self._load(identity)
        if (
            not result.get("apply_available")
            or result.get("applied")
            or (folder / "apply.json").exists()
        ):
            raise ValueError(
                "trial is incomplete,already applied,or has unsupported source changes"
            )
        baseline = json.loads((folder / "baseline.json").read_bytes())
        candidate = json.loads((folder / "candidate.json").read_bytes())
        if (
            digest(encoded(baseline)) != result["baseline_sha256"]
            or digest(encoded(candidate)) != result["candidate_sha256"]
        ):
            raise ValueError("saved trial source identity changed")
        if self._scan(self.workspace.root) != baseline:
            raise ValueError("trial is stale;live source versions changed")
        changes = result["changes"]
        written = []
        created_directories = []
        # Validate every path/payload before any live write,including caller scope.
        for change in changes:
            self.workspace.path(change["path"], write=True)
            if change["after"]["kind"] == "file":
                raw, _ = self._read(folder / "payloads" / change["path"])
                if digest(raw) != change["after"]["sha256"]:
                    raise ValueError("saved candidate payload changed")
        rollback = folder / "rollback"
        rollback.mkdir(exist_ok=False)
        audit = dict(
            trial_id=identity,
            status="applying",
            written=[],
            rolled_back=[],
            conflicts=[],
            in_progress_path=None,
        )
        self._write(folder / "apply.json", audit)
        try:
            for change in sorted(changes, key=lambda item: (item["path"].count("/"), item["path"])):
                self._guard()
                name = change["path"]
                path = self.workspace.path(name, write=True)
                before, after = change["before"], change["after"]
                audit["in_progress_path"] = name
                self._write(folder / "apply.json", audit)
                if after["kind"] == "directory":
                    with self._parent(name) as (parent, leaf):
                        os.mkdir(leaf, mode=after["mode"], dir_fd=parent)
                        child = os.open(
                            leaf, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
                        )
                        try:
                            os.fchmod(child, after["mode"])
                        finally:
                            os.close(child)
                    created_directories.append(name)
                    audit["in_progress_path"] = None
                    self._write(folder / "apply.json", audit)
                    continue
                raw, _ = self._read(folder / "payloads" / name)
                if before:
                    original, _ = self._read(path)
                    backup = rollback / name
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    backup.write_bytes(original)
                self._replace(
                    name,
                    raw,
                    after["mode"],
                    before["sha256"] if before else None,
                    expected_mode=before["mode"] if before else None,
                )
                written.append(change)
                audit["written"].append(name)
                audit["in_progress_path"] = None
                self._write(folder / "apply.json", audit)
            if self._scan(self.workspace.root) != candidate:
                raise ValueError("source changed during multi-file apply")
            result["applied"] = True
            self._write(folder / "result.json", result)
            audit.update(
                status="applied",
                candidate_revision=digest(encoded(candidate)),
                independent_acceptance=False,
                requires_original_verification=True,
            )
        except BaseException as error:
            audit.update(status="apply_failed", error=type(error).__name__ + ": " + str(error))
            # Best-effort rollback only of still-owned bytes;never overwrite an
            # external writer and never replay the model edit script.
            for change in reversed(written):
                name = change["path"]
                try:
                    with self._parent(name) as (parent, leaf):
                        current, _ = self._at(parent, leaf, guard=False)
                    if current is None or digest(current) != change["after"]["sha256"]:
                        raise ValueError("external writer conflict")
                    if change["before"]:
                        saved = (rollback / name).read_bytes()
                        if digest(saved) != change["before"]["sha256"]:
                            raise ValueError("rollback payload changed")
                        self._replace(
                            name,
                            saved,
                            change["before"]["mode"],
                            change["after"]["sha256"],
                            guard=False,
                            expected_mode=change["after"]["mode"],
                        )
                    else:
                        with self._parent(name) as (parent, leaf):
                            current, _ = self._at(parent, leaf, guard=False)
                            if current is None or digest(current) != change["after"]["sha256"]:
                                raise ValueError("external writer conflict")
                            os.unlink(leaf, dir_fd=parent)
                    audit["rolled_back"].append(name)
                except (OSError, ValueError) as rollback_error:
                    audit["conflicts"].append(dict(path=name, error=str(rollback_error)))
            for name in reversed(created_directories):
                try:
                    with self._parent(name) as (parent, leaf):
                        os.rmdir(leaf, dir_fd=parent)
                except (OSError, ValueError):
                    pass
            audit["retained_changed_paths"] = [
                item["path"] for item in written if item["path"] not in audit["rolled_back"]
            ]
            audit["unconfirmed_write_path"] = audit["in_progress_path"]
        finally:
            result["apply_available"] = False
            result["apply_outcome_ref"] = str(folder / "apply.json")
            self._write(folder / "result.json", result)
            self._write(folder / "apply.json", audit)
            self.record("trial_apply", audit)
        return audit
