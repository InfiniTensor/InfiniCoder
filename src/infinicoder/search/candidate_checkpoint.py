"""Complete, bounded checkpoints for an explicitly exclusive task workspace.

Unlike an experiment probe, this restores created/deleted helpers and caches as
well as source. It never executes a saved command or traverses external links.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import stat
import tempfile

from infinicoder.search.candidate_trial import TrialLimits


class CheckpointConflict(ValueError):
    """The live or saved tree no longer has its declared version."""


def revision(manifest):
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class CandidateCheckpoint:
    name: str
    revision: str
    manifest_ref: str
    tree_ref: str


class CandidateCheckpointStore:
    """Caller-owned source state, not a checkpoint of services or dependencies.

    All entries are included: there is no guessed build/cache/git exclusion.
    Existing trial file/byte bounds are retained; unavailable trees must use
    the separately declared ordinary path before generation, not a partial copy.
    """

    def __init__(self, workspace, directory, *, exclusive_workspace, remaining):
        if exclusive_workspace is not True or not callable(remaining):
            raise ValueError("explicit exclusive workspace and shared deadline required")
        if os.name != "posix":
            raise ValueError("candidate checkpoint restore requires POSIX task execution")
        self.workspace = Path(workspace).absolute()
        self.directory = Path(directory).absolute()
        for path in (self.workspace, self.directory):
            if any(p.is_symlink() for p in (path, *path.parents)):
                raise ValueError("checkpoint roots cannot have linked components")
        self.workspace = self.workspace.resolve()
        self.directory = self.directory.resolve()
        if (
            not self.workspace.is_dir()
            or self.directory.exists()
            or self.directory.is_relative_to(self.workspace)
            or self.workspace.is_relative_to(self.directory)
        ):
            raise ValueError("existing workspace and fresh separate checkpoint directory required")
        self.remaining = remaining
        self.limits = TrialLimits()
        self.root_device = self.workspace.stat().st_dev
        self.directory.mkdir(mode=0o700)
        self.records = {}

    def _guard(self):
        if self.remaining() <= 0:
            raise TimeoutError("checkpoint shares the original task deadline")

    @staticmethod
    def _identity(info):
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns

    def _read(self, path, *, root, guard=True):
        if guard:
            self._guard()
        relative = path.relative_to(root)
        parent = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for component in relative.parts[:-1]:
                child = os.open(
                    component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
                )
                os.close(parent)
                parent = child
            fd = os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            with os.fdopen(fd, "rb") as stream:
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode) or before.st_size > self.limits.file_bytes:
                    raise ValueError("checkpoint needs bounded regular files")
                data = stream.read(self.limits.file_bytes + 1)
                after = os.fstat(stream.fileno())
            current = os.stat(relative.name, dir_fd=parent, follow_symlinks=False)
        finally:
            os.close(parent)
        if (
            len(data) > self.limits.file_bytes
            or self._identity(before) != self._identity(after)
            or self._identity(after) != self._identity(current)
        ):
            raise CheckpointConflict("file changed while being captured")
        if guard:
            self._guard()
        return data, stat.S_IMODE(after.st_mode)

    def scan(self, root=None, *, guard=True):
        root = Path(root) if root is not None else self.workspace
        entries = {}
        total = 0
        if root.is_symlink() or not root.is_dir():
            raise CheckpointConflict("checkpoint tree root changed")
        device = root.stat().st_dev

        def failed_walk(error):
            raise error

        for folder, directories, files in os.walk(root, followlinks=False, onerror=failed_walk):
            if guard:
                self._guard()
            for name in sorted(directories + files):
                path = Path(folder, name)
                key = path.relative_to(root).as_posix()
                info = path.lstat()
                mode = stat.S_IMODE(info.st_mode)
                if mode & 0o7000 or info.st_dev != device:
                    raise ValueError("special permissions and nested mounts are not checkpointed")
                if stat.S_ISLNK(info.st_mode):
                    if name in directories:
                        directories.remove(name)
                    raw = os.readlink(path)
                    lexical = Path(os.path.abspath(path.parent / raw))
                    # A live absolute internal link remains internal when its
                    # top-level entry is temporarily in the rollback journal.
                    owner = root if lexical.is_relative_to(root) else self.workspace
                    if not lexical.is_relative_to(owner):
                        raise ValueError("checkpoint link leaves the task tree")
                    try:
                        resolved = path.resolve()
                    except RuntimeError as error:
                        raise ValueError("cyclic checkpoint link") from error
                    if not (
                        resolved.is_relative_to(root) or resolved.is_relative_to(self.workspace)
                    ):
                        raise ValueError("checkpoint link resolves outside the task tree")
                    entries[key] = dict(kind="link", target=lexical.relative_to(owner).as_posix())
                elif stat.S_ISDIR(info.st_mode):
                    entries[key] = dict(kind="directory", mode=mode)
                elif stat.S_ISREG(info.st_mode):
                    raw, checked_mode = self._read(path, root=root, guard=guard)
                    total += len(raw)
                    if total > self.limits.bytes:
                        raise ValueError("workspace exceeds complete checkpoint byte bound")
                    entries[key] = dict(
                        kind="file",
                        mode=checked_mode,
                        bytes=len(raw),
                        sha256=hashlib.sha256(raw).hexdigest(),
                    )
                else:
                    raise ValueError("special files are not checkpointed")
                if len(entries) > self.limits.files:
                    raise ValueError("workspace exceeds complete checkpoint entry bound")
        return dict(root_mode=stat.S_IMODE(root.stat().st_mode), entries=entries)

    def _copy(self, source, target, manifest):
        target.mkdir(mode=0o700)
        for key, item in sorted(
            manifest["entries"].items(), key=lambda pair: (pair[0].count("/"), pair[0])
        ):
            self._guard()
            path = target / key
            if item["kind"] == "directory":
                path.mkdir(mode=0o700)
            elif item["kind"] == "link":
                path.symlink_to(os.path.relpath(target / item["target"], path.parent))
            else:
                raw, mode = self._read(source / key, root=source)
                if hashlib.sha256(raw).hexdigest() != item["sha256"] or mode != item["mode"]:
                    raise CheckpointConflict("saved or live checkpoint source changed")
                with path.open("xb") as stream:
                    stream.write(raw)
                path.chmod(mode)
                original = (source / key).stat()
                os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
        for key, item in sorted(manifest["entries"].items(), key=lambda pair: -pair[0].count("/")):
            if item["kind"] == "directory":
                (target / key).chmod(item["mode"])
        target.chmod(manifest["root_mode"])
        if self.scan(source) != manifest or self.scan(target) != manifest:
            raise CheckpointConflict("complete checkpoint copy has a different version")

    def capture(self, name):
        if name not in ("baseline", "A", "B") or name in self.records:
            raise ValueError("capture each fixed candidate checkpoint once")
        manifest = self.scan()
        folder = self.directory / name
        folder.mkdir(mode=0o700)
        tree = folder / "tree"
        self._copy(self.workspace, tree, manifest)
        path = folder / "manifest.json"
        path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf8")
        checkpoint = CandidateCheckpoint(name, revision(manifest), str(path), str(tree))
        self.records[name] = checkpoint
        return checkpoint

    def load(self, checkpoint):
        if self.records.get(checkpoint.name) != checkpoint:
            raise ValueError("checkpoint does not belong to this task store")
        manifest = json.loads(Path(checkpoint.manifest_ref).read_bytes())
        if (
            revision(manifest) != checkpoint.revision
            or self.scan(Path(checkpoint.tree_ref)) != manifest
        ):
            raise CheckpointConflict("saved checkpoint changed")
        return manifest

    def fork(self, checkpoint, target):
        """Create a complete independent branch beneath this exclusive store.

        This is source copying,not an OS sandbox. The collaboration owner must
        still bind branch execution and public checks to its own environment.
        First-version branches reject links instead of sharing writable state.
        """
        self._guard()
        manifest = self.load(checkpoint)
        if any(row["kind"] not in ("file", "directory") for row in manifest["entries"].values()):
            raise ValueError("collaboration forks require link-free source trees")
        target = Path(target).absolute()
        if (
            not target.is_relative_to(self.directory)
            or target == self.directory
            or target.exists()
            or any(p.is_symlink() for p in (target, *target.parents))
        ):
            raise ValueError(
                "fresh link-free target beneath the exclusive checkpoint store required"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        self._copy(Path(checkpoint.tree_ref), target, manifest)
        return dict(
            workspace=str(target),
            revision=checkpoint.revision,
            source_manifest_ref=checkpoint.manifest_ref,
        )

    @staticmethod
    def _part(manifest, name):
        return {
            k: v for k, v in manifest["entries"].items() if k == name or k.startswith(name + "/")
        }

    def restore(self, checkpoint, *, expected_revision):
        """Apply a complete tree transaction;never replay the candidate's edits."""
        desired = self.load(checkpoint)
        current = self.scan()
        if revision(current) != expected_revision:
            raise CheckpointConflict("live workspace changed before selection")
        if current == desired:
            return dict(
                status="already_current", selected=checkpoint.name, revision=checkpoint.revision
            )
        names = sorted(
            {key.split("/")[0] for key in current["entries"]}
            | {key.split("/")[0] for key in desired["entries"]}
        )
        names = [name for name in names if self._part(current, name) != self._part(desired, name)]
        temporary = Path(
            tempfile.mkdtemp(prefix=".ic-candidate-restore-", dir=self.workspace.parent)
        )
        replacement, backup = temporary / "next", temporary / "previous"
        audit = dict(
            status="preparing",
            selected=checkpoint.name,
            expected_revision=expected_revision,
            selected_revision=checkpoint.revision,
            transaction_directory=str(temporary),
            moved_old=[],
            promoted=[],
            rolled_back=[],
            conflicts=[],
        )
        audit_path = self.directory / ("restore-" + secrets.token_hex(8) + ".json")

        def save():
            audit_path.write_text(json.dumps(audit, indent=2), encoding="utf8")

        save()
        try:
            self._copy(Path(checkpoint.tree_ref), replacement, desired)
            replacement.chmod(0o700)
            backup.mkdir(mode=0o700)
            if self.scan() != current:
                raise CheckpointConflict("live workspace changed while selection was prepared")
            self.workspace.chmod(0o700)
            for name in names:
                self._guard()
                source = self.workspace / name
                if name in current["entries"]:
                    os.replace(source, backup / name)
                    audit["moved_old"].append(name)
                if name in desired["entries"]:
                    os.replace(replacement / name, source)
                    audit["promoted"].append(name)
                save()
            self.workspace.chmod(desired["root_mode"])
            if self.scan() != desired:
                raise CheckpointConflict(
                    "selected workspace does not match its complete checkpoint"
                )
            audit["status"] = "selected"
            save()
        except BaseException as error:
            audit.update(status="selection_failed", error=type(error).__name__ + ": " + str(error))
            self.workspace.chmod(0o700)
            discarded = temporary / "unselected"
            discarded.mkdir(mode=0o700)
            for name in reversed(audit["promoted"]):
                try:
                    if self._part(self.scan(guard=False), name) != self._part(desired, name):
                        raise CheckpointConflict("external writer changed selected entry")
                    os.replace(self.workspace / name, discarded / name)
                except (OSError, ValueError) as conflict:
                    audit["conflicts"].append(dict(path=name, error=str(conflict)))
            for name in reversed(audit["moved_old"]):
                try:
                    live = self.workspace / name
                    if live.exists() or live.is_symlink():
                        raise CheckpointConflict("rollback destination has another writer")
                    os.replace(backup / name, live)
                    audit["rolled_back"].append(name)
                except (OSError, ValueError) as conflict:
                    audit["conflicts"].append(dict(path=name, error=str(conflict)))
            self.workspace.chmod(current["root_mode"])
            audit["rollback_complete"] = (
                not audit["conflicts"] and self.scan(guard=False) == current
            )
            save()
            raise
        else:
            # Only the newly created, verified task-private sibling is removed.
            if temporary.resolve().parent != self.workspace.parent or temporary.is_symlink():
                raise CheckpointConflict("restore transaction directory identity changed")
            if shutil.rmtree.avoids_symlink_attacks:
                try:
                    shutil.rmtree(temporary)
                except OSError as error:
                    audit["retained_transaction_directory"] = str(temporary)
                    audit["cleanup_error"] = type(error).__name__ + ": " + str(error)
                    save()
        return audit | dict(audit_ref=str(audit_path))
