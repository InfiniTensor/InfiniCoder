"""Scope contracts and source leases;real OS/SDK execution is a Linux group."""

from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
import unittest

from infinicoder.runtime.agent_execution import (
    CoreBudget,
    CoreRequest,
    UnsupportedCore,
    validate_native_request,
)
from infinicoder.models.budget import ModelProfile
from infinicoder.execution.native_sandbox import NativeSandboxExecution
from infinicoder.execution.native_scoped_execution import with_scoped_native_execution
from infinicoder.planning.task_contract import TaskSpec, Check


class Caller:
    identity = "owned-contract-control"

    def validate(self, request):
        pass

    @contextmanager
    def open(self, *args):
        yield

    @contextmanager
    def public_command(self, *args):
        yield

    def verify(self, request):
        raise AssertionError("these contract controls must never call acceptance")


class NativeScopeControls(unittest.TestCase):
    def test_original_Clang_body_merge_consumer_is_preserved(self):
        from infinicoder.execution.native_scope import restore
        import infinicoder.planning.native_plan  # noqa: F401 -- exercise the existing Clang consumer import

        fragment = "int f(int x) { return x; }"
        prefix = "// original declaration\n"
        scope = dict(
            source=fragment, original=prefix + fragment + "\n", offset=len(prefix), qualname="f"
        )
        result = restore(scope, "int f(int x) { return x + 1; }")
        self.assertEqual(result, prefix + "int f(int x) { return x + 1; }\n")
        with self.assertRaises(ValueError):
            restore(scope, "long f(int x) { return x + 1; }")

    def make(self, root):
        workspace = root / "source"
        (workspace / "package").mkdir(parents=True)
        (workspace / "package/api.py").write_text("def call(): return 0\n")
        (workspace / "caller.py").write_text("from package.api import call\n")
        builder = lambda source, command: [
            "bwrap",
            "--die-with-parent",
            "--tmpfs",
            "/tmp",
            "--dev",
            "/dev",
            "--bind",
            str(source),
            "/workspace",
            "--chdir",
            "/workspace",
            *command,
        ]
        execution = NativeSandboxExecution(builder, root / "scratch")
        spec = TaskSpec(
            "Fix call without changing caller.py.",
            allowed_paths=["package/"],
            required_artifacts=["package/api.py"],
            checks=[Check("public", ["python", "caller.py"], protected_paths=["caller.py"])],
        )
        return CoreRequest(
            spec,
            workspace,
            root / "state",
            ModelProfile(16384, 4096),
            CoreBudget(),
            execution,
            Caller(),
            Caller(),
        )

    def test_complete_spec_preserved_and_known_namespace_transformed(self):
        with tempfile.TemporaryDirectory() as name:
            request = self.make(Path(name))
            scoped = with_scoped_native_execution(request)
            validate_native_request(scoped)
            self.assertIs(scoped.spec, request.spec)
            argv = scoped.execution_adapter._argv(
                request.execution_adapter.sandbox_argv, request.workspace, ["/bin/true"]
            )
            self.assertIn("--unshare-pid", argv)
            self.assertEqual(
                argv[argv.index("--ro-bind") :][:3],
                ["--ro-bind", str(request.workspace), "/workspace"],
            )
            self.assertIn(str(request.workspace / "package"), argv)
            self.assertEqual(scoped.spec.checks[0].protected_paths, ["caller.py"])

    def test_file_and_unconsumed_stage_contracts_fail_before_execution(self):
        with tempfile.TemporaryDirectory() as name:
            request = self.make(Path(name))
            with self.assertRaises(UnsupportedCore):
                validate_native_request(request)
            request.spec.allowed_paths = ["package/api.py"]
            with self.assertRaisesRegex(UnsupportedCore, "exact-file"):
                validate_native_request(with_scoped_native_execution(request))
            request.spec.allowed_paths = ["package/"]
            request.spec.milestones = [dict(goal="retain caller stage", check_ids=["public"])]
            with self.assertRaisesRegex(UnsupportedCore, "stages"):
                validate_native_request(with_scoped_native_execution(request))
            self.assertEqual(request.spec.milestones[0]["goal"], "retain caller stage")

    def test_protected_source_and_contract_changes_revoke_lease(self):
        with tempfile.TemporaryDirectory() as name:
            request = with_scoped_native_execution(self.make(Path(name)))
            request.execution_adapter.validate(request)
            (request.workspace / "caller.py").write_text("changed by a caller\n")
            with self.assertRaisesRegex(UnsupportedCore, "protected workspace"):
                request.execution_adapter.verify_scope(request)
        with tempfile.TemporaryDirectory() as name:
            request = with_scoped_native_execution(self.make(Path(name)))
            request.execution_adapter.validate(request)
            request.spec.required_artifacts.append("package/extra.py")
            with self.assertRaisesRegex(UnsupportedCore, "TaskSpec"):
                request.execution_adapter.verify_scope(request)

    def test_hardlink_alias_cannot_make_protected_inode_writable(self):
        with tempfile.TemporaryDirectory() as name:
            request = self.make(Path(name))
            try:
                os.link(request.workspace / "caller.py", request.workspace / "package/alias.py")
            except OSError:
                self.skipTest("hardlinks unavailable")
            with self.assertRaisesRegex(UnsupportedCore, "hardlink"):
                validate_native_request(with_scoped_native_execution(request))

    def test_unknown_mount_or_overlapping_checker_is_not_silently_admitted(self):
        with tempfile.TemporaryDirectory() as name:
            request = with_scoped_native_execution(self.make(Path(name)))
            request.execution_adapter.validate(request)
            base = request.execution_adapter.execution.sandbox_argv

            def extra(source, command):
                argv = base(source, command)
                argv[-len(command) : -len(command)] = ["--tmpfs", "/workspace/extra"]
                return argv

            with self.assertRaisesRegex(UnsupportedCore, "nested workspace"):
                request.execution_adapter._argv(extra, request.workspace, ["/bin/true"])
            request.spec.checks[0].protected_paths = ["package/api.py"]
            with self.assertRaisesRegex(UnsupportedCore, "overlapping"):
                request.execution_adapter.validate(request)


if __name__ == "__main__":
    unittest.main()
