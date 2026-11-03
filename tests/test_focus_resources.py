"""Finite native-focus checks against caller-provided real Clang headers."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from infinicoder.analysis.local_edit import invalid, scope_for, select
from infinicoder.planning.task_contract import TaskSpec
from infinicoder.execution.workspace_tools import Workspace

RESOURCE = os.environ.get("IC_OWNED_CLANG_RESOURCE_DIR")
SOURCE = "#include <stddef.h>\nint combine(int a,int b) { size_t n=2; return a-b+int(n)-2; }\n"


@unittest.skipUnless(RESOURCE, "requires the caller-owned existing Clang resource package")
class NativeFocusResources(unittest.TestCase):
    def fixture(self, root, flags=()):
        source = root / "workspace"
        (source / "engine").mkdir(parents=True)
        (source / "engine/math.cc").write_text(SOURCE)
        database = source / "compile_commands.json"
        database.write_text(
            json.dumps(
                [
                    dict(
                        directory=".",
                        file="engine/math.cc",
                        arguments=["g++", "-std=c++17", *flags, "-c", "engine/math.cc"],
                    )
                ]
            )
        )
        workspace = Workspace(source, ["engine/"])
        digest = hashlib.sha256(SOURCE.encode()).hexdigest()
        inspected = [
            dict(path="engine/math.cc", sha256=digest, spans=[[1, 2]], full_file_visible=True)
        ]
        unit = type(
            "Unit", (), dict(kind="implement", worker_path=None, write_set=["engine/"], id="owned")
        )()
        args = dict(
            path="engine/math.cc",
            symbol="combine",
            expected_hash=digest,
            reason="Owned source localization control.",
        )
        spec = TaskSpec("Repair combine without changing its interface.", allowed_paths=["engine/"])
        return workspace, database, lambda: select(spec, args, workspace, inspected, unit, 0)

    def resource(self, root, name):
        owned = root / name
        owned.mkdir()
        (owned / "include").symlink_to(Path(RESOURCE) / "include", target_is_directory=True)
        (owned / "receipt.json").write_text(
            json.dumps(
                dict(
                    libclang_version=importlib.metadata.version("libclang"),
                    llvm_commit="owned-identity",
                )
            )
        )
        return owned

    def test_real_headers_consumed_and_configuration_changes_revoke_focus(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = self.resource(root, "resource-a")
            second = self.resource(root, "resource-b")
            workspace, database, choose = self.fixture(root)
            with patch.dict(os.environ, {"INFINICODER_CLANG_RESOURCE_DIR": str(first)}):
                focus = choose()
                self.assertEqual(
                    focus["resource_configuration"]["effective_origin"], "caller_environment"
                )
                self.assertTrue(focus["resource_configuration"]["version_verified"])
                self.assertFalse(invalid(focus, workspace))
                self.assertEqual(scope_for(focus, workspace)["qualname"], "combine")
                with patch.dict(os.environ, {"INFINICODER_CLANG_RESOURCE_DIR": str(second)}):
                    self.assertIn("native_analysis_configuration", invalid(focus, workspace))
                    with self.assertRaisesRegex(ValueError, "configuration changed"):
                        scope_for(focus, workspace)
                (first / "receipt.json").write_text(
                    json.dumps(
                        dict(
                            libclang_version=importlib.metadata.version("libclang"),
                            llvm_commit="owned-changed",
                        )
                    )
                )
                self.assertIn("native_analysis_configuration", invalid(focus, workspace))
                with self.assertRaisesRegex(ValueError, "configuration changed"):
                    scope_for(focus, workspace)
                fresh = choose()
                value = json.loads(database.read_bytes())
                value[0]["arguments"].insert(1, "-DOWNED_ARGUMENT_CHANGE=1")
                database.write_text(json.dumps(value))
                self.assertIn("native_analysis_configuration", invalid(fresh, workspace))

    def test_compile_database_resource_wins_without_environment_rescue(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            resource = self.resource(root, "explicit-resource")
            workspace, database, choose = self.fixture(root, ["-resource-dir", str(resource)])
            with patch.dict(
                os.environ, {"INFINICODER_CLANG_RESOURCE_DIR": str(root / "missing-env-resource")}
            ):
                focus = choose()
                self.assertEqual(
                    focus["resource_configuration"]["effective_origin"], "compile_database"
                )
                self.assertEqual(focus["resource_configuration"]["directory"], str(resource))
                self.assertFalse(invalid(focus, workspace))
                scope_for(focus, workspace)
            value = json.loads(database.read_bytes())
            value[0]["arguments"][3] = str(root / "missing-database-resource")
            database.write_text(json.dumps(value))
            with patch.dict(os.environ, {"INFINICODER_CLANG_RESOURCE_DIR": str(resource)}):
                with self.assertRaisesRegex(ValueError, "Clang"):
                    choose()


if __name__ == "__main__":
    unittest.main()
