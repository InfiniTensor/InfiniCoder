"""Incremental relation scope and verified refresh of invalidated callers."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from infinicoder.analysis.code_context import CodeIndex


class IncrementalRelations(unittest.TestCase):
    def read(self, workspace):
        def source(path):
            raw = (workspace / path).read_bytes()
            return raw, raw.decode("utf-8-sig"), hashlib.sha256(raw).hexdigest()

        return source

    def test_native_bindings_are_resolved_without_python_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            workspace = root / "workspace"
            workspace.mkdir()
            with CodeIndex(
                workspace, root / "index.sqlite", native_index_policy="on_demand"
            ) as index:
                build = dict(
                    native_bindings=dict(
                        symbols=[
                            dict(symbol="unit.cpp::caller", usr="caller-USR"),
                            dict(symbol="unit.cpp::callee", usr="callee-USR"),
                        ],
                        refs=[
                            dict(
                                src="unit.cpp::caller",
                                target="callee",
                                line=3,
                                dispatch="direct",
                                usr="callee-USR",
                            )
                        ],
                    )
                )
                index.db.execute(
                    "INSERT INTO files VALUES(?,?,?,?,?,?,?)",
                    (
                        "unit.cpp",
                        "owned-hash",
                        "cpp",
                        "int callee(); int caller(){return callee();}",
                        "",
                        "{}",
                        json.dumps(build),
                    ),
                )
                for name, line in (("callee", 1), ("caller", 3)):
                    index.db.execute(
                        "INSERT INTO symbols VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            "unit.cpp::" + name,
                            "unit.cpp",
                            name,
                            name,
                            "function",
                            line,
                            line,
                            "int " + name + "()",
                            "",
                            "[]",
                            "clang",
                        ),
                    )
                index.db.execute(
                    "INSERT INTO refs VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        "unit.cpp",
                        "unit.cpp::caller",
                        "callee",
                        "calls",
                        3,
                        "",
                        None,
                        "unresolved",
                        0,
                    ),
                )
                index._resolve()
                row = index.db.execute("SELECT dst,confidence FROM refs").fetchone()
                self.assertEqual(tuple(row), ("unit.cpp::callee", "clang_static_candidate"))

    def test_selected_refresh_does_not_rebind_unobserved_callers(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "caller.py").write_bytes(
                b"from callee import target\ndef caller():\n    return target()\n"
            )
            (workspace / "callee.py").write_bytes(b"def target():\n    return 1\n")
            with CodeIndex(
                workspace, root / "index.sqlite", native_index_policy="on_demand"
            ) as index:
                index.refresh_paths(("caller.py", "callee.py"), read_source=self.read(workspace))
                index.db.execute("UPDATE refs SET dst=NULL,confidence='owned-unobserved'")
                index._resolve(paths=("callee.py",))
                row = index.db.execute(
                    "SELECT dst,confidence FROM refs WHERE path='caller.py'"
                ).fetchone()
                self.assertEqual(tuple(row), (None, "owned-unobserved"))

    def test_reobserving_unchanged_caller_restores_its_changed_target_relation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "caller.py").write_bytes(
                b"from callee import target\ndef caller():\n    return target()\n"
            )
            (workspace / "callee.py").write_bytes(b"def target():\n    return 1\n")
            with CodeIndex(
                workspace, root / "index.sqlite", native_index_policy="on_demand"
            ) as index:
                read = self.read(workspace)
                index.refresh_paths(("caller.py", "callee.py"), read_source=read)
                (workspace / "callee.py").write_bytes(b"def target():\n    return 2\n")
                index.refresh_paths(("callee.py",), read_source=read)
                row = index.db.execute(
                    "SELECT dst,confidence FROM refs WHERE path='caller.py'"
                ).fetchone()
                self.assertEqual(tuple(row), (None, "target_changed;refresh_required"))
                result = index.refresh_paths(("caller.py",), read_source=read)
                self.assertEqual(result["changed"], [])
                row = index.db.execute(
                    "SELECT dst,confidence FROM refs WHERE path='caller.py'"
                ).fetchone()
                self.assertEqual(tuple(row), ("callee.py::target", "static_candidate"))


if __name__ == "__main__":
    unittest.main()
