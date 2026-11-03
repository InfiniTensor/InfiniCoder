"""Reject unsafe mount declarations before invoking Docker."""

from pathlib import Path
import tempfile
import unittest
from infinicoder import DockerTaskHistoryMount, UnsupportedCore


class DockerTaskHistoryDeclarations(unittest.TestCase):
    def test_export_only_is_separate_fresh_and_readonly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            workspace = root / "workspace"
            workspace.mkdir()
            mount = DockerTaskHistoryMount(root / "state", workspace)
            self.assertEqual(list(mount.directory.iterdir()), [])
            self.assertTrue(mount.mount_arguments[1].endswith("dst=/tmp/ic-task-history,readonly"))
            with self.assertRaises(FileExistsError):
                DockerTaskHistoryMount(root / "state", workspace)
            with self.assertRaises(UnsupportedCore):
                DockerTaskHistoryMount(workspace / "state", workspace)
            with self.assertRaises(UnsupportedCore):
                DockerTaskHistoryMount(root / "state,readonly=false", workspace)


if __name__ == "__main__":
    unittest.main()
