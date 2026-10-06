"""The remote start command: detached, and explicit about its environment."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from improvements.sweep import remote  # noqa: E402


class DetachCommandTest(unittest.TestCase):
    def test_detaches_and_redirects(self):
        command = remote.detach_command(
            "/srv/wavCSE", ["uv", "run", "--locked", "python", "-m", "x"],
            "/srv/wavCSE/state/supervisor.log")
        self.assertIn("cd /srv/wavCSE", command)
        self.assertIn("setsid nohup", command)
        self.assertIn("< /dev/null", command)
        self.assertIn("/srv/wavCSE/state/supervisor.log", command)

    def test_env_file_is_sourced_before_the_command(self):
        command = remote.detach_command(
            "/srv/wavCSE", ["python", "-m", "x"], "/tmp/log", "/root/.sweep-env")
        self.assertIn(". /root/.sweep-env", command)
        self.assertLess(command.index(". /root/.sweep-env"),
                        command.index("setsid nohup"))

    def test_no_env_file_means_no_sourcing(self):
        command = remote.detach_command("/srv/wavCSE", ["python"], "/tmp/log")
        self.assertNotIn("set -a", command)

    def test_arguments_with_spaces_are_quoted(self):
        command = remote.detach_command(
            "/srv/my checkout", ["python", "-m", "x", "--note", "a b"],
            "/tmp/a b.log")
        self.assertIn("'/srv/my checkout'", command)
        self.assertIn("'a b'", command)

    def test_no_secret_value_is_ever_in_the_command(self):
        """Only the env-file *path* may appear; values are read on the pod."""
        command = remote.detach_command(
            "/srv/wavCSE", ["python"], "/tmp/log", "/root/.sweep-env")
        for forbidden in ("MLFLOW_TRACKING_PASSWORD", "Bearer", "token="):
            self.assertNotIn(forbidden, command)


if __name__ == "__main__":
    unittest.main()
