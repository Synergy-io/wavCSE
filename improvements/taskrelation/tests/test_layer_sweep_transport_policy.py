"""A public layer sweep cannot launch an unaccounted paid worker."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from improvements.sweep import remote


class RemotePolicyTests(unittest.TestCase):
    def test_missing_explicit_infra_binding_refuses(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(remote.RemoteError, "WAVCSE_INFRA_CHECKOUT"):
                remote.exec("university", ["true"])

    def test_paid_worker_never_receives_a_sweep_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src/wavcse_infra").mkdir(parents=True)
            (root / "src/wavcse_infra/__init__.py").write_text("", encoding="utf-8")
            (root / "pyproject.toml").write_text(
                '[project]\nname = "wavcse-infra"\n', encoding="utf-8"
            )
            binary = root / ".venv/bin/infra"
            binary.parent.mkdir(parents=True)
            marker = root / "executed"
            binary.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "if sys.argv[1:3] == ['worker', 'show']:\n"
                "    print(json.dumps({'provider': 'runpod', 'id': 'paid-pod'}))\n"
                "else:\n"
                f"    pathlib.Path({str(marker)!r}).write_text('mutated')\n",
                encoding="utf-8",
            )
            binary.chmod(0o755)
            with mock.patch.dict(os.environ, {"WAVCSE_INFRA_CHECKOUT": str(root)}):
                with self.assertRaisesRegex(remote.RemoteError, "static_ssh"):
                    remote.start_detached(
                        "paid-pod", "/workspace/model", ["python", "-m", "trainer"],
                        "/workspace/logs/sweep.log",
                    )
            self.assertFalse(marker.exists())
