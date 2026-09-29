"""A failed arm must not exit successfully.

The control plane treats exit 0 as a successful execution and persists declared
outputs on that basis, so an entry point that prints a traceback and returns 0
reports a crashed run as a result. A *scientifically negative* run is a different
thing: it completed and wrote its outputs, so it exits 0 and is interpreted later
as evidence.
"""

import os
import subprocess
import sys
import unittest
from unittest import mock

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from improvements import run_improvements  # noqa: E402

REAL_CONFIG = os.path.join(
    "improvements", "taskrelation", "01-mtrl", "mtrl_config.yml"
)


class ExitCodeTests(unittest.TestCase):
    def test_a_successful_run_exits_zero(self):
        argv = ["--model", "mtrl", "--task_type", "ks_si_er", "--config", REAL_CONFIG]
        with mock.patch.object(run_improvements, "run_single_model",
                               return_value=object()) as runner:
            self.assertEqual(run_improvements.main(argv), 0)
        self.assertEqual(runner.call_count, 1)

    def test_a_failing_arm_exits_nonzero(self):
        argv = ["--model", "mtrl", "--task_type", "ks_si_er", "--config", REAL_CONFIG]
        with mock.patch.object(run_improvements, "run_single_model",
                               side_effect=RuntimeError("boom")):
            self.assertNotEqual(run_improvements.main(argv), 0)

    def test_a_missing_config_exits_nonzero(self):
        argv = ["--model", "mtrl", "--task_type", "ks_si_er",
                "--config", os.path.join("does", "not", "exist.yml")]
        with mock.patch.object(run_improvements, "run_single_model",
                               side_effect=AssertionError("must not run")):
            self.assertNotEqual(run_improvements.main(argv), 0)

    def test_one_failing_arm_fails_the_whole_invocation(self):
        argv = ["--model", "all"]
        calls = []

        def flaky(model_type, *args, **kwargs):
            calls.append(model_type)
            if model_type == "tsm":
                raise RuntimeError("only this one fails")
            return object()

        with mock.patch.object(run_improvements, "run_single_model", side_effect=flaky):
            self.assertNotEqual(run_improvements.main(argv), 0)
        # The remaining arms still ran: one failure does not abort the others.
        self.assertIn("mtrl", calls)

    def test_the_module_exits_nonzero_in_a_real_process(self):
        """The end-to-end property: a failed invocation is not exit 0."""

        completed = subprocess.run(
            [sys.executable, "-m", "improvements.run_improvements",
             "--model", "mtrl", "--task_type", "ks_si_er",
             "--config", os.path.join("does", "not", "exist.yml")],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            universal_newlines=True, timeout=600,
        )
        self.assertNotEqual(completed.returncode, 0, completed.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()
