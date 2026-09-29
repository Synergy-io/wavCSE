"""The worker wrapper must not publish empty required evidence."""

import os
import unittest

from improvements.compute import worker_stage
from improvements.compute.tests.fakes import ComputeTestCase


class WorkerStageTests(ComputeTestCase):
    def test_empty_required_results_file_is_not_staged(self):
        results = os.path.join(self.home, "results")
        output = os.path.join(self.home, "output")
        os.makedirs(results)
        with open(os.path.join(results, "metrics.txt"), "wb"):
            pass
        plan = {"outputs": [{"name": "metrics", "kind": "results_file",
                             "required": True}]}
        staged, missing = worker_stage.stage_outputs(
            plan, {"arm": "mtrl"}, 0, {"results_dir": results}, output)
        self.assertEqual(staged, [])
        self.assertIn("empty", missing[0])
        self.assertFalse(os.path.exists(os.path.join(output, "metrics.txt")))


if __name__ == "__main__":
    unittest.main()
