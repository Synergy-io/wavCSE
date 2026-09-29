"""Evaluation must bind to this run, never to the newest directory.

`MultiTasksModelEvaluator` falls back to `_latest_run_id(root)` when it is not
given explicit identifiers. Two runs sharing a root can then evaluate each
other's checkpoints, with nothing in the output to show it. The entry points in
`improvements/` therefore pass the identifiers belonging to the trainer they just
built, and this test proves the fallback is not consulted in that path.
"""

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
DOWNSTREAM_DIR = os.path.join(REPO_ROOT, "downstream")
for path in (DOWNSTREAM_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from evaluator.evaluator_model import MultiTasksModelEvaluator  # noqa: E402
from improvements.eval_utils import evaluation_run_ids  # noqa: E402


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 2)


class RunIdentityTests(unittest.TestCase):
    def test_identity_comes_from_the_trainer_not_from_disk(self):
        trainer = SimpleNamespace(
            results_dir="/tmp/run_root/2026_09_29_10_00_00",
            ckpt_dir="/tmp/ckpt_root/2026_09_29_10_00_00",
        )
        self.assertEqual(
            evaluation_run_ids(trainer),
            ("2026_09_29_10_00_00", "2026_09_29_10_00_00"),
        )

    def test_a_trainer_without_run_directories_is_refused(self):
        with self.assertRaises(RuntimeError):
            evaluation_run_ids(SimpleNamespace())

    def test_an_empty_directory_name_is_refused(self):
        with self.assertRaises(RuntimeError):
            evaluation_run_ids(SimpleNamespace(results_dir="/", ckpt_dir="/"))


class CrossLoadTests(unittest.TestCase):
    """Two neighbouring runs in one root: the requested one must win."""

    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="eval-identity-")
        self.results_root = os.path.join(self.home, "results")
        self.checkpoints_root = os.path.join(self.home, "checkpoints")
        self.wanted = "2026_09_29_09_00_00"
        self.newer = "2026_09_29_11_00_00"
        self.model = TinyModel()
        for run_id in (self.wanted, self.newer):
            os.makedirs(os.path.join(self.results_root, run_id), exist_ok=True)
            checkpoint_dir = os.path.join(self.checkpoints_root, run_id)
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(self.model.state_dict(),
                       os.path.join(checkpoint_dir, "ks_si_er_opt.pth"))
        # Make the *wanted* run older on disk, which is what "latest" would skip.
        os.utime(os.path.join(self.checkpoints_root, self.wanted), (10**9, 10**9))

    def test_the_requested_run_is_used_even_when_another_looks_newer(self):
        dataset = [(torch.zeros(4), torch.zeros(3))]
        with mock.patch.object(
            MultiTasksModelEvaluator, "_latest_run_id",
            side_effect=AssertionError("'latest' must never be consulted here"),
        ):
            evaluator = MultiTasksModelEvaluator(
                model=self.model,
                device=torch.device("cpu"),
                task_type="ks_si_er",
                evaluation_cfg={"batch_size": 1},
                results_root=self.results_root,
                dataset=dataset,
                checkpoints_root=self.checkpoints_root,
                checkpoint_tag="opt",
                ignore_index=-1,
                results_run_id=self.wanted,
                checkpoint_run_id=self.wanted,
            )
        self.assertEqual(os.path.basename(evaluator.results_dir), self.wanted)
        self.assertEqual(os.path.basename(evaluator.checkpoints_dir), self.wanted)

    def test_omitting_the_identifiers_would_fall_back_to_latest(self):
        """The hazard this fix removes, demonstrated on the frozen downstream code."""

        dataset = [(torch.zeros(4), torch.zeros(3))]
        evaluator = MultiTasksModelEvaluator(
            model=self.model,
            device=torch.device("cpu"),
            task_type="ks_si_er",
            evaluation_cfg={"batch_size": 1},
            results_root=self.results_root,
            dataset=dataset,
            checkpoints_root=self.checkpoints_root,
            checkpoint_tag="opt",
            ignore_index=-1,
        )
        self.assertEqual(os.path.basename(evaluator.checkpoints_dir), self.newer)
        self.assertNotEqual(os.path.basename(evaluator.checkpoints_dir), self.wanted)


if __name__ == "__main__":
    unittest.main()
