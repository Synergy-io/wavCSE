"""Data-free checks for the DG-0005 training-only sampling control."""

import os
import sys
import tempfile
import unittest

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
DOWNSTREAM_DIR = os.path.join(REPO_ROOT, "downstream")
for path in (DOWNSTREAM_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from improvements.taskrelation.weighted_sampling import (
    TaskWeightedTrainer,
    build_task_weighted_sampler,
)


class PatternDataset(Dataset):
    def __init__(self, size, index_pattern):
        self.size = size
        self.index_pattern = index_pattern

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        raise AssertionError("Sampler construction must not load embeddings")


class CombinedDatasetStub(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, index):
        raise AssertionError("Sampler construction must not load embeddings")


class TaskWeightedSamplerTests(unittest.TestCase):
    def setUp(self):
        self.root = CombinedDatasetStub([
            PatternDataset(100, "10"),
            Subset(PatternDataset(25, "01"), list(range(20))),
        ])
        self.dataset = Subset(self.root, list(range(len(self.root))))

    def test_sampler_is_deterministic_and_preserves_optimizer_exposure(self):
        first = list(build_task_weighted_sampler(
            self.dataset,
            ["ks", "er"],
            {"er": 5.0},
            seed=42,
        ))
        second = list(build_task_weighted_sampler(
            self.dataset,
            ["ks", "er"],
            {"er": 5.0},
            seed=42,
        ))

        self.assertEqual(first, second)
        self.assertEqual(len(first), len(self.dataset))
        er_count = sum(index >= 100 for index in first)
        ks_count = len(first) - er_count
        self.assertGreaterEqual(er_count, 0.5 * ks_count)

    def test_sampler_derives_membership_without_reading_samples(self):
        sampler = build_task_weighted_sampler(
            self.dataset,
            ["ks", "er"],
            {"er": 5.0},
            seed=7,
        )

        self.assertEqual(sampler.num_samples, len(self.dataset))
        self.assertTrue(torch.equal(
            sampler.weights,
            torch.tensor([1.0] * 100 + [5.0] * 20, dtype=torch.double),
        ))

    def test_sampler_rejects_unknown_tasks_and_missing_seed(self):
        with self.assertRaisesRegex(ValueError, "Unknown task_sampling_weights"):
            build_task_weighted_sampler(
                self.dataset,
                ["ks", "er"],
                {"si": 1.0},
                seed=42,
            )
        with self.assertRaisesRegex(ValueError, "run seed"):
            build_task_weighted_sampler(
                self.dataset,
                ["ks", "er"],
                {"er": 5.0},
                seed=None,
            )


    def test_training_loader_uses_weighted_batches(self):
        class SampledCombinedDataset(CombinedDatasetStub):
            def __getitem__(self, index):
                task = 1 if index >= len(self.datasets[0]) else 0
                return torch.tensor([float(index)]), torch.tensor([task, -1])

        dataset = Subset(SampledCombinedDataset(self.root.datasets), list(range(120)))
        with tempfile.TemporaryDirectory() as directory:
            trainer = TaskWeightedTrainer(
                model=nn.Linear(1, 2),
                device=torch.device("cpu"),
                task_type="ks_er",
                training_cfg={
                    "num_epochs": 1,
                    "batch_size": 12,
                    "learning_rate": 0.001,
                    "shuffle_train": True,
                    "drop_last_train": False,
                    "pin_memory": False,
                    "task_sampling_weights": {"er": 5.0},
                },
                results_root=os.path.join(directory, "results"),
                checkpoints_root=os.path.join(directory, "checkpoints"),
                training_data=dataset,
                validation_data=dataset,
                seed=42,
            )
            sampled_indices = [
                int(value)
                for embeddings, _ in trainer.train_dataloader
                for value in embeddings[:, 0]
            ]
            validation_indices = [
                int(value)
                for embeddings, _ in trainer.val_dataloader
                for value in embeddings[:, 0]
            ]

        self.assertEqual(len(sampled_indices), len(dataset))
        er_count = sum(index >= 100 for index in sampled_indices)
        self.assertGreaterEqual(er_count, (len(dataset) - er_count) * 0.5)
        self.assertEqual(validation_indices, list(range(len(dataset))))

if __name__ == "__main__":
    unittest.main()
