"""Synthetic correctness checks for the RAMUSA model and regularizer."""

import os
import sys
import tempfile
import unittest

import torch
from torch.utils.data import Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
for path in (ROOT, os.path.join(ROOT, "downstream")):
    if path not in sys.path:
        sys.path.insert(0, path)

from improvements.lowrank.models.ramusa_model import DownstreamMultiTaskModelRAMUSA
from improvements.lowrank.trainers.ramusa_trainer import MultiTasksModelTrainerRAMUSA
from improvements.lowrank.utils.ramusa import compute_truncated_trace_penalty, update_effective_rank


class SyntheticDataset(Dataset):
    def __init__(self):
        self.inputs = torch.randn(6, 3, 768)
        self.labels = torch.tensor([[0, -1, -1], [-1, 1, -1], [-1, -1, 2], [3, -1, -1], [-1, 4, -1], [-1, -1, 1]])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]


def model():
    return DownstreamMultiTaskModelRAMUSA(
        upstream_model_type="wavlm_base", task_type="ks_si_er",
        embedding_dim_shared1=8, embedding_dim_shared2=6,
        layer_pooling_type="mean", layer_pooling_param=None,
        dropout_prob_shared1=0.0, dropout_prob_shared2=0.0, adapter_dim=4,
    )


class RAMUSATests(unittest.TestCase):
    def test_shapes_and_task_guard(self):
        network = model()
        self.assertEqual(len(network.adapters), 3)
        self.assertEqual(tuple(network.build_ramusa_parameter_matrix().shape), (24, 3))
        self.assertEqual(network.get_ramusa_singular_values().numel(), 3)
        self.assertNotIn("ic", network.TASK_NAMES)
        with self.assertRaises(ValueError):
            DownstreamMultiTaskModelRAMUSA(
                upstream_model_type="wavlm_base", task_type="ks_si_er_ic",
                embedding_dim_shared1=8, embedding_dim_shared2=6,
                layer_pooling_type="mean", dropout_prob_shared1=0.0,
                dropout_prob_shared2=0.0,
            )

    def test_penalty_and_gradients(self):
        matrix = torch.diag(torch.tensor([4.0, 2.0, 1.0]))
        self.assertAlmostEqual(compute_truncated_trace_penalty(matrix, 0).item(), 7.0)
        self.assertAlmostEqual(compute_truncated_trace_penalty(matrix, 1).item(), 3.0)
        self.assertEqual(compute_truncated_trace_penalty(matrix, 3).item(), 0.0)
        network = model()
        penalty = compute_truncated_trace_penalty(network.build_ramusa_parameter_matrix(), 0)
        penalty.backward()
        self.assertTrue(all(adapter.weight.grad is not None for adapter in network.adapters))

    def test_rank_threshold_and_initial_stage(self):
        matrix = torch.diag(torch.tensor([4.0, 2.0, 1.0]))
        self.assertEqual(update_effective_rank(matrix, 2.0), 2)
        with tempfile.TemporaryDirectory() as directory:
            trainer = MultiTasksModelTrainerRAMUSA(
                model=model(), device=torch.device("cpu"), task_type="ks_si_er",
                training_cfg={"num_epochs": 1, "batch_size": 6, "learning_rate": .001, "weight_decay": 0,
                              "shuffle_train": False, "shuffle_val": False, "pin_memory": False,
                              "drop_last_train": False, "drop_last_val": False, "num_workers": 0,
                              "l1_lambda": 0, "l2_lambda": 0, "saved_checkpoint_count": 1},
                ramusa_cfg={"enabled": True, "lambda": .1, "tau": 2, "num_stages": 1, "epochs_per_stage": 1},
                results_root=os.path.join(directory, "results"), checkpoints_root=os.path.join(directory, "checkpoints"),
                training_data=SyntheticDataset(), validation_data=SyntheticDataset(), ignore_index=-1,
            )
            self.assertEqual(trainer.effective_rank, 0)


if __name__ == "__main__":
    unittest.main()
