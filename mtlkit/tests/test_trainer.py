"""Unit tests for mtlkit/trainer.py.

Includes the 4-task parity gate: `process_batch` + `UniformAverageCombine`
reproduces `downstream/trainer/trainer_model.py`'s base
`_process_batch`'s per-task-loss-combination arithmetic exactly (the L1/L2
regularization and optimizer step are NOT part of this seam's claim — see
mtlkit/trainer.py's module docstring).
"""

import unittest

import torch
import torch.nn as nn

import mtlkit.trainer as mtlkit_trainer
from mtlkit.combine import UniformAverageCombine
from mtlkit.heads import MultiTaskModel, build_heads, build_trunk
from mtlkit.tasks import TASK_REGISTRY


def _build_model(task_keys, pooling_type="mean", pooling_param=None):
    tasks = [TASK_REGISTRY.get(k) for k in task_keys]
    torch.manual_seed(3)
    trunk = build_trunk(
        upstream_model_type="wavlm_large",
        embedding_dim_shared1=16,
        embedding_dim_shared2=8,
        layer_pooling_type=pooling_type,
        dropout_prob_shared1=0.0,
        dropout_prob_shared2=0.0,
        layer_pooling_param=pooling_param,
    )
    heads = build_heads(tasks, in_features=8)
    return MultiTaskModel(trunk, heads)


class ProcessBatchTests(unittest.TestCase):
    def test_head_params_returns_live_parameters(self):
        model = _build_model(["ks", "si"])
        params = mtlkit_trainer.head_params(model)
        self.assertEqual(len(params), 2)
        for weight, bias in params:
            self.assertTrue(weight.requires_grad)
            self.assertTrue(bias.requires_grad)

    def test_process_batch_produces_backprop_ready_loss(self):
        model = _build_model(["ks", "si", "er"])
        torch.manual_seed(11)
        input_seq = torch.randn(4, 25, 1024)
        labels_list = [
            torch.randint(0, 12, (4,)),
            torch.full((4,), -1),  # no valid SI labels this batch
            torch.randint(0, 4, (4,)),
        ]
        result = mtlkit_trainer.process_batch(
            model, input_seq, labels_list, nn.CrossEntropyLoss(), UniformAverageCombine()
        )
        self.assertEqual(result.loss_all.dim(), 0)
        result.loss_all.backward()
        self.assertIsNotNone(model.heads[0].weight.grad)
        # SI head had no valid samples this batch, so its loss term was
        # skipped entirely -- it never entered loss_all's graph, so its
        # grad stays None (not zero) after backward.
        self.assertIsNone(model.heads[1].weight.grad)

    def test_valid_count_and_loss_task_bookkeeping(self):
        model = _build_model(["ks", "si"])
        torch.manual_seed(5)
        input_seq = torch.randn(3, 25, 1024)
        labels_list = [torch.tensor([0, 1, -1]), torch.tensor([-1, -1, -1])]
        result = mtlkit_trainer.process_batch(
            model, input_seq, labels_list, nn.CrossEntropyLoss(), UniformAverageCombine()
        )
        self.assertEqual(result.valid_count_task[0], 2)
        self.assertEqual(result.valid_count_task[1], 0)
        self.assertEqual(result.loss_task[1], 0.0)


class FourTaskParityGateTests(unittest.TestCase):
    """Reproduces trainer_model.py's base `_process_batch` per-task-loss
    arithmetic (excluding L1/L2 and the optimizer step, which are generic
    trainer-loop concerns outside combine()'s claim) on a synthetic 4-task
    batch."""

    def _reference_loss_all(self, logits_tuple, labels_list, loss_fn, num_tasks, ignore_index=-1):
        from trainer.trainer_utils import masked_ce_loss as reference_masked_ce_loss

        loss_weight = 1.0 / float(num_tasks)
        loss_all = torch.tensor(0.0)
        for t in range(num_tasks):
            loss_t, _ = reference_masked_ce_loss(
                logits_tuple[t], labels_list[t], loss_fn, ignore_index
            )
            if loss_t is not None:
                loss_all = loss_all + loss_t * loss_weight
        return loss_all

    def test_4task_combine_matches_reference_arithmetic(self):
        import os
        import sys

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        downstream_dir = os.path.join(repo_root, "downstream")
        for path in (downstream_dir, repo_root):
            if path not in sys.path:
                sys.path.insert(0, path)

        model = _build_model(["ks", "si", "er", "ic"])
        torch.manual_seed(99)
        input_seq = torch.randn(5, 25, 1024)
        labels_list = [
            torch.randint(0, 12, (5,)),
            torch.tensor([-1, -1, 5, -1, -1]),
            torch.randint(0, 4, (5,)),
            torch.full((5,), -1),
        ]
        loss_fn = nn.CrossEntropyLoss()

        outputs = model(input_seq)
        result = mtlkit_trainer.process_batch(
            model, input_seq, labels_list, loss_fn, UniformAverageCombine()
        )
        reference_loss = self._reference_loss_all(
            outputs.logits, labels_list, loss_fn, num_tasks=4
        )
        self.assertTrue(torch.allclose(result.loss_all, reference_loss, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
