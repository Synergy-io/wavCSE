"""Unit tests for mtlkit/heads.py.

Includes the 4-task parity check from the seam table: masked_ce_loss/
masked_accuracy and the trunk+heads forward pass match today's
`downstream/trainer/trainer_utils.py` and `downstream/model/downstream_model.py`
numerically, given the same seeded weights and the same input.
"""

import os
import sys
import unittest

import torch
import torch.nn as nn

import mtlkit.heads as mtlkit_heads
from mtlkit.tasks import TASK_REGISTRY

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DOWNSTREAM_DIR = os.path.join(REPO_ROOT, "downstream")
for path in (DOWNSTREAM_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from model.downstream_model import DownstreamMultiTaskModel  # noqa: E402
from trainer.trainer_utils import masked_ce_loss as downstream_masked_ce_loss  # noqa: E402
from trainer.trainer_utils import masked_accuracy as downstream_masked_accuracy  # noqa: E402


class MaskedLossParityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.randn(6, 4)
        self.labels = torch.tensor([0, -1, 2, -1, 1, 3])
        self.loss_fn = nn.CrossEntropyLoss()

    def test_masked_ce_loss_matches_downstream(self):
        mtlkit_loss, mtlkit_n = mtlkit_heads.masked_ce_loss(
            self.logits, self.labels, self.loss_fn
        )
        downstream_loss, downstream_n = downstream_masked_ce_loss(
            self.logits, self.labels, self.loss_fn
        )
        self.assertEqual(mtlkit_n, downstream_n)
        self.assertTrue(torch.allclose(mtlkit_loss, downstream_loss))

    def test_masked_ce_loss_all_invalid_returns_none_zero(self):
        labels = torch.full((4,), -1)
        loss, n = mtlkit_heads.masked_ce_loss(torch.randn(4, 3), labels, self.loss_fn)
        self.assertIsNone(loss)
        self.assertEqual(n, 0)

    def test_masked_accuracy_matches_downstream(self):
        pred = torch.tensor([0, 1, 2, 3, 1, 2])
        mtlkit_correct, mtlkit_total = mtlkit_heads.masked_accuracy(pred, self.labels)
        downstream_correct, downstream_total = downstream_masked_accuracy(pred, self.labels)
        self.assertEqual((mtlkit_correct, mtlkit_total), (downstream_correct, downstream_total))

    def test_masked_accuracy_all_invalid_returns_zero_zero(self):
        labels = torch.full((4,), -1)
        pred = torch.zeros(4, dtype=torch.long)
        correct, total = mtlkit_heads.masked_accuracy(pred, labels)
        self.assertEqual((correct, total), (0, 0))


class InputDimFromUpstreamTests(unittest.TestCase):
    def test_large_variant(self):
        self.assertEqual(mtlkit_heads.input_dim_from_upstream("wavlm_large"), 1024)

    def test_base_variant(self):
        self.assertEqual(mtlkit_heads.input_dim_from_upstream("wavlm_base"), 768)

    def test_unknown_variant_raises(self):
        with self.assertRaises(ValueError):
            mtlkit_heads.input_dim_from_upstream("wavlm_huge")


class BuildHeadsTests(unittest.TestCase):
    def test_build_heads_output_dims_match_task_num_classes(self):
        tasks = [TASK_REGISTRY.get("ks"), TASK_REGISTRY.get("si"), TASK_REGISTRY.get("er")]
        heads = mtlkit_heads.build_heads(tasks, in_features=16)
        self.assertEqual(len(heads), 3)
        self.assertEqual(heads[0].out_features, 12)   # ks
        self.assertEqual(heads[1].out_features, 1251)  # si
        self.assertEqual(heads[2].out_features, 4)     # er


class ForwardParityWithDownstreamModelTests(unittest.TestCase):
    """Same seeded weights, same input -> identical logits/predictions as
    today's DownstreamMultiTaskModel, for every pooling type that needs no
    extra param (the parity-relevant default in build_model.yml is mean)."""

    def _build_mtlkit_model(self, task_type, pooling_type="mean", pooling_param=None):
        tasks = [TASK_REGISTRY.get(t) for t in task_type.split("_")]
        torch.manual_seed(7)
        trunk = mtlkit_heads.build_trunk(
            upstream_model_type="wavlm_large",
            embedding_dim_shared1=32,
            embedding_dim_shared2=16,
            layer_pooling_type=pooling_type,
            dropout_prob_shared1=0.0,
            dropout_prob_shared2=0.0,
            layer_pooling_param=pooling_param,
        )
        heads = mtlkit_heads.build_heads(tasks, in_features=16)
        return mtlkit_heads.MultiTaskModel(trunk, heads)

    def _build_downstream_model(self, task_type, pooling_type="mean", pooling_param=None):
        torch.manual_seed(7)
        return DownstreamMultiTaskModel(
            upstream_model_type="wavlm_large",
            task_type=task_type,
            embedding_dim_shared1=32,
            embedding_dim_shared2=16,
            layer_pooling_type=pooling_type,
            dropout_prob_shared1=0.0,
            dropout_prob_shared2=0.0,
            layer_pooling_param=pooling_param,
        )

    def test_forward_logits_match_for_3task_mean_pooling(self):
        mtlkit_model = self._build_mtlkit_model("ks_si_er")
        downstream_model = self._build_downstream_model("ks_si_er")

        torch.manual_seed(123)
        input_seq = torch.randn(2, 25, 1024)

        mtlkit_out = mtlkit_model(input_seq)
        downstream_out = downstream_model(input_seq)

        self.assertEqual(len(mtlkit_out.logits), len(downstream_out.logits))
        for i, (a, b) in enumerate(zip(mtlkit_out.logits, downstream_out.logits)):
            self.assertTrue(torch.allclose(a, b, atol=1e-6), msg=f"logits mismatch at head {i}")
        for i, (a, b) in enumerate(zip(mtlkit_out.prediction, downstream_out.prediction)):
            self.assertTrue(torch.equal(a, b), msg=f"prediction mismatch at head {i}")

    def test_forward_logits_match_for_4task_weighted_pooling(self):
        mtlkit_model = self._build_mtlkit_model("ks_si_er_ic", pooling_type="weighted", pooling_param=25)
        downstream_model = self._build_downstream_model(
            "ks_si_er_ic", pooling_type="weighted", pooling_param=25
        )

        torch.manual_seed(456)
        input_seq = torch.randn(3, 25, 1024)

        mtlkit_out = mtlkit_model(input_seq)
        downstream_out = downstream_model(input_seq)

        for i, (a, b) in enumerate(zip(mtlkit_out.logits, downstream_out.logits)):
            self.assertTrue(torch.allclose(a, b, atol=1e-6), msg=f"logits mismatch at head {i}")

    def test_get_pooling_weights_uses_correct_attribute_name(self):
        # Regression guard for the downstream_model.py bug documented in
        # mtlkit/heads.py's module docstring (`pooling_weights` vs
        # `position_weights`) — mtlkit's port must not reproduce it.
        model = self._build_mtlkit_model("ks_si", pooling_type="weighted", pooling_param=25)
        weights = model.get_pooling_weights()
        self.assertIsNotNone(weights)
        self.assertEqual(weights.shape, (25,))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=5)

    def test_get_pooling_weights_none_for_non_weighted(self):
        model = self._build_mtlkit_model("ks_si", pooling_type="mean")
        self.assertIsNone(model.get_pooling_weights())


if __name__ == "__main__":
    unittest.main()
