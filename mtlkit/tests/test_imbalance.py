"""Unit tests for mtlkit/imbalance.py.

Imbalance seam success criteria (Success Criteria + Eng Review decision 4B):
- class-reweighted masked loss reproduces a hand-computed reference value
  on a pinned batch (mechanical gate)
- on that same batch, the under-represented class's head gradient norm with
  reweighting is >= 1.5x its value without reweighting (pinned assertion,
  replacing the original qualitative "gradients demonstrably shift")
"""

import unittest

import torch
import torch.nn as nn

from mtlkit.heads import masked_ce_loss
from mtlkit.imbalance import build_weighted_loss_fn, inverse_frequency_weights


class InverseFrequencyWeightsTests(unittest.TestCase):
    def test_weights_are_inversely_proportional_to_count(self):
        weights = inverse_frequency_weights([100, 10, 1])
        # class 2 (count=1) should have the largest weight, class 0 the smallest
        self.assertLess(weights[0], weights[1])
        self.assertLess(weights[1], weights[2])

    def test_weights_normalized_to_mean_one_over_nonzero_classes(self):
        weights = inverse_frequency_weights([10, 20, 30])
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=5)

    def test_zero_count_class_gets_zero_weight(self):
        weights = inverse_frequency_weights([10, 0, 5])
        self.assertEqual(float(weights[1]), 0.0)

    def test_uniform_counts_give_uniform_weights(self):
        weights = inverse_frequency_weights([50, 50, 50])
        self.assertTrue(torch.allclose(weights, torch.ones(3)))


class MechanicalGateHandComputedTests(unittest.TestCase):
    """Class-reweighted masked loss reproduces a hand-computed reference
    value on a pinned batch."""

    def test_weighted_loss_matches_hand_computed_value(self):
        # 2 classes, counts [1, 4] -> inverse freq [1, 0.25], normalized to
        # mean 1 over both (mean of [1, 0.25] = 0.625) -> [1.6, 0.4]
        class_counts = [1, 4]
        weights = inverse_frequency_weights(class_counts)
        expected_weights = torch.tensor([1.6, 0.4])
        self.assertTrue(torch.allclose(weights, expected_weights, atol=1e-4))

        # Pinned batch: logits chosen so cross-entropy has a clean closed form.
        logits = torch.tensor([[0.0, 0.0], [0.0, 0.0]])  # uniform prediction -> CE = ln(2) per row
        labels = torch.tensor([0, 1])
        loss_fn = build_weighted_loss_fn(class_counts)
        loss, valid = masked_ce_loss(logits, labels, loss_fn)

        ln2 = torch.log(torch.tensor(2.0))
        # nn.CrossEntropyLoss(weight=w, reduction="mean") computes
        # sum(w[y_i] * ce_i) / sum(w[y_i]) for the batch.
        expected = (expected_weights[0] * ln2 + expected_weights[1] * ln2) / (
            expected_weights[0] + expected_weights[1]
        )
        self.assertEqual(valid, 2)
        self.assertTrue(torch.allclose(loss, expected, atol=1e-4))


class PinnedGradientRatioTests(unittest.TestCase):
    """Eng Review decision 4B: on the same pinned batch, the under-
    represented class's head gradient norm with reweighting is >= 1.5x its
    value without reweighting."""

    def _gradient_norm_for_underrepresented_class(self, weighted: bool):
        torch.manual_seed(0)
        head = nn.Linear(8, 3)  # 3 classes

        # Pinned batch: one sample per class (balanced within THIS batch),
        # but class_counts reflects a severely imbalanced dataset overall
        # (class 2 occurs 1000x less often than 0/1) -- isolates the
        # reweighting's effect on class 2's gradient from in-batch sample
        # count, rather than conflating the two.
        torch.manual_seed(1)
        embedding = torch.randn(3, 8)
        labels = torch.tensor([0, 1, 2])
        class_counts = [1000, 1000, 1]

        logits = head(embedding)
        if weighted:
            loss_fn = build_weighted_loss_fn(class_counts)
        else:
            loss_fn = nn.CrossEntropyLoss()

        loss, _ = masked_ce_loss(logits, labels, loss_fn)
        loss.backward()

        # Gradient norm on the row of the weight matrix feeding class 2's
        # logit -- the under-represented class's own head parameters.
        return float(head.weight.grad[2].norm())

    def test_reweighted_gradient_at_least_1_5x_unweighted(self):
        unweighted_norm = self._gradient_norm_for_underrepresented_class(weighted=False)
        weighted_norm = self._gradient_norm_for_underrepresented_class(weighted=True)

        self.assertGreater(unweighted_norm, 0.0)
        self.assertGreaterEqual(
            weighted_norm,
            1.5 * unweighted_norm,
            msg=(
                f"reweighted grad norm {weighted_norm:.6f} is not >= 1.5x "
                f"unweighted {unweighted_norm:.6f}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
