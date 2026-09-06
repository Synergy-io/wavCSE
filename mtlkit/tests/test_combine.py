"""Unit tests for mtlkit/combine.py."""

import unittest

import torch

from mtlkit.combine import COMBINE_REGISTRY, UniformAverageCombine, build_combine_strategy


class CombineRegistryTests(unittest.TestCase):
    def test_uniform_average_registered(self):
        self.assertIn("uniform_average", COMBINE_REGISTRY)

    def test_unknown_strategy_raises_with_valid_options(self):
        with self.assertRaises(KeyError) as ctx:
            COMBINE_REGISTRY.get("bogus")
        self.assertIn("uniform_average", str(ctx.exception))

    def test_build_combine_strategy_returns_instance(self):
        strategy = build_combine_strategy("uniform_average")
        self.assertIsInstance(strategy, UniformAverageCombine)


class UniformAverageCombineTests(unittest.TestCase):
    def setUp(self):
        self.strategy = UniformAverageCombine()

    def test_averages_across_all_valid_tasks(self):
        losses = [torch.tensor(4.0), torch.tensor(8.0)]
        masks = [torch.tensor([True]), torch.tensor([True])]
        out = self.strategy(losses, masks, head_params=[])
        # 1/2 * 4 + 1/2 * 8 = 6
        self.assertAlmostEqual(float(out), 6.0)

    def test_skips_none_losses_but_keeps_the_original_denominator(self):
        # 4 task slots, only 2 have a valid loss this batch — the weight is
        # still 1/4 per slot (matches downstream/trainer_model.py:279's
        # loss_weight = 1/num_tasks, computed from num_tasks, not
        # "num tasks with a valid sample this batch").
        losses = [torch.tensor(4.0), None, torch.tensor(8.0), None]
        masks = [torch.tensor([True]), torch.tensor([False]), torch.tensor([True]), torch.tensor([False])]
        out = self.strategy(losses, masks, head_params=[])
        self.assertAlmostEqual(float(out), 1.0 + 2.0)  # 4/4 + 8/4

    def test_all_none_returns_zero(self):
        out = self.strategy([None, None], [torch.tensor([False])] * 2, head_params=[])
        self.assertAlmostEqual(float(out), 0.0)

    def test_epoch_hooks_are_no_ops(self):
        # Must not raise — base class default no-op behavior.
        self.strategy.on_epoch_begin(1)
        self.strategy.on_epoch_end(1)

    def test_output_is_a_scalar_tensor(self):
        out = self.strategy([torch.tensor(1.0)], [torch.tensor([True])], head_params=[])
        self.assertEqual(out.dim(), 0)

    def test_gradient_flows_through_combined_loss(self):
        w = torch.tensor(2.0, requires_grad=True)
        loss_t = w * 3.0
        out = self.strategy([loss_t], [torch.tensor([True])], head_params=[])
        out.backward()
        self.assertIsNotNone(w.grad)
        self.assertAlmostEqual(float(w.grad), 3.0)  # d(1/1 * w*3)/dw = 3


if __name__ == "__main__":
    unittest.main()
