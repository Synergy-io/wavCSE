"""Unit tests for mtlkit/compose.py.

Includes the composition-wiring success criterion: the fixture strategy
stacks with a shared trunk via declared-order composition and trains
(forward + backward) without shape/gradient errors on a synthetic 4-task
setup.
"""

import unittest

import torch
import torch.nn as nn

from mtlkit.compose import (
    COMPOSE_REGISTRY,
    ComposeStrategy,
    FixtureIdentityStrategy,
    FixtureScaleStrategy,
    build_compose_strategy,
    compose,
)
from mtlkit.heads import MultiTaskModel, build_heads, build_trunk
from mtlkit.tasks import TASK_REGISTRY


class ComposeRegistryTests(unittest.TestCase):
    def test_fixture_strategies_registered(self):
        self.assertIn("fixture_scale", COMPOSE_REGISTRY)
        self.assertIn("fixture_identity", COMPOSE_REGISTRY)

    def test_build_compose_strategy_returns_instance(self):
        strategy = build_compose_strategy("fixture_scale", init_scale=0.5)
        self.assertIsInstance(strategy, FixtureScaleStrategy)
        self.assertAlmostEqual(float(strategy.scale), 0.5)

    def test_unknown_strategy_raises_with_valid_options(self):
        with self.assertRaises(KeyError) as ctx:
            COMPOSE_REGISTRY.get("bogus")
        self.assertIn("fixture_scale", str(ctx.exception))


class ComposeSemanticsTests(unittest.TestCase):
    def test_no_strategies_returns_input_unchanged(self):
        x = torch.randn(3, 4)
        self.assertTrue(torch.equal(compose(x, []), x))

    def test_single_additive_strategy(self):
        x = torch.ones(2, 3)
        strategy = FixtureScaleStrategy(init_scale=0.5)
        out = compose(x, [strategy])
        # out = x + 0.5*x = 1.5*x
        self.assertTrue(torch.allclose(out, 1.5 * x))

    def test_two_additive_strategies_sum_independently_of_original(self):
        x = torch.ones(2, 3)
        a = FixtureScaleStrategy(init_scale=0.5)
        b = FixtureScaleStrategy(init_scale=0.25)
        out = compose(x, [a, b])
        # out = x + 0.5x + 0.25x = 1.75x — both see the ORIGINAL x, not each other's output
        self.assertTrue(torch.allclose(out, 1.75 * x))

    def test_additive_strategy_order_is_commutative(self):
        x = torch.randn(2, 3)
        a = FixtureScaleStrategy(init_scale=0.5)
        b = FixtureScaleStrategy(init_scale=0.25)
        self.assertTrue(torch.allclose(compose(x, [a, b]), compose(x, [b, a])))

    def test_sequential_strategy_chains_on_running_output(self):
        x = torch.ones(2, 3)
        sequential = FixtureIdentityStrategy()
        additive = FixtureScaleStrategy(init_scale=0.5)
        # sequential first (no-op on running `out`), then additive computed
        # against the ORIGINAL shared_output (per module docstring semantics)
        out = compose(x, [sequential, additive])
        self.assertTrue(torch.allclose(out, 1.5 * x))

    def test_unknown_combine_op_raises_valueerror(self):
        class BadStrategy(ComposeStrategy):
            combine_op = "bogus_op"

            def forward(self, shared_output):
                return shared_output

        with self.assertRaises(ValueError) as ctx:
            compose(torch.ones(2, 2), [BadStrategy()])
        self.assertIn("bogus_op", str(ctx.exception))
        self.assertIn("BadStrategy", str(ctx.exception))


class CompositionWiringCriterionTests(unittest.TestCase):
    """Fixture-only success criterion: stack a second strategy with the
    shared trunk via declared-order composition and train end-to-end
    (forward + backward, no shape/gradient errors) on a synthetic 4-task
    setup."""

    def test_composed_trunk_trains_without_shape_or_gradient_errors(self):
        tasks = [TASK_REGISTRY.get(k) for k in ("ks", "si", "er", "ic")]
        torch.manual_seed(0)
        trunk = build_trunk(
            upstream_model_type="wavlm_large",
            embedding_dim_shared1=16,
            embedding_dim_shared2=8,
            layer_pooling_type="mean",
            dropout_prob_shared1=0.0,
            dropout_prob_shared2=0.0,
        )
        heads = build_heads(tasks, in_features=8)
        model = MultiTaskModel(trunk, heads)
        fixture = FixtureScaleStrategy(init_scale=0.1)

        class ComposedModel(nn.Module):
            def __init__(self, base_model, fixture_strategy):
                super().__init__()
                self.base_model = base_model
                self.fixture_strategy = fixture_strategy

            def forward(self, input_seq):
                embedding = self.base_model.trunk(input_seq)
                composed = compose(embedding, [self.fixture_strategy])
                logits_list = [head(composed) for head in self.base_model.heads]
                pred_list = [torch.argmax(logits, dim=1) for logits in logits_list]
                from mtlkit.heads import MultiClassifierOutput

                return MultiClassifierOutput(logits=tuple(logits_list), prediction=tuple(pred_list))

        composed_model = ComposedModel(model, fixture)

        torch.manual_seed(1)
        input_seq = torch.randn(5, 25, 1024)
        outputs = composed_model(input_seq)

        self.assertEqual(len(outputs.logits), 4)
        for logits, task in zip(outputs.logits, tasks):
            self.assertEqual(logits.shape, (5, task.num_classes))

        loss = torch.stack([logits.sum() for logits in outputs.logits]).sum()
        loss.backward()

        self.assertIsNotNone(fixture.scale.grad)
        self.assertIsNotNone(trunk.projector_layer.weight.grad)
        for head in heads:
            self.assertIsNotNone(head.weight.grad)


if __name__ == "__main__":
    unittest.main()
