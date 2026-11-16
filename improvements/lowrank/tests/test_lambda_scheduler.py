"""Unit tests for the torch-independent RAMUSA lambda scheduler."""

import unittest

from improvements.lowrank.utils.lambda_scheduler import get_ramusa_lambda


class RamusaLambdaSchedulerTests(unittest.TestCase):
    def test_fixed_schedule(self):
        values = [get_ramusa_lambda("fixed", epoch, 0.1) for epoch in range(6)]
        self.assertEqual(values, [0.1] * 6)

    def test_linear_warmup_and_clipping(self):
        values = [
            get_ramusa_lambda("linear_warmup", epoch, 0.1, warmup_epochs=5)
            for epoch in range(6)
        ]
        for actual, expected in zip(values, (0.02, 0.04, 0.06, 0.08, 0.1, 0.1)):
            self.assertAlmostEqual(actual, expected)
        self.assertTrue(all(value <= 0.1 for value in values))

    def test_delayed_linear_warmup(self):
        values = [
            get_ramusa_lambda(
                "delayed_linear_warmup", epoch, 0.1,
                warmup_epochs=5, delay_epochs=2,
            )
            for epoch in range(8)
        ]
        for actual, expected in zip(values, (0.0, 0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.1)):
            self.assertAlmostEqual(actual, expected)

    def test_zero_warmup_and_validation(self):
        self.assertEqual(get_ramusa_lambda("linear_warmup", 0, 0.1), 0.1)
        self.assertEqual(
            get_ramusa_lambda("delayed_linear_warmup", 4, 0.1, delay_epochs=2), 0.1
        )
        with self.assertRaises(ValueError):
            get_ramusa_lambda("fixed", 0, -0.1)
        with self.assertRaises(ValueError):
            get_ramusa_lambda("fixed", 0, float("nan"))
        with self.assertRaises(ValueError):
            get_ramusa_lambda("unknown", 0, 0.1)


if __name__ == "__main__":
    unittest.main()