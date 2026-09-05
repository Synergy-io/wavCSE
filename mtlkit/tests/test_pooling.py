"""Unit tests for mtlkit/pooling.py.

Includes a dispatch-equivalence check against `downstream/pooling/pooling.py`
(today's implementation) per the seam table's "Validated by" column.
"""

import os
import sys
import unittest

import torch

import mtlkit.pooling as mtlkit_pooling

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DOWNSTREAM_DIR = os.path.join(REPO_ROOT, "downstream")
for path in (DOWNSTREAM_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from pooling.pooling import Pooling as DownstreamPooling  # noqa: E402


class PoolingRegistryTests(unittest.TestCase):
    def test_all_ten_strategies_registered(self):
        self.assertEqual(
            mtlkit_pooling.POOLING_REGISTRY.list(),
            ["auto", "gated", "lnp", "lse", "max", "mean", "mix", "sap", "smp", "weighted"],
        )

    def test_unsupported_pooling_type_raises_valueerror(self):
        with self.assertRaises(ValueError) as ctx:
            mtlkit_pooling.Pooling("bogus")
        self.assertEqual(str(ctx.exception), "Unsupported pooling type: bogus")


class PoolingBasicOpsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(4, 6, 8)  # [B, L, D]

    def test_mean_pooling_shape_and_value(self):
        p = mtlkit_pooling.Pooling("mean")
        out = p.get_vector_after_pooling(self.x, dim=1)
        self.assertEqual(out.shape, (4, 8))
        self.assertTrue(torch.allclose(out, self.x.mean(dim=1)))

    def test_max_pooling(self):
        p = mtlkit_pooling.Pooling("max")
        out = p.get_vector_after_pooling(self.x, dim=1)
        self.assertTrue(torch.allclose(out, self.x.max(dim=1).values))

    def test_weighted_pooling_requires_param(self):
        with self.assertRaises(ValueError):
            mtlkit_pooling.Pooling("weighted")

    def test_sap_requires_param(self):
        with self.assertRaises(ValueError):
            mtlkit_pooling.Pooling("sap")

    def test_get_last_aux_populated_for_weighted(self):
        p = mtlkit_pooling.Pooling("weighted", pooling_param=6)
        p.get_vector_after_pooling(self.x, dim=1)
        aux = p.get_last_aux()
        self.assertIn("position_weights", aux)
        self.assertIn("weighted_weights", aux)


class DispatchEquivalenceWithDownstreamTests(unittest.TestCase):
    """Today's pooling strategy through mtlkit's registry reproduces the
    exact numeric output of `downstream/pooling/pooling.py` on the same
    input and the same seeded parameters."""

    def _assert_equivalent(self, pooling_type, pooling_param=None, dim=1):
        torch.manual_seed(42)
        x = torch.randn(3, 5, 7)

        torch.manual_seed(1)
        mtlkit_p = mtlkit_pooling.Pooling(pooling_type, pooling_param=pooling_param)
        torch.manual_seed(1)
        downstream_p = DownstreamPooling(pooling_type, pooling_param=pooling_param)

        out_mtlkit = mtlkit_p.get_vector_after_pooling(x, dim=dim)
        out_downstream = downstream_p.get_vector_after_pooling(x, dim=dim)
        self.assertTrue(
            torch.allclose(out_mtlkit, out_downstream),
            msg=f"mismatch for pooling_type={pooling_type!r}",
        )

    def test_mean(self):
        self._assert_equivalent("mean")

    def test_max(self):
        self._assert_equivalent("max")

    def test_mix(self):
        self._assert_equivalent("mix", pooling_param=0.3)

    def test_lnp(self):
        self._assert_equivalent("lnp", pooling_param=2)

    def test_smp(self):
        self._assert_equivalent("smp", pooling_param=1.0)

    def test_lse(self):
        self._assert_equivalent("lse", pooling_param=1.0)

    def test_weighted(self):
        self._assert_equivalent("weighted", pooling_param=5)

    def test_gated(self):
        self._assert_equivalent("gated", pooling_param=5)

    def test_auto(self):
        self._assert_equivalent("auto", pooling_param=0.01)

    def test_sap(self):
        self._assert_equivalent("sap", pooling_param=7)


if __name__ == "__main__":
    unittest.main()
