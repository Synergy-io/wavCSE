"""Pooling modules for aggregating transformer layer outputs.

Thin compatibility wrapper (Next Step 5 / Eng Review decision D1): the real
implementation moved to `mtlkit/pooling.py`, moved verbatim and now
registry-dispatched. This module re-exports the exact same `Pooling` class
under its original import path so every existing consumer
(`from pooling.pooling import Pooling`) keeps working unmodified — see
`mtlkit/tests/test_facade_parity.py` for the parity proof.

Author: Braveenan Sritharan
Created: 2026-01-19
"""

from mtlkit.pooling import Pooling  # noqa: F401
