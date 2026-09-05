"""mtlkit/imbalance.py — per-task class reweighting inside the masked loss.

Builds an inverse-frequency-weighted `nn.CrossEntropyLoss` per task, for use
as `mtlkit.heads.masked_ce_loss`'s `loss_fn` argument. Weights are
normalized to mean 1.0 over classes that actually occur, so the overall
loss scale stays comparable to an unweighted `CrossEntropyLoss` — a
reweighted run's loss values are still readable next to an unweighted run's,
not inflated by an arbitrary constant.
"""

from typing import Sequence

import torch
import torch.nn as nn


def inverse_frequency_weights(class_counts: Sequence[int]) -> torch.Tensor:
    """Inverse-frequency class weights: ``weight[c] = 1 / count[c]``,
    normalized so the mean weight over classes with ``count > 0`` is 1.0.
    A class with zero count gets weight 0 (it never occurs in this split;
    a 1/0 weight would be undefined and can't affect any real loss term).
    """
    counts = torch.tensor(list(class_counts), dtype=torch.float)
    weights = torch.zeros_like(counts)
    nonzero = counts > 0
    weights[nonzero] = 1.0 / counts[nonzero]

    mean_nonzero = weights[nonzero].mean() if bool(nonzero.any()) else torch.tensor(1.0)
    if float(mean_nonzero) > 0:
        weights = weights / mean_nonzero
    return weights


def build_weighted_loss_fn(class_counts: Sequence[int], ignore_index: int = -1) -> nn.Module:
    """`nn.CrossEntropyLoss` configured with inverse-frequency class weights
    for one task — drop-in `loss_fn` for `mtlkit.heads.masked_ce_loss`."""
    weights = inverse_frequency_weights(class_counts)
    return nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)
