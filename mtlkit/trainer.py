"""mtlkit/trainer.py — thin loop hosting the combine() hook.

Not itself a seam (per the seam table) — `process_batch` below is the
generalized replacement for the per-task-loss-accumulation portion of
`downstream/trainer/trainer_model.py`'s `_process_batch` /
`improvements/taskrelation/01-mtrl/mtrl_trainer.py`'s duplicated override,
with the actual weighting logic delegated to a `mtlkit.combine.CombineStrategy`.

    input_seq, labels_list
         |
         v  model(input_seq) -> MultiClassifierOutput
    logits_tuple, pred_tuple
         |
         v  masked_ce_loss / masked_accuracy per task   (mtlkit.heads)
    per_task_losses, per_task_masks, correct_task, samples_task
         |
         v  combine_strategy(per_task_losses, per_task_masks, head_params)
    loss_all (scalar, backprop-ready)          (mtlkit.combine)

L1/L2 weight regularization and the optimizer step are NOT this module's
job — they're generic trainer-loop concerns applied around `process_batch`'s
result by whatever owns the full training loop (the Next Step 5 wrapper for
the base 4-task run, MTRL's migrated trainer for its own L1/L2 + Omega
regularizer). `process_batch` only replaces the per-task-loss-combination
step that combine() seam claims.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import torch
import torch.nn as nn

from mtlkit.combine import CombineStrategy, HeadParams
from mtlkit.heads import masked_accuracy, masked_ce_loss


@dataclass
class BatchResult:
    loss_all: torch.Tensor  # scalar, backprop-ready — combine_strategy's output
    loss_task: Dict[int, float] = field(default_factory=dict)
    valid_count_task: Dict[int, int] = field(default_factory=dict)
    correct_task: Dict[int, int] = field(default_factory=dict)
    samples_task: Dict[int, int] = field(default_factory=dict)


def head_params(model: nn.Module) -> List[HeadParams]:
    """Extract (weight, bias) — live, autograd-connected — from a
    `mtlkit.heads.MultiTaskModel`'s per-task classifier heads, in task-slot
    order. This is exactly the `head_params` combine() strategies receive."""
    return [(head.weight, head.bias) for head in model.heads]


def process_batch(
    model: nn.Module,
    input_seq: torch.Tensor,
    labels_list: List[torch.Tensor],
    loss_fn: nn.Module,
    combine_strategy: CombineStrategy,
    ignore_index: int = -1,
) -> BatchResult:
    """Forward pass + per-task masked loss/accuracy + combine() — the
    seam-ified replacement for `_process_batch`'s per-task-loss half.

    Does NOT call `.backward()` or step an optimizer — the caller owns the
    training loop and decides when (or whether, e.g. eval mode) to do that.
    """
    outputs = model(input_seq)
    logits_tuple = outputs.logits
    pred_tuple = outputs.prediction
    num_tasks = len(logits_tuple)

    per_task_losses = []
    per_task_masks = []
    valid_count_task: Dict[int, int] = {}
    loss_task: Dict[int, float] = {}

    for t in range(num_tasks):
        loss_t, valid = masked_ce_loss(
            logits=logits_tuple[t],
            labels=labels_list[t],
            loss_fn=loss_fn,
            ignore_index=ignore_index,
        )
        per_task_losses.append(loss_t)
        per_task_masks.append(labels_list[t] != ignore_index)
        valid_count_task[t] = valid
        loss_task[t] = float(loss_t.item()) if loss_t is not None else 0.0

    loss_all = combine_strategy(per_task_losses, per_task_masks, head_params(model))

    correct_task: Dict[int, int] = {}
    samples_task: Dict[int, int] = {}
    for t in range(num_tasks):
        c, s = masked_accuracy(pred_tuple[t], labels_list[t], ignore_index=ignore_index)
        correct_task[t] = c
        samples_task[t] = s

    return BatchResult(
        loss_all=loss_all,
        loss_task=loss_task,
        valid_count_task=valid_count_task,
        correct_task=correct_task,
        samples_task=samples_task,
    )
