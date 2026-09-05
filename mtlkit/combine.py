"""mtlkit/combine.py — the loss-combination hook (closes issue #10).

Before this seam, adding a new weighting strategy meant copy-pasting the
whole trainer's batch-processing method (MTRL's ~85-line
`_process_batch` override is the proof case). `combine()` isolates exactly
the "per-task losses -> one scalar" step behind one strategy interface.

Per-step contract (tightened by Eng Review cross-model tension, 2026-09-06):
    in:  per_task_losses  — one Optional[Tensor] per task slot, in task-slot
                            order. None means "no valid sample for this task
                            in this batch" (mtlkit.heads.masked_ce_loss's
                            contract) — a strategy MUST skip None entries,
                            never treat None as zero loss silently.
         per_task_masks   — one boolean Tensor [B] per task slot (labels !=
                            ignore_index), for strategies that need row-level
                            detail beyond just "which tasks had valid rows".
         head_params      — one (weight, bias) tuple per classifier head, in
                            task-slot order. These are the LIVE,
                            autograd-connected `nn.Parameter` tensors from
                            the model — never detached copies, or a strategy
                            that folds them into its loss (e.g. MTRL's
                            tr(W Omega^-1 W^T) regularizer) silently breaks
                            backprop to those heads. Any task-specific
                            repacking of these raw tensors (MTRL's
                            `get_task_parameter_matrix`-style transform) is
                            the STRATEGY's own responsibility, not this
                            seam's — `combine()` only guarantees the raw,
                            live inputs.
    out: one scalar Tensor — the combined loss, ready for `.backward()`.

Strategy objects are stateful and get epoch-lifecycle hooks
(`on_epoch_begin`/`on_epoch_end`) because a real strategy (MTRL) has a
warmup period and an epoch-scheduled analytic buffer refresh that must not
run on every batch.
"""

from typing import List, Optional, Tuple

import torch

from mtlkit.registry import Registry

# (weight, bias) — LIVE autograd-connected parameters, see module docstring.
HeadParams = Tuple[torch.Tensor, torch.Tensor]


class CombineStrategy:
    """Base class for loss-combination strategies.

    Subclasses implement `__call__`; the epoch-lifecycle hooks default to
    no-ops for strategies (like `uniform_average`) that carry no state.
    """

    def on_epoch_begin(self, epoch: int) -> None:
        pass

    def on_epoch_end(self, epoch: int) -> None:
        pass

    def __call__(
        self,
        per_task_losses: List[Optional[torch.Tensor]],
        per_task_masks: List[torch.Tensor],
        head_params: List[HeadParams],
    ) -> torch.Tensor:
        raise NotImplementedError


COMBINE_REGISTRY: Registry[type] = Registry("combine strategy")


def register_combine_strategy(key: str):
    """Class decorator: ``@register_combine_strategy("my_strategy")``."""

    def _decorator(cls: type) -> type:
        COMBINE_REGISTRY.register(key, cls)
        return cls

    return _decorator


def build_combine_strategy(key: str, **kwargs) -> CombineStrategy:
    cls = COMBINE_REGISTRY.get(key)
    return cls(**kwargs)


@register_combine_strategy("uniform_average")
class UniformAverageCombine(CombineStrategy):
    """Default strategy: reproduces the original inlined ``1/num_tasks``
    arithmetic exactly — ``downstream/trainer/trainer_model.py``'s
    ``loss_weight = 1.0 / float(self.num_tasks)`` followed by
    ``loss_all = loss_all + loss_task * loss_weight`` per task, summed.

    Known limitation (Eng Review D2, documented in the design doc's Success
    Criteria, accepted as-is): the weight is per task-SLOT, not per
    valid-task-per-sample — a sample valid at multiple task slots (e.g. FSC's
    3 heads) contributes proportionally more gradient weight than a
    single-task sample. Not addressed by this strategy.
    """

    def __call__(self, per_task_losses, per_task_masks, head_params):
        num_tasks = len(per_task_losses)
        loss_weight = 1.0 / float(num_tasks)

        device = torch.device("cpu")
        for loss_t in per_task_losses:
            if loss_t is not None:
                device = loss_t.device
                break

        loss_all = torch.zeros((), device=device)
        for loss_t in per_task_losses:
            if loss_t is not None:
                loss_all = loss_all + loss_t * loss_weight
        return loss_all
