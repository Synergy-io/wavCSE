"""mtlkit/heads.py — masked loss/accuracy, shared trunk, and per-task heads.

Closes issue #8's loss-function half: `masked_ce_loss`/`masked_accuracy` are
now defined once here instead of independently in
`downstream/trainer/trainer_utils.py` and `downstream/evaluator/evaluator_utils.py`.
`build_trunk`/`build_heads` assemble the shared backbone + per-task
classification heads that `downstream/model/downstream_model.py`'s
`DownstreamMultiTaskModel` currently builds inline (Next Step 2).

    input_seq [B, L, input_dim]
         |
         v  projector_layer (Linear) + dropout
    [B, L, shared1]
         |
         v  Pooling.get_vector_after_pooling(dim=1)   (mtlkit.pooling)
    [B, shared1]
         |
         v  hidden_layer (Linear) + dropout
    [B, shared2]
         |
         +--> head[0] (Linear) -> logits[0] -> argmax -> pred[0]
         +--> head[1] (Linear) -> logits[1] -> argmax -> pred[1]
         +--> ...                                          (one per task)

Note on the port: `downstream/model/downstream_model.py:135`'s
`get_pooling_weights` reads `self.pooling.pooling_weights`, but `Pooling`
(both the original and mtlkit's port) stores that parameter as
`position_weights` — calling it on a "weighted"-pooling model raises
`AttributeError` today. `downstream/` is frozen outside Next Step 5's
reviewed rewrite, so this is flagged, not silently patched there; mtlkit's
`MultiTaskModel.get_pooling_weights` below uses the correct attribute name
and does not reproduce the bug.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mtlkit.pooling import Pooling
from mtlkit.tasks import TaskSpec

# ---------------------------------------------------------------------------
# Masked loss / accuracy — unified copy of trainer_utils.py / evaluator_utils.py
# ---------------------------------------------------------------------------


def masked_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: nn.Module,
    ignore_index: int = -1,
) -> Tuple[Optional[torch.Tensor], int]:
    """Cross-entropy over only the rows where ``labels != ignore_index``.

    Returns ``(None, 0)`` when no row in this batch is valid for this task —
    callers (mtlkit.combine's default strategy) must skip a ``None`` loss
    rather than treat it as zero.
    """
    mask = labels != ignore_index
    labels_masked = labels[mask]
    if labels_masked.numel() == 0:
        return None, 0
    logits_masked = logits[mask, :]
    loss = loss_fn(logits_masked, labels_masked)
    return loss, int(labels_masked.size(0))


def masked_accuracy(
    pred: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -1,
) -> Tuple[int, int]:
    """(correct, total) over only the rows where ``labels != ignore_index``."""
    mask = labels != ignore_index
    labels_masked = labels[mask]
    if labels_masked.numel() == 0:
        return 0, 0
    pred_masked = pred[mask]
    correct = int((pred_masked == labels_masked).sum().item())
    total = int(labels_masked.size(0))
    return correct, total


# ---------------------------------------------------------------------------
# Model assembly
# ---------------------------------------------------------------------------


class MultiClassifierOutput:
    def __init__(self, logits=None, prediction=None):
        self.logits = logits
        self.prediction = prediction


_UPSTREAM_INPUT_DIMS = {"base": 768, "large": 1024}


def input_dim_from_upstream(upstream_model_type: str) -> int:
    """Same convention as `DownstreamMultiTaskModel._input_dim_from_upstream`:
    the upstream model's size variant (suffix of `upstream_model_type`, e.g.
    "wavlm_large") determines its hidden size."""
    variation = upstream_model_type.split("_")[-1].lower()
    try:
        return _UPSTREAM_INPUT_DIMS[variation]
    except KeyError:
        raise ValueError(
            f"Unknown upstream_model_variation='{variation}' "
            f"from upstream_model_type='{upstream_model_type}'. "
            f"Expected one of: {', '.join(_UPSTREAM_INPUT_DIMS)}."
        ) from None


class Trunk(nn.Module):
    """Shared backbone: projector -> layer pooling -> hidden layer.

    Equivalent to the shared-parameter portion of
    `DownstreamMultiTaskModel.__init__`/`forward` (everything before the
    per-task classifier heads).
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim_shared1: int,
        embedding_dim_shared2: int,
        layer_pooling_type: str,
        dropout_prob_shared1: float,
        dropout_prob_shared2: float,
        layer_pooling_param: Optional[Union[int, float]] = None,
    ):
        super().__init__()
        self.projector_layer = nn.Linear(input_dim, embedding_dim_shared1)
        self.dropout_shared1 = nn.Dropout(p=dropout_prob_shared1)

        self.hidden_layer = nn.Linear(embedding_dim_shared1, embedding_dim_shared2)
        self.dropout_shared2 = nn.Dropout(p=dropout_prob_shared2)

        self.layer_pooling_type = layer_pooling_type
        self.pooling = Pooling(layer_pooling_type, pooling_param=layer_pooling_param)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        embedding_shared = self.projector_layer(input_seq)  # [B, L, shared1]
        embedding_shared = self.pooling.get_vector_after_pooling(embedding_shared, dim=1)
        embedding_shared = self.dropout_shared1(embedding_shared)

        embedding_shared = self.hidden_layer(embedding_shared)  # [B, shared2]
        embedding_shared = self.dropout_shared2(embedding_shared)
        return embedding_shared

    def get_pooling_weights(self):
        if self.layer_pooling_type != "weighted":
            return None
        return F.softmax(self.pooling.position_weights, dim=0)


def build_trunk(
    upstream_model_type: str,
    embedding_dim_shared1: int,
    embedding_dim_shared2: int,
    layer_pooling_type: str,
    dropout_prob_shared1: float,
    dropout_prob_shared2: float,
    layer_pooling_param: Optional[Union[int, float]] = None,
) -> Trunk:
    input_dim = input_dim_from_upstream(upstream_model_type)
    return Trunk(
        input_dim=input_dim,
        embedding_dim_shared1=embedding_dim_shared1,
        embedding_dim_shared2=embedding_dim_shared2,
        layer_pooling_type=layer_pooling_type,
        dropout_prob_shared1=dropout_prob_shared1,
        dropout_prob_shared2=dropout_prob_shared2,
        layer_pooling_param=layer_pooling_param,
    )


def build_heads(tasks: List[TaskSpec], in_features: int) -> nn.ModuleList:
    """One `nn.Linear(in_features, task.num_classes)` per task, in the given
    order — the order the caller supplies IS the task-slot order used
    throughout combine()/compose() (per-task losses/masks are positional)."""
    return nn.ModuleList([nn.Linear(in_features, task.num_classes) for task in tasks])


class MultiTaskModel(nn.Module):
    """Shared trunk + per-task heads, assembled from `build_trunk`/`build_heads`.

    Equivalent to `DownstreamMultiTaskModel`, but trunk and heads are
    injected rather than constructed from a `task_type` string internally —
    task/dataset resolution is `mtlkit.tasks`'s job, not the model's.
    """

    def __init__(self, trunk: Trunk, heads: nn.ModuleList):
        super().__init__()
        self.trunk = trunk
        self.heads = heads

    def forward(self, input_seq: torch.Tensor) -> MultiClassifierOutput:
        embedding_shared = self.trunk(input_seq)

        logits_list = []
        pred_list = []
        for head in self.heads:
            logits = head(embedding_shared)
            pred = torch.argmax(logits, dim=1)
            logits_list.append(logits)
            pred_list.append(pred)

        return MultiClassifierOutput(logits=tuple(logits_list), prediction=tuple(pred_list))

    def get_all_embeddings(self, input_seq: torch.Tensor) -> torch.Tensor:
        return self.trunk(input_seq)

    def get_pooling_weights(self):
        return self.trunk.get_pooling_weights()
