import logging
import os
import sys
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


_DOWNSTREAM_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "downstream")
)
if _DOWNSTREAM_DIR not in sys.path:
    sys.path.insert(0, _DOWNSTREAM_DIR)

from model.downstream_model import DownstreamMultiTaskModel, MultiClassifierOutput


class LowRankTaskUpdate(nn.Module):
    """Bias-free low-rank update delta(x) = U(Vx)"""

    def __init__(self, dim: int, rank: int):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if rank > dim:
            raise ValueError(f"rank ({rank}) cannot exceed dim ({dim})")

        self.dim = int(dim)
        self.rank = int(rank)
        self.down = nn.Linear(self.dim, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))

    def delta_weight(self) -> torch.Tensor:
        """Materialize ``U @ V`` on demand for diagnostics only."""
        return self.up.weight @ self.down.weight

    def delta_frobenius_norm(self) -> torch.Tensor:
        """Compute ``||U @ V||_F`` without materializing the dense update."""
        gram_up = self.up.weight.transpose(0, 1) @ self.up.weight
        gram_down = self.down.weight @ self.down.weight.transpose(0, 1)
        squared_norm = torch.sum(gram_up * gram_down.transpose(0, 1))
        return torch.sqrt(torch.clamp(squared_norm, min=0.0))


class DownstreamMultiTaskModelFTN(DownstreamMultiTaskModel):
    """Three-task wavCSE model with FTN-inspired low-rank task updates.

    The decomposition block is placed after FC2 and its existing dropout and
    before the task classifiers. For task ``t`` it computes
    ``z_t = W_shared h + U_t(V_t h) + b_shared``.
    """

    SUPPORTED_TASK_TYPE = "ks_si_er"
    TASK_ORDER = ("ks", "si", "er")

    def __init__(
        self,
        upstream_model_type: str,
        task_type: str,
        embedding_dim_shared1: int,
        embedding_dim_shared2: int,
        layer_pooling_type: str,
        dropout_prob_shared1: float,
        dropout_prob_shared2: float,
        ftn_rank: int = 8,
        layer_pooling_param: Optional[Union[int, float]] = None,
    ):
        if task_type != self.SUPPORTED_TASK_TYPE:
            raise ValueError(
                "The first FTN decomposition implementation supports only the "
                f"exact ordered task type '{self.SUPPORTED_TASK_TYPE}', got '{task_type}'."
            )

        super().__init__(
            upstream_model_type=upstream_model_type,
            task_type=task_type,
            embedding_dim_shared1=embedding_dim_shared1,
            embedding_dim_shared2=embedding_dim_shared2,
            layer_pooling_type=layer_pooling_type,
            dropout_prob_shared1=dropout_prob_shared1,
            dropout_prob_shared2=dropout_prob_shared2,
            layer_pooling_param=layer_pooling_param,
        )

        self.ftn_rank = int(ftn_rank)
        self.shared_adapter = nn.Linear(
            embedding_dim_shared2, embedding_dim_shared2, bias=True
        )
        self.task_updates = nn.ModuleList(
            [
                LowRankTaskUpdate(embedding_dim_shared2, self.ftn_rank)
                for _ in self.TASK_ORDER
            ]
        )
        self._reset_shared_adapter()

        logging.info(
            "FTN decomposition: task_type=%s, rank=%d, task_order=%s",
            task_type,
            self.ftn_rank,
            self.TASK_ORDER,
        )

    def _reset_shared_adapter(self) -> None:
        nn.init.eye_(self.shared_adapter.weight)
        nn.init.zeros_(self.shared_adapter.bias)

    def _common_representation(
        self, input_seq: torch.Tensor, apply_dropout: bool
    ) -> torch.Tensor:
        h = self.projector_layer(input_seq)
        h = self.pooling.get_vector_after_pooling(h, dim=1)
        if apply_dropout:
            h = self.dropout_shared1(h)
        h = self.hidden_layer(h)
        if apply_dropout:
            h = self.dropout_shared2(h)
        return h

    def _adapted_representations(self, h: torch.Tensor) -> tuple[torch.Tensor, ...]:
        shared_z = self.shared_adapter(h)
        return tuple(
            shared_z + task_update(h) for task_update in self.task_updates
        )

    def forward(self, input_seq: torch.Tensor) -> MultiClassifierOutput:
        h = self._common_representation(input_seq, apply_dropout=True)
        adapted = self._adapted_representations(h)

        logits = tuple(
            classifier(z) for classifier, z in zip(self.classifiers, adapted)
        )
        predictions = tuple(torch.argmax(task_logits, dim=1) for task_logits in logits)
        return MultiClassifierOutput(logits=logits, prediction=predictions)

    def get_all_embeddings(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Return the common FC2 representation before decomposition."""
        return self._common_representation(input_seq, apply_dropout=False)

    def get_task_adapted_embeddings(
        self, input_seq: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        h = self._common_representation(input_seq, apply_dropout=False)
        return dict(zip(self.TASK_ORDER, self._adapted_representations(h)))

    def get_delta_weights(self) -> Dict[str, torch.Tensor]:
        return {
            task: update.delta_weight()
            for task, update in zip(self.TASK_ORDER, self.task_updates)
        }

    def get_delta_norms(self) -> Dict[str, float]:
        with torch.no_grad():
            return {
                task: float(update.delta_frobenius_norm().item())
                for task, update in zip(self.TASK_ORDER, self.task_updates)
            }

    def get_shared_adapter_identity_distance(self) -> float:
        """Return ``||W_shared - I||_F`` without allocating an identity matrix."""
        with torch.no_grad():
            weight = self.shared_adapter.weight
            squared_distance = (
                torch.sum(weight.square())
                + weight.shape[0]
                - 2.0 * torch.trace(weight)
            )
            return float(torch.sqrt(torch.clamp(squared_distance, min=0.0)).item())

    def get_pooling_weights(self):
        """Preserve the baseline analysis API for learnable weighted pooling."""
        if self.layer_pooling_type != "weighted":
            return None
        return F.softmax(self.pooling.position_weights, dim=0)
