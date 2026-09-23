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
    """Bias-free rectangular low-rank update ``delta(x) = U(Vx)``."""

    def __init__(self, input_dim: int, output_dim: int, rank: int):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        max_rank = min(input_dim, output_dim)
        if rank > max_rank:
            raise ValueError(
                f"rank ({rank}) cannot exceed min(input_dim, output_dim) "
                f"({max_rank})"
            )

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.rank = int(rank)
        self.down = nn.Linear(self.input_dim, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.output_dim, bias=False)
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
    """Three-task wavCSE model with independent low-rank FC2 updates.

    The existing FC2 and its bias are shared; each task uses the bias-free
    residual ``U_t V_t`` on the pre-FC2 input. This is FTN-inspired, not an
    exact reproduction of FTN.
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
        fc2_input_dim = self.hidden_layer.in_features
        fc2_output_dim = self.hidden_layer.out_features
        self.task_updates = nn.ModuleList(
            [
                LowRankTaskUpdate(fc2_input_dim, fc2_output_dim, self.ftn_rank)
                for _ in self.TASK_ORDER
            ]
        )

        logging.info(
            "FTN-inspired FC2 decomposition: task_type=%s, rank=%d, "
            "fc2_shape=%dx%d, task_order=%s",
            task_type,
            self.ftn_rank,
            fc2_output_dim,
            fc2_input_dim,
            self.TASK_ORDER,
        )

    def _pre_fc2_representation(
        self, input_seq: torch.Tensor, apply_dropout: bool
    ) -> torch.Tensor:
        x = self.projector_layer(input_seq)
        x = self.pooling.get_vector_after_pooling(x, dim=1)
        if apply_dropout:
            x = self.dropout_shared1(x)
        return x

    def _adapted_representations(
        self, x: torch.Tensor, apply_dropout: bool
    ) -> tuple[torch.Tensor, ...]:
        shared_h = self.hidden_layer(x)
        adapted = tuple(
            shared_h + task_update(x) for task_update in self.task_updates
        )
        if apply_dropout:
            adapted = tuple(self.dropout_shared2(h) for h in adapted)
        return adapted

    def _shared_fc2_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the original shared FC2, including its shared bias."""
        return self.hidden_layer(x)

    def forward(self, input_seq: torch.Tensor) -> MultiClassifierOutput:
        x = self._pre_fc2_representation(input_seq, apply_dropout=True)
        adapted = self._adapted_representations(x, apply_dropout=True)

        logits = tuple(
            classifier(z) for classifier, z in zip(self.classifiers, adapted)
        )
        predictions = tuple(torch.argmax(task_logits, dim=1) for task_logits in logits)
        return MultiClassifierOutput(logits=logits, prediction=predictions)

    def get_all_embeddings(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Return the shared baseline FC2 representation with dropout disabled."""
        x = self._pre_fc2_representation(input_seq, apply_dropout=False)
        return self._shared_fc2_representation(x)

    def get_task_adapted_embeddings(
        self, input_seq: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        x = self._pre_fc2_representation(input_seq, apply_dropout=False)
        return dict(
            zip(
                self.TASK_ORDER,
                self._adapted_representations(x, apply_dropout=False),
            )
        )

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

    def get_shared_weight_norm(self) -> float:
        """Return the Frobenius norm of the existing shared FC2 weight."""
        with torch.no_grad():
            return float(torch.linalg.vector_norm(self.hidden_layer.weight).item())

    def get_relative_delta_norms(self) -> Dict[str, float]:
        """Return ``||Delta_W_t||_F / ||W_shared||_F`` for each task."""
        delta_norms = self.get_delta_norms()
        shared_norm = self.get_shared_weight_norm()
        if shared_norm == 0.0:
            return {task: 0.0 for task in self.TASK_ORDER}
        return {task: norm / shared_norm for task, norm in delta_norms.items()}

    def get_delta_cosine_similarities(self) -> Dict[str, float]:
        """Materialize update matrices and compare them for diagnostics only."""
        with torch.no_grad():
            deltas = self.get_delta_weights()
            similarities = {}
            for left, right in (("ks", "si"), ("ks", "er"), ("si", "er")):
                left_flat = deltas[left].reshape(-1)
                right_flat = deltas[right].reshape(-1)
                if left_flat.norm() == 0 or right_flat.norm() == 0:
                    similarity = 0.0
                else:
                    similarity = float(
                        F.cosine_similarity(left_flat, right_flat, dim=0).item()
                    )
                similarities[f"{left}_{right}"] = similarity
            return similarities

    def get_pooling_weights(self):
        """Preserve the baseline analysis API for learnable weighted pooling."""
        if self.layer_pooling_type != "weighted":
            return None
        return F.softmax(self.pooling.position_weights, dim=0)
