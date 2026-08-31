"""RAMUSA-inspired neural low-rank multi-task model."""

from typing import List

import torch
import torch.nn as nn

from model.downstream_model import DownstreamMultiTaskModel, MultiClassifierOutput


class DownstreamMultiTaskModelRAMUSA(DownstreamMultiTaskModel):
    """Shared wavCSE representation followed by three task adapters.

    This is a neural adaptation of RAMUSA: autograd optimizes the adapters
    using a truncated nuclear-norm penalty; no singular vectors are removed.
    """

    SUPPORTED_TASK_TYPE = "ks_si_er"
    TASK_NAMES = ("ks", "si", "er")

    def __init__(self, *args, adapter_dim=None, **kwargs):
        task_type = kwargs.get("task_type")
        if task_type is None and len(args) >= 2:
            task_type = args[1]
        if task_type != self.SUPPORTED_TASK_TYPE:
            raise ValueError(
                "RAMUSA supports only task_type='ks_si_er'; intent classification is not supported."
            )
        super().__init__(*args, **kwargs)
        shared_dim = int(self.hidden_layer.out_features)
        adapter_dim = shared_dim if adapter_dim is None else int(adapter_dim)
        if adapter_dim <= 0:
            raise ValueError("adapter_dim must be positive")

        self.adapter_dim = adapter_dim
        self.adapters = nn.ModuleList(
            nn.Linear(shared_dim, adapter_dim) for _ in self.TASK_NAMES
        )
        output_dims = self._output_dims_from_task_type(task_type)
        self.classifiers = nn.ModuleList(
            nn.Linear(adapter_dim, output_dim) for output_dim in output_dims
        )

    def _shared_embedding(self, input_seq: torch.Tensor, apply_dropout: bool) -> torch.Tensor:
        embedding = self.projector_layer(input_seq)
        embedding = self.pooling.get_vector_after_pooling(embedding, dim=1)
        if apply_dropout:
            embedding = self.dropout_shared1(embedding)
        embedding = self.hidden_layer(embedding)
        if apply_dropout:
            embedding = self.dropout_shared2(embedding)
        return embedding

    def forward(self, input_seq):
        shared_embedding = self._shared_embedding(input_seq, apply_dropout=True)
        logits_list = []
        prediction_list = []
        for adapter, classifier in zip(self.adapters, self.classifiers):
            logits = classifier(adapter(shared_embedding))
            logits_list.append(logits)
            prediction_list.append(torch.argmax(logits, dim=1))
        return MultiClassifierOutput(
            logits=tuple(logits_list), prediction=tuple(prediction_list)
        )

    def get_all_embeddings(self, input_seq):
        return self._shared_embedding(input_seq, apply_dropout=False)

    def build_ramusa_parameter_matrix(self) -> torch.Tensor:
        """Stack adapter weight vectors as columns, preserving gradients."""
        return torch.stack([adapter.weight.reshape(-1) for adapter in self.adapters], dim=1)

    def get_ramusa_singular_values(self) -> torch.Tensor:
        return torch.linalg.svdvals(self.build_ramusa_parameter_matrix())

    def get_adapter_weight_tensors(self) -> List[torch.Tensor]:
        return [adapter.weight for adapter in self.adapters]
