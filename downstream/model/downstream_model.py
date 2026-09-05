"""
Downstream multi task classification model.

This module defines a shared representation based downstream
multi task learning model that operates on pre computed
upstream embeddings. It supports:
- Layer wise pooling across transformer layers
- Shared projection and hidden layers
- Multiple task specific classification heads
- Retrieval of pooled embeddings and pooling weights

The model is designed to be used with upstream SSL
representations such as WavLM.

Thin compatibility wrapper (Next Step 5 / Eng Review decision D1): task_type
parsing and output-dim resolution now delegate to `mtlkit.tasks`/
`mtlkit.heads` (closes part of issue #8), and `Pooling` is `mtlkit.pooling`'s
(via `downstream/pooling/pooling.py`'s re-export).

Structural note: this class keeps its ORIGINAL flat `nn.Module` attribute
layout (`projector_layer`, `dropout_shared1`, `hidden_layer`,
`dropout_shared2`, `classifiers`, `pooling`, `layer_pooling_type`) as direct
instance attributes rather than delegating assembly to
`mtlkit.heads.build_trunk`/`build_heads`/`MultiTaskModel` (which nest them
under a `trunk` submodule) -- `improvements/clustering/models/ncmtl_model.py`
and `improvements/decomposition/models/ftn_model.py` both subclass this
class and reach directly into that flat structure (`self.hidden_layer`,
`self.projector_layer`, etc., not just the public constructor/forward
API). A nested-trunk assembly silently broke both subclasses (caught by
running `improvements/clustering/tests/test_ncmtl.py`, a pre-existing,
untouched consumer test) before this facade-parity wrapper landed. mtlkit's
own native model assembly (`mtlkit.heads.MultiTaskModel`) is unaffected and
used by callers that don't need this legacy flat shape.

Deliberate deviation from strict byte-for-byte parity, documented rather
than silently carried forward: the original `get_pooling_weights` read
`self.pooling.pooling_weights`, an attribute `Pooling` never defined (the
real attribute is `position_weights`) -- calling this method on a
"weighted"-pooling model always raised `AttributeError`, so no consumer
could have been relying on that crash succeeding. This wrapper fixes it.

Author: Braveenan Sritharan
Created: 2026-01-19
"""

import logging
from typing import Optional, Union

import torch
import torch.nn.functional as F
import torch.nn as nn

from pooling.pooling import Pooling
from mtlkit.heads import MultiClassifierOutput, input_dim_from_upstream  # noqa: F401
from mtlkit.tasks import TASK_REGISTRY, dataset_array_from_task_type
from utils.pooling_id import make_pooling_name


def _tasks_from_task_type(task_type: str):
    """Ordered, de-duplicated TaskSpec list for a task_type string -- same
    tokenization/dedup/error-message behavior as the original
    `_build_dataset_id_from_task_type`, sourced from mtlkit.tasks so this
    wrapper and mtlkit.tasks can never drift apart on what a token means."""
    task_tokens = task_type.split("_")
    seen_keys = []
    for t in task_tokens:
        if TASK_REGISTRY.try_get(t) is None:
            raise ValueError(
                f"Invalid task type token: '{t}' in task_type='{task_type}'"
            )
        if t not in seen_keys:
            seen_keys.append(t)
    return [TASK_REGISTRY.get(key) for key in seen_keys]


class DownstreamMultiTaskModel(nn.Module):
    def __init__(
        self,
        upstream_model_type: str,
        task_type: str,
        embedding_dim_shared1: int,
        embedding_dim_shared2: int,
        layer_pooling_type: str,
        dropout_prob_shared1: float,
        dropout_prob_shared2: float,
        layer_pooling_param: Optional[Union[int, float]] = None,
    ):
        super().__init__()

        input_dim = input_dim_from_upstream(upstream_model_type)
        tasks = _tasks_from_task_type(task_type)
        output_dim_array = [task.num_classes for task in tasks]

        self.projector_layer = nn.Linear(input_dim, embedding_dim_shared1)
        self.dropout_shared1 = nn.Dropout(p=dropout_prob_shared1)

        self.hidden_layer = nn.Linear(embedding_dim_shared1, embedding_dim_shared2)
        self.dropout_shared2 = nn.Dropout(p=dropout_prob_shared2)

        self.classifiers = nn.ModuleList([
            nn.Linear(embedding_dim_shared2, out_dim) for out_dim in output_dim_array
        ])

        # Pooling owns learnable parameters only when needed (weighted)
        self.layer_pooling_type = layer_pooling_type
        self.pooling = Pooling(layer_pooling_type, pooling_param=layer_pooling_param)

        layer_pool_name = make_pooling_name(layer_pooling_type, layer_pooling_param)
        logging.info(f"Layer pooling type: {layer_pool_name}")

    def _build_dataset_id_from_task_type(self, task_type: str):
        """Preserved for any external caller relying on this internal
        method's name; delegates to mtlkit.tasks (see module docstring)."""
        return dataset_array_from_task_type(task_type)

    def _output_dims_from_task_type(self, task_type: str):
        """Preserved for any external caller relying on this internal
        method's name (e.g. NCMTL's `__init__`)."""
        return [task.num_classes for task in _tasks_from_task_type(task_type)]

    def forward(self, input_seq):
        # input_seq: [B, L, input_dim] e.g., [2048, 25, 1024]
        embedding_shared = self.projector_layer(input_seq)  # [B, L, shared1]
        embedding_shared = self.pooling.get_vector_after_pooling(embedding_shared, dim=1)  # [B, shared1]
        embedding_shared = self.dropout_shared1(embedding_shared)

        embedding_shared = self.hidden_layer(embedding_shared)  # [B, shared2]
        embedding_shared = self.dropout_shared2(embedding_shared)

        logits_list = []
        pred_list = []
        for head in self.classifiers:
            logits = head(embedding_shared)
            pred = torch.argmax(logits, dim=1)
            logits_list.append(logits)
            pred_list.append(pred)

        return MultiClassifierOutput(
            logits=tuple(logits_list),
            prediction=tuple(pred_list)
        )

    def get_all_embeddings(self, input_seq):
        embedding_shared = self.projector_layer(input_seq)
        embedding_shared = self.pooling.get_vector_after_pooling(embedding_shared, dim=1)
        embedding_shared = self.hidden_layer(embedding_shared)
        return embedding_shared

    def get_pooling_weights(self):
        if self.layer_pooling_type != "weighted":
            return None

        return F.softmax(self.pooling.position_weights, dim=0)
