"""
wavCSE-MTRL: Multi-Task Relationship Learning (Zhang & Yeung, 2010/2012)

Zhang & Yeung proposed MTRL ("A convex formulation for learning task
relationships in multi-task learning", UAI 2010 / journal version 2012),
the central method of the Task Relation Learning approach (Zhang & Yang
survey, Section 2.4). Key ideas:
1. Stack all task parameters as columns of a matrix W (d features x m
   tasks) and place a matrix-normal prior on W: W ~ MN(0, I, Omega), where
   Omega is an [m, m] task covariance matrix.
2. Jointly optimize:
       min_{W, Omega}  Loss(W) + lambda1 * ||W||_F^2 + lambda2 * tr(W Omega^-1 W^T)
       s.t. Omega >= 0 (PSD), tr(Omega) <= 1
   This is jointly convex in (W, Omega).
3. Given W, the optimal Omega has a closed-form solution:
       Omega = (W^T W)^{1/2} / tr((W^T W)^{1/2})
   (a matrix square root, computed via eigendecomposition, followed by
   trace-normalization to satisfy tr(Omega) <= 1). No gradient descent is
   needed for this step -- this is what makes MTRL "convex" rather than a
   learned/gradient-based relationship matrix.

Applied to wavCSE:
- wavCSE's task heads have very different numbers of classes (ks: 12,
  si: 1251, er: 4), so we cannot follow the paper's original d x m matrix
  literally (naively flattening each head's weight+bias into unequal-length
  rows can't be stacked -- see the docstring of get_task_parameter_matrix
  below for why we instead use a fixed-length, mean-pooled representation
  per task).
- Omega is NOT a learned nn.Parameter here (contrast with wavCSE-PMR, whose
  precision matrix omega_chol IS learned via backprop). MTRL's Omega has a
  closed-form optimum given W, so it is a register_buffer, refreshed
  analytically by update_omega() (called by the trainer on a schedule),
  never touched by the optimizer.
- The regularization term tr(W Omega^-1 W^T) IS differentiated with respect
  to the live classifier-head weights (not a detached snapshot), so it
  actually pulls related tasks' parameters together during the normal
  backward pass -- this is what the alternating scheme in the paper means
  by "solve for W given Omega fixed".

Key difference from PMR (Goncalves et al. 2016): PMR learns a *precision*
matrix via gradient descent, jointly with W, using a single combined loss.
MTRL learns a *covariance* matrix via an alternating closed-form step,
with no gradient ever touching Omega directly -- Omega always exactly
solves the convex sub-problem given the current W.
"""

import logging
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../downstream'))

from pooling.pooling import Pooling
from utils.constant_mapping import LabelKeywordMapping, TaskDatasetMapping
from utils.pooling_id import make_pooling_name


class MultiClassifierOutput:
    def __init__(self, logits=None, prediction=None):
        self.logits = logits
        self.prediction = prediction


class DownstreamMultiTaskModelMTRL(nn.Module):
    """
    wavCSE with Multi-Task Relationship Learning (MTRL).

    Keeps independent task heads but couples them through a task covariance
    matrix Omega ([num_tasks, num_tasks]) that has a closed-form optimum
    given the current task parameters, computed analytically (not learned
    by gradient descent) -- see update_omega().
    """

    def __init__(
        self,
        upstream_model_type: str,
        task_type: str,
        embedding_dim_shared1: int,
        embedding_dim_shared2: int,
        layer_pooling_type: str,
        dropout_prob_shared1: float,
        dropout_prob_shared2: float,
        mtrl_lambda: float = 0.01,     # NEW: strength of the tr(W Omega^-1 W^T) regularizer
        omega_epsilon: float = 1e-4,   # NEW: numerical floor before inverting Omega
        normalize_w: bool = False,     # NEW: unit-normalize each task row of W before forming Omega
        layer_pooling_param: Optional[Union[int, float]] = None
    ):
        super().__init__()

        input_dim = self._input_dim_from_upstream(upstream_model_type)
        output_dim_array = self._output_dims_from_task_type(task_type)
        self.num_tasks = len(output_dim_array)
        self.mtrl_lambda = mtrl_lambda
        self.omega_epsilon = omega_epsilon
        self.normalize_w = normalize_w

        # Shared backbone (unchanged from wavCSE)
        self.projector_layer = nn.Linear(input_dim, embedding_dim_shared1)
        self.dropout_shared1 = nn.Dropout(p=dropout_prob_shared1)

        self.hidden_layer = nn.Linear(embedding_dim_shared1, embedding_dim_shared2)
        self.dropout_shared2 = nn.Dropout(p=dropout_prob_shared2)

        # Task-specific heads (unchanged structure)
        self.classifiers = nn.ModuleList([
            nn.Linear(embedding_dim_shared2, out_dim) for out_dim in output_dim_array
        ])

        # === MTRL MODIFICATION START ===
        # Task covariance matrix Omega: [num_tasks, num_tasks].
        # NOT an nn.Parameter -- it has a closed-form optimum given W (see
        # update_omega()), so it is refreshed analytically, never by the
        # optimizer. Initialized to I/num_tasks so tr(Omega) == 1, matching
        # the paper's tr(Omega) <= 1 constraint at the identity/uninformative
        # prior (no task assumed more related to any other yet).
        init_omega = torch.eye(self.num_tasks) / float(self.num_tasks)
        self.register_buffer('omega', init_omega.clone())
        self.register_buffer('omega_inv', torch.inverse(
            init_omega + omega_epsilon * torch.eye(self.num_tasks)
        ))
        # === MTRL MODIFICATION END ===

        self.layer_pooling_type = layer_pooling_type
        self.pooling = Pooling(layer_pooling_type, pooling_param=layer_pooling_param)

        layer_pool_name = make_pooling_name(layer_pooling_type, layer_pooling_param)
        logging.info(f"Layer pooling type: {layer_pool_name}")
        logging.info(f"MTRL lambda (regularizer strength): {mtrl_lambda}")
        logging.info(f"MTRL omega epsilon: {omega_epsilon}")
        logging.info(f"MTRL normalize W rows: {normalize_w}")

    def _input_dim_from_upstream(self, upstream_model_type: str) -> int:
        upstream_model_variation = upstream_model_type.split("_")[-1].lower()
        if upstream_model_variation == "base":
            return 768
        elif upstream_model_variation == "large":
            return 1024
        else:
            raise ValueError(
                f"Unknown upstream_model_variation='{upstream_model_variation}' "
                f"from upstream_model_type='{upstream_model_type}'."
            )

    def _build_dataset_id_from_task_type(self, task_type: str) -> list:
        task_tokens = task_type.split("_")
        dataset_keys = []
        for t in task_tokens:
            ds = TaskDatasetMapping.get_dataset_key(t)
            if ds is None:
                raise ValueError(f"Invalid task type token: '{t}' in task_type='{task_type}'")
            if ds not in dataset_keys:
                dataset_keys.append(ds)
        return dataset_keys

    def _output_dims_from_task_type(self, task_type: str) -> list:
        dataset_id_array = self._build_dataset_id_from_task_type(task_type)
        output_dim_array = []
        for ds in dataset_id_array:
            label2index, _ = LabelKeywordMapping.get_label_mapping(ds)
            output_dim_array.append(len(label2index))
        return output_dim_array

    def get_task_parameter_matrix(self):
        """
        Collect a fixed-length parameter vector per task, stacked into W.

        wavCSE's task heads have very different output dimensions (e.g. ks:
        12 classes, si: 1251 classes, er: 4 classes), so a raw
        flatten-and-stack of each head's [out_dim, hidden_dim] weight +
        [out_dim] bias produces rows of different lengths that cannot be
        stacked into one matrix, and would in any case let the task with
        the most classes dominate the covariance purely because it has more
        raw parameters, not because it is "more complex" in the MTRL sense.

        Instead, each task is represented by the mean of its head's weight
        rows (mean(weight, dim=0) -> [hidden_dim]) concatenated with the
        mean of its bias (-> scalar): a fixed (hidden_dim + 1)-length
        "direction" for that task in the shared hidden space, independent
        of its class count.

        If normalize_w is True, each row is additionally unit-normalized
        (with a small epsilon floor), so that WW^T becomes the cosine-
        similarity matrix between task directions and the head with the
        largest parameter norm can no longer dominate Omega's diagonal
        (observed with er in the 60-epoch run: its diagonal was 0.837 vs
        0.151/0.011, which through Omega^-1 mostly decoupled er from the
        regularizer entirely).

        Built from the live parameters (no .detach()) so that gradients
        flow back into the classifier heads when this is used inside
        get_mtrl_regularizer_loss() -- see that method's docstring.
        """
        rows = []
        for head in self.classifiers:
            w_mean = head.weight.mean(dim=0)          # [hidden_dim]
            b_mean = head.bias.mean().unsqueeze(0)     # [1]
            rows.append(torch.cat([w_mean, b_mean]))   # [hidden_dim + 1]
        W = torch.stack(rows, dim=0)  # [num_tasks, hidden_dim + 1]

        if self.normalize_w:
            row_norms = W.norm(dim=1, keepdim=True)
            W = W / (row_norms + self.omega_epsilon)

        return W

    @torch.no_grad()
    def update_omega(self):
        """
        Closed-form M-step: recompute Omega given the current task
        parameters W, per Zhang & Yeung (2010):

            Omega = (W^T W)^{1/2} / tr((W^T W)^{1/2})

        Note the paper's W is d x m (features x tasks); here
        get_task_parameter_matrix() returns [num_tasks, d], so the relevant
        Gram matrix over tasks is W @ W.T (not W.T @ W). This is an
        analytic step, not a gradient step, hence @torch.no_grad() -- it
        must not be part of the loss backward pass. Also refreshes the
        cached omega_inv used by get_mtrl_regularizer_loss(), so that
        method doesn't need to invert Omega on every batch.
        """
        W = self.get_task_parameter_matrix()  # [num_tasks, hidden_dim + 1]
        WWt = W @ W.T                          # [num_tasks, num_tasks], symmetric PSD

        eigvals, eigvecs = torch.linalg.eigh(WWt)
        eigvals = eigvals.clamp(min=0.0)
        sqrt_WWt = eigvecs @ torch.diag(eigvals.sqrt()) @ eigvecs.T

        trace = torch.trace(sqrt_WWt)
        if trace.item() <= self.omega_epsilon:
            # Degenerate case (e.g. all task heads still near their random
            # init) -- fall back to the uninformative identity/m prior
            # rather than dividing by ~0.
            new_omega = torch.eye(self.num_tasks, device=W.device) / float(self.num_tasks)
        else:
            new_omega = sqrt_WWt / trace

        self.omega.copy_(new_omega)
        omega_reg = new_omega + self.omega_epsilon * torch.eye(self.num_tasks, device=W.device)
        self.omega_inv.copy_(torch.inverse(omega_reg))

    def get_mtrl_regularizer_loss(self):
        """
        MTRL regularization term: lambda * tr(W Omega^-1 W^T).

        Built from the live (non-detached) task parameter matrix, so this
        term contributes real gradients to the classifier heads during
        backward() -- it is the "solve for W given Omega fixed" half of the
        alternating scheme. Omega/omega_inv themselves are treated as
        constants here (they are buffers, refreshed separately and
        analytically by update_omega(), never by this loss).
        """
        W = self.get_task_parameter_matrix()          # [num_tasks, hidden_dim + 1]
        W_omega_inv = W.T @ self.omega_inv             # [hidden_dim + 1, num_tasks]
        trace_term = torch.trace(W_omega_inv @ W)      # scalar
        return self.mtrl_lambda * trace_term

    def get_omega_matrix(self):
        """Return the current task covariance matrix Omega for analysis."""
        return self.omega.detach().cpu()

    def forward(self, input_seq):
        # Shared backbone (unchanged)
        embedding_shared = self.projector_layer(input_seq)
        embedding_shared = self.pooling.get_vector_after_pooling(embedding_shared, dim=1)
        embedding_shared = self.dropout_shared1(embedding_shared)

        embedding_shared = self.hidden_layer(embedding_shared)
        embedding_shared = self.dropout_shared2(embedding_shared)

        # Task heads (unchanged forward pass -- MTRL only adds a loss term,
        # it does not change how predictions are computed)
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
        return F.softmax(self.pooling.pooling_weights, dim=0)
