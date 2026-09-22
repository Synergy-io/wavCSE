"""
wavCSE-MSSL: Multi-task Sparse Structure Learning (Goncalves, Von Zuben &
Banerjee, JMLR 17(33):1-30, 2016) -- the p-MSSL instantiation.

Published method implemented here (paper equations in parentheses):
  - Rows of the task-parameter matrix W are i.i.d. N(0, Sigma), with precision
    Omega = Sigma^{-1}; Omega_ij = 0 means tasks i and j are conditionally
    independent given the others (Section 3.1, 3.3).
  - Objective (Eq. 3), with our per-task losses already being means so the
    paper's 1/n_k scaling is satisfied by construction:
        L(W) + lambda_0 * tr(W Omega W^T) - d * log|Omega| + lambda_1*|W|_1 + lambda_2*|Omega|_1
  - Alternating minimization (Algorithm 1): given Omega, descend on W
    including the coupling term; given W, solve for Omega.
  - The Omega step is the graphical-lasso problem (Eq. 8)
        min_{Omega > 0}  lambda_0 * tr(S Omega) - log|Omega| + (lambda_2/d)||Omega||_1,
        S = (1/d) W^T W  (in our [tasks, features] convention: S = (1/d) W W^T),
    solved by ADMM (Eq. 9, 10a-c, 11) with an eigendecomposition update for
    Omega and element-wise soft-thresholding for Z.

Why this is the arm worth running: classical MTRL in this repository forms
Omega by the closed-form (W^T W)^{1/2} trace-normalisation, which saturates
near +/-1/3 and loses pair-specific information (findings F5/F7/F9). MSSL's
Omega is a *precision* estimated by a proper penalised-likelihood problem:
the -log|Omega| barrier plus l1 shrinkage is a bounded, regularised
parameterisation that does not saturate by construction. The paper states
this contrast explicitly ("we ... learn the inverse of the covariance matrix
directly, which tends to be more stable than computing covariance and then
inverting it").

Deviations from the paper, recorded in studies/TR-0007/PLAN.md before any run:
  1. The paper's per-task parameter vector (a whole linear model) does not
     exist in a shared-trunk model with heterogeneous heads (12 / 1251 / 4
     classes). Following this repository's existing convention for the MTRL
     control, each task contributes a fixed-length summary: the mean of its
     classifier weight rows concatenated with its mean bias. This is the SAME
     adapter the in-category control uses, so the comparison isolates the
     estimator. It is a reduction, not a faithful reproduction of the paper's
     setting, and results must be reported as such.
  2. lambda_1 (sparsity on W) is disabled by default: in the paper's linear
     model it selects features, whereas our W is a mean-of-head-rows summary
     for which an l1 penalty has no meaningful analogue. lambda_2 (sparsity on
     Omega) is the mechanism and is retained. The knob exists so the deviation
     is explicit rather than silent.

Unlike the quarantined `models/pmr_model.py` (DEC-0004), which learns a
precision by gradient descent inside one combined loss and never trained,
this is the published alternating scheme: Omega is solved by ADMM and is
never touched by the optimizer.
"""

import logging
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../downstream"))

from pooling.pooling import Pooling
from utils.constant_mapping import LabelKeywordMapping, TaskDatasetMapping
from utils.pooling_id import make_pooling_name

ADMM_OMEGA_RIDGE = 1e-8


class MultiClassifierOutput:
    def __init__(self, logits=None, prediction=None):
        self.logits = logits
        self.prediction = prediction


def soft_threshold(matrix: torch.Tensor, threshold: float) -> torch.Tensor:
    """Element-wise soft-thresholding S_t(x) = sign(x) * max(|x| - t, 0)."""
    return torch.sign(matrix) * torch.clamp(matrix.abs() - threshold, min=0.0)


def graphical_lasso_admm(
    sample_covariance: torch.Tensor,
    lambda_2: float,
    lambda_0: float = 1.0,
    rho: float = 1.0,
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
) -> torch.Tensor:
    """Solve the paper's Eq. (8) by ADMM (Eq. 9, 10a-c, 11).

        min_{Omega > 0} lambda_0 * tr(S Omega) - log|Omega| + lambda_2 * ||Omega||_1

    The 1/d factor the paper folds into lambda_2 (Eq. 8) is applied by the
    caller, which passes S already divided by d.

    Omega-update (10a) solves rho*Omega - Omega^{-1} = rho*(Z - U) - lambda_0*S
    in closed form: for M = rho*(Z-U) - lambda_0*S = Q diag(m) Q^T,

        Omega = Q diag( (m_i + sqrt(m_i^2 + 4*rho)) / (2*rho) ) Q^T

    which is the eigendecomposition update the paper refers to. Z-update (10b)
    is Eq. (11), element-wise soft-thresholding at lambda_2/rho. U-update is
    (10c). Returns the symmetric positive-definite Omega.
    """
    num_tasks = sample_covariance.shape[0]
    identity = torch.eye(num_tasks, dtype=sample_covariance.dtype,
                         device=sample_covariance.device)
    omega = identity.clone()
    z = identity.clone()
    dual = torch.zeros_like(identity)

    # Boyd et al. (2011) 3.3.1 stopping rule: BOTH the primal residual
    # r = Omega - Z and the dual residual s = rho*(Z - Z_prev) must be small
    # relative to the current iterates. A Z-only criterion is unsafe for this
    # problem: with a large lambda_2 the Z iterate collapses to the zero
    # matrix, which would declare convergence while the primal iterate is
    # still far from the solution (verified against sklearn's graphical_lasso
    # before this rule was adopted).
    sqrt_num_tasks = float(num_tasks) ** 0.5
    absolute_tolerance = 1e-8
    relative_tolerance = tolerance
    for _ in range(max_iterations):
        previous_z = z.clone()

        # (10a) Omega update via eigendecomposition of rho*(Z - U) - lambda_0*S.
        m = rho * (z - dual) - lambda_0 * sample_covariance
        m = 0.5 * (m + m.transpose(0, 1))
        eigenvalues, eigenvectors = torch.linalg.eigh(m)
        shifted = 0.5 * (
            eigenvalues + torch.sqrt(eigenvalues.square() + 4.0 * rho)
        ) / rho
        omega = (eigenvectors * shifted) @ eigenvectors.transpose(0, 1)
        omega = 0.5 * (omega + omega.transpose(0, 1))

        # (10b) Z update -- Eq. (11). The l1 penalty applies to the
        # OFF-DIAGONAL entries only, following the graphical lasso of
        # Friedman, Hastie & Tibshirani (2008) that the paper cites: only
        # off-diagonal zeros carry the conditional-independence meaning, so
        # penalising the diagonal would shrink precisions in a way that has no
        # graph interpretation. The paper's Eq. (8) writes ||Omega||_1 without
        # specifying; this choice also allows validation against sklearn's
        # independent graphical_lasso implementation.
        z = soft_threshold(omega + dual, lambda_2 / rho)
        # Restore the unpenalised diagonal (fill_diagonal_ only accepts a
        # scalar, so the diagonal is blended back with an identity mask).
        z = z * (1.0 - identity) + (omega + dual) * identity

        # (10c) dual update.
        dual = dual + omega - z

        primal_residual = torch.linalg.matrix_norm(omega - z)
        dual_residual = rho * torch.linalg.matrix_norm(z - previous_z)
        primal_limit = absolute_tolerance * sqrt_num_tasks + relative_tolerance * max(
            torch.linalg.matrix_norm(omega), torch.linalg.matrix_norm(z)
        )
        dual_limit = (
            absolute_tolerance * sqrt_num_tasks
            + relative_tolerance * rho * torch.linalg.matrix_norm(dual)
        )
        if primal_residual < primal_limit and dual_residual < dual_limit:
            break

    # Numerical hygiene: symmetrise and floor the diagonal so |Omega| > 0.
    omega = 0.5 * (omega + omega.transpose(0, 1))
    reg = 0.5 * (omega + omega.transpose(0, 1))
    return reg + ADMM_OMEGA_RIDGE * identity


class DownstreamMultiTaskModelMSSL(nn.Module):
    """wavCSE with p-MSSL: sparse task precision learned by ADMM graphical lasso."""

    def __init__(
        self,
        upstream_model_type: str,
        task_type: str,
        embedding_dim_shared1: int,
        embedding_dim_shared2: int,
        layer_pooling_type: str,
        dropout_prob_shared1: float,
        dropout_prob_shared2: float,
        mssl_lambda_2: float = 0.05,
        mssl_lambda_0: float = 1.0,
        mssl_lambda_1: float = 0.0,
        mssl_admm_rho: float = 1.0,
        mssl_admm_iterations: int = 1000,
        normalize_w: bool = False,
        layer_pooling_param: Optional[Union[int, float]] = None,
    ):
        super().__init__()

        input_dim = self._input_dim_from_upstream(upstream_model_type)
        output_dim_array = self._output_dims_from_task_type(task_type)
        self.num_tasks = len(output_dim_array)

        self.mssl_lambda_2 = float(mssl_lambda_2)
        self.mssl_lambda_0 = float(mssl_lambda_0)
        self.mssl_lambda_1 = float(mssl_lambda_1)
        self.mssl_admm_rho = float(mssl_admm_rho)
        self.mssl_admm_iterations = int(mssl_admm_iterations)
        self.normalize_w = normalize_w
        self.omega_epsilon = 1e-4

        # Shared backbone (unchanged from wavCSE).
        self.projector_layer = nn.Linear(input_dim, embedding_dim_shared1)
        self.dropout_shared1 = nn.Dropout(p=dropout_prob_shared1)
        self.hidden_layer = nn.Linear(embedding_dim_shared1, embedding_dim_shared2)
        self.dropout_shared2 = nn.Dropout(p=dropout_prob_shared2)

        # Task heads (unchanged structure).
        self.classifiers = nn.ModuleList([
            nn.Linear(embedding_dim_shared2, out_dim) for out_dim in output_dim_array
        ])

        # === MSSL MODIFICATION START ===
        # Omega is a PRECISION matrix (Sigma^{-1}), solved analytically by ADMM
        # graphical lasso at each Omega step. It is a buffer, never an
        # nn.Parameter, so the optimizer never touches it.
        identity = torch.eye(self.num_tasks)
        self.register_buffer("omega", identity.clone())
        self.register_buffer("last_admm_rho", torch.tensor(self.mssl_admm_rho))
        # === MSSL MODIFICATION END ===

        self.layer_pooling_type = layer_pooling_type
        self.pooling = Pooling(layer_pooling_type, pooling_param=layer_pooling_param)

        logging.info(f"Layer pooling type: {make_pooling_name(layer_pooling_type, layer_pooling_param)}")
        logging.info(
            "MSSL | lambda_0=%.4g | lambda_1(W, disabled by default)=%.4g | "
            "lambda_2(Omega sparsity)=%.4g | admm_rho=%.4g | admm_iters=%d | normalize_w=%s",
            self.mssl_lambda_0, self.mssl_lambda_1, self.mssl_lambda_2,
            self.mssl_admm_rho, self.mssl_admm_iterations, self.normalize_w,
        )

    def _input_dim_from_upstream(self, upstream_model_type: str) -> int:
        variation = upstream_model_type.split("_")[-1].lower()
        if variation == "base":
            return 768
        if variation == "large":
            return 1024
        raise ValueError(
            f"Unknown upstream_model_variation='{variation}' "
            f"from upstream_model_type='{upstream_model_type}'."
        )

    def _build_dataset_id_from_task_type(self, task_type: str) -> list:
        dataset_keys = []
        for token in task_type.split("_"):
            key = TaskDatasetMapping.get_dataset_key(token)
            if key is None:
                raise ValueError(f"Invalid task type token: '{token}' in task_type='{task_type}'")
            if key not in dataset_keys:
                dataset_keys.append(key)
        return dataset_keys

    def _output_dims_from_task_type(self, task_type: str) -> list:
        dims = []
        for key in self._build_dataset_id_from_task_type(task_type):
            label2index, _ = LabelKeywordMapping.get_label_mapping(key)
            dims.append(len(label2index))
        return dims

    def get_task_parameter_matrix(self) -> torch.Tensor:
        """Fixed-length per-task summary W [num_tasks, hidden_dim + 1].

        Identical adapter to the in-category MTRL control (mean classifier
        weight row + mean bias), so the estimator is the only difference
        between the two arms. Built from live parameters so the coupling term
        produces real gradients on the heads.
        """
        rows = []
        for head in self.classifiers:
            w_mean = head.weight.mean(dim=0)        # [hidden_dim]
            b_mean = head.bias.mean().unsqueeze(0)  # [1]
            rows.append(torch.cat([w_mean, b_mean]))
        W = torch.stack(rows, dim=0)                # [num_tasks, hidden + 1]

        if self.normalize_w:
            W = W / (W.norm(dim=1, keepdim=True) + self.omega_epsilon)
        return W

    @torch.no_grad()
    def update_omega(self) -> torch.Tensor:
        """Omega step: graphical lasso on S = (1/d) W W^T (Eq. 8, 10a-c).

        No closed-form trace-normalisation here (contrast with MTRL): the
        precision is estimated by the penalised likelihood, which is what
        keeps it from saturating.
        """
        W = self.get_task_parameter_matrix()
        num_rows = W.shape[1]
        # Paper: S = (1/d) W^T W with W of shape [d, m]; our W is [m, d], so
        # the task-space Gram matrix is W W^T and the 1/d factor is explicit.
        sample_covariance = (W @ W.transpose(0, 1)) / float(num_rows)

        new_omega = graphical_lasso_admm(
            sample_covariance=sample_covariance,
            lambda_2=self.mssl_lambda_2,
            lambda_0=self.mssl_lambda_0,
            rho=self.mssl_admm_rho,
            max_iterations=self.mssl_admm_iterations,
        )
        self.omega.copy_(new_omega)
        return new_omega

    def get_relation_loss(self) -> torch.Tensor:
        """lambda_0 * tr(W Omega W^T) + optional lambda_1 * |W|_1  (Eq. 3).

        Uses the live W so gradients reach the classifier heads; Omega is a
        buffer fixed by the last Omega step (the "solve for W given Omega"
        half of the alternation).
        """
        W = self.get_task_parameter_matrix()
        # The paper's W is [d features, m tasks] and the prior is on its rows,
        # so the coupling is tr(W Omega W^T) = tr(W^T Omega W) in our
        # [m tasks, d features] convention. Omega is m x m, so the trace must
        # be taken over the feature dimension: W^T @ Omega @ W is [d, d].
        loss = self.mssl_lambda_0 * torch.trace(
            W.transpose(0, 1) @ self.omega @ W
        )
        if self.mssl_lambda_1 > 0.0:
            loss = loss + self.mssl_lambda_1 * W.abs().sum()
        return loss

    def get_omega_matrix(self) -> torch.Tensor:
        """Current precision matrix Omega (detached, CPU) for analysis."""
        return self.omega.detach().cpu()

    def get_partial_correlations(self) -> torch.Tensor:
        """Partial correlations -Omega_ij / sqrt(Omega_ii Omega_jj) (paper 3.1)."""
        omega = self.omega.detach()
        diag = torch.diagonal(omega).clamp(min=self.omega_epsilon).sqrt()
        denom = diag.unsqueeze(0) * diag.unsqueeze(1)
        partial = -omega / denom
        partial.fill_diagonal_(0.0)
        return partial.cpu()

    def forward(self, input_seq):
        embedding_shared = self.projector_layer(input_seq)
        embedding_shared = self.pooling.get_vector_after_pooling(embedding_shared, dim=1)
        embedding_shared = self.dropout_shared1(embedding_shared)
        embedding_shared = self.hidden_layer(embedding_shared)
        embedding_shared = self.dropout_shared2(embedding_shared)

        logits_list, pred_list = [], []
        for head in self.classifiers:
            logits = head(embedding_shared)
            logits_list.append(logits)
            pred_list.append(torch.argmax(logits, dim=1))

        return MultiClassifierOutput(
            logits=tuple(logits_list),
            prediction=tuple(pred_list),
        )

    def get_all_embeddings(self, input_seq):
        embedding_shared = self.projector_layer(input_seq)
        embedding_shared = self.pooling.get_vector_after_pooling(embedding_shared, dim=1)
        return self.hidden_layer(embedding_shared)

    def get_pooling_weights(self):
        if self.layer_pooling_type != "weighted":
            return None
        return F.softmax(self.pooling.pooling_weights, dim=0)
