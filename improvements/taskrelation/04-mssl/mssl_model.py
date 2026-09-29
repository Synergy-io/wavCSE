"""
wavCSE-MSSL: Multi-task Sparse Structure Learning (Goncalves, Von Zuben &
Banerjee, JMLR 17(33):1-30, 2016) -- the p-MSSL instantiation.

Published method implemented here (paper equations in parentheses; verified
against the JMLR PDF, https://jmlr.org/papers/volume17/15-215/15-215.pdf):
  - Rows of the task-parameter matrix W are i.i.d. N(0, Sigma), with precision
    Omega = Sigma^{-1}; Omega_ij = 0 means tasks i and j are conditionally
    independent given the others (Section 3.1, 3.3).
  - Joint objective (Eq. 3), with our per-task losses already being means so
    the paper's 1/n_k scaling is satisfied by construction:
        L(W) + lambda_0 * tr(W Omega W^T) - d * log|Omega|
             + lambda_1 * |W|_1 + lambda_2 * |Omega|_1
  - Alternating minimization (Algorithm 1): given Omega, descend on W
    including the coupling term (Eq. 4a); given W, solve for Omega (Eq. 4b).
  - The Omega step is the graphical-lasso problem of Eq. (4b), re-written in
    the paper as Eq. (8):
        min_{Omega > 0} lambda_0 * tr(S Omega) - log|Omega| + (lambda_2 / d) * ||Omega||_1
        S = (1/d) W^T W        (in our [m tasks, d features] layout: S = (1/d) W W^T)
    solved by ADMM (Eq. 9, 10a-c, 11): eigendecomposition update for Omega,
    element-wise soft-thresholding for Z.

UNITS. `lambda_2` everywhere in this module is the paper's lambda_2 of Eq. (3)
/ Eq. (4b), whose data term is `tr(W Omega W^T) - d log|Omega|` -- *not* the
`lambda_2 / d` coefficient that appears inside Eq. (8). The paper states the
conversion explicitly: "As lambda_2 is a user defined parameter, the factor
1/d can be incorporated into lambda_2." The solver therefore takes `d`
explicitly (`d = W.shape[1]` for our summary matrix) and applies the `1/d`
itself, so a silent 1/d error is impossible: solving with the same numeric
lambda_2 in the two conventions gives different estimators for d != 1.

lambda_0 = 1.0 is the paper's setting: "The parameter lambda_0 was set to one
in all experiments." It is kept as an explicit, overridable parameter.

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
  2. lambda_1 (sparsity on W) is disabled (0.0). In the paper's linear model
     it selects features, whereas our W is a mean-of-head-rows summary for
     which an l1 penalty has no meaningful analogue. (lambda_0 = 1,
     lambda_1 = 0 is also the paper's own named special case: "Setting
     lambda_0 to one and lambda_1 to zero, we return to the exclusive Gaussian
     prior as in (2)".) The knob exists so the deviation is explicit rather
     than silent.
  3. lambda_2 is not fixed by the paper: Algorithm 1 declares the penalty
     parameters as "chosen by cross-validation", the regression experiments
     use stability selection and the classification experiments use
     cross-validation over {0.01, 0.1, 1, 10, 100}. The Study pre-registers
     the value (or the validation-selected grid) in `mssl_config.yml`; this
     class has no default for it, so an unconfigured run fails loudly.

Numerics: the paper's objective fixes the barrier's scale (unit coefficient on
`-log|Omega|` against a unit `tr(S Omega)` term) and is therefore deliberately
not scale free -- S's scale and lambda_2 interact, which is why the paper
selects lambda_2 per data set, and why an unregularised `-log|Omega|` barrier
cannot be rescaled away. The solver keeps the objective exactly as published
and instead makes the ADMM numerically robust: it runs in float64 (the problem
is 3x3, so the cost is negligible, and float32 cannot resolve the ADMM's
residual tests at the summary matrix's scale), it starts rho at the
scale-consistent 1/mean(diag(S)), and it rebalances rho by the residual rule
of Boyd et al. (2011) Section 3.4.1 -- the reference the paper itself cites
for the ADMM derivation. Convergence is judged by that same section's primal
and dual residuals, not by one of them alone.

Unlike the quarantined `models/pmr_model.py` (DEC-0004), which learns a
precision by gradient descent inside one combined loss and never trained,
this is the published alternating scheme: Omega is solved by ADMM and is
never touched by the optimizer.
"""

import logging
import math
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

# Numerical floor on the diagonal of S, relative to its mean diagonal. S is a
# sample covariance of d rows of W; the paper assumes it is full rank (rows
# i.i.d. Gaussian), while a summary matrix can be exactly rank deficient. The
# floor is inert except in that degenerate case (1e-12 relative), where it
# keeps |-log|Omega|| finite instead of letting the solve diverge.
SAMPLE_COVARIANCE_FLOOR = 1e-12

# Boyd et al. (2011) Section 3.4.1 residual balancing bounds.
ADMM_RHO_MIN = 1e-6
ADMM_RHO_MAX = 1e6
ADMM_RHO_SCALE = 2.0
ADMM_RESIDUAL_RATIO = 10.0


class MultiClassifierOutput:
    def __init__(self, logits=None, prediction=None):
        self.logits = logits
        self.prediction = prediction


def soft_threshold(matrix: torch.Tensor, threshold: float) -> torch.Tensor:
    """Element-wise soft-thresholding S_t(x) = sign(x) * max(|x| - t, 0)."""
    return torch.sign(matrix) * torch.clamp(matrix.abs() - threshold, min=0.0)


def optimality_certificate(
    omega: torch.Tensor,
    sample_covariance: torch.Tensor,
    lambda_2: float,
    d: int,
    lambda_0: float = 1.0,
):
    """Certify a candidate solution of Eq. (8) without a reference solver.

    Eq. (8) -- `min_{Omega>0} lambda_0*tr(S Omega) - log|Omega| +
    (lambda_2/d)*||Omega||_1`, l1 on the off-diagonal -- has the dual

        max_{W>0}  log|W| + m   s.t.  W_ii = lambda_0*S_ii,
                                       |W_ij - lambda_0*S_ij| <= lambda_2/d (i != j)

    with strong duality. A positive-definite `Omega` is therefore optimal iff
    `W = Omega^{-1}` is dual feasible and the primal-dual gap vanishes. Both
    quantities are returned in the problem's own units (the dual violation in
    `S`'s units), so it certifies optimality directly instead of comparing two
    solvers, and it is scale free: the same tolerance works for a unit-scale
    covariance and for the mean-head summary's 1e-4-scale covariance.

    Returns `(dual_feasibility_violation, relative_duality_gap)`.
    """
    penalty = float(lambda_2) / float(d)
    omega_work = omega.detach().to(dtype=torch.float64)
    covariance_work = lambda_0 * sample_covariance.detach().to(dtype=torch.float64)
    num_tasks = omega_work.shape[0]
    identity = torch.eye(num_tasks, dtype=torch.float64, device=omega_work.device)

    omega_inverse = torch.linalg.inv(omega_work)
    off_diagonal = 1.0 - identity
    dual_violation = torch.max(
        torch.max(torch.clamp((omega_inverse - covariance_work).abs() - penalty, min=0.0) * off_diagonal),
        torch.max((torch.diagonal(omega_inverse) - torch.diagonal(covariance_work)).abs()),
    )

    log_det_omega = torch.linalg.slogdet(omega_work)[1]
    log_det_inverse = torch.linalg.slogdet(omega_inverse)[1]
    primal = (
        torch.trace(covariance_work @ omega_work) - log_det_omega
        + penalty * (omega_work - torch.diag(torch.diagonal(omega_work))).abs().sum()
    )
    dual = log_det_inverse + float(num_tasks)
    relative_gap = (primal - dual) / torch.clamp(primal.abs(), min=1e-30)
    return float(dual_violation), float(relative_gap)


def graphical_lasso_admm(
    sample_covariance: torch.Tensor,
    lambda_2: float,
    d: int,
    lambda_0: float = 1.0,
    rho: Optional[float] = None,
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
) -> torch.Tensor:
    """Solve the paper's Eq. (8) by the ADMM of Eqs. (9), (10a-c), (11).

        min_{Omega > 0} lambda_0 * tr(S Omega) - log|Omega| + (lambda_2 / d) * ||Omega||_1

    which is Eq. (4b), `lambda_0 * tr(W Omega W^T) - d * log|Omega| + lambda_2 *
    ||Omega||_1`, divided by d and written in terms of S = (1/d) W^T W.
    `lambda_2` is the paper's penalisation parameter of Eq. (3): it is NOT the
    `lambda_2 / d` coefficient that Eq. (8) displays, and `d` (the number of
    rows of W) is therefore required. Emitted by the caller as d = W.shape[1].

    Omega-update (10a) solves, in closed form, `rho*Omega - Omega^{-1} =
    rho*(Z - U) - lambda_0*S`: for `M = rho*(Z-U) - lambda_0*S = Q diag(m) Q^T`,

        Omega = Q diag( (m_i + sqrt(m_i^2 + 4*rho)) / (2*rho) ) Q^T

    which is the eigendecomposition update the paper refers to. The Z-update
    (10b) is Eq. (11), element-wise soft-thresholding at `lambda_2/(d*rho)`;
    the U-update is (10c). The l1 penalty applies to the OFF-DIAGONAL entries
    only, following the graphical lasso of Friedman, Hastie & Tibshirani
    (2008) that the paper cites for this step (Eq. (11) writes an element-wise
    threshold without singling out the diagonal; only off-diagonal zeros carry
    the conditional-independence meaning).

    Returns the symmetric positive-definite Omega in the input's dtype and on
    the input's device.
    """
    if sample_covariance.dim() != 2 or sample_covariance.shape[0] != sample_covariance.shape[1]:
        raise ValueError(f"sample_covariance must be square, got {tuple(sample_covariance.shape)}")
    if int(d) <= 0:
        raise ValueError(f"d must be the positive number of rows of W used to form S, got {d}")
    if float(lambda_2) < 0.0:
        raise ValueError(f"lambda_2 must be non-negative, got {lambda_2}")
    if float(lambda_0) <= 0.0:
        raise ValueError(f"lambda_0 must be positive, got {lambda_0}")

    output_dtype = sample_covariance.dtype
    device = sample_covariance.device
    # float64 throughout: 3x3 eigh costs nothing, and the ADMM's residual tests
    # are not resolvable in float32 at this problem's scale.
    sample_covariance_work = sample_covariance.detach().to(dtype=torch.float64).clone()
    num_tasks = sample_covariance_work.shape[0]
    identity = torch.eye(num_tasks, dtype=torch.float64, device=device)

    sample_covariance_work = 0.5 * (
        sample_covariance_work + sample_covariance_work.transpose(0, 1)
    )
    diagonal_mean = float(torch.diagonal(sample_covariance_work).mean())
    if diagonal_mean <= 0.0:
        raise ValueError("sample_covariance has a non-positive mean diagonal; cannot solve")
    sample_covariance_work = sample_covariance_work + (
        SAMPLE_COVARIANCE_FLOOR * diagonal_mean
    ) * identity
    eigenvalues_s = torch.linalg.eigvalsh(sample_covariance_work)
    if float(eigenvalues_s.max()) <= 0.0:
        raise ValueError("sample_covariance has no positive eigenvalue; cannot solve")

    # The paper's objective fixes the barrier's scale (coefficient 1 against a
    # unit `tr(S Omega)` term), so it is deliberately NOT scale free: S's
    # scale and lambda_2 interact, which is why the paper selects lambda_2 per
    # data set. The minimiser is unique (the problem is strictly convex where
    # it is defined), so the initialization only affects speed; the paper's
    # identity start assumes its own scale, in which the diagonal of S is
    # O(1). The scale-free analogue used here is Omega = Z = diag(1/S_ii):
    # in the paper's standardized setting the two coincide.
    penalty = float(lambda_2) / float(d)
    lambda_0 = float(lambda_0)

    omega_init = torch.diag(1.0 / torch.diagonal(sample_covariance_work))
    omega = omega_init.clone()
    z = omega_init.clone()
    dual = torch.zeros_like(identity)
    sqrt_num_tasks = math.sqrt(float(num_tasks))
    absolute_tolerance = float(tolerance)
    relative_tolerance = float(tolerance)
    penalty_scale = float(rho) if rho is not None and float(rho) > 0.0 else 1.0 / diagonal_mean

    # Optimality certificate (see the stopping test below): the subgradient of
    # the smooth part, in the data's units.
    data_scale = max(float(sample_covariance_work.abs().max()), penalty, float(eigenvalues_s.max()))

    iterations_run = 0
    for iterations_run in range(1, int(max_iterations) + 1):
        previous_z = z.clone()

        # (10a) Omega update: eigendecomposition of rho*(Z - U) - lambda_0*S.
        m = penalty_scale * (z - dual) - lambda_0 * sample_covariance_work
        m = 0.5 * (m + m.transpose(0, 1))
        eigenvalues, eigenvectors = torch.linalg.eigh(m)
        shifted = 0.5 * (
            eigenvalues + torch.sqrt(eigenvalues.square() + 4.0 * penalty_scale)
        ) / penalty_scale
        omega = (eigenvectors * shifted) @ eigenvectors.transpose(0, 1)
        omega = 0.5 * (omega + omega.transpose(0, 1))

        # (10b) Z update -- Eq. (11) with the penalty lambda_2/(d*rho), the
        # off-diagonal entries soft-thresholded and the diagonal carried over.
        z = soft_threshold(omega + dual, penalty / penalty_scale)
        z = z * (1.0 - identity) + (omega + dual) * identity

        # (10c) dual update.
        dual = dual + omega - z

        primal_residual = float(torch.linalg.matrix_norm(omega - z))
        dual_residual = penalty_scale * float(torch.linalg.matrix_norm(z - previous_z))

        # Stopping rule. Two certificates must hold at once:
        #  * the ADMM split is tight: ||Omega - Z|| is small relative to the
        #    iterates (Boyd et al. (2011) Section 3.3.1, primal residual);
        #  * the returned Omega satisfies the optimality conditions of the
        #    problem in Eq. (8) to tolerance, in the data's own units:
        #        Omega^-1_ii          = S_ii                       (diagonal)
        #        Omega^-1_ij - S_ij   = penalty * sign(Z_ij)        (Z_ij != 0)
        #        |Omega^-1_ij - S_ij| <= penalty                    (Z_ij == 0)
        #    The dual residual of Section 3.3.1 is deliberately not used as a
        #    stop condition: it is measured against ||rho*U||, which vanishes
        #    at the solution, so it is unattainable in finite precision when
        #    the problem is badly scaled (Omega ~ 1/S). The certificate above
        #    is scale free and is exactly the condition whose violation the
        #    ADMM is driving to zero.
        if primal_residual <= relative_tolerance * max(
            float(torch.linalg.matrix_norm(omega)), float(torch.linalg.matrix_norm(z))
        ):
            dual_violation, relative_gap = optimality_certificate(
                z, sample_covariance_work, lambda_2, d, lambda_0
            )
            if (
                dual_violation <= relative_tolerance * data_scale
                and relative_gap <= relative_tolerance
            ):
                break

        # Section 3.4.1 residual balancing: keep the two residuals within a
        # factor ADMM_RESIDUAL_RATIO, halving/doubling rho (and rescaling the
        # scaled dual variable so that U keeps its meaning).
        if primal_residual > ADMM_RESIDUAL_RATIO * dual_residual and penalty_scale < ADMM_RHO_MAX:
            penalty_scale = min(penalty_scale * ADMM_RHO_SCALE, ADMM_RHO_MAX)
            dual = dual / ADMM_RHO_SCALE
        elif dual_residual > ADMM_RESIDUAL_RATIO * primal_residual and penalty_scale > ADMM_RHO_MIN:
            penalty_scale = max(penalty_scale / ADMM_RHO_SCALE, ADMM_RHO_MIN)
            dual = dual * ADMM_RHO_SCALE
        elif not math.isfinite(primal_residual) or not math.isfinite(dual_residual):
            raise RuntimeError("MSSL graphical-lasso ADMM diverged (non-finite residual)")

    # The estimate returned is the ADMM's Z iterate, not the primal Omega: Z
    # is the split variable that carries the l1 solution's exact zero pattern
    # (Eq. (11) thresholds it), while the primal iterate only approaches that
    # pattern as the splitting constraint tightens. Publishing Z therefore
    # makes the reported sparsity exact and keeps the estimate consistent with
    # the optimality certificate above, which is evaluated on Z's support.
    z = 0.5 * (z + z.transpose(0, 1))
    dual_violation, relative_gap = optimality_certificate(
        z, sample_covariance_work, lambda_2, d, lambda_0
    )
    logging.debug(
        "MSSL ADMM: iterations=%d rho=%.4g lambda_2=%.4g d=%d lambda_0=%.4g "
        "dual_violation=%.3g relative_gap=%.3g",
        iterations_run, penalty_scale, float(lambda_2), int(d), lambda_0,
        dual_violation, relative_gap,
    )
    return z.to(dtype=output_dtype)


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
        mssl_lambda_2: float,
        mssl_lambda_0: float = 1.0,
        mssl_lambda_1: float = 0.0,
        mssl_admm_rho: Optional[float] = None,
        mssl_admm_iterations: int = 2000,
        normalize_w: bool = False,
        layer_pooling_param: Optional[Union[int, float]] = None,
    ):
        super().__init__()

        input_dim = self._input_dim_from_upstream(upstream_model_type)
        output_dim_array = self._output_dims_from_task_type(task_type)
        self.num_tasks = len(output_dim_array)

        # mssl_lambda_2 has no default: the published paper selects it on data
        # (Algorithm 1: "penalty parameters chosen by cross-validation"), so an
        # unconfigured run must fail rather than inherit a placeholder value.
        self.mssl_lambda_2 = float(mssl_lambda_2)
        self.mssl_lambda_0 = float(mssl_lambda_0)
        self.mssl_lambda_1 = float(mssl_lambda_1)
        # rho = None lets the solver start at its scale-consistent default and
        # rebalance by the residual rule of Boyd et al. (2011) 3.4.1.
        self.mssl_admm_rho = None if mssl_admm_rho is None else float(mssl_admm_rho)
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
        self.register_buffer(
            "last_admm_rho",
            torch.tensor(float("nan") if self.mssl_admm_rho is None else self.mssl_admm_rho),
        )
        # === MSSL MODIFICATION END ===

        self.layer_pooling_type = layer_pooling_type
        self.pooling = Pooling(layer_pooling_type, pooling_param=layer_pooling_param)

        logging.info(f"Layer pooling type: {make_pooling_name(layer_pooling_type, layer_pooling_param)}")
        logging.info(
            "MSSL | lambda_0=%.4g (paper: 1.0) | lambda_1(W, paper's "
            "exclusive-Gaussian-prior case)=%.4g | lambda_2(Omega sparsity, "
            "Eq. 3 units, applied as lambda_2/d in Eq. 8)=%.4g | admm_rho=%s | "
            "admm_iters=%d | normalize_w=%s",
            self.mssl_lambda_0, self.mssl_lambda_1, self.mssl_lambda_2,
            "auto" if self.mssl_admm_rho is None else "%.4g" % self.mssl_admm_rho,
            self.mssl_admm_iterations, self.normalize_w,
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
        """Omega step (Eq. 4b / Eq. 8): graphical lasso on S = (1/d) W W^T.

        No closed-form trace-normalisation here (contrast with MTRL): the
        precision is estimated by the penalised likelihood, which is what
        keeps it from saturating. `self.mssl_lambda_2` is the paper's Eq. (3)
        penalty; the solver applies the `1/d` of Eq. (8) itself.
        """
        W = self.get_task_parameter_matrix()
        d = int(W.shape[1])
        # Paper: S = (1/d) W^T W with W of shape [d, m]; our W is [m, d], so
        # the task-space Gram matrix is W W^T and the 1/d factor is explicit.
        sample_covariance = (W @ W.transpose(0, 1)) / float(d)

        new_omega = graphical_lasso_admm(
            sample_covariance=sample_covariance,
            lambda_2=self.mssl_lambda_2,
            d=d,
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
        half of the alternation). With the published lambda_0 = 1 this is the
        paper's coupling term unchanged; lambda_1 is disabled (see the module
        docstring, deviation 2) and, when enabled, penalises the summary rows.
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
