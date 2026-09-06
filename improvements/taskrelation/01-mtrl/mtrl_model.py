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
  literally -- see `mtrl_combine.MTRLCombineStrategy._task_parameter_matrix`'s
  docstring for why a fixed-length, mean-pooled representation per task is
  used instead.
- The regularization term tr(W Omega^-1 W^T) IS differentiated with respect
  to the live classifier-head weights (not a detached snapshot), so it
  actually pulls related tasks' parameters together during the normal
  backward pass -- this is what the alternating scheme in the paper means
  by "solve for W given Omega fixed".

Migrated onto mtlkit (Next Step 6, Eng Review decision D1): the shared
backbone + heads now delegate to `mtlkit.heads`/`mtlkit.tasks` (drops the
`sys.path.insert` hack this file used to reach `downstream/`), and Omega/
Omega^-1 and the warmup/refresh schedule moved to `mtrl_combine.py`'s
`MTRLCombineStrategy` -- this model is now a plain multi-task model with no
MTRL-specific state of its own. See `mtrl_trainer.py` for how the strategy
plugs into `mtlkit.trainer.process_batch`.

Key difference from PMR (Goncalves et al. 2016): PMR learns a *precision*
matrix via gradient descent, jointly with W, using a single combined loss.
MTRL learns a *covariance* matrix via an alternating closed-form step,
with no gradient ever touching Omega directly -- Omega always exactly
solves the convex sub-problem given the current W.
"""

import logging
from typing import Optional, Union

import mtlkit.heads as mtlkit_heads
from mtlkit.heads import MultiClassifierOutput  # noqa: F401 -- re-export, same class the inherited forward() constructs
from mtlkit.tasks import TASK_REGISTRY
from utils.pooling_id import make_pooling_name


def _tasks_from_task_type(task_type: str):
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


class DownstreamMultiTaskModelMTRL(mtlkit_heads.MultiTaskModel):
    """
    wavCSE with Multi-Task Relationship Learning (MTRL).

    Plain shared-trunk + per-task-heads model -- MTRL's task-coupling
    mechanism (the Omega covariance matrix and its regularizer) now lives
    entirely in `mtrl_combine.MTRLCombineStrategy`, not here. This class
    keeps its own flat structure via `mtlkit.heads.MultiTaskModel`
    (`self.trunk`, `self.heads`) since -- unlike `DownstreamMultiTaskModel`
    in `downstream/model/downstream_model.py` -- nothing in
    `improvements/` subclasses this model expecting a legacy flat layout.
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
        layer_pooling_param: Optional[Union[int, float]] = None,
    ):
        tasks = _tasks_from_task_type(task_type)
        trunk = mtlkit_heads.build_trunk(
            upstream_model_type=upstream_model_type,
            embedding_dim_shared1=embedding_dim_shared1,
            embedding_dim_shared2=embedding_dim_shared2,
            layer_pooling_type=layer_pooling_type,
            dropout_prob_shared1=dropout_prob_shared1,
            dropout_prob_shared2=dropout_prob_shared2,
            layer_pooling_param=layer_pooling_param,
        )
        heads = mtlkit_heads.build_heads(tasks, in_features=embedding_dim_shared2)
        super().__init__(trunk, heads)

        self.num_tasks = len(tasks)

        layer_pool_name = make_pooling_name(layer_pooling_type, layer_pooling_param)
        logging.info(f"Layer pooling type: {layer_pool_name}")
