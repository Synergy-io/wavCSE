"""MTRL's combine() strategy (Zhang & Yeung 2010/2012) -- Next Step 6.

Migrates the alternating-minimization mechanism (previously duplicated
across `MultiTasksModelTrainerMTRL._process_batch`'s ~85-line override and
`DownstreamMultiTaskModelMTRL`'s Omega buffers/methods) onto mtlkit's
`combine()` seam. Per Eng Review cross-model tension 3 (2026-09-06): Omega
and omega_inv are owned by THIS strategy object, not the model -- the
strategy's epoch-lifecycle hooks (`on_epoch_begin`/`on_epoch_end`) exist
specifically to host this warmup + scheduled-refresh state.

The full loss (unchanged from the original docstring):
    L_total = L_classification + l1_lambda*|W|_1 + l2_lambda*|W|_2^2
              + mtrl_lambda * tr(W Omega^-1 W^T)   [after warmup]

L1/L2 stays a generic trainer-loop concern (mtlkit/trainer.py's module
docstring) applied around this strategy's output by `MultiTasksModelTrainerMTRL`
-- only the classification-loss-combination + Omega-regularizer term lives
here.
"""

import logging
from typing import List, Optional

import torch

from mtlkit.combine import CombineStrategy, HeadParams, register_combine_strategy


@register_combine_strategy("mtrl")
class MTRLCombineStrategy(CombineStrategy):
    def __init__(
        self,
        num_tasks: int,
        mtrl_lambda: float = 0.01,
        omega_epsilon: float = 1e-4,
        normalize_w: bool = False,
        mtrl_warmup_epochs: int = 3,
        omega_update_frequency: int = 1,
    ):
        self.num_tasks = num_tasks
        self.mtrl_lambda = mtrl_lambda
        self.omega_epsilon = omega_epsilon
        self.normalize_w = normalize_w
        self.mtrl_warmup_epochs = mtrl_warmup_epochs
        self.omega_update_frequency = omega_update_frequency

        init_omega = torch.eye(num_tasks) / float(num_tasks)
        self.omega = init_omega.clone()
        self.omega_inv = torch.inverse(init_omega + omega_epsilon * torch.eye(num_tasks))

        self._current_epoch = 0
        self._last_head_params: Optional[List[HeadParams]] = None

        # Full per-epoch Omega history -- same role as the original
        # trainer's self.omega_history (used by _save_omega_history).
        self.omega_history: List[dict] = []

    def _task_parameter_matrix(self, head_params: List[HeadParams]) -> torch.Tensor:
        """Same fixed-length, mean-pooled per-task representation as the
        original `DownstreamMultiTaskModelMTRL.get_task_parameter_matrix`
        (see that method's docstring for why raw flatten-and-stack doesn't
        work), built from LIVE head weight/bias tensors so gradients flow
        back to the classifier heads when used inside a loss."""
        rows = []
        for weight, bias in head_params:
            w_mean = weight.mean(dim=0)          # [hidden_dim]
            b_mean = bias.mean().unsqueeze(0)    # [1]
            rows.append(torch.cat([w_mean, b_mean]))
        W = torch.stack(rows, dim=0)             # [num_tasks, hidden_dim + 1]

        if self.normalize_w:
            row_norms = W.norm(dim=1, keepdim=True)
            W = W / (row_norms + self.omega_epsilon)
        return W

    def __call__(
        self,
        per_task_losses: List[Optional[torch.Tensor]],
        per_task_masks: List[torch.Tensor],
        head_params: List[HeadParams],
    ) -> torch.Tensor:
        self._last_head_params = head_params

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

        if self._current_epoch >= self.mtrl_warmup_epochs:
            if self.omega_inv.device != device:
                self.omega_inv = self.omega_inv.to(device)
            W = self._task_parameter_matrix(head_params)
            W_omega_inv = W.T @ self.omega_inv
            trace_term = torch.trace(W_omega_inv @ W)
            loss_all = loss_all + self.mtrl_lambda * trace_term

        return loss_all

    def on_epoch_begin(self, epoch: int) -> None:
        self._current_epoch = epoch

    def on_epoch_end(self, epoch: int) -> None:
        if epoch >= self.mtrl_warmup_epochs and epoch % self.omega_update_frequency == 0:
            self._update_omega()
            self._record_omega(epoch)

    @torch.no_grad()
    def _update_omega(self) -> None:
        """Closed-form M-step -- see the original
        `DownstreamMultiTaskModelMTRL.update_omega`'s docstring for the
        derivation. Identical math, now over `self._last_head_params`
        (the most recent batch's live head tensors) instead of
        `self.classifiers`."""
        if self._last_head_params is None:
            return

        W = self._task_parameter_matrix(self._last_head_params)
        WWt = W @ W.T

        eigvals, eigvecs = torch.linalg.eigh(WWt)
        eigvals = eigvals.clamp(min=0.0)
        sqrt_WWt = eigvecs @ torch.diag(eigvals.sqrt()) @ eigvecs.T

        trace = torch.trace(sqrt_WWt)
        if trace.item() <= self.omega_epsilon:
            new_omega = torch.eye(self.num_tasks, device=W.device) / float(self.num_tasks)
        else:
            new_omega = sqrt_WWt / trace

        self.omega = new_omega
        omega_reg = new_omega + self.omega_epsilon * torch.eye(self.num_tasks, device=W.device)
        self.omega_inv = torch.inverse(omega_reg)

    def get_omega_matrix(self) -> torch.Tensor:
        """Return the current task covariance matrix Omega for analysis."""
        return self.omega.detach().cpu()

    def _record_omega(self, epoch: int) -> None:
        """Snapshot the full Omega matrix for the end-of-training
        omega_history.json artifact and log its evolution as MLflow
        metrics, exactly as the original trainer's `_record_omega` did."""
        import mlflow

        Omega = self.get_omega_matrix()
        omega_list = Omega.tolist()
        self.omega_history.append({"epoch": epoch, "omega": omega_list})

        if mlflow.active_run() is not None:
            metrics = {}
            for i in range(self.num_tasks):
                for j in range(self.num_tasks):
                    if j < i:
                        continue
                    key = f"omega_{i}_{j}" if i != j else f"omega_diag_{i}"
                    metrics[key] = omega_list[i][j]
            mlflow.log_metrics(metrics, step=epoch)

        off_diag_mask = 1.0 - torch.eye(self.num_tasks)
        mean_off_diag = (Omega * off_diag_mask).abs().sum().item() / max(
            self.num_tasks * (self.num_tasks - 1), 1
        )
        logging.info(
            f"MTRL Omega updated at epoch {epoch}: "
            f"mean|off-diagonal|={mean_off_diag:.6f}, trace={torch.trace(Omega).item():.6f}"
        )
