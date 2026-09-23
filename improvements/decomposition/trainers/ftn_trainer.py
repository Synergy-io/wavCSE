"""FTN trainer with optional safeguards and shared-FC2 gradient diagnostics."""

import math
import os
import sys

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau


_DOWNSTREAM_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "downstream")
)
if _DOWNSTREAM_DIR not in sys.path:
    sys.path.insert(0, _DOWNSTREAM_DIR)

from trainer.trainer_model import MultiTasksModelTrainer


class MultiTasksModelTrainerFTN(MultiTasksModelTrainer):
    """Standard loop with optional GradNorm and shared-FC2 diagnostics."""

    GRADIENT_PAIRS = (("ks", "si"), ("ks", "er"), ("si", "er"))

    def __init__(self, *args, **kwargs):
        diagnostics_cfg = kwargs.pop("diagnostics_cfg", None) or {}
        gradnorm_cfg = kwargs.pop("gradnorm_cfg", None) or {}
        self.shared_gradient_analysis = bool(
            diagnostics_cfg.get("shared_gradient_analysis", True)
        )
        self.shared_gradient_every_n_batches = int(
            diagnostics_cfg.get("shared_gradient_every_n_batches", 10)
        )
        if self.shared_gradient_every_n_batches < 1:
            raise ValueError("shared_gradient_every_n_batches must be >= 1")
        super().__init__(*args, **kwargs)
        self.task_weighting = self.training_cfg.get("task_weighting", "equal")
        if self.task_weighting not in {"equal", "gradnorm"}:
            raise ValueError("task_weighting must be 'equal' or 'gradnorm'")
        self._gradnorm_collect = False
        if self.task_weighting == "gradnorm":
            self.gradnorm_alpha = float(gradnorm_cfg.get("alpha", 1.5))
            self.gradnorm_weight_lr = float(gradnorm_cfg.get("weight_lr", 0.025))
            self.gradnorm_min_weight = float(gradnorm_cfg.get("min_weight", 1e-6))
            if (not math.isfinite(self.gradnorm_alpha) or self.gradnorm_alpha < 0
                    or not math.isfinite(self.gradnorm_weight_lr) or self.gradnorm_weight_lr <= 0):
                raise ValueError("GradNorm alpha must be >= 0 and weight_lr must be > 0")
            if not math.isfinite(self.gradnorm_min_weight) or not 0 < self.gradnorm_min_weight < 1:
                raise ValueError("GradNorm min_weight must be between 0 and 1")
            self.task_weights = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.ones((), device=self.device)) for _ in self.task_array]
            )
            self.weight_optimizer = torch.optim.Adam(
                self.task_weights.parameters(), lr=self.gradnorm_weight_lr
            )
            self.initial_task_losses = [None] * self.num_tasks
            self._reset_gradnorm_epoch_stats()
        self._reset_shared_gradient_stats()
        self._training_batch_index = 0
        self._collect_shared_gradient = False
        self.max_grad_norm = float(self.training_cfg.get("max_grad_norm", 0.0))
        self.grad_norm_sum = 0.0
        self.grad_norm_max = 0.0
        self.grad_norm_steps = 0

        if self.max_grad_norm > 0.0:
            self._grad_clip_handle = self.optimizer.register_step_pre_hook(
                self._clip_gradients
            )

        scheduler_min_lr = float(self.training_cfg.get("scheduler_min_lr", 0.0))
        if scheduler_min_lr > 0.0:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=int(self.training_cfg.get("scheduler_patience", 1)),
                factor=float(self.training_cfg.get("scheduler_factor", 0.5)),
                min_lr=scheduler_min_lr,
            )

    def _clip_gradients(self, _optimizer, _args, _kwargs):
        total_norm = clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        total_norm_value = float(total_norm.detach().item())
        self.grad_norm_sum += total_norm_value
        self.grad_norm_max = max(self.grad_norm_max, total_norm_value)
        self.grad_norm_steps += 1

    def _reset_shared_gradient_stats(self):
        self._gradient_norm_sums = {task: 0.0 for task in self.task_array}
        self._gradient_norm_max = {task: 0.0 for task in self.task_array}
        self._gradient_norm_counts = {task: 0 for task in self.task_array}
        self._gradient_cosine_sums = {pair: 0.0 for pair in self.GRADIENT_PAIRS}
        self._gradient_conflict_counts = {pair: 0 for pair in self.GRADIENT_PAIRS}
        self._gradient_pair_counts = {pair: 0 for pair in self.GRADIENT_PAIRS}

    def _process_data_loader(self, data_loader, train_mode: bool):
        if train_mode:
            self._training_batch_index = 0
            self._reset_shared_gradient_stats()
            if self.task_weighting == "gradnorm":
                self._reset_gradnorm_epoch_stats()
        return super()._process_data_loader(data_loader, train_mode)

    def _calculate_total_loss(self, loss_task, loss_all, loss_weight):
        """Inspect each unweighted masked task loss before the parent's backward."""
        if self._gradnorm_collect:
            index = self._gradnorm_task_index
            task = self.task_array[index]
            self._gradnorm_task_index += 1
            if loss_task is not None:
                self._batch_task_losses[index] = loss_task
                if self.initial_task_losses[index] is None:
                    initial_loss = float(loss_task.detach().item())
                    if math.isfinite(initial_loss):
                        self.initial_task_losses[index] = max(initial_loss, 1e-12)
                gradient = torch.autograd.grad(
                    loss_task,
                    self.model.hidden_layer.weight,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if gradient is not None:
                    gradient = gradient.detach()
                    if gradient.numel() and bool(torch.isfinite(gradient).all()):
                        norm = float(torch.linalg.vector_norm(gradient).item())
                        if math.isfinite(norm):
                            self._batch_raw_gradient_norms[index] = norm
                            if self._collect_shared_gradient:
                                self._record_shared_gradient(task, gradient, norm)
            if self._gradnorm_task_index == self.num_tasks:
                if self._collect_shared_gradient:
                    self._record_gradient_pairs()
                    self._batch_shared_gradients.clear()
                self._update_gradnorm_weights()
                loss_all = loss_all + loss_weight * sum(
                    self.task_weights[task_index].detach() * task_loss
                    for task_index, task_loss in self._batch_task_losses.items()
                )
            if loss_task is None:
                return torch.tensor(0.0, device=self.device), loss_all, 0
            return loss_task, loss_all, 1

        if self._collect_shared_gradient:
            task = self.task_array[self._diagnostic_task_index]
            self._diagnostic_task_index += 1
            if loss_task is not None:
                gradient = torch.autograd.grad(
                    loss_task,
                    self.model.hidden_layer.weight,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if gradient is not None:
                    gradient = gradient.detach()
                    if gradient.numel() and bool(torch.isfinite(gradient).all()):
                        norm = float(torch.linalg.vector_norm(gradient).item())
                        if math.isfinite(norm):
                            self._record_shared_gradient(task, gradient, norm)
            if self._diagnostic_task_index == self.num_tasks:
                self._record_gradient_pairs()
                self._batch_shared_gradients.clear()
        return super()._calculate_total_loss(loss_task, loss_all, loss_weight)

    def _record_shared_gradient(self, task, gradient, norm):
        self._gradient_norm_sums[task] += norm
        self._gradient_norm_max[task] = max(self._gradient_norm_max[task], norm)
        self._gradient_norm_counts[task] += 1
        if norm > 1e-12:
            self._batch_shared_gradients[task] = gradient

    def _reset_gradnorm_epoch_stats(self):
        self._gradnorm_metric_sums = {}
        self._gradnorm_metric_counts = {}

    def _add_gradnorm_metric(self, name, value):
        if math.isfinite(value):
            self._gradnorm_metric_sums[name] = self._gradnorm_metric_sums.get(name, 0.0) + value
            self._gradnorm_metric_counts[name] = self._gradnorm_metric_counts.get(name, 0) + 1

    def _update_gradnorm_weights(self):
        eligible = [
            index for index in self._batch_raw_gradient_norms
            if index in self._batch_task_losses and self.initial_task_losses[index] is not None
        ]
        if len(eligible) < 2:
            return
        ratios = torch.stack([
            self._batch_task_losses[index].detach()
            / self.initial_task_losses[index]
            for index in eligible
        ])
        if not bool(torch.isfinite(ratios).all()):
            return
        relative_rates = ratios / ratios.mean().clamp_min(1e-12)
        weighted_norms = torch.stack([
            self.task_weights[index] * self._batch_raw_gradient_norms[index]
            for index in eligible
        ])
        targets = (
            weighted_norms.mean().detach()
            * relative_rates.pow(self.gradnorm_alpha)
        ).detach()
        balancing_loss = torch.abs(weighted_norms - targets).sum()
        if not bool(torch.isfinite(balancing_loss)):
            return
        self.weight_optimizer.zero_grad(set_to_none=True)
        balancing_loss.backward()
        self.weight_optimizer.step()

        # Inactive parameters receive no optimizer gradient and stay unchanged.
        # Normalize only the active weights to keep the total at num_tasks.
        with torch.no_grad():
            inactive_sum = sum(
                self.task_weights[index].item()
                for index in range(self.num_tasks) if index not in eligible
            )
            active_budget = self.num_tasks - inactive_sum
            excess = torch.stack([
                (self.task_weights[index] - self.gradnorm_min_weight).clamp_min(0.0)
                for index in eligible
            ])
            if float(excess.sum().item()) <= 1e-12:
                normalized = torch.full_like(excess, active_budget / len(eligible))
            else:
                normalized = self.gradnorm_min_weight + excess * (
                    (active_budget - len(eligible) * self.gradnorm_min_weight)
                    / excess.sum()
                )
            for position, index in enumerate(eligible):
                self.task_weights[index].copy_(normalized[position])
                task = self.task_array[index]
                self._add_gradnorm_metric(
                    f"gradnorm_weighted_grad_norm_{task}",
                    float((self.task_weights[index] * self._batch_raw_gradient_norms[index]).item()),
                )
                self._add_gradnorm_metric(
                    f"gradnorm_relative_rate_{task}",
                    float(relative_rates[position].item()),
                )
                self._add_gradnorm_metric(
                    f"gradnorm_target_grad_norm_{task}",
                    float(targets[position].item()),
                )
            self._add_gradnorm_metric("gradnorm_loss", float(balancing_loss.item()))

    def _record_gradient_pairs(self):
        for pair in self.GRADIENT_PAIRS:
            left, right = pair
            if left not in self._batch_shared_gradients or right not in self._batch_shared_gradients:
                continue
            cosine = float(
                F.cosine_similarity(
                    self._batch_shared_gradients[left].flatten(),
                    self._batch_shared_gradients[right].flatten(),
                    dim=0,
                    eps=1e-12,
                ).item()
            )
            if not math.isfinite(cosine):
                continue
            self._gradient_cosine_sums[pair] += cosine
            self._gradient_conflict_counts[pair] += int(cosine < 0.0)
            self._gradient_pair_counts[pair] += 1

    def _process_batch(self, batch, train_mode: bool):
        """Exclude parameter regularization from predictive validation loss."""
        if train_mode:
            self._training_batch_index += 1
            self._gradnorm_collect = self.task_weighting == "gradnorm"
            if self._gradnorm_collect:
                self._gradnorm_task_index = 0
                self._batch_task_losses = {}
                self._batch_raw_gradient_norms = {}
            self._collect_shared_gradient = (
                self.shared_gradient_analysis
                and (self._training_batch_index - 1)
                % self.shared_gradient_every_n_batches == 0
            )
            if self._collect_shared_gradient:
                self._diagnostic_task_index = 0
                self._batch_shared_gradients = {}
            try:
                return super()._process_batch(batch, train_mode=True)
            finally:
                self._collect_shared_gradient = False
                self._gradnorm_collect = False
                if hasattr(self, "_batch_shared_gradients"):
                    self._batch_shared_gradients.clear()
                if hasattr(self, "_batch_task_losses"):
                    self._batch_task_losses.clear()

        l1_lambda, l2_lambda = self.l1_lambda, self.l2_lambda
        try:
            self.l1_lambda = 0.0
            self.l2_lambda = 0.0
            return super()._process_batch(batch, train_mode=False)
        finally:
            self.l1_lambda, self.l2_lambda = l1_lambda, l2_lambda

    def consume_shared_gradient_stats(self):
        """Return measured training-epoch scalars; omit unobserved tasks/pairs."""
        metrics = {}
        for task, count in self._gradient_norm_counts.items():
            if count:
                metrics[f"shared_fc2_grad_norm_{task}"] = (
                    self._gradient_norm_sums[task] / count
                )
                metrics[f"shared_fc2_grad_norm_max_{task}"] = (
                    self._gradient_norm_max[task]
                )
        for pair, count in self._gradient_pair_counts.items():
            if count:
                name = "_".join(pair)
                metrics[f"shared_fc2_grad_cosine_{name}"] = (
                    self._gradient_cosine_sums[pair] / count
                )
                metrics[f"shared_fc2_grad_conflict_rate_{name}"] = (
                    self._gradient_conflict_counts[pair] / count
                )
        self._reset_shared_gradient_stats()
        return metrics

    def consume_gradnorm_stats(self):
        if self.task_weighting != "gradnorm":
            return {}
        metrics = {
            name: total / self._gradnorm_metric_counts[name]
            for name, total in self._gradnorm_metric_sums.items()
        }
        metrics.update({
            f"gradnorm_weight_{task}": float(self.task_weights[index].detach().item())
            for index, task in enumerate(self.task_array)
        })
        self._reset_gradnorm_epoch_stats()
        return metrics

    def save_training_state(self, epoch, validation_loss):
        """Save resumable GradNorm state separately from evaluator model checkpoints."""
        if self.task_weighting != "gradnorm":
            return None
        path = os.path.join(self.ckpt_dir, "gradnorm_training_state.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "task_weights": [weight.detach().cpu() for weight in self.task_weights],
            "weight_optimizer_state_dict": self.weight_optimizer.state_dict(),
            "initial_task_losses": self.initial_task_losses,
            # Parent scheduler.step(val_loss) runs after the epoch report hook.
            "pending_scheduler_loss": validation_loss,
        }, path)
        return path

    def load_training_state(self, path):
        """Load a full GradNorm state or a legacy model-only checkpoint."""
        if self.task_weighting != "gradnorm":
            raise ValueError("GradNorm training state requires task_weighting='gradnorm'")
        state = torch.load(path, map_location=self.device, weights_only=False)
        if "model_state_dict" not in state:
            self.model.load_state_dict(state)
            return 0
        required = ("optimizer_state_dict", "scheduler_state_dict", "task_weights",
                    "weight_optimizer_state_dict", "initial_task_losses")
        if any(key not in state for key in required):
            raise ValueError("Incomplete GradNorm training state; expected model-only or full state")
        if len(state["task_weights"]) != self.num_tasks or len(state["initial_task_losses"]) != self.num_tasks:
            raise ValueError("GradNorm training state task count does not match this trainer")
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        with torch.no_grad():
            for parameter, value in zip(self.task_weights, state["task_weights"]):
                parameter.copy_(value.to(self.device))
        self.weight_optimizer.load_state_dict(state["weight_optimizer_state_dict"])
        self.initial_task_losses = list(state["initial_task_losses"])
        if state.get("pending_scheduler_loss") is not None:
            self.scheduler.step(state["pending_scheduler_loss"])
        return int(state.get("epoch", 0))

    def consume_gradient_norm_stats(self):
        """Return and reset accumulated pre-clipping gradient statistics."""
        if self.grad_norm_steps == 0:
            return None
        stats = {
            "mean": self.grad_norm_sum / self.grad_norm_steps,
            "max": self.grad_norm_max,
        }
        self.grad_norm_sum = 0.0
        self.grad_norm_max = 0.0
        self.grad_norm_steps = 0
        return stats
