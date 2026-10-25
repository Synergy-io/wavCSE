"""FTN trainer with optional safeguards and shared-FC2 gradient diagnostics."""

import math
import os
import sys

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau


_DOWNSTREAM_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "downstream")
)
if _DOWNSTREAM_DIR not in sys.path:
    sys.path.insert(0, _DOWNSTREAM_DIR)

from trainer.trainer_model import MultiTasksModelTrainer


class _TaskLossDispatcher(nn.Module):
    """Select one fixed cross-entropy criterion per task in call order."""

    def __init__(self, task_array, er_label_smoothing: float):
        super().__init__()
        if not 0.0 <= er_label_smoothing <= 1.0:
            raise ValueError(
                "er_label_smoothing must be between 0.0 and 1.0, got "
                f"{er_label_smoothing}"
            )
        self.task_array = tuple(task_array)
        self.criteria = nn.ModuleList(
            [
                nn.CrossEntropyLoss(
                    label_smoothing=(
                        er_label_smoothing if task == "er" else 0.0
                    )
                )
                for task in self.task_array
            ]
        )
        self._next_task = 0

    def reset(self) -> None:
        self._next_task = 0

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self._next_task >= len(self.criteria):
            raise RuntimeError("Task loss dispatcher was called too many times in one batch")
        criterion = self.criteria[self._next_task]
        self._next_task += 1
        return criterion(logits, labels)

    def smoothing_by_task(self):
        return {
            task: float(criterion.label_smoothing)
            for task, criterion in zip(self.task_array, self.criteria)
        }


class MultiTasksModelTrainerFTN(MultiTasksModelTrainer):
    """Standard loop with training-only regularization and optional safeguards."""

    GRADIENT_PAIRS = (("ks", "si"), ("ks", "er"), ("si", "er"))

    def __init__(self, *args, **kwargs):
        diagnostics_cfg = kwargs.pop("diagnostics_cfg", None) or {}
        training_cfg = kwargs.get("training_cfg") or {}
        self.er_label_smoothing = float(
            training_cfg.get("er_label_smoothing", 0.0)
        )
        self.shared_gradient_analysis = bool(
            diagnostics_cfg.get("shared_gradient_analysis", True)
        )
        self.shared_gradient_every_n_batches = int(
            diagnostics_cfg.get("shared_gradient_every_n_batches", 10)
        )
        if self.shared_gradient_every_n_batches < 1:
            raise ValueError("shared_gradient_every_n_batches must be >= 1")
        super().__init__(*args, **kwargs)
        self.loss_fn = _TaskLossDispatcher(
            self.task_array, self.er_label_smoothing
        )
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
        return super()._process_data_loader(data_loader, train_mode)

    def _calculate_total_loss(self, loss_task, loss_all, loss_weight):
        """Inspect each unweighted masked task loss before the parent's backward."""
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
                            self._gradient_norm_sums[task] += norm
                            self._gradient_norm_max[task] = max(
                                self._gradient_norm_max[task], norm
                            )
                            self._gradient_norm_counts[task] += 1
                            if norm > 1e-12:
                                self._batch_shared_gradients[task] = gradient
            if self._diagnostic_task_index == self.num_tasks:
                self._record_gradient_pairs()
                self._batch_shared_gradients.clear()
        return super()._calculate_total_loss(loss_task, loss_all, loss_weight)

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
        self.loss_fn.reset()
        if train_mode:
            self._training_batch_index += 1
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
                if hasattr(self, "_batch_shared_gradients"):
                    self._batch_shared_gradients.clear()

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
