"""Thin stabilization extension for the standard downstream trainer."""

import os
import sys

from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau


_DOWNSTREAM_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "downstream")
)
if _DOWNSTREAM_DIR not in sys.path:
    sys.path.insert(0, _DOWNSTREAM_DIR)

from trainer.trainer_model import MultiTasksModelTrainer


class MultiTasksModelTrainerFTN(MultiTasksModelTrainer):
    """Reuse the standard loop with optional clipping and scheduler floor."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
