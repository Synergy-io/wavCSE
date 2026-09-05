"""
MTRL Trainer: Alternating minimization for wavCSE-MTRL.

Zhang & Yeung (2010/2012) alternate between:
1. Update W (task parameters) with Omega fixed -> standard training step,
   with the regularizer tr(W Omega^-1 W^T) added to the classification
   loss (see `mtrl_combine.MTRLCombineStrategy.__call__`).
2. Update Omega (task covariance) with W fixed -> closed-form analytic
   step (matrix square root, no gradient descent) -- see the strategy's
   `on_epoch_end` hook.

Unlike wavCSE-TSM's alternation (a separate Adam optimizer looping over
the full training set every `alternate_frequency` epochs) and wavCSE-PMR's
(a single joint loss, Omega learned by backprop every batch), MTRL's
Omega-step is a cheap analytic computation on a small [num_tasks,
num_tasks] matrix, so it can run every epoch (or as configured via
`omega_update_frequency`) with negligible cost.

The full loss is:
  L_total = L_classification + l1_lambda*|W|_1 + l2_lambda*|W|_2^2
            + mtrl_lambda * tr(W Omega^-1 W^T)   [after warmup]

Migrated onto mtlkit's combine() seam (Next Step 6, Eng Review decision
D1): `_process_batch`'s ~85-line duplicated override collapses into
`mtlkit.trainer.process_batch` + `MTRLCombineStrategy` (this trainer's job
is now: drive the epoch-lifecycle hooks, apply the generic L1/L2
regularization the base trainer also applies, and do the backward/step --
the classification-loss-combination + Omega math live in
`mtrl_combine.py`). Drops the `sys.path.insert` hack this file used to
reach `downstream/` -- `mtlkit` is now an installed package; `downstream/`
still needs to be on `sys.path` for the base trainer import below, exactly
as `improvements/run_improvements.py` (this trainer's real entry point)
already arranges.
"""

import json
import logging
import os
import sys
from typing import Dict

import torch
import torch.nn as nn
import mlflow

import mtlkit.trainer as mtlkit_trainer

# 01-mtrl/ starts with a digit, so it isn't a valid Python package name --
# add just this directory (not a reach into a different subtree, unlike the
# sys.path.insert hack this migration drops) so the sibling mtrl_combine.py
# module resolves regardless of how this file itself was loaded.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from mtrl_combine import MTRLCombineStrategy

from trainer.trainer_model import MultiTasksModelTrainer
from trainer.trainer_utils import BatchStats


class MultiTasksModelTrainerMTRL(MultiTasksModelTrainer):
    """
    Trainer for wavCSE-MTRL with alternating minimization between the task
    parameters (gradient descent, every batch) and the task covariance
    matrix Omega (closed-form analytic update, every `omega_update_frequency`
    epochs, after `mtrl_warmup_epochs`).
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        task_type: str,
        training_cfg: Dict,
        results_root: str,
        checkpoints_root: str,
        training_data=None,
        validation_data=None,
        ignore_index: int = -1,
        mtrl_warmup_epochs: int = 3,     # epochs before enabling the MTRL regularizer / Omega updates
        omega_update_frequency: int = 1, # how often (in epochs) to refresh Omega analytically
        mtrl_lambda: float = 0.01,
        omega_epsilon: float = 1e-4,
        normalize_w: bool = False,
    ):
        super().__init__(
            model=model,
            device=device,
            task_type=task_type,
            training_cfg=training_cfg,
            results_root=results_root,
            checkpoints_root=checkpoints_root,
            training_data=training_data,
            validation_data=validation_data,
            ignore_index=ignore_index,
        )

        self.combine_strategy = MTRLCombineStrategy(
            num_tasks=self.num_tasks,
            mtrl_lambda=mtrl_lambda,
            omega_epsilon=omega_epsilon,
            normalize_w=normalize_w,
            mtrl_warmup_epochs=mtrl_warmup_epochs,
            omega_update_frequency=omega_update_frequency,
        )

        logging.info(
            f"MTRL warmup epochs: {mtrl_warmup_epochs}, "
            f"omega_update_frequency: {omega_update_frequency}"
        )

    def _save_omega_history(self) -> None:
        """
        Write the full per-epoch Omega history (plus the final matrix) to
        results_dir/omega_history.json. This directory is already uploaded
        wholesale as an MLflow artifact by run_improvements.py's
        `mlflow.log_artifacts(trainer.results_dir, artifact_path="results")`
        call, so no separate log_artifact call is needed here -- but do it
        anyway (under "omega/") so it's easy to find without digging through
        the full results/ tree.
        """
        history = self.combine_strategy.omega_history
        if not history:
            return

        path = os.path.join(self.results_dir, "omega_history.json")
        payload = {
            "task_array": self.task_array,
            "history": history,
            "final_omega": history[-1]["omega"],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

        if mlflow.active_run() is not None:
            mlflow.log_artifact(path, artifact_path="omega")

        logging.info(f"Omega history ({len(history)} snapshots) saved to {path}")

    def _process_batch(self, batch, train_mode: bool) -> BatchStats:
        """
        Per-task masked losses + the MTRL Omega regularizer now come from
        `mtlkit.trainer.process_batch` + `self.combine_strategy` (see module
        docstring). L1/L2 regularization stays here -- it's the base
        trainer's generic concern too (see trainer_model.py's own
        `_process_batch`), applied around combine()'s result.
        """
        input_seq, labels_list = self._unpack_batch(batch)

        if train_mode:
            self.optimizer.zero_grad(set_to_none=True)

        result = mtlkit_trainer.process_batch(
            model=self.model,
            input_seq=input_seq,
            labels_list=labels_list,
            loss_fn=self.loss_fn,
            combine_strategy=self.combine_strategy,
            ignore_index=self.ignore_index,
        )
        loss_all = result.loss_all
        batch_all = 1

        # L1 + L2 regularization (standard, over every parameter)
        l1_reg = torch.tensor(0.0, device=self.device)
        l2_reg = torch.tensor(0.0, device=self.device)
        for p in self.model.parameters():
            l1_reg = l1_reg + torch.sum(torch.abs(p))
            l2_reg = l2_reg + torch.sum(torch.square(p))
        loss_all = loss_all + self.l1_lambda * l1_reg + self.l2_lambda * l2_reg

        if train_mode:
            loss_all.backward()
            self.optimizer.step()

        correct_all = sum(result.correct_task.values())
        samples_all = sum(result.samples_task.values())
        batch_task = {t: int(v > 0) for t, v in result.valid_count_task.items()}

        self._save_debug_counts("train" if train_mode else "val", result.valid_count_task)

        return BatchStats(
            loss_all=float(loss_all.item()),
            batch_all=batch_all,
            loss_task=result.loss_task,
            batch_task=batch_task,
            correct_task=result.correct_task,
            samples_task=result.samples_task,
            correct_all=correct_all,
            samples_all=samples_all,
            valid_count_task=result.valid_count_task,
        )

    def train(self) -> None:
        logging.info(
            "training_start_MTRL | task_type=%s | num_tasks=%d | epochs=%d | batch_size=%d | device=%s",
            self.task_type, self.num_tasks, self.num_epochs,
            self.batch_size, str(self.device)
        )

        for ep in range(1, self.num_epochs + 1):
            self.combine_strategy.on_epoch_begin(ep)
            logging.info("epoch_start | epoch=%d/%d", ep, self.num_epochs)

            train_stats = self._process_data_loader(self.train_dataloader, train_mode=True)
            val_stats = self._process_data_loader(self.val_dataloader, train_mode=False)

            # Alternating minimization: closed-form Omega update, post-warmup
            # (the strategy's on_epoch_end decides whether this epoch is due,
            # per mtrl_warmup_epochs/omega_update_frequency)
            self.combine_strategy.on_epoch_end(ep)

            # History tracking
            self.train_losses_all.append(train_stats.avg_loss_all)
            self.train_acc_all.append(train_stats.accuracy_all)
            self.val_losses_all.append(val_stats.avg_loss_all)
            self.val_acc_all.append(val_stats.accuracy_all)

            for t in range(self.num_tasks):
                self.train_losses_task[t].append(train_stats.avg_loss_task[t])
                self.train_acc_task[t].append(train_stats.accuracy_task[t])
                self.val_losses_task[t].append(val_stats.avg_loss_task[t])
                self.val_acc_task[t].append(val_stats.accuracy_task[t])

            # Metrics logging
            self.metrics_writer.write_metrics_all(
                self._epoch_report_line(ep, "train", train_stats))
            self.metrics_writer.write_metrics_all(
                self._epoch_report_line(ep, "val", val_stats))

            for t, task in enumerate(self.task_array):
                self.metrics_writer.write_metrics_task(
                    t,
                    f"Epoch {ep}/{self.num_epochs} | {task} | "
                    f"train_loss={train_stats.avg_loss_task[t]:.4f} "
                    f"train_acc={train_stats.accuracy_task[t]:.4f} | "
                    f"val_loss={val_stats.avg_loss_task[t]:.4f} "
                    f"val_acc={val_stats.accuracy_task[t]:.4f}"
                )

            lr = float(self.optimizer.param_groups[0]["lr"])
            self.learning_rate_array.append(lr)

            logging.info(
                "epoch_all | epoch=%d | lr=%.6g | train(loss=%.4f acc=%.4f) | val(loss=%.4f acc=%.4f)",
                ep, lr,
                train_stats.avg_loss_all, train_stats.accuracy_all,
                val_stats.avg_loss_all, val_stats.accuracy_all
            )

            for t, task in enumerate(self.task_array):
                logging.info(
                    "epoch_task | epoch=%d | task=%s | train(loss=%.4f acc=%.4f) | "
                    "val(loss=%.4f acc=%.4f) | samples=%d",
                    ep, task,
                    train_stats.avg_loss_task[t], train_stats.accuracy_task[t],
                    val_stats.avg_loss_task[t], val_stats.accuracy_task[t],
                    int(val_stats.total_samples_task.get(t, 0))
                )

            # Checkpoints + scheduler
            self.ckpt.save_epoch_checkpoint(ep)
            self.ckpt.maybe_save_best_and_opt(
                test_accuracy_all=val_stats.accuracy_all,
                test_accuracy_task=val_stats.accuracy_task,
                num_tasks=self.num_tasks
            )
            self.scheduler.step(val_stats.avg_loss_all)

        # Final summary
        self.metrics_writer.write_metrics_all(
            f"Best model accuracy - all: {self.ckpt.best_accuracy_all_threshold:.6f}")
        for t, task in enumerate(self.task_array):
            self.metrics_writer.write_metrics_all(
                f"Best model accuracy - {task}: {self.ckpt.best_accuracy_task_thresholds.get(t, 0.0):.6f}"
            )
        self.metrics_writer.write_metrics_all(
            f"Opt model accuracy - all: {self.ckpt.opt_accuracy_all_threshold:.6f}")
        for t, task in enumerate(self.task_array):
            self.metrics_writer.write_metrics_all(
                f"Opt model accuracy - {task}: {self.ckpt.opt_accuracy_task_thresholds.get(t, 0.0):.6f}"
            )

        logging.info("training_end | run_id=%s | best_all=%.6f | opt_all=%.6f",
                     os.path.basename(self.results_dir),
                     self.ckpt.best_accuracy_all_threshold,
                     self.ckpt.opt_accuracy_all_threshold)

        self._save_omega_history()
        self.plot_metrics()
