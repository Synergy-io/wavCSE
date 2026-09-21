"""
MTRL Trainer: Alternating minimization for wavCSE-MTRL.

Zhang & Yeung (2010/2012) alternate between:
1. Update W (task parameters) with Omega fixed -> standard training step,
   with the regularizer tr(W Omega^-1 W^T) added to the classification
   loss (see _process_batch).
2. Update Omega (task covariance) with W fixed -> closed-form analytic
   step (matrix square root, no gradient descent) -- see train()'s call
   to self.model.update_omega().

Unlike wavCSE-TSM's alternation (a separate Adam optimizer looping over
the full training set every `alternate_frequency` epochs) and wavCSE-PMR's
(a single joint loss, Omega learned by backprop every batch), MTRL's
Omega-step is a cheap analytic computation on a small [num_tasks,
num_tasks] matrix, so it can run every epoch (or as configured via
`omega_update_frequency`) with negligible cost.

The full loss is:
  L_total = L_classification + l1_lambda*|W|_1 + l2_lambda*|W|_2^2
            + mtrl_lambda * tr(W Omega^-1 W^T)   [after warmup]
"""

import os
import json
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import mlflow

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../downstream'))

from trainer.trainer_model import MultiTasksModelTrainer
from trainer.trainer_utils import BatchStats, masked_ce_loss, masked_accuracy


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

        self.mtrl_warmup_epochs = mtrl_warmup_epochs
        self.omega_update_frequency = omega_update_frequency
        self.current_epoch = 0

        # Full per-epoch Omega history, kept in memory and written to disk
        # (and uploaded as an MLflow artifact) at the end of train() -- see
        # _record_omega() and train()'s final section. The per-epoch summary
        # log line alone isn't enough to reconstruct the actual matrix later.
        self.omega_history = []

        logging.info(
            f"MTRL warmup epochs: {mtrl_warmup_epochs}, "
            f"omega_update_frequency: {omega_update_frequency}"
        )

    def _record_omega(self, epoch: int) -> None:
        """
        After model.update_omega(), snapshot the full Omega matrix: append it
        to self.omega_history (for the end-of-training omega_history.json
        artifact) and, if an MLflow run is active, log every entry as a
        per-epoch metric (omega_<task_i>_<task_j>) so its evolution is
        viewable as a chart on DagsHub, not just a scalar summary in the log.
        """
        Omega = self.model.get_omega_matrix()  # [num_tasks, num_tasks], detached CPU tensor
        omega_list = Omega.tolist()

        self.omega_history.append({"epoch": epoch, "omega": omega_list})

        if mlflow.active_run() is not None:
            metrics = {}
            for i, task_i in enumerate(self.task_array):
                for j, task_j in enumerate(self.task_array):
                    if j < i:
                        continue  # Omega is symmetric -- log each pair once
                    key = f"omega_{task_i}_{task_j}" if i != j else f"omega_diag_{task_i}"
                    metrics[key] = omega_list[i][j]
            mlflow.log_metrics(metrics, step=epoch)

        off_diag_mask = 1.0 - torch.eye(self.model.num_tasks)
        mean_off_diag = (Omega * off_diag_mask).abs().sum().item() / max(
            self.model.num_tasks * (self.model.num_tasks - 1), 1
        )
        logging.info(
            f"MTRL Omega updated at epoch {epoch}: "
            f"mean|off-diagonal|={mean_off_diag:.6f}, trace={torch.trace(Omega).item():.6f}"
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
        if not self.omega_history:
            return

        path = os.path.join(self.results_dir, "omega_history.json")
        payload = {
            "task_array": self.task_array,
            "history": self.omega_history,
            "final_omega": self.omega_history[-1]["omega"],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

        if mlflow.active_run() is not None:
            mlflow.log_artifact(path, artifact_path="omega")

        logging.info(f"Omega history ({len(self.omega_history)} snapshots) saved to {path}")

    def _process_batch(self, batch, train_mode: bool):
        """
        Override batch processing to add the MTRL regularization term
        (post-warmup). Structurally identical to the base trainer's
        _process_batch, with one addition (see "MTRL MODIFICATION").
        """
        input_seq, labels_list = self._unpack_batch(batch)

        if train_mode:
            self.optimizer.zero_grad(set_to_none=True)

        outputs = self.model(input_seq=input_seq)
        logits_tuple = outputs.logits
        pred_tuple = outputs.prediction

        loss_weight = 1.0 / float(self.num_tasks)

        loss_all = torch.tensor(0.0, device=self.device)
        batch_all = 1

        loss_task = {}
        batch_task = {}
        correct_task = {}
        samples_task = {}
        valid_count_task = {}

        per_task_losses = {}
        for t in range(self.num_tasks):
            loss_t, valid = masked_ce_loss(
                logits=logits_tuple[t],
                labels=labels_list[t],
                loss_fn=self.loss_fn,
                ignore_index=self.ignore_index
            )
            per_task_losses[t] = loss_t
            valid_count_task[t] = valid

        for t in range(self.num_tasks):
            lt_raw = per_task_losses[t]
            lt, loss_all, bt = self._calculate_total_loss(lt_raw, loss_all, loss_weight)
            loss_task[t] = float(lt.item()) if lt_raw is not None else 0.0
            batch_task[t] = int(bt)

        # L1 + L2 regularization (standard, over every parameter)
        l1_reg = torch.tensor(0.0, device=self.device)
        l2_reg = torch.tensor(0.0, device=self.device)
        for p in self.model.parameters():
            l1_reg = l1_reg + torch.sum(torch.abs(p))
            l2_reg = l2_reg + torch.sum(torch.square(p))
        loss_all = loss_all + self.l1_lambda * l1_reg + self.l2_lambda * l2_reg

        # === MTRL MODIFICATION START ===
        # Add the tr(W Omega^-1 W^T) regularizer after warmup. Uses the
        # live classifier-head weights (see get_mtrl_regularizer_loss's
        # docstring), so this genuinely pulls related tasks' parameters
        # together via the normal backward pass.
        if self.current_epoch >= self.mtrl_warmup_epochs:
            mtrl_loss = self.model.get_mtrl_regularizer_loss()
            loss_all = loss_all + mtrl_loss
        # === MTRL MODIFICATION END ===

        if train_mode:
            loss_all.backward()
            self.optimizer.step()

        correct_all = 0
        samples_all = 0
        for t in range(self.num_tasks):
            c, s = masked_accuracy(pred_tuple[t], labels_list[t], ignore_index=self.ignore_index)
            correct_task[t] = c
            samples_task[t] = s
            correct_all += c
            samples_all += s

        self._save_debug_counts("train" if train_mode else "val", valid_count_task)

        return BatchStats(
            loss_all=float(loss_all.item()),
            batch_all=batch_all,
            loss_task=loss_task,
            batch_task=batch_task,
            correct_task=correct_task,
            samples_task=samples_task,
            correct_all=correct_all,
            samples_all=samples_all,
            valid_count_task=valid_count_task
        )

    def train(self) -> None:
        logging.info(
            "training_start_MTRL | task_type=%s | num_tasks=%d | epochs=%d | batch_size=%d | device=%s",
            self.task_type, self.num_tasks, self.num_epochs,
            self.batch_size, str(self.device)
        )

        for ep in range(1, self.num_epochs + 1):
            self.current_epoch = ep
            logging.info("epoch_start | epoch=%d/%d", ep, self.num_epochs)

            train_stats = self._process_data_loader(self.train_dataloader, train_mode=True)
            val_stats = self._process_data_loader(self.val_dataloader, train_mode=False)

            # Alternating minimization: closed-form Omega update, post-warmup
            if (ep >= self.mtrl_warmup_epochs
                    and ep % self.omega_update_frequency == 0):
                self.model.update_omega()
                self._record_omega(ep)

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
