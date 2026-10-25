"""
MSSL trainer: alternating minimization for wavCSE-MSSL (Goncalves et al.,
JMLR 17(33), 2016), p-MSSL instantiation.

Mirrors the structure of `01-mtrl/mtrl_trainer.py` so the two arms differ only
in the relation mechanism:
  1. W step (every batch): standard training plus the coupling term
     lambda_0 * tr(W Omega W^T) -- `get_relation_loss()`.
  2. Omega step (every `omega_update_frequency` epochs, after
     `mssl_warmup_epochs`): solve the graphical-lasso problem by ADMM in
     closed form at the level of the Omega subproblem (eigendecomposition +
     soft-thresholding), i.e. `model.update_omega()`. Omega is never touched
     by the optimizer.

Both arms record a per-epoch relation snapshot to `omega_history.json` in the
same JSON shape, so relation behaviour is comparable across arms. MSSL's
entries additionally carry the Omega solver's own optimality certificate for
that snapshot (`dual_violation`, `relative_duality_gap` -- see
`mssl_model.optimality_certificate`), which the MTRL arm's analytic step has
no analogue for.

MSSL's entries also carry the mechanism bundle this arm's study needs (the
coupling term's *scale* is the TR-0007 finding), all measured per epoch and
never fed back into training:

* `omega_eigenvalues`, `trace`, `support_edges` / `zero_offdiagonals` and the
  partial correlations -- the relation object itself;
* `summary_cosines`, `summary_gram`, `summary_gram_eigenvalues`,
  `summary_row_norms` and `summary_raw_row_norms` -- the task-parameter
  summary's geometry, which is what the coupling reshapes;
* `coupling_value` = `lambda_0 * tr(W Omega W^T)` at that snapshot, read off
  the same `W` the coupling term uses.

Separately, `coupling_scale.json` records one *gradient-scale* probe per epoch
on the first training batch -- `||d L_task/d theta||` over all trainable
parameters and over the classifier heads, `||d coupling/d theta||` over the
same sets, their ratio, and the coupling value on that batch. The probe uses
`torch.autograd.grad(..., retain_graph=True)`, which never writes optimizer
`.grad` buffers, so the measured step is the same step the run would have
taken without instrumentation (`test_mssl_mechanism_probe.py` asserts this
bit-exactly).
"""

import os
import json
import math
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import mlflow

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../downstream"))
# The Omega solver's optimality certificate lives in the sibling
# `mssl_model.py`. This folder has no importable package path (a leading digit
# plus a hyphen), so import it off its own directory, the same way
# run_improvements.py loads these modules by file path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainer.trainer_model import MultiTasksModelTrainer
from trainer.trainer_utils import BatchStats, masked_ce_loss, masked_accuracy
from mssl_model import optimality_certificate


class MultiTasksModelTrainerMSSL(MultiTasksModelTrainer):
    """Alternating minimization between task parameters and task precision Omega."""

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
        mssl_warmup_epochs: int = 3,
        omega_update_frequency: int = 1,
        lambda_2_selection: str = "unspecified",
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

        self.mssl_warmup_epochs = mssl_warmup_epochs
        self.omega_update_frequency = omega_update_frequency
        self.lambda_2_selection = str(lambda_2_selection)
        self.current_epoch = 0
        self.omega_history = []
        # Mechanism instrumentation (see the module docstring): one gradient-scale
        # probe per epoch on the first training batch, plus the coupling value and
        # the summary/Omega geometry recorded with each Omega snapshot. Everything
        # here is read-only with respect to training.
        self.coupling_history: List[Dict] = []
        self._probe_batch_seen = 0
        self._relation_parameter_names = tuple(
            name for name, _ in self.model.named_parameters()
            if name.startswith("classifiers.")
        )

        logging.info(
            "MSSL warmup epochs: %d, omega_update_frequency: %d",
            mssl_warmup_epochs,
            omega_update_frequency,
        )

    def _optimality_certificate(self, omega: torch.Tensor):
        """Score an Omega snapshot against the solver's own optimality conditions.

        Reconstructs exactly the sample covariance `model.update_omega()` solved
        against -- S = (1/d) W W^T for the current summary W, with d the number
        of summary columns -- and evaluates Eq. (8)'s primal-dual certificate
        (see `mssl_model.optimality_certificate`). This reports the quantity the
        ADMM stopped on, so a value above the solver's tolerance means the
        snapshot is not a certified solution of the Omega step.
        """
        with torch.no_grad():
            W = self.model.get_task_parameter_matrix().detach()
            d = int(W.shape[1])
            sample_covariance = (W @ W.transpose(0, 1)) / float(d)
        return optimality_certificate(
            omega,
            sample_covariance.cpu(),
            self.model.mssl_lambda_2,
            d,
            self.model.mssl_lambda_0,
        )

    def _summary_geometry(self) -> Dict:
        """Summary Gram, row norms and pairwise cosines at the current parameters.

        `summary_cosines` are the Gram's off-diagonals of the normalised summary
        (the adapter's output when `normalize_w` is on); `summary_raw_row_norms`
        are the rows' norms before that normalisation, which the normalised
        matrix cannot show. Reporting only, never fed back into training.
        """
        summary = self.model.get_task_parameter_matrix().detach().float().cpu()
        gram = summary @ summary.transpose(0, 1)
        raw_norms = self.model.get_raw_task_parameter_matrix().norm(dim=1)
        return {
            "summary_gram": [[self._round(value, 6) for value in row] for row in gram],
            "summary_cosines": [
                self._round(gram[0, 1], 6),
                self._round(gram[0, 2], 6),
                self._round(gram[1, 2], 6),
            ],
            "summary_gram_eigenvalues": [
                self._round(value, 6) for value in torch.linalg.eigvalsh(gram.double())
            ],
            "summary_row_norms": [self._round(value, 6) for value in summary.norm(dim=1)],
            "summary_raw_row_norms": [self._round(value, 6) for value in raw_norms],
        }

    def _coupling_value(self, omega: torch.Tensor) -> float:
        """`lambda_0 * tr(W Omega W^T)` exactly, from the same W the loss uses."""
        with torch.no_grad():
            W = self.model.get_task_parameter_matrix().detach().float().cpu()
            value = float(self.model.mssl_lambda_0) * float(
                torch.trace(W.double().transpose(0, 1) @ omega.double() @ W.double())
            )
        return self._round(value, 4)

    @staticmethod
    def _round(value, digits: int) -> float:
        return round(float(value), digits)

    def _record_omega(self, epoch: int) -> None:
        omega = self.model.get_omega_matrix()
        omega_list = omega.tolist()
        partial_list = self.model.get_partial_correlations().tolist()
        dual_violation, relative_duality_gap = self._optimality_certificate(omega)
        num_tasks = self.model.num_tasks
        omega_double = omega.double()
        off_diagonal = omega_double - torch.diag(torch.diagonal(omega_double))
        # Exact support of the relation object. The edges are the strict-upper
        # index pairs; the count has to be taken over those pairs explicitly,
        # because `torch.triu` returns the *whole* matrix with its lower part
        # zeroed, so counting zeros in its output also counts the zeroed region
        # (a 3-task snapshot with two absent edges then reported eight zeros and
        # a negative edge count -- fixed 2026-09-30; the raw Omega was recorded
        # correctly throughout, so the earlier records are recoverable from it).
        edge_pairs = [(i, j) for i in range(num_tasks) for j in range(i + 1, num_tasks)]
        zero_offdiagonals = sum(
            1 for i, j in edge_pairs if float(off_diagonal[i, j]) == 0.0
        )
        support_edges = len(edge_pairs) - zero_offdiagonals

        entry = {
            "epoch": epoch,
            "omega": omega_list,
            "partial_correlations": partial_list,
            "dual_violation": dual_violation,
            "relative_duality_gap": relative_duality_gap,
            "lambda_2": float(self.model.mssl_lambda_2),
            "lambda_0": float(self.model.mssl_lambda_0),
            "omega_trace": self._round(torch.trace(omega.double()), 4),
            "omega_eigenvalues": [
                self._round(value, 4) for value in torch.linalg.eigvalsh(omega.double())
            ],
            "mean_abs_offdiagonal": self._round(
                off_diagonal.abs().sum() / max(num_tasks * (num_tasks - 1), 1), 4
            ),
            "zero_offdiagonals": zero_offdiagonals,
            "support_edges": support_edges,
            "coupling_value": self._coupling_value(omega),
        }
        entry.update(self._summary_geometry())
        self.omega_history.append(entry)

        if mlflow.active_run() is not None:
            metrics = {}
            for i, task_i in enumerate(self.task_array):
                for j, task_j in enumerate(self.task_array):
                    if j < i:
                        continue
                    key = (
                        f"omega_{task_i}_{task_j}" if i != j
                        else f"omega_diag_{task_i}"
                    )
                    metrics[key] = omega_list[i][j]
            metrics["omega_dual_violation"] = dual_violation
            metrics["omega_relative_duality_gap"] = relative_duality_gap
            mlflow.log_metrics(metrics, step=epoch)

        num_tasks = self.model.num_tasks
        off_diag_mask = 1.0 - torch.eye(num_tasks)
        off_diag = (omega * off_diag_mask).abs()
        zero_off_diag = int(((off_diag == 0.0).sum()).item())
        possible_off_diag = num_tasks * (num_tasks - 1)
        logging.info(
            "MSSL Omega updated at epoch %d: mean|off-diagonal|=%.6f, "
            "trace=%.6f, zero off-diagonals=%d/%d, dual_violation=%.3g, "
            "relative_duality_gap=%.3g",
            epoch,
            off_diag.sum().item() / max(possible_off_diag, 1),
            torch.trace(omega).item(),
            zero_off_diag,
            possible_off_diag,
            dual_violation,
            relative_duality_gap,
        )

    def _save_omega_history(self) -> None:
        if not self.omega_history:
            return
        path = os.path.join(self.results_dir, "omega_history.json")
        payload = {
            "task_array": self.task_array,
            "method": "mssl",
            "history": self.omega_history,
            "final_omega": self.omega_history[-1]["omega"],
            "final_partial_correlations": self.omega_history[-1][
                "partial_correlations"
            ],
        }
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2)

        if mlflow.active_run() is not None:
            mlflow.log_artifact(path, artifact_path="omega")

        logging.info("MSSL Omega history (%d snapshots) saved to %s",
                     len(self.omega_history), path)

    # ---- mechanism instrumentation (read-only; see the module docstring) ----

    def _trainable_parameters(self) -> List[torch.nn.Parameter]:
        return [parameter for parameter in self.model.parameters()
                if parameter.requires_grad]

    def _head_parameters(self) -> List[torch.nn.Parameter]:
        return [parameter for name, parameter in self.model.named_parameters()
                if parameter.requires_grad and name.startswith("classifiers.")]

    @staticmethod
    def _gradient_norm(output: torch.Tensor,
                       parameters: List[torch.nn.Parameter]) -> Optional[float]:
        """||d output / d parameters||_2, without writing optimizer `.grad` buffers.

        `torch.autograd.grad` returns the gradients instead of accumulating them
        into `parameter.grad`, and the graph is kept so the training step's own
        `backward()` still sees it. Returns None when `output` does not depend on
        any of `parameters`.
        """
        gradients = torch.autograd.grad(
            output, parameters, retain_graph=True, allow_unused=True
        )
        total = 0.0
        for gradient in gradients:
            if gradient is not None:
                total += float(gradient.detach().double().pow(2).sum().item())
        if total == 0.0 and all(gradient is None for gradient in gradients):
            return None
        return math.sqrt(total)

    def _probe_gradient_scales(self, task_loss: torch.Tensor,
                               coupling: Optional[torch.Tensor]) -> None:
        """Record one epoch's task/coupling gradient scales on the probe batch."""
        all_parameters = self._trainable_parameters()
        head_parameters = self._head_parameters()
        task_all = self._gradient_norm(task_loss, all_parameters)
        task_heads = self._gradient_norm(task_loss, head_parameters)
        record = {
            "epoch": self.current_epoch,
            "warmup_epochs": int(self.mssl_warmup_epochs),
            "task_loss_value": self._round(task_loss.detach().item(), 6),
            "task_grad_norm_all": None if task_all is None else self._round(task_all, 6),
            "task_grad_norm_heads": (
                None if task_heads is None else self._round(task_heads, 6)
            ),
            "relation_value": None,
            "relation_grad_norm_all": None,
            "relation_grad_norm_heads": None,
            "relation_over_task_grad_ratio_heads": None,
            "relation_over_task_grad_ratio_all": None,
        }
        if coupling is not None:
            relation_all = self._gradient_norm(coupling, all_parameters)
            relation_heads = self._gradient_norm(coupling, head_parameters)
            record.update({
                "relation_value": self._round(coupling.detach().item(), 6),
                "relation_grad_norm_all": (
                    None if relation_all is None else self._round(relation_all, 6)
                ),
                "relation_grad_norm_heads": (
                    None if relation_heads is None else self._round(relation_heads, 6)
                ),
            })
            if relation_heads is not None and task_heads:
                record["relation_over_task_grad_ratio_heads"] = self._round(
                    relation_heads / task_heads, 6
                )
            if relation_all is not None and task_all:
                record["relation_over_task_grad_ratio_all"] = self._round(
                    relation_all / task_all, 6
                )
        self.coupling_history.append(record)

        if mlflow.active_run() is not None:
            metrics = {"task_grad_norm_heads": record["task_grad_norm_heads"]}
            if record["relation_grad_norm_heads"] is not None:
                metrics["relation_grad_norm_heads"] = record["relation_grad_norm_heads"]
                metrics["relation_value"] = record["relation_value"]
            if record["relation_over_task_grad_ratio_heads"] is not None:
                metrics["relation_over_task_grad_ratio"] = (
                    record["relation_over_task_grad_ratio_heads"]
                )
            mlflow.log_metrics(
                {key: value for key, value in metrics.items() if value is not None},
                step=self.current_epoch,
            )

        logging.info(
            "MSSL gradient scale probe | epoch=%d | ||g_task||_heads=%.6g | "
            "||g_coupling||_heads=%s | ratio=%s | coupling=%.6g",
            self.current_epoch,
            record["task_grad_norm_heads"],
            record["relation_grad_norm_heads"],
            record["relation_over_task_grad_ratio_heads"],
            record["relation_value"] if record["relation_value"] is not None else 0.0,
        )

    def _save_coupling_scale(self) -> None:
        if not self.coupling_history:
            return
        onset = [
            record["epoch"] for record in self.coupling_history
            if record["relation_value"] is not None
        ]
        path = os.path.join(self.results_dir, "coupling_scale.json")
        payload = {
            "task_array": self.task_array,
            "method": "mssl",
            "probe": "first training batch of every epoch",
            "gradient_parameter_sets": {
                "all": "every trainable parameter of the model",
                "heads": "parameters whose name starts with 'classifiers.'",
            },
            "lambda_0": float(self.model.mssl_lambda_0),
            "lambda_1": float(self.model.mssl_lambda_1),
            "lambda_2": float(self.model.mssl_lambda_2),
            "lambda_2_selection": self.lambda_2_selection,
            "normalize_w": bool(self.model.normalize_w),
            "warmup_epochs": int(self.mssl_warmup_epochs),
            "coupling_onset_epoch": onset[0] if onset else None,
            "history": self.coupling_history,
        }
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2)

        if mlflow.active_run() is not None:
            mlflow.log_artifact(path, artifact_path="mechanism")

        logging.info("MSSL coupling scale history (%d epochs) saved to %s",
                     len(self.coupling_history), path)

    def _process_batch(self, batch, train_mode: bool):
        input_seq, labels_list = self._unpack_batch(batch)

        if train_mode:
            self.optimizer.zero_grad(set_to_none=True)

        outputs = self.model(input_seq=input_seq)
        logits_tuple = outputs.logits
        pred_tuple = outputs.prediction

        loss_weight = 1.0 / float(self.num_tasks)
        loss_all = torch.tensor(0.0, device=self.device)
        batch_all = 1

        loss_task, batch_task = {}, {}
        correct_task, samples_task, valid_count_task = {}, {}, {}

        per_task_losses = {}
        for t in range(self.num_tasks):
            loss_t, valid = masked_ce_loss(
                logits=logits_tuple[t],
                labels=labels_list[t],
                loss_fn=self.loss_fn,
                ignore_index=self.ignore_index,
            )
            per_task_losses[t] = loss_t
            valid_count_task[t] = valid

        for t in range(self.num_tasks):
            lt_raw = per_task_losses[t]
            lt, loss_all, bt = self._calculate_total_loss(lt_raw, loss_all, loss_weight)
            loss_task[t] = float(lt.item()) if lt_raw is not None else 0.0
            batch_task[t] = int(bt)

        l1_reg = torch.tensor(0.0, device=self.device)
        l2_reg = torch.tensor(0.0, device=self.device)
        for parameter in self.model.parameters():
            l1_reg = l1_reg + torch.sum(torch.abs(parameter))
            l2_reg = l2_reg + torch.sum(torch.square(parameter))
        # The task term (the batch-mean weighted cross-entropy) before the
        # regularizers and the coupling: the scale the coupling is compared with.
        task_loss_term = loss_all
        loss_all = loss_all + self.l1_lambda * l1_reg + self.l2_lambda * l2_reg

        # === MSSL MODIFICATION: coupling term tr(W Omega W^T), post-warmup ===
        coupling_term = None
        if self.current_epoch >= self.mssl_warmup_epochs:
            coupling_term = self.model.get_relation_loss()
            loss_all = loss_all + coupling_term
        # === MSSL MODIFICATION END ===

        if train_mode:
            self._probe_batch_seen += 1
            is_probe_batch = self._probe_batch_seen == 1
            if is_probe_batch:
                # Read-only measurement on the epoch's first training batch: it
                # writes no `.grad` buffer and keeps the graph, so the backward
                # below and the optimizer step are unchanged.
                self._probe_gradient_scales(task_loss_term, coupling_term)
            loss_all.backward()
            self.optimizer.step()

        correct_all = 0
        samples_all = 0
        for t in range(self.num_tasks):
            c, s = masked_accuracy(pred_tuple[t], labels_list[t],
                                   ignore_index=self.ignore_index)
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
            valid_count_task=valid_count_task,
        )

    def train(self) -> None:
        logging.info(
            "training_start_MSSL | task_type=%s | num_tasks=%d | epochs=%d | batch_size=%d | device=%s",
            self.task_type, self.num_tasks, self.num_epochs,
            self.batch_size, str(self.device),
        )

        for ep in range(1, self.num_epochs + 1):
            self.current_epoch = ep
            self._probe_batch_seen = 0
            logging.info("epoch_start | epoch=%d/%d", ep, self.num_epochs)

            train_stats = self._process_data_loader(self.train_dataloader, train_mode=True)
            val_stats = self._process_data_loader(self.val_dataloader, train_mode=False)

            # Alternating minimization: ADMM graphical-lasso Omega step.
            if (ep >= self.mssl_warmup_epochs
                    and ep % self.omega_update_frequency == 0):
                self.model.update_omega()
                self._record_omega(ep)

            self.train_losses_all.append(train_stats.avg_loss_all)
            self.train_acc_all.append(train_stats.accuracy_all)
            self.val_losses_all.append(val_stats.avg_loss_all)
            self.val_acc_all.append(val_stats.accuracy_all)

            for t in range(self.num_tasks):
                self.train_losses_task[t].append(train_stats.avg_loss_task[t])
                self.train_acc_task[t].append(train_stats.accuracy_task[t])
                self.val_losses_task[t].append(val_stats.avg_loss_task[t])
                self.val_acc_task[t].append(val_stats.accuracy_task[t])

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

        logging.info(
            "training_end | run_id=%s | best_all=%.6f | opt_all=%.6f",
            os.path.basename(self.results_dir),
            self.ckpt.best_accuracy_all_threshold,
            self.ckpt.opt_accuracy_all_threshold
        )

        self._save_omega_history()
        self._save_coupling_scale()
        self.plot_metrics()
