"""Stage-wise trainer for the neural RAMUSA adaptation."""

import copy
import json
import logging
import os
import shutil
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from trainer.trainer_model import MultiTasksModelTrainer
from trainer.trainer_utils import BatchStats, masked_accuracy, masked_ce_loss
from improvements.lowrank.utils.ramusa import update_effective_rank
from improvements.lowrank.utils.adaptive_tau import AdaptiveTauManager
from improvements.lowrank.utils.lambda_scheduler import get_ramusa_lambda


class MultiTasksModelTrainerRAMUSA(MultiTasksModelTrainer):
    def __init__(self, *args, ramusa_cfg: dict, **kwargs):
        super().__init__(*args, **kwargs)
        if self.task_type != "ks_si_er":
            raise ValueError("RAMUSA supports only task_type='ks_si_er'.")
        self.ramusa_enabled = bool(ramusa_cfg.get("enabled", True))
        schedule_cfg = ramusa_cfg.get("lambda_schedule", {})
        self.lambda_schedule_enabled = bool(schedule_cfg.get("enabled", False))
        self.lambda_schedule_type = (
            str(schedule_cfg.get("type", "fixed"))
            if self.lambda_schedule_enabled else "fixed"
        )
        legacy_lambda = float(ramusa_cfg.get("lambda", schedule_cfg.get("lambda_max", 0.01)))
        self.lambda_max = float(
            schedule_cfg.get("lambda_max", legacy_lambda)
            if self.lambda_schedule_enabled else legacy_lambda
        )
        self.warmup_epochs = int(schedule_cfg.get("warmup_epochs", 0))
        self.delay_epochs = int(schedule_cfg.get("delay_epochs", 0))
        self.ramusa_lambda = self.lambda_max
        self.current_ramusa_lambda = self.lambda_max
        self.tau = float(ramusa_cfg.get("tau", 1.0))
        self.num_stages = int(ramusa_cfg.get("num_stages", 1))
        self.epochs_per_stage = int(ramusa_cfg.get("epochs_per_stage", self.num_epochs))
        if self.lambda_max < 0 or self.tau < 0:
            raise ValueError("RAMUSA lambda and tau must be non-negative")
        if self.warmup_epochs < 0 or self.delay_epochs < 0:
            raise ValueError("RAMUSA warmup_epochs and delay_epochs must be non-negative")
        get_ramusa_lambda(
            self.lambda_schedule_type, 0, self.lambda_max,
            self.warmup_epochs, self.delay_epochs,
        )
        if self.num_stages <= 0 or self.epochs_per_stage <= 0:
            raise ValueError("RAMUSA stages and epochs_per_stage must be positive")
        if self.num_stages * self.epochs_per_stage != self.num_epochs:
            raise ValueError("training.num_epochs must equal num_stages * epochs_per_stage")

        # Initialize adaptive tau manager
        adaptive_tau_cfg = ramusa_cfg.get("adaptive_tau", {})
        self.adaptive_tau_enabled = adaptive_tau_cfg.get("enabled", False)
        if self.adaptive_tau_enabled:
            self.adaptive_tau_manager = AdaptiveTauManager(adaptive_tau_cfg)
            logging.info("Adaptive tau enabled with config: %s", self.adaptive_tau_manager.get_config_dict())
        else:
            self.adaptive_tau_manager = None

        self.current_stage = 1
        self.effective_rank = 0
        self.global_epoch = 0
        self.stage_history = []
        self.lambda_history = []
        self.training_loss_fn = nn.CrossEntropyLoss(
            label_smoothing=float(self.training_cfg.get("label_smoothing", 0.0))
        )
        self.validation_loss_fn = nn.CrossEntropyLoss()
        self.stage_state_path = os.path.join(self.results_dir, "ramusa_state.json")
        self.training_state_path = os.path.join(self.ckpt_dir, "ramusa_training_state.pth")
        
        # Store stage boundary state for validation probes
        self._stage_boundary_state = None

    def _process_batch(self, batch, train_mode: bool) -> BatchStats:
        input_seq, labels_list = self._unpack_batch(batch)
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(input_seq=input_seq)
        loss_weight = 1.0 / float(self.num_tasks)
        loss_all = torch.tensor(0.0, device=self.device)
        loss_task: Dict[int, float] = {}
        batch_task: Dict[int, int] = {}
        correct_task: Dict[int, int] = {}
        samples_task: Dict[int, int] = {}
        valid_count_task: Dict[int, int] = {}
        loss_fn = self.training_loss_fn if train_mode else self.validation_loss_fn

        per_task_losses = {}
        for task_index in range(self.num_tasks):
            loss, valid = masked_ce_loss(
                outputs.logits[task_index], labels_list[task_index], loss_fn, self.ignore_index
            )
            per_task_losses[task_index] = loss
            valid_count_task[task_index] = valid
        mtl_loss = torch.tensor(0.0, device=self.device)
        for task_index, raw_loss in per_task_losses.items():
            task_loss, loss_all, present = self._calculate_total_loss(
                raw_loss, loss_all, loss_weight
            )
            if raw_loss is not None:
                mtl_loss = mtl_loss + raw_loss * loss_weight
            loss_task[task_index] = float(task_loss.item()) if raw_loss is not None else 0.0
            batch_task[task_index] = present

        rank_penalty = torch.tensor(0.0, device=self.device)
        singular_values = torch.empty(0, device=self.device)
        if train_mode and self.ramusa_enabled:
            singular_values = self.model.get_ramusa_singular_values()
            rank_penalty = singular_values[self.effective_rank:].sum()
            loss_all = loss_all + self.current_ramusa_lambda * rank_penalty
        l1_reg = sum(torch.sum(torch.abs(p)) for p in self.model.parameters())
        l2_reg = sum(torch.sum(torch.square(p)) for p in self.model.parameters())
        loss_all = loss_all + self.l1_lambda * l1_reg + self.l2_lambda * l2_reg
        if train_mode:
            loss_all.backward()
            self.optimizer.step()

        for task_index in range(self.num_tasks):
            correct, samples = masked_accuracy(
                outputs.prediction[task_index], labels_list[task_index], self.ignore_index
            )
            correct_task[task_index] = correct
            samples_task[task_index] = samples
        correct_all = sum(correct_task.values())
        samples_all = sum(samples_task.values())
        self._save_debug_counts("train" if train_mode else "val", valid_count_task)
        self._last_batch_details = {
            "mtl_loss": float(mtl_loss.item()),
            "rank_penalty": float(rank_penalty.detach().item()),
            "weighted_rank_penalty": float((self.current_ramusa_lambda * rank_penalty).detach().item()),
            "lambda_current": self.current_ramusa_lambda,
            "total_loss": float(loss_all.detach().item()),
            "singular_values": [float(value) for value in singular_values.detach().cpu()],
        }
        if train_mode and hasattr(self, "_epoch_batch_details"):
            self._epoch_batch_details.append(self._last_batch_details)
        return BatchStats(
            loss_all=float(loss_all.detach().item()), batch_all=1,
            loss_task=loss_task, batch_task=batch_task,
            correct_task=correct_task, samples_task=samples_task,
            correct_all=correct_all, samples_all=samples_all,
            valid_count_task=valid_count_task,
        )

    def _write_stage_state(self):
        state = {
            "stage": self.current_stage, "r": self.effective_rank,
            "tau": self.tau, "lambda": self.lambda_max,
            "global_epoch": self.global_epoch,
            "epoch_in_stage": self.global_epoch % self.epochs_per_stage,
            "lambda_schedule": self._lambda_schedule_state(),
            "lambda_current": self.current_ramusa_lambda,
            "lambda_history": self.lambda_history,
            "history": self.stage_history,
        }
        with open(self.stage_state_path, "w") as state_file:
            json.dump(state, state_file, indent=2)

    def _lambda_schedule_state(self) -> Dict[str, Any]:
        return {
            "enabled": self.lambda_schedule_enabled,
            "type": self.lambda_schedule_type,
            "lambda_max": self.lambda_max,
            "warmup_epochs": self.warmup_epochs,
            "delay_epochs": self.delay_epochs,
        }

    def _save_training_state(self) -> None:
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_epoch": self.global_epoch,
            "current_stage": self.current_stage,
            "epoch_in_stage": self.global_epoch % self.epochs_per_stage,
            "r": self.effective_rank,
            "tau": self.tau,
            "lambda_schedule": self._lambda_schedule_state(),
            "lambda_current": self.current_ramusa_lambda,
            "lambda_history": self.lambda_history,
            "stage_history": self.stage_history,
            "history": {
                "train_losses_all": self.train_losses_all,
                "train_acc_all": self.train_acc_all,
                "val_losses_all": self.val_losses_all,
                "val_acc_all": self.val_acc_all,
                "train_losses_task": self.train_losses_task,
                "train_acc_task": self.train_acc_task,
                "val_losses_task": self.val_losses_task,
                "val_acc_task": self.val_acc_task,
                "learning_rate_array": self.learning_rate_array,
            },
            "checkpoint_metrics": {
                "best_accuracy_all_threshold": self.ckpt.best_accuracy_all_threshold,
                "best_accuracy_task_thresholds": self.ckpt.best_accuracy_task_thresholds,
                "opt_accuracy_all_threshold": self.ckpt.opt_accuracy_all_threshold,
                "opt_accuracy_task_thresholds": self.ckpt.opt_accuracy_task_thresholds,
                "opt_accuracy_avg_threshold": self.ckpt.opt_accuracy_avg_threshold,
            },
            "adaptive_tau_state": {
                "previous_tau": self.adaptive_tau_manager.previous_tau,
                "previous_r": self.adaptive_tau_manager.previous_r,
            } if self.adaptive_tau_manager is not None else None,
        }, self.training_state_path)

    def resume_from_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        saved_schedule = checkpoint.get("lambda_schedule")
        if saved_schedule is not None and saved_schedule != self._lambda_schedule_state():
            raise ValueError("RAMUSA lambda schedule does not match the checkpoint")
        self.global_epoch = int(checkpoint.get("global_epoch", 0))
        self.current_stage = min(
            self.num_stages,
            self.global_epoch // self.epochs_per_stage + 1,
        )
        if "current_stage" in checkpoint:
            self.current_stage = int(checkpoint["current_stage"])
        self.effective_rank = int(checkpoint["r"])
        self.tau = float(checkpoint["tau"])
        self.current_ramusa_lambda = get_ramusa_lambda(
            self.lambda_schedule_type, self.global_epoch, self.lambda_max,
            self.warmup_epochs, self.delay_epochs,
        )
        self.lambda_history = list(checkpoint.get("lambda_history", []))
        self.stage_history = list(checkpoint.get("stage_history", []))
        history = checkpoint.get("history", {})
        for name in (
            "train_losses_all", "train_acc_all", "val_losses_all", "val_acc_all",
            "train_losses_task", "train_acc_task", "val_losses_task", "val_acc_task",
            "learning_rate_array",
        ):
            if name in history:
                setattr(self, name, history[name])
        checkpoint_metrics = checkpoint.get("checkpoint_metrics", {})
        for name, value in checkpoint_metrics.items():
            if hasattr(self.ckpt, name):
                setattr(self.ckpt, name, value)
        adaptive_tau_state = checkpoint.get("adaptive_tau_state")
        if adaptive_tau_state is not None and self.adaptive_tau_manager is not None:
            self.adaptive_tau_manager.previous_tau = adaptive_tau_state.get("previous_tau")
            self.adaptive_tau_manager.previous_r = adaptive_tau_state.get("previous_r")

        previous_checkpoint_dir = os.path.dirname(os.path.abspath(path))
        for tag in ("best", "opt"):
            previous_model_path = os.path.join(
                previous_checkpoint_dir,
                os.path.basename(self.model_checkpoint_path).replace(".pth", f"_{tag}.pth"),
            )
            current_model_path = self.model_checkpoint_path.replace(".pth", f"_{tag}.pth")
            if os.path.exists(previous_model_path):
                shutil.copy2(previous_model_path, current_model_path)

    def _finish_stage(self, previous_rank: int):
        with torch.no_grad():
            matrix = self.model.build_ramusa_parameter_matrix()
            singular_values = self.model.get_ramusa_singular_values()
            rank_limit = singular_values.numel()

        if self.adaptive_tau_enabled and self.adaptive_tau_manager is not None:
            # Use adaptive tau selection
            validation_metrics = self._compute_stage_boundary_metrics()
            candidate_validator = (
                self._create_validation_probe_callback() 
                if self.adaptive_tau_manager.validation_probe_enabled 
                else None
            )
            new_tau, new_r = self.adaptive_tau_manager.update_tau_between_stages(
                model=self.model,
                singular_values=singular_values,
                stage=self.current_stage - 1,  # 0-indexed
                validation_metrics_base=validation_metrics,
                candidate_validator=candidate_validator,
            )
            if new_tau is not None:
                self.tau = new_tau
            self.effective_rank = new_r
        else:
            # Use original fixed-tau approach
            next_rank = update_effective_rank(matrix, self.tau)
            self.effective_rank = max(0, min(next_rank, rank_limit))

        values = [float(value) for value in singular_values.cpu()]
        self.stage_history.append({
            "stage": self.current_stage, "previous_r": previous_rank,
            "updated_r": self.effective_rank, "singular_values": values,
            "tau": self.tau,
        })
        logging.info(
            "RAMUSA stage %d completed | singular_values=%s | tau=%g | previous_r=%d | updated_r=%d",
            self.current_stage, values, self.tau, previous_rank, self.effective_rank,
        )
        self._write_stage_state()

    def _epoch_report_line(self, epoch, phase, stats):
        try:
            import mlflow
            import mlflow_utils

            if mlflow.active_run() is not None:
                learning_rate = None
                if phase == "val":
                    learning_rate = float(self.optimizer.param_groups[0]["lr"])
                mlflow_utils.log_epoch_stats(
                    epoch, phase, stats, self.task_array,
                    learning_rate=learning_rate,
                )
        except Exception as error:
            logging.warning("Could not log MLflow epoch metrics: %s", error)
        line = super()._epoch_report_line(epoch, phase, stats)
        details = getattr(self, "_epoch_train_details", getattr(self, "_last_batch_details", {}))
        return "%s | stage=%d | r=%d | lambda=%.6g | mtl_loss=%.4f | rank_penalty=%.4f | weighted_rank_penalty=%.4f | total_loss=%.4f | singular_values=%s" % (
            line, self.current_stage, self.effective_rank,
            self.current_ramusa_lambda,
            details.get("mtl_loss", stats.avg_loss_all),
            details.get("rank_penalty", 0.0), details.get("weighted_rank_penalty", 0.0),
            details.get("total_loss", stats.avg_loss_all),
            details.get("singular_values", []),
        )

    def _compute_stage_boundary_metrics(self) -> Dict[str, float]:
        """Compute validation metrics at the current stage boundary.
        
        Returns:
            metrics: Dictionary with keys 'ks', 'si', 'er' containing task accuracies.
        """
        self.model.eval()
        task_accuracies = {0: 0.0, 1: 0.0, 2: 0.0}  # ks, si, er
        task_samples = {0: 0, 1: 0, 2: 0}
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                input_seq, labels_list = self._unpack_batch(batch)
                outputs = self.model(input_seq=input_seq)
                
                for task_index in range(self.num_tasks):
                    correct, samples = masked_accuracy(
                        outputs.prediction[task_index], 
                        labels_list[task_index], 
                        self.ignore_index
                    )
                    task_accuracies[task_index] += correct
                    task_samples[task_index] += samples
        
        # Compute per-task accuracies
        metrics = {}
        task_names = ["ks", "si", "er"]
        for idx, name in enumerate(task_names):
            if task_samples[idx] > 0:
                metrics[name] = task_accuracies[idx] / task_samples[idx]
            else:
                metrics[name] = 0.0
        
        return metrics

    def _save_stage_boundary_state(self):
        """Save model, optimizer, and scheduler state at stage boundary."""
        self._stage_boundary_state = {
            "model_state": copy.deepcopy(self.model.state_dict()),
            "optimizer_state": copy.deepcopy(self.optimizer.state_dict()),
            "scheduler_state": copy.deepcopy(self.scheduler.state_dict()),
        }

    def _restore_stage_boundary_state(self):
        """Restore model, optimizer, and scheduler to stage boundary state."""
        if self._stage_boundary_state is None:
            return
        
        self.model.load_state_dict(self._stage_boundary_state["model_state"])
        self.optimizer.load_state_dict(self._stage_boundary_state["optimizer_state"])
        self.scheduler.load_state_dict(self._stage_boundary_state["scheduler_state"])

    def _run_validation_probe(
        self, r_candidate: int, probe_epochs: int = 1
    ) -> Dict[str, float]:
        """Run a short validation probe with a specific rank.
        
        Args:
            r_candidate: Rank to probe.
            probe_epochs: Number of epochs to train.
            
        Returns:
            metrics: Dictionary with task accuracies after probing.
        """
        original_r = self.effective_rank
        self.effective_rank = r_candidate
        
        self.model.train()
        for _ in range(probe_epochs):
            _ = self._process_data_loader(self.train_dataloader, train_mode=True)
        
        # Evaluate on validation set
        metrics = self._compute_stage_boundary_metrics()
        
        # Restore original state
        self.effective_rank = original_r
        self._restore_stage_boundary_state()
        
        return metrics

    def _create_validation_probe_callback(self):
        """Create a callback function for validation probe evaluation.
        
        Returns:
            callback: Function that takes candidates and returns best (tau, r, improvement).
        """
        # Save stage boundary state first
        self._save_stage_boundary_state()
        base_metrics = self._compute_stage_boundary_metrics()
        
        def probe_candidates(
            candidates: list,
            base_metrics: Dict[str, float],
        ) -> Tuple[float, int, float]:
            """Probe each unique candidate and select the best one.
            
            Args:
                candidates: List of (tau, r) tuples.
                base_metrics: Baseline validation metrics.
                
            Returns:
                (best_tau, best_r, improvement): Best candidate and its improvement score.
            """
            # Deduplicate candidates by rank
            unique_ranks = {}
            for tau, r in candidates:
                if r not in unique_ranks:
                    unique_ranks[r] = tau
            
            best_score = -float('inf')
            best_tau = candidates[0][0]
            best_r = candidates[0][1]
            
            probe_epochs = self.adaptive_tau_manager.validation_probe_epochs
            
            logging.info(f"Running validation probes for {len(unique_ranks)} unique ranks")
            
            for r, tau in unique_ranks.items():
                logging.info(f"Probing rank r={r}...")
                
                # Restore to clean state
                self._restore_stage_boundary_state()
                
                # Run probe
                probe_metrics = self._run_validation_probe(r, probe_epochs=probe_epochs)
                
                # Compute improvement score
                improvement, deltas = self.adaptive_tau_manager.compute_normalized_improvement(
                    base_metrics, probe_metrics
                )
                
                # Add negative transfer penalty
                penalty = self.adaptive_tau_manager.compute_negative_transfer_penalty(deltas)
                score = improvement + penalty
                
                logging.info(
                    f"Rank r={r}: improvement={improvement:.6f}, "
                    f"penalty={penalty:.6f}, score={score:.6f}"
                )
                
                if score > best_score:
                    best_score = score
                    best_tau = tau
                    best_r = r
            
            # Final restoration
            self._restore_stage_boundary_state()
            
            logging.info(f"Selected rank r={best_r} with score={best_score:.6f}")
            return (best_tau, best_r, best_score)
        
        return probe_candidates

    def train(self):
        logging.info("RAMUSA neural adaptation start | stages=%d | epochs_per_stage=%d | tau=%g", self.num_stages, self.epochs_per_stage, self.tau)
        while self.global_epoch < self.num_epochs:
            self.current_stage = min(
                self.num_stages,
                self.global_epoch // self.epochs_per_stage + 1,
            )
            epoch_in_stage = self.global_epoch % self.epochs_per_stage + 1
            self.current_ramusa_lambda = get_ramusa_lambda(
                self.lambda_schedule_type, self.global_epoch, self.lambda_max,
                self.warmup_epochs, self.delay_epochs,
            )
            epoch = self.global_epoch + 1
            logging.info(
                "ramusa_epoch_start | global_epoch=%d | stage=%d | epoch_in_stage=%d | lambda_schedule=%s | lambda=%.6g | lambda_max=%.6g | warmup_epochs=%d | delay_epochs=%d | r=%d | tau=%.6g",
                epoch, self.current_stage, epoch_in_stage, self.lambda_schedule_type,
                self.current_ramusa_lambda, self.lambda_max, self.warmup_epochs,
                self.delay_epochs, self.effective_rank, self.tau,
            )
            self._epoch_batch_details = []
            train_stats = self._process_data_loader(self.train_dataloader, train_mode=True)
            batch_details = self._epoch_batch_details
            self._epoch_train_details = {
                key: sum(details[key] for details in batch_details) / len(batch_details)
                for key in ("mtl_loss", "rank_penalty", "weighted_rank_penalty", "total_loss")
            } if batch_details else dict(getattr(self, "_last_batch_details", {}))
            if batch_details:
                self._epoch_train_details["singular_values"] = batch_details[-1]["singular_values"]
            val_stats = self._process_data_loader(self.val_dataloader, train_mode=False)
            self.train_losses_all.append(train_stats.avg_loss_all)
            self.train_acc_all.append(train_stats.accuracy_all)
            self.val_losses_all.append(val_stats.avg_loss_all)
            self.val_acc_all.append(val_stats.accuracy_all)
            for task_index in range(self.num_tasks):
                self.train_losses_task[task_index].append(train_stats.avg_loss_task[task_index])
                self.train_acc_task[task_index].append(train_stats.accuracy_task[task_index])
                self.val_losses_task[task_index].append(val_stats.avg_loss_task[task_index])
                self.val_acc_task[task_index].append(val_stats.accuracy_task[task_index])
            self.metrics_writer.write_metrics_all(self._epoch_report_line(epoch, "train", train_stats))
            self.metrics_writer.write_metrics_all(self._epoch_report_line(epoch, "val", val_stats))
            self.learning_rate_array.append(float(self.optimizer.param_groups[0]["lr"]))
            self.ckpt.maybe_save_best_and_opt(val_stats.accuracy_all, val_stats.accuracy_task, self.num_tasks)
            self.scheduler.step(val_stats.avg_loss_all)

            details = self._epoch_train_details
            self.lambda_history.append({
                "global_epoch": epoch,
                "stage": self.current_stage,
                "epoch_in_stage": epoch_in_stage,
                "lambda": self.current_ramusa_lambda,
                "ramusa_penalty": details.get("rank_penalty", 0.0),
                "weighted_ramusa_penalty": details.get("weighted_rank_penalty", 0.0),
                "multitask_loss": details.get("mtl_loss", train_stats.avg_loss_all),
                "total_loss": details.get("total_loss", train_stats.avg_loss_all),
                "r": self.effective_rank,
                "tau": self.tau,
            })
            logging.info(
                "ramusa_epoch | global_epoch=%d | stage=%d | epoch_in_stage=%d | lambda=%.6g | lambda_max=%.6g | mtl_loss=%.6f | ramusa_penalty=%.6f | weighted_ramusa_penalty=%.6f | total_loss=%.6f | r=%d | tau=%.6g | singular_values=%s",
                epoch, self.current_stage, epoch_in_stage, self.current_ramusa_lambda,
                self.lambda_max, details.get("mtl_loss", train_stats.avg_loss_all),
                details.get("rank_penalty", 0.0), details.get("weighted_rank_penalty", 0.0),
                details.get("total_loss", train_stats.avg_loss_all), self.effective_rank, self.tau,
                details.get("singular_values", []),
            )

            self.global_epoch += 1
            if epoch_in_stage == self.epochs_per_stage:
                self._finish_stage(self.stage_history[-1]["updated_r"] if self.stage_history else 0)
            self._write_stage_state()
            self._save_training_state()
        self.plot_metrics()
