"""Stage-wise trainer for the neural RAMUSA adaptation."""

import json
import logging
import os
from typing import Dict

import torch
import torch.nn as nn

from trainer.trainer_model import MultiTasksModelTrainer
from trainer.trainer_utils import BatchStats, masked_accuracy, masked_ce_loss
from improvements.lowrank.utils.ramusa import update_effective_rank


class MultiTasksModelTrainerRAMUSA(MultiTasksModelTrainer):
    def __init__(self, *args, ramusa_cfg: dict, **kwargs):
        super().__init__(*args, **kwargs)
        if self.task_type != "ks_si_er":
            raise ValueError("RAMUSA supports only task_type='ks_si_er'.")
        self.ramusa_enabled = bool(ramusa_cfg.get("enabled", True))
        self.ramusa_lambda = float(ramusa_cfg.get("lambda", 0.01))
        self.tau = float(ramusa_cfg.get("tau", 1.0))
        self.num_stages = int(ramusa_cfg.get("num_stages", 1))
        self.epochs_per_stage = int(ramusa_cfg.get("epochs_per_stage", self.num_epochs))
        if self.ramusa_lambda < 0 or self.tau < 0:
            raise ValueError("RAMUSA lambda and tau must be non-negative")
        if self.num_stages <= 0 or self.epochs_per_stage <= 0:
            raise ValueError("RAMUSA stages and epochs_per_stage must be positive")
        if self.num_stages * self.epochs_per_stage != self.num_epochs:
            raise ValueError("training.num_epochs must equal num_stages * epochs_per_stage")

        self.current_stage = 1
        self.effective_rank = 0
        self.stage_history = []
        self.training_loss_fn = nn.CrossEntropyLoss(
            label_smoothing=float(self.training_cfg.get("label_smoothing", 0.0))
        )
        self.validation_loss_fn = nn.CrossEntropyLoss()
        self.stage_state_path = os.path.join(self.results_dir, "ramusa_state.json")

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
            loss_all = loss_all + self.ramusa_lambda * rank_penalty
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
            "total_loss": float(loss_all.detach().item()),
            "singular_values": [float(value) for value in singular_values.detach().cpu()],
        }
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
            "tau": self.tau, "lambda": self.ramusa_lambda,
            "history": self.stage_history,
        }
        with open(self.stage_state_path, "w") as state_file:
            json.dump(state, state_file, indent=2)

    def resume_from_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.current_stage = int(checkpoint["stage"])
        self.effective_rank = int(checkpoint["r"])
        if float(checkpoint["tau"]) != self.tau or float(checkpoint["lambda"]) != self.ramusa_lambda:
            raise ValueError("RAMUSA tau/lambda do not match the checkpoint")

    def _finish_stage(self, previous_rank: int):
        with torch.no_grad():
            matrix = self.model.build_ramusa_parameter_matrix()
            singular_values = self.model.get_ramusa_singular_values()
            rank_limit = singular_values.numel()
            next_rank = update_effective_rank(matrix, self.tau)
        self.effective_rank = max(0, min(next_rank, rank_limit))
        values = [float(value) for value in singular_values.cpu()]
        self.stage_history.append({
            "stage": self.current_stage, "previous_r": previous_rank,
            "updated_r": self.effective_rank, "singular_values": values,
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
        details = getattr(self, "_last_batch_details", {})
        return "%s | stage=%d | r=%d | mtl_loss=%.4f | rank_penalty=%.4f | total_loss=%.4f | singular_values=%s" % (
            line, self.current_stage, self.effective_rank,
            details.get("mtl_loss", stats.avg_loss_all),
            details.get("rank_penalty", 0.0), details.get("total_loss", stats.avg_loss_all),
            details.get("singular_values", []),
        )

    def train(self):
        logging.info("RAMUSA neural adaptation start | stages=%d | epochs_per_stage=%d | tau=%g", self.num_stages, self.epochs_per_stage, self.tau)
        for stage in range(1, self.num_stages + 1):
            self.current_stage = stage
            for _ in range(self.epochs_per_stage):
                epoch = len(self.train_losses_all) + 1
                train_stats = self._process_data_loader(self.train_dataloader, train_mode=True)
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
            self._finish_stage(self.stage_history[-1]["updated_r"] if self.stage_history else 0)
        self.plot_metrics()
