"""Observational per-task gradient diagnostics for matched MTL studies.

The wrapper returned by :func:`make_gradient_diagnostic_trainer` samples
per-task gradients from the same forward pass used for training. It uses
``torch.autograd.grad`` without writing optimizer ``.grad`` buffers, then
executes the original baseline/MTRL batch objective and optimizer step.
"""

import json
import logging
import math
import os
import statistics
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import mlflow
import torch

from downstream.trainer.trainer_utils import (
    BatchStats,
    masked_accuracy,
    masked_ce_loss,
)


def _summary(values: Iterable[float]) -> Dict[str, Optional[float]]:
    values = list(values)
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _phase(progress: float) -> str:
    if progress < (1.0 / 3.0):
        return "early"
    if progress < (2.0 / 3.0):
        return "middle"
    return "late"


def _pair_key(task_a: str, task_b: str) -> str:
    return f"{task_a}_{task_b}"


def _summarize_records(records: Sequence[Dict], task_array: Sequence[str]) -> Dict:
    phases = ("all", "early", "middle", "late")
    pairs = list(combinations(task_array, 2))
    summary = {}

    for phase_name in phases:
        selected = records if phase_name == "all" else [
            record for record in records if record["phase"] == phase_name
        ]
        phase_summary = {
            "sample_count": len(selected),
            "task_norms": {},
            "pairwise_cosines": {},
        }

        mean_norms = []
        for task in task_array:
            norm_summary = _summary(
                record["gradient_norms"][task] for record in selected
            )
            phase_summary["task_norms"][task] = norm_summary
            if norm_summary["mean"] is not None:
                mean_norms.append(norm_summary["mean"])

        positive_norms = [value for value in mean_norms if value > 0.0]
        phase_summary["max_min_mean_norm_ratio"] = (
            max(positive_norms) / min(positive_norms)
            if len(positive_norms) >= 2 else None
        )

        for task_a, task_b in pairs:
            key = _pair_key(task_a, task_b)
            values = [record["pairwise_cosines"][key] for record in selected]
            pair_summary = _summary(values)
            pair_summary["negative_count"] = sum(value < 0.0 for value in values)
            pair_summary["negative_frequency"] = (
                pair_summary["negative_count"] / len(values) if values else None
            )
            phase_summary["pairwise_cosines"][key] = pair_summary

        summary[phase_name] = phase_summary

    return summary


def make_gradient_diagnostic_trainer(base_trainer_cls):
    """Wrap a baseline or MTRL trainer with sampled gradient recording."""

    class GradientDiagnosticTrainer(base_trainer_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            config = self.training_cfg.get("gradient_diagnostics", {})
            self._gradient_diagnostics_enabled = bool(config.get("enabled", False))
            self._gradient_sample_interval = int(
                config.get("sample_interval_steps", 20)
            )
            if self._gradient_sample_interval <= 0:
                raise ValueError("gradient diagnostic sample interval must be positive")

            exclude_prefixes = tuple(
                config.get("exclude_parameter_prefixes", ["classifiers."])
            )
            self._gradient_parameter_items = [
                (name, parameter)
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
                and not name.startswith(exclude_prefixes)
            ]
            if not self._gradient_parameter_items:
                raise ValueError("gradient diagnostics found no shared parameters")

            self._gradient_total_steps = self.num_epochs * len(self.train_dataloader)
            self._gradient_step = 0
            self._gradient_records: List[Dict] = []
            self._gradient_skipped_missing_tasks = 0
            self._gradient_artifact_path = os.path.join(
                self.results_dir, "gradient_diagnostics.json"
            )
            logging.info(
                "gradient_diagnostics | enabled=%s | interval=%d | total_steps=%d | shared_parameters=%d",
                self._gradient_diagnostics_enabled,
                self._gradient_sample_interval,
                self._gradient_total_steps,
                sum(parameter.numel() for _, parameter in self._gradient_parameter_items),
            )

        def _process_batch(self, batch, train_mode: bool):
            if not self._gradient_diagnostics_enabled:
                return super()._process_batch(batch, train_mode)

            input_seq, labels_list = self._unpack_batch(batch)
            if train_mode:
                self.optimizer.zero_grad(set_to_none=True)

            outputs = self.model(input_seq=input_seq)
            logits_tuple = outputs.logits
            pred_tuple = outputs.prediction
            loss_weight = 1.0 / float(self.num_tasks)

            loss_all = torch.tensor(0.0, device=self.device)
            loss_task = {}
            batch_task = {}
            correct_task = {}
            samples_task = {}
            valid_count_task = {}
            per_task_losses = {}

            for task_index in range(self.num_tasks):
                loss_value, valid = masked_ce_loss(
                    logits=logits_tuple[task_index],
                    labels=labels_list[task_index],
                    loss_fn=self.loss_fn,
                    ignore_index=self.ignore_index,
                )
                per_task_losses[task_index] = loss_value
                valid_count_task[task_index] = valid

            if train_mode:
                self._record_task_gradient_diagnostics(
                    per_task_losses,
                    valid_count_task,
                )

            for task_index in range(self.num_tasks):
                raw_loss = per_task_losses[task_index]
                weighted_loss, loss_all, active = self._calculate_total_loss(
                    raw_loss,
                    loss_all,
                    loss_weight,
                )
                loss_task[task_index] = (
                    float(weighted_loss.item()) if raw_loss is not None else 0.0
                )
                batch_task[task_index] = int(active)

            l1_reg = torch.tensor(0.0, device=self.device)
            l2_reg = torch.tensor(0.0, device=self.device)
            for parameter in self.model.parameters():
                l1_reg = l1_reg + parameter.abs().sum()
                l2_reg = l2_reg + parameter.square().sum()
            loss_all = (
                loss_all
                + self.l1_lambda * l1_reg
                + self.l2_lambda * l2_reg
            )

            if (
                hasattr(self, "mtrl_warmup_epochs")
                and self.current_epoch >= self.mtrl_warmup_epochs
            ):
                loss_all = loss_all + self.model.get_mtrl_regularizer_loss()

            if train_mode:
                loss_all.backward()
                self.optimizer.step()

            correct_all = 0
            samples_all = 0
            for task_index in range(self.num_tasks):
                correct, samples = masked_accuracy(
                    pred_tuple[task_index],
                    labels_list[task_index],
                    ignore_index=self.ignore_index,
                )
                correct_task[task_index] = correct
                samples_task[task_index] = samples
                correct_all += correct
                samples_all += samples

            self._save_debug_counts(
                "train" if train_mode else "val",
                valid_count_task,
            )
            return BatchStats(
                loss_all=float(loss_all.item()),
                batch_all=1,
                loss_task=loss_task,
                batch_task=batch_task,
                correct_task=correct_task,
                samples_task=samples_task,
                correct_all=correct_all,
                samples_all=samples_all,
                valid_count_task=valid_count_task,
            )

        def _should_sample_gradient_step(self, step: int) -> bool:
            return (
                step == 1
                or step == self._gradient_total_steps
                or step % self._gradient_sample_interval == 0
            )

        def _record_task_gradient_diagnostics(
            self,
            per_task_losses: Dict[int, Optional[torch.Tensor]],
            valid_count_task: Dict[int, int],
        ) -> None:
            if not self._gradient_diagnostics_enabled:
                return

            self._gradient_step += 1
            if not self._should_sample_gradient_step(self._gradient_step):
                return
            if any(per_task_losses.get(index) is None for index in range(self.num_tasks)):
                self._gradient_skipped_missing_tasks += 1
                return

            parameters = [parameter for _, parameter in self._gradient_parameter_items]
            gradients: List[Tuple[Optional[torch.Tensor], ...]] = []
            squared_norms = []

            for task_index in range(self.num_tasks):
                task_gradients = torch.autograd.grad(
                    per_task_losses[task_index],
                    parameters,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )
                squared_norm = torch.zeros((), device=self.device)
                for gradient in task_gradients:
                    if gradient is not None:
                        squared_norm = squared_norm + gradient.detach().square().sum()
                gradients.append(task_gradients)
                squared_norms.append(squared_norm)

            norms = [math.sqrt(max(float(value.item()), 0.0)) for value in squared_norms]
            if any(not math.isfinite(value) or value == 0.0 for value in norms):
                raise RuntimeError(
                    f"non-finite or zero task-gradient norm at step {self._gradient_step}: {norms}"
                )

            pairwise_cosines = {}
            for task_a, task_b in combinations(range(self.num_tasks), 2):
                dot = torch.zeros((), device=self.device)
                for gradient_a, gradient_b in zip(
                    gradients[task_a], gradients[task_b]
                ):
                    if gradient_a is not None and gradient_b is not None:
                        dot = dot + (
                            gradient_a.detach() * gradient_b.detach()
                        ).sum()
                pairwise_cosines[_pair_key(
                    self.task_array[task_a], self.task_array[task_b]
                )] = float(dot.item()) / (norms[task_a] * norms[task_b])

            progress = (
                (self._gradient_step - 1) / (self._gradient_total_steps - 1)
                if self._gradient_total_steps > 1 else 1.0
            )
            record = {
                "step": self._gradient_step,
                "progress": progress,
                "phase": _phase(progress),
                "valid_examples": {
                    self.task_array[index]: int(valid_count_task[index])
                    for index in range(self.num_tasks)
                },
                "gradient_norms": {
                    self.task_array[index]: norms[index]
                    for index in range(self.num_tasks)
                },
                "pairwise_cosines": pairwise_cosines,
            }
            self._gradient_records.append(record)

            if mlflow.active_run() is not None:
                metrics = {
                    f"gradient_norm_{task}": value
                    for task, value in record["gradient_norms"].items()
                }
                for pair, cosine in pairwise_cosines.items():
                    metrics[f"gradient_cosine_{pair}"] = cosine
                    metrics[f"gradient_conflict_{pair}"] = float(cosine < 0.0)
                mlflow.log_metrics(metrics, step=self._gradient_step)

            del gradients

        def _save_gradient_diagnostics(self) -> None:
            if not self._gradient_diagnostics_enabled:
                return
            payload = {
                "task_array": self.task_array,
                "sample_interval_steps": self._gradient_sample_interval,
                "total_training_steps": self._gradient_total_steps,
                "sampled_steps": len(self._gradient_records),
                "skipped_sample_steps_missing_tasks": self._gradient_skipped_missing_tasks,
                "shared_parameter_names": [
                    name for name, _ in self._gradient_parameter_items
                ],
                "shared_parameter_count": sum(
                    parameter.numel()
                    for _, parameter in self._gradient_parameter_items
                ),
                "records": self._gradient_records,
                "summary": _summarize_records(
                    self._gradient_records, self.task_array
                ),
            }
            with open(self._gradient_artifact_path, "w") as artifact:
                json.dump(payload, artifact, indent=2)
            if mlflow.active_run() is not None:
                mlflow.log_artifact(
                    self._gradient_artifact_path,
                    artifact_path="gradient_diagnostics",
                )
            logging.info(
                "gradient diagnostics saved | samples=%d | skipped_missing=%d | path=%s",
                len(self._gradient_records),
                self._gradient_skipped_missing_tasks,
                self._gradient_artifact_path,
            )

        def train(self):
            try:
                return super().train()
            finally:
                self._save_gradient_diagnostics()

    GradientDiagnosticTrainer.__name__ = (
        f"GradientDiagnostic{base_trainer_cls.__name__}"
    )
    return GradientDiagnosticTrainer
