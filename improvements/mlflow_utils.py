"""
Shared MLflow logging helpers for wavCSE improvements runs.

Reusable by any owner folder under improvements/ (base, taskrelation,
clustering, lowrank, decomposition) without requiring changes to downstream/.
Every helper here only reads public attributes/return values already exposed
by MultiTasksModelTrainer and MultiTasksModelEvaluator (downstream/trainer,
downstream/evaluator) -- no subclassing.

Author: Kevin Sanjula
"""

from datetime import datetime

import mlflow


def _flatten_dict(d, parent_key="", sep="."):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def setup_mlflow(cfg):
    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri")
    experiment_name = mlflow_cfg.get("experiment_name", "default")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_config_params(cfg):
    flat = _flatten_dict({k: v for k, v in cfg.items() if k != "mlflow"})
    mlflow.log_params({k: v for k, v in flat.items() if v is not None})


def build_run_name(category, model, task_type, suffix=None):
    """Standard run-name pattern shared by every owner folder:
    {category}_{model}_{task_type}_{timestamp}[_{suffix}].

    `category` is one of base/base-improvement/taskrelation/lowrank/
    clustering/decomposition; `model` is the architecture variant within it
    (e.g. "gbc", "original"). Keeps run names self-identifying even outside
    their experiment, and centralizes the timestamp format so every category
    sorts/reads the same way.
    """
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    name = f"{category}_{model}_{task_type}_{timestamp}"
    if suffix:
        name = f"{name}_{suffix}"
    return name


def set_standard_tags(category, model, cfg, extra_tags=None):
    """Sets the coarse cross-experiment grouping tags every run should carry.

    Unlike params (log_config_params), MLflow tags are what
    MlflowClient.search_runs()'s filter_string can combine with a list of
    experiment_ids -- so these are what let a report script slice runs by
    category/model/pooling *across* every wavcse-*/taskrelation-*/lowrank-*/
    clustering-*/decomposition-* experiment at once, not just within one.
    """
    pooling_cfg = cfg.get("pooling", {})
    tags = {
        "category": category,
        "model": model,
        "pooling_frame": pooling_cfg.get("frame_pooling_type"),
        "pooling_layer": pooling_cfg.get("layer_pooling_type"),
    }
    if extra_tags:
        tags.update(extra_tags)
    mlflow.set_tags({k: v for k, v in tags.items() if v is not None})


def make_live_trainer(base_trainer_cls):
    """Factory returning a subclass of base_trainer_cls that pushes each
    epoch's train/val stats to MLflow immediately via log_epoch_stats,
    instead of waiting for the whole run to finish (see log_epoch_stats'
    docstring for why). Overrides only _epoch_report_line -- the smallest
    hook point in trainer_model.py's train() loop that already receives the
    full per-phase EpochStats.

    Centralizes the pattern originally defined inline as
    improvements/base/run_base.py's _LiveMlflowTrainer, so every owner
    folder (base, taskrelation, and future lowrank/clustering/decomposition)
    can reuse it against their own trainer class (including TSM/PMR's
    alternating-minimization trainers) instead of redefining it.

    Coupling risk: relies on _epoch_report_line's private signature
    (epoch, phase, stats) staying stable across every trainer it's applied to.
    """

    class _LiveMlflowTrainer(base_trainer_cls):
        def _epoch_report_line(self, epoch, phase, stats):
            learning_rate = None
            if phase == "val":
                learning_rate = float(self.optimizer.param_groups[0]["lr"])
            log_epoch_stats(epoch, phase, stats, self.task_array, learning_rate=learning_rate)
            return super()._epoch_report_line(epoch, phase, stats)

    return _LiveMlflowTrainer


def log_trainer_history(trainer):
    """Replays a FINISHED trainer's full per-epoch history into step-indexed MLflow
    metrics, all at once. Simple, but gives zero visibility if the process dies
    mid-training -- prefer log_epoch_stats() (below) for long/unattended runs."""
    task_array = trainer.task_array

    for epoch in range(trainer.num_epochs):
        metrics = {
            "train_loss_all": trainer.train_losses_all[epoch],
            "train_acc_all": trainer.train_acc_all[epoch],
            "val_loss_all": trainer.val_losses_all[epoch],
            "val_acc_all": trainer.val_acc_all[epoch],
            "learning_rate": trainer.learning_rate_array[epoch],
        }
        for t, task in enumerate(task_array):
            metrics[f"train_{task}_loss"] = trainer.train_losses_task[t][epoch]
            metrics[f"train_{task}_acc"] = trainer.train_acc_task[t][epoch]
            metrics[f"val_{task}_loss"] = trainer.val_losses_task[t][epoch]
            metrics[f"val_{task}_acc"] = trainer.val_acc_task[t][epoch]

        mlflow.log_metrics(metrics, step=epoch + 1)


def log_epoch_stats(epoch, phase, stats, task_array, learning_rate=None):
    """Logs one phase's ("train" or "val") EpochStats for one epoch immediately.

    Meant to be called live, during training, from a hook into one epoch at a
    time -- so a run killed mid-training still leaves a partial, inspectable
    metric curve on the tracking server, instead of nothing at all (see
    log_trainer_history's caveat above). See make_live_trainer() above for
    the trainer-side hook this pairs with.
    """
    metrics = {
        f"{phase}_loss_all": stats.avg_loss_all,
        f"{phase}_acc_all": stats.accuracy_all,
    }
    for t, task in enumerate(task_array):
        metrics[f"{phase}_{task}_loss"] = stats.avg_loss_task[t]
        metrics[f"{phase}_{task}_acc"] = stats.accuracy_task[t]
    if learning_rate is not None:
        metrics["learning_rate"] = learning_rate

    mlflow.log_metrics(metrics, step=epoch)


def log_eval_stats(stats, tag, task_array):
    """Logs an EvalStats instance (from MultiTasksModelEvaluator.write_metrics()) under a checkpoint tag."""
    metrics = {
        f"test_{tag}_loss_all": stats.avg_loss_all,
        f"test_{tag}_acc_all": stats.accuracy_all,
    }
    for t, task in enumerate(task_array):
        metrics[f"test_{tag}_{task}_loss"] = stats.avg_loss_task[t]
        metrics[f"test_{tag}_{task}_acc"] = stats.accuracy_task[t]

    mlflow.log_metrics(metrics)
