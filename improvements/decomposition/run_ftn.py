"""Train and evaluate the three-task FTN-inspired wavCSE model.

Usage:
    python run_ftn.py --task_type ks_si_er --config configs/ftn_config.yml \
        --device_index 0
"""

import argparse
import os
import sys
from datetime import datetime

from dotenv import load_dotenv


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_IMPROVEMENTS_DIR = os.path.dirname(_THIS_DIR)
_REPO_ROOT = os.path.dirname(_IMPROVEMENTS_DIR)
_DOWNSTREAM_DIR = os.path.join(_REPO_ROOT, "downstream")

for path in (_REPO_ROOT, _DOWNSTREAM_DIR, _IMPROVEMENTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import mlflow
import mlflow_utils
from seed_utils import set_seed

from dataset.load_embedding import LoadEmbedding
from improvements.decomposition.models.ftn_model import DownstreamMultiTaskModelFTN
from improvements.decomposition.trainers.ftn_trainer import MultiTasksModelTrainerFTN
from evaluator.evaluator_model import MultiTasksModelEvaluator
from utils.load_config import load_config
from utils.parse_transformer_layers import parse_transformer_layers
from utils.setup_device import set_device
from utils.setup_logging import setup_logging


class _LiveMlflowTrainer(MultiTasksModelTrainerFTN):
    """Add MLflow diagnostics and configurable clipping to the standard trainer."""

    def _epoch_report_line(self, epoch, phase, stats):
        learning_rate = None
        if phase == "val":
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
        mlflow_utils.log_epoch_stats(
            epoch, phase, stats, self.task_array, learning_rate=learning_rate
        )

        balanced_accuracy = sum(stats.accuracy_task.values()) / len(
            stats.accuracy_task
        )
        diagnostics = {f"{phase}_balanced_task_acc": balanced_accuracy}

        if phase == "train":
            gradient_stats = self.consume_gradient_norm_stats()
            if gradient_stats is not None:
                diagnostics.update(
                    {
                        "train_grad_norm_mean_before_clip": gradient_stats["mean"],
                        "train_grad_norm_max_before_clip": gradient_stats["max"],
                    }
                )

        if phase == "val":
            diagnostics["shared_adapter_identity_distance"] = (
                self.model.get_shared_adapter_identity_distance()
            )
            diagnostics.update(
                {
                    f"delta_norm_{task}": norm
                    for task, norm in self.model.get_delta_norms().items()
                }
            )

        mlflow.log_metrics(diagnostics, step=epoch)
        return super()._epoch_report_line(epoch, phase, stats)


def _owner_relative_path(path: str) -> str:
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    return os.path.join(_THIS_DIR, expanded)


def _config_path(path: str) -> str:
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    cwd_candidate = os.path.abspath(expanded)
    if os.path.isfile(cwd_candidate):
        return cwd_candidate
    return os.path.join(_THIS_DIR, expanded)


def _parameter_counts(model: DownstreamMultiTaskModelFTN) -> dict[str, int]:
    updates = [sum(p.numel() for p in module.parameters()) for module in model.task_updates]
    return {
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "shared_adapter_parameters": sum(
            p.numel() for p in model.shared_adapter.parameters()
        ),
        "task_update_parameters_total": sum(updates),
        "task_update_parameters_per_task": updates[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate FTN-inspired wavCSE decomposition"
    )
    parser.add_argument("--task_type", required=True, help="Must be ks_si_er")
    parser.add_argument(
        "--config",
        default=os.path.join(_THIS_DIR, "configs", "ftn_config.yml"),
        help="Path to the FTN YAML configuration",
    )
    parser.add_argument(
        "--device_index", type=int, default=None, help="Override the config GPU index"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (overrides config file's seed:, default 42 if neither set)",
    )
    args = parser.parse_args()

    if args.task_type != DownstreamMultiTaskModelFTN.SUPPORTED_TASK_TYPE:
        parser.error("FTN v1 supports only --task_type ks_si_er")

    load_dotenv(os.path.join(_REPO_ROOT, ".env"))
    cfg = load_config(_config_path(args.config))
    setup_logging(log_level=cfg["log_level"])

    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    set_seed(seed)
    cfg["seed"] = seed

    device_index = (
        args.device_index if args.device_index is not None else cfg["device"]["index"]
    )
    device = set_device(cfg["device"]["type"], device_index)

    transformer_layers = parse_transformer_layers(
        transformer_layers=cfg["upstream"]["selected_transformer_layers"],
        upstream_model_type=cfg["upstream"]["model_type"],
    )
    layer_pooling_type = cfg["pooling"]["layer_pooling_type"]
    layer_pooling_param = (
        len(transformer_layers)
        if layer_pooling_type in {"weighted", "gated"}
        else cfg["pooling"]["layer_pooling_param"]
    )

    results_root = _owner_relative_path(cfg["paths"]["results_root"])
    checkpoints_root = _owner_relative_path(cfg["paths"]["checkpoints_root"])
    task_type = args.task_type

    mlflow_utils.setup_mlflow(cfg)
    ftn_rank = cfg["model"]["ftn_rank"]
    run_name = (
        f"ftn_r{ftn_rank}_{task_type}_"
        f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    )

    with mlflow.start_run(run_name=run_name):
        mlflow_utils.log_config_params(cfg)
        mlflow.log_param("task_type", task_type)
        mlflow.log_param("ftn_rank", ftn_rank)

        loader = LoadEmbedding(
            root_data_path=cfg["paths"]["root_data_path"],
            root_emb_path=cfg["paths"]["root_emb_path"],
            upstream_model_type=cfg["upstream"]["model_type"],
            frame_pooling_type=cfg["pooling"]["frame_pooling_type"],
            frame_pooling_param=cfg["pooling"]["frame_pooling_param"],
            transformer_layer_array=transformer_layers,
            device=device,
        )
        train_data, val_data, test_data = loader.load_embedding(
            task_type=task_type,
            subset_percentage=cfg["dataset"]["subset_percentage"],
        )

        model_cfg = cfg["model"]
        model = DownstreamMultiTaskModelFTN(
            upstream_model_type=cfg["upstream"]["model_type"],
            task_type=task_type,
            embedding_dim_shared1=model_cfg["embedding_dim_shared1"],
            embedding_dim_shared2=model_cfg["embedding_dim_shared2"],
            layer_pooling_type=layer_pooling_type,
            layer_pooling_param=layer_pooling_param,
            dropout_prob_shared1=model_cfg["dropout_prob_shared1"],
            dropout_prob_shared2=model_cfg["dropout_prob_shared2"],
            ftn_rank=model_cfg["ftn_rank"],
        ).to(device)
        mlflow.log_params(_parameter_counts(model))

        trainer = _LiveMlflowTrainer(
            model=model,
            device=device,
            task_type=task_type,
            training_cfg=cfg["training"],
            results_root=results_root,
            checkpoints_root=checkpoints_root,
            training_data=train_data,
            validation_data=val_data,
            ignore_index=cfg["dataset"]["ignore_index"],
        )
        trainer.train()

        results_run_id = os.path.basename(trainer.results_dir)
        checkpoint_run_id = os.path.basename(trainer.ckpt_dir)
        for tag in ("opt", "best", "epoch"):
            evaluator = MultiTasksModelEvaluator(
                model=model,
                device=device,
                task_type=task_type,
                evaluation_cfg=cfg["evaluation"],
                results_root=results_root,
                dataset=test_data,
                checkpoints_root=checkpoints_root,
                checkpoint_tag=tag,
                ignore_index=cfg["dataset"]["ignore_index"],
                results_run_id=results_run_id,
                checkpoint_run_id=checkpoint_run_id,
            )
            stats = evaluator.write_metrics()
            evaluator.write_predictions_csv()
            mlflow_utils.log_eval_stats(stats, tag, trainer.task_array)

        mlflow.log_metrics(
            {
                f"final_delta_norm_{task}": norm
                for task, norm in model.get_delta_norms().items()
            }
        )
        mlflow.log_artifacts(trainer.ckpt_dir, artifact_path="checkpoints")
        mlflow.log_artifacts(trainer.results_dir, artifact_path="results")


if __name__ == "__main__":
    main()
