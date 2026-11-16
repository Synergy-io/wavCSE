"""Train and evaluate the RAMUSA-inspired low-rank model.

Usage:
    python run.py --task_type ks_si_er --config configs/ramusa_config.yml
"""

import argparse
import logging
import os
import sys
from contextlib import nullcontext
from datetime import datetime

import torch
from dotenv import load_dotenv

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_IMPROVEMENTS_DIR = os.path.dirname(_THIS_DIR)
_REPO_ROOT = os.path.dirname(_IMPROVEMENTS_DIR)
_DOWNSTREAM_DIR = os.path.join(_REPO_ROOT, "downstream")

sys.path = [
    path
    for path in sys.path
    if os.path.abspath(path or os.getcwd()) != _THIS_DIR
]
sys.path.insert(0, _IMPROVEMENTS_DIR)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _DOWNSTREAM_DIR)

from dataset.load_embedding import LoadEmbedding
from evaluator.evaluator_model import MultiTasksModelEvaluator
from improvements.lowrank.models.ramusa_model import DownstreamMultiTaskModelRAMUSA
from improvements.lowrank.trainers.ramusa_trainer import MultiTasksModelTrainerRAMUSA
from improvements.tensorboard_utils import tensorboard_trainer
from utils.load_config import load_config
from utils.parse_transformer_layers import parse_transformer_layers
from utils.setup_device import set_device
from utils.setup_logging import setup_logging


def _mlflow_context(cfg, task_type):
    try:
        import mlflow
        import mlflow_utils
        mlflow_utils.setup_mlflow(cfg)
        return mlflow.start_run(
            run_name=f"ramusa_{task_type}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        )
    except Exception as error:
        logging.warning("MLflow unavailable; continuing locally: %s", error)
        return nullcontext(None)


def _log_final_mlflow_state(trainer):
    try:
        import mlflow

        for tag in ("opt", "best"):
            path = trainer.model_checkpoint_path.replace(".pth", f"_{tag}.pth")
            if os.path.exists(path):
                mlflow.log_artifact(path, artifact_path="checkpoints")
        mlflow.log_artifacts(trainer.results_dir, artifact_path="results")
    except Exception as error:
        logging.warning("Could not log final MLflow artifacts: %s", error)
 

def main():
    parser = argparse.ArgumentParser(description="Train the RAMUSA low-rank wavCSE model")
    parser.add_argument("--task_type", default="ks_si_er")
    parser.add_argument("--config", default=os.path.join(_THIS_DIR, "configs", "ramusa_config.yml"))
    parser.add_argument("--device_index", type=int, default=None)
    args = parser.parse_args()
    if args.task_type != "ks_si_er":
        parser.error("RAMUSA supports only --task_type ks_si_er")

    load_dotenv(os.path.join(_REPO_ROOT, ".env"))
    cfg = load_config(args.config)
    setup_logging(cfg["log_level"])
    device = set_device(
        cfg["device"]["type"],
        args.device_index if args.device_index is not None else cfg["device"]["index"],
    )
    layers = parse_transformer_layers(
        cfg["upstream"]["selected_transformer_layers"], cfg["upstream"]["model_type"]
    )
    pooling_type = cfg["pooling"]["layer_pooling_type"]
    pooling_param = cfg["pooling"].get("layer_pooling_param")
    if pooling_type in ("weighted", "gated"):
        pooling_param = len(layers)
    loader = LoadEmbedding(
        root_data_path=os.path.expanduser(cfg["paths"]["root_data_path"]),
        root_emb_path=os.path.expanduser(cfg["paths"]["root_emb_path"]),
        upstream_model_type=cfg["upstream"]["model_type"],
        frame_pooling_type=cfg["pooling"]["frame_pooling_type"],
        frame_pooling_param=cfg["pooling"].get("frame_pooling_param"),
        transformer_layer_array=layers,
        device=device,
    )
    train_data, val_data, test_data = loader.load_embedding(
        args.task_type, subset_percentage=cfg["dataset"]["subset_percentage"]
    )

    with _mlflow_context(cfg, args.task_type) as active_run:
        if active_run is not None:
            import mlflow
            import mlflow_utils
            mlflow_utils.log_config_params(cfg)
            mlflow.log_param("task_type", args.task_type)
        model = DownstreamMultiTaskModelRAMUSA(
            upstream_model_type=cfg["upstream"]["model_type"], task_type=args.task_type,
            embedding_dim_shared1=cfg["model"]["embedding_dim_shared1"],
            embedding_dim_shared2=cfg["model"]["embedding_dim_shared2"],
            layer_pooling_type=pooling_type, layer_pooling_param=pooling_param,
            dropout_prob_shared1=cfg["model"]["dropout_prob_shared1"],
            dropout_prob_shared2=cfg["model"]["dropout_prob_shared2"],
            adapter_dim=cfg["model"].get("adapter_dim"),
        ).to(device)
        trainer_class = tensorboard_trainer(MultiTasksModelTrainerRAMUSA)
        trainer = trainer_class(
            model=model, device=device, task_type=args.task_type,
            training_cfg=cfg["training"], ramusa_cfg=cfg["ramusa"],
            results_root=cfg["paths"]["results_root"],
            checkpoints_root=cfg["paths"]["checkpoints_root"],
            training_data=train_data, validation_data=val_data,
            ignore_index=cfg["dataset"]["ignore_index"],
            tensorboard_cfg=cfg.get("tensorboard", {}),
        )
        trainer.train()
        run_id = os.path.basename(trainer.results_dir)
        for tag in ("opt", "best"):
            evaluator = MultiTasksModelEvaluator(
                model=model, device=device, task_type=args.task_type,
                evaluation_cfg=cfg["evaluation"], results_root=cfg["paths"]["results_root"],
                dataset=test_data, checkpoints_root=cfg["paths"]["checkpoints_root"],
                checkpoint_tag=tag, ignore_index=cfg["dataset"]["ignore_index"],
                results_run_id=run_id, checkpoint_run_id=run_id,
            )
            stats = evaluator.write_metrics()
            evaluator.write_predictions_csv()
            if active_run is not None:
                try:
                    import mlflow_utils
                    mlflow_utils.log_eval_stats(stats, tag, trainer.task_array)
                except Exception as error:
                    logging.warning("Could not log MLflow evaluation metrics: %s", error)
        if active_run is not None:
            _log_final_mlflow_state(trainer)


if __name__ == "__main__":
    main()
