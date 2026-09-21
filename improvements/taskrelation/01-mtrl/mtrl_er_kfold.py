"""
10-fold leave-one-speaker-out (LOSO) cross-validation for the ER task,
run inside wavCSE-MTRL (joint ks_si_er training with the task-relation
regularizer), tracked in MLflow (backed by a DagsHub-hosted tracking server).

Sibling of improvements/base/run_base_er_kfold.py -- same LOSO protocol,
same _LOSOLoadEmbedding/build_loso_fold reused unmodified from there, but
builds DownstreamMultiTaskModelMTRL + MultiTasksModelTrainerMTRL instead of
the plain model/trainer, so the MTRL-vs-baseline `er` comparison can finally
be measured on the leakage-free split instead of downstream/'s speaker-leaky
one. See improvements/base/README.md's "ER 10-fold cross-validation"
section for the full design rationale (why LOSO, why only `er` is folded,
why 5 epochs/fold) and its baseline results this run is compared against.

Usage:
    python mtrl_er_kfold.py --task_type ks_si_er --config mtrl_kfold_config.yml [--num_folds 10] [--device_index 0]
"""

import os
import sys
import json
import shutil
import argparse
import statistics

from dotenv import load_dotenv

# ----------------------------
# Path setup (robust to cwd)
# ----------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))                       # .../taskrelation/01-mtrl
_TASKRELATION_DIR = os.path.dirname(_THIS_DIR)                               # .../taskrelation
_IMPROVEMENTS_DIR = os.path.dirname(_TASKRELATION_DIR)                       # .../improvements
_REPO_ROOT = os.path.dirname(_IMPROVEMENTS_DIR)                              # .../wavCSE (git repo root)
_DOWNSTREAM_DIR = os.path.join(_REPO_ROOT, "downstream")
_BASE_DIR = os.path.join(_IMPROVEMENTS_DIR, "base")

sys.path.insert(0, _DOWNSTREAM_DIR)     # exposes dataset/, model/, trainer/, evaluator/, utils/ as top-level
sys.path.insert(0, _IMPROVEMENTS_DIR)   # exposes mlflow_utils as top-level
sys.path.insert(0, _BASE_DIR)           # exposes kfold_iemocap, _LOSOLoadEmbedding (via run_base_er_kfold)
sys.path.insert(0, _THIS_DIR)           # exposes mtrl_model, mtrl_trainer as top-level

import mlflow
import mlflow_utils
from loading_utils import get_loader_device
from seed_utils import set_seed

from evaluator.evaluator_model import MultiTasksModelEvaluator
from utils.load_config import load_config
from utils.setup_device import set_device
from utils.setup_logging import setup_logging
from utils.parse_transformer_layers import parse_transformer_layers

# Reused unmodified from the baseline LOSO script -- same fold-assignment
# logic and the same IEMOCAP-only re-slicing wrapper around LoadEmbedding.
from run_base_er_kfold import _LOSOLoadEmbedding

import mtrl_model
import mtrl_trainer

_LiveMlflowTrainerMTRL = mlflow_utils.make_live_trainer(mtrl_trainer.MultiTasksModelTrainerMTRL)


def _run_fold(fold_index, cfg, task_type, device, results_root, checkpoints_root):
    root_data_path = cfg["paths"]["root_data_path"]
    root_emb_path = cfg["paths"]["root_emb_path"]

    upstream_model_type = cfg["upstream"]["model_type"]
    selected_transformer_layers = cfg["upstream"]["selected_transformer_layers"]
    transformer_layer_array = parse_transformer_layers(
        transformer_layers=selected_transformer_layers,
        upstream_model_type=upstream_model_type
    )

    subset_percentage = cfg["dataset"]["subset_percentage"]
    ignore_index = cfg["dataset"]["ignore_index"]

    frame_pooling_type = cfg["pooling"]["frame_pooling_type"]
    layer_pooling_type = cfg["pooling"]["layer_pooling_type"]
    frame_pooling_param = cfg["pooling"]["frame_pooling_param"]
    layer_pooling_param = (
        cfg["pooling"]["layer_pooling_param"]
        if layer_pooling_type not in ["weighted", "gated"]
        else len(transformer_layer_array)
    )

    model_cfg = cfg["model"]
    embedding_dim_shared1 = model_cfg["embedding_dim_shared1"]
    embedding_dim_shared2 = model_cfg["embedding_dim_shared2"]
    dropout_prob_shared1 = model_cfg["dropout_prob_shared1"]
    dropout_prob_shared2 = model_cfg["dropout_prob_shared2"]

    training_cfg = cfg["training"]
    evaluation_cfg = cfg["evaluation"]
    mtrl_cfg = cfg.get("mtrl", {})

    loader = _LOSOLoadEmbedding(
        root_data_path=root_data_path,
        root_emb_path=root_emb_path,
        upstream_model_type=upstream_model_type,
        frame_pooling_type=frame_pooling_type,
        frame_pooling_param=frame_pooling_param,
        transformer_layer_array=transformer_layer_array,
        device=get_loader_device(),
        fold_index=fold_index,
    )

    train_data, val_data, test_data = loader.load_embedding(
        task_type=task_type,
        subset_percentage=subset_percentage
    )

    mlflow.log_params({
        "fold_index": fold_index,
        "held_out_test_speaker": loader.held_out_test_speaker,
        "held_out_val_speaker": loader.held_out_val_speaker,
    })

    model = mtrl_model.DownstreamMultiTaskModelMTRL(
        upstream_model_type=upstream_model_type,
        task_type=task_type,
        embedding_dim_shared1=embedding_dim_shared1,
        embedding_dim_shared2=embedding_dim_shared2,
        layer_pooling_type=layer_pooling_type,
        layer_pooling_param=layer_pooling_param,
        dropout_prob_shared1=dropout_prob_shared1,
        dropout_prob_shared2=dropout_prob_shared2,
        mtrl_lambda=model_cfg.get("mtrl_lambda", 0.01),
        omega_epsilon=model_cfg.get("omega_epsilon", 1e-4),
        normalize_w=model_cfg.get("normalize_w", False),
    )
    model.to(device)

    fold_results_root = os.path.join(results_root, f"fold_{fold_index}")
    fold_checkpoints_root = os.path.join(checkpoints_root, f"fold_{fold_index}")

    trainer = _LiveMlflowTrainerMTRL(
        model=model,
        device=device,
        task_type=task_type,
        training_cfg=training_cfg,
        results_root=fold_results_root,
        checkpoints_root=fold_checkpoints_root,
        training_data=train_data,
        validation_data=val_data,
        ignore_index=ignore_index,
        mtrl_warmup_epochs=mtrl_cfg.get("warmup_epochs", 3),
        omega_update_frequency=mtrl_cfg.get("omega_update_frequency", 1),
    )
    trainer.train()

    for tag in ["best", "opt"]:
        ckpt_path = trainer.model_checkpoint_path.replace(".pth", f"_{tag}.pth")
        if os.path.exists(ckpt_path):
            mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")

    task_array = trainer.task_array
    er_idx = task_array.index("er")

    fold_er_metrics = {}
    for tag in ["opt", "best", "epoch"]:
        evaluator = MultiTasksModelEvaluator(
            model=model,
            device=device,
            task_type=task_type,
            evaluation_cfg=evaluation_cfg,
            results_root=fold_results_root,
            dataset=test_data,
            checkpoints_root=fold_checkpoints_root,
            checkpoint_tag=tag,
            ignore_index=ignore_index
        )
        stats = evaluator.write_metrics()
        evaluator.write_predictions_csv()
        mlflow_utils.log_eval_stats(stats, tag, task_array)

        fold_er_metrics[tag] = {
            "acc": stats.accuracy_task[er_idx],
            "loss": stats.avg_loss_task[er_idx],
        }

    # Final Omega for this fold, for analysis (mirrors mtrl_trainer's own
    # per-epoch history, which is already saved/uploaded from results_dir).
    final_omega = model.get_omega_matrix().tolist()
    mlflow.log_dict({"task_array": task_array, "omega": final_omega}, "omega/final_omega.json")

    mlflow.log_artifacts(trainer.results_dir, artifact_path="results")

    return {
        "fold_index": fold_index,
        "held_out_test_speaker": loader.held_out_test_speaker,
        "held_out_val_speaker": loader.held_out_val_speaker,
        "er_metrics": fold_er_metrics,
        "final_omega": final_omega,
    }


def main():
    parser = argparse.ArgumentParser(
        description="10-fold leave-one-speaker-out cross-validation for the er task, on wavCSE-MTRL"
    )
    parser.add_argument("--task_type", type=str, default="ks_si_er",
                         help="Task string joined by underscores; must include 'er'")
    parser.add_argument("--config", type=str,
                         default=os.path.join(_THIS_DIR, "mtrl_kfold_config.yml"),
                         help="Path to YAML configuration file")
    parser.add_argument("--num_folds", type=int, default=10,
                         help="Number of LOSO folds (must equal the number of IEMOCAP speakers, 10)")
    parser.add_argument("--device_index", type=int, default=None,
                         help="GPU index to use (overrides config file)")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed (overrides config file's seed:, default 42 if neither set)")
    args = parser.parse_args()

    task_type = args.task_type
    if "er" not in task_type.split("_"):
        raise ValueError(f"task_type must include 'er' for ER cross-validation, got: {task_type}")

    load_dotenv(os.path.join(_REPO_ROOT, ".env"))

    cfg = load_config(args.config)

    # One seed for the whole 10-fold sequence -- see run_base_er_kfold.py's
    # identical comment and improvements/seed_utils.py.
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    set_seed(seed)
    cfg["seed"] = seed

    log_level = cfg["log_level"]
    device_type = cfg["device"]["type"]
    device_index = args.device_index if args.device_index is not None else cfg["device"]["index"]

    results_root = cfg["paths"]["results_root"]
    checkpoints_root = cfg["paths"]["checkpoints_root"]

    # Pre-run disk guard -- see run_base_er_kfold.py's identical guard for
    # the rationale (shared machine, root disk has repeatedly hit ~0 bytes
    # free mid-run).
    for root in (checkpoints_root, results_root):
        expanded_root = os.path.expanduser(root)
        os.makedirs(expanded_root, exist_ok=True)
        free_gb = shutil.disk_usage(expanded_root).free / (1024 ** 3)
        if free_gb < 2.0:
            raise RuntimeError(
                f"Only {free_gb:.1f} GB free on the filesystem holding "
                f"{expanded_root} -- aborting before training to avoid a "
                f"mid-run checkpoint failure. Free space on the shared disk "
                f"and retry (see CLAUDE.md gotchas)."
            )

    setup_logging(log_level=log_level)
    device = set_device(device_type=device_type, device_index=device_index)

    mlflow_utils.setup_mlflow(cfg)
    run_name = mlflow_utils.build_run_name("taskrelation", "mtrl", task_type, suffix="kfold")

    fold_results = []

    with mlflow.start_run(run_name=run_name):
        mlflow_utils.log_config_params(cfg)
        mlflow_utils.set_standard_tags("taskrelation", "mtrl", cfg)
        mlflow.log_param("task_type", task_type)
        mlflow.log_params({
            "kfold_protocol": "leave_one_speaker_out",
            "kfold_num_folds": args.num_folds,
            "kfold_epochs_per_fold": cfg["training"]["num_epochs"],
        })

        for fold_index in range(args.num_folds):
            with mlflow.start_run(run_name=f"fold_{fold_index}", nested=True):
                fold_result = _run_fold(
                    fold_index, cfg, task_type, device, results_root, checkpoints_root
                )
                fold_results.append(fold_result)

        summary = {"folds": fold_results, "aggregate": {}}
        for tag in ["opt", "best", "epoch"]:
            accs = [f["er_metrics"][tag]["acc"] for f in fold_results]
            losses = [f["er_metrics"][tag]["loss"] for f in fold_results]

            acc_mean, acc_std = statistics.mean(accs), statistics.pstdev(accs)
            loss_mean, loss_std = statistics.mean(losses), statistics.pstdev(losses)

            summary["aggregate"][tag] = {
                "acc_mean": acc_mean, "acc_std": acc_std,
                "loss_mean": loss_mean, "loss_std": loss_std,
            }

            mlflow.log_metrics({
                f"kfold_er_{tag}_acc_mean": acc_mean,
                f"kfold_er_{tag}_acc_std": acc_std,
                f"kfold_er_{tag}_loss_mean": loss_mean,
                f"kfold_er_{tag}_loss_std": loss_std,
            })

        os.makedirs(results_root, exist_ok=True)
        summary_path = os.path.join(results_root, "kfold_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(summary_path)


if __name__ == "__main__":
    main()
