"""
10-fold leave-one-speaker-out (LOSO) cross-validation for the ER task,
run inside the plain wavCSE downstream model (joint ks_si_er training),
tracked in MLflow (backed by a DagsHub-hosted tracking server).

Why: the default IEMOCAP split in downstream/dataset/load_embedding.py's
_load_iemocap() pools all 10 speakers together and takes a fixed positional
stride for val/test -- train and val/test almost certainly share speakers,
so the resulting er accuracy is inflated by speaker memorization rather
than genuine emotion generalization. LOSO cross-validation is the standard
SER evaluation protocol on IEMOCAP specifically because it removes that
leakage. See improvements/base/README.md for the full design rationale.

Only the er/IEMOCAP split is folded -- ks (SpeechCommand) and si (VoxCeleb)
keep their normal official splits every fold, since they're loaded via
independent methods (_load_speechcommand/_load_voxceleb) that this script
does not touch. downstream/ itself is never modified: _LOSOLoadEmbedding
below reuses LoadEmbedding._load_iemocap()'s IEMOCAPEmbedding construction
unmodified and only re-slices its output into LOSO indices.

Usage:
    python run_base_er_kfold.py --task_type ks_si_er --config configs/base_kfold_config.yml [--num_folds 10] [--device_index 0]
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
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_IMPROVEMENTS_DIR = os.path.dirname(_THIS_DIR)              # .../improvements
_REPO_ROOT = os.path.dirname(_IMPROVEMENTS_DIR)              # .../wavCSE (git repo root)
_DOWNSTREAM_DIR = os.path.join(_REPO_ROOT, "downstream")

sys.path.insert(0, _DOWNSTREAM_DIR)     # exposes dataset/, model/, trainer/, evaluator/, utils/ as top-level
sys.path.insert(0, _IMPROVEMENTS_DIR)   # exposes mlflow_utils as top-level
sys.path.insert(0, _THIS_DIR)           # exposes kfold_iemocap, run_base as top-level

import mlflow
import mlflow_utils
from loading_utils import get_loader_device

from torch.utils.data import Subset

from dataset.load_embedding import LoadEmbedding
from model.downstream_model import DownstreamMultiTaskModel
from evaluator.evaluator_model import MultiTasksModelEvaluator
from utils.load_config import load_config
from utils.setup_device import set_device
from utils.setup_logging import setup_logging
from utils.parse_transformer_layers import parse_transformer_layers

from kfold_iemocap import build_loso_fold
from run_base import _LiveMlflowTrainer


class _LOSOLoadEmbedding(LoadEmbedding):
    """LoadEmbedding, with IEMOCAP's split replaced by a LOSO fold.

    Reuses the parent's _load_iemocap() to build the IEMOCAPEmbedding
    instance (same filtering/index_pattern), then discards its stride-based
    indices and re-slices by speaker instead. ks/si loading is inherited
    unmodified.
    """

    def __init__(self, *args, fold_index=0, num_folds=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.fold_index = fold_index
        self.num_folds = num_folds
        self.held_out_test_speaker = None
        self.held_out_val_speaker = None

    def _load_iemocap(self):
        train_subset, _val_subset, _test_subset = super()._load_iemocap()
        iemocap_total_data = train_subset.dataset  # underlying IEMOCAPEmbedding, shared by all 3 Subsets

        train_idx, val_idx, test_idx, test_spk, val_spk = build_loso_fold(
            iemocap_total_data, self.fold_index, self.num_folds
        )
        self.held_out_test_speaker = test_spk
        self.held_out_val_speaker = val_spk

        return (
            Subset(iemocap_total_data, train_idx),
            Subset(iemocap_total_data, val_idx),
            Subset(iemocap_total_data, test_idx),
        )


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

    embedding_dim_shared1 = cfg["model"]["embedding_dim_shared1"]
    embedding_dim_shared2 = cfg["model"]["embedding_dim_shared2"]
    dropout_prob_shared1 = cfg["model"]["dropout_prob_shared1"]
    dropout_prob_shared2 = cfg["model"]["dropout_prob_shared2"]

    training_cfg = cfg["training"]
    evaluation_cfg = cfg["evaluation"]

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

    model = DownstreamMultiTaskModel(
        upstream_model_type=upstream_model_type,
        task_type=task_type,
        embedding_dim_shared1=embedding_dim_shared1,
        embedding_dim_shared2=embedding_dim_shared2,
        layer_pooling_type=layer_pooling_type,
        layer_pooling_param=layer_pooling_param,
        dropout_prob_shared1=dropout_prob_shared1,
        dropout_prob_shared2=dropout_prob_shared2
    )
    model.to(device)

    fold_results_root = os.path.join(results_root, f"fold_{fold_index}")
    fold_checkpoints_root = os.path.join(checkpoints_root, f"fold_{fold_index}")

    trainer = _LiveMlflowTrainer(
        model=model,
        device=device,
        task_type=task_type,
        training_cfg=training_cfg,
        results_root=fold_results_root,
        checkpoints_root=fold_checkpoints_root,
        training_data=train_data,
        validation_data=val_data,
        ignore_index=ignore_index
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

    mlflow.log_artifacts(trainer.results_dir, artifact_path="results")

    return {
        "fold_index": fold_index,
        "held_out_test_speaker": loader.held_out_test_speaker,
        "held_out_val_speaker": loader.held_out_val_speaker,
        "er_metrics": fold_er_metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="10-fold leave-one-speaker-out cross-validation for the er task"
    )
    parser.add_argument("--task_type", type=str, default="ks_si_er",
                         help="Task string joined by underscores; must include 'er'")
    parser.add_argument("--config", type=str,
                         default=os.path.join(_THIS_DIR, "configs", "base_kfold_config.yml"),
                         help="Path to YAML configuration file")
    parser.add_argument("--num_folds", type=int, default=10,
                         help="Number of LOSO folds (must equal the number of IEMOCAP speakers, 10)")
    parser.add_argument("--device_index", type=int, default=None,
                         help="GPU index to use (overrides config file)")
    args = parser.parse_args()

    task_type = args.task_type
    if "er" not in task_type.split("_"):
        raise ValueError(f"task_type must include 'er' for ER cross-validation, got: {task_type}")

    load_dotenv(os.path.join(_REPO_ROOT, ".env"))

    cfg = load_config(args.config)

    log_level = cfg["log_level"]
    device_type = cfg["device"]["type"]
    device_index = args.device_index if args.device_index is not None else cfg["device"]["index"]

    results_root = cfg["paths"]["results_root"]
    checkpoints_root = cfg["paths"]["checkpoints_root"]

    # ----------------------------
    # Pre-run disk guard: this is a shared machine whose root disk has
    # repeatedly hit ~0 bytes free mid-run (a checkpoint save then fails
    # with "PytorchStreamWriter failed writing file", silently corrupting
    # the run -- see run_base.py's/run_improvements.py's identical guard).
    # Abort loudly before training instead of dying mid-checkpoint. This
    # script is the most exposed of the three -- 10 full joint trainings
    # per invocation.
    # ----------------------------
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
    run_name = mlflow_utils.build_run_name("base", "original", task_type, suffix="kfold")

    fold_results = []

    with mlflow.start_run(run_name=run_name):
        mlflow_utils.log_config_params(cfg)
        mlflow_utils.set_standard_tags("base", "original", cfg)
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
