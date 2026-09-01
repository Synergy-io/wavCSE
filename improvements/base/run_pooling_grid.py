"""
Grid search over every layer-pooling method (downstream/pooling/pooling.py
supports 10) for the plain wavCSE baseline, at one layer count per process.

Reuses downstream/dataset/load_embedding.py, downstream/model/downstream_model.py,
downstream/trainer/trainer_model.py, downstream/evaluator/evaluator_model.py
UNMODIFIED -- downstream/ is not touched. Only orchestration and MLflow
logging are new (mirrors improvements/base/run_base.py's pattern).

Two-stage search per layer count:
  1. Screening: every (pooling_type, pooling_param) combo, 10 epochs, ranked
     by trainer.ckpt.opt_accuracy_all_threshold (the same "opt" checkpoint
     selection criterion -- avg-of-per-task accuracy -- used to report every
     other number in this project's campaigns, for consistency). No full
     evaluator pass (its batch_size=1 eval takes minutes; screening only
     needs a fast, comparable ranking signal). Embeddings loaded ONCE and
     reused across every combo in this layer count.
  2. Confirmation: top 3 combos re-run at the full 30-epoch budget with the
     full evaluator pass (opt/best/epoch tags), directly comparable to every
     other reported number.

Usage:
    python run_pooling_grid.py --num_layers 16 --device_index 0
    python run_pooling_grid.py --num_layers 25 --device_index 1

Run the two invocations concurrently (one per GPU) -- see
improvements/base/POOLING_GRID_SEARCH.md for the launch commands and results.
"""

import os
import sys
import json
import shutil
import argparse

from dotenv import load_dotenv

# ----------------------------
# Path setup (robust to cwd) -- same pattern as run_base.py
# ----------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_IMPROVEMENTS_DIR = os.path.dirname(_THIS_DIR)
_REPO_ROOT = os.path.dirname(_IMPROVEMENTS_DIR)
_DOWNSTREAM_DIR = os.path.join(_REPO_ROOT, "downstream")

sys.path.insert(0, _DOWNSTREAM_DIR)
sys.path.insert(0, _IMPROVEMENTS_DIR)

import mlflow
import mlflow_utils
from loading_utils import get_loader_device

from dataset.load_embedding import LoadEmbedding
from model.downstream_model import DownstreamMultiTaskModel
from trainer.trainer_model import MultiTasksModelTrainer
from evaluator.evaluator_model import MultiTasksModelEvaluator
from utils.setup_device import set_device
from utils.setup_logging import setup_logging
from utils.parse_transformer_layers import parse_transformer_layers

_LiveMlflowTrainer = mlflow_utils.make_live_trainer(MultiTasksModelTrainer)

TASK_TYPE = "ks_si_er"
MLFLOW_URI = "https://dagshub.com/Ke-vin-S/wavCSE.mlflow"
EXPERIMENT = "wavcse-base-poolingsweep"
EMBEDDING_DIM_SHARED1 = 512  # sap's d_model = the PROJECTED dim pooling runs on, not the raw 1024
LAYERS_16 = "6,1,0,3,2,5,4,7,8,9,10,11,12,17,14,13"

# 6 pooling types with no tunable param worth sweeping; 4 with a small sweep.
FIXED_POOLINGS = ["mean", "max", "weighted", "gated", "auto", "sap"]
SWEEP_GRID = {
    "mix": [0.25, 0.5, 0.75],
    "lnp": [4, 8, 16],
    "smp": [0.5, 1, 2],
    "lse": [1, 2, 5],
}

EVAL_CFG = {"batch_size": 1, "shuffle": False, "pin_memory": True, "drop_last": False, "num_workers": 4}


def build_pooling_combos(num_layers):
    combos = []
    for ptype in FIXED_POOLINGS:
        if ptype in ("weighted", "gated"):
            param = num_layers
        elif ptype == "sap":
            param = EMBEDDING_DIM_SHARED1
        else:
            param = None
        combos.append((ptype, param))
    for ptype, values in SWEEP_GRID.items():
        for v in values:
            combos.append((ptype, v))
    return combos


def disk_guard(*roots, min_gb=2.0):
    for root in roots:
        expanded = os.path.expanduser(root)
        os.makedirs(expanded, exist_ok=True)
        free_gb = shutil.disk_usage(expanded).free / (1024 ** 3)
        if free_gb < min_gb:
            raise RuntimeError(
                f"Only {free_gb:.1f} GB free on {expanded} -- aborting before "
                f"training to avoid a mid-run checkpoint failure."
            )


def build_cfg(pooling_type, layer_pooling_param, num_layers, device_index,
              num_epochs, results_root, checkpoints_root):
    return {
        "log_level": "INFO",
        "device": {"type": "cuda", "index": device_index},
        "paths": {
            "root_data_path": "~/voice_dataset",
            "root_emb_path": "~/dataset/embedding",
            "results_root": results_root,
            "checkpoints_root": checkpoints_root,
        },
        "upstream": {
            "model_type": "wavlm_large",
            "selected_transformer_layers": "all" if num_layers == 25 else LAYERS_16,
        },
        "dataset": {"subset_percentage": 100, "ignore_index": -1},
        "pooling": {
            "frame_pooling_type": "mean",
            "frame_pooling_param": None,
            "layer_pooling_type": pooling_type,
            "layer_pooling_param": layer_pooling_param,
        },
        "model": {
            "embedding_dim_shared1": EMBEDDING_DIM_SHARED1,
            "embedding_dim_shared2": 2000,
            "dropout_prob_shared1": 0.4,
            "dropout_prob_shared2": 0.6,
        },
        "training": {
            "num_epochs": num_epochs,
            "batch_size": 2048,
            "learning_rate": 0.0025,
            "weight_decay": 0.00000005,
            "saved_checkpoint_count": 1,
            "shuffle_train": True,
            "shuffle_val": True,
            "pin_memory": True,
            "drop_last_train": True,
            "drop_last_val": False,
            "num_workers": 4,
            "patience": 1,
            "factor": 0.5,
            "l1_lambda": 0.0000001,
            "l2_lambda": 0.00001,
        },
        "evaluation": EVAL_CFG,
        "mlflow": {"tracking_uri": MLFLOW_URI, "experiment_name": EXPERIMENT},
    }


def run_one(pooling_type, pooling_param, num_layers, device, device_index,
            transformer_layer_array, train_data, val_data, test_data,
            num_epochs, stage, results_root, checkpoints_root, full_eval):
    layer_pooling_param = pooling_param
    if pooling_type in ("weighted", "gated"):
        layer_pooling_param = len(transformer_layer_array)

    cfg = build_cfg(pooling_type, layer_pooling_param, num_layers, device_index,
                     num_epochs, results_root, checkpoints_root)

    disk_guard(results_root, checkpoints_root)

    mlflow_utils.setup_mlflow(cfg)
    model_slug = f"pool-{pooling_type}" + (
        f"-{pooling_param}" if pooling_type in ("mix", "lnp", "smp", "lse") else ""
    )
    run_name = mlflow_utils.build_run_name("base", model_slug, TASK_TYPE, suffix=f"{num_layers}L_{stage}")

    result = {
        "pooling_type": pooling_type, "pooling_param": pooling_param,
        "num_layers": num_layers, "stage": stage,
    }

    with mlflow.start_run(run_name=run_name):
        mlflow_utils.log_config_params(cfg)
        mlflow_utils.set_standard_tags("base", model_slug, cfg, extra_tags={
            "stage": stage,
            "pooling_type": pooling_type,
            "pooling_param": str(pooling_param),
            "num_layers": num_layers,
        })
        mlflow.log_param("task_type", TASK_TYPE)

        model = DownstreamMultiTaskModel(
            upstream_model_type="wavlm_large",
            task_type=TASK_TYPE,
            embedding_dim_shared1=EMBEDDING_DIM_SHARED1,
            embedding_dim_shared2=2000,
            layer_pooling_type=pooling_type,
            layer_pooling_param=layer_pooling_param,
            dropout_prob_shared1=0.4,
            dropout_prob_shared2=0.6,
        )
        model.to(device)

        trainer = _LiveMlflowTrainer(
            model=model,
            device=device,
            task_type=TASK_TYPE,
            training_cfg=cfg["training"],
            results_root=results_root,
            checkpoints_root=checkpoints_root,
            training_data=train_data,
            validation_data=val_data,
            ignore_index=-1,
        )
        trainer.train()

        # Ranking signal: same "opt" checkpoint-selection criterion (best
        # avg-of-per-task accuracy) used to report test_opt_acc_all
        # everywhere else in this project, applied here to VALIDATION stats
        # (that's what maybe_save_best_and_opt is actually called with).
        result["val_opt_acc_all"] = trainer.ckpt.opt_accuracy_all_threshold
        result["val_best_acc_all"] = trainer.ckpt.best_accuracy_all_threshold
        result["val_opt_acc_task"] = dict(trainer.ckpt.opt_accuracy_task_thresholds)
        mlflow.log_metric("screen_val_opt_acc_all", result["val_opt_acc_all"])
        mlflow.log_metric("screen_val_best_acc_all", result["val_best_acc_all"])

        if full_eval:
            for tag in ["best", "opt"]:
                ckpt_path = trainer.model_checkpoint_path.replace(".pth", f"_{tag}.pth")
                if os.path.exists(ckpt_path):
                    mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")

            for tag in ["opt", "best", "epoch"]:
                try:
                    evaluator = MultiTasksModelEvaluator(
                        model=model, device=device, task_type=TASK_TYPE,
                        checkpoints_root=checkpoints_root, results_root=results_root,
                        evaluation_cfg=EVAL_CFG, dataset=test_data,
                        checkpoint_tag=tag, ignore_index=-1,
                    )
                    stats = evaluator.write_metrics()
                    evaluator.write_predictions_csv()
                    mlflow_utils.log_eval_stats(stats, tag, trainer.task_array)
                    if tag == "opt":
                        result["test_opt_acc_all"] = stats.accuracy_all
                        result["test_opt_acc_task"] = {
                            trainer.task_array[t]: stats.accuracy_task[t]
                            for t in range(len(trainer.task_array))
                        }
                except Exception as e:
                    print(f"  eval tag={tag} failed: {e}")

            mlflow.log_artifacts(trainer.results_dir, artifact_path="results")
            result["mlflow_run_id"] = mlflow.active_run().info.run_id

    return result


def main():
    parser = argparse.ArgumentParser(description="Pooling grid search (one layer count per process)")
    parser.add_argument("--num_layers", type=int, required=True, choices=[16, 25])
    parser.add_argument("--device_index", type=int, required=True)
    parser.add_argument("--screen_epochs", type=int, default=10)
    parser.add_argument("--confirm_epochs", type=int, default=30)
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    load_dotenv(os.path.join(_REPO_ROOT, ".env"))
    setup_logging("INFO")

    num_layers = args.num_layers
    device = set_device("cuda", args.device_index)
    layer_str = "all" if num_layers == 25 else LAYERS_16
    transformer_layer_array = parse_transformer_layers(layer_str, "wavlm_large")

    print(f"\n{'='*60}\n  Pooling grid search: {num_layers} layers (GPU {args.device_index})\n{'='*60}\n")

    loader = LoadEmbedding(
        root_emb_path=os.path.expanduser("~/dataset/embedding"),
        root_data_path=os.path.expanduser("~/voice_dataset"),
        upstream_model_type="wavlm_large",
        frame_pooling_type="mean",
        frame_pooling_param=None,
        transformer_layer_array=transformer_layer_array,
        device=get_loader_device(),
    )
    train_data, val_data, test_data = loader.load_embedding(TASK_TYPE, subset_percentage=100)

    combos = build_pooling_combos(num_layers)
    print(f"Screening {len(combos)} pooling combos at {args.screen_epochs} epochs each...")

    screen_results_root = f"results_poolingsweep_screen_{num_layers}L"
    screen_ckpt_root = f"checkpoints_poolingsweep_screen_{num_layers}L"

    screen_results = []
    for i, (pooling_type, pooling_param) in enumerate(combos):
        print(f"\n[{i+1}/{len(combos)}] {num_layers}L screen: {pooling_type} param={pooling_param}")
        try:
            r = run_one(
                pooling_type, pooling_param, num_layers, device, args.device_index,
                transformer_layer_array, train_data, val_data, test_data,
                num_epochs=args.screen_epochs, stage="screen",
                results_root=screen_results_root, checkpoints_root=screen_ckpt_root,
                full_eval=False,
            )
            screen_results.append(r)
            print(f"  -> val_opt_acc_all={r['val_opt_acc_all']:.4f}")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Screening checkpoints/results are throwaway -- only the logged MLflow
    # metrics matter. Reclaim disk immediately.
    shutil.rmtree(screen_results_root, ignore_errors=True)
    shutil.rmtree(screen_ckpt_root, ignore_errors=True)

    screen_results.sort(key=lambda r: r["val_opt_acc_all"], reverse=True)
    print(f"\n{'='*60}\n  Screening ranking ({num_layers}L)\n{'='*60}")
    for r in screen_results:
        print(f"  {r['val_opt_acc_all']:.4f}  {r['pooling_type']:10s} param={r['pooling_param']}")

    top_k = screen_results[: args.top_k]
    print(f"\nConfirming top {len(top_k)} at {args.confirm_epochs} epochs (full evaluator)...")

    confirm_results_root = f"results_poolingsweep_confirm_{num_layers}L"
    confirm_ckpt_root = f"checkpoints_poolingsweep_confirm_{num_layers}L"

    confirm_results = []
    for r in top_k:
        print(f"\nConfirm {num_layers}L: {r['pooling_type']} param={r['pooling_param']}")
        try:
            cr = run_one(
                r["pooling_type"], r["pooling_param"], num_layers, device, args.device_index,
                transformer_layer_array, train_data, val_data, test_data,
                num_epochs=args.confirm_epochs, stage="confirm",
                results_root=confirm_results_root, checkpoints_root=confirm_ckpt_root,
                full_eval=True,
            )
            confirm_results.append(cr)
            print(f"  -> test_opt_acc_all={cr.get('test_opt_acc_all')}")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    out_path = os.path.join(_THIS_DIR, f"pooling_grid_results_{num_layers}L.json")
    with open(out_path, "w") as f:
        json.dump({"num_layers": num_layers, "screen": screen_results, "confirm": confirm_results}, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
