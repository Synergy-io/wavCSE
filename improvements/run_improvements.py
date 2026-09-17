"""
Run script for wavCSE architectural improvements.

This script benchmarks three improved wavCSE architectures against the original:

1. wavCSE-GBC: Global Bias Coupling (original design for this project --
   see improvements/taskrelation/03-gbc/gbc_model.py's docstring for a
   2026-09-01 citation correction; no named published method matches it)
   - Adds a shared global bias that couples all task heads
   - Lightest change, most similar to original

2. wavCSE-TSM: Task Structure Matrix (Ciliberto et al. 2015)
   - Replaces independent heads with latent classifier basis + structure matrix A
   - Alternating minimization between model and structure
   - l1 sparsity on task relations

3. wavCSE-PMR: Precision Matrix Regularization (Goncalves et al. 2016)
   - Adds learnable sparse precision matrix Omega over task parameters
   - Gaussian graphical model interpretation
   - Trace + log-det regularization

Original wavCSE contributions (layer selection, frame/layer pooling) are PRESERVED.

Usage:
    python run_improvements.py --model [gbc|tsm|pmr|all] --task_type ks_si_er
"""

import os
import sys
import argparse
import logging
import shutil
import yaml
from datetime import datetime

import torch
import mlflow
from dotenv import load_dotenv

# Add downstream path for imports
DOWNSTREAM_PATH = os.path.join(
    os.path.dirname(__file__), '../downstream'
)
sys.path.insert(0, DOWNSTREAM_PATH)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)  # .../wavCSE (git repo root)

from dataset.load_embedding import LoadEmbedding
from utils.load_config import load_config
from utils.setup_device import set_device
from utils.setup_logging import setup_logging
from utils.parse_transformer_layers import parse_transformer_layers

from improvements.loading_utils import get_loader_device
from improvements.seed_utils import set_seed
from improvements import mlflow_utils

# "gbc"/"tsm"/"pmr" are all task-relation-learning variants (Kevin's branch).
# When other members add their own architecture folders (lowrank/, clustering/,
# decomposition/), extend this lookup accordingly -- it drives both the MLflow
# `category` tag and (per the experiment-naming convention) which experiments
# carry the "wavcse-" prefix (base/base-improvement only; other categories are
# their own top-level architectures, not baseline variants).
MODEL_CATEGORY = {
    "original": "base",
    "gbc": "taskrelation",
    "tsm": "taskrelation",
    "pmr": "taskrelation",
    "mtrl": "taskrelation",
}

# "01-mtrl" (and any future "0N-<name>" self-contained architecture folder)
# is not a valid Python package path -- a leading digit and a hyphen both
# break dotted imports (`from improvements.taskrelation.01-mtrl...` is a
# syntax error). Load such folders' modules directly from their file path
# instead of relying on sys.path + package-style imports.
def _load_module_from_path(module_name: str, file_path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Config lookup for model types whose config doesn't live in the flat
# taskrelation/configs/ dir (see CONFIG_PATH_OVERRIDES usage in main()).
CONFIG_PATH_OVERRIDES = {
    "mtrl": os.path.join("taskrelation", "01-mtrl", "mtrl_config.yml"),
    "gbc": os.path.join("taskrelation", "03-gbc", "gbc_config.yml"),
}


def build_model(model_type: str, cfg: dict, task_type: str, layer_pooling_param):
    """Factory for building the appropriate model variant."""
    model_cfg = cfg.get("model", {})

    common_args = dict(
        upstream_model_type=cfg["upstream"]["model_type"],
        task_type=task_type,
        embedding_dim_shared1=model_cfg.get("embedding_dim_shared1", 512),
        embedding_dim_shared2=model_cfg.get("embedding_dim_shared2", 2000),
        layer_pooling_type=cfg["pooling"]["layer_pooling_type"],
        dropout_prob_shared1=model_cfg.get("dropout_prob_shared1", 0.4),
        dropout_prob_shared2=model_cfg.get("dropout_prob_shared2", 0.6),
        layer_pooling_param=layer_pooling_param,
    )

    if model_type == "gbc":
        gbc_dir = os.path.join(os.path.dirname(__file__), "taskrelation", "03-gbc")
        gbc_module = _load_module_from_path(
            "gbc_model", os.path.join(gbc_dir, "gbc_model.py")
        )
        return gbc_module.DownstreamMultiTaskModelGBC(
            **common_args,
            gbc_global_dim=model_cfg.get("gbc_global_dim", 64),
        )
    elif model_type == "tsm":
        from improvements.taskrelation.models.tsm_model import DownstreamMultiTaskModelTSM
        return DownstreamMultiTaskModelTSM(
            **common_args,
            num_latent_classifiers=model_cfg.get("num_latent_classifiers", 8),
            structure_sparsity_lambda=model_cfg.get("structure_sparsity_lambda", 0.01),
        )
    elif model_type == "pmr":
        from improvements.taskrelation.models.pmr_model import DownstreamMultiTaskModelPMR
        return DownstreamMultiTaskModelPMR(
            **common_args,
            pmr_lambda=model_cfg.get("pmr_lambda", 0.01),
            pmr_gamma=model_cfg.get("pmr_gamma", 0.1),
        )
    elif model_type == "mtrl":
        mtrl_dir = os.path.join(os.path.dirname(__file__), "taskrelation", "01-mtrl")
        mtrl_module = _load_module_from_path(
            "mtrl_model", os.path.join(mtrl_dir, "mtrl_model.py")
        )
        return mtrl_module.DownstreamMultiTaskModelMTRL(
            **common_args,
            mtrl_lambda=model_cfg.get("mtrl_lambda", 0.01),
            omega_epsilon=model_cfg.get("omega_epsilon", 1e-4),
            normalize_w=model_cfg.get("normalize_w", False),
        )
    elif model_type == "original":
        from model.downstream_model import DownstreamMultiTaskModel
        return DownstreamMultiTaskModel(**common_args)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def build_trainer(model_type: str, model, device, task_type, cfg, training_data,
                  validation_data, ignore_index):
    """Factory for building the appropriate trainer."""
    training_cfg = cfg["training"]

    if model_type == "tsm":
        from improvements.taskrelation.trainers.tsm_trainer import MultiTasksModelTrainerTSM
        trainer_cls = mlflow_utils.make_live_trainer(MultiTasksModelTrainerTSM)
        tsm_cfg = cfg.get("tsm", {})
        return trainer_cls(
            model=model,
            device=device,
            task_type=task_type,
            training_cfg=training_cfg,
            results_root=cfg["paths"]["results_root"],
            checkpoints_root=cfg["paths"]["checkpoints_root"],
            training_data=training_data,
            validation_data=validation_data,
            ignore_index=ignore_index,
            alternate_frequency=tsm_cfg.get("alternate_frequency", 5),
            structure_epochs=tsm_cfg.get("structure_epochs", 1),
            structure_lr=tsm_cfg.get("structure_lr", 0.001),
        )
    elif model_type == "pmr":
        from improvements.taskrelation.trainers.pmr_trainer import MultiTasksModelTrainerPMR
        trainer_cls = mlflow_utils.make_live_trainer(MultiTasksModelTrainerPMR)
        pmr_cfg = cfg.get("pmr", {})
        return trainer_cls(
            model=model,
            device=device,
            task_type=task_type,
            training_cfg=training_cfg,
            results_root=cfg["paths"]["results_root"],
            checkpoints_root=cfg["paths"]["checkpoints_root"],
            training_data=training_data,
            validation_data=validation_data,
            ignore_index=ignore_index,
            pmr_warmup_epochs=pmr_cfg.get("warmup_epochs", 3),
        )
    elif model_type == "mtrl":
        mtrl_dir = os.path.join(os.path.dirname(__file__), "taskrelation", "01-mtrl")
        mtrl_trainer_module = _load_module_from_path(
            "mtrl_trainer", os.path.join(mtrl_dir, "mtrl_trainer.py")
        )
        trainer_cls = mlflow_utils.make_live_trainer(
            mtrl_trainer_module.MultiTasksModelTrainerMTRL
        )
        mtrl_cfg = cfg.get("mtrl", {})
        return trainer_cls(
            model=model,
            device=device,
            task_type=task_type,
            training_cfg=training_cfg,
            results_root=cfg["paths"]["results_root"],
            checkpoints_root=cfg["paths"]["checkpoints_root"],
            training_data=training_data,
            validation_data=validation_data,
            ignore_index=ignore_index,
            mtrl_warmup_epochs=mtrl_cfg.get("warmup_epochs", 3),
            omega_update_frequency=mtrl_cfg.get("omega_update_frequency", 1),
        )
    else:
        # GBC and original use the standard trainer
        from trainer.trainer_model import MultiTasksModelTrainer
        trainer_cls = mlflow_utils.make_live_trainer(MultiTasksModelTrainer)
        return trainer_cls(
            model=model,
            device=device,
            task_type=task_type,
            training_cfg=training_cfg,
            results_root=cfg["paths"]["results_root"],
            checkpoints_root=cfg["paths"]["checkpoints_root"],
            training_data=training_data,
            validation_data=validation_data,
            ignore_index=ignore_index,
        )


def run_single_model(model_type: str, task_type: str, config_path: str,
                     device_index: int, seed: int = None):
    """Run a single model variant end-to-end."""
    print(f"\n{'='*60}")
    print(f"  Running wavCSE-{model_type.upper()}")
    print(f"  Task: {task_type}")
    print(f"{'='*60}\n")

    cfg = load_config(config_path)
    setup_logging(cfg.get("log_level", "INFO"))

    # Resolved fresh per model, not once at the top of main() -- an
    # `--model all` run calls this once per architecture, and each needs
    # its own configured seed applied cleanly rather than inheriting
    # whatever RNG state the previous architecture's training left behind.
    resolved_seed = seed if seed is not None else cfg.get("seed", 42)
    set_seed(resolved_seed)
    cfg["seed"] = resolved_seed

    # Pre-run disk guard: this is a shared machine whose root disk has
    # repeatedly hit ~0 bytes free mid-run (a checkpoint save then fails
    # with "PytorchStreamWriter failed writing file", silently corrupting
    # the run). Abort loudly before training instead of dying mid-checkpoint.
    for root_key in ("checkpoints_root", "results_root"):
        root = os.path.expanduser(cfg["paths"][root_key])
        os.makedirs(root, exist_ok=True)
        free_gb = shutil.disk_usage(root).free / (1024 ** 3)
        if free_gb < 2.0:
            raise RuntimeError(
                f"Only {free_gb:.1f} GB free on the filesystem holding "
                f"{root_key} ({root}) -- aborting before training to avoid a "
                f"mid-run checkpoint failure. Free space on the shared disk "
                f"and retry (see CLAUDE.md gotchas)."
            )
    logging.info(
        "Disk guard passed: %.1f GB free on %s",
        shutil.disk_usage(os.path.expanduser(cfg["paths"]["checkpoints_root"])).free / (1024 ** 3),
        os.path.expanduser(cfg["paths"]["checkpoints_root"])
    )

    device = set_device(
        cfg.get("device", {}).get("type", "cuda"),
        device_index if device_index is not None else cfg.get("device", {}).get("index", 0)
    )

    # Parse transformer layers
    transformer_layer_array = parse_transformer_layers(
        cfg["upstream"]["selected_transformer_layers"],
        cfg["upstream"]["model_type"]
    )
    logging.info(f"Selected transformer layers: {transformer_layer_array}")

    # Load embeddings
    loader = LoadEmbedding(
        root_emb_path=os.path.expanduser(cfg["paths"]["root_emb_path"]),
        root_data_path=os.path.expanduser(cfg["paths"]["root_data_path"]),
        upstream_model_type=cfg["upstream"]["model_type"],
        frame_pooling_type=cfg["pooling"]["frame_pooling_type"],
        frame_pooling_param=cfg["pooling"].get("frame_pooling_param"),
        transformer_layer_array=transformer_layer_array,
        device=get_loader_device(),
    )

    subset_pct = cfg["dataset"].get("subset_percentage", 100)
    train_data, val_data, test_data = loader.load_embedding(
        task_type, subset_percentage=subset_pct
    )

    # Determine layer pooling param
    layer_pooling_type = cfg["pooling"]["layer_pooling_type"]
    layer_pooling_param = cfg["pooling"].get("layer_pooling_param")
    if layer_pooling_type in ["weighted", "gated"]:
        layer_pooling_param = len(transformer_layer_array)

    ignore_index = cfg["dataset"].get("ignore_index", -1)

    # ----------------------------
    # MLflow setup
    # ----------------------------
    category = MODEL_CATEGORY.get(model_type, "taskrelation")
    mlflow_utils.setup_mlflow(cfg)
    run_name = mlflow_utils.build_run_name(category, model_type, task_type)

    from evaluator.evaluator_model import MultiTasksModelEvaluator

    with mlflow.start_run(run_name=run_name):
        mlflow_utils.log_config_params(cfg)
        mlflow_utils.set_standard_tags(category, model_type, cfg)
        mlflow.log_param("task_type", task_type)

        # Build model
        model = build_model(model_type, cfg, task_type, layer_pooling_param)
        model.to(device)

        # Build trainer
        trainer = build_trainer(
            model_type, model, device, task_type, cfg,
            train_data, val_data, ignore_index
        )

        # Train
        trainer.train()

        for tag in ["best", "opt"]:
            ckpt_path = trainer.model_checkpoint_path.replace(".pth", f"_{tag}.pth")
            if os.path.exists(ckpt_path):
                mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")

        # Evaluate
        task_array = trainer.task_array
        for tag in ["opt", "best", "epoch"]:
            try:
                evaluator = MultiTasksModelEvaluator(
                    model=model,
                    device=device,
                    task_type=task_type,
                    checkpoints_root=cfg["paths"]["checkpoints_root"],
                    results_root=cfg["paths"]["results_root"],
                    evaluation_cfg=cfg.get("evaluation", {}),
                    dataset=test_data,
                    checkpoint_tag=tag,
                    ignore_index=ignore_index,
                )
                stats = evaluator.write_metrics()
                evaluator.write_predictions_csv()
                mlflow_utils.log_eval_stats(stats, tag, task_array)
            except Exception as e:
                logging.warning(f"Evaluation with tag='{tag}' failed: {e}")

        mlflow.log_artifacts(trainer.results_dir, artifact_path="results")

    print(f"\n  wavCSE-{model_type.upper()} completed.\n")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Run wavCSE architectural improvements"
    )
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["gbc", "tsm", "pmr", "mtrl", "original", "all"],
        help="Which model variant to run"
    )
    parser.add_argument(
        "--task_type", type=str, default="ks_si_er",
        help="Task type (e.g., ks_si_er)"
    )
    parser.add_argument(
        "--device_index", type=int, default=None,
        help="GPU device index"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (overrides each config's seed:, default 42 if neither set)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Override the config file path (relative to the repo root or "
             "absolute). Only valid with a single --model (not 'all'); used "
             "e.g. by taskrelation/02-lnp/ whose configs live outside "
             "taskrelation/configs/."
    )
    args = parser.parse_args()

    if args.config is not None and args.model == "all":
        parser.error("--config cannot be combined with --model all")

    load_dotenv(os.path.join(_REPO_ROOT, ".env"))

    # Config dir is hardcoded to taskrelation/ -- extend this once lowrank/,
    # clustering/, decomposition/ have their own config dirs (see MODEL_CATEGORY above).
    config_dir = os.path.join(os.path.dirname(__file__), "taskrelation", "configs")

    if args.model == "all":
        models_to_run = ["gbc", "tsm", "pmr", "mtrl"]
    else:
        models_to_run = [args.model]

    results = {}
    for mtype in models_to_run:
        if args.config is not None:
            # Explicit --config override (see the argparse help; used by
            # cross-folder experiments like taskrelation/02-lnp).
            config_path = os.path.join(_REPO_ROOT, args.config) \
                if not os.path.isabs(args.config) else args.config
        elif mtype in CONFIG_PATH_OVERRIDES:
            config_path = os.path.join(
                os.path.dirname(__file__), CONFIG_PATH_OVERRIDES[mtype]
            )
        else:
            config_path = os.path.join(config_dir, f"{mtype}_config.yml")
        if not os.path.exists(config_path):
            print(f"Warning: No config found for {mtype} at {config_path}")
            continue
        try:
            results[mtype] = run_single_model(
                mtype, args.task_type, config_path, args.device_index, args.seed
            )
        except Exception as e:
            print(f"Error running {mtype}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("  All model runs completed.")
    print("="*60)

    return results


if __name__ == "__main__":
    main()
