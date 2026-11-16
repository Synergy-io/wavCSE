"""Generate, validate, and summarize RAMUSA layer-reduction experiments."""

import argparse
import copy
import csv
import fcntl
import json
import os
import re
import tempfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = ROOT / "configs" / "layer_reduction"
LOG_ROOT = ROOT / "logs" / "layer_reduction"
BASE_CONFIG = ROOT / "configs" / "ramusa_config.yml"
MASTER_FIELDS = [
    "strategy", "k", "seed", "gpu", "preserved_layers", "removed_layers",
    "num_preserved_layers", "test_ks_acc", "test_sid_acc", "test_er_acc",
    "test_overall_acc", "macro_accuracy", "test_loss", "run_id", "results_dir",
    "checkpoint", "status",
]

STRATEGIES = {
    "top_layer_dropping": {
        "display": "Top-layer dropping",
        "prefix": "top_layers",
        "gpu": 0,
        "layers": {
            2: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22),
            4: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
            6: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18),
            8: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
        },
    },
    "even_alternate_dropping": {
        "display": "Even alternate dropping",
        "prefix": "even_alt",
        "gpu": 0,
        "layers": {
            2: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23),
            4: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 21, 23),
            6: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 19, 21, 23),
            8: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 17, 19, 21, 23),
        },
    },
    "odd_alternate_dropping": {
        "display": "Odd alternate dropping",
        "prefix": "odd_alt",
        "gpu": 0,
        "layers": {
            2: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24),
            4: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24),
            6: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24),
            8: (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24),
        },
    },
    "symmetric_dropping": {
        "display": "Symmetric dropping",
        "prefix": "symmetric",
        "gpu": 1,
        "layers": {
            2: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24),
            4: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24),
            6: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24),
            8: (0, 1, 2, 3, 4, 5, 6, 7, 8, 17, 18, 19, 20, 21, 22, 23, 24),
        },
    },
    "bottom_layer_dropping": {
        "display": "Bottom-layer dropping",
        "prefix": "bottom_layers",
        "gpu": 1,
        "layers": {
            2: (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24),
            4: (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24),
            6: (6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24),
            8: (8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24),
        },
    },
    "contribution_based_dropping": {
        "display": "Contribution-based dropping",
        "prefix": "contribution",
        "gpu": 1,
        "layers": {
            2: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24),
            4: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24),
            6: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21, 22, 23, 24),
            8: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 22, 23, 24),
        },
    },
}


def _load_yaml(path):
    with open(path, encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def _config_path(strategy, k):
    prefix = STRATEGIES[strategy]["prefix"]
    return CONFIG_ROOT / strategy / f"ramusa_{prefix}_k{k}_seed42.yml"


def generate_configs():
    baseline = _load_yaml(BASE_CONFIG)
    baseline["seed"] = 42
    if baseline["upstream"]["selected_transformer_layers"] != "all":
        raise ValueError("Baseline config must retain all layers")

    for strategy, spec in STRATEGIES.items():
        strategy_dir = CONFIG_ROOT / strategy
        strategy_dir.mkdir(parents=True, exist_ok=True)
        for k, layers in spec["layers"].items():
            path = _config_path(strategy, k)
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing config: {path}")
            config = copy.deepcopy(baseline)
            config["seed"] = 42
            config["device"]["index"] = spec["gpu"]
            config["paths"]["results_root"] = (
                f"results_ramusa{'_gpu1' if spec['gpu'] == 1 else ''}/"
                f"layer_reduction/{strategy}/k{k}_seed42"
            )
            config["paths"]["checkpoints_root"] = (
                f"checkpoints_ramusa{'_gpu1' if spec['gpu'] == 1 else ''}/"
                f"layer_reduction/{strategy}/k{k}_seed42"
            )
            config["upstream"]["selected_transformer_layers"] = ",".join(
                str(layer) for layer in layers
            )
            config["run_name"] = f"ramusa_{strategy}_k{k}_seed42"
            kept = ",".join(str(layer) for layer in layers)
            removed = ",".join(str(layer) for layer in sorted(set(range(25)) - set(layers)))
            comment = (
                f"# Strategy: {spec['display']}; k={k}; seed=42; GPU={spec['gpu']}\n"
                f"# Preserved layers: {kept}\n# Removed layers: {removed}\n"
            )
            with open(path, "w", encoding="utf-8") as config_file:
                config_file.write(comment)
                yaml.safe_dump(config, config_file, sort_keys=False)

        summary = strategy_dir / "results_summary.log"
        if not summary.exists():
            _write_strategy_summary(strategy, {})
        gpu_log_dir = LOG_ROOT / f"gpu{spec['gpu']}" / strategy
        gpu_log_dir.mkdir(parents=True, exist_ok=True)


def _expected_paths(strategy, k):
    gpu = STRATEGIES[strategy]["gpu"]
    gpu_suffix = "_gpu1" if gpu == 1 else ""
    return (
        f"results_ramusa{gpu_suffix}/layer_reduction/{strategy}/k{k}_seed42",
        f"checkpoints_ramusa{gpu_suffix}/layer_reduction/{strategy}/k{k}_seed42",
    )


def validate_configs():
    baseline = _load_yaml(BASE_CONFIG)
    if baseline.get("seed") != 42:
        raise ValueError("Baseline seed must be exactly 42")
    if baseline["upstream"].get("selected_transformer_layers") != "all":
        raise ValueError("Baseline must remain the all-layer config")

    configs = sorted(CONFIG_ROOT.rglob("*.yml"))
    if len(configs) != 24:
        raise ValueError(f"Expected exactly 24 experiment YAMLs, found {len(configs)}")
    results_roots = set()
    checkpoint_roots = set()
    gpu_counts = {0: 0, 1: 0}

    for strategy, spec in STRATEGIES.items():
        summary = CONFIG_ROOT / strategy / "results_summary.log"
        if not summary.is_file():
            raise FileNotFoundError(f"Missing required summary file: {summary}")
        for k, expected_layers in spec["layers"].items():
            path = _config_path(strategy, k)
            if not path.is_file():
                raise FileNotFoundError(f"Missing config: {path}")
            config = _load_yaml(path)
            if config.get("seed") != 42:
                raise ValueError(f"{path}: seed must be exactly 42")
            raw_layers = config["upstream"].get("selected_transformer_layers")
            if str(raw_layers).strip().lower() == "all":
                raise ValueError(f"{path}: all-layer selection is not allowed")
            try:
                layers = [int(item.strip()) for item in str(raw_layers).split(",")]
            except ValueError as error:
                raise ValueError(f"{path}: selected layers must be comma-separated integers") from error
            if len(layers) != len(set(layers)):
                raise ValueError(f"{path}: selected layers contain duplicates")
            if any(layer < 0 or layer > 24 for layer in layers):
                raise ValueError(f"{path}: selected layers must be between 0 and 24")
            if layers != sorted(layers):
                raise ValueError(f"{path}: selected layers must be in ascending order")
            if tuple(layers) != expected_layers:
                raise ValueError(f"{path}: selected layers do not match the experiment specification")
            if len(layers) != 25 - k or len(set(range(25)) - set(layers)) != k:
                raise ValueError(f"{path}: incorrect retained/dropped layer count for k={k}")
            if config["device"].get("type") != "cuda" or config["device"].get("index") != spec["gpu"]:
                raise ValueError(f"{path}: incorrect GPU assignment")
            expected_results, expected_checkpoints = _expected_paths(strategy, k)
            if config["paths"].get("results_root") != expected_results:
                raise ValueError(f"{path}: results_root is not isolated as specified")
            if config["paths"].get("checkpoints_root") != expected_checkpoints:
                raise ValueError(f"{path}: checkpoints_root is not isolated as specified")
            if config.get("run_name") != f"ramusa_{strategy}_k{k}_seed42":
                raise ValueError(f"{path}: missing unique MLflow run_name")

            comparable = copy.deepcopy(config)
            comparable.pop("run_name", None)
            comparable["device"]["index"] = baseline["device"]["index"]
            comparable["paths"]["results_root"] = baseline["paths"]["results_root"]
            comparable["paths"]["checkpoints_root"] = baseline["paths"]["checkpoints_root"]
            comparable["upstream"]["selected_transformer_layers"] = baseline["upstream"]["selected_transformer_layers"]
            if comparable != baseline:
                raise ValueError(f"{path}: non-experiment settings differ from the baseline")

            if expected_results in results_roots or expected_checkpoints in checkpoint_roots:
                raise ValueError(f"{path}: duplicate result or checkpoint root")
            results_roots.add(expected_results)
            checkpoint_roots.add(expected_checkpoints)
            gpu_counts[spec["gpu"]] += 1

    if gpu_counts != {0: 12, 1: 12}:
        raise ValueError(f"Expected 12 configs on each GPU, found {gpu_counts}")
    print("Preflight passed: all 24 configs, layer sets, GPU assignments, and baseline settings match.")


def _atomic_write(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as temp_file:
        temp_file.write(contents)
        temp_path = Path(temp_file.name)
    os.replace(temp_path, path)


def _write_strategy_summary(strategy, rows):
    spec = STRATEGIES[strategy]
    lines = [f"Strategy: {spec['display']}", "Seed: 42", "", (
        "k | preserved_layers | test_ks_acc | test_sid_acc | test_er_acc | "
        "test_overall_acc | macro_accuracy | test_loss | run_id | checkpoint | status"
    )]
    for k, layers in spec["layers"].items():
        record = rows.get((strategy, str(k), "42"), {})
        kept = ",".join(str(layer) for layer in layers)
        fields = [
            str(k), kept,
            record.get("test_ks_acc", "-"), record.get("test_sid_acc", "-"),
            record.get("test_er_acc", "-"), record.get("test_overall_acc", "-"),
            record.get("macro_accuracy", "-"), record.get("test_loss", "-"),
            record.get("run_id", "-"), record.get("checkpoint", "-"),
            record.get("status", "pending"),
        ]
        lines.append(" | ".join(fields))
    summary_path = CONFIG_ROOT / strategy / "results_summary.log"
    _atomic_write(summary_path, "\n".join(lines) + "\n")


def _read_master(master_path):
    if not master_path.is_file():
        return {}
    with open(master_path, newline="", encoding="utf-8") as master_file:
        return {
            (row["strategy"], row["k"], row["seed"]): row
            for row in csv.DictReader(master_file)
        }


def record_result(config_path):
    config_path = Path(config_path).resolve()
    strategy = config_path.parent.name
    match = re.search(r"_k(2|4|6|8)_seed42\.yml$", config_path.name)
    if strategy not in STRATEGIES or match is None:
        raise ValueError(f"Not a layer-reduction config: {config_path}")
    k = int(match.group(1))
    config = _load_yaml(config_path)
    results_root = (ROOT / config["paths"]["results_root"]).resolve()
    metric_files = list(results_root.glob("*/layer_reduction_metrics.json"))
    if not metric_files:
        raise FileNotFoundError(f"No structured evaluation metrics found under {results_root}")
    metrics_path = max(metric_files, key=lambda path: path.stat().st_mtime_ns)
    with open(metrics_path, encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)
    if metrics.get("seed") != 42 or metrics.get("selected_layers") != list(STRATEGIES[strategy]["layers"][k]):
        raise ValueError(f"Metrics metadata does not match {config_path}")
    if not Path(metrics["checkpoint"]).is_file():
        raise FileNotFoundError(f"Best checkpoint does not exist: {metrics['checkpoint']}")

    preserved = STRATEGIES[strategy]["layers"][k]
    removed = sorted(set(range(25)) - set(preserved))
    new_row = {
        "strategy": strategy,
        "k": str(k),
        "seed": "42",
        "gpu": str(STRATEGIES[strategy]["gpu"]),
        "preserved_layers": ",".join(map(str, preserved)),
        "removed_layers": ",".join(map(str, removed)),
        "num_preserved_layers": str(len(preserved)),
        "test_ks_acc": str(metrics["test_ks_acc"]),
        "test_sid_acc": str(metrics["test_sid_acc"]),
        "test_er_acc": str(metrics["test_er_acc"]),
        "test_overall_acc": str(metrics["test_overall_acc"]),
        "macro_accuracy": str(metrics["macro_accuracy"]),
        "test_loss": str(metrics["test_loss"]),
        "run_id": str(metrics["run_id"]),
        "results_dir": str(metrics["results_dir"]),
        "checkpoint": str(metrics["checkpoint"]),
        "status": "completed",
    }

    master_path = LOG_ROOT / "master_results.csv"
    lock_path = LOG_ROOT / ".master_results.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        rows = _read_master(master_path)
        key = (strategy, str(k), "42")
        rows[key] = new_row
        with tempfile.NamedTemporaryFile(
            "w", newline="", encoding="utf-8", dir=master_path.parent, delete=False
        ) as temp_file:
            writer = csv.DictWriter(temp_file, fieldnames=MASTER_FIELDS)
            writer.writeheader()
            writer.writerows(sorted(rows.values(), key=lambda row: (row["gpu"], row["strategy"], int(row["k"]))))
            temp_path = Path(temp_file.name)
        os.replace(temp_path, master_path)
        strategy_rows = {
            row_key: row for row_key, row in rows.items() if row_key[0] == strategy
        }
        _write_strategy_summary(strategy, strategy_rows)
        fcntl.flock(lock_file, fcntl.LOCK_UN)
    print(f"Recorded {strategy} k={k}: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("generate", help="Create the 24 configs and required summary files")
    subparsers.add_parser("validate", help="Validate all experiment configs against the baseline")
    record_parser = subparsers.add_parser("record", help="Update summary files from structured test metrics")
    record_parser.add_argument("--config", required=True)
    args = parser.parse_args()
    if args.command == "generate":
        generate_configs()
        print("Generated 24 layer-reduction configs and six strategy summaries.")
    elif args.command == "validate":
        validate_configs()
    else:
        record_result(args.config)


if __name__ == "__main__":
    main()