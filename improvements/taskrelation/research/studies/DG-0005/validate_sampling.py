"""Validate DG-0005 A1 sampling composition without loading embeddings."""

import os
import sys
from collections import Counter
from pathlib import Path

import torch


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[4]
DOWNSTREAM_DIR = REPO_ROOT / "downstream"
IMPROVEMENTS_DIR = REPO_ROOT / "improvements"
for path in (REPO_ROOT, DOWNSTREAM_DIR, IMPROVEMENTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset.custom_emb_dataloader import build_task_weighted_sampler
from dataset.load_embedding import LoadEmbedding
from loading_utils import get_loader_device
from utils.load_config import load_config
from utils.parse_transformer_layers import parse_transformer_layers


def main():
    config_path = STUDY_DIR / "configs" / "a1_er_weighted.yml"
    cfg = load_config(str(config_path))
    layers = parse_transformer_layers(
        cfg["upstream"]["selected_transformer_layers"],
        cfg["upstream"]["model_type"],
    )
    loader = LoadEmbedding(
        root_data_path=os.path.expanduser(cfg["paths"]["root_data_path"]),
        root_emb_path=os.path.expanduser(cfg["paths"]["root_emb_path"]),
        upstream_model_type=cfg["upstream"]["model_type"],
        frame_pooling_type=cfg["pooling"]["frame_pooling_type"],
        frame_pooling_param=cfg["pooling"].get("frame_pooling_param"),
        transformer_layer_array=layers,
        device=get_loader_device(),
    )
    training_data, _, _ = loader.load_embedding(
        task_type="ks_si_er",
        subset_percentage=cfg["dataset"]["subset_percentage"],
    )
    sampler = build_task_weighted_sampler(
        training_data,
        ["ks", "si", "er"],
        cfg["training"]["task_sampling_weights"],
        cfg["seed"],
    )
    root = training_data.dataset
    boundaries = [len(root.datasets[0]), len(root.datasets[0]) + len(root.datasets[1])]
    sampled_tasks = Counter()
    for index in sampler:
        root_index = training_data.indices[index]
        if root_index < boundaries[0]:
            sampled_tasks["ks"] += 1
        elif root_index < boundaries[1]:
            sampled_tasks["si"] += 1
        else:
            sampled_tasks["er"] += 1
    total = sum(sampled_tasks.values())
    payload = {
        "dataset_samples": len(training_data),
        "sampler_samples": sampler.num_samples,
        "task_counts": dict(sampled_tasks),
        "expected_per_batch": {
            task: count * cfg["training"]["batch_size"] / total
            for task, count in sampled_tasks.items()
        },
        "er_to_ks_ratio": sampled_tasks["er"] / sampled_tasks["ks"],
    }
    print(payload)
    if sampler.num_samples != len(training_data):
        raise RuntimeError("Sampler changed optimizer exposure")
    if payload["er_to_ks_ratio"] < 0.5:
        raise RuntimeError("ER sampling contribution missed the pre-registered gate")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
