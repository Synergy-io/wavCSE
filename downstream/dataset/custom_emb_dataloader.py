"""
Custom DataLoader for embedding based datasets.

This module defines a lightweight wrapper around PyTorch's
DataLoader for handling pre computed embedding datasets.
It provides a custom collate function that stacks embedding
tensors and corresponding label tensors into batched inputs
suitable for downstream training.

Author: Braveenan Sritharan
Created: 2026-01-19
"""

import math

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

def build_task_weighted_sampler(
    dataset,
    task_array,
    task_sampling_weights,
    seed,
):
    """Build a deterministic sampler without loading any embedding tensors."""
    if seed is None:
        raise ValueError("A run seed is required when task sampling is enabled.")
    if not isinstance(task_sampling_weights, dict) or not task_sampling_weights:
        raise ValueError("task_sampling_weights must be a non-empty mapping.")

    unknown_tasks = set(task_sampling_weights) - set(task_array)
    if unknown_tasks:
        raise ValueError(
            f"Unknown task_sampling_weights keys: {sorted(unknown_tasks)}"
        )

    weights_by_task = torch.ones(len(task_array), dtype=torch.double)
    for task, value in task_sampling_weights.items():
        weight = float(value)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(
                f"Sampling weight for task '{task}' must be finite and positive."
            )
        weights_by_task[task_array.index(task)] = weight

    sample_indices = list(range(len(dataset)))
    root_dataset = dataset
    while isinstance(root_dataset, Subset):
        sample_indices = [root_dataset.indices[index] for index in sample_indices]
        root_dataset = root_dataset.dataset

    component_datasets = getattr(root_dataset, "datasets", None)
    if not component_datasets:
        raise TypeError(
            "Task-weighted sampling requires a CombinedDataset, optionally "
            "wrapped in torch.utils.data.Subset."
        )

    component_task_indices = []
    cumulative_sizes = []
    cumulative_size = 0
    for component in component_datasets:
        pattern = getattr(component, "index_pattern", None)
        if (
            not isinstance(pattern, str)
            or len(pattern) != len(task_array)
            or pattern.count("1") != 1
            or any(value not in ("0", "1") for value in pattern)
        ):
            raise ValueError(
                "Each component dataset must expose a one-hot index_pattern "
                "matching task_array."
            )
        component_task_indices.append(pattern.index("1"))
        cumulative_size += len(component)
        cumulative_sizes.append(cumulative_size)

    if cumulative_size != len(root_dataset):
        raise ValueError("CombinedDataset component lengths are inconsistent.")

    root_indices = torch.as_tensor(sample_indices, dtype=torch.long)
    if torch.any(root_indices < 0) or torch.any(root_indices >= cumulative_size):
        raise IndexError("Subset contains an index outside the CombinedDataset.")
    boundaries = torch.as_tensor(cumulative_sizes[:-1], dtype=torch.long)
    component_indices = torch.bucketize(root_indices, boundaries, right=True)
    task_indices = torch.as_tensor(
        component_task_indices, dtype=torch.long
    )[component_indices]
    sample_weights = weights_by_task[task_indices]

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )


class CustomEmbDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        num_workers=0,
        persistent_workers=False,
        sampler=None,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False if sampler is not None else shuffle,
            sampler=sampler,
            collate_fn=self.defined_collate,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

    def defined_collate(self, batch):
        # Expect each item as: (embedding, label_tensor)
        sequences, labels = zip(*batch)
        input_seq = torch.stack(sequences)
        labels = torch.stack(labels).long()
        return input_seq, labels
