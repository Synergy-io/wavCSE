"""Task-weighted training for improvement runs; the downstream baseline stays unchanged."""

import logging
import math

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from trainer.trainer_model import MultiTasksModelTrainer


def build_task_weighted_sampler(dataset, task_array, task_sampling_weights, seed):
    """Sample tasks reproducibly without reading embedding tensors."""
    if seed is None:
        raise ValueError("A run seed is required when task sampling is enabled.")
    if not isinstance(task_sampling_weights, dict) or not task_sampling_weights:
        raise ValueError("task_sampling_weights must be a non-empty mapping.")

    unknown_tasks = set(task_sampling_weights) - set(task_array)
    if unknown_tasks:
        raise ValueError(f"Unknown task_sampling_weights keys: {sorted(unknown_tasks)}")

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
        pattern_source = component
        while isinstance(pattern_source, Subset):
            pattern_source = pattern_source.dataset
        pattern = getattr(pattern_source, "index_pattern", None)
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
    task_indices = torch.as_tensor(component_task_indices, dtype=torch.long)[
        component_indices
    ]
    sample_weights = weights_by_task[task_indices]

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )


class TaskWeightedTrainer(MultiTasksModelTrainer):
    """Add the DG-0005 training-only sampler without modifying the baseline."""

    def __init__(
        self,
        model,
        device,
        task_type,
        training_cfg,
        results_root,
        checkpoints_root,
        training_data=None,
        validation_data=None,
        ignore_index=-1,
        seed=None,
    ):
        weights = training_cfg.get("task_sampling_weights")
        sampler = None
        if weights is not None and training_data is not None:
            tasks = [task for task in task_type.split("_") if task.strip()]
            sampler = build_task_weighted_sampler(training_data, tasks, weights, seed)

        super().__init__(
            model=model,
            device=device,
            task_type=task_type,
            training_cfg=training_cfg,
            results_root=results_root,
            checkpoints_root=checkpoints_root,
            training_data=training_data,
            validation_data=validation_data,
            ignore_index=ignore_index,
        )
        if sampler is None:
            return

        # The frozen trainer creates its ordinary loader in __init__. Replace only
        # the training loader; keep its collator and all configured loader flags.
        ordinary_loader = self.train_dataloader
        self.train_dataloader = DataLoader(
            training_data,
            batch_size=ordinary_loader.batch_size,
            shuffle=False,
            sampler=sampler,
            collate_fn=ordinary_loader.collate_fn,
            drop_last=ordinary_loader.drop_last,
            pin_memory=ordinary_loader.pin_memory,
            num_workers=ordinary_loader.num_workers,
            persistent_workers=ordinary_loader.persistent_workers,
        )
        logging.info(
            "Task-weighted training sampler enabled | weights=%s | "
            "num_samples=%d | seed=%d",
            weights,
            len(training_data),
            seed,
        )
