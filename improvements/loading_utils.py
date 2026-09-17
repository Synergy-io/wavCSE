"""
Shared data-loading device helper for wavCSE improvements run scripts.

Reusable by any owner folder under improvements/ (base, taskrelation,
clustering, lowrank, decomposition) via run_improvements.py or a standalone
script, without requiring changes to downstream/.

Author: Kevin Sanjula
"""

import torch


def get_loader_device():
    """Device to pass into LoadEmbedding (and therefore into each dataset's
    torch.load(..., map_location=...) per sample) -- always CPU, regardless
    of the actual training device.

    MultiTasksModelTrainer and MultiTasksModelEvaluator already move each
    batch to the training device explicitly right before use
    (trainer/trainer_model.py's `x = batch[0].to(self.device)`, and the
    evaluator's equivalent), so the embedding load itself doesn't need to
    target the GPU. Loading to CPU instead lets the DataLoader use
    num_workers > 0 safely -- CUDA contexts can't be shared across forked
    worker processes, which is why downstream/configs/build_model.yml ships
    num_workers: 0 with a comment flagging exactly this risk. Loading to
    CPU sidesteps it entirely, independent of what num_workers is set to.
    """
    return torch.device("cpu")
