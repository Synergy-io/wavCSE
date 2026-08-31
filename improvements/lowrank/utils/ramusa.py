"""Differentiable RAMUSA reduced-rank helpers."""

import torch


def compute_truncated_trace_penalty(task_parameter_matrix: torch.Tensor, r: int) -> torch.Tensor:
    """Return the sum of singular values after the first ``r`` values."""
    singular_values = torch.linalg.svdvals(task_parameter_matrix)
    rank_limit = singular_values.numel()
    if not 0 <= r <= rank_limit:
        raise ValueError(f"r must satisfy 0 <= r <= {rank_limit}, got {r}")
    return singular_values[r:].sum()


def update_effective_rank(task_parameter_matrix: torch.Tensor, tau: float) -> int:
    """Count singular values at or above the fixed RAMUSA threshold."""
    if tau < 0:
        raise ValueError("tau must be non-negative")
    singular_values = torch.linalg.svdvals(task_parameter_matrix)
    return int((singular_values >= tau).sum().item())
