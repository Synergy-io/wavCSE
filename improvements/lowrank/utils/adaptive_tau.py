"""Adaptive tau (threshold) manager for dynamic rank selection in RAMUSA.

This module implements the algorithm described in README_ADAPTIVE_TAU_RAMUSA.md.
The tau threshold is updated between training stages based on:
  1. Singular-value spectrum of task-parameter matrix
  2. Relative spectral gaps
  3. Validation performance of candidate structures
  4. Temporal smoothing (EMA)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AdaptiveTauManager:
    """Manages adaptive tau selection between RAMUSA training stages."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the adaptive tau manager.

        Args:
            config: Dictionary with keys:
                - min_spectral_gap: Minimum relative gap to consider (default 0.20)
                - smoothing_beta: EMA smoothing coefficient (default 0.70)
                - epsilon: Numerical constant for gap computation (default 1e-8)
                - allow_rank_zero_after_stage1: Allow r=0 after stage 1 (default True)
                - allow_zero_penalty_rank: Allow r=3 (no penalty) (default False)
                - min_validation_gain: Min improvement to accept new tau (default 0.001)
                - validation_probe: Dict with 'enabled' and 'epochs' keys
                - negative_transfer_penalty: Penalty for task collapse (default 0.5)
        """
        self.min_spectral_gap = config.get("min_spectral_gap", 0.20)
        self.smoothing_beta = config.get("smoothing_beta", 0.70)
        self.epsilon = config.get("epsilon", 1e-8)
        self.allow_rank_zero_after_stage1 = config.get("allow_rank_zero_after_stage1", True)
        self.allow_zero_penalty_rank = config.get("allow_zero_penalty_rank", False)
        self.min_validation_gain = config.get("min_validation_gain", 0.001)
        self.negative_transfer_penalty = config.get("negative_transfer_penalty", 0.5)
        
        self.validation_probe_config = config.get("validation_probe", {})
        self.validation_probe_enabled = self.validation_probe_config.get("enabled", False)
        self.validation_probe_epochs = self.validation_probe_config.get("epochs", 1)

        self.previous_tau: Optional[float] = None
        self.previous_r: Optional[int] = None
        self.stage_count = 0

    def compute_relative_spectral_gaps(
        self, singular_values: torch.Tensor
    ) -> torch.Tensor:
        """Compute relative spectral gaps between consecutive singular values.

        For three tasks (3 singular values), compute:
            g_1 = (sigma_1 - sigma_2) / (sigma_1 + eps)
            g_2 = (sigma_2 - sigma_3) / (sigma_2 + eps)

        Args:
            singular_values: Tensor of shape (R,) with singular values in descending order.
                            R is typically 3 (number of tasks).

        Returns:
            gaps: Tensor of shape (R-1,) with relative gaps.
        """
        num_singular = singular_values.shape[0]
        gaps = []

        for i in range(num_singular - 1):
            s_i = singular_values[i]
            s_i_plus_1 = singular_values[i + 1]
            gap = (s_i - s_i_plus_1) / (s_i + self.epsilon)
            gaps.append(gap.item())

        return np.array(gaps)

    def generate_candidate_taus(
        self,
        singular_values: torch.Tensor,
        gaps: np.ndarray,
        previous_tau: Optional[float],
        previous_r: Optional[int],
    ) -> List[Tuple[float, int]]:
        """Generate candidate (tau, r) pairs based on spectral gaps.

        Args:
            singular_values: Tensor of shape (R,) with ordered singular values.
            gaps: Array of relative gaps between consecutive singular values.
            previous_tau: Previous stage's threshold (for fallback candidate).
            previous_r: Previous stage's rank (for fallback candidate).

        Returns:
            candidates: List of (tau, r) tuples representing meaningful thresholds.
        """
        s = singular_values.cpu().numpy().astype(np.float64)
        candidates_dict = {}  # Use dict to deduplicate by rank

        # For each gap that exceeds the minimum threshold, generate a candidate
        for i, gap in enumerate(gaps):
            if gap >= self.min_spectral_gap:
                # Use geometric mean between s[i] and s[i+1]
                tau_candidate = np.sqrt(s[i] * s[i + 1])
                r_candidate = i + 1  # r is number of singular values >= tau

                # Only add if not already in dict
                if r_candidate not in candidates_dict:
                    candidates_dict[r_candidate] = tau_candidate
                    logger.debug(
                        f"Candidate rank r={r_candidate}: gap={gap:.4f}, "
                        f"tau={tau_candidate:.6f}"
                    )

        # Add fallback candidate (previous configuration)
        if previous_r is not None and previous_tau is not None:
            if previous_r not in candidates_dict:
                candidates_dict[previous_r] = previous_tau
                logger.debug(
                    f"Fallback candidate r={previous_r}: tau={previous_tau:.6f}"
                )

        # Convert dict to list of (tau, r) tuples
        candidates = [(tau, r) for r, tau in candidates_dict.items()]

        if not candidates:
            # If no meaningful gaps, keep previous configuration
            if previous_r is not None and previous_tau is not None:
                candidates = [(previous_tau, previous_r)]
            else:
                # First stage fallback: r=1 with geometric mean tau
                if len(s) > 1:
                    tau_fallback = float(np.sqrt(s[0] * s[1]))
                else:
                    tau_fallback = float(s[0]) if len(s) > 0 else 1.0
                candidates = [(tau_fallback, 1)]

        return candidates

    def compute_r_from_tau(
        self, singular_values: torch.Tensor, tau: float
    ) -> int:
        """Compute effective rank given tau and singular values.

        r = count of singular values >= tau

        Args:
            singular_values: Tensor of singular values in descending order.
            tau: Threshold value.

        Returns:
            r: Effective rank.
        """
        s = singular_values.cpu().numpy().astype(np.float64)
        r = np.sum(s >= tau - self.epsilon)
        return int(r)

    def clamp_tau_to_rank_interval(
        self, tau: float, singular_values: torch.Tensor, target_r: int
    ) -> float:
        """Clamp tau to preserve the target rank after smoothing.

        For rank r, the valid interval is: sigma_r >= tau > sigma_{r+1}

        Args:
            tau: Smoothed tau value.
            singular_values: Tensor of singular values.
            target_r: Desired rank to preserve.

        Returns:
            tau_clamped: Tau value clamped to the rank interval.
        """
        s = singular_values.cpu().numpy().astype(np.float64)
        R = len(s)

        if target_r <= 0 or target_r > R:
            logger.warning(f"Invalid target rank {target_r}; clamping to [1, {R}]")
            target_r = max(1, min(R, target_r))

        # Upper bound: must be <= sigma_{target_r - 1}
        if target_r == 0:
            upper = s[0] + self.epsilon  # r=0 means tau > sigma_1
        else:
            upper = s[target_r - 1]

        # Lower bound: must be > sigma_{target_r}
        if target_r >= R:
            lower = s[-1] - self.epsilon  # r=R means tau <= sigma_R
        else:
            lower = s[target_r]

        # Add safety margin
        width = max(upper - lower, self.epsilon)
        margin = max(self.epsilon, 0.01 * width)
        safe_lower = lower + margin
        safe_upper = upper - margin

        if safe_lower >= safe_upper:
            logger.warning(
                f"Singular values too close; using raw tau for rank {target_r}"
            )
            # If singular values are almost equal, use the geometric mean
            if target_r > 0 and target_r < R:
                tau_clamped = np.sqrt(s[target_r - 1] * s[target_r])
            else:
                tau_clamped = tau
        else:
            tau_clamped = float(np.clip(tau, safe_lower, safe_upper))

        # Verify rank is preserved
        r_check = self.compute_r_from_tau(singular_values, tau_clamped)
        if r_check != target_r:
            logger.warning(
                f"Clamping failed to preserve rank: "
                f"target_r={target_r}, actual_r={r_check}; using fallback"
            )
            # Use geometric mean as fallback
            if target_r > 0 and target_r < R:
                tau_clamped = float(np.sqrt(s[target_r - 1] * s[target_r]))
            else:
                tau_clamped = float(s[target_r - 1] if target_r > 0 else s[0])

        return tau_clamped

    def smooth_tau(
        self, raw_tau: float, previous_tau: Optional[float]
    ) -> float:
        """Apply EMA smoothing to the raw threshold.

        tau_smooth = beta * tau_prev + (1 - beta) * tau_raw

        Args:
            raw_tau: New threshold value.
            previous_tau: Threshold from previous stage.

        Returns:
            tau_smoothed: Smoothed threshold value.
        """
        if previous_tau is None:
            return raw_tau

        tau_smoothed = (
            self.smoothing_beta * previous_tau + 
            (1.0 - self.smoothing_beta) * raw_tau
        )
        return tau_smoothed

    def select_best_candidate_spectral_only(
        self, candidates: List[Tuple[float, int]], gaps: np.ndarray
    ) -> Tuple[float, int]:
        """Select best candidate using only spectral gap information.

        Uses the largest gap to guide selection.

        Args:
            candidates: List of (tau, r) candidate pairs.
            gaps: Array of relative spectral gaps.

        Returns:
            (tau_selected, r_selected): Best candidate based on spectral gaps.
        """
        if not candidates:
            return (self.previous_tau, self.previous_r)

        # Find the gap with largest value
        if gaps.size > 0:
            best_gap_idx = np.argmax(gaps)
            best_gap = gaps[best_gap_idx]

            if best_gap >= self.min_spectral_gap:
                # Select candidate corresponding to this gap
                # Note: r = gap_index + 1
                for tau, r in candidates:
                    if r == best_gap_idx + 1:
                        logger.debug(
                            f"Selected r={r} (gap={best_gap:.4f}) from spectral analysis"
                        )
                        return (tau, r)

        # Fallback: select candidate closest to previous_r
        if self.previous_r is not None:
            best_candidate = min(
                candidates,
                key=lambda x: abs(x[1] - self.previous_r)
            )
        else:
            best_candidate = candidates[0]

        logger.debug(
            f"Selected r={best_candidate[1]} (fallback to closest previous rank)"
        )
        return best_candidate

    def compute_normalized_improvement(
        self,
        base_metrics: Dict[str, float],
        candidate_metrics: Dict[str, float],
        task_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute normalized validation improvement score.

        Args:
            base_metrics: Baseline metrics {task: value}.
            candidate_metrics: Candidate metrics {task: value}.
            task_weights: Optional task weights for aggregation.

        Returns:
            (overall_score, delta_dict): Improvement score and per-task deltas.
        """
        if task_weights is None:
            task_weights = {"ks": 1.0, "si": 1.0, "er": 1.0}

        deltas = {}
        weighted_sum = 0.0
        weight_sum = 0.0

        for task in base_metrics.keys():
            base_val = base_metrics[task]
            cand_val = candidate_metrics.get(task, base_val)
            weight = task_weights.get(task, 1.0)

            # Assume higher is better (e.g., accuracy)
            delta = (cand_val - base_val) / (abs(base_val) + self.epsilon)
            deltas[task] = delta
            weighted_sum += weight * delta
            weight_sum += weight

        overall_score = weighted_sum / weight_sum if weight_sum > 0 else 0.0
        return overall_score, deltas

    def compute_negative_transfer_penalty(
        self, deltas: Dict[str, float]
    ) -> float:
        """Compute penalty for negative transfer (task collapse).

        Args:
            deltas: Per-task normalized improvements.

        Returns:
            penalty: Penalty term (non-positive).
        """
        min_delta = min(deltas.values()) if deltas else 0.0
        penalty = self.negative_transfer_penalty * min(0.0, min_delta)
        return penalty

    def update_tau_between_stages(
        self,
        model: nn.Module,
        singular_values: torch.Tensor,
        stage: int,
        validation_metrics_base: Optional[Dict[str, float]] = None,
        candidate_validator: Optional[Any] = None,
    ) -> Tuple[float, int]:
        """Update tau dynamically between training stages.

        Main entry point for adaptive tau selection.

        Args:
            model: The RAMUSA model (for building W_tasks if needed).
            singular_values: Current singular values of task-parameter matrix.
            stage: Current stage number (0-indexed).
            validation_metrics_base: Baseline validation metrics at stage boundary.
            candidate_validator: Optional callback to validate candidates.

        Returns:
            (tau_final, r_final): Updated threshold and rank for next stage.
        """
        self.stage_count = stage

        # Stage 1: Use r=0, no threshold selection yet
        if stage == 0:
            self.previous_r = 0
            self.previous_tau = None
            logger.info("Stage 1: Using r=0 (all singular values penalized)")
            return (None, 0)

        # Compute spectral gaps
        gaps = self.compute_relative_spectral_gaps(singular_values)
        logger.debug(f"Spectral gaps: {gaps}")

        # Generate candidate (tau, r) pairs
        candidates = self.generate_candidate_taus(
            singular_values=singular_values,
            gaps=gaps,
            previous_tau=self.previous_tau,
            previous_r=self.previous_r,
        )
        logger.info(f"Candidate (tau, r) pairs: {candidates}")

        # Select best candidate
        if self.validation_probe_enabled and candidate_validator is not None:
            # Use validation probes (expensive but accurate)
            logger.info("Running validation probes for candidate selection...")
            selected_tau, selected_r, improvement = candidate_validator(
                candidates=candidates,
                base_metrics=validation_metrics_base,
            )
            logger.info(f"Validation probe selected r={selected_r}, improvement={improvement:.6f}")

            # Check minimum validation gain threshold
            if improvement < self.min_validation_gain:
                logger.info(
                    f"Improvement {improvement:.6f} < min_validation_gain "
                    f"{self.min_validation_gain}; keeping previous configuration"
                )
                selected_tau = self.previous_tau
                selected_r = self.previous_r
        else:
            # Use spectral gaps only
            logger.info("Using spectral analysis for candidate selection...")
            selected_tau, selected_r = self.select_best_candidate_spectral_only(
                candidates, gaps
            )

        # Smooth the selected tau
        if selected_tau is None:
            logger.warning("Selected tau is None; using geometric mean fallback")
            # This should not happen, but handle gracefully
            s_vals = singular_values.cpu().numpy().astype(np.float64)
            selected_tau = float(np.sqrt(s_vals[0] * s_vals[1]) if len(s_vals) > 1 else s_vals[0])
            logger.debug(f"Fallback tau computed: {selected_tau:.6f}")
        
        if self.previous_tau is not None:
            tau_smoothed = self.smooth_tau(selected_tau, self.previous_tau)
            logger.debug(
                f"Smoothing: prev_tau={self.previous_tau:.6f}, "
                f"selected_tau={selected_tau:.6f}, "
                f"smoothed_tau={tau_smoothed:.6f}"
            )
        else:
            tau_smoothed = selected_tau
            logger.debug(f"First tau after stage 1: tau={tau_smoothed:.6f}")

        # Clamp tau to preserve selected rank
        tau_final = self.clamp_tau_to_rank_interval(
            tau=tau_smoothed,
            singular_values=singular_values,
            target_r=selected_r,
        )

        # Verify rank is preserved
        r_final = self.compute_r_from_tau(singular_values, tau_final)
        if r_final != selected_r:
            logger.error(
                f"Failed to preserve rank: selected_r={selected_r}, "
                f"final_r={r_final}; this should not happen"
            )

        # Update state for next iteration
        self.previous_tau = tau_final
        self.previous_r = r_final

        logger.info(
            f"Stage {stage+1}: Updated tau={tau_final:.6f}, r={r_final} "
            f"(previous: tau={self.previous_tau}, r={self.previous_r})"
        )

        return (tau_final, r_final)

    def get_config_dict(self) -> Dict[str, Any]:
        """Export current configuration as dictionary."""
        return {
            "min_spectral_gap": self.min_spectral_gap,
            "smoothing_beta": self.smoothing_beta,
            "epsilon": self.epsilon,
            "allow_rank_zero_after_stage1": self.allow_rank_zero_after_stage1,
            "allow_zero_penalty_rank": self.allow_zero_penalty_rank,
            "min_validation_gain": self.min_validation_gain,
            "negative_transfer_penalty": self.negative_transfer_penalty,
            "validation_probe_enabled": self.validation_probe_enabled,
            "validation_probe_epochs": self.validation_probe_epochs,
        }
