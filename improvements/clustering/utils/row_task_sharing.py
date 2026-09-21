"""Frozen closest-pair row sharing for the NCMTL row-level ablation."""

import csv
import json
import os

import torch

from .candidate_row_distances import compute_candidate_row_distances


class RowTaskSharing:
    """Assign each corresponding candidate row to its closest task pair."""

    PAIRS = (("ks_si", 0, 1), ("ks_er", 0, 2), ("si_er", 1, 2))

    def __init__(self, results_dir: str, task_names):
        if list(task_names) != ["ks", "si", "er"]:
            raise ValueError("Row task sharing requires tasks ['ks', 'si', 'er']")
        self.task_names = list(task_names)
        self.assignments = None
        self.assignment_epoch = None
        self.assignment_csv_path = os.path.join(
            results_dir, "row_pair_assignments.csv"
        )
        self.assignment_summary_path = os.path.join(
            results_dir, "row_pair_assignment_summary.json"
        )
        self.stability_path = os.path.join(
            results_dir, "row_pair_assignment_stability.csv"
        )
        with open(self.stability_path, "w", newline="") as stability_file:
            csv.writer(stability_file).writerow(
                [
                    "epoch", "rows", "matching_frozen_assignment",
                    "match_rate", "observed_ks_si", "observed_ks_er",
                    "observed_si_er",
                ]
            )

    @property
    def initialized(self) -> bool:
        return self.assignments is not None

    def initialize(self, weights, epoch: int) -> None:
        distances = compute_candidate_row_distances(weights, self.task_names)
        stacked = torch.stack([distances[name] for name, _, _ in self.PAIRS])
        sorted_distances, sorted_indices = torch.sort(stacked, dim=0)
        self.assignments = sorted_indices[0].to(dtype=torch.long)
        self.assignment_epoch = int(epoch)

        with open(self.assignment_csv_path, "w", newline="") as assignment_file:
            writer = csv.writer(assignment_file)
            writer.writerow(
                [
                    "row", "pair_id", "selected_pair", "selected_distance",
                    "second_distance", "absolute_margin", "relative_margin",
                    "ks_si_distance", "ks_er_distance", "si_er_distance",
                ]
            )
            for row_index in range(stacked.shape[1]):
                selected = int(self.assignments[row_index].item())
                best = float(sorted_distances[0, row_index].item())
                second = float(sorted_distances[1, row_index].item())
                margin = second - best
                writer.writerow(
                    [
                        row_index,
                        selected,
                        self.PAIRS[selected][0],
                        best,
                        second,
                        margin,
                        margin / second if second > 0.0 else 0.0,
                        float(stacked[0, row_index].item()),
                        float(stacked[1, row_index].item()),
                        float(stacked[2, row_index].item()),
                    ]
                )

        counts = self.assignment_counts()
        total = int(self.assignments.numel())
        summary = {
            "strategy": "closest_pair",
            "assignment_epoch": self.assignment_epoch,
            "frozen": True,
            "num_rows": total,
            "pair_ids": {name: pair_id for pair_id, (name, _, _) in enumerate(self.PAIRS)},
            "counts": counts,
            "proportions": {name: count / total for name, count in counts.items()},
        }
        with open(self.assignment_summary_path, "w") as summary_file:
            json.dump(summary, summary_file, indent=2)

    def assignment_counts(self) -> dict[str, int]:
        if not self.initialized:
            return {name: 0 for name, _, _ in self.PAIRS}
        return {
            name: int(torch.sum(self.assignments == pair_id).item())
            for pair_id, (name, _, _) in enumerate(self.PAIRS)
        }

    @torch.no_grad()
    def share(self, weights) -> None:
        if not self.initialized:
            return
        for pair_id, (_, first, second) in enumerate(self.PAIRS):
            row_mask = (self.assignments == pair_id).to(weights[first].device)
            if not bool(torch.any(row_mask)):
                continue
            row_indices = torch.nonzero(row_mask, as_tuple=False).squeeze(1)
            center = (weights[first][row_mask] + weights[second][row_mask]) / 2.0
            weights[first].index_copy_(0, row_indices, center)
            weights[second].index_copy_(0, row_indices, center)

    def cluster_loss(self, weights) -> torch.Tensor:
        if not self.initialized:
            return weights[0].new_zeros(())
        loss = weights[0].new_zeros(())
        for pair_id, (_, first, second) in enumerate(self.PAIRS):
            row_mask = (self.assignments == pair_id).to(weights[first].device)
            if not bool(torch.any(row_mask)):
                continue
            center = (
                (weights[first][row_mask] + weights[second][row_mask]) / 2.0
            ).detach()
            loss = loss + torch.sum((weights[first][row_mask] - center) ** 2)
            loss = loss + torch.sum((weights[second][row_mask] - center) ** 2)
        return loss

    def record_observed_stability(self, epoch: int, raw_values: dict) -> None:
        """Compare current closest pairs with the frozen warm-up assignments."""
        if not self.initialized:
            return
        stacked = torch.tensor(
            [raw_values[name] for name, _, _ in self.PAIRS], dtype=torch.float32
        )
        observed = torch.argmin(stacked, dim=0)
        matches = int(torch.sum(observed == self.assignments).item())
        counts = {
            name: int(torch.sum(observed == pair_id).item())
            for pair_id, (name, _, _) in enumerate(self.PAIRS)
        }
        total = int(observed.numel())
        with open(self.stability_path, "a", newline="") as stability_file:
            csv.writer(stability_file).writerow(
                [
                    epoch, total, matches, matches / total,
                    counts["ks_si"], counts["ks_er"], counts["si_er"],
                ]
            )
