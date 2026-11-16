"""Tests for layer-reduction result aggregation."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from improvements.lowrank import layer_reduction


class LayerReductionSummaryTests(unittest.TestCase):
    def test_record_result_replaces_existing_row(self):
        strategy = "top_layer_dropping"
        layers = list(layer_reduction.STRATEGIES[strategy]["layers"][2])
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            config_root = temp_root / "configs" / "layer_reduction"
            log_root = temp_root / "logs" / "layer_reduction"
            config_path = config_root / strategy / "ramusa_top_layers_k2_seed42.yml"
            config_path.parent.mkdir(parents=True)
            results_root = temp_root / "results"
            run_dir = results_root / "run_1"
            run_dir.mkdir(parents=True)
            checkpoint = temp_root / "best.pth"
            checkpoint.touch()

            with open(config_path, "w", encoding="utf-8") as config_file:
                yaml.safe_dump({"paths": {"results_root": str(results_root)}}, config_file)
            with open(run_dir / "layer_reduction_metrics.json", "w", encoding="utf-8") as metrics_file:
                json.dump({
                    "seed": 42,
                    "selected_layers": layers,
                    "run_id": "run_1",
                    "test_ks_acc": 0.9,
                    "test_sid_acc": 0.8,
                    "test_er_acc": 0.7,
                    "test_overall_acc": 0.82,
                    "macro_accuracy": 0.8,
                    "test_loss": 0.5,
                    "checkpoint": str(checkpoint),
                    "results_dir": str(run_dir),
                }, metrics_file)

            with patch.multiple(
                layer_reduction,
                ROOT=temp_root,
                CONFIG_ROOT=config_root,
                LOG_ROOT=log_root,
            ):
                layer_reduction.record_result(config_path)
                layer_reduction.record_result(config_path)

            with open(log_root / "master_results.csv", newline="", encoding="utf-8") as master_file:
                rows = list(csv.DictReader(master_file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "completed")
            summary = (config_root / strategy / "results_summary.log").read_text(encoding="utf-8")
            self.assertIn("| 0.8 | 0.5 | run_1 |", summary)
            self.assertIn("| completed", summary)


if __name__ == "__main__":
    unittest.main()