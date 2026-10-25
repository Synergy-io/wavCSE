"""Public TR-0013 configs vary only the published p-MSSL lambda_2 axis.

The preflight, authorization and run records live in private wavcse-research.
"""

import unittest
from pathlib import Path

import yaml

from downstream.utils.parse_transformer_layers import parse_transformer_layers
from improvements import mlflow_utils

ROOT = Path(__file__).resolve().parents[3]
CANDIDATE_DIR = ROOT / "improvements/taskrelation/experiments/TR-0013/configs"
CONTROL_DIR = ROOT / "improvements/taskrelation/experiments/TR-0007/configs"
MSSL_SOURCE = ROOT / "improvements/taskrelation/04-mssl/mssl_config.yml"
LAMBDA_2 = {"mssl-l2-0p01": 0.01, "mssl-l2-0p1": 0.1}
ARMS = (*LAMBDA_2, "classical-mtrl", "wavcse-baseline")
ALLOWED_DIFFS = {
    ("paths", "results_root"), ("paths", "checkpoints_root"),
    ("device", "index"), ("model", "mssl_lambda_2"),
    ("mssl", "lambda_2_selection"),
}


def load(path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def differences(left, right, prefix=()):
    if isinstance(left, dict) and isinstance(right, dict):
        changed = set()
        for key in set(left) | set(right):
            child = prefix + (key,)
            changed |= ({child} if key not in left or key not in right
                        else differences(left[key], right[key], child))
        return changed
    return {prefix} if left != right else set()


def unexpected(changed):
    return {key for key in changed if key not in ALLOWED_DIFFS and key[0] != "research"}


class CandidateAxisTests(unittest.TestCase):
    def test_published_axis_uses_exactly_the_two_registered_values(self):
        observed = {name: load(CANDIDATE_DIR / (name + ".yml"))["model"]["mssl_lambda_2"]
                    for name in LAMBDA_2}
        self.assertEqual(observed, LAMBDA_2)

    def test_only_registered_axes_differ_from_the_source_arm(self):
        source = load(MSSL_SOURCE)
        for arm in LAMBDA_2:
            with self.subTest(arm=arm):
                config = load(CANDIDATE_DIR / (arm + ".yml"))
                self.assertEqual(unexpected(differences(source, config)), set())
                self.assertEqual(config["mssl"]["lambda_2_selection"],
                                 "validation-selected-over-published-grid-0.01-0.1")
                self.assertEqual(config["model"]["mssl_lambda_0"], 1.0)
                self.assertEqual(config["model"]["mssl_lambda_1"], 0.0)
                self.assertEqual(config["mssl"]["omega_update_frequency"], 1)

    def test_two_candidates_differ_only_in_lambda_2(self):
        first = load(CANDIDATE_DIR / "mssl-l2-0p01.yml")
        second = load(CANDIDATE_DIR / "mssl-l2-0p1.yml")
        self.assertEqual(unexpected(differences(first, second)), set())
        self.assertIn(("model", "mssl_lambda_2"), differences(first, second))


class FrozenProtocolTests(unittest.TestCase):
    def test_all_arms_use_identical_layer_selection_and_training_conditions(self):
        configs = {name: load(CANDIDATE_DIR / (name + ".yml")) for name in ARMS}
        for name, config in configs.items():
            with self.subTest(arm=name):
                self.assertEqual(config["upstream"]["model_type"], "wavlm_large")
                self.assertEqual(list(parse_transformer_layers(
                    config["upstream"]["selected_transformer_layers"], "wavlm_large"
                )), list(range(25)))
                self.assertEqual(config["pooling"]["frame_pooling_type"], "mean")
                self.assertEqual(config["pooling"]["layer_pooling_type"], "smp")
                self.assertEqual(config["pooling"]["layer_pooling_param"], 0.5)
                self.assertEqual(config["training"]["num_epochs"], 30)
                self.assertEqual(config["training"]["batch_size"], 2048)
                self.assertEqual(config["seed"], 42)
                self.assertEqual(config["research"]["study_id"], "TR-0013")
                self.assertEqual(config["research"]["stage"], "screen")
                self.assertTrue(mlflow_utils.resolve_run_note(config))

    def test_controls_remain_their_TR0007_counterparts(self):
        for name in ("classical-mtrl", "wavcse-baseline"):
            with self.subTest(arm=name):
                current = load(CANDIDATE_DIR / (name + ".yml"))
                source = load(CONTROL_DIR / (name + ".yml"))
                self.assertEqual(unexpected(differences(current, source)), set())
                self.assertEqual(current["model"], source["model"])
                self.assertEqual(current["training"], source["training"])
