"""Executable validation of the registered TR-0013 screen.

TR-0013 tests *one* independent variable -- the `lambda_2` policy of the
faithful published p-MSSL arm (researcher-fixed 0.01 in TR-0007 -> validation-
selected over the paper's two smallest grid values {0.01, 0.1}). Everything
else is the frozen protocol TR-0007 ran. So the tests here are mostly
*difference* tests: they resolve each of the four registered configs and prove
that the only keys that moved are the ones the study registers -- the `lambda_2`
value, its selection label, the study-local output roots and the study identity
block -- against the configs TR-0007 actually ran.

They also pin the plan's own contract: four arms at seed 42, all 25 wavLM
layers, the canonical embedding set and layout, the declared output contract,
one Secure worker inside the granted envelope, and no confirmation stage.
"""

import json
import os
import sys
import unittest

import yaml

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(TESTS_DIR, "..", "..", "..", ".."))
for path in (REPO_ROOT, os.path.join(REPO_ROOT, "downstream")):
    if path not in sys.path:
        sys.path.insert(0, path)

from downstream.utils.parse_transformer_layers import parse_transformer_layers  # noqa: E402

from improvements import mlflow_utils  # noqa: E402
from improvements.compute import artifacts, jobspec  # noqa: E402

STUDY_DIR = os.path.join(REPO_ROOT, "improvements", "taskrelation", "research",
                         "studies", "TR-0013")
PLAN_PATH = os.path.join(STUDY_DIR, "compute", "plan.json")
INPUTS_PATH = os.path.join(STUDY_DIR, "compute", "inputs.json")
PLAN_TEXT = open(os.path.join(STUDY_DIR, "PLAN.md"), "r", encoding="utf-8").read()
AUTHORIZATION_PATH = os.path.join(
    REPO_ROOT, "improvements", "taskrelation", "research", "authorizations",
    "TR-0013.yaml"
)
MSSL_ARM_CONFIG = os.path.join(REPO_ROOT, "improvements", "taskrelation", "04-mssl",
                               "mssl_config.yml")
TR0007_STUDY_DIR = os.path.join(REPO_ROOT, "improvements", "taskrelation", "research",
                                "studies", "TR-0007")

WAVLM_LARGE_LAYERS = 25
CANDIDATE_LAMBDA_2 = {"mssl-l2-0p01": 0.01, "mssl-l2-0p1": 0.1}
# The only paths allowed to move between TR-0007's configs and TR-0013's:
# the study-local output roots, the registered lambda_2, its selection label,
# and the study identity block.
ALLOWED_DIFFS = {
    ("paths", "results_root"),
    ("paths", "checkpoints_root"),
    # The architecture folder's config pins device index 1 for the shared
    # workstation; a study config that runs on a one-GPU worker carries 0, and
    # the job's own `--device_index 0` overrides it either way. TR-0007's study
    # configs made the same change.
    ("device", "index"),
    ("model", "mssl_lambda_2"),
    ("mssl", "lambda_2_selection"),
}


def _load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _diff(left, right, prefix=()):
    """Key paths whose values differ between two nested mappings."""
    if isinstance(left, dict) and isinstance(right, dict):
        paths = set()
        for key in set(left) | set(right):
            child = tuple(list(prefix) + [key])
            if key not in left or key not in right:
                paths.add(child)
                continue
            paths |= _diff(left[key], right[key], child)
        return paths
    if left != right:
        return {prefix}
    return set()


def _unexpected(changed):
    """Changed key paths this study does not register (identity block exempt)."""
    return {path for path in changed
            if path not in ALLOWED_DIFFS and path[0] != "research"}


class CandidateAxisTests(unittest.TestCase):
    def setUp(self):
        self.plan = jobspec.load_plan(PLAN_PATH)
        self.configs = {
            arm["arm"]: _load(os.path.join(REPO_ROOT, arm["config"]))
            for arm in self.plan["arms"]
        }
        self.source = _load(MSSL_ARM_CONFIG)

    def test_exactly_two_candidate_values_from_the_published_grid(self):
        candidates = {arm: self.configs[arm]["model"]["mssl_lambda_2"]
                      for arm in CANDIDATE_LAMBDA_2}
        self.assertEqual(candidates, CANDIDATE_LAMBDA_2)
        self.assertEqual(sorted(candidates.values()), [0.01, 0.1])
        for arm, config in self.configs.items():
            self.assertEqual(
                config.get("mssl", {}).get("lambda_2_selection",
                                           "not-applicable-no-relation-penalty"),
                ("validation-selected-over-published-grid-0.01-0.1"
                 if arm in CANDIDATE_LAMBDA_2
                 else "not-applicable-no-relation-penalty"),
            )

    def test_candidates_differ_from_tr0007s_candidate_only_in_registered_axes(self):
        for arm in CANDIDATE_LAMBDA_2:
            changed = _diff(self.source, self.configs[arm])
            self.assertEqual(_unexpected(changed), set(),
                             "{} changed a key outside the registered axis".format(arm))
            # The axis itself must actually have moved relative to TR-0007 --
            # for the 0.01 arm it is the roots and the identity only.
            if arm == "mssl-l2-0p1":
                self.assertIn(("model", "mssl_lambda_2"), changed)
            self.assertIn(("mssl", "lambda_2_selection"), changed)

    def test_the_two_candidates_differ_from_each_other_only_by_the_axis(self):
        changed = _diff(self.configs["mssl-l2-0p01"], self.configs["mssl-l2-0p1"])
        self.assertEqual(_unexpected(changed), set())
        self.assertIn(("model", "mssl_lambda_2"), changed)

    def test_frozen_mssl_internals_are_untouched(self):
        for arm in CANDIDATE_LAMBDA_2:
            config = self.configs[arm]
            self.assertEqual(config["model"]["mssl_lambda_0"], 1.0)
            self.assertEqual(config["model"]["mssl_lambda_1"], 0.0)
            self.assertIsNone(config["model"]["mssl_admm_rho"])
            self.assertEqual(config["model"]["mssl_admm_iterations"], 2000)
            self.assertTrue(config["model"]["normalize_w"])
            self.assertEqual(config["mssl"]["warmup_epochs"], 3)
            self.assertEqual(config["mssl"]["omega_update_frequency"], 1)


class FrozenProtocolTests(unittest.TestCase):
    def setUp(self):
        self.plan = jobspec.load_plan(PLAN_PATH)
        self.configs = {
            arm["arm"]: _load(os.path.join(REPO_ROOT, arm["config"]))
            for arm in self.plan["arms"]
        }

    def test_every_arm_resolves_to_all_25_wavlm_layers(self):
        for arm, config in self.configs.items():
            self.assertEqual(config["upstream"]["model_type"], "wavlm_large", arm)
            self.assertEqual(config["upstream"]["selected_transformer_layers"], "all", arm)
            layers = parse_transformer_layers(
                config["upstream"]["selected_transformer_layers"],
                config["upstream"]["model_type"],
            )
            self.assertEqual(list(layers), list(range(WAVLM_LARGE_LAYERS)), arm)

    def test_every_arm_holds_the_protocol_conditions_identical(self):
        for arm, config in self.configs.items():
            self.assertEqual(config["pooling"]["frame_pooling_type"], "mean", arm)
            self.assertEqual(config["pooling"]["layer_pooling_type"], "smp", arm)
            self.assertEqual(config["pooling"]["layer_pooling_param"], 0.5, arm)
            self.assertEqual(config["dataset"]["subset_percentage"], 100, arm)
            self.assertEqual(config["dataset"]["ignore_index"], -1, arm)
            for key, value in (("embedding_dim_shared1", 512),
                               ("embedding_dim_shared2", 2000),
                               ("dropout_prob_shared1", 0.4),
                               ("dropout_prob_shared2", 0.6)):
                self.assertEqual(config["model"][key], value, arm)
            training = config["training"]
            self.assertEqual(training["num_epochs"], 30, arm)
            self.assertEqual(training["batch_size"], 2048, arm)
            self.assertEqual(training["learning_rate"], 0.0025, arm)
            self.assertEqual(training["weight_decay"], 5e-8, arm)
            self.assertEqual(training["l1_lambda"], 1e-7, arm)
            self.assertEqual(training["l2_lambda"], 1e-5, arm)
            self.assertEqual(training["factor"], 0.5, arm)
            self.assertTrue(training["drop_last_train"], arm)
            self.assertEqual(training["num_workers"], 4, arm)
            self.assertEqual(config["seed"], 42, arm)
            self.assertEqual(config["device"], {"type": "cuda", "index": 0}, arm)

    def test_controls_are_tr0007s_controls_untouched(self):
        per_arm = {
            arm["arm"]: os.path.join(REPO_ROOT, arm["config"])
            for arm in self.plan["arms"]
        }
        for arm, original in (("classical-mtrl", "classical-mtrl.yml"),
                              ("wavcse-baseline", "wavcse-baseline.yml")):
            copy = _load(per_arm[arm])
            source = _load(os.path.join(TR0007_STUDY_DIR, "configs", original))
            changed = _diff(copy, source)
            unexpected = {
                path for path in changed
                if path not in ALLOWED_DIFFS and path[0] != "research"
            }
            self.assertEqual(unexpected, set(), arm)
            self.assertEqual(copy["model"], source["model"], arm)
            self.assertEqual(copy["training"], source["training"], arm)

    def test_every_arm_config_carries_the_study_identity(self):
        for arm in self.plan["arms"]:
            path = os.path.join(REPO_ROOT, arm["config"])
            config = _load(path)
            research = config.get("research")
            self.assertIsNotNone(research, "{} has no research block".format(path))
            self.assertEqual(research["study_id"], "TR-0013")
            self.assertEqual(research["stage"], "screen")
            self.assertEqual(research["method"], arm["method"])
            self.assertEqual(research["protocol"], "variant-benchmark-protocol")
            self.assertTrue(os.path.exists(os.path.join(REPO_ROOT, research["run_note_file"])))
            self.assertEqual(
                mlflow_utils.build_research_run_name(config, arm["arm"], "ks_si_er"),
                "TR-0013__screen__{}__ks_si_er__smp25__s42".format(arm["method"]),
            )
            self.assertTrue(config["mlflow"]["experiment_name"])


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.plan = jobspec.load_plan(PLAN_PATH)
        self.inputs = artifacts.load(INPUTS_PATH)

    def test_plan_registers_four_arms_at_seed_42_and_no_confirmation_stage(self):
        self.assertEqual(self.plan["study"], "TR-0013")
        self.assertEqual(self.plan["task_type"], "ks_si_er")
        self.assertEqual(self.plan["stages"]["screen"]["seeds"], [42])
        self.assertEqual(sorted(self.plan["stages"]), ["screen"])
        self.assertEqual(
            sorted(arm["arm"] for arm in self.plan["arms"]),
            ["classical-mtrl", "mssl-l2-0p01", "mssl-l2-0p1", "wavcse-baseline"],
        )

    def test_each_candidates_labels_match_its_own_config(self):
        for arm in self.plan["arms"]:
            config = _load(os.path.join(REPO_ROOT, arm["config"]))
            labels = arm["labels"]
            self.assertEqual(labels["layer_policy"], "all-25")
            if arm["arm"] in CANDIDATE_LAMBDA_2:
                self.assertEqual(float(labels["lambda_2"]),
                                 config["model"]["mssl_lambda_2"])
                self.assertEqual(labels["lambda_1"], "0.0")
                self.assertEqual(labels["lambda_0"], "1.0")
                self.assertEqual(labels["normalize_w"], "true")
            else:
                self.assertEqual(labels["lambda_2"], "not-applicable")

    def test_plan_declares_the_canonical_embeddings_and_layout(self):
        requirements = self.inputs["requirements"]
        self.assertEqual(len(requirements), 15)
        declared = {requirement["artifact"] for requirement in requirements}
        for requirement in requirements:
            self.assertRegex(requirement["sha256"], r"^[0-9a-f]{64}$")
            self.assertTrue(requirement["required"])
        layout = self.plan["embedding_layout"]
        self.assertEqual(layout["root"], "embedding")
        layout_keys = []
        for entry in layout["datasets"]:
            layout_keys.extend(entry.get("inputs") or [entry["input"]])
        self.assertEqual(len(layout_keys), 15)
        self.assertEqual(set(layout_keys), declared)
        self.assertEqual(sorted(entry["dataset"] for entry in layout["datasets"]),
                         ["iemocap", "speechcommand", "voxceleb"])

    def test_plan_declares_the_output_contract_and_one_compatible_worker(self):
        kinds = {(output["kind"], output.get("tag")) for output in self.plan["outputs"]}
        self.assertIn(("checkpoint", "best"), kinds)
        self.assertIn(("checkpoint", "opt"), kinds)
        self.assertIn(("results_file", None), kinds)
        worker = self.plan["worker"]
        self.assertEqual(worker["cloud"], "SECURE")
        self.assertEqual(worker["gpu_count"], 1)
        self.assertEqual(worker["data_centers"], ["EU-RO-1"])
        self.assertEqual(worker["network_volume"]["datacenter"], "EU-RO-1")
        # The pinned image's PyTorch tops out at sm_90, so a Blackwell part
        # cannot run this workload (DG-0007 recorded that failure).
        self.assertNotIn("Blackwell", worker["gpu_type"])
        self.assertTrue(worker["image"])

    def test_declared_inputs_are_the_tr0007_artifact_set_byte_for_byte(self):
        with open(os.path.join(TR0007_STUDY_DIR, "compute", "inputs.json"),
                  "r", encoding="utf-8") as handle:
            tr0007 = json.load(handle)
        self.assertEqual(
            sorted((item["artifact"], item["sha256"], item["size_bytes"])
                   for item in self.inputs["requirements"]),
            sorted((item["artifact"], item["sha256"], item["size_bytes"])
                   for item in tr0007["requirements"]),
        )


class AuthorizationTests(unittest.TestCase):
    def test_envelope_is_scoped_to_this_screen_only(self):
        envelope = _load(AUTHORIZATION_PATH)
        self.assertEqual(envelope["scope"], "TR-0013")
        self.assertEqual(envelope["budget"]["max_gpu_hourly_usd"], 0.80)
        self.assertEqual(envelope["budget"]["max_total_gpu_usd"], 3.00)
        self.assertEqual(envelope["budget"]["max_wall_clock_hours"], 4)
        self.assertEqual(envelope["concurrency"]["max_simultaneous_workers"], 1)
        self.assertTrue(envelope["concurrency"]["replacement_workers_allowed"])
        self.assertTrue(envelope["resources"]["existing_network_volume_allowed"])
        self.assertFalse(envelope["resources"]["new_persistent_resources"])
        self.assertEqual(envelope["resources"]["container_disk_gb_max"], 120)
        self.assertTrue(envelope["stop_policy"]["destroy_on_completion"])
        self.assertEqual(envelope["stop_policy"]["retain_for_reuse_hours"], 0)
        # The plan asks for less disk than the envelope allows.
        plan = jobspec.load_plan(PLAN_PATH)
        self.assertLessEqual(plan["worker"]["container_disk_gb"],
                             envelope["resources"]["container_disk_gb_max"])

    def test_plan_states_the_prediction_the_falsifiers_and_the_run_count(self):
        flat = " ".join(PLAN_TEXT.split())
        self.assertIn(
            "validation-selected over the paper's two smallest published grid values",
            flat,
        )
        self.assertIn("Falsification of H-scale", flat)
        self.assertIn("Falsification of H-λ", flat)
        self.assertIn("Four runs, not five", flat)
        self.assertIn("Not a λ sweep", flat)
        self.assertIn("not a scale fix", flat)


if __name__ == "__main__":
    unittest.main()
