"""Generated configs: the diff must be exactly the keys a sweep may move."""

import os
import shutil
import sys
import tempfile
import unittest

import yaml

from improvements.sweep import config_gen, manifest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layer_sweep_fixtures import make_study  # noqa: E402

EXPECTED_DIFF = set(config_gen.MUTABLE_KEYS) | set(config_gen.RESEARCH_MUTABLE_KEYS)


class RenderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.spec = manifest.load_spec(make_study(self.tmp, {
            "combos": [
                {"id": "all", "layers": "all"},
                {"id": "top-k2", "layers": "0-22", "group": "top layer dropping", "k": 2},
            ],
            "policy": {"workers_per_run": 2},
        })["spec_path"])
        self.by_id = {manifest.combo_id(combo): combo for combo in self.spec["combos"]}

    def test_two_combos_differ_only_in_the_allowed_keys(self):
        left = config_gen.render(self.spec, "screen", self.by_id["all"])
        right = config_gen.render(self.spec, "screen", self.by_id["top-k2"])
        differing = set(config_gen.differing_keys(left, right))
        self.assertTrue(differing <= EXPECTED_DIFF,
                        "unexpected diff: {}".format(sorted(differing - EXPECTED_DIFF)))
        for key in ("upstream.selected_transformer_layers", "research.combo",
                    "research.group", "research.k", "research.layers",
                    "research.layer_count", "research.representation",
                    "research.run_note", "paths.results_root",
                    "paths.checkpoints_root"):
            self.assertIn(key, differing, key)

    def test_layer_string_is_written_verbatim(self):
        document = config_gen.render(self.spec, "screen", self.by_id["top-k2"])
        self.assertEqual(document["upstream"]["selected_transformer_layers"], "0-22")

    def test_output_roots_are_repo_relative_under_the_study(self):
        document = config_gen.render(self.spec, "screen", self.by_id["top-k2"])
        self.assertEqual(
            document["paths"]["results_root"],
            "improvements/taskrelation/research/studies/TR-TEST/outputs/top-k2/results")
        self.assertEqual(
            document["paths"]["checkpoints_root"],
            "improvements/taskrelation/research/studies/TR-TEST/outputs/top-k2/checkpoints")

    def test_grouping_is_written_into_the_config(self):
        research = config_gen.render(self.spec, "screen", self.by_id["top-k2"])["research"]
        self.assertEqual(research["group"], "top layer dropping")
        self.assertEqual(research["k"], 2)
        self.assertEqual(research["combo"], "top-k2")
        self.assertEqual(research["layers"], "0-22")
        self.assertEqual(research["layer_count"], 23)

    def test_representation_names_pooling_and_layer_count(self):
        self.assertEqual(
            config_gen.render(self.spec, "screen", self.by_id["top-k2"])["research"]
            ["representation"], "smp23L")
        self.assertEqual(
            config_gen.render(self.spec, "screen", self.by_id["all"])["research"]
            ["representation"], "smp25L")

    def test_run_name_template_is_written_and_per_seed_unique(self):
        research = config_gen.render(self.spec, "screen", self.by_id["top-k2"])["research"]
        self.assertIn("{combo}", research["run_name"])
        self.assertIn("{seed:02d}", research["run_name"])

    def test_note_details_layers_group_and_k(self):
        note = config_gen.render(self.spec, "screen", self.by_id["top-k2"])["research"]
        self.assertIn("Group: top layer dropping", note["run_note"])
        self.assertIn("k (layers dropped): 2", note["run_note"])
        self.assertIn("Layers kept (23 of 25): 0,1,2,", note["run_note"])
        self.assertIn("Layers dropped: 23,24", note["run_note"])

    def test_inherited_note_file_is_cleared(self):
        """The template's run_note_file would otherwise win over the inline note."""
        research = config_gen.render(self.spec, "screen", self.by_id["top-k2"])["research"]
        self.assertIsNone(research["run_note_file"])

    def test_worker_count_comes_from_the_policy(self):
        document = config_gen.render(self.spec, "screen", self.by_id["all"])
        self.assertEqual(document["training"]["num_workers"], 2)
        self.assertEqual(document["evaluation"]["num_workers"], 2)

    def test_identity_block_is_filled_from_the_spec(self):
        research = config_gen.render(self.spec, "screen", self.by_id["all"])["research"]
        self.assertEqual(research["study_id"], "TR-TEST")
        self.assertEqual(research["stage"], "screen")
        self.assertEqual(research["method"], "mtrl_norm_corrected")
        self.assertEqual(research["sweep_id"], "layersweep-test")

    def test_experiment_name_is_the_sweep_experiment(self):
        self.assertEqual(config_gen.render(self.spec, "screen", self.by_id["all"])["mlflow"]
                         ["experiment_name"], "taskrelation-mtrl-layersweep")

    def test_untouched_template_values_survive(self):
        document = config_gen.render(self.spec, "screen", self.by_id["all"])
        self.assertEqual(document["pooling"]["layer_pooling_type"], "smp")
        self.assertEqual(document["training"]["num_epochs"], 30)
        self.assertEqual(document["model"]["normalize_w"], False)


class WriteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.spec = manifest.load_spec(make_study(self.tmp, {
            "combos": [{"id": "all", "layers": "all"},
                       {"id": "top-k2", "layers": "0-22"}]})["spec_path"])
        self.by_id = {manifest.combo_id(combo): combo for combo in self.spec["combos"]}

    def test_write_then_refuse_to_clobber(self):
        path = config_gen.write(self.spec, "screen", self.by_id["all"])
        self.assertTrue(os.path.exists(path))
        with self.assertRaises(config_gen.ConfigError):
            config_gen.write(self.spec, "screen", self.by_id["all"])
        config_gen.write(self.spec, "screen", self.by_id["all"], force=True)

    def test_written_config_is_valid_yaml_with_a_provenance_header(self):
        path = config_gen.write(self.spec, "screen", self.by_id["all"])
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("Generated by `python -m improvements.sweep configs`", text)
        document = yaml.safe_load(text)
        self.assertEqual(document["research"]["combo"], "all")

    def test_relative_config_path_uses_the_combo_id(self):
        config_gen.write(self.spec, "screen", self.by_id["top-k2"])
        self.assertEqual(
            config_gen.relative_config_path(self.spec, "screen", self.by_id["top-k2"]),
            "improvements/taskrelation/research/studies/TR-TEST/configs/screen/top-k2.yml")

    def test_write_stage_writes_every_combo(self):
        paths = config_gen.write_stage(self.spec, "screen")
        self.assertEqual(len(paths), 2)
        self.assertEqual({os.path.basename(path) for path in paths},
                         {"all.yml", "top-k2.yml"})


if __name__ == "__main__":
    unittest.main()
