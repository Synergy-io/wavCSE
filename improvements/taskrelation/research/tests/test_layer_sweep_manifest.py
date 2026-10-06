"""The sweep spec: layer grammar, identity, and what must be refused."""

import json
import os
import shutil
import sys
import tempfile
import unittest

from improvements.sweep import manifest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layer_sweep_fixtures import make_study  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))


def _downstream_parser():
    """Load the frozen pipeline's parser by path (no sys.path mutation)."""
    import importlib.util
    path = os.path.join(_REPO_ROOT, "downstream", "utils", "parse_transformer_layers.py")
    spec = importlib.util.spec_from_file_location("downstream_parse_layers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_transformer_layers


class LayerGrammarTest(unittest.TestCase):
    """The local parser must not drift from the frozen pipeline's parser."""

    def test_parity_with_downstream_parser(self):
        downstream_parse = _downstream_parser()

        cases = ["all", "0", "0-12", "1,2,3", "6,1,0,3,2,5,4,7,8,9,10,11,12,17,14,13"]
        for combo in cases:
            self.assertEqual(manifest.parse_layers(combo, "wavlm_large"),
                             downstream_parse(combo, "wavlm_large"), combo)

    def test_all_resolves_against_the_model_variant(self):
        self.assertEqual(len(manifest.parse_layers("all", "wavlm_large")), 25)
        self.assertEqual(len(manifest.parse_layers("all", "wavlm_base")), 13)

    def test_order_is_preserved(self):
        self.assertEqual(manifest.parse_layers("6,1,0,3", "wavlm_large"), [6, 1, 0, 3])

    def test_inverted_range_is_refused(self):
        with self.assertRaises(manifest.SpecError):
            manifest.parse_layers("12-3", "wavlm_large")

    def test_combo_slug_is_readable(self):
        self.assertEqual(manifest.combo_slug("all"), "all")
        self.assertEqual(manifest.combo_slug("0-15"), "0-15")
        self.assertEqual(manifest.combo_slug("6,1,0,3"), "6_1_0_3")


class JobIdentityTest(unittest.TestCase):
    def test_key_is_deterministic_and_input_sensitive(self):
        first = manifest.job_key("TR-1", "0-15", 42, "a" * 40)
        self.assertEqual(first, manifest.job_key("TR-1", "0-15", 42, "A" * 40))
        self.assertNotEqual(first, manifest.job_key("TR-1", "0-16", 42, "a" * 40))
        self.assertNotEqual(first, manifest.job_key("TR-1", "0-15", 43, "a" * 40))
        self.assertEqual(len(first), 12)


class LoadSpecTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.study = make_study(self.tmp)

    def _spec(self, overrides):
        return make_study(self.tmp, overrides)["spec_path"]

    def test_loads_and_normalizes_policy(self):
        spec = manifest.load_spec(self.study["spec_path"])
        self.assertEqual(spec["policy"]["max_concurrent"], "auto")
        self.assertEqual(spec["policy"]["vram_per_run_gb"], 2.5)
        self.assertEqual(spec["method"], "mtrl_norm_corrected")
        self.assertTrue(spec["experiment_name"].endswith("layersweep"))

    def test_empty_combo_list_is_refused(self):
        with self.assertRaises(manifest.SpecError) as caught:
            manifest.load_spec(self._spec({"combos": []}))
        self.assertIn("non-empty", str(caught.exception))

    def test_duplicate_combos_are_refused(self):
        with self.assertRaises(manifest.SpecError) as caught:
            manifest.load_spec(self._spec({"combos": ["all", "all"]}))
        self.assertIn("duplicate", str(caught.exception))

    def test_unparsable_combo_is_refused(self):
        with self.assertRaises(manifest.SpecError):
            manifest.load_spec(self._spec({"combos": ["one-two"]}))

    def test_non_integer_seed_is_refused(self):
        with self.assertRaises(manifest.SpecError):
            manifest.load_spec(self._spec({"stages": {"screen": {"seeds": ["42"]}}}))

    def test_stage_naming_an_unknown_combo_is_refused(self):
        with self.assertRaises(manifest.SpecError):
            manifest.load_spec(self._spec({
                "stages": {"screen": {"seeds": [42], "combos": ["0-99"]}}}))

    def test_unknown_policy_key_is_refused(self):
        with self.assertRaises(manifest.SpecError):
            manifest.load_spec(self._spec({"policy": {"max_concurrent": 2,
                                                      "turbo": True}}))

    def test_non_positive_policy_value_is_refused(self):
        with self.assertRaises(manifest.SpecError):
            manifest.load_spec(self._spec({"policy": {"cpu_cores_per_run": 0}}))

    def test_missing_template_is_refused(self):
        with self.assertRaises(manifest.SpecError) as caught:
            manifest.load_spec(self._spec({"config_template": "nope.yml"}))
        self.assertIn("does not exist", str(caught.exception))

    def test_repo_relative_template_resolves(self):
        path = self._spec({"config_template": "template.yml"})
        spec = manifest.load_spec(path)
        self.assertTrue(os.path.exists(spec["config_template"]))

    def test_missing_required_key_is_refused(self):
        path = self.study["spec_path"]
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
        document.pop("combos")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        with self.assertRaises(manifest.SpecError) as caught:
            manifest.load_spec(path)
        self.assertIn("combos", str(caught.exception))


class StageExpansionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.spec = manifest.load_spec(make_study(self.tmp, {
            "combos": ["all", "0-15", "6,1,0,3"],
            "stages": {
                "screen": {"seeds": [42]},
                "confirm": {"seeds": [0, 1], "combos": ["all"]},
            },
        })["spec_path"])

    def test_stage_without_subset_runs_every_combo(self):
        self.assertEqual([manifest.combo_id(combo)
                          for combo in manifest.stage_combos(self.spec, "screen")],
                         ["all", "0-15", "6,1,0,3"])

    def test_stage_subset_narrows_the_selection(self):
        self.assertEqual([manifest.combo_id(combo)
                          for combo in manifest.stage_combos(self.spec, "confirm")],
                         ["all"])

    def test_jobs_expand_combo_major(self):
        self.assertEqual([(manifest.combo_id(combo), seed)
                          for combo, seed in manifest.stage_jobs(self.spec, "confirm")],
                         [("all", 0), ("all", 1)])

    def test_unknown_stage_is_refused(self):
        with self.assertRaises(manifest.SpecError):
            manifest.stage_jobs(self.spec, "nope")


class ComboRecordTest(unittest.TestCase):
    """A combo may be a bare string or a record carrying its grouping."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _spec(self, combos, stages=None):
        return manifest.load_spec(make_study(self.tmp, {
            "combos": combos,
            "stages": stages or {"screen": {"seeds": [42]}},
        })["spec_path"])

    def test_bare_string_normalizes_to_a_record(self):
        spec = self._spec(["0-15"])
        self.assertEqual(spec["combos"], [{"id": None, "layers": "0-15",
                                           "group": None, "k": None}])

    def test_record_keeps_group_and_k(self):
        spec = self._spec([{"id": "top-k2", "layers": "0-22",
                            "group": "top layer dropping", "k": 2}])
        combo = spec["combos"][0]
        self.assertEqual(manifest.combo_id(combo), "top-k2")
        self.assertEqual(combo["group"], "top layer dropping")
        self.assertEqual(combo["k"], 2)
        self.assertEqual(combo["layers"], "0-22")

    def test_duplicate_ids_are_refused(self):
        with self.assertRaises(manifest.SpecError) as caught:
            self._spec([{"id": "dup", "layers": "0-15"},
                        {"id": "dup", "layers": "0-14"}])
        self.assertIn("duplicate", str(caught.exception))

    def test_record_without_layers_is_refused(self):
        with self.assertRaises(manifest.SpecError):
            self._spec([{"id": "x", "group": "g", "k": 2}])

    def test_non_string_group_is_refused(self):
        with self.assertRaises(manifest.SpecError):
            self._spec([{"layers": "0-15", "group": 7}])

    def test_non_integer_k_is_refused(self):
        with self.assertRaises(manifest.SpecError):
            self._spec([{"layers": "0-15", "k": "two"}])

    def test_stage_selects_by_id(self):
        spec = self._spec(
            [{"id": "top-k2", "layers": "0-22"},
             {"id": "bottom-k2", "layers": "2-24"}],
            stages={"screen": {"seeds": [42]},
                    "confirm": {"seeds": [0], "combos": ["bottom-k2"]}})
        self.assertEqual([manifest.combo_id(combo)
                          for combo in manifest.stage_combos(spec, "confirm")],
                         ["bottom-k2"])

    def test_stage_selects_by_layer_string(self):
        spec = self._spec([{"id": "top-k2", "layers": "0-22"}],
                          stages={"screen": {"seeds": [42]},
                                  "confirm": {"seeds": [0], "combos": ["0-22"]}})
        self.assertEqual(len(manifest.stage_combos(spec, "confirm")), 1)

    def test_stage_naming_an_absent_id_is_refused(self):
        with self.assertRaises(manifest.SpecError) as caught:
            self._spec([{"id": "top-k2", "layers": "0-22"}],
                       stages={"screen": {"seeds": [42], "combos": ["nope"]}})
        self.assertIn("nope", str(caught.exception))

    def test_slug_prefers_the_id(self):
        self.assertEqual(manifest.combo_slug({"id": "top-k2", "layers": "0-22"}),
                         "top-k2")
        self.assertEqual(manifest.combo_slug("0-15"), "0-15")


if __name__ == "__main__":
    unittest.main()
