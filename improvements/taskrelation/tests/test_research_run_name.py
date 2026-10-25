"""The Study-aware run name, including the config-supplied template.

`build_research_run_name` is shared by every improvements/ owner folder, so the
first case here is the regression that matters most: a config that does not ask
for a template must produce exactly the name it always did.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from improvements import mlflow_utils  # noqa: E402


def config(**research):
    block = {"study_id": "TR-0001", "stage": "screen",
             "method": "mtrl_norm_corrected", "representation": "smp25L"}
    block.update(research)
    return {"seed": 42, "research": block}


class DefaultNameTest(unittest.TestCase):
    def test_default_shape_is_unchanged(self):
        self.assertEqual(
            mlflow_utils.build_research_run_name(config(), "mtrl", "ks_si_er"),
            "TR-0001__screen__mtrl_norm_corrected__ks_si_er__smp25L__s42")

    def test_seed_comes_from_the_config(self):
        cfg = config()
        cfg["seed"] = 7
        name = mlflow_utils.build_research_run_name(cfg, "mtrl", "ks_si_er")
        self.assertIsNotNone(name)
        self.assertTrue(name.endswith("__s07"))

    def test_no_study_id_means_no_research_name(self):
        self.assertIsNone(mlflow_utils.build_research_run_name(
            {"seed": 1, "research": {}}, "mtrl", "ks_si_er"))


class TemplateNameTest(unittest.TestCase):
    def test_template_wins_and_formats(self):
        name = mlflow_utils.build_research_run_name(
            config(combo="top-k2",
                   run_name="{study_id}__{stage}__{combo}__{task_type}__"
                            "{representation}__s{seed:02d}"),
            "mtrl", "ks_si_er")
        self.assertEqual(name,
                         "TR-0001__screen__top-k2__ks_si_er__smp25L__s42")

    def test_layer_count_placeholder_is_available(self):
        name = mlflow_utils.build_research_run_name(
            config(combo="sel", layer_count=23,
                   run_name="{combo}-L{layer_count}-s{seed}"),
            "mtrl", "ks_si_er")
        self.assertEqual(name, "sel-L23-s42")

    def test_spaces_and_slashes_are_cleaned_from_placeholders(self):
        name = mlflow_utils.build_research_run_name(
            config(combo="a b/c", run_name="{combo}"),
            "mtrl", "ks_si_er")
        self.assertEqual(name, "a-b-c")

    def test_unknown_placeholder_fails_loudly(self):
        with self.assertRaises(ValueError) as caught:
            mlflow_utils.build_research_run_name(
                config(combo="x", run_name="{nope}"), "mtrl", "ks_si_er")
        message = str(caught.exception)
        self.assertIn("nope", message)
        self.assertIn("combo", message)

    def test_placeholder_values_are_reusable_across_seeds(self):
        template = config(combo="sym-k2", run_name="{combo}__s{seed:02d}")
        names = []
        for seed in (0, 4):
            template["seed"] = seed
            names.append(mlflow_utils.build_research_run_name(
                template, "mtrl", "ks_si_er"))
        self.assertEqual(names, ["sym-k2__s00", "sym-k2__s04"])


class NoteResolutionTest(unittest.TestCase):
    def test_inline_note_wins_over_a_cleared_file(self):
        """resolve_run_note returns the inline note verbatim when no file is set."""
        cfg = config(run_note_file=None, run_note="Group: top\nk: 2")
        self.assertEqual(mlflow_utils.resolve_run_note(cfg), "Group: top\nk: 2")

    def test_missing_note_is_none(self):
        self.assertIsNone(mlflow_utils.resolve_run_note(config()))


if __name__ == "__main__":
    unittest.main()
