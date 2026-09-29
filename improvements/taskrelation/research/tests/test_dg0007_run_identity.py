"""DG-0007 must be identifiable as DG-0007 at runtime, in every arm.

Audit review `5f72acd` §H1: the successor was pre-registered without
study-identity configs, so run as committed none of the three arms would carry
the study's identity -- `build_research_run_name` falls back to a legacy
timestamped name, `resolve_run_note` returns None, and `set_standard_tags`
publishes `study_id`, `stage`, `parent_study` and `representation` as None,
contradicting `/AGENTS.md` invariant 12 and
`VARIANT_BENCHMARK_PROTOCOL.md` §7.

These tests run the REAL helpers over the REAL committed configs -- they parse
YAML only to hand it to the same functions the runner calls -- and they fail if
any arm loses its identity. `ProvenanceInvariant` is deliberately asserted both
ways: it passes for the six DG-0007 execution configs and raises for the
pre-existing configs they are derived from, so the invariant cannot be vacuous.
"""

import copy
import os
import sys
import unittest
from unittest import mock

import yaml

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
STUDY_DIR = os.path.join(
    REPO_ROOT, "improvements", "taskrelation", "research", "studies", "DG-0007"
)
CONFIG_DIR = os.path.join(STUDY_DIR, "configs")
for path in (REPO_ROOT, os.path.join(REPO_ROOT, "downstream")):
    if path not in sys.path:
        sys.path.insert(0, path)

from improvements import mlflow_utils  # noqa: E402
from improvements.run_identity import research_identity  # noqa: E402
from improvements.run_improvements import MODEL_CATEGORY  # noqa: E402
from utils.parse_transformer_layers import parse_transformer_layers  # noqa: E402

STUDY_ID = "DG-0007"

# arm -> (runner --model, research.method, category the runner tags with)
ARMS = {
    "norm_corrected_mtrl": ("mtrl", "mtrl_norm_corrected", "taskrelation"),
    "historical_mtrl": ("mtrl", "mtrl", "taskrelation"),
    "baseline": ("original", "wavcse-baseline", "base"),
}

# The configs each arm's science is derived from. They predate the study and
# carry no research block; the study configs add identity only.
SOURCES = {
    "norm_corrected_mtrl":
        "improvements/taskrelation/01-mtrl/mtrl_norm_corrected_25L_config.yml",
    "historical_mtrl":
        "improvements/taskrelation/01-mtrl/mtrl_poolingwinner_25L_config.yml",
    "baseline":
        "improvements/base/configs/base_poolingwinner_25L_config.yml",
}

REQUIRED_TAGS = (
    "study_id", "stage", "method", "representation", "task_set", "family",
    "hypothesis_slug", "parent_study", "baseline_study", "agent_generated",
    "status", "layers", "pooling", "git_commit",
)


def config_path(arm, stage):
    name = f"{arm}.yml" if stage == "screen" else f"confirm_{arm}.yml"
    return os.path.join(CONFIG_DIR, name)


def load(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def flatten(mapping, prefix=""):
    flat = {}
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten(value, path))
        else:
            flat[path] = value
    return flat


def assert_run_provenance(test, cfg, model, task_type="ks_si_er", seed=42):
    """The invariant a launchable study run must satisfy. Returns the tags.

    Raises AssertionError when any part of the study/arm identity is missing, so
    a config that would silently fall back to a legacy run name fails here
    rather than producing anonymous evidence.
    """
    cfg = copy.deepcopy(cfg)
    cfg["seed"] = seed

    run_name = mlflow_utils.build_research_run_name(cfg, model, task_type)
    test.assertIsNotNone(run_name, "the run name must be study-scoped")
    test.assertTrue(run_name.startswith(STUDY_ID + "__"))

    test.assertTrue(mlflow_utils.resolve_run_note(cfg), "the run note must resolve")

    captured = {}
    with mock.patch.object(
        mlflow_utils.mlflow, "set_tags", side_effect=lambda tags: captured.update(tags)
    ):
        mlflow_utils.set_standard_tags(
            MODEL_CATEGORY.get(model, "taskrelation"), model, cfg,
            extra_tags={"task_set": task_type},
        )
    for name in REQUIRED_TAGS:
        test.assertIn(name, captured, f"{name} must be published")
        test.assertIsNotNone(captured[name], f"{name} must not be null")
    test.assertEqual(captured["study_id"], STUDY_ID)
    test.assertEqual(captured["seed"], seed)
    test.assertNotEqual(captured["pooling"], "None")
    return captured, run_name


class ArmIdentity(unittest.TestCase):
    """Every arm, every stage, carries a complete DG-0007 identity."""

    def test_run_name_is_study_scoped_and_arm_specific(self):
        expected = {
            ("norm_corrected_mtrl", "screen"):
                "DG-0007__screen__mtrl_norm_corrected__ks_si_er__smp25__s42",
            ("norm_corrected_mtrl", "confirm"):
                "DG-0007__confirm__mtrl_norm_corrected__ks_si_er__smp25__s42",
            ("historical_mtrl", "screen"):
                "DG-0007__screen__mtrl__ks_si_er__smp25__s42",
            ("historical_mtrl", "confirm"):
                "DG-0007__confirm__mtrl__ks_si_er__smp25__s42",
            ("baseline", "screen"):
                "DG-0007__screen__wavcse-baseline__ks_si_er__smp25__s42",
            ("baseline", "confirm"):
                "DG-0007__confirm__wavcse-baseline__ks_si_er__smp25__s42",
        }
        for (arm, stage), name in expected.items():
            with self.subTest(arm=arm, stage=stage):
                model = ARMS[arm][0]
                cfg = load(config_path(arm, stage))
                cfg["seed"] = 42
                self.assertEqual(
                    mlflow_utils.build_research_run_name(cfg, model, "ks_si_er"), name
                )

    def test_every_arm_publishes_a_complete_research_identity(self):
        for arm, (model, method, category) in ARMS.items():
            for stage in ("screen", "confirm"):
                with self.subTest(arm=arm, stage=stage):
                    cfg = load(config_path(arm, stage))
                    tags, run_name = assert_run_provenance(self, cfg, model)
                    self.assertEqual(tags["method"], method)
                    self.assertEqual(tags["stage"], stage)
                    self.assertEqual(tags["representation"], "smp25")
                    self.assertEqual(tags["task_set"], "ks_si_er")
                    self.assertEqual(tags["family"], "task_relation_learning")
                    self.assertEqual(tags["hypothesis_slug"],
                                     "mtrl-normalization-faithfulness")
                    self.assertEqual(tags["parent_study"], "DG-0002")
                    self.assertEqual(tags["layers"], "all")
                    self.assertEqual(tags["pooling"], "smp:0.5")
                    self.assertEqual(
                        MODEL_CATEGORY.get(model, "taskrelation"), category
                    )
                    self.assertIn(method, run_name)

    def test_the_run_note_is_the_study_note(self):
        for arm in ARMS:
            for stage in ("screen", "confirm"):
                with self.subTest(arm=arm, stage=stage):
                    note = mlflow_utils.resolve_run_note(load(config_path(arm, stage)))
                    self.assertIn("DG-0007", note)
                    self.assertIn("normalization-corrected", note)

    def test_real_runner_tag_call_shape_is_exercised(self):
        """The helper is called exactly as `run_improvements.main` calls it."""
        cfg = load(config_path("norm_corrected_mtrl", "screen"))
        cfg["seed"] = 7
        captured = {}
        with mock.patch.object(
            mlflow_utils.mlflow, "set_tags",
            side_effect=lambda tags: captured.update(tags),
        ):
            mlflow_utils.set_standard_tags(
                MODEL_CATEGORY["mtrl"], "mtrl", cfg, extra_tags={"task_set": "ks_si_er"}
            )
        self.assertEqual(captured["study_id"], STUDY_ID)
        self.assertEqual(captured["seed"], 7)
        self.assertEqual(captured["method"], "mtrl_norm_corrected")
        # the run name follows the CLI seed too, because the runner writes the
        # resolved seed back into cfg before naming the run
        self.assertTrue(
            mlflow_utils.build_research_run_name(cfg, "mtrl", "ks_si_er").endswith("s07")
        )


class IdentityRecord(unittest.TestCase):
    """ARC_RUN_IDENTITY must separate the two MTRL arms of one study."""

    def test_record_carries_study_stage_arm_and_representation(self):
        for arm, (model, method, _category) in ARMS.items():
            with self.subTest(arm=arm):
                cfg = load(config_path(arm, "screen"))
                record = research_identity(
                    cfg["research"], "0" * 40, default_method=model
                )
                self.assertEqual(record["study_id"], STUDY_ID)
                self.assertEqual(record["stage"], "screen")
                self.assertEqual(record["method"], method)
                self.assertEqual(record["representation"], "smp25")
                self.assertEqual(record["git_commit"], "0" * 40)

    def test_the_two_mtrl_arms_are_told_apart_by_the_record(self):
        """`model` is "mtrl" for both arms; only `method` distinguishes them."""
        records = {}
        for arm in ("norm_corrected_mtrl", "historical_mtrl"):
            cfg = load(config_path(arm, "screen"))
            records[arm] = research_identity(cfg["research"], "a" * 40,
                                             default_method="mtrl")
        self.assertNotEqual(records["norm_corrected_mtrl"]["method"],
                            records["historical_mtrl"]["method"])
        self.assertEqual(records["norm_corrected_mtrl"]["method"],
                         "mtrl_norm_corrected")
        self.assertEqual(records["historical_mtrl"]["method"], "mtrl")
        # the runner passes model="mtrl" for both, which is exactly why the
        # arm cannot be read off `model` alone
        self.assertNotIn("model", records["norm_corrected_mtrl"])

    def test_a_config_without_a_research_block_yields_no_identity(self):
        plain = load(os.path.join(REPO_ROOT, SOURCES["historical_mtrl"]))
        record = research_identity(plain.get("research"), "a" * 40, default_method="mtrl")
        self.assertIsNone(record["study_id"])
        self.assertIsNone(record["stage"])
        self.assertIsNone(record["representation"])


class ProvenanceInvariant(unittest.TestCase):
    """The invariant is not vacuous: the study configs are what satisfy it."""

    def test_the_invariant_fails_for_the_pre_existing_configs(self):
        for arm, source in SOURCES.items():
            with self.subTest(source=source):
                cfg = load(os.path.join(REPO_ROOT, source))
                self.assertNotIn(
                    "research", cfg,
                    "the source config is deliberately identity-free",
                )
                with self.assertRaises(AssertionError):
                    assert_run_provenance(self, cfg, ARMS[arm][0])

    def test_the_invariant_passes_only_with_a_research_block(self):
        cfg = load(config_path("baseline", "screen"))
        self.assertIn("research", cfg)
        assert_run_provenance(self, cfg, "original")
        del cfg["research"]
        with self.assertRaises(AssertionError):
            assert_run_provenance(self, cfg, "original")


class StudyConfigIsolation(unittest.TestCase):
    """Each study config adds identity only; the arm delta stays one variable."""

    def test_study_configs_differ_from_their_source_only_in_identity_and_outputs(self):
        for arm, source in SOURCES.items():
            for stage in ("screen", "confirm"):
                with self.subTest(arm=arm, stage=stage):
                    study = flatten(load(config_path(arm, stage)))
                    original = flatten(load(os.path.join(REPO_ROOT, source)))
                    added = set(study) - set(original)
                    self.assertTrue(added)
                    self.assertTrue(
                        all(key.startswith("research.") for key in added), added
                    )
                    changed = {
                        key for key in original if original[key] != study[key]
                    }
                    self.assertEqual(
                        changed,
                        {"paths.results_root", "paths.checkpoints_root"},
                    )

    def test_the_two_mtrl_arms_differ_only_in_the_corrected_variable(self):
        corrected = flatten(load(config_path("norm_corrected_mtrl", "screen")))
        historical = flatten(load(config_path("historical_mtrl", "screen")))
        self.assertEqual(sorted(corrected), sorted(historical))
        self.assertEqual(
            {k for k in corrected if corrected[k] != historical[k]},
            {
                "model.normalize_w",
                "paths.results_root",
                "paths.checkpoints_root",
                "research.method",
            },
        )
        self.assertIs(corrected["model.normalize_w"], False)
        self.assertIs(historical["model.normalize_w"], True)

    def test_screen_and_confirm_differ_only_in_stage_status_and_outputs(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                screen = flatten(load(config_path(arm, "screen")))
                confirm = flatten(load(config_path(arm, "confirm")))
                self.assertEqual(sorted(screen), sorted(confirm))
                self.assertEqual(
                    {k for k in screen if screen[k] != confirm[k]},
                    {
                        "paths.results_root",
                        "paths.checkpoints_root",
                        "research.stage",
                        "research.status",
                    },
                )


class AllTwentyFiveLayers(unittest.TestCase):
    def test_every_arm_trains_on_all_twenty_five_layers(self):
        for arm in ARMS:
            for stage in ("screen", "confirm"):
                with self.subTest(arm=arm, stage=stage):
                    cfg = load(config_path(arm, stage))
                    self.assertEqual(
                        parse_transformer_layers(
                            cfg["upstream"]["selected_transformer_layers"],
                            cfg["upstream"]["model_type"],
                        ),
                        list(range(25)),
                    )
                    self.assertEqual(cfg["pooling"]["layer_pooling_type"], "smp")
                    self.assertEqual(cfg["pooling"]["layer_pooling_param"], 0.5)

    def test_a_short_embedding_store_fails_loudly_rather_than_subsetting(self):
        """The loader indexes the store with the layer array, so 25 means 25.

        `downstream/dataset/preprocess_embedding.py` does
        `embedding = embedding[transformer_layer_array, :]`. On a store with the
        canonical 25 rows that is the identity over every layer; on a shorter
        store it raises rather than silently training on a subset. The store
        itself is upstream-frozen and absent here, so this pins the behaviour
        the loader relies on rather than the store's row count (see the audit's
        pre-run materialization note).
        """
        import torch

        layers = parse_transformer_layers("all", "wavlm_large")
        store = torch.arange(25 * 4, dtype=torch.float32).reshape(25, 4)
        self.assertTrue(torch.equal(store[layers, :], store))
        with self.assertRaises(IndexError):
            store[:24, :][layers, :]


if __name__ == "__main__":
    unittest.main()
