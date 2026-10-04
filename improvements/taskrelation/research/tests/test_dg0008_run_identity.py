"""DG-0008 must carry the Study's canonical provenance on every member run.

The registered DG-0008 implementation wrote ``run.json``/``run_identity.json``
but never opened an MLflow run, so ``AGENTS.md`` invariant 12 and
``VARIANT_BENCHMARK_PROTOCOL.md`` §7 were unsatisfied for every one of the 201
members.  These tests run the REAL committed config through the REAL helpers
``run_dg0008.py`` calls -- ``member_method``, ``open_member_run``,
``build_research_run_name``, ``set_standard_tags``, ``log_config_params``,
``log_trainer_history`` and ``resolve_run_note`` -- and assert that:

* every member run name is study-scoped and arm/cell/seed specific;
* ``set_standard_tags`` publishes the complete identity and a resolvable note;
* a run genuinely opens and the trainer history, the fixed-final ER endpoint,
  the manifest/identity digests and the per-step traces all land, against a
  local file-backed tracking URI (no network);
* tracking failure is fatal, never a silent provenance skip.
"""

import copy
import importlib.util
import os
import sys
import tempfile
import unittest
from unittest import mock

import yaml

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
STUDY_DIR = os.path.join(
    REPO_ROOT, "improvements", "taskrelation", "research", "studies", "DG-0008"
)
CONFIG_PATH = os.path.join(STUDY_DIR, "configs", "dg0008.yaml")
for path in (REPO_ROOT, os.path.join(REPO_ROOT, "downstream")):
    if path not in sys.path:
        sys.path.insert(0, path)

from improvements import mlflow_utils  # noqa: E402

STUDY_ID = "DG-0008"
STAGE = "stage1_screen"
ARMS = ("pair", "control")
CELLS = ("ks_er", "si_er")
SEEDS = (0, 1, 2, 3, 4)

REQUIRED_TAGS = (
    "study_id", "stage", "method", "representation", "task_set", "family",
    "hypothesis_slug", "baseline_study", "agent_generated", "status", "seed",
    "layers", "pooling", "git_commit",
)


def _load_runner():
    path = os.path.join(STUDY_DIR, "run_dg0008.py")
    spec = importlib.util.spec_from_file_location("dg0008_runner_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_dg0008 = _load_runner()


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def member_config(cfg, cell, arm, seed):
    """Reproduce run_dg0008.build_member's member-identity resolution exactly."""

    config = copy.deepcopy(cfg)
    config["seed"] = seed
    config["research"] = dict(cfg["research"])
    method = run_dg0008.member_method(config["research"], arm)
    config["research"]["method"] = method
    config["research"]["task_set"] = cell
    return config, method


def expected_run_name(method, cell, seed):
    return "{}__{}__{}__{}__smp25__s{:02d}".format(
        STUDY_ID, STAGE, method, cell, seed
    )


class RunNameTests(unittest.TestCase):
    """Every member, every arm, carries a Study-scoped, member-specific name."""

    def test_names_follow_the_study_pattern(self):
        cfg = load_config()
        names = set()
        for arm in ARMS:
            for cell in CELLS:
                for seed in SEEDS:
                    config, method = member_config(cfg, cell, arm, seed)
                    name = mlflow_utils.build_research_run_name(config, method, cell)
                    with self.subTest(arm=arm, cell=cell, seed=seed):
                        self.assertEqual(name, expected_run_name(method, cell, seed))
                        self.assertTrue(name.startswith(STUDY_ID + "__"))
                    names.add(name)
        # 2 arms x 2 cells x 5 seeds: the name is not vacuous.
        self.assertEqual(len(names), len(ARMS) * len(CELLS) * len(SEEDS))

    def test_arm_methods_come_from_the_committed_config(self):
        cfg = load_config()
        methods = cfg["research"]["methods"]
        self.assertEqual(set(methods), set(ARMS))
        self.assertEqual(run_dg0008.member_method(cfg["research"], "pair"),
                         methods["pair"])
        self.assertEqual(run_dg0008.member_method(cfg["research"], "control"),
                         methods["control"])
        self.assertNotEqual(methods["pair"], methods["control"])


class TagIdentityTests(unittest.TestCase):
    """The invariant `test_dg0007_run_identity` asserts, for DG-0008."""

    def test_complete_tag_set_run_name_and_resolvable_note(self):
        cfg = load_config()
        for arm in ARMS:
            for cell in CELLS:
                config, method = member_config(cfg, cell, arm, 3)
                with self.subTest(arm=arm, cell=cell):
                    self.assertTrue(
                        mlflow_utils.resolve_run_note(config),
                        "the run note must resolve",
                    )
                    self.assertEqual(
                        mlflow_utils.build_research_run_name(config, method, cell),
                        expected_run_name(method, cell, 3),
                    )

                    captured = {}
                    with mock.patch.object(
                        mlflow_utils.mlflow, "set_tags",
                        side_effect=lambda tags: captured.update(tags),
                    ):
                        mlflow_utils.set_standard_tags(
                            run_dg0008.MEMBER_CATEGORY, run_dg0008.MEMBER_MODEL,
                            config, extra_tags={"task_set": cell},
                        )

                    for name in REQUIRED_TAGS:
                        self.assertIn(name, captured, "{} must be published".format(name))
                        self.assertIsNotNone(captured[name], "{} must not be null".format(name))
                    self.assertEqual(captured["study_id"], STUDY_ID)
                    self.assertEqual(captured["stage"], STAGE)
                    self.assertEqual(captured["method"], method)
                    self.assertEqual(captured["task_set"], cell)
                    self.assertEqual(captured["representation"], "smp25")
                    self.assertEqual(captured["family"], "task_relation_learning")
                    self.assertEqual(captured["hypothesis_slug"],
                                     "exact-opportunity-er-target-residual")
                    self.assertEqual(captured["baseline_study"], "DG-0001")
                    self.assertEqual(captured["seed"], 3)
                    self.assertEqual(captured["layers"], "all")
                    self.assertEqual(captured["pooling"], "smp:0.5")
                    self.assertIn(STUDY_ID, captured["mlflow.note.content"])

    def test_the_run_note_is_the_study_note(self):
        cfg = load_config()
        config, _ = member_config(cfg, "ks_er", "pair", 0)
        note = mlflow_utils.resolve_run_note(config)
        self.assertIn(STUDY_ID, note)
        self.assertIn("exact-opportunity", note)


class FailClosedTests(unittest.TestCase):
    """Tracking failure aborts the member; provenance is never skipped."""

    def test_tracking_start_failure_is_fatal(self):
        cfg = load_config()
        config, method = member_config(cfg, "ks_er", "pair", 0)
        with mock.patch.object(
            mlflow_utils, "setup_mlflow",
            side_effect=RuntimeError("tracking uri unreachable"),
        ):
            with self.assertRaises(RuntimeError):
                run_dg0008.open_member_run(config, method, "ks_er")

    def test_missing_study_identity_is_refused(self):
        cfg = load_config()
        config, method = member_config(cfg, "ks_er", "pair", 0)
        config.pop("research")
        with mock.patch.object(mlflow_utils, "setup_mlflow"):
            with self.assertRaises(run_dg0008.ManifestError):
                run_dg0008.open_member_run(config, method, "ks_er")

    def test_unconfigured_arm_method_is_refused(self):
        with self.assertRaises(run_dg0008.ManifestError):
            run_dg0008.member_method({}, "pair")


class _FakeTrainer:
    """The finished-trainer surface ``log_trainer_history`` replays."""

    num_epochs = 5
    task_array = ["ks", "er"]
    final_validation_er_accuracy = 0.75

    def __init__(self):
        self.train_losses_all = [1.0, 0.9, 0.8, 0.7, 0.6]
        self.train_acc_all = [0.10, 0.20, 0.30, 0.40, 0.50]
        self.val_losses_all = [1.1, 1.0, 0.9, 0.8, 0.7]
        self.val_acc_all = [0.11, 0.22, 0.33, 0.44, 0.55]
        self.learning_rate_array = [0.0025] * 5
        self.train_losses_task = [[0.5] * 5, [0.6] * 5]
        self.train_acc_task = [[0.1] * 5, [0.2] * 5]
        self.val_losses_task = [[0.7] * 5, [0.8] * 5]
        self.val_acc_task = [[0.3] * 5, [0.75] * 5]


class OfflineEndToEndLoggingTests(unittest.TestCase):
    """A real run against a local file-backed tracking URI, no network."""

    def test_member_run_publishes_identity_history_endpoint_and_artifacts(self):
        from mlflow.tracking import MlflowClient

        cfg = load_config()
        uri = "file://" + tempfile.mkdtemp(prefix="dg0008-mlflow-")
        cfg["mlflow"] = {
            "tracking_uri": uri,
            "experiment_name": "wavcse-baseline-dg0008-offline-test",
        }
        config, method = member_config(cfg, "si_er", "pair", 2)

        artifact_dir = tempfile.mkdtemp(prefix="dg0008-member-")
        provenance_path = os.path.join(artifact_dir, "provenance.json")
        traces_path = os.path.join(artifact_dir, "step_traces.json")
        for path in (provenance_path, traces_path):
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("{}")

        trainer = _FakeTrainer()
        with run_dg0008.open_member_run(config, method, "si_er"):
            mlflow_utils.set_standard_tags(
                run_dg0008.MEMBER_CATEGORY, run_dg0008.MEMBER_MODEL, config,
                extra_tags={"task_set": "si_er"},
            )
            mlflow_utils.log_config_params(config)
            mlflow_utils.log_trainer_history(trainer)
            mlflow_utils.mlflow.log_metric(
                "final_validation_er_accuracy",
                trainer.final_validation_er_accuracy,
            )
            mlflow_utils.mlflow.log_artifact(provenance_path)
            mlflow_utils.mlflow.log_artifact(traces_path)
            run_id = mlflow_utils.mlflow.active_run().info.run_id

        client = MlflowClient(tracking_uri=uri)
        run = client.get_run(run_id)
        self.assertEqual(run.data.tags["study_id"], STUDY_ID)
        self.assertEqual(run.data.tags["stage"], STAGE)
        self.assertEqual(run.data.tags["method"], method)
        self.assertEqual(run.data.tags["task_set"], "si_er")
        self.assertEqual(run.data.tags["representation"], "smp25")
        self.assertTrue(run.data.tags["mlflow.note.content"])
        self.assertEqual(
            run.data.metrics["final_validation_er_accuracy"],
            trainer.final_validation_er_accuracy,
        )
        # Per-epoch history plus the two provenance artifacts.
        self.assertIn("train_loss_all", run.data.metrics)
        self.assertIn("val_er_acc", run.data.metrics)
        artifacts = {artifact.path for artifact in client.list_artifacts(run_id)}
        self.assertIn("provenance.json", artifacts)
        self.assertIn("step_traces.json", artifacts)


if __name__ == "__main__":
    unittest.main()
