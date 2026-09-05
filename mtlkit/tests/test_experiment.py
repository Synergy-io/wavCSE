"""Unit tests for mtlkit/experiment.py (Eng Review decision 1B: Hydra adapter).

Includes the config-driven-run success criterion (a full experiment config
resolves end-to-end from a YAML file alone; the override escape hatch
overrides at least one YAML-expressed field) and the negative-path config
test (Eng Review decision 5A): a malformed/incomplete config raises a clear,
actionable ConfigError before training starts.
"""

import os
import tempfile
import unittest

import mtlkit.experiment as experiment


def _write_yaml(directory: str, name: str, content: str) -> None:
    with open(os.path.join(directory, f"{name}.yaml"), "w") as f:
        f.write(content)


class LoadExperimentConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.config_dir = self.tmpdir.name

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_loads_full_config_from_yaml_alone(self):
        _write_yaml(
            self.config_dir,
            "build_model",
            """
task_type: ks_si_er
upstream_model_type: wavlm_large
embedding_dim_shared1: 256
embedding_dim_shared2: 128
layer_pooling_type: mean
num_epochs: 5
batch_size: 64
mlflow_experiment_name: wavcse-mtlkit
""",
        )
        config = experiment.load_experiment_config(self.config_dir, "build_model")
        self.assertEqual(config.task_type, "ks_si_er")
        self.assertEqual(config.upstream_model_type, "wavlm_large")
        self.assertEqual(config.num_epochs, 5)
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.mlflow_experiment_name, "wavcse-mtlkit")

    def test_override_escape_hatch_overrides_a_yaml_expressed_field(self):
        _write_yaml(
            self.config_dir,
            "build_model",
            """
task_type: ks_si_er
upstream_model_type: wavlm_large
num_epochs: 5
""",
        )
        config = experiment.load_experiment_config(
            self.config_dir, "build_model", overrides=["num_epochs=50"]
        )
        self.assertEqual(config.num_epochs, 50)
        # untouched fields keep their YAML-expressed values
        self.assertEqual(config.task_type, "ks_si_er")

    def test_unknown_fields_land_in_extra_not_dropped(self):
        _write_yaml(
            self.config_dir,
            "build_model",
            """
task_type: ks_si
upstream_model_type: wavlm_large
dropout_prob_shared1: 0.3
""",
        )
        config = experiment.load_experiment_config(self.config_dir, "build_model")
        self.assertEqual(config.extra["dropout_prob_shared1"], 0.3)

    def test_apply_python_override_mutates_in_place(self):
        _write_yaml(
            self.config_dir,
            "build_model",
            "task_type: ks_si\nupstream_model_type: wavlm_large\n",
        )
        config = experiment.load_experiment_config(self.config_dir, "build_model")

        def _bump_batch_size(cfg):
            cfg.batch_size = 128

        result = experiment.apply_python_override(config, _bump_batch_size)
        self.assertIs(result, config)
        self.assertEqual(config.batch_size, 128)


class NegativePathConfigTests(unittest.TestCase):
    """Eng Review decision 5A: malformed/incomplete config raises a clear
    error before training starts, not a stack trace deep inside the trainer."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.config_dir = self.tmpdir.name

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_missing_required_field_raises_configerror(self):
        _write_yaml(self.config_dir, "build_model", "num_epochs: 5\n")
        with self.assertRaises(experiment.ConfigError) as ctx:
            experiment.load_experiment_config(self.config_dir, "build_model")
        message = str(ctx.exception)
        self.assertIn("task_type", message)
        self.assertIn("upstream_model_type", message)

    def test_partial_required_fields_still_raises(self):
        _write_yaml(self.config_dir, "build_model", "task_type: ks_si\n")
        with self.assertRaises(experiment.ConfigError) as ctx:
            experiment.load_experiment_config(self.config_dir, "build_model")
        self.assertIn("upstream_model_type", str(ctx.exception))
        self.assertNotIn("task_type,", str(ctx.exception))

    def test_missing_config_file_raises_configerror_not_deep_stacktrace(self):
        with self.assertRaises(experiment.ConfigError):
            experiment.load_experiment_config(self.config_dir, "does_not_exist")

    def test_missing_config_dir_raises_configerror(self):
        with self.assertRaises(experiment.ConfigError) as ctx:
            experiment.load_experiment_config(
                os.path.join(self.config_dir, "nope"), "build_model"
            )
        self.assertIn("does not exist", str(ctx.exception))

    def test_malformed_yaml_raises_configerror(self):
        _write_yaml(self.config_dir, "build_model", "task_type: [unclosed\n")
        with self.assertRaises(experiment.ConfigError):
            experiment.load_experiment_config(self.config_dir, "build_model")


class BuildRunNameTests(unittest.TestCase):
    def test_matches_improvements_mlflow_utils_pattern(self):

        name = experiment.build_run_name("mtlkit", "combine", "ks_si_er")
        self.assertRegex(
            name, r"^mtlkit_combine_ks_si_er_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$"
        )

    def test_suffix_appended(self):
        name = experiment.build_run_name("mtlkit", "combine", "ks_si", suffix="fold0")
        self.assertTrue(name.endswith("_fold0"))


class StartMlflowRunTests(unittest.TestCase):
    def test_run_logs_params_and_ends_cleanly(self):
        import mlflow

        with tempfile.TemporaryDirectory() as tmp_tracking_dir:
            config = experiment.ExperimentConfig(
                task_type="ks_si",
                upstream_model_type="wavlm_large",
                mlflow_tracking_uri=f"file://{tmp_tracking_dir}",
                mlflow_experiment_name="mtlkit-test",
            )
            with experiment.start_mlflow_run(config, category="mtlkit", model="test") as run:
                run_id = run.info.run_id

            client = mlflow.tracking.MlflowClient(tracking_uri=config.mlflow_tracking_uri)
            finished_run = client.get_run(run_id)
            self.assertEqual(finished_run.data.params["task_type"], "ks_si")
            self.assertEqual(finished_run.info.status, "FINISHED")

    def test_run_ends_on_exception(self):
        import mlflow

        with tempfile.TemporaryDirectory() as tmp_tracking_dir:
            config = experiment.ExperimentConfig(
                task_type="ks_si",
                upstream_model_type="wavlm_large",
                mlflow_tracking_uri=f"file://{tmp_tracking_dir}",
                mlflow_experiment_name="mtlkit-test-exc",
            )
            with self.assertRaises(RuntimeError):
                with experiment.start_mlflow_run(config, category="mtlkit", model="test"):
                    raise RuntimeError("boom")
            self.assertIsNone(mlflow.active_run())


if __name__ == "__main__":
    unittest.main()
