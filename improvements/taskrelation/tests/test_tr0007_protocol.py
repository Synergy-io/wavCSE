"""Public MSSL model and config behavior under the TR-0007 matched protocol.

Private compute plans, authorization records and scientific decisions are
validated in wavcse-research; this suite tests the executable model only.
"""

import os
import sys
import unittest

import torch
import yaml

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(TESTS_DIR, "..", "..", ".."))
for path in (REPO_ROOT, os.path.join(REPO_ROOT, "downstream")):
    if path not in sys.path:
        sys.path.insert(0, path)

from downstream.utils.parse_transformer_layers import parse_transformer_layers  # noqa: E402
from pooling.pooling import Pooling  # noqa: E402

from improvements import mlflow_utils  # noqa: E402

MSSL_DIR = os.path.join(REPO_ROOT, "improvements", "taskrelation", "04-mssl")
CONFIG_PATH = os.path.join(MSSL_DIR, "mssl_config.yml")
MSSL_MODEL_PATH = os.path.join(MSSL_DIR, "mssl_model.py")
WAVLM_LARGE_LAYERS = 25


def _load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_mssl_model_class():
    import importlib.util

    spec = importlib.util.spec_from_file_location("tr0007_mssl_model", MSSL_MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DownstreamMultiTaskModelMSSL


class LayerPolicyTests(unittest.TestCase):
    def test_config_resolves_to_all_25_wavlm_layers(self):
        config = _load_config()
        self.assertEqual(config["upstream"]["model_type"], "wavlm_large")
        self.assertEqual(config["upstream"]["selected_transformer_layers"], "all")

        layers = parse_transformer_layers(
            config["upstream"]["selected_transformer_layers"],
            config["upstream"]["model_type"],
        )
        self.assertEqual(list(layers), list(range(WAVLM_LARGE_LAYERS)))
        self.assertEqual(len(layers), WAVLM_LARGE_LAYERS)

    def test_layer_pooling_consumes_every_layer_row(self):
        # smp pools the layer axis with a non-degenerate weight for every
        # layer, so every one of the 25 rows must receive gradient.
        config = _load_config()
        pooling = Pooling(
            config["pooling"]["layer_pooling_type"],
            pooling_param=config["pooling"]["layer_pooling_param"],
        )
        inputs = torch.randn(2, WAVLM_LARGE_LAYERS, 1024, requires_grad=True)
        pooled = pooling.get_vector_after_pooling(inputs, dim=1)
        self.assertEqual(tuple(pooled.shape), (2, 1024))
        pooled.sum().backward()
        per_layer = inputs.grad.abs().sum(dim=(0, 2))
        self.assertEqual(int(per_layer.shape[0]), WAVLM_LARGE_LAYERS)
        self.assertTrue(
            bool((per_layer > 0).all()),
            "every one of the 25 layer rows must participate in pooling",
        )

    def test_relation_path_reaches_every_layer_row(self):
        # End to end on a tiny model built from the arm's own config: the
        # relation loss (the MSSL coupling term) must produce gradient on all
        # 25 layers, not only on the trunk's last layer.
        config = _load_config()
        torch.manual_seed(0)
        model_cls = _load_mssl_model_class()
        model = model_cls(
            upstream_model_type=config["upstream"]["model_type"],
            task_type="ks_si_er",
            embedding_dim_shared1=8,
            embedding_dim_shared2=4,
            layer_pooling_type=config["pooling"]["layer_pooling_type"],
            dropout_prob_shared1=0.0,
            dropout_prob_shared2=0.0,
            mssl_lambda_2=config["model"]["mssl_lambda_2"],
            mssl_lambda_0=config["model"]["mssl_lambda_0"],
            mssl_lambda_1=config["model"]["mssl_lambda_1"],
            mssl_admm_rho=config["model"]["mssl_admm_rho"],
            mssl_admm_iterations=config["model"]["mssl_admm_iterations"],
            normalize_w=config["model"]["normalize_w"],
            layer_pooling_param=config["pooling"]["layer_pooling_param"],
        )
        model.update_omega()

        inputs = torch.randn(2, WAVLM_LARGE_LAYERS, 1024, requires_grad=True)
        outputs = model(input_seq=inputs)
        loss = sum(logits.sum() for logits in outputs.logits) + model.get_relation_loss()
        loss.backward()
        per_layer = inputs.grad.abs().sum(dim=(0, 2))
        self.assertTrue(
            bool((per_layer > 0).all()),
            "the MSSL relation path must reach every one of the 25 layer rows",
        )


class FrozenProtocolTests(unittest.TestCase):
    def test_config_matches_the_registered_protocol_conditions(self):
        config = _load_config()
        self.assertEqual(config["pooling"]["frame_pooling_type"], "mean")
        self.assertEqual(config["pooling"]["layer_pooling_type"], "smp")
        self.assertEqual(config["pooling"]["layer_pooling_param"], 0.5)
        self.assertEqual(config["dataset"]["subset_percentage"], 100)
        self.assertEqual(config["dataset"]["ignore_index"], -1)
        self.assertEqual(config["model"]["embedding_dim_shared1"], 512)
        self.assertEqual(config["model"]["embedding_dim_shared2"], 2000)
        self.assertEqual(config["model"]["dropout_prob_shared1"], 0.4)
        self.assertEqual(config["model"]["dropout_prob_shared2"], 0.6)
        training = config["training"]
        self.assertEqual(training["num_epochs"], 30)
        self.assertEqual(training["batch_size"], 2048)
        self.assertEqual(training["learning_rate"], 0.0025)
        self.assertEqual(training["weight_decay"], 5e-8)
        self.assertEqual(training["l1_lambda"], 1e-7)
        self.assertEqual(training["l2_lambda"], 1e-5)
        self.assertEqual(training["factor"], 0.5)
        self.assertTrue(training["drop_last_train"])
        self.assertFalse(training["drop_last_val"])
        self.assertTrue(training["pin_memory"])
        self.assertEqual(training["num_workers"], 4)
        self.assertEqual(config["seed"], 42)

    def test_config_is_matched_to_the_in_category_control(self):
        # Protocol §5: the arm may vary only the relation mechanism. Every
        # shared block must therefore equal the control's, byte for byte.
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            arm = yaml.safe_load(handle)
        control_path = os.path.join(REPO_ROOT, "improvements", "taskrelation",
                                    "01-mtrl", "mtrl_poolingwinner_25L_config.yml")
        with open(control_path, "r", encoding="utf-8") as handle:
            control = yaml.safe_load(handle)
        for block in ("upstream", "dataset", "pooling", "training", "evaluation"):
            self.assertEqual(arm[block], control[block], "block {!r} differs".format(block))
        self.assertEqual(arm["device"], control["device"])
        for key in ("embedding_dim_shared1", "embedding_dim_shared2",
                    "dropout_prob_shared1", "dropout_prob_shared2"):
            self.assertEqual(arm["model"][key], control["model"][key])
        # The mechanism-specific blocks are the only difference.
        self.assertNotIn("mtrl_lambda", arm["model"])
        self.assertIn("mtrl_lambda", control["model"])


class PublishedFormulationTests(unittest.TestCase):
    def test_lambda_values_and_the_lambda_2_labelling(self):
        config = _load_config()
        model_cfg = config["model"]
        self.assertEqual(model_cfg["mssl_lambda_0"], 1.0)
        self.assertEqual(model_cfg["mssl_lambda_1"], 0.0)
        self.assertIsNone(model_cfg["mssl_admm_rho"])
        self.assertGreaterEqual(model_cfg["mssl_admm_iterations"], 1000)
        self.assertTrue(model_cfg["normalize_w"])
        # lambda_2 = 0.01 is a researcher-fixed screening value (DEC-0016), not
        # a paper default: the source paper cross-validates lambda_1/lambda_2
        # and publishes neither a value nor a transferable scale.
        self.assertEqual(model_cfg["mssl_lambda_2"], 0.01)
        self.assertEqual(config["mssl"]["lambda_2_selection"], "researcher-fixed")
        # The label is machine-readable provenance, so it reaches MLflow with
        # the rest of the flattened config.

    def test_lambda_2_has_no_default_in_the_implementation(self):
        model_cls = _load_mssl_model_class()
        with self.assertRaises(TypeError):
            model_cls(
                upstream_model_type="wavlm_large",
                task_type="ks_si_er",
                embedding_dim_shared1=8,
                embedding_dim_shared2=4,
                layer_pooling_type="smp",
                dropout_prob_shared1=0.0,
                dropout_prob_shared2=0.0,
                layer_pooling_param=0.5,
            )


class ProvenanceTests(unittest.TestCase):
    def test_run_name_and_research_tags_identify_the_study_stage_arm_and_seed(self):
        config = _load_config()
        self.assertEqual(
            mlflow_utils.build_research_run_name(config, "mssl", "ks_si_er"),
            "TR-0007__screen__p-mssl__ks_si_er__smp25__s42",
        )
        research = config["research"]
        self.assertEqual(research["study_id"], "TR-0007")
        self.assertEqual(research["stage"], "screen")
        self.assertEqual(research["method"], "p-mssl")
        self.assertEqual(research["family"], "task_relation_learning")
        self.assertEqual(research["task_set"], "ks_si_er")
        self.assertEqual(research["representation"], "smp25")

    def test_tracking_uri_and_experiment_follow_the_project_convention(self):
        config = _load_config()
        self.assertEqual(
            config["mlflow"]["tracking_uri"],
            "https://dagshub.com/Ke-vin-S/wavCSE.mlflow",
        )
        self.assertEqual(config["mlflow"]["experiment_name"], "taskrelation-mssl")


if __name__ == "__main__":
    unittest.main()
