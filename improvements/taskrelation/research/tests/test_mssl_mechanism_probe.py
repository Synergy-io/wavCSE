"""The MSSL mechanism probe must be read-only with respect to training.

TR-0013 records its mechanism bundle from inside the arm's own batch step, so
the instrumentation itself has to be proven not to change what the run does:
`torch.autograd.grad(..., retain_graph=True)` returns gradients instead of
accumulating them into `parameter.grad`, and the graph the training step's own
`backward()` consumes stays intact. These tests build the *real* MSSL model at
tiny dimensions and the real trainer methods by construction (no rewritten
copy of the batch step), then assert:

1. measuring writes no `.grad` buffer and leaves the subsequent backward's
   gradients and the optimizer's parameter update bit-identical to a run with
   no measurement at all;
2. the recorded bundle carries the fields the study's evidence layer reads
   (scale, support, partials, eigenvalues, geometry, coupling value, and the
   gradient norms and ratio);
3. the coupling's gradient reaches only the classifier heads, so the
   head-scoped and all-parameter relation norms agree.
"""

import importlib.util
import json
import os
import sys
import tempfile
import unittest

import torch

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
for path in (REPO_ROOT, os.path.join(REPO_ROOT, "downstream")):
    if path not in sys.path:
        sys.path.insert(0, path)

from trainer.trainer_utils import masked_ce_loss  # noqa: E402

MSSL_DIR = os.path.join(REPO_ROOT, "improvements", "taskrelation", "04-mssl")
MODEL_PATH = os.path.join(MSSL_DIR, "mssl_model.py")
TRAINER_PATH = os.path.join(MSSL_DIR, "mssl_trainer.py")
NUM_LAYERS = 25


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_model(module, lambda_2=0.01):
    torch.manual_seed(0)
    model = module.DownstreamMultiTaskModelMSSL(
        upstream_model_type="wavlm_large",
        task_type="ks_si_er",
        embedding_dim_shared1=8,
        embedding_dim_shared2=4,
        layer_pooling_type="smp",
        dropout_prob_shared1=0.0,
        dropout_prob_shared2=0.0,
        mssl_lambda_2=lambda_2,
        mssl_lambda_0=1.0,
        mssl_lambda_1=0.0,
        mssl_admm_rho=None,
        mssl_admm_iterations=200,
        normalize_w=True,
        layer_pooling_param=0.5,
    )
    model.update_omega()
    return model


def _bare_trainer(trainer_module, model):
    """The real trainer's methods without the full training harness.

    Nothing about the batch step, the probe or the recorders is reimplemented
    here: only the attributes those methods read are provided.
    """
    trainer = object.__new__(trainer_module.MultiTasksModelTrainerMSSL)
    trainer.model = model
    trainer.device = torch.device("cpu")
    trainer.task_array = ["ks", "si", "er"]
    trainer.num_tasks = 3
    trainer.current_epoch = 4
    trainer.mssl_warmup_epochs = 3
    trainer.omega_update_frequency = 1
    trainer.lambda_2_selection = "validation-selected-over-published-grid-0.01-0.1"
    trainer.omega_history = []
    trainer.coupling_history = []
    trainer._probe_batch_seen = 0
    trainer._relation_parameter_names = tuple(
        name for name, _ in model.named_parameters()
        if name.startswith("classifiers.")
    )
    trainer.training_cfg = {}
    return trainer


def _forward(model):
    inputs = torch.randn(2, NUM_LAYERS, 1024)
    return model(input_seq=inputs), inputs


def _task_loss(model, outputs):
    loss = torch.zeros(())
    loss_function = torch.nn.CrossEntropyLoss()
    for logits in outputs.logits:
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        component, _ = masked_ce_loss(
            logits=logits, labels=labels, loss_fn=loss_function, ignore_index=-1,
        )
        if component is not None:
            loss = loss + component / 3.0
    return loss


class MechanismProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_module = _load(MODEL_PATH, "tr0013_mssl_model")
        cls.trainer_module = _load(TRAINER_PATH, "tr0013_mssl_trainer")

    def _run_step(self, probe):
        model = _build_model(self.model_module)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = _bare_trainer(self.trainer_module, model)
        outputs, _ = _forward(model)
        task_loss = _task_loss(model, outputs)
        coupling = model.get_relation_loss()
        optimizer.zero_grad(set_to_none=True)
        if probe:
            trainer._probe_gradient_scales(task_loss, coupling)
        (task_loss + coupling).backward()
        optimizer.step()
        return {
            "params": {name: parameter.detach().clone()
                       for name, parameter in model.named_parameters()},
            "history": list(trainer.coupling_history),
        }

    def test_probe_does_not_change_the_optimizer_step(self):
        reference = self._run_step(probe=False)
        measured = self._run_step(probe=True)
        self.assertEqual(set(reference["params"]), set(measured["params"]))
        for name, control in reference["params"].items():
            torch.testing.assert_close(measured["params"][name], control)
        self.assertEqual(reference["history"], [])

    def test_probe_records_the_studys_required_scale_fields(self):
        measured = self._run_step(probe=True)
        self.assertEqual(len(measured["history"]), 1)
        record = measured["history"][0]
        for key in ("epoch", "task_loss_value", "task_grad_norm_all",
                    "task_grad_norm_heads", "relation_value",
                    "relation_grad_norm_all", "relation_grad_norm_heads",
                    "relation_over_task_grad_ratio_heads",
                    "relation_over_task_grad_ratio_all"):
            self.assertIn(key, record)
        self.assertGreater(record["task_grad_norm_heads"], 0.0)
        self.assertGreater(record["relation_grad_norm_heads"], 0.0)
        # The published coupling term depends on the parameter summaries only,
        # so it reaches the classifier heads and nothing else.
        self.assertAlmostEqual(
            record["relation_grad_norm_all"], record["relation_grad_norm_heads"],
            places=9,
        )
        self.assertAlmostEqual(
            record["relation_over_task_grad_ratio_heads"],
            record["relation_grad_norm_heads"] / record["task_grad_norm_heads"],
            delta=1e-3,
        )

    def test_probe_writes_no_optimizer_grad_buffers(self):
        model = _build_model(self.model_module)
        trainer = _bare_trainer(self.trainer_module, model)
        outputs, _ = _forward(model)
        task_loss = _task_loss(model, outputs)
        coupling = model.get_relation_loss()
        for parameter in model.parameters():
            parameter.grad = None
        trainer._probe_gradient_scales(task_loss, coupling)
        self.assertTrue(
            all(parameter.grad is None for parameter in model.parameters()),
            "the probe must not accumulate into .grad",
        )
        # And the graph it did not consume is still usable for the real step.
        (task_loss + coupling).backward()


class MechanismRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_module = _load(MODEL_PATH, "tr0013_mssl_model_for_record")
        cls.trainer_module = _load(TRAINER_PATH, "tr0013_mssl_trainer_for_record")

    def test_omega_snapshot_carries_the_required_bundle(self):
        for lambda_2 in (0.01, 0.1):
            model = _build_model(self.model_module, lambda_2=lambda_2)
            trainer = _bare_trainer(self.trainer_module, model)
            trainer._record_omega(3)
            entry = trainer.omega_history[-1]
            for key in ("epoch", "omega", "partial_correlations", "dual_violation",
                        "relative_duality_gap", "lambda_2", "lambda_0", "omega_trace",
                        "omega_eigenvalues", "mean_abs_offdiagonal",
                        "zero_offdiagonals", "support_edges", "coupling_value",
                        "summary_gram", "summary_cosines", "summary_gram_eigenvalues",
                        "summary_row_norms", "summary_raw_row_norms"):
                self.assertIn(key, entry, "lambda_2={}".format(lambda_2))
            self.assertEqual(entry["lambda_2"], lambda_2)
            self.assertEqual(entry["support_edges"] + entry["zero_offdiagonals"], 3)
            self.assertLessEqual(entry["support_edges"], 3)
            self.assertEqual(len(entry["omega_eigenvalues"]), 3)
            self.assertEqual(len(entry["summary_cosines"]), 3)
            # Rows are unit-normalised by the adapter (down to its epsilon
            # floor), and the raw row norms report the scale that normalisation
            # removed: the two must be consistent with the adapter's own rule.
            self.assertEqual(len(entry["summary_row_norms"]), 3)
            self.assertEqual(len(entry["summary_raw_row_norms"]), 3)
            for normalised, raw in zip(entry["summary_row_norms"],
                                       entry["summary_raw_row_norms"]):
                self.assertTrue(0.0 < normalised <= 1.0)
                self.assertAlmostEqual(
                    normalised, raw / (raw + model.omega_epsilon), places=4
                )

    def test_support_counts_only_the_strict_upper_edges(self):
        # Regression: `torch.triu` returns the whole matrix with its lower part
        # zeroed, so counting zeros in its output counted that zeroed region too
        # and reported e.g. 8 zeros / -5 edges for the 3-task snapshot below.
        # The raw Omega was always recorded correctly; this pins the derived
        # counts to the strict-upper edge pairs.
        model = _build_model(self.model_module, lambda_2=0.01)
        trainer = _bare_trainer(self.trainer_module, model)
        snapshot = torch.tensor([[10.0, -9.0, 0.0],
                                 [-9.0, 10.0, 0.0],
                                 [0.0, 0.0, 20.0]])
        model.omega.copy_(snapshot)
        trainer._record_omega(7)
        entry = trainer.omega_history[-1]
        self.assertEqual(entry["zero_offdiagonals"], 2)
        self.assertEqual(entry["support_edges"], 1)
        self.assertEqual(entry["support_edges"] + entry["zero_offdiagonals"], 3)
        self.assertEqual(entry["partial_correlations"][0][2], 0.0)
        self.assertGreater(entry["partial_correlations"][0][1], 0.0)

    def test_coupling_scale_artifact_is_written_with_its_provenance(self):
        model = _build_model(self.model_module, lambda_2=0.1)
        trainer = _bare_trainer(self.trainer_module, model)
        outputs, _ = _forward(model)
        trainer._probe_gradient_scales(
            _task_loss(model, outputs), model.get_relation_loss()
        )
        with tempfile.TemporaryDirectory() as directory:
            trainer.results_dir = directory
            trainer._save_coupling_scale()
            with open(os.path.join(directory, "coupling_scale.json")) as handle:
                payload = json.load(handle)
        self.assertEqual(payload["method"], "mssl")
        self.assertEqual(payload["lambda_0"], 1.0)
        self.assertEqual(payload["lambda_1"], 0.0)
        self.assertEqual(payload["lambda_2"], 0.1)
        self.assertEqual(
            payload["lambda_2_selection"],
            "validation-selected-over-published-grid-0.01-0.1",
        )
        self.assertEqual(payload["coupling_onset_epoch"], 4)
        self.assertEqual(len(payload["history"]), 1)


if __name__ == "__main__":
    unittest.main()
