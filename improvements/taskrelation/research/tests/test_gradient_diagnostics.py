"""Data-free checks for observational gradient diagnostics."""

import copy
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
DOWNSTREAM_DIR = os.path.join(REPO_ROOT, "downstream")
for path in (DOWNSTREAM_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from improvements.gradient_diagnostics import make_gradient_diagnostic_trainer
from trainer.trainer_model import MultiTasksModelTrainer


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(4, 3)
        self.classifiers = nn.ModuleList([nn.Linear(3, 2), nn.Linear(3, 2)])

    def forward(self, input_seq):
        hidden = torch.tanh(self.shared(input_seq))
        logits = tuple(head(hidden) for head in self.classifiers)
        predictions = tuple(value.argmax(dim=1) for value in logits)
        return SimpleNamespace(logits=logits, prediction=predictions)

    def task_losses(self, inputs, targets):
        outputs = self(inputs)
        return {
            task: nn.functional.cross_entropy(outputs.logits[task], targets[task])
            for task in range(len(self.classifiers))
        }


class TinyTrainer:
    def __init__(self, model, results_dir, enabled):
        self.model = model
        self.device = torch.device("cpu")
        self.training_cfg = {
            "gradient_diagnostics": {
                "enabled": enabled,
                "sample_interval_steps": 1,
                "exclude_parameter_prefixes": ["classifiers."],
            }
        }
        self.num_epochs = 1
        self.train_dataloader = [None, None, None]
        self.task_array = ["a", "b"]
        self.num_tasks = 2
        self.results_dir = results_dir
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        self.gradients_were_clean_before_backward = True

    def train(self):
        inputs = torch.tensor([
            [0.2, -0.1, 0.4, 0.3],
            [-0.5, 0.6, 0.1, -0.2],
            [0.7, 0.2, -0.4, 0.5],
        ])
        targets = {
            0: torch.tensor([0, 1, 0]),
            1: torch.tensor([1, 0, 1]),
        }
        for _ in self.train_dataloader:
            self.optimizer.zero_grad(set_to_none=True)
            losses = self.model.task_losses(inputs, targets)
            self._record_task_gradient_diagnostics(losses, {0: 3, 1: 3})
            self.gradients_were_clean_before_backward &= all(
                parameter.grad is None for parameter in self.model.parameters()
            )
            sum(losses.values()).backward()
            self.optimizer.step()


class GradientDiagnosticTests(unittest.TestCase):
    def _run(self, enabled):
        torch.manual_seed(7)
        model = TinyModel()
        temporary_directory = tempfile.TemporaryDirectory()
        trainer_cls = make_gradient_diagnostic_trainer(TinyTrainer)
        trainer = trainer_cls(model, temporary_directory.name, enabled)
        trainer.train()
        return model, trainer, temporary_directory

    def _build_real_trainer(self, model, root, diagnostic):
        inputs = torch.tensor([
            [0.2, -0.1, 0.4, 0.3],
            [-0.5, 0.6, 0.1, -0.2],
            [0.7, 0.2, -0.4, 0.5],
        ])
        labels = torch.tensor([[0, 1], [1, 0], [0, 1]])
        dataset = TensorDataset(inputs, labels)
        training_config = {
            "num_epochs": 1,
            "batch_size": 3,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "saved_checkpoint_count": 1,
            "shuffle_train": False,
            "shuffle_val": False,
            "pin_memory": False,
            "drop_last_train": False,
            "drop_last_val": False,
            "num_workers": 0,
            "l1_lambda": 0.0,
            "l2_lambda": 0.0,
            "gradient_diagnostics": {
                "enabled": diagnostic,
                "sample_interval_steps": 1,
                "exclude_parameter_prefixes": ["classifiers."],
            },
        }
        trainer_cls = (
            make_gradient_diagnostic_trainer(MultiTasksModelTrainer)
            if diagnostic else MultiTasksModelTrainer
        )
        return trainer_cls(
            model=model,
            device=torch.device("cpu"),
            task_type="ks_si",
            training_cfg=training_config,
            results_root=os.path.join(root, "results"),
            checkpoints_root=os.path.join(root, "checkpoints"),
            training_data=dataset,
            validation_data=dataset,
            ignore_index=-1,
        ), (inputs, labels)

    def test_records_shared_gradients_without_populating_optimizer_buffers(self):
        model, trainer, temporary_directory = self._run(enabled=True)
        self.addCleanup(temporary_directory.cleanup)

        self.assertTrue(trainer.gradients_were_clean_before_backward)
        with open(trainer._gradient_artifact_path, "r") as artifact:
            payload = json.load(artifact)

        self.assertEqual(payload["sampled_steps"], 3)
        self.assertEqual(
            payload["shared_parameter_names"], ["shared.weight", "shared.bias"]
        )
        self.assertEqual(
            [record["phase"] for record in payload["records"]],
            ["early", "middle", "late"],
        )
        self.assertEqual(payload["summary"]["all"]["sample_count"], 3)
        self.assertEqual(
            payload["summary"]["all"]["pairwise_cosines"]["a_b"]["count"],
            3,
        )
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_observation_does_not_change_parameter_updates(self):
        observed_model, _, observed_directory = self._run(enabled=True)
        control_model, _, control_directory = self._run(enabled=False)
        self.addCleanup(observed_directory.cleanup)
        self.addCleanup(control_directory.cleanup)

        for name, observed in observed_model.state_dict().items():
            torch.testing.assert_close(observed, control_model.state_dict()[name])

    def test_wrapped_batch_matches_original_trainer_update(self):
        torch.manual_seed(17)
        control_model = TinyModel()
        observed_model = copy.deepcopy(control_model)

        with tempfile.TemporaryDirectory() as control_root:
            with tempfile.TemporaryDirectory() as observed_root:
                control_trainer, batch = self._build_real_trainer(
                    control_model, control_root, diagnostic=False
                )
                observed_trainer, _ = self._build_real_trainer(
                    observed_model, observed_root, diagnostic=True
                )
                control_stats = control_trainer._process_batch(batch, train_mode=True)
                observed_stats = observed_trainer._process_batch(batch, train_mode=True)

        self.assertAlmostEqual(control_stats.loss_all, observed_stats.loss_all)
        self.assertEqual(control_stats.batch_all, observed_stats.batch_all)
        for task_index in range(2):
            self.assertAlmostEqual(
                control_stats.loss_task[task_index],
                observed_stats.loss_task[task_index],
            )
        self.assertEqual(control_stats.batch_task, observed_stats.batch_task)
        self.assertEqual(control_stats.correct_task, observed_stats.correct_task)
        self.assertEqual(control_stats.samples_task, observed_stats.samples_task)
        self.assertEqual(control_stats.correct_all, observed_stats.correct_all)
        self.assertEqual(control_stats.samples_all, observed_stats.samples_all)
        self.assertEqual(
            control_stats.valid_count_task,
            observed_stats.valid_count_task,
        )
        for name, observed in observed_model.state_dict().items():
            torch.testing.assert_close(observed, control_model.state_dict()[name])


if __name__ == "__main__":
    unittest.main()
