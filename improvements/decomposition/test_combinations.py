"""Small CPU checks for every supported FTN task combination."""

import os
import tempfile
import unittest

import torch
from torch.utils.data import TensorDataset

from improvements.decomposition.models.ftn_model import DownstreamMultiTaskModelFTN
from improvements.decomposition.trainers.ftn_trainer import MultiTasksModelTrainerFTN
from evaluator.evaluator_model import MultiTasksModelEvaluator


COMBINATIONS = ("ks", "si", "er", "ks_si", "ks_er", "si_er", "ks_si_er")


class CombinationTests(unittest.TestCase):
    def test_forward_backward_and_smoke_training(self):
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as root:
            for task_type in COMBINATIONS:
                with self.subTest(task_type=task_type):
                    model = DownstreamMultiTaskModelFTN(
                        upstream_model_type="wavlm_large",
                        task_type=task_type,
                        embedding_dim_shared1=8,
                        embedding_dim_shared2=8,
                        layer_pooling_type="smp",
                        layer_pooling_param=1.0,
                        dropout_prob_shared1=0.0,
                        dropout_prob_shared2=0.0,
                        ftn_rank=2,
                    )
                    tasks = task_type.split("_")
                    self.assertEqual(tuple(model.active_tasks), tuple(tasks))
                    self.assertEqual(len(model.task_updates), len(tasks))
                    x = torch.randn(2, 25, 1024, requires_grad=True)
                    output = model(x)
                    self.assertEqual(len(output.logits), len(tasks))
                    for logits, head in zip(output.logits, model.classifiers):
                        self.assertEqual(tuple(logits.shape), (2, head.out_features))
                    loss = sum(logits.square().mean() for logits in output.logits)
                    loss.backward()
                    self.assertTrue(torch.all(x.grad.abs().sum(dim=(0, 2)) > 0))
                    self.assertIsNotNone(model.hidden_layer.weight.grad)
                    for update in model.task_updates:
                        self.assertIsNotNone(update.up.weight.grad)
                    with self.assertRaises(ValueError):
                        model(torch.randn(2, 24, 1024))

                    labels = torch.zeros(2, len(tasks), dtype=torch.long)
                    dataset = TensorDataset(x.detach(), labels)
                    run_root = os.path.join(root, task_type)
                    trainer = MultiTasksModelTrainerFTN(
                        model=model, device=torch.device("cpu"), task_type=task_type,
                        training_cfg={"num_epochs": 1, "batch_size": 2,
                                      "learning_rate": 0.001, "saved_checkpoint_count": 1,
                                      "drop_last_train": False, "shuffle_train": False,
                                      "pin_memory": False},
                        results_root=os.path.join(run_root, "results"),
                        checkpoints_root=os.path.join(run_root, "checkpoints"),
                        training_data=dataset, validation_data=dataset,
                        diagnostics_cfg={"shared_gradient_analysis": False},
                    )
                    trainer.train()
                    evaluator = MultiTasksModelEvaluator(
                        model=model, device=torch.device("cpu"), task_type=task_type,
                        evaluation_cfg={"batch_size": 2},
                        results_root=os.path.join(run_root, "results"), dataset=dataset,
                        checkpoints_root=os.path.join(run_root, "checkpoints"),
                        checkpoint_tag="opt",
                        results_run_id=os.path.basename(trainer.results_dir),
                        checkpoint_run_id=os.path.basename(trainer.ckpt_dir),
                    )
                    stats = evaluator.write_metrics()
                    self.assertEqual(len(stats.accuracy_task), len(tasks))
                    self.assertTrue(os.path.isfile(evaluator.metrics_path_all))


if __name__ == "__main__":
    unittest.main()
