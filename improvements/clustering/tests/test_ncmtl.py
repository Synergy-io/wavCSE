"""Synthetic, data-free checks for NCMTL task combinations."""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DOWNSTREAM_DIR = os.path.join(REPO_ROOT, "downstream")
for path in (DOWNSTREAM_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from improvements.clustering.models.ncmtl_model import DownstreamMultiTaskModelNCMTL
from improvements.clustering.trainers.ncmtl_trainer import MultiTasksModelTrainerNCMTL
from improvements.clustering.utils.ncmtl_clustering import (
    canonicalize_cluster_labels,
    cluster_candidate_weights,
)
from improvements.clustering.utils.candidate_row_distances import (
    compute_candidate_row_distances,
)
from improvements.clustering.utils.row_task_sharing import RowTaskSharing
from evaluator.evaluator_model import MultiTasksModelEvaluator


def build_model(
    task_type="ks_si_er",
    identical_candidate_initialization=False,
    shared_activation="none",
):
    return DownstreamMultiTaskModelNCMTL(
        upstream_model_type="wavlm_base",
        task_type=task_type,
        embedding_dim_shared1=16,
        embedding_dim_shared2=8,
        layer_pooling_type="mean",
        layer_pooling_param=None,
        dropout_prob_shared1=0.0,
        dropout_prob_shared2=0.0,
        shared_activation=shared_activation,
        identical_candidate_initialization=identical_candidate_initialization,
    )


class SyntheticDataset(Dataset):
    def __init__(self):
        generator = torch.Generator().manual_seed(7)
        self.inputs = torch.randn(6, 3, 768, generator=generator)
        self.labels = torch.tensor([
            [0, -1, -1], [-1, 1, -1], [-1, -1, 2],
            [3, -1, -1], [-1, 4, -1], [-1, -1, 1],
        ])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]


class SyntheticSubsetDataset(Dataset):
    def __init__(self, task_type):
        source = SyntheticDataset()
        indices = [{"ks": 0, "si": 1, "er": 2}[task] for task in task_type.split("_")]
        self.inputs = source.inputs
        self.labels = source.labels[:, indices]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]


def training_config():
    return {
        "num_epochs": 1, "batch_size": 6, "learning_rate": 0.001,
        "label_smoothing": 0.1,
        "gradient_clip_norm": 1.0,
        "weight_decay": 0.0, "saved_checkpoint_count": 1,
        "shuffle_train": False, "shuffle_val": False, "pin_memory": False,
        "drop_last_train": False, "drop_last_val": False, "num_workers": 0,
        "l1_lambda": 0.0, "l2_lambda": 0.0,
    }


def ncmtl_config(**overrides):
    config = {
        "alpha": 0.001, "num_clusters": 2, "cluster_every_n_batches": 1,
        "warmup_epochs": 10, "kmeans_random_state": 42, "kmeans_n_init": 1,
        "kmeans_max_iter": 100, "freeze_on_stability": True,
        "stability_patience": 50, "min_epochs_before_freeze": 1,
        "max_recluster_epochs": 4,
    }
    config.update(overrides)
    return config


class NCMTLModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_forward_candidate_and_flattened_shapes(self):
        model = build_model()
        inputs = torch.randn(2, 16, 768)
        outputs = model(input_seq=inputs)
        self.assertEqual([tuple(x.shape) for x in outputs.logits], [(2, 12), (2, 1251), (2, 4)])
        self.assertEqual(len(model.candidate_layers), 3)
        shared_dim = model.hidden_layer.out_features
        self.assertEqual(model.candidate_dim, shared_dim)
        self.assertTrue(all(
            layer.in_features == shared_dim and layer.out_features == shared_dim
            for layer in model.candidate_layers
        ))
        self.assertTrue(all(
            layer.weight.shape == torch.Size([shared_dim, shared_dim])
            for layer in model.candidate_layers
        ))
        self.assertEqual(
            [classifier.in_features for classifier in model.classifiers],
            [shared_dim, shared_dim, shared_dim],
        )
        self.assertEqual(
            [classifier.out_features for classifier in model.classifiers],
            [12, 1251, 4],
        )
        self.assertEqual(
            tuple(model.get_flattened_candidate_weights().shape),
            (3, shared_dim * shared_dim),
        )
        self.assertEqual(tuple(model.get_all_embeddings(inputs).shape), (2, 8))

    def test_identical_candidate_initialization_uses_separate_parameters(self):
        model = build_model(identical_candidate_initialization=True)
        first, second, third = model.candidate_layers

        self.assertTrue(torch.equal(first.weight, second.weight))
        self.assertTrue(torch.equal(first.weight, third.weight))
        self.assertIsNot(first.weight, second.weight)
        self.assertNotEqual(first.weight.data_ptr(), second.weight.data_ptr())

        with torch.no_grad():
            first.weight.add_(1.0)
        self.assertFalse(torch.equal(first.weight, second.weight))

    def test_independent_candidate_initialization_remains_available(self):
        model = build_model(identical_candidate_initialization=False)
        self.assertFalse(torch.equal(
            model.candidate_layers[0].weight,
            model.candidate_layers[1].weight,
        ))

    def test_configurable_shared_activation_runs_after_fc1_and_fc2(self):
        expected_types = {
            "none": torch.nn.Identity,
            "relu": torch.nn.ReLU,
            "gelu": torch.nn.GELU,
        }
        inputs = torch.randn(2, 3, 768)
        for name, expected_type in expected_types.items():
            with self.subTest(activation=name):
                model = build_model(shared_activation=name)
                calls = []
                handle = model.shared_activation.register_forward_hook(
                    lambda module, args, output: calls.append(tuple(output.shape))
                )
                outputs = model(inputs)
                handle.remove()

                self.assertIsInstance(model.shared_activation, expected_type)
                self.assertEqual(model.shared_activation_name, name)
                self.assertEqual(calls, [(2, 3, 16), (2, 8)])
                self.assertEqual(tuple(outputs.logits[0].shape), (2, 12))

    def test_invalid_shared_activation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "none, relu, gelu"):
            build_model(shared_activation="tanh")

    def test_production_candidate_and_classifier_dimensions(self):
        model = DownstreamMultiTaskModelNCMTL(
            upstream_model_type="wavlm_large",
            task_type="ks_si_er",
            embedding_dim_shared1=512,
            embedding_dim_shared2=2000,
            layer_pooling_type="mix",
            layer_pooling_param=0.5,
            dropout_prob_shared1=0.4,
            dropout_prob_shared2=0.6,
        )
        self.assertEqual(model.hidden_layer.out_features, 2000)
        self.assertTrue(all(
            layer.weight.shape == torch.Size([2000, 2000])
            and layer.bias is None
            for layer in model.candidate_layers
        ))
        self.assertEqual(
            [(head.in_features, head.out_features) for head in model.classifiers],
            [(2000, 12), (2000, 1251), (2000, 4)],
        )
        self.assertEqual(
            tuple(model.get_flattened_candidate_weights().shape),
            (3, 2000 * 2000),
        )
        inputs = torch.randn(2, 16, 1024)
        outputs = model(input_seq=inputs)
        self.assertEqual(
            [tuple(logits.shape) for logits in outputs.logits],
            [(2, 12), (2, 1251), (2, 4)],
        )
        self.assertEqual(tuple(model.get_all_embeddings(inputs).shape), (2, 2000))

    def test_candidate_row_distances_preserve_every_row(self):
        model = build_model(identical_candidate_initialization=True)
        with torch.no_grad():
            model.candidate_layers[1].weight[0].add_(1.0)
            model.candidate_layers[2].weight[1].add_(2.0)

        distances = compute_candidate_row_distances(
            model.get_candidate_weight_tensors(), ["ks", "si", "er"]
        )
        self.assertEqual(set(distances), {"ks_si", "ks_er", "si_er"})
        self.assertTrue(all(tuple(values.shape) == (8,) for values in distances.values()))
        self.assertGreater(float(distances["ks_si"][0]), 0.0)
        self.assertEqual(float(distances["ks_si"][1]), 0.0)

    def test_hard_sharing_and_cluster_loss(self):
        model = build_model()
        with torch.no_grad():
            model.candidate_layers[0].weight.fill_(1.0)
            model.candidate_layers[1].weight.fill_(3.0)
            model.candidate_layers[2].weight.fill_(8.0)
        model.set_cluster_assignments([0, 0, 1])
        model.share_candidate_weights_by_cluster()
        self.assertTrue(torch.equal(model.candidate_layers[0].weight, model.candidate_layers[1].weight))
        self.assertAlmostEqual(float(model.get_cluster_loss().item()), 0.0, places=7)

    def test_kmeans_and_canonicalization(self):
        self.assertEqual(canonicalize_cluster_labels([0, 1, 1]), canonicalize_cluster_labels([1, 0, 0]))
        vectors = torch.tensor([[0.0, 0.0], [0.1, 0.1], [10.0, 10.0]])
        labels = cluster_candidate_weights(vectors, 2, random_state=42)
        self.assertEqual(len(labels), 3)
        self.assertEqual(len(set(labels)), 2)

    def test_checkpoint_round_trip(self):
        model = build_model()
        model.set_cluster_assignments([0, 1, 0])
        model.cluster_frozen.fill_(True)
        with tempfile.NamedTemporaryFile(suffix=".pth") as checkpoint:
            torch.save(model.state_dict(), checkpoint.name)
            restored = build_model()
            restored.load_state_dict(torch.load(checkpoint.name, map_location="cpu"))
            outputs = restored(torch.randn(2, 3, 768))
        self.assertEqual(restored.get_cluster_state(), model.get_cluster_state())
        self.assertEqual(tuple(outputs.logits[1].shape), (2, 1251))

    def test_task_combinations_and_guard(self):
        dimensions = {"ks": 12, "si": 1251, "er": 4}
        for task_type in ("ks", "si", "er", "ks_si", "si_er", "ks_er", "ks_si_er"):
            with self.subTest(task_type=task_type):
                model = build_model(task_type)
                outputs = model(torch.randn(2, 3, 768))
                self.assertEqual(
                    [logits.shape[1] for logits in outputs.logits],
                    [dimensions[task] for task in task_type.split("_")],
                )
                self.assertEqual(len(model.candidate_layers), len(task_type.split("_")))
        for invalid in ("ks_si_er_ic", "ks_ks", "", "ks__er"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "unique tasks"):
                    build_model(invalid)

    def test_pairwise_soft_sharing_gates_each_row_by_distance(self):
        for task_names in (["ks", "si"], ["si", "er"], ["ks", "er"]):
            with self.subTest(task_names=task_names), tempfile.TemporaryDirectory() as directory:
                sharing = RowTaskSharing(
                    directory, task_names, sharing_mode="soft",
                    soft_min_margin=0.05, soft_full_margin=0.20,
                )
                weights = [
                    torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]]),
                    torch.tensor([[1.0, 0.0], [0.85, 0.0], [0.5, 0.0], [0.0, 0.0]]),
                ]
                sharing.initialize(weights, epoch=3)
                self.assertEqual(sharing.shared_assignment_counts(), {"_".join(task_names): 3})
                coefficients = sharing.sharing_coefficients
                self.assertEqual(float(coefficients[0]), 1.0)
                self.assertGreater(float(coefficients[1]), 0.0)
                self.assertLess(float(coefficients[1]), 1.0)
                self.assertEqual(float(coefficients[2]), 0.0)
                self.assertEqual(float(coefficients[3]), 1.0)
                before = [weight.clone() for weight in weights]
                sharing.share(weights)
                self.assertTrue(torch.equal(weights[0][2], before[0][2]))
                self.assertTrue(torch.equal(weights[1][2], before[1][2]))
                self.assertLess(float((weights[0][0] - weights[1][0]).abs().max()), 1e-6)
                self.assertLess(
                    float((weights[0][1] - weights[1][1]).abs().max()),
                    float((before[0][1] - before[1][1]).abs().max()),
                )


class NCMTLTrainerTests(unittest.TestCase):
    def test_three_pair_and_single_training_and_evaluation_paths(self):
        for task_type in ("ks_si_er", "si_er", "ks"):
            with self.subTest(task_type=task_type), tempfile.TemporaryDirectory() as directory:
                dataset = SyntheticSubsetDataset(task_type)
                trainer = MultiTasksModelTrainerNCMTL(
                    model=build_model(task_type), device=torch.device("cpu"),
                    task_type=task_type, training_cfg=training_config(),
                    results_root=os.path.join(directory, "results"),
                    checkpoints_root=os.path.join(directory, "checkpoints"),
                    training_data=dataset, validation_data=dataset,
                    ignore_index=-1,
                    ncmtl_cfg=ncmtl_config(
                        sharing_granularity="row", row_sharing_mode="soft",
                        row_warmup_mode="adaptive",
                        log_candidate_row_distances=True,
                    ),
                )
                trainer.train()
                with open(trainer.cluster_summary_path) as summary_file:
                    summary = json.load(summary_file)
                self.assertEqual(summary["task_type"], task_type)
                if task_type == "ks":
                    self.assertEqual(summary["strategy"], "no_pair_single_task")
                else:
                    self.assertTrue(os.path.exists(trainer.row_distance_logger.summary_path))
                evaluator = MultiTasksModelEvaluator(
                    model=trainer.model, device=torch.device("cpu"),
                    task_type=task_type,
                    evaluation_cfg={"batch_size": 6, "pin_memory": False},
                    results_root=os.path.join(directory, "results"),
                    dataset=dataset,
                    checkpoints_root=os.path.join(directory, "checkpoints"),
                    checkpoint_tag="epoch",
                    results_run_id=os.path.basename(trainer.results_dir),
                    checkpoint_run_id=os.path.basename(trainer.ckpt_dir),
                )
                stats = evaluator.write_metrics()
                evaluator.write_predictions_csv()
                self.assertEqual(len(stats.accuracy_task), len(task_type.split("_")))

    def _build_trainer(self, directory, **overrides):
        dataset = SyntheticDataset()
        return MultiTasksModelTrainerNCMTL(
            model=build_model(), device=torch.device("cpu"), task_type="ks_si_er",
            training_cfg=training_config(), results_root=os.path.join(directory, "results"),
            checkpoints_root=os.path.join(directory, "checkpoints"), training_data=dataset,
            validation_data=dataset, ignore_index=-1, ncmtl_cfg=ncmtl_config(**overrides),
        )

    def test_masked_labels_produce_finite_loss_and_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self._build_trainer(directory)
            self.assertEqual(trainer.training_loss_fn.label_smoothing, 0.1)
            self.assertEqual(trainer.validation_loss_fn.label_smoothing, 0.0)
            batch = next(iter(trainer.train_dataloader))
            trainer.current_epoch = 1
            stats = trainer._process_batch(batch, train_mode=True)
            self.assertTrue(torch.isfinite(torch.tensor(stats.loss_all)))
            self.assertTrue(any(p.grad is not None for p in trainer.model.parameters()))

    def test_gradient_clipping_runs_before_optimizer_step(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self._build_trainer(directory)
            batch = next(iter(trainer.train_dataloader))
            trainer.current_epoch = 1
            events = []

            original_step = trainer.optimizer.step

            def tracked_step(*args, **kwargs):
                events.append("step")
                return original_step(*args, **kwargs)

            trainer.optimizer.step = tracked_step
            with patch(
                "torch.nn.utils.clip_grad_norm_",
                side_effect=lambda *args, **kwargs: events.append("clip") or torch.tensor(0.0),
            ) as clip_mock:
                trainer._process_batch(batch, train_mode=True)

            self.assertEqual(events[:2], ["clip", "step"])
            self.assertEqual(clip_mock.call_args.kwargs["max_norm"], 1.0)
            self.assertTrue(clip_mock.call_args.kwargs["error_if_nonfinite"])

    def test_gradient_clipping_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            config = training_config()
            config["gradient_clip_norm"] = None
            dataset = SyntheticDataset()
            trainer = MultiTasksModelTrainerNCMTL(
                model=build_model(), device=torch.device("cpu"),
                task_type="ks_si_er", training_cfg=config,
                results_root=os.path.join(directory, "results"),
                checkpoints_root=os.path.join(directory, "checkpoints"),
                training_data=dataset, validation_data=dataset,
                ignore_index=-1, ncmtl_cfg=ncmtl_config(),
            )
            self.assertIsNone(trainer.gradient_clip_norm)

    def test_validation_has_no_clustering_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self._build_trainer(directory, warmup_epochs=0)
            trainer.model.set_cluster_assignments([0, 0, 1])
            assignments_before = trainer.model.cluster_assignments.clone()
            weights_before = [x.weight.detach().clone() for x in trainer.model.candidate_layers]
            trainer._process_batch(next(iter(trainer.val_dataloader)), train_mode=False)
            self.assertTrue(torch.equal(assignments_before, trainer.model.cluster_assignments))
            for before, layer in zip(weights_before, trainer.model.candidate_layers):
                self.assertTrue(torch.equal(before, layer.weight))

    def test_frozen_assignment_continues_hard_sharing(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self._build_trainer(directory, warmup_epochs=0)
            trainer.current_epoch = 2
            trainer.model.set_cluster_assignments([0, 0, 1])
            trainer.model.cluster_frozen.fill_(True)
            trainer._process_batch(
                next(iter(trainer.train_dataloader)), train_mode=True
            )
            self.assertTrue(torch.equal(
                trainer.model.candidate_layers[0].weight,
                trainer.model.candidate_layers[1].weight,
            ))

    def test_row_distance_diagnostics_write_epoch_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self._build_trainer(
                directory,
                log_candidate_row_distances=True,
            )
            trainer._process_data_loader(trainer.train_dataloader, train_mode=True)

            with open(trainer.row_distance_logger.matrices_path) as matrices_file:
                matrices = json.load(matrices_file)
            with open(trainer.row_distance_logger.values_path) as values_file:
                values = json.load(values_file)

            self.assertEqual(len(matrices), 1)
            self.assertEqual(len(values), 1)
            self.assertEqual(matrices[0]["stage"], "post_optimizer_pre_sharing")
            self.assertEqual(len(matrices[0]["matrix"]), 3)
            self.assertEqual(set(values[0]["values"]), {"ks_si", "ks_er", "si_er"})
            self.assertTrue(all(
                len(pair_values) == 8
                for pair_values in values[0]["values"].values()
            ))

    def test_row_sharing_freezes_warmup_assignments_and_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self._build_trainer(
                directory,
                sharing_granularity="row",
                warmup_epochs=1,
                log_candidate_row_distances=True,
            )
            trainer._process_data_loader(trainer.train_dataloader, train_mode=True)
            self.assertFalse(trainer.row_task_sharing.initialized)

            trainer._process_data_loader(trainer.train_dataloader, train_mode=True)
            self.assertTrue(trainer.row_task_sharing.initialized)
            self.assertEqual(trainer.row_task_sharing.assignment_epoch, 1)
            self.assertEqual(
                sum(trainer.row_task_sharing.assignment_counts().values()), 8
            )
            self.assertTrue(os.path.exists(
                trainer.row_task_sharing.assignment_csv_path
            ))
            self.assertTrue(os.path.exists(
                trainer.row_task_sharing.assignment_summary_path
            ))

            weights = trainer.model.get_candidate_weight_tensors()
            assignments = trainer.row_task_sharing.assignments
            pair_indices = ((0, 1), (0, 2), (1, 2))
            for pair_id, (first, second) in enumerate(pair_indices):
                mask = assignments == pair_id
                if bool(torch.any(mask)):
                    self.assertTrue(torch.equal(
                        weights[first][mask], weights[second][mask]
                    ))

    def test_adaptive_row_warmup_freezes_after_stability_patience(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self._build_trainer(
                directory,
                sharing_granularity="row",
                row_warmup_mode="adaptive",
                row_warmup_min_epochs=2,
                row_warmup_max_epochs=5,
                row_warmup_stability_threshold=0.0,
                row_warmup_stability_patience=2,
                log_candidate_row_distances=True,
            )

            for _ in range(3):
                trainer._process_data_loader(
                    trainer.train_dataloader, train_mode=True
                )

            self.assertFalse(trainer.row_task_sharing.initialized)
            self.assertTrue(trainer.row_task_sharing.ready_to_freeze)
            self.assertEqual(trainer.row_task_sharing.assignment_epoch, 3)
            self.assertEqual(trainer.row_task_sharing.stable_transition_count, 2)

            trainer._process_data_loader(trainer.train_dataloader, train_mode=True)
            self.assertTrue(trainer.row_task_sharing.initialized)
            self.assertTrue(os.path.exists(
                trainer.row_task_sharing.warmup_stability_path
            ))
            self.assertTrue(os.path.exists(
                trainer.row_task_sharing.warmup_summary_path
            ))

    def test_confidence_gate_leaves_uncertain_rows_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self._build_trainer(
                directory,
                sharing_granularity="row",
                row_warmup_mode="fixed",
                warmup_epochs=1,
                row_min_relative_margin=1.0,
            )
            trainer._process_data_loader(trainer.train_dataloader, train_mode=True)
            trainer._process_data_loader(trainer.train_dataloader, train_mode=True)

            self.assertTrue(trainer.row_task_sharing.initialized)
            self.assertEqual(
                sum(trainer.row_task_sharing.shared_assignment_counts().values()),
                0,
            )
            self.assertEqual(
                int(torch.sum(~trainer.row_task_sharing.shared_row_mask).item()),
                8,
            )

    def test_confidence_aware_soft_sharing_uses_frozen_coefficients(self):
        with tempfile.TemporaryDirectory() as directory:
            sharing = RowTaskSharing(
                directory,
                ["ks", "si", "er"],
                sharing_mode="soft",
                soft_min_margin=0.10,
                soft_full_margin=0.50,
            )
            weights = [
                torch.tensor([[0.0], [0.0], [0.0]]),
                torch.tensor([[0.7], [0.95], [2.0]]),
                torch.tensor([[1.7], [1.95], [10.0]]),
            ]
            sharing.initialize(weights, epoch=3)

            self.assertTrue(torch.allclose(
                sharing.sharing_coefficients,
                torch.tensor([0.5, 0.0, 1.0]),
                atol=1e-6,
            ))
            coefficients_before = sharing.sharing_coefficients.clone()
            self.assertAlmostEqual(float(sharing.cluster_loss(weights)), 2.1225, places=5)

            sharing.share(weights)
            self.assertTrue(torch.equal(
                sharing.sharing_coefficients, coefficients_before
            ))
            self.assertTrue(torch.allclose(weights[0][0], torch.tensor([0.175])))
            self.assertTrue(torch.allclose(weights[1][0], torch.tensor([0.525])))
            self.assertTrue(torch.allclose(weights[0][1], torch.tensor([0.0])))
            self.assertTrue(torch.allclose(weights[1][1], torch.tensor([0.95])))
            self.assertTrue(torch.allclose(weights[0][2], weights[1][2]))

            with open(sharing.assignment_summary_path) as summary_file:
                summary = json.load(summary_file)
            self.assertEqual(summary["sharing_mode"], "soft")
            self.assertEqual(summary["independent_rows"], 1)
            self.assertEqual(summary["partially_shared_rows"], 1)
            self.assertEqual(summary["fully_shared_rows"], 1)

    def test_soft_sharing_configuration_is_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ValueError, "greater than row_soft_min_margin"
            ):
                RowTaskSharing(
                    directory,
                    ["ks", "si", "er"],
                    sharing_mode="soft",
                    soft_min_margin=0.20,
                    soft_full_margin=0.20,
                )


if __name__ == "__main__":
    unittest.main()
