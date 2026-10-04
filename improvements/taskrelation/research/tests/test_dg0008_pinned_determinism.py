"""DG-0008 pinned execution: tensor determinism, the aux-loss switch, fixed LR,
checkpoint/restart semantics, the held-out gate and the frozen classification.

A tiny real training run (a few synthetic embeddings, batch 4) exercises the
actual trainer over the actual manifest-driven schedule, so the arm switche and
the deterministic trace are observed rather than described.
"""

import os
import sys
import tempfile
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
for path in (REPO_ROOT, os.path.join(REPO_ROOT, "downstream")):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch  # noqa: E402

from improvements.seed_utils import set_seed  # noqa: E402
from improvements.taskrelation.research.dg0008 import analysis as analysis_module  # noqa: E402
from improvements.taskrelation.research.dg0008 import gate as gate_module  # noqa: E402
from improvements.taskrelation.research.dg0008 import trainer as trainer_module  # noqa: E402
from improvements.taskrelation.research.dg0008.manifest import ManifestError  # noqa: E402
from improvements.taskrelation.research.dg0008.sampler import (  # noqa: E402
    key_to_index,
    plain_dataset,
)
from model.downstream_model import DownstreamMultiTaskModel  # noqa: E402

UPSTREAM = "wavlm_large"
POOL = "mean"
KEYS_0 = ["bed/{:03d}.wav".format(i) for i in range(4)]
KEYS_1 = ["Session1/sentences/wav/Ses01F_impro01/{:03d}.wav".format(i) for i in range(4)]
VECTORS = [(KEYS_0, [0, 1, 2, 3]), (KEYS_1, [0, 1, 2, 3])]
STEPS = [
    [[0, key] for key in KEYS_0],
    [[1, key] for key in KEYS_1],
]
N_ER = [0, 4]


def _training_cfg():
    return {
        "num_epochs": 5,
        "batch_size": 4,
        "learning_rate": 0.0025,
        "weight_decay": 0.00000005,
        "saved_checkpoint_count": 1,
        "shuffle_train": False,
        "shuffle_val": False,
        "pin_memory": False,
        "drop_last_train": True,
        "drop_last_val": False,
        "num_workers": 0,
        "l1_lambda": 0.0000001,
        "l2_lambda": 0.00001,
    }


class DeterminismPinningTests(unittest.TestCase):
    def setUp(self):
        self.previous = torch.are_deterministic_algorithms_enabled()
        self.previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()

    def tearDown(self):
        torch.use_deterministic_algorithms(self.previous, warn_only=self.previous_warn)

    def test_apply_determinism_sets_every_required_flag(self):
        trainer_module.apply_determinism()
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)


class _Harness(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.embedding_root = os.path.join(self.root, "embedding", UPSTREAM, POOL)
        self._write_embeddings()
        self.dataset = plain_dataset(
            VECTORS, "ks_er", embedding_root=self.embedding_root,
            upstream_model_type=UPSTREAM, frame_pool_id=POOL,
            transformer_layer_array=list(range(25)),
        )
        self._previous = torch.are_deterministic_algorithms_enabled()
        self._previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
        trainer_module.apply_determinism()

    def tearDown(self):
        torch.use_deterministic_algorithms(self._previous, warn_only=self._previous_warn)
        self._tmp.cleanup()

    def _write_embeddings(self):
        for component_dir, keys in (("speechcommand", KEYS_0), ("iemocap", KEYS_1)):
            for key in keys:
                relative = key.replace(".wav", "_{}_{}.pt".format(UPSTREAM, POOL))
                path = os.path.join(self.embedding_root, component_dir, relative)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(torch.zeros(25, 1024), path)

    def _model(self):
        return DownstreamMultiTaskModel(
            upstream_model_type=UPSTREAM,
            task_type="ks_er",
            embedding_dim_shared1=8,
            embedding_dim_shared2=4,
            layer_pooling_type="smp",
            layer_pooling_param=0.5,
            dropout_prob_shared1=0.4,
            dropout_prob_shared2=0.6,
        )

    def _train(self, aux, seed=0, name=None):
        set_seed(seed)
        model = self._model()
        directory = os.path.join(self.root, name or ("pair" if aux else "control"))
        trainer = trainer_module.DG0008Trainer(
            model=model,
            device=torch.device("cpu"),
            task_type="ks_er",
            training_cfg=_training_cfg(),
            results_root=os.path.join(directory, "results"),
            checkpoints_root=os.path.join(directory, "checkpoints"),
            training_data=self.dataset,
            validation_data=self.dataset,
            ignore_index=-1,
            seed=seed,
            aux_loss_enabled=aux,
            epoch_steps=[STEPS for _ in range(5)],
            target_task="er",
        )
        trainer.bind_key_index(key_to_index(VECTORS), [N_ER for _ in range(5)])
        trainer.train()
        return trainer


class FixedLRAndTraceTests(_Harness):
    def test_learning_rate_is_constant_at_every_step(self):
        trainer = self._train(aux=True)
        self.assertTrue(trainer.step_lr_trace)
        self.assertTrue(all(lr == 0.0025 for lr in trainer.step_lr_trace))
        self.assertEqual(len(trainer.step_loss_trace), 10)

    def test_consumed_er_counts_equal_the_frozen_epoch_indexed_counts(self):
        trainer = self._train(aux=False)
        self.assertTrue(trainer.er_counts_match)
        self.assertEqual(trainer.consumed_n_er, [N_ER for _ in range(5)])


class ArmAndRepeatTests(_Harness):
    def test_arms_share_initialization_and_pre_forward_rng_sequence(self):
        pair = self._train(aux=True, name="pair")
        control = self._train(aux=False, name="control")
        self.assertEqual(pair.initialization_digest, control.initialization_digest)
        self.assertEqual(
            trainer_module.rng_sequence_digest(pair.rng_trace),
            trainer_module.rng_sequence_digest(control.rng_trace),
        )
        self.assertEqual(len(pair.rng_trace), 10)

    def test_aux_switch_changes_the_trajectory(self):
        pair = self._train(aux=True, name="pair2")
        control = self._train(aux=False, name="control2")
        self.assertNotEqual(
            trainer_module.model_state_digest(pair.model),
            trainer_module.model_state_digest(control.model),
        )

    def test_same_arm_repeat_is_bitwise_identical(self):
        first = self._train(aux=False, seed=0, name="r1")
        second = self._train(aux=False, seed=0, name="r2")
        self.assertEqual(first.step_loss_trace, second.step_loss_trace)
        self.assertEqual(
            first.final_validation_er_accuracy, second.final_validation_er_accuracy
        )
        self.assertEqual(
            first.fixed_final_checkpoint_sha256, second.fixed_final_checkpoint_sha256
        )

    def test_checkpoint_is_created_read_back_and_hashed(self):
        trainer = self._train(aux=True)
        path = trainer.fixed_final_checkpoint_path
        self.assertTrue(os.path.exists(path))
        self.assertEqual(trainer_module.file_sha256(path), trainer.fixed_final_checkpoint_sha256)
        torch.load(path, map_location="cpu")


class RestartRefusalTests(_Harness):
    def test_resume_is_refused_once_a_run_record_exists(self):
        directory = os.path.join(self.root, "resume")
        os.makedirs(directory, exist_ok=True)
        self.assertTrue(trainer_module.refuse_resume(directory))
        with open(os.path.join(directory, "run.json"), "wb") as handle:
            handle.write(b"{}")
        with self.assertRaises(ManifestError):
            trainer_module.refuse_resume(directory)


class HeldOutGateTests(unittest.TestCase):
    def _valid_gate(self, reads=0):
        return {
            "schema": "dg0008.validity-gate.v1",
            "stage1_valid": True,
            "counts": {"held_out_reads_before_gate": reads},
        }

    def test_admits_only_a_valid_gate_with_no_premature_read(self):
        self.assertTrue(gate_module.assert_held_out_admitted(self._valid_gate()))
        with self.assertRaises(ManifestError):
            gate_module.assert_held_out_admitted(self._valid_gate(reads=1))
        invalid = self._valid_gate()
        invalid["stage1_valid"] = False
        with self.assertRaises(ManifestError):
            gate_module.assert_held_out_admitted(invalid)
        with self.assertRaises(ManifestError):
            gate_module.assert_held_out_admitted({"schema": "other"})

    def test_pair_mismatch_detection(self):
        base = {field: 1 for field in gate_module.PAIR_SHARED_FIELDS}
        self.assertEqual(gate_module.pair_mismatches(base, dict(base)), [])
        changed = dict(base)
        changed["identity_digest"] = 2
        self.assertEqual(gate_module.pair_mismatches(base, changed), ["identity_digest"])

    def test_repeat_mismatch_detection(self):
        base = {field: 1 for field in gate_module.REPEAT_FIELDS}
        self.assertEqual(gate_module.repeat_mismatches(base, dict(base)), [])
        changed = dict(base)
        changed["loss_trace"] = 2
        self.assertIn("loss_trace", gate_module.repeat_mismatches(base, changed))

    def test_embedding_and_envelope_checks(self):
        self.assertEqual(
            gate_module.embedding_digest_mismatches({"a": "1"}, {"a": "1"}), 0
        )
        self.assertEqual(
            gate_module.embedding_digest_mismatches({"a": "1"}, {"a": "2"}), 1
        )
        self.assertEqual(gate_module.s_envelope_violations([], 1, 9), [])
        self.assertEqual(
            len(gate_module.s_envelope_violations(
                [{"cell": "ks_er", "fold": 0, "step_counts": [0]}], 1, 9)), 1
        )

    def test_stage_gate_invalid_without_the_full_matrix_and_repeat(self):
        gate = gate_module.stage1_gate(
            [],
            expected_index={"runs": {}, "epochs": {}},
            expected_identity_digest="x",
            observed_identity_digest="x",
            expected_embeddings={"a": "1"},
            observed_embeddings={"a": "1"},
            s_envelope=(1, 9),
            repeat_first=None,
            repeat_second=None,
            held_out_reads=0,
            expected_matrix_members={("ks_er", 0, 0, "pair"), ("ks_er", 0, 0, "control")},
        )
        self.assertFalse(gate["stage1_valid"])
        classes = {problem["class"] for problem in gate["problems"]}
        self.assertIn("MISSING_MATRIX_MEMBERS", classes)
        self.assertIn("MISSING_REPEAT", classes)

    def _consistent_gate(self, mutate=None):
        def record(cell, fold, seed, arm):
            return {
                "cell": cell, "fold": fold, "seed": seed, "arm": arm,
                "identity_digest": "i", "run_manifest_digest": "r",
                "epoch_digests": ["e"] * 5, "consumed_n_er": [[0]] * 5,
                "step_counts": [1] * 5, "target_coefficient": 0.5,
                "lr_sequence": [0.0025], "initialization_sha256": "x",
                "rng_sequence_sha256": "y", "er_counts_match": True,
                "checkpoint_readback_ok": True,
            }

        pair = record("ks_er", 0, 0, "pair")
        control = record("ks_er", 0, 0, "control")
        repeat = {"loss_trace": 1, "final_validation_er_accuracy": 2,
                  "fixed_final_checkpoint_sha256": 3}
        if mutate:
            mutate(pair, control)
        return gate_module.stage1_gate(
            [pair, control],
            expected_index={
                "runs": {"ks_er|0|0": "r"},
                "epochs": {"ks_er|0|0|{}".format(epoch): "e" for epoch in range(5)},
            },
            expected_identity_digest="i",
            observed_identity_digest="i",
            expected_embeddings={"a": "1"},
            observed_embeddings={"a": "1"},
            s_envelope=(1, 9),
            repeat_first=repeat,
            repeat_second=dict(repeat),
            held_out_reads=0,
            expected_matrix_members={("ks_er", 0, 0, "pair"), ("ks_er", 0, 0, "control")},
        )

    def test_consistent_records_pass_and_any_drift_fails(self):
        gate = self._consistent_gate()
        self.assertTrue(gate["stage1_valid"], gate)
        self.assertTrue(all(value == 0 for value in gate["counts"].values()))

        drift = self._consistent_gate(lambda pair, control: control.__setitem__("identity_digest", "j"))
        self.assertFalse(drift["stage1_valid"])
        self.assertEqual(drift["counts"]["identity_manifest_mismatches"], 1)
        self.assertEqual(drift["counts"]["pair_equality_mismatch_members"], 1)


class FrozenAnalysisTests(unittest.TestCase):
    def test_interval_of_a_constant_is_degenerate(self):
        summary = analysis_module.summarize([0.01] * 5)
        self.assertEqual(summary["sd"], 0.0)
        self.assertEqual(summary["half_width"], 0.0)

    def test_half_width_uses_sample_sd_and_four_degrees_of_freedom(self):
        summary = analysis_module.summarize([1.0, 2.0, 3.0, 4.0, 5.0])
        expected_sd = (2.5) ** 0.5
        self.assertAlmostEqual(summary["sd"], expected_sd, places=12)
        self.assertAlmostEqual(
            summary["half_width"],
            analysis_module.t_critical_975_df4() * expected_sd / (5 ** 0.5),
            places=12,
        )

    def _interval(self, mean, half):
        return {"mean": mean, "half_width": half, "low": mean - half, "high": mean + half}

    def test_classification_regions_and_strict_boundaries(self):
        self.assertEqual(
            analysis_module.classify(
                self._interval(-0.03, 0.01), self._interval(0.0, 0.005), self._interval(-0.03, 0.01)
            ),
            "H1",
        )
        self.assertEqual(
            analysis_module.classify(
                self._interval(-0.03, 0.01), self._interval(-0.03, 0.01), self._interval(0.0, 0.005)
            ),
            "H2",
        )
        self.assertEqual(
            analysis_module.classify(
                self._interval(0.0, 0.005), self._interval(0.0, 0.005), self._interval(0.0, 0.005)
            ),
            "H3",
        )
        # Boundary equality is not success: high == -delta is not "wholly below".
        self.assertEqual(
            analysis_module.classify(
                self._interval(-0.02, 0.01), self._interval(0.0, 0.005), self._interval(-0.03, 0.01)
            ),
            "INCONCLUSIVE",
        )

    def test_analyse_returns_the_classification(self):
        result = analysis_module.analyse(
            [-0.03] * 5, [0.0] * 5, nested={"note": "description only"}
        )
        self.assertEqual(result["classification"], "H1")
        self.assertEqual(result["delta"], analysis_module.DELTA_DG)
        self.assertEqual(result["nested"], {"note": "description only"})


if __name__ == "__main__":
    unittest.main()
