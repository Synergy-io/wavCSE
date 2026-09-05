"""Facade-parity test suite (Next Step 5 / Eng Review decision D1).

For each rewritten symbol, constructs BOTH the live (post-rewrite,
mtlkit-backed) downstream/ module and the pre-refactor reference frozen at
commit a4b9823's ancestor a6b9823 (see facade_parity_utils.py), and asserts
identical behavior: same inputs -> same outputs, same attribute names, same
exceptions on bad input -- per the design doc's Success Criteria.

Documented, deliberate exception: DownstreamMultiTaskModel.get_pooling_weights
diverges (fixes a bug nobody could have depended on -- see
downstream/model/downstream_model.py's module docstring). Covered by its own
test below asserting the FIX, not byte-for-byte parity with the crash.
"""

import os
import sys
import unittest

import torch

from mtlkit.tests.facade_parity_utils import reference_downstream

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DOWNSTREAM_DIR = os.path.join(REPO_ROOT, "downstream")
for path in (DOWNSTREAM_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)


class PoolingFacadeParityTests(unittest.TestCase):
    def test_live_pooling_matches_reference_for_every_strategy(self):
        import pooling.pooling as live_pooling_module

        cases = [
            ("mean", None),
            ("max", None),
            ("mix", 0.3),
            ("lnp", 2),
            ("smp", 1.0),
            ("lse", 1.0),
            ("weighted", 5),
            ("gated", 5),
            ("auto", 0.01),
            ("sap", 7),
        ]
        with reference_downstream() as _:
            import pooling.pooling as ref_pooling_module

            for pooling_type, pooling_param in cases:
                torch.manual_seed(42)
                x = torch.randn(3, 5, 7)

                torch.manual_seed(1)
                live_p = live_pooling_module.Pooling(pooling_type, pooling_param=pooling_param)
                torch.manual_seed(1)
                ref_p = ref_pooling_module.Pooling(pooling_type, pooling_param=pooling_param)

                out_live = live_p.get_vector_after_pooling(x, dim=1)
                out_ref = ref_p.get_vector_after_pooling(x, dim=1)
                self.assertTrue(
                    torch.allclose(out_live, out_ref),
                    msg=f"mismatch for pooling_type={pooling_type!r}",
                )

    def test_unsupported_pooling_type_same_exception(self):
        import pooling.pooling as live_pooling_module

        with reference_downstream() as _:
            import pooling.pooling as ref_pooling_module

            with self.assertRaises(ValueError) as live_ctx:
                live_pooling_module.Pooling("bogus")
            with self.assertRaises(ValueError) as ref_ctx:
                ref_pooling_module.Pooling("bogus")
            self.assertEqual(str(live_ctx.exception), str(ref_ctx.exception))


class ConstantMappingFacadeParityTests(unittest.TestCase):
    def test_label_mappings_identical(self):
        import utils.constant_mapping as live_mapping

        with reference_downstream() as _:
            import utils.constant_mapping as ref_mapping

            for key in ("speechcommand", "voxceleb", "iemocap", "fluentspeechcommand"):
                live_l2i, live_i2l = live_mapping.LabelKeywordMapping.get_label_mapping(key)
                ref_l2i, ref_i2l = ref_mapping.LabelKeywordMapping.get_label_mapping(key)
                self.assertEqual(live_l2i, ref_l2i, msg=f"label2index mismatch for {key}")
                self.assertEqual(live_i2l, ref_i2l, msg=f"index2label mismatch for {key}")

    def test_task_and_dataset_keyword_mappings_identical(self):
        import utils.constant_mapping as live_mapping

        with reference_downstream() as _:
            import utils.constant_mapping as ref_mapping

            for key in ("ks", "si", "er", "ic"):
                self.assertEqual(
                    live_mapping.TaskKeywordMapping.get_task_name(key),
                    ref_mapping.TaskKeywordMapping.get_task_name(key),
                )
                self.assertEqual(
                    live_mapping.TaskDatasetMapping.get_dataset_key(key),
                    ref_mapping.TaskDatasetMapping.get_dataset_key(key),
                )

    def test_attribute_names_preserved(self):
        import utils.constant_mapping as live_mapping

        # Every attribute the original class exposed is still there, same name.
        for attr in (
            "LABEL2INDEX_SPEECHCOMMANDv1",
            "INDEX2LABEL_SPEECHCOMMANDv1",
            "LABEL2INDEX_VOXCELEB1",
            "LABEL2INDEX_IEMOCAP",
            "LABEL2INDEX_FLUENTSPEECHCOMMAND",
            "speechcommand",
            "voxceleb",
            "iemocap",
            "fluentspeechcommand",
        ):
            self.assertTrue(hasattr(live_mapping.LabelKeywordMapping, attr), msg=f"missing {attr}")


class MaskedLossFacadeParityTests(unittest.TestCase):
    def test_trainer_utils_masked_ce_loss_matches_reference(self):
        import torch.nn as nn

        import trainer.trainer_utils as live_trainer_utils

        with reference_downstream() as _:
            import trainer.trainer_utils as ref_trainer_utils

            torch.manual_seed(0)
            logits = torch.randn(6, 4)
            labels = torch.tensor([0, -1, 2, -1, 1, 3])
            loss_fn = nn.CrossEntropyLoss()

            live_loss, live_n = live_trainer_utils.masked_ce_loss(logits, labels, loss_fn)
            ref_loss, ref_n = ref_trainer_utils.masked_ce_loss(logits, labels, loss_fn)
            self.assertEqual(live_n, ref_n)
            self.assertTrue(torch.allclose(live_loss, ref_loss))

    def test_evaluator_utils_masked_ce_loss_matches_reference(self):
        import torch.nn as nn

        import evaluator.evaluator_utils as live_evaluator_utils

        with reference_downstream() as _:
            import evaluator.evaluator_utils as ref_evaluator_utils

            torch.manual_seed(0)
            logits = torch.randn(6, 4)
            labels = torch.tensor([0, -1, 2, -1, 1, 3])
            loss_fn = nn.CrossEntropyLoss()

            live_loss, live_n = live_evaluator_utils.masked_ce_loss(logits, labels, loss_fn)
            ref_loss, ref_n = ref_evaluator_utils.masked_ce_loss(logits, labels, loss_fn)
            self.assertEqual(live_n, ref_n)
            self.assertTrue(torch.allclose(live_loss, ref_loss))

    def test_masked_accuracy_matches_reference_both_modules(self):
        import trainer.trainer_utils as live_trainer_utils

        with reference_downstream() as _:
            import trainer.trainer_utils as ref_trainer_utils

            labels = torch.tensor([0, -1, 2, -1, 1, 3])
            pred = torch.tensor([0, 1, 2, 3, 1, 2])
            self.assertEqual(
                live_trainer_utils.masked_accuracy(pred, labels),
                ref_trainer_utils.masked_accuracy(pred, labels),
            )


class DownstreamMultiTaskModelFacadeParityTests(unittest.TestCase):
    def test_forward_logits_and_predictions_match_reference(self):
        import model.downstream_model as live_model_module

        with reference_downstream() as _:
            import model.downstream_model as ref_model_module

            torch.manual_seed(7)
            live_model = live_model_module.DownstreamMultiTaskModel(
                upstream_model_type="wavlm_large",
                task_type="ks_si_er_ic",
                embedding_dim_shared1=32,
                embedding_dim_shared2=16,
                layer_pooling_type="mean",
                dropout_prob_shared1=0.0,
                dropout_prob_shared2=0.0,
            )
            torch.manual_seed(7)
            ref_model = ref_model_module.DownstreamMultiTaskModel(
                upstream_model_type="wavlm_large",
                task_type="ks_si_er_ic",
                embedding_dim_shared1=32,
                embedding_dim_shared2=16,
                layer_pooling_type="mean",
                dropout_prob_shared1=0.0,
                dropout_prob_shared2=0.0,
            )

            torch.manual_seed(123)
            input_seq = torch.randn(2, 25, 1024)
            live_out = live_model(input_seq)
            ref_out = ref_model(input_seq)

            for i, (a, b) in enumerate(zip(live_out.logits, ref_out.logits)):
                self.assertTrue(torch.allclose(a, b, atol=1e-6), msg=f"logits mismatch at head {i}")
            for i, (a, b) in enumerate(zip(live_out.prediction, ref_out.prediction)):
                self.assertTrue(torch.equal(a, b), msg=f"prediction mismatch at head {i}")

    def test_invalid_task_type_token_same_exception_message(self):
        import model.downstream_model as live_model_module

        def _build(module):
            return module.DownstreamMultiTaskModel(
                upstream_model_type="wavlm_large",
                task_type="ks_bogus",
                embedding_dim_shared1=32,
                embedding_dim_shared2=16,
                layer_pooling_type="mean",
                dropout_prob_shared1=0.0,
                dropout_prob_shared2=0.0,
            )

        with reference_downstream() as _:
            import model.downstream_model as ref_model_module

            with self.assertRaises(ValueError) as live_ctx:
                _build(live_model_module)
            with self.assertRaises(ValueError) as ref_ctx:
                _build(ref_model_module)
            self.assertEqual(str(live_ctx.exception), str(ref_ctx.exception))

    def test_get_pooling_weights_deliberately_diverges_by_fixing_the_bug(self):
        """Documented deviation: the reference crashes (AttributeError); the
        live wrapper doesn't, and returns correctly-shaped, correctly-summed
        softmax weights instead. No consumer could have depended on the
        crash succeeding."""
        import model.downstream_model as live_model_module

        live_model = live_model_module.DownstreamMultiTaskModel(
            upstream_model_type="wavlm_large",
            task_type="ks_si",
            embedding_dim_shared1=32,
            embedding_dim_shared2=16,
            layer_pooling_type="weighted",
            dropout_prob_shared1=0.0,
            dropout_prob_shared2=0.0,
            layer_pooling_param=25,
        )
        weights = live_model.get_pooling_weights()
        self.assertIsNotNone(weights)
        self.assertEqual(weights.shape, (25,))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=5)

        with reference_downstream() as _:
            import model.downstream_model as ref_model_module

            ref_model = ref_model_module.DownstreamMultiTaskModel(
                upstream_model_type="wavlm_large",
                task_type="ks_si",
                embedding_dim_shared1=32,
                embedding_dim_shared2=16,
                layer_pooling_type="weighted",
                dropout_prob_shared1=0.0,
                dropout_prob_shared2=0.0,
                layer_pooling_param=25,
            )
            with self.assertRaises(AttributeError):
                ref_model.get_pooling_weights()


if __name__ == "__main__":
    unittest.main()
