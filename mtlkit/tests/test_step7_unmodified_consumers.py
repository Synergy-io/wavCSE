"""Step 7 / Success Criteria: PMR, GBC, TSM, NCMTL, and FTN's existing
entrypoints run without modification against the wrapped downstream/ --
verified by importing and instantiating each branch's model against the
rewritten wrapper, confirming no import error, no signature mismatch, and a
successful synthetic-data forward+backward pass. No code changes to any of
these branches (Constraints).

NCMTL already has its own comprehensive suite
(improvements/clustering/tests/test_ncmtl.py, unmodified, run separately as
its own regression proof); this file covers PMR, GBC, TSM, and FTN, which
had none.
"""

import importlib.util
import os
import sys
import unittest

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DOWNSTREAM_DIR = os.path.join(REPO_ROOT, "downstream")
for path in (DOWNSTREAM_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)


def _load_module_from_path(module_name: str, file_path: str):
    """Matches improvements/run_improvements.py's own loader -- GBC and
    MTRL live under numbered directories (03-gbc, 01-mtrl) that aren't
    valid Python package names."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _common_args(task_type="ks_si_er"):
    return dict(
        upstream_model_type="wavlm_large",
        task_type=task_type,
        embedding_dim_shared1=32,
        embedding_dim_shared2=16,
        layer_pooling_type="mean",
        dropout_prob_shared1=0.0,
        dropout_prob_shared2=0.0,
    )


def _smoke_forward_backward(model, batch_size=3, seq_len=25, input_dim=1024):
    torch.manual_seed(0)
    input_seq = torch.randn(batch_size, seq_len, input_dim)
    outputs = model(input_seq)
    loss = torch.stack([logits.float().sum() for logits in outputs.logits]).sum()
    loss.backward()
    return outputs


class PMRUnmodifiedConsumerTests(unittest.TestCase):
    def test_import_instantiate_forward_backward(self):
        from improvements.taskrelation.models.pmr_model import DownstreamMultiTaskModelPMR

        model = DownstreamMultiTaskModelPMR(**_common_args())
        outputs = _smoke_forward_backward(model)
        self.assertEqual(len(outputs.logits), 3)
        self.assertIsNotNone(model.projector_layer.weight.grad)
        # omega_chol isn't wired into this smoke loss (only the classifier
        # heads are summed) -- confirms accessing it doesn't error, not that
        # it receives a gradient.
        self.assertIsInstance(model.omega_chol, torch.nn.Parameter)


class GBCUnmodifiedConsumerTests(unittest.TestCase):
    def test_import_instantiate_forward_backward(self):
        gbc_dir = os.path.join(REPO_ROOT, "improvements", "taskrelation", "03-gbc")
        gbc_module = _load_module_from_path("gbc_model_step7", os.path.join(gbc_dir, "gbc_model.py"))

        model = gbc_module.DownstreamMultiTaskModelGBC(**_common_args())
        outputs = _smoke_forward_backward(model)
        self.assertEqual(len(outputs.logits), 3)
        self.assertIsNotNone(model.projector_layer.weight.grad)
        self.assertIsNotNone(model.global_bias.grad)


class TSMUnmodifiedConsumerTests(unittest.TestCase):
    def test_import_instantiate_forward_backward(self):
        from improvements.taskrelation.models.tsm_model import DownstreamMultiTaskModelTSM

        model = DownstreamMultiTaskModelTSM(**_common_args())
        outputs = _smoke_forward_backward(model)
        self.assertEqual(len(outputs.logits), 3)
        self.assertIsNotNone(model.projector_layer.weight.grad)
        self.assertIsNotNone(model.structure_matrix_A.grad)


class MTRLUnmodifiedConsumerTests(unittest.TestCase):
    def test_import_instantiate_forward_backward(self):
        mtrl_dir = os.path.join(REPO_ROOT, "improvements", "taskrelation", "01-mtrl")
        mtrl_module = _load_module_from_path("mtrl_model_step7", os.path.join(mtrl_dir, "mtrl_model.py"))

        model = mtrl_module.DownstreamMultiTaskModelMTRL(**_common_args())
        outputs = _smoke_forward_backward(model)
        self.assertEqual(len(outputs.logits), 3)
        self.assertIsNotNone(model.projector_layer.weight.grad)

        # MTRL-specific: analytic Omega update and regularizer loss both work
        model.update_omega()
        reg_loss = model.get_mtrl_regularizer_loss()
        self.assertEqual(reg_loss.dim(), 0)


class FTNUnmodifiedConsumerTests(unittest.TestCase):
    def test_import_instantiate_forward_backward(self):
        from improvements.decomposition.models.ftn_model import DownstreamMultiTaskModelFTN

        model = DownstreamMultiTaskModelFTN(**_common_args(task_type="ks_si_er"))
        outputs = _smoke_forward_backward(model)
        self.assertEqual(len(outputs.logits), 3)
        # FTN genuinely subclasses DownstreamMultiTaskModel -- this exercises
        # the flat-attribute-layout fix directly.
        self.assertIsNotNone(model.projector_layer.weight.grad)
        self.assertIsNotNone(model.shared_adapter.weight.grad)

    def test_wrong_task_type_rejected_same_as_before(self):
        from improvements.decomposition.models.ftn_model import DownstreamMultiTaskModelFTN

        with self.assertRaises(ValueError):
            DownstreamMultiTaskModelFTN(**_common_args(task_type="ks_si"))


if __name__ == "__main__":
    unittest.main()
