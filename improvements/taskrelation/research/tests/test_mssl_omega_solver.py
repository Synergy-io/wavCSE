"""Sanity checks for the MSSL (Goncalves et al. 2016) relation mechanism.

These verify the two claims that make the arm meaningful:
  1. the Omega step really solves the graphical-lasso subproblem (Eq. 8) --
     checked against the known closed form at lambda_2 = 0, against exact
     sparsity at large lambda_2, and for symmetry/positive-definiteness;
  2. the coupling term tr(W Omega W^T) (Eq. 3) produces the analytic gradient
     2*lambda_0 * W @ Omega on the live head parameters, verified through
     autograd on a real (small) wavCSE-MSSL model.
"""

import os
import sys
import unittest

import torch

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
DOWNSTREAM_DIR = os.path.join(REPO_ROOT, "downstream")
MSSL_DIR = os.path.join(
    REPO_ROOT, "improvements", "taskrelation", "04-mssl"
)
for path in (DOWNSTREAM_DIR, REPO_ROOT, MSSL_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from mssl_model import (  # noqa: E402
    DownstreamMultiTaskModelMSSL,
    graphical_lasso_admm,
    soft_threshold,
)


def _well_conditioned_covariance(num_tasks, seed):
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(num_tasks, num_tasks, generator=generator)
    covariance = (base @ base.transpose(0, 1)) / num_tasks + 0.5 * torch.eye(num_tasks)
    return covariance


class GraphicalLassoSolverTests(unittest.TestCase):
    def test_zero_penalty_recovers_the_inverse_covariance(self):
        covariance = _well_conditioned_covariance(4, seed=0)
        expected = torch.inverse(covariance)

        omega = graphical_lasso_admm(
            sample_covariance=covariance,
            lambda_2=0.0,
            rho=1.0,
            max_iterations=2000,
            tolerance=1e-12,
        )

        torch.testing.assert_close(omega, expected, rtol=1e-3, atol=1e-4)

    def test_large_penalty_drives_off_diagonals_to_exactly_zero(self):
        covariance = _well_conditioned_covariance(4, seed=1)

        omega = graphical_lasso_admm(
            sample_covariance=covariance,
            lambda_2=5.0,
            rho=1.0,
            max_iterations=2000,
            tolerance=1e-12,
        )

        off_diagonal = omega - torch.diag(torch.diagonal(omega))
        self.assertEqual(int((off_diagonal == 0).sum()), 12, "expected a sparse precision")
        self.assertTrue(torch.all(torch.diagonal(omega) > 0))

    def test_output_is_symmetric_positive_definite_across_penalties(self):
        covariance = _well_conditioned_covariance(3, seed=2)

        for lambda_2 in (0.0, 0.01, 0.1, 1.0):
            omega = graphical_lasso_admm(
                sample_covariance=covariance,
                lambda_2=lambda_2,
                rho=1.0,
                max_iterations=1000,
                tolerance=1e-12,
            )
            torch.testing.assert_close(omega, omega.transpose(0, 1), rtol=0, atol=1e-6)
            eigenvalues = torch.linalg.eigvalsh(omega)
            self.assertGreater(
                float(eigenvalues.min()), 0.0,
                f"Omega not positive definite at lambda_2={lambda_2}",
            )

    def test_soft_threshold_matches_definition(self):
        values = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
        expected = torch.tensor([-1.5, 0.0, 0.0, 0.0, 1.5])
        torch.testing.assert_close(soft_threshold(values, 0.5), expected)


class RelationCouplingTests(unittest.TestCase):
    def _tiny_model(self, lambda_2=0.05, lambda_0=1.0):
        torch.manual_seed(3)
        return DownstreamMultiTaskModelMSSL(
            upstream_model_type="wavlm_large",
            task_type="ks_si_er",
            embedding_dim_shared1=16,
            embedding_dim_shared2=8,
            layer_pooling_type="smp",
            dropout_prob_shared1=0.0,
            dropout_prob_shared2=0.0,
            mssl_lambda_2=lambda_2,
            mssl_lambda_0=lambda_0,
            mssl_admm_iterations=200,
            layer_pooling_param=0.5,
        )

    def test_summary_matrix_and_precision_shapes_survive_heterogeneous_heads(self):
        model = self._tiny_model()
        # ks/si/er have 12 / 1251 / 4 classes -- no aligned parameter columns.
        self.assertEqual([head.out_features for head in model.classifiers], [12, 1251, 4])

        summary = model.get_task_parameter_matrix()
        self.assertEqual(tuple(summary.shape), (3, 8 + 1))

        omega = model.update_omega()
        self.assertEqual(tuple(omega.shape), (3, 3))
        eigenvalues = torch.linalg.eigvalsh(omega)
        self.assertGreater(float(eigenvalues.min()), 0.0)

    def test_coupling_gradient_matches_analytic_derivative(self):
        lambda_0 = 1.0
        model = self._tiny_model(lambda_0=lambda_0)
        model.update_omega()

        # Analytic: d/dW [lambda_0 tr(W Omega W^T)] = 2 lambda_0 W Omega.
        with torch.no_grad():
            summary = model.get_task_parameter_matrix()
        expected_grad_w = 2.0 * lambda_0 * summary @ model.omega
        expected_grad_w = expected_grad_w.detach()

        model.zero_grad(set_to_none=True)
        loss = model.get_relation_loss()
        loss.backward()

        for task_index, head in enumerate(model.classifiers):
            num_classes = head.out_features
            grad = head.weight.grad
            self.assertIsNotNone(grad)
            # W's row for this task is the mean of the head's weight rows, so
            # the penalty pulls every row identically -- an artifact of the
            # summary adapter, recorded in the Study plan.
            uniform_target = expected_grad_w[task_index, :8] / float(num_classes)
            for row in range(min(num_classes, 3)):
                torch.testing.assert_close(
                    grad[row, :8], uniform_target, rtol=1e-4, atol=1e-6
                )
            torch.testing.assert_close(
                head.bias.grad,
                (expected_grad_w[task_index, 8] / float(num_classes)).expand_as(
                    head.bias.grad
                ),
                rtol=1e-4, atol=1e-6,
            )

    def test_relation_loss_is_finite_before_and_after_the_omega_step(self):
        model = self._tiny_model()
        loss_before = float(model.get_relation_loss().item())
        model.update_omega()
        loss_after = float(model.get_relation_loss().item())
        self.assertTrue(torch.isfinite(torch.tensor(loss_before)))
        self.assertTrue(torch.isfinite(torch.tensor(loss_after)))

    def test_penalty_sparsifies_the_precision(self):
        model = self._tiny_model(lambda_2=1e6)
        omega = model.update_omega()
        off_diagonal = omega - torch.diag(torch.diagonal(omega))
        self.assertEqual(
            int((off_diagonal.abs() < 1e-9).sum()), 6,
            "a very large lambda_2 must zero every off-diagonal",
        )


if __name__ == "__main__":
    unittest.main()
