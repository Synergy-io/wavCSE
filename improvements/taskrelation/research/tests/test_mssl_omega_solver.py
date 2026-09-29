"""Sanity checks for the MSSL (Goncalves et al. 2016) relation mechanism.

These verify the claims that make the arm meaningful, in the convention the
TR-0007 study pre-registers (Option A: faithful published formulation):

  1. the Omega step solves the paper's Eq. (8),
         min_{Omega>0} lambda_0*tr(S Omega) - log|Omega| + (lambda_2/d)*||Omega||_1,
         S = (1/d) W^T W,
     with lambda_2 in the paper's Eq. (3) units -- so the 1/d of Eq. (8) is
     applied by the solver, not by the caller. Checked against closed forms
     (lambda_2 = 0, lambda_0 scaling, the diagonal solution for a large
     penalty), against the problem's own optimality conditions, and for
     symmetry/positive-definiteness;
  2. the coupling term lambda_0*tr(W Omega W^T) (Eq. 3/4a) produces the
     analytic gradient 2*lambda_0*Omega*W on the live head parameters,
     verified through autograd on a real (small) wavCSE-MSSL model.

The two defects the previous version of this file had are corrected here:
the analytic gradient was written with the multiplication in the wrong order
(`summary @ omega`, which does not even multiply: [m, d] @ [m, m]), and the
sparsity assertions demanded bit-exact float32 zeros from the ADMM's primal
iterate, which is not the variable that carries the l1 support (the returned
split variable Z does, and its zeros are exact).
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
    optimality_certificate,
    soft_threshold,
)


def _well_conditioned_covariance(num_tasks, seed):
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(num_tasks, num_tasks, generator=generator)
    covariance = (base @ base.transpose(0, 1)) / num_tasks + 0.5 * torch.eye(num_tasks)
    return covariance


def _summary_scale_covariance(num_tasks, seed):
    """A covariance at the scale the mean-head summary actually produces.

    The summary rows are means of classifier weight rows over d = 2001
    coordinates, so S enters Eq. (8) with diagonal entries ~1e-5 while its
    inverse (the unregularised Omega) has entries ~1e5. This is the regime
    that a unit-scaled solver silently fails to converge in.
    """
    covariance = _well_conditioned_covariance(num_tasks, seed)
    return covariance * 1e-4


class GraphicalLassoSolverTests(unittest.TestCase):
    def test_zero_penalty_recovers_the_inverse_covariance(self):
        covariance = _well_conditioned_covariance(4, seed=0)
        expected = torch.inverse(covariance)

        omega = graphical_lasso_admm(
            sample_covariance=covariance,
            lambda_2=0.0,
            d=len(covariance),
            max_iterations=20000,
            tolerance=1e-12,
        )

        torch.testing.assert_close(omega, expected, rtol=1e-3, atol=1e-4)

    def test_lambda_0_scales_the_data_term(self):
        # With no penalty the minimiser of lambda_0*tr(S Omega) - log|Omega|
        # is (lambda_0*S)^{-1}, so lambda_0 must appear exactly there.
        covariance = _well_conditioned_covariance(3, seed=7)
        omega = graphical_lasso_admm(
            sample_covariance=covariance,
            lambda_2=0.0,
            d=len(covariance),
            lambda_0=2.0,
            max_iterations=20000,
            tolerance=1e-12,
        )
        torch.testing.assert_close(
            omega, torch.inverse(2.0 * covariance), rtol=1e-3, atol=1e-4
        )

    def test_lambda_2_is_in_the_paper_units_of_eq_3(self):
        # Eq. (8) carries lambda_2/d. Solving with d = 1 must therefore equal
        # solving with lambda_2/d at d = 1, and must NOT equal solving with the
        # same numeric lambda_2 at d = 1 (the rescaled convention the draft
        # used). This pins the units of the pre-registered hyperparameter.
        covariance = _well_conditioned_covariance(3, seed=11)
        num_rows = 9
        lambda_2 = 0.9

        paper_units = graphical_lasso_admm(
            sample_covariance=covariance, lambda_2=lambda_2, d=num_rows,
            max_iterations=20000, tolerance=1e-12,
        )
        equivalent = graphical_lasso_admm(
            sample_covariance=covariance, lambda_2=lambda_2 / num_rows, d=1,
            max_iterations=20000, tolerance=1e-12,
        )
        rescaled_convention = graphical_lasso_admm(
            sample_covariance=covariance, lambda_2=lambda_2, d=1,
            max_iterations=20000, tolerance=1e-12,
        )

        torch.testing.assert_close(paper_units, equivalent, rtol=1e-6, atol=1e-8)
        self.assertFalse(
            torch.allclose(paper_units, rescaled_convention, rtol=1e-3, atol=1e-6),
            "lambda_2 must be interpreted in the paper's Eq. (3) units, so the "
            "1/d of Eq. (8) changes the estimator",
        )

    def test_large_penalty_returns_the_diagonal_precision(self):
        # If the penalty exceeds every off-diagonal magnitude of S, the
        # diagonal matrix diag(1/S_ii) satisfies the optimality conditions
        # exactly, and its off-diagonals are exactly zero.
        covariance = _well_conditioned_covariance(4, seed=1)
        off_diagonal_max = float(
            (covariance - torch.diag(torch.diagonal(covariance))).abs().max()
        )
        lambda_2 = 10.0 * off_diagonal_max * len(covariance)

        omega = graphical_lasso_admm(
            sample_covariance=covariance,
            lambda_2=lambda_2,
            d=len(covariance),
            max_iterations=20000,
            tolerance=1e-12,
        )

        off_diagonal = omega - torch.diag(torch.diagonal(omega))
        off_diagonal_mask = 1.0 - torch.eye(len(covariance))
        self.assertEqual(
            int(((off_diagonal == 0).to(torch.float64) * off_diagonal_mask).sum()),
            12,
            "expected a sparse precision",
        )
        self.assertTrue(torch.all(torch.diagonal(omega) > 0))
        torch.testing.assert_close(
            omega, torch.diag(1.0 / torch.diagonal(covariance)), rtol=1e-6, atol=1e-8
        )

    def test_support_shrinks_monotonically_with_the_penalty(self):
        covariance = _well_conditioned_covariance(4, seed=5)
        counts = []
        for lambda_2 in (0.0, 0.05, 0.2, 1.0, 5.0):
            omega = graphical_lasso_admm(
                sample_covariance=covariance, lambda_2=lambda_2, d=len(covariance),
                max_iterations=20000, tolerance=1e-12,
            )
            off_diagonal = omega - torch.diag(torch.diagonal(omega))
            counts.append(int((off_diagonal != 0).sum()))
        self.assertEqual(counts, sorted(counts, reverse=True), f"counts={counts}")
        self.assertGreater(counts[0], counts[-1], f"counts={counts}")

    def test_solution_satisfies_the_optimality_conditions(self):
        # The certificate needs no reference solver: it is the optimality
        # condition of Eq. (8) itself, measured in S's units. It is asserted
        # at both a unit scale and the summary's 1e-4 scale, which is where a
        # solver whose rho is not scale aware silently returns a wrong point.
        for label, covariance in (
            ("unit scale", _well_conditioned_covariance(3, seed=2)),
            ("summary scale", _summary_scale_covariance(3, seed=2)),
        ):
            for lambda_2 in (0.0, 0.01, 0.1, 10.0):
                with self.subTest(case=label, lambda_2=lambda_2):
                    omega = graphical_lasso_admm(
                        sample_covariance=covariance,
                        lambda_2=lambda_2,
                        d=len(covariance),
                        max_iterations=20000,
                        tolerance=1e-12,
                    )
                    dual_violation, relative_gap = optimality_certificate(
                        omega, covariance, lambda_2, len(covariance)
                    )
                    scale = max(
                        float(covariance.abs().max()), lambda_2 / len(covariance)
                    )
                    self.assertLessEqual(
                        dual_violation, 1e-6 * scale,
                        f"{label}: dual violation {dual_violation} exceeds 1e-6*{scale}",
                    )
                    self.assertLessEqual(
                        abs(relative_gap), 1e-6,
                        f"{label}: relative duality gap {relative_gap}",
                    )

    def test_output_is_symmetric_positive_definite_across_penalties(self):
        covariance = _well_conditioned_covariance(3, seed=2)

        for lambda_2 in (0.0, 0.01, 0.1, 1.0):
            omega = graphical_lasso_admm(
                sample_covariance=covariance,
                lambda_2=lambda_2,
                d=len(covariance),
                max_iterations=20000,
                tolerance=1e-12,
            )
            torch.testing.assert_close(omega, omega.transpose(0, 1), rtol=0, atol=1e-6)
            eigenvalues = torch.linalg.eigvalsh(omega)
            self.assertGreater(
                float(eigenvalues.min()), 0.0,
                f"Omega not positive definite at lambda_2={lambda_2}",
            )

    def test_rejects_invalid_inputs(self):
        covariance = _well_conditioned_covariance(3, seed=2)
        with self.assertRaises(ValueError):
            graphical_lasso_admm(covariance, lambda_2=-1.0, d=3)
        with self.assertRaises(ValueError):
            graphical_lasso_admm(covariance, lambda_2=0.1, d=0)
        with self.assertRaises(ValueError):
            graphical_lasso_admm(covariance, lambda_2=0.1, d=3, lambda_0=0.0)
        with self.assertRaises(ValueError):
            graphical_lasso_admm(torch.ones(2, 3), lambda_2=0.1, d=3)

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
            mssl_admm_iterations=2000,
            layer_pooling_param=0.5,
        )

    def test_lambda_2_has_no_default(self):
        # The published paper selects lambda_2 on data, so an unconfigured run
        # must fail loudly rather than inherit a placeholder.
        torch.manual_seed(3)
        with self.assertRaises(TypeError):
            DownstreamMultiTaskModelMSSL(
                upstream_model_type="wavlm_large",
                task_type="ks_si_er",
                embedding_dim_shared1=16,
                embedding_dim_shared2=8,
                layer_pooling_type="smp",
                dropout_prob_shared1=0.0,
                dropout_prob_shared2=0.0,
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
        eigenvalues = torch.linalg.eigvalsh(omega.double())
        self.assertGreater(float(eigenvalues.min()), 0.0)

    def test_omega_step_uses_the_summary_length_as_d(self):
        # S = (1/d) W W^T and Eq. (8)'s lambda_2/d must both use the summary
        # length; rebuilding the solve by hand with that d must reproduce it.
        model = self._tiny_model(lambda_2=0.05)
        summary = model.get_task_parameter_matrix().detach()
        d = int(summary.shape[1])
        expected = graphical_lasso_admm(
            sample_covariance=(summary @ summary.transpose(0, 1)) / float(d),
            lambda_2=0.05,
            d=d,
            lambda_0=model.mssl_lambda_0,
            rho=model.mssl_admm_rho,
            max_iterations=model.mssl_admm_iterations,
        )
        torch.testing.assert_close(model.update_omega(), expected, rtol=1e-6, atol=1e-10)

    def test_coupling_gradient_matches_analytic_derivative(self):
        lambda_0 = 1.0
        model = self._tiny_model(lambda_0=lambda_0)
        model.update_omega()

        # Analytic: d/dW [lambda_0 tr(W^T Omega W)] = 2 lambda_0 Omega W, with
        # W of shape [num_tasks, features] and Omega [num_tasks, num_tasks].
        with torch.no_grad():
            summary = model.get_task_parameter_matrix()
        expected_grad_w = 2.0 * lambda_0 * (model.omega.detach() @ summary)

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
        off_diagonal_mask = 1.0 - torch.eye(model.num_tasks)
        self.assertEqual(
            int(((off_diagonal == 0).to(torch.float64) * off_diagonal_mask).sum()), 6,
            "a very large lambda_2 must zero every off-diagonal",
        )
        self.assertTrue(torch.all(torch.diagonal(omega) > 0))


if __name__ == "__main__":
    unittest.main()
