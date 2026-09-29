"""Faithfulness of the classical MTRL arm to the published mathematics.

Audit: `improvements/taskrelation/research/audits/2026-09-29-mtrl-theory-to-implementation-audit.md`
(with its correction log). Corrected after the independent review, commit
`5f72acd`: the historical arm's mathematics is now asserted directly, and no
mathematical assertion sits behind a configuration assertion.

Zhang & Yeung's MTRL (*A Regularization Approach to Learning Task
Relationships in Multi-Task Learning*, ACM TKDD 8(3):12, 2014 -- journal
version of UAI 2010) defines the task parameter matrix `W = (w_1,...,w_m)`
(their Eq. (8): `W` is `d x m`, columns are the task parameter vectors), the
regularizer `(lambda_2/2) tr(W Omega^-1 W^T)`, the constraint
`Omega >= 0, tr(Omega) <= 1`, the initialization `Omega_0 = I_m/m`, and the
closed-form relation subproblem solution

    Omega = (W^T W)^(1/2) / tr((W^T W)^(1/2))                     (Eq. (14))

with `tr(Omega^-1 W^T W)` attaining its minimum `(tr (W^T W)^(1/2))^2`.

`mtrl_model.py` builds `W` as the project's declared adapter (per-task
mean-pooled head summary, `[m, hidden+1]`), a documented deviation acknowledged
in `literature/zhang-yeung-2014-mtrl-asymmetric.md` (Gate 3). That adapter is
NOT under test here. What is under test:

* the published equations, as implemented (section A);
* that the historical in-category control's row normalization (`normalize_w:
  true`, in every committed MTRL config) moves the executed mathematics off the
  published ones (section B, asserted first, on the mathematics);
* that removing that normalization restores them (section C);
* and that the two are separated by the same criterion, so the distinction is
  a demonstrated mathematical fact rather than a config-diff claim (section D).

Nothing in section B depends on a YAML assertion: `assert_published_W_invariants`
returns measured values, and the configuration's own `normalize_w` value is
asserted separately in section E.
"""

import copy
import os
import sys
import unittest

import torch
import yaml

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
MTRL_DIR = os.path.join(REPO_ROOT, "improvements", "taskrelation", "01-mtrl")
for path in (REPO_ROOT, os.path.join(REPO_ROOT, "downstream")):
    if path not in sys.path:
        sys.path.insert(0, path)

from improvements.run_improvements import build_model  # noqa: E402
from utils.parse_transformer_layers import parse_transformer_layers  # noqa: E402

HISTORICAL_CONFIG = os.path.join(MTRL_DIR, "mtrl_poolingwinner_25L_config.yml")
CORRECTED_CONFIG = os.path.join(MTRL_DIR, "mtrl_norm_corrected_25L_config.yml")

HISTORICAL_CONFIGS = [
    "mtrl_config.yml",
    "mtrl_alllayers_config.yml",
    "mtrl_poolingwinner_16L_config.yml",
    "mtrl_poolingwinner_25L_config.yml",
    "mtrl_kfold_config.yml",
]

# Measured on the protocol configs in float64, seeds 0-11 (audit and its
# independent review): the historical configuration's Omega sits +48.26 % to
# +52.07 % above the published subproblem minimum for the matrix the model
# trains, and its penalty scale ratio is 1.0031-1.0032 (omega_epsilon-limited;
# it tends to 1 as omega_epsilon -> 0). The corrected configuration's gap is
# within ~3e-6 and its ratio is exactly 9. Thresholds below are far outside
# float noise and far inside the measured separation.
GAP_THRESHOLD = 0.40
CORRECTED_GAP_TOLERANCE = 1e-3
SCALE_RATIO_CEILING = 1.01
SEEDS = (0, 1, 2)

TOL = 1e-5


def load_config(path):
    with open(path) as handle:
        return yaml.safe_load(handle)


def build_from_config(cfg, seed=0, normalize_w=None, omega_epsilon=None):
    """Build the MTRL model exactly as `run_improvements.build_model` does."""
    cfg = copy.deepcopy(cfg)
    if normalize_w is not None:
        cfg["model"]["normalize_w"] = normalize_w
    if omega_epsilon is not None:
        cfg["model"]["omega_epsilon"] = omega_epsilon
    torch.manual_seed(seed)
    layers = parse_transformer_layers(
        cfg["upstream"]["selected_transformer_layers"], cfg["upstream"]["model_type"]
    )
    pooling_param = cfg["pooling"].get("layer_pooling_param")
    if cfg["pooling"]["layer_pooling_type"] == "weighted" and pooling_param is None:
        pooling_param = len(layers)
    return build_model("mtrl", cfg, "ks_si_er", pooling_param)


def raw_task_parameter_matrix(model):
    """The adapter's W with the row normalization forced off."""
    previous = model.normalize_w
    model.normalize_w = False
    try:
        return model.get_task_parameter_matrix().detach()
    finally:
        model.normalize_w = previous


def gram(matrix):
    return matrix @ matrix.T


def sqrt_psd(matrix):
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    return (
        eigenvectors
        @ torch.diag(eigenvalues.clamp(min=0.0).sqrt())
        @ eigenvectors.T
    )


def published_minimum(matrix):
    """(tr(W^T W)^(1/2))^2, TKDD Eq. (14)'s minimum over tr(Omega) = 1."""
    return torch.trace(sqrt_psd(gram(matrix)).double()) ** 2


def relation_value(omega, matrix):
    """tr(Omega^-1 W^T W) in float64, evaluated on the paper's orientation."""
    return torch.trace(
        torch.linalg.solve(omega.double(), gram(matrix).double())
    )


def scale_ratio(model, scale=3.0, omega_epsilon=None):
    """penalty(scale*W)/penalty(W), with Omega re-derived as the scheme would.

    Computed through the model's own W construction, so the ratio is `scale**2`
    exactly when the regularized object is the task parameter matrix, and ~1
    when a row normalization makes it scale-free. `omega_epsilon` may be
    overridden to expose the epsilon floor.
    """
    if omega_epsilon is not None:
        model.omega_epsilon = omega_epsilon
    model.update_omega()
    base = model.get_mtrl_regularizer_loss().item()
    with torch.no_grad():
        for head in model.classifiers:
            head.weight.mul_(scale)
            head.bias.mul_(scale)
    model.update_omega()  # re-derive Omega for the scaled parameters
    return model.get_mtrl_regularizer_loss().item() / base


def assert_published_W_invariants(test, model):
    """The gate named in the audit: does the executed Omega solve Eq. (14) for W?

    Returns the measured invariants; asserts nothing about `normalize_w` or any
    other configuration value, so the caller always exercises the mathematics.
    """
    raw = raw_task_parameter_matrix(model)
    minimum = published_minimum(raw)
    attained = relation_value(model.omega, raw)
    gap = (attained - minimum) / minimum

    test.assertAlmostEqual(
        torch.trace(model.omega).item(), 1.0, places=6,
        msg="TKDD Eq. (8): the feasible Omega has tr(Omega) = 1",
    )
    closed_form = sqrt_psd(gram(raw)) / torch.trace(sqrt_psd(gram(raw)))
    closed_form_residual = (model.omega.double() - closed_form.double()).abs().max()
    return {"gap": gap.item(), "closed_form_residual": closed_form_residual.item()}


class A_PublishedOmegaSolution(unittest.TestCase):
    """The paper's own equations, as implemented, for the W each policy uses."""

    def test_omega_equals_the_published_closed_form(self):
        """TKDD Eq. (14): Omega = (W^T W)^(1/2)/tr((W^T W)^(1/2))."""
        for normalize_w in (False, True):
            with self.subTest(normalize_w=normalize_w):
                model = build_from_config(
                    load_config(CORRECTED_CONFIG), normalize_w=normalize_w
                )
                model.update_omega()
                W = model.get_task_parameter_matrix().detach()
                root = sqrt_psd(gram(W))
                self.assertTrue(
                    torch.allclose(model.omega, root / torch.trace(root),
                                   atol=TOL, rtol=0)
                )

    def test_omega_is_symmetric_psd_and_satisfies_the_trace_constraint(self):
        """TKDD Eq. (8): Omega >= 0 and tr(Omega) <= 1 (attained at equality)."""
        model = build_from_config(load_config(CORRECTED_CONFIG), normalize_w=False)
        model.update_omega()
        omega = model.omega
        self.assertTrue(torch.allclose(omega, omega.T, atol=TOL))
        self.assertGreaterEqual(torch.linalg.eigvalsh(omega).min().item(), -TOL)
        self.assertAlmostEqual(torch.trace(omega).item(), 1.0, places=6)

    def test_initial_omega_is_identity_over_m_before_any_update(self):
        """TKDD section 2.2: "We set the initial value of Omega to (1/m) I_m"."""
        model = build_from_config(load_config(CORRECTED_CONFIG), normalize_w=False)
        self.assertAlmostEqual(torch.trace(model.omega).item(), 1.0, places=6)
        self.assertTrue(
            torch.allclose(
                model.omega, torch.eye(model.num_tasks) / model.num_tasks, atol=TOL
            )
        )

    def test_omega_attains_the_published_minimum_of_the_relation_subproblem(self):
        """Substituting Omega* gives (tr(W^T W)^(1/2))^2 -- the trace-norm value."""
        for normalize_w in (False, True):
            with self.subTest(normalize_w=normalize_w):
                model = build_from_config(
                    load_config(CORRECTED_CONFIG), normalize_w=normalize_w
                )
                model.update_omega()
                W = model.get_task_parameter_matrix().detach()
                minimum = published_minimum(W)
                self.assertLess(
                    abs(relation_value(model.omega, W) - minimum) / minimum,
                    1e-4,
                )


class B_HistoricalNormalizedMathematics(unittest.TestCase):
    """The D2 finding: the historical arm's mathematics is off the published ones.

    Every test here builds the model IN MEMORY with `normalize_w=True` (or from
    the historical config, which sets it) and asserts the mathematics directly.
    No configuration assertion precedes these.
    """

    def test_historical_omega_is_not_the_minimiser_for_the_trained_parameters(self):
        """Omega must solve Eq. (14) for the matrix the regularizer differentiates.

        Under row normalization it solves it for a rescaled copy, so the logged
        Omega sits ~50 % above the published subproblem minimum for the raw W.
        """
        for seed in SEEDS:
            with self.subTest(seed=seed):
                model = build_from_config(
                    load_config(HISTORICAL_CONFIG), seed=seed
                )  # as written: normalize_w: true
                self.assertTrue(model.normalize_w)
                model.update_omega()
                measured = assert_published_W_invariants(self, model)
                self.assertGreater(
                    measured["gap"],
                    GAP_THRESHOLD,
                    "the historical configuration's Omega must sit well above the "
                    "published minimum for the trained parameter matrix",
                )
                self.assertGreater(
                    measured["closed_form_residual"],
                    1e-4,
                    "Omega must not coincide with the published closed form for W",
                )

    def test_historical_penalty_is_scale_invariant_not_quadratic(self):
        """The published term is degree-two homogeneous in W; the normalized one is not."""
        for seed in SEEDS:
            with self.subTest(seed=seed):
                model = build_from_config(load_config(HISTORICAL_CONFIG), seed=seed)
                self.assertTrue(model.normalize_w)
                ratio = scale_ratio(model)
                self.assertLess(ratio, SCALE_RATIO_CEILING)
                self.assertGreater(ratio, 0.99)

    def test_historical_scale_ratio_is_epsilon_limited_and_tends_to_one(self):
        """1.0031 is the omega_epsilon floor, not the modification.

        With omega_epsilon -> 0 the normalized ratio is exactly 1: the penalty
        carries no information about the parameter scale at all.
        """
        def ratio_at(epsilon):
            model = build_from_config(
                load_config(HISTORICAL_CONFIG), normalize_w=True
            )
            model.omega_epsilon = epsilon
            model.update_omega()
            base = model.get_mtrl_regularizer_loss().item()
            with torch.no_grad():
                for head in model.classifiers:
                    head.weight.mul_(3.0)
                    head.bias.mul_(3.0)
            model.update_omega()
            return model.get_mtrl_regularizer_loss().item() / base

        coarse = ratio_at(1e-4)
        fine = ratio_at(1e-9)
        self.assertLess(abs(fine - 1.0), abs(coarse - 1.0))
        self.assertLess(abs(fine - 1.0), 1e-3)

    def test_normalized_penalty_is_confined_to_lambda_times_m_to_m_squared(self):
        """With unit rows the attained value is in [m, m^2]: it cannot grow with scale.

        Endpoints exactly: A = I gives m^2, A = s s^T (perfect +/- collinearity,
        the modified objective's own attractor) gives m.
        """
        m = 3
        self.assertAlmostEqual(
            published_minimum(torch.eye(m)).item(), float(m * m), places=6
        )
        # rows equal to s = (+1, +1, +1) give Gram = s s^T, the collinear endpoint
        self.assertAlmostEqual(
            published_minimum(torch.ones(m, 1)).item(), float(m), places=5
        )
        generator = torch.Generator().manual_seed(0)
        for _ in range(200):
            rows = torch.randn(m, 20, generator=generator, dtype=torch.float64)
            rows = rows / rows.norm(dim=1, keepdim=True)
            value = published_minimum(rows).item()
            self.assertGreaterEqual(value, m - 1e-6)
            self.assertLessEqual(value, m * m + 1e-6)

    def test_task_scale_information_is_lost_but_direction_information_is_not(self):
        """The precise consequence: the diagonal tracks directions, never scale.

        `diag(Omega) ~= 1/m` is NOT a general invariant of the normalized arm
        (the 25L smp Phase-A run reports [0.3027, 0.3076, 0.3897]); what the
        normalization removes is the tasks' parameter *magnitude*. Scaling one
        task's parameters changes the published Omega and cannot meaningfully
        change the normalized one. Measured with a 40x rescale of task 1:
        ||dOmega||/||Omega|| is 2.80e-3 under normalization -- the residual
        omega_epsilon floor, which vanishes with epsilon -- against 9.74e-1
        without it, where the diagonal reorganises from [0.351, 0.035, 0.614]
        to [0.149, 0.591, 0.260].
        """
        def movement(normalize_w, config):
            model = build_from_config(load_config(config), normalize_w=normalize_w)
            model.update_omega()
            before = model.omega.clone()
            with torch.no_grad():
                model.classifiers[1].weight.mul_(40.0)
                model.classifiers[1].bias.mul_(40.0)
            model.update_omega()
            return (model.omega - before).norm().item() / before.norm().item()

        normalized = movement(True, HISTORICAL_CONFIG)
        corrected = movement(False, CORRECTED_CONFIG)
        self.assertLess(
            normalized,
            1e-2,
            "under row normalization a per-task rescaling moves Omega only by "
            "the epsilon floor",
        )
        self.assertGreater(
            corrected,
            0.5,
            "without normalization the published Omega must respond to task scale",
        )


class C_NormalizationCorrectedMathematics(unittest.TestCase):
    """Removing the normalization restores the published mathematics."""

    def test_corrected_config_satisfies_the_published_W_invariants(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                model = build_from_config(load_config(CORRECTED_CONFIG), seed=seed)
                self.assertFalse(model.normalize_w)
                model.update_omega()
                measured = assert_published_W_invariants(self, model)
                self.assertLess(abs(measured["gap"]), CORRECTED_GAP_TOLERANCE)
                self.assertLess(measured["closed_form_residual"], 1e-5)

    def test_corrected_penalty_is_quadratic_in_the_parameter_scale(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                model = build_from_config(load_config(CORRECTED_CONFIG), seed=seed)
                self.assertFalse(model.normalize_w)
                self.assertAlmostEqual(scale_ratio(model), 9.0, places=3)

    def test_corrected_penalty_is_unbounded_in_the_parameter_scale(self):
        """The published penalty has no scale ceiling: it grows as ||W||^2."""
        model = build_from_config(load_config(CORRECTED_CONFIG), normalize_w=False)
        model.update_omega()
        omega = model.omega.detach().double()
        W = raw_task_parameter_matrix(model).double()
        self.assertAlmostEqual(
            relation_value(omega, W * 10.0).item(),
            100.0 * relation_value(omega, W).item(),
            places=6,
        )


class D_Discriminator(unittest.TestCase):
    """One criterion, opposite verdicts: the distinction is mathematics, not config."""

    def _verdict(self, normalize_w):
        """True when the executed Omega solves Eq. (14) for the trained W."""
        model = build_from_config(
            load_config(HISTORICAL_CONFIG), normalize_w=normalize_w, seed=0
        )
        model.update_omega()
        raw = raw_task_parameter_matrix(model)
        minimum = published_minimum(raw)
        attained = relation_value(model.omega, raw)
        return bool(abs((attained - minimum) / minimum) < CORRECTED_GAP_TOLERANCE)

    def test_same_criterion_passes_without_normalization_and_fails_with_it(self):
        """The historical-specific failure disappears when only the flag changes.

        Same code, same config, same seed, same criterion -- only the row
        normalization differs, and the verdict flips. This is the mutation the
        audit's D2 claim rests on.
        """
        self.assertFalse(self._verdict(True), "historical mathematics must fail")
        self.assertTrue(self._verdict(False), "corrected mathematics must pass")

    def test_the_criterion_really_discriminates(self):
        """Guard: the criterion is not trivially true of every Omega."""
        model = build_from_config(load_config(CORRECTED_CONFIG), normalize_w=False)
        model.update_omega()
        raw = raw_task_parameter_matrix(model)
        minimum = published_minimum(raw)
        swapped = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ) / 3.0
        self.assertGreater(
            abs((relation_value(swapped, raw) - minimum) / minimum).item(),
            CORRECTED_GAP_TOLERANCE,
        )


class E_PublishedRegularizer(unittest.TestCase):
    """The regularizer the trainer differentiates is lambda*tr(W Omega^-1 W^T)."""

    def test_penalty_equals_lambda_times_the_published_trace_expression(self):
        for normalize_w in (False, True):
            with self.subTest(normalize_w=normalize_w):
                model = build_from_config(
                    load_config(CORRECTED_CONFIG), normalize_w=normalize_w
                )
                model.update_omega()
                W = model.get_task_parameter_matrix().detach()
                W_paper = W.T  # the paper's W is d x m; the code stores [m, d]
                expected = model.mtrl_lambda * torch.trace(
                    W_paper.double()
                    @ model.omega_inv.double()
                    @ W_paper.T.double()
                ).item()
                self.assertAlmostEqual(
                    model.get_mtrl_regularizer_loss().item(),
                    expected,
                    delta=1e-4 * max(abs(expected), 1.0),
                )

    def test_orientation_is_the_paper_transpose(self):
        """W_code is [m, d]; the paper's W is [d, m] = W_code^T.

        tr(W_code Omega^-1 W_code^T) must equal the paper's
        tr(W Omega^-1 W^T) for W = W_code^T.
        """
        model = build_from_config(load_config(CORRECTED_CONFIG), normalize_w=False)
        model.update_omega()
        W = raw_task_parameter_matrix(model).double()
        omega = model.omega.detach().double()
        self.assertAlmostEqual(
            torch.trace(torch.linalg.solve(omega, gram(W))).item(),
            torch.trace(W.T @ torch.linalg.solve(omega, W)).item(),
            places=8,
        )

    def test_gradient_reaches_every_class_row_of_a_head_identically(self):
        """The adapter's consequence (D1): only a head's mean direction is regularized."""
        model = build_from_config(load_config(CORRECTED_CONFIG), normalize_w=False)
        model.update_omega()
        model.zero_grad(set_to_none=True)
        model.get_mtrl_regularizer_loss().backward()
        for index, head in enumerate(model.classifiers):
            grad = head.weight.grad
            self.assertIsNotNone(grad, f"head {index} must receive gradient")
            self.assertLess(
                (grad - grad[0].unsqueeze(0)).abs().max().item(), 1e-6
            )
        self.assertIsNone(
            model.hidden_layer.weight.grad,
            "the regularizer acts on the task heads only, like the paper's W",
        )

    def test_gradient_equals_two_lambda_omega_inverse_w(self):
        model = build_from_config(load_config(CORRECTED_CONFIG), normalize_w=False)
        model.update_omega()
        model.zero_grad(set_to_none=True)
        model.get_mtrl_regularizer_loss().backward()
        W = model.get_task_parameter_matrix().detach()
        dW = 2.0 * model.mtrl_lambda * (model.omega_inv @ W)
        head = model.classifiers[0]
        expected_row = dW[0, : head.weight.shape[1]] / head.weight.shape[0]
        self.assertLess((head.weight.grad[0] - expected_row).abs().max().item(), 1e-5)


class F_ConfigurationRecord(unittest.TestCase):
    """What the committed configurations record - asserted separately, on purpose."""

    def test_every_committed_historical_mtrl_config_enables_normalize_w(self):
        for name in HISTORICAL_CONFIGS:
            with self.subTest(config=name):
                cfg = load_config(os.path.join(MTRL_DIR, name))
                self.assertIs(cfg["model"]["normalize_w"], True)
                self.assertEqual(cfg["model"]["mtrl_lambda"], 0.01)
                self.assertEqual(cfg["model"]["omega_epsilon"], 0.0001)
                self.assertEqual(cfg["mtrl"]["omega_update_frequency"], 1)

    def test_corrected_and_historical_configs_differ_only_in_the_intended_keys(self):
        historical = load_config(HISTORICAL_CONFIG)
        corrected = load_config(CORRECTED_CONFIG)

        def flatten(mapping, prefix=""):
            flat = {}
            for key, value in mapping.items():
                path = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    flat.update(flatten(value, path))
                else:
                    flat[path] = value
            return flat

        fh, fc = flatten(historical), flatten(corrected)
        self.assertEqual(sorted(fh), sorted(fc))
        self.assertEqual(
            {k for k in fh if fh[k] != fc[k]},
            {
                "model.normalize_w",
                "paths.results_root",
                "paths.checkpoints_root",
            },
        )
        self.assertIs(fh["model.normalize_w"], True)
        self.assertIs(fc["model.normalize_w"], False)


class G_AllTwentyFiveLayerPolicy(unittest.TestCase):
    """The shared protocol fixes layer pooling over all 25 WavLM-Large layers."""

    def test_all_resolves_to_the_twenty_five_layer_wavlm_stack(self):
        self.assertEqual(parse_transformer_layers("all", "wavlm_large"), list(range(25)))

    def test_control_and_successor_train_on_all_twenty_five_layers(self):
        for path in (HISTORICAL_CONFIG, CORRECTED_CONFIG):
            with self.subTest(config=os.path.basename(path)):
                cfg = load_config(path)
                self.assertEqual(
                    parse_transformer_layers(
                        cfg["upstream"]["selected_transformer_layers"],
                        cfg["upstream"]["model_type"],
                    ),
                    list(range(25)),
                )
                self.assertEqual(cfg["pooling"]["layer_pooling_type"], "smp")
                self.assertEqual(cfg["pooling"]["layer_pooling_param"], 0.5)

                model = build_from_config(cfg)
                projected = model.projector_layer(torch.randn(2, 25, 1024))
                weights = torch.softmax(0.5 * projected, dim=1)
                self.assertEqual(weights.shape, (2, 25, 512))
                self.assertTrue(
                    torch.allclose(weights.sum(dim=1), torch.ones(2, 512), atol=1e-5),
                    "smp must pool over all 25 layer slots, dropping none",
                )

    def test_weighted_layer_pooling_sizes_its_weights_to_the_layer_array(self):
        """Layer participation is mechanical: one weight per selected layer."""
        for name, expected in (("mtrl_config.yml", 16), ("mtrl_alllayers_config.yml", 25)):
            with self.subTest(config=name):
                model = build_from_config(load_config(os.path.join(MTRL_DIR, name)))
                self.assertEqual(model.pooling.position_weights.numel(), expected)


if __name__ == "__main__":
    unittest.main()
