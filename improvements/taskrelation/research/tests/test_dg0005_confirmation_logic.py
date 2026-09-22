"""Data-free checks for DG-0005's pre-registered confirmation gates."""

import os
import sys
import unittest

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
STUDY_DIR = os.path.join(
    REPO_ROOT, "improvements", "taskrelation", "research", "studies", "DG-0005"
)
DOWNSTREAM_DIR = os.path.join(REPO_ROOT, "downstream")
for path in (STUDY_DIR, DOWNSTREAM_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from analyze_confirmation import (  # noqa: E402
    CONFIRMATION_COMMIT,
    _exposure_check,
    _study_decision,
    _summary,
)


def _gradient_record(*, step_count=2820, ks=539.1, si=1461.7, er=47.2):
    return {
        "artifact": "synthetic",
        "task_array": ["ks", "si", "er"],
        "sample_interval_steps": 20,
        "total_training_steps": step_count,
        "sampled_steps": 142,
        "skipped_sample_steps_missing_tasks": 0,
        "shared_parameter_names": ["a", "b"],
        "shared_parameter_count": 1550800,
        "valid_examples_per_sampled_batch": {
            "ks": {"mean": ks, "min": ks, "max": ks},
            "si": {"mean": si, "min": si, "max": si},
            "er": {"mean": er, "min": er, "max": er},
        },
        "record_steps": list(range(1, 283, 20)),
    }


def _cross_seed(
    *,
    a0_late,
    a1_late,
    a1_middle=None,
    commit=CONFIRMATION_COMMIT,
    exposure=True,
):
    """Build the check dict that PLAN.md's decision rule consumes.

    `a1_middle` defaults to `a1_late`; pass it explicitly to exercise a case
    where the middle and late phases disagree.
    """
    a1_middle = a1_late if a1_middle is None else a1_middle
    seeds = range(1, len(a0_late) + 1)
    return {
        "confirmation_checks": {
            "all_exposure_checks_passed": exposure,
            "all_runs_same_git_commit": True,
            "git_commits": [commit],
            "all_runs_have_notes": True,
            "a0_late_ratio_at_least_3_by_seed": {
                str(seed): ratio >= 3.0
                for seed, ratio in zip(seeds, a0_late)
            },
            "a1_late_ratio_below_3_by_seed": {
                str(seed): ratio < 3.0
                for seed, ratio in zip(seeds, a1_late)
            },
            "a1_middle_and_late_ratio_at_least_3_by_seed": {
                str(seed): late >= 3.0 and middle >= 3.0
                for seed, late, middle in zip(seeds, a1_late, a1_middle)
            },
            "a0_middle_ratio_at_least_3_by_seed": {
                str(seed): ratio >= 3.0 for seed, ratio in zip(seeds, a0_late)
            },
            "a1_middle_ratio_below_3_by_seed": {
                str(seed): ratio < 3.0
                for seed, ratio in zip(seeds, a1_middle)
            },
        },
        "paired_ratio_delta_a1_minus_a0": {
            phase: _summary([
                a1 - a0 for a0, a1 in zip(a0_late, a1_late)
            ])
            for phase in ("middle", "late")
        },
    }


A0_LATE = [7.2, 7.6, 8.1, 6.9, 7.4]


class ExposureGateTests(unittest.TestCase):
    def test_accepts_a_ER_weighted_composition_that_reaches_ks_scale(self):
        check = _exposure_check(
            _gradient_record(),
            _gradient_record(ks=439.2, si=1174.0, er=434.8),
        )

        self.assertTrue(check["passed"])
        self.assertAlmostEqual(check["a1_mean_batch_total"], 2048.0)
        self.assertGreater(check["a1_er_to_ks_mean_count_ratio"], 0.5)

    def test_rejects_an_arm_that_changed_optimizer_exposure(self):
        check = _exposure_check(
            _gradient_record(),
            _gradient_record(step_count=2821, ks=439.2, si=1174.0, er=434.8),
        )

        self.assertFalse(check["passed"])
        self.assertIn("total_training_steps", check["mismatches"])

    def test_rejects_a_composition_that_does_not_reach_the_er_gate(self):
        check = _exposure_check(
            _gradient_record(),
            _gradient_record(ks=539.1, si=1461.7, er=47.2),
        )

        self.assertFalse(check["passed"])


class DecisionRuleTests(unittest.TestCase):
    def test_confirms_when_a1_removes_late_dominance_in_every_seed(self):
        decision = _study_decision(_cross_seed(
            a0_late=A0_LATE,
            a1_late=[2.8, 2.6, 2.9, 2.4, 2.7],
        ))

        self.assertEqual(decision["decision"], "CONFIRMED")
        self.assertTrue(decision["decision_basis"]["protocol_valid"])

    def test_rejects_when_the_candidate_keeps_middle_and_late_dominance(self):
        decision = _study_decision(_cross_seed(
            a0_late=A0_LATE,
            a1_late=[7.4, 7.5, 8.0, 7.1, 7.3],
        ))

        self.assertEqual(decision["decision"], "REJECTED")
        self.assertTrue(
            decision["decision_basis"][
                "a1_middle_and_late_ratio_at_least_3_all_seeds"
            ]
        )

    def test_is_inconclusive_when_only_some_seeds_cross_the_threshold(self):
        decision = _study_decision(_cross_seed(
            a0_late=A0_LATE,
            a1_late=[2.8, 3.4, 2.9, 2.4, 2.7],
        ))

        self.assertEqual(decision["decision"], "INCONCLUSIVE")

    def test_is_inconclusive_when_no_baseline_dominance_exists_to_remove(self):
        decision = _study_decision(_cross_seed(
            a0_late=[2.2, 2.6, 2.1, 2.9, 2.4],
            a1_late=[2.0, 2.1, 1.9, 2.2, 2.0],
        ))

        self.assertEqual(decision["decision"], "INCONCLUSIVE")

    def test_confirms_on_the_pre_registered_late_condition_alone(self):
        """PLAN.md's support rule names the late phase only.

        A middle phase that stays above 3.0 while late falls below it still
        satisfies the pre-registered support condition, so it must not be
        silently re-gated after the fact; the discrepancy is reported instead.
        """
        cross_seed = _cross_seed(
            a0_late=A0_LATE,
            a1_late=[2.8, 2.6, 2.9, 2.4, 2.7],
            a1_middle=[3.5, 3.4, 3.2, 3.1, 3.6],
        )

        decision = _study_decision(cross_seed)

        self.assertEqual(decision["decision"], "CONFIRMED")
        self.assertFalse(all(
            cross_seed["confirmation_checks"][
                "a1_middle_ratio_below_3_by_seed"
            ].values()
        ))

    def test_cannot_confirm_from_a_different_implementation_commit(self):
        decision = _study_decision(_cross_seed(
            a0_late=A0_LATE,
            a1_late=[2.8, 2.6, 2.9, 2.4, 2.7],
            commit="0" * 40,
        ))

        self.assertEqual(decision["decision"], "INCONCLUSIVE")
        self.assertFalse(decision["decision_basis"]["protocol_valid"])

    def test_cannot_confirm_when_an_exposure_check_failed(self):
        decision = _study_decision(_cross_seed(
            a0_late=A0_LATE,
            a1_late=[2.8, 2.6, 2.9, 2.4, 2.7],
            exposure=False,
        ))

        self.assertEqual(decision["decision"], "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
