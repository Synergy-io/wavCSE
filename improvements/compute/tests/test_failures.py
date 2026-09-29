"""Failure taxonomy: bounds, dispositions, and the two invariants that protect science."""

import unittest

from improvements.compute import failures
from improvements.compute.errors import (
    ArtifactIntegrityError,
    AuthorizationError,
    CapacityError,
    CostError,
    ImplementationBugError,
    ReconcilableError,
    ScientificFailureError,
    SecurityError,
    TransientInfraError,
)


class ExceptionMappingTests(unittest.TestCase):
    def test_each_typed_failure_maps_to_its_class(self):
        cases = [
            (TransientInfraError("x"), failures.TRANSIENT_INFRA),
            (ReconcilableError("x"), failures.TRANSIENT_INFRA),
            (CapacityError("x"), failures.CAPACITY),
            (CostError("x"), failures.COST),
            (AuthorizationError("x"), failures.AUTHORIZATION),
            (ArtifactIntegrityError("x"), failures.ARTIFACT_INTEGRITY),
            (SecurityError("x"), failures.SECURITY),
            (ImplementationBugError("x"), failures.IMPLEMENTATION_BUG),
            (ScientificFailureError("x"), failures.SCIENTIFIC_FAILURE),
        ]
        for exc, expected in cases:
            with self.subTest(exc=type(exc).__name__):
                self.assertEqual(failures.assess_exception(exc).klass, expected)

    def test_no_class_ever_allows_a_scientific_change(self):
        for exc in (TransientInfraError("x"), CapacityError("x"),
                    ReconcilableError("x"), ArtifactIntegrityError("x")):
            self.assertFalse(failures.assess_exception(exc).scientific_change_allowed)

    def test_security_is_never_retried(self):
        self.assertFalse(failures.assess_exception(SecurityError("x")).retryable)

    def test_ambiguous_outcome_must_be_reconciled_first(self):
        assessment = failures.assess_exception(ReconcilableError("x"))
        self.assertTrue(assessment.reconcile_first)
        allowed, reason = failures.decide_retry(assessment, 0, reconciled=False)
        self.assertFalse(allowed)
        self.assertIn("reconciled", reason)


class JobOutcomeTests(unittest.TestCase):
    def test_completed_run_is_evidence_not_a_defect(self):
        assessment = failures.assess_job_outcome(
            {"state": "SUCCEEDED", "exit_code": 0}, outputs_verified=True
        )
        self.assertEqual(assessment.klass, failures.SCIENTIFIC_FAILURE)
        self.assertFalse(assessment.retryable)
        self.assertFalse(assessment.repairable)

    def test_success_with_unverified_outputs_is_an_integrity_failure(self):
        assessment = failures.assess_job_outcome(
            {"state": "SUCCEEDED", "exit_code": 0}, outputs_verified=False
        )
        self.assertEqual(assessment.klass, failures.ARTIFACT_INTEGRITY)

    def test_oom_is_a_resource_failure(self):
        assessment = failures.assess_job_outcome(
            {"state": "FAILED", "exit_code": 137},
            log_text="torch.cuda.OutOfMemoryError: CUDA out of memory",
        )
        self.assertEqual(assessment.klass, failures.RESOURCE_OOM)
        self.assertTrue(assessment.retryable)
        self.assertFalse(assessment.scientific_change_allowed)
        self.assertIn("no scientific setting may change", assessment.reason)

    def test_plain_nonzero_exit_is_an_implementation_bug(self):
        assessment = failures.assess_job_outcome(
            {"state": "FAILED", "exit_code": 1},
            log_text="Traceback (most recent call last): KeyError: 'paths'",
        )
        self.assertEqual(assessment.klass, failures.IMPLEMENTATION_BUG)

    def test_retry_budget_is_bounded(self):
        assessment = failures.assess_exception(TransientInfraError("x"))
        allowed, _ = failures.decide_retry(assessment, 0)
        self.assertTrue(allowed)
        allowed, reason = failures.decide_retry(assessment, assessment.max_attempts)
        self.assertFalse(allowed)
        self.assertIn("exhausted", reason)

    def test_capacity_reselection_is_bounded(self):
        assessment = failures.assess_exception(CapacityError("x"))
        self.assertEqual(assessment.max_attempts, 2)
        allowed, _ = failures.decide_retry(assessment, 2)
        self.assertFalse(allowed)

    def test_authorization_and_cost_are_never_retried(self):
        for exc in (AuthorizationError("x"), CostError("x")):
            assessment = failures.assess_exception(exc)
            self.assertFalse(assessment.retryable)
            allowed, _ = failures.decide_retry(assessment, 0)
            self.assertFalse(allowed)


if __name__ == "__main__":
    unittest.main()
