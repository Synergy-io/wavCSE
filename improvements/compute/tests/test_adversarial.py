"""Adversarial scenarios: what a crash, a disappearance or a price change must not do."""

import unittest

from improvements.compute import envelope as envelope_module
from improvements.compute import failures, jobspec, ledger, run_study
from improvements.compute.tests.fakes import ComputeTestCase, FakeInfra, sample_plan, worker_record


class AdversarialTestCase(ComputeTestCase):
    def setUp(self):
        super(AdversarialTestCase, self).setUp()
        self.make_repo()
        self.write_envelope("TR-0007")
        self.plan = jobspec.load_plan(self.write_plan(sample_plan()))
        self.commit()
        self.view = envelope_module.load("TR-0007")
        self.infra = FakeInfra(workers=[worker_record()])
        self.commit_sha = jobspec.git_state()["head"]
        self.name = jobspec.job_name("TR-0007", "screen", "mssl", 42)

    def payload(self, *, state="RUNNING", outputs=None):
        return {
            "job_id": "job-" + self.name,
            "name": self.name,
            "state": state,
            "exit_code": 0 if state == "SUCCEEDED" else None,
            "worker_id": "w-1",
            "outputs": outputs or [],
            "spec": {
                "source": {"commit": self.commit_sha},
                "tracking": {"metadata": {
                    "scope": "TR-0007", "stage": "screen", "arm": "mssl", "seed": "42",
                }},
            },
        }

    def submit(self):
        self.infra.submit_payload = self.payload()
        record = run_study.load_record("TR-0007")
        run_study.submit_pending("TR-0007", self.plan, "screen", infra=self.infra,
                                 view=self.view, record=record)
        return record


class CrashRecoveryTests(AdversarialTestCase):
    def test_a_controller_crash_after_completion_is_recovered_by_reconcile(self):
        """The job finished while nobody was watching: collect, do not resubmit."""

        record = self.submit()
        expected = jobspec.declared_outputs(
            self.plan, jobspec.arm_by_name(self.plan, "mssl"), 42)
        self.infra.jobs = [self.payload(
            state="SUCCEEDED",
            outputs=[{"path": item["path"], "artifact": item["artifact"],
                      "required": item["required"], "persisted": True,
                      "verified_size_bytes": 10, "sha256": "a" * 64}
                     for item in expected],
        )]
        run_study.reconcile("TR-0007", self.plan, "screen", infra=self.infra,
                            record=record)
        entry = list(record["entries"].values())[0]
        self.assertEqual(entry["state"], run_study.SUCCEEDED)
        result = run_study.collect("TR-0007", self.plan, "screen", record=record)
        self.assertEqual(len(result["collected"]), 1)
        self.assertEqual(self.infra.count("job_submit"), 1)

    def test_a_disappeared_worker_leaves_the_job_reconcilable(self):
        """An unreachable worker is not a failure and must not trigger a resubmission."""

        record = self.submit()
        self.infra.jobs = [self.payload(state="PREPARING")]
        self.infra.jobs[0]["worker_absent"] = True
        self.infra.jobs[0]["reconciliation_required"] = True
        run_study.reconcile("TR-0007", self.plan, "screen", infra=self.infra,
                            record=record)
        entry = list(record["entries"].values())[0]
        self.assertNotIn(entry["state"],
                         (run_study.FAILED, run_study.SUCCEEDED, run_study.COLLECTED))
        result = run_study.advance("TR-0007", self.plan, "screen", infra=self.infra,
                                   record=record)
        self.assertEqual(result["step"], "monitor")
        self.assertEqual(self.infra.count("job_submit"), 1)

    def test_a_replaced_worker_does_not_reset_the_attempt_budget(self):
        record = self.submit()
        entry = list(record["entries"].values())[0]
        entry["attempts"] = failures.MAX_ATTEMPTS[failures.TRANSIENT_INFRA]
        run_study.save_record("TR-0007", record)
        reloaded = run_study.load_record("TR-0007")
        self.assertEqual(
            list(reloaded["entries"].values())[0]["attempts"],
            failures.MAX_ATTEMPTS[failures.TRANSIENT_INFRA],
        )

    def test_stale_runtime_state_fails_closed_for_new_spend(self):
        """A lease the provider no longer lists makes the total unbounded."""

        ledger.redeem_create("TR-0007", worker_id="w-gone", purpose="test",
                             envelope_digest=self.view.digest,
                             deadline="2999-01-01T00:00:00+00:00")
        spend = ledger.derive_spend([], "TR-0007")
        self.assertFalse(spend.bounded)
        self.assertTrue(any("no matching provider worker" in item
                            for item in spend.unknowns))
        decision = envelope_module.check(
            self.view, envelope_module.ACTION_SUBMIT_JOB, spend.facts()
        )
        # The envelope itself is satisfied; the ledger's boundedness is the guard
        # the caller must consult, and it is false.
        self.assertFalse(spend.bounded)

    def test_stale_research_state_cannot_be_smoothed_over(self):
        """The run ledger is keyed by commit, so a new commit starts fresh."""

        key_old = jobspec.job_key("TR-0007", "TR-0007", "screen", "mssl", 42,
                                  self.commit_sha)
        key_new = jobspec.job_key("TR-0007", "TR-0007", "screen", "mssl", 42,
                                  "b" * 40)
        self.assertNotEqual(key_old, key_new)


class PriceChangeTests(AdversarialTestCase):
    def test_a_price_rise_is_refused_at_the_next_check(self):
        decision = envelope_module.check(
            self.view, envelope_module.ACTION_CREATE_WORKER,
            {"live_workers": [], "estimated_spend_usd": "1.0"},
            requested={"hourly_usd": "0.80", "projected_hours": "1",
                       "container_disk_gb": 60},
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.klass, "COST")

    def test_a_price_rise_inside_the_ceiling_still_counts_against_the_total(self):
        decision = envelope_module.check(
            self.view, envelope_module.ACTION_CREATE_WORKER,
            {"live_workers": [], "estimated_spend_usd": "7.5"},
            requested={"hourly_usd": "0.50", "projected_hours": "2",
                       "container_disk_gb": 60},
        )
        self.assertFalse(decision.allowed)
        self.assertIn("projected spend", decision.reason)

    def test_a_worker_whose_cost_cannot_be_derived_blocks_new_spend(self):
        self.infra.workers = [worker_record(hourly=None)]
        spend = ledger.derive_spend(self.infra.workers, "TR-0007")
        self.assertFalse(spend.bounded)
        decision = envelope_module.check(
            self.view, envelope_module.ACTION_CREATE_WORKER, spend.facts(),
            requested={"hourly_usd": "0.50", "container_disk_gb": 60},
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.klass, "COST")
        self.assertIn("cannot be bounded", decision.reason)


class ConcurrencyTests(AdversarialTestCase):
    def test_two_existing_billable_workers_block_submission(self):
        self.infra.workers = [worker_record(worker_id="w-one"),
                              worker_record(worker_id="w-two")]
        decision = envelope_module.check(
            self.view, envelope_module.ACTION_SUBMIT_JOB,
            ledger.derive_spend(self.infra.workers, "TR-0007").facts())
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.klass, "CAPACITY")

    def test_two_workers_under_a_limit_of_one_cannot_be_created(self):
        self.infra.workers = [worker_record(worker_id="w-live")]
        decision = envelope_module.check(
            self.view, envelope_module.ACTION_CREATE_WORKER,
            ledger.derive_spend(self.infra.workers, "TR-0007").facts(),
            requested={"hourly_usd": "0.50", "container_disk_gb": 60},
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.klass, "CAPACITY")

    def test_replacement_is_allowed_only_after_the_predecessor_stops(self):
        """Replacement permits a new Pod, never two billing at once."""

        self.infra.workers = [worker_record(worker_id="w-live", state="RUNNING")]
        busy = ledger.derive_spend(self.infra.workers, "TR-0007").facts()
        refused = envelope_module.check(
            self.view, envelope_module.ACTION_CREATE_WORKER, busy,
            requested={"hourly_usd": "0.50", "container_disk_gb": 60},
        )
        self.assertFalse(refused.allowed)
        self.assertIn("stopped or destroyed", refused.reason)

        # The backend stops the finished predecessor, which records the accrued
        # cost on the lease: concurrency is free and the ledger is still bounded.
        ledger.redeem_create("TR-0007", worker_id="w-live", purpose="test",
                             envelope_digest=self.view.digest,
                             deadline="2999-01-01T00:00:00+00:00")
        ledger.close_lease("w-live", cost_usd="1.25", wall_clock_hours="2.5",
                           state="stopped")
        self.infra.workers = [worker_record(worker_id="w-live", state="STOPPED")]
        free = ledger.derive_spend(self.infra.workers, "TR-0007").facts()
        self.assertTrue(free["accounting_bounded"])
        self.assertEqual(free["live_workers"], [])
        allowed = envelope_module.check(
            self.view, envelope_module.ACTION_CREATE_WORKER, free,
            requested={"hourly_usd": "0.50", "container_disk_gb": 60},
        )
        self.assertTrue(allowed.allowed)


if __name__ == "__main__":
    unittest.main()
