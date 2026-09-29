"""A vanished worker after partial progress: keep the evidence, bound the cost, finish.

The scenario is the one that used to deadlock the cycle: two seeds collected, a third
running, and the Pod gone from provider inventory. The completed seeds are preserved, the
predecessor's spend is bounded from the last authoritative observation instead of guessed,
and only the missing work is scheduled — and only when the bound still fits the
authorization. When no bound can be proven, the scope stays fail-closed.
"""

import unittest

from improvements.compute import envelope as envelope_module
from improvements.compute import failures, jobspec, ledger, reaper, run_study
from improvements.compute import state as state_module
from improvements.compute import worker as worker_module
from improvements.compute.errors import CostError
from improvements.compute.tests.fakes import (
    ComputeTestCase,
    FakeInfra,
    sample_plan,
    staged_evidence,
    worker_record,
)


class VanishedWorkerTestCase(ComputeTestCase):
    SEEDS = (0, 1, 2)

    def setUp(self):
        super(VanishedWorkerTestCase, self).setUp()
        self.make_repo()
        self.write_envelope("TR-0007", overrides={
            "budget": {"max_gpu_hourly_usd": 0.5, "max_total_gpu_usd": 40.0,
                       "max_wall_clock_hours": 12},
        })
        self.plan = jobspec.load_plan(self.write_plan(sample_plan(
            stages={"confirm": {"seeds": list(self.SEEDS)}},
        )))
        self.commit()
        self.allow_remote_commit()
        self.view = envelope_module.load("TR-0007")
        self.infra = FakeInfra(workers=[worker_record(worker_id="w-1")])
        self.commit_sha = jobspec.git_state()["head"]
        self.lease()

    def lease(self, *, created="2026-09-01T00:00:00+00:00"):
        lease = ledger.redeem_create(
            "TR-0007", worker_id="w-1", purpose="test",
            envelope_digest=self.view.digest,
            deadline="2999-01-01T00:00:00+00:00",
        )
        document = ledger._leases_document()[1]
        for item in document["leases"]:
            item["created_at"] = created
        ledger._save(ledger.leases_path(), document)
        return lease

    def payload(self, seed, *, state="RUNNING", outputs=None):
        name = jobspec.job_name("TR-0007", "confirm", "mssl", seed)
        return {
            "job_id": "job-" + name,
            "name": name,
            "state": state,
            "exit_code": 0 if state == "SUCCEEDED" else None,
            "worker_id": "w-1",
            "outputs": outputs or [],
            "spec": {
                "source": {"commit": self.commit_sha},
                "tracking": {"metadata": {
                    "scope": "TR-0007", "stage": "confirm", "arm": "mssl",
                    "seed": str(seed),
                }},
            },
        }

    def next_unsubmitted_seed(self):
        """The seed one submission would carry: the first in plan order without a job."""

        record = run_study.load_record("TR-0007")
        submitted = {entry["seed"] for entry in record["entries"].values()
                     if entry.get("job_id")}
        for seed in self.SEEDS:
            if seed not in submitted:
                return seed
        raise AssertionError("every seed already holds a job")

    def submit_next(self):
        """Submit the next seed in plan order, with the payload that describes it."""

        seed = self.next_unsubmitted_seed()
        self.infra.submit_payload = self.payload(seed)
        record = run_study.load_record("TR-0007")
        run_study.submit_pending("TR-0007", self.plan, "confirm", infra=self.infra,
                                 view=self.view, record=record)
        return seed

    def collect_seed(self, seed):
        """Drive one seed through the real submit -> reconcile -> collect path."""

        self.assertEqual(self.submit_next(), seed)
        name = jobspec.job_name("TR-0007", "confirm", "mssl", seed)
        objects, outputs, _manifest = staged_evidence(
            self.plan, stage="confirm", arm="mssl", seed=seed, commit=self.commit_sha,
            job_id="job-" + name)
        self.infra.objects.update(objects)
        self.infra.jobs = [self.payload(seed, state="SUCCEEDED", outputs=outputs)]
        record = run_study.load_record("TR-0007")
        run_study.reconcile("TR-0007", self.plan, "confirm", infra=self.infra,
                            record=record)
        run_study.collect("TR-0007", self.plan, "confirm", record=record,
                          reader=self.infra)
        return record

    def progress_to(self, seed, *, observe=True):
        """Collect every earlier seed, then leave `seed` submitted and running."""

        for earlier in self.SEEDS:
            if earlier == seed:
                break
            self.collect_seed(earlier)
        if observe:
            self.observe_running()
        self.assertEqual(self.submit_next(), seed)
        return run_study.load_record("TR-0007")

    def seed_entry(self, record, seed):
        return [entry for entry in record["entries"].values()
                if entry["seed"] == seed][0]

    def worker_vanishes(self):
        """The Pod leaves inventory; the control plane keeps the job record it wrote."""

        self.infra.workers = []
        self.infra.jobs = []
        for record in self.infra.records.values():
            if str(record.get("state") or "").upper() != "SUCCEEDED":
                record["state"] = "FAILED"
                record["worker_absent"] = True
                record["remote_status"] = "worker_absent"
                record["exit_code"] = None

    def observe_running(self, *, hours_ago=3):
        """Persist the observation a live pass would have made three hours ago.

        Anchored to the real clock so a transition that reads ``utc_now()`` itself sees a
        bound of roughly ``hours_ago`` at the observed rate, exactly as production would.
        """

        from datetime import timedelta

        now = state_module.utc_now()
        started = now - timedelta(hours=hours_ago)
        ledger.observe_workers(
            [worker_record(worker_id="w-1", hourly="0.50",
                           created=started.isoformat())],
            scope="TR-0007", now=now,
        )
        return started


class RecoveryTests(VanishedWorkerTestCase):
    def partial_progress(self):
        """Seeds 0 and 1 collected; seed 2 submitted and running."""

        record = self.progress_to(2)
        states = {entry["seed"]: entry["state"] for entry in record["entries"].values()}
        self.assertEqual(states, {0: run_study.COLLECTED, 1: run_study.COLLECTED,
                                  2: run_study.SUBMITTED})
        return record

    def test_completed_seeds_survive_the_disappearance(self):
        record = self.partial_progress()
        self.worker_vanishes()

        run_study.advance("TR-0007", self.plan, "confirm", infra=self.infra,
                          record=record)

        record = run_study.load_record("TR-0007")
        preserved = [entry for entry in record["entries"].values()
                     if entry["seed"] in (0, 1)]
        self.assertTrue(all(entry["state"] == run_study.COLLECTED
                            for entry in preserved))
        self.assertTrue(all(entry.get("evidence_verified") for entry in preserved))

    def test_the_predecessor_cost_is_reconciled_from_an_authoritative_observation(self):
        record = self.partial_progress()
        self.worker_vanishes()

        run_study.advance("TR-0007", self.plan, "confirm", infra=self.infra,
                          record=record)

        lease = ledger.all_leases()[0]
        self.assertEqual(lease["state"], "vanished")
        self.assertEqual(lease["finalization"], ledger.FINALIZATION_UPPER_BOUND)
        self.assertEqual(lease["finalization_basis"]["hourly_cost_usd"], "0.50")
        # Roughly three hours at the observed $0.50/hour rate, and never less than that:
        # the figure is a ceiling, so it must not undercount the time already spent.
        cost = ledger.Decimal(lease["closed_cost_usd"])
        self.assertGreaterEqual(cost, ledger.Decimal("1.50"))
        self.assertLess(cost, ledger.Decimal("1.75"))
        hours = ledger.Decimal(lease["closed_wall_clock_hours"])
        self.assertGreaterEqual(hours, ledger.Decimal("3.0"))
        spend = ledger.derive_spend([], "TR-0007")
        self.assertTrue(spend.bounded)
        self.assertEqual(spend.estimated_spend_usd, cost)

    def test_only_the_missing_seed_is_scheduled_after_recovery(self):
        record = self.partial_progress()
        self.worker_vanishes()

        first = run_study.advance("TR-0007", self.plan, "confirm", infra=self.infra,
                                  record=record)
        # Reconciliation bounds the cost and schedules the missing seed, but there is
        # nothing left to run it on, and the transition says so instead of guessing.
        self.assertEqual(first["step"], "needs-worker")
        self.assertTrue(ledger.derive_spend([], "TR-0007").bounded)
        before = self.infra.count("job_submit")

        # Only now is a replacement worth creating: the ledger is bounded again, and the
        # retry is aimed at the new Pod rather than at the one that vanished.
        self.infra.create_worker_record = worker_record(worker_id="w-9")
        worker_module.ensure_worker(self.plan, self.view, infra=self.infra)
        self.infra.submit_payload = dict(self.payload(2), job_id="job-second")

        second = run_study.advance("TR-0007", self.plan, "confirm", infra=self.infra,
                                   record=run_study.load_record("TR-0007"))

        self.assertEqual(second["step"], "submitted")
        self.assertEqual(self.infra.count("job_submit"), before + 1)
        entries = run_study.load_record("TR-0007")["entries"].values()
        resubmitted = [item for item in entries if item["seed"] == 2][0]
        self.assertEqual(resubmitted["state"], run_study.SUBMITTED)
        self.assertEqual(resubmitted["worker_id"], "w-9")
        self.assertEqual(resubmitted["attempts"], 2)
        collected = [item for item in run_study.load_record("TR-0007")["entries"].values()
                     if item["seed"] in (0, 1)]
        self.assertTrue(all(item["state"] == run_study.COLLECTED for item in collected))

    def test_the_reaper_alone_recovers_the_same_situation(self):
        """No OMP, no cycle: the timer's command performs the same reconciliation."""

        record = self.partial_progress()
        self.worker_vanishes()

        report = reaper.reap(self.infra, execute=True)

        self.assertEqual(report["scopes"][0]["stopped"], [])
        finalized = report["scopes"][0]["finalized_leases"]
        self.assertEqual(len(finalized), 1)
        self.assertEqual(finalized[0]["state"], "vanished")
        self.assertEqual(finalized[0]["precision"], ledger.FINALIZATION_UPPER_BOUND)
        self.assertTrue(ledger.derive_spend([], "TR-0007").bounded)


class FailClosedTests(VanishedWorkerTestCase):
    def test_a_worker_never_observed_remains_unbounded(self):
        """No observation means no bound, so the scope stays blocked rather than guessed."""

        self.progress_to(2, observe=False)
        self.worker_vanishes()

        report = reaper.reap(self.infra, execute=True)

        unbounded = report["scopes"][0]["unbounded_leases"]
        self.assertEqual(len(unbounded), 1)
        self.assertIn("never observed", "; ".join(unbounded[0]["missing"]))
        lease = ledger.all_leases()[0]
        self.assertEqual(lease["state"], "active")
        spend = ledger.derive_spend([], "TR-0007")
        self.assertFalse(spend.bounded)

    def test_an_unbounded_ledger_still_blocks_new_spend(self):
        self.progress_to(2, observe=False)
        self.worker_vanishes()

        with self.assertRaises(CostError):
            worker_module.ensure_worker(self.plan, self.view, infra=self.infra)

    def test_an_observed_worker_without_a_rate_remains_unbounded(self):
        self.progress_to(2, observe=False)
        ledger.observe_workers(
            [worker_record(worker_id="w-1", hourly=None)], scope="TR-0007",
            now=state_module.utc_now(),
        )
        self.worker_vanishes()

        report = reaper.reap(self.infra, execute=True)

        unbounded = report["scopes"][0]["unbounded_leases"]
        self.assertEqual(len(unbounded), 1)
        self.assertIn("no observed hourly price", "; ".join(unbounded[0]["missing"]))
        self.assertFalse(ledger.derive_spend([], "TR-0007").bounded)

    def test_finalizing_twice_never_resets_the_predecessor_cost(self):
        self.progress_to(2)
        self.worker_vanishes()

        from datetime import timedelta

        observed_at = state_module.utc_now()
        first = ledger.reconcile_absent_leases("TR-0007", [], now=observed_at)
        second = ledger.reconcile_absent_leases(
            "TR-0007", [], now=observed_at + timedelta(hours=3))

        self.assertEqual(len(first["finalized"]), 1)
        self.assertEqual(second["finalized"], [])
        self.assertAlmostEqual(float(ledger.all_leases()[0]["closed_cost_usd"]), 1.5,
                               places=2)

    def test_a_worker_the_provider_still_lists_is_never_finalized(self):
        self.progress_to(2)
        ledger.observe_workers([worker_record(worker_id="w-1")], scope="TR-0007")

        result = ledger.reconcile_absent_leases(
            "TR-0007", [worker_record(worker_id="w-1")])

        self.assertEqual(result["finalized"], [])
        self.assertEqual(ledger.all_leases()[0]["state"], "active")

    def test_an_externally_stopped_worker_is_bounded_and_then_continues(self):
        """A Pod stopped outside this controller is the same accounting problem."""

        self.progress_to(2)
        stopped = worker_record(worker_id="w-1", state="STOPPED")

        from datetime import timedelta

        result = ledger.reconcile_absent_leases(
            "TR-0007", [stopped], now=state_module.utc_now() + timedelta(hours=1))

        self.assertEqual(len(result["finalized"]), 1)
        lease = ledger.all_leases()[0]
        self.assertEqual(lease["state"], "stopped")
        self.assertAlmostEqual(float(lease["closed_cost_usd"]), 2.0, places=2)
        self.assertTrue(ledger.derive_spend([stopped], "TR-0007").bounded)

    def test_an_unrecoverable_job_is_retried_as_infrastructure_not_as_science(self):
        self.progress_to(2)
        self.worker_vanishes()
        record = run_study.load_record("TR-0007")

        run_study.reconcile("TR-0007", self.plan, "confirm", infra=self.infra,
                            record=record)
        outcomes = run_study.assess(record, stage="confirm")

        entry = self.seed_entry(record, 2)
        self.assertEqual(entry["failure_class"], failures.TRANSIENT_INFRA)
        self.assertTrue(entry["retry_allowed"])
        self.assertEqual(entry["state"], run_study.PENDING)
        # Infrastructure failure is never scientific evidence, and the retry is not
        # aimed at the Pod that is gone.
        self.assertNotEqual(entry["failure_class"], failures.SCIENTIFIC_FAILURE)
        self.assertIsNone(entry["worker_id"])
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0]["class"], failures.TRANSIENT_INFRA)
        self.assertEqual(outcomes[0]["job_key"], entry["job_key"])


if __name__ == "__main__":
    unittest.main()
