"""Sweep: classification, dry-run default, and what it must never touch."""

import unittest

from improvements.compute import ledger, sweep as sweep_module
from improvements.compute.tests.fakes import ComputeTestCase, FakeInfra, worker_record

DEADLINE_PAST = "2000-01-01T00:00:00+00:00"
DEADLINE_FUTURE = "2999-01-01T00:00:00+00:00"


class SweepTests(ComputeTestCase):
    def setUp(self):
        super(SweepTests, self).setUp()
        self.make_repo()

    def lease(self, worker_id, deadline=DEADLINE_FUTURE, provenance="arc"):
        lease = ledger.redeem_create(
            "TR-0007", worker_id=worker_id, purpose="test",
            envelope_digest="d", deadline=deadline,
        )
        if provenance != "arc":
            lease["provenance"] = provenance
            ledger._save(ledger.leases_path(), ledger._leases_document()[1])
        return lease

    def test_dry_run_performs_nothing(self):
        self.lease("w-1", deadline=DEADLINE_PAST)
        infra = FakeInfra(workers=[worker_record(worker_id="w-1")])
        plan = sweep_module.sweep(infra, "TR-0007", execute=False)
        self.assertTrue(plan["dry_run"])
        self.assertEqual(infra.stop_calls, [])
        stale = [entry for entry in plan["employees"]
                 if entry["classification"] == sweep_module.STALE]
        self.assertEqual(len(stale), 1)

    def test_execute_stops_only_the_stale_worker(self):
        self.lease("w-1", deadline=DEADLINE_PAST)
        self.lease("w-2", deadline=DEADLINE_FUTURE)
        infra = FakeInfra(workers=[
            worker_record(worker_id="w-1", name="wavcse-tr-0007-a"),
            worker_record(worker_id="w-2", name="wavcse-tr-0007-b"),
        ])
        plan = sweep_module.sweep(infra, "TR-0007", execute=True)
        self.assertEqual(infra.stop_calls, ["w-1"])
        self.assertEqual(len(plan["performed"]), 1)

    def test_deadline_stops_a_running_job_before_spend_can_continue(self):
        self.lease("w-1", deadline=DEADLINE_PAST)
        infra = FakeInfra(
            workers=[worker_record(worker_id="w-1")],
            jobs=[{"job_id": "job-1", "state": "RUNNING", "worker_id": "w-1"}],
        )
        plan = sweep_module.sweep(infra, "TR-0007", execute=True)
        self.assertEqual(infra.stop_calls, ["w-1"])
        self.assertEqual(plan["employees"][0]["classification"], sweep_module.STALE)

    def test_unknown_worker_is_reported_and_untouched(self):
        infra = FakeInfra(workers=[worker_record(
            worker_id="w-x", name="wavcse-someone-else-abc")])
        plan = sweep_module.sweep(infra, "TR-0007", execute=True)
        self.assertEqual(len(plan["unowned_workers"]), 1)
        self.assertEqual(infra.stop_calls, [])

    def test_volumes_are_never_in_the_plan(self):
        self.lease("w-1", deadline=DEADLINE_PAST)
        infra = FakeInfra(
            workers=[worker_record(worker_id="w-1")],
            volumes=[{"id": "vol-1", "size_gb": 100}],
        )
        plan = sweep_module.sweep(infra, "TR-0007", execute=True)
        self.assertEqual(plan["volumes_touched"], [])
        self.assertNotIn("worker_destroy", infra.verbs())
        for call in infra.calls:
            self.assertNotIn("volume_destroy", str(call))
            self.assertNotIn("volume_create", str(call))

    def test_adopted_stale_worker_is_stopped_never_destroyed(self):
        self.lease("w-1", deadline=DEADLINE_PAST, provenance="adopted")
        infra = FakeInfra(workers=[worker_record(worker_id="w-1")])
        sweep_module.sweep(infra, "TR-0007", execute=True)
        self.assertEqual(infra.stop_calls, ["w-1"])
        self.assertEqual(infra.destroy_calls, [])

    def test_sweep_is_idempotent(self):
        self.lease("w-1", deadline=DEADLINE_PAST)
        infra = FakeInfra(workers=[worker_record(worker_id="w-1")])
        first = sweep_module.sweep(infra, "TR-0007", execute=False)
        second = sweep_module.sweep(infra, "TR-0007", execute=False)
        self.assertEqual(
            [entry["worker_id"] for entry in first["employees"]],
            [entry["worker_id"] for entry in second["employees"]],
        )
        self.assertEqual(first["unowned_workers"], second["unowned_workers"])

    def test_fully_stopped_workers_do_not_block(self):
        infra = FakeInfra(workers=[worker_record(worker_id="w-1", state="STOPPED")])
        plan = sweep_module.sweep(infra, "TR-0007", execute=False)
        self.assertEqual(infra.stop_calls, [])
        self.assertEqual(plan["employees"], [])
        self.assertEqual(len(plan["inactive"]), 1)


if __name__ == "__main__":
    unittest.main()
