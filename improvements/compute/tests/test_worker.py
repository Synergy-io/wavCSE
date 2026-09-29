"""Worker acquisition: intents, duplicate-create safety, price enforcement."""

import unittest
from decimal import Decimal

from improvements.compute import envelope as envelope_module
from improvements.compute import ledger, worker as worker_module
from improvements.compute.errors import CapacityError, CostError, ReconcilableError
from improvements.compute.tests.fakes import ComputeTestCase, FakeInfra, sample_plan, worker_record
from improvements.compute import jobspec


class EnsureWorkerTests(ComputeTestCase):
    def setUp(self):
        super(EnsureWorkerTests, self).setUp()
        self.make_repo()
        self.write_envelope("TR-0007")
        self.view = envelope_module.load("TR-0007")
        self.plan = jobspec.load_plan(self.write_plan(sample_plan()))

    def ready_record(self, **overrides):
        record = worker_record(**overrides)
        record.setdefault("readiness", "READY")
        return record

    def test_existing_worker_is_reused_without_creating(self):
        infra = FakeInfra(workers=[self.ready_record()])
        created, actions = worker_module.ensure_worker(self.plan, self.view, infra=infra)
        self.assertEqual(created["id"], "w-1")
        self.assertEqual(infra.count("worker_create"), 0)
        self.assertEqual(infra.count("worker_bootstrap"), 1)
        self.assertIn("reused worker w-1", actions)

    def test_missing_worker_is_created_once_and_leased(self):
        infra = FakeInfra()
        infra.create_worker_record = self.ready_record(worker_id="w-9")
        created, actions = worker_module.ensure_worker(self.plan, self.view, infra=infra)
        self.assertEqual(created["id"], "w-9")
        self.assertEqual(infra.count("worker_create"), 1)
        leases = ledger.leases_for("TR-0007")
        self.assertEqual(len(leases), 1)
        self.assertEqual(leases[0]["worker_id"], "w-9")
        self.assertEqual(leases[0]["envelope_digest"], self.view.digest)

    def test_ceiling_is_passed_to_the_control_plane_guard(self):
        infra = FakeInfra()
        infra.create_worker_record = self.ready_record()
        worker_module.ensure_worker(self.plan, self.view, infra=infra)
        self.assertEqual(infra.last_create_kwargs["max_price"], 0.5)

    def test_create_intent_is_recorded_before_the_paid_call(self):
        infra = FakeInfra()

        def spy(**kwargs):
            intents = ledger.pending_creates("TR-0007")
            self.assertEqual(len(intents), 1, "intent must exist before the create")
            return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        infra.worker_create = spy
        with self.assertRaises(ReconcilableError):
            worker_module.ensure_worker(self.plan, self.view, infra=infra)

    def test_lost_acknowledgement_never_creates_a_second_worker(self):
        infra = FakeInfra()
        # The create succeeds at the provider but the controller sees nothing.
        infra.create_worker_record = None
        with self.assertRaises(ReconcilableError) as caught:
            worker_module.ensure_worker(self.plan, self.view, infra=infra)
        self.assertEqual(caught.exception.action, "create-worker")
        self.assertEqual(infra.count("worker_create"), 1)
        self.assertEqual(len(ledger.pending_creates("TR-0007")), 1)

        # A later attempt must reconcile, not issue a second paid request.
        with self.assertRaises(Exception):
            worker_module.ensure_worker(self.plan, self.view, infra=infra)
        self.assertEqual(infra.count("worker_create"), 1)

    def test_failed_create_closes_the_intent(self):
        infra = FakeInfra()
        infra.create_returncode = 1
        infra.create_stderr = "ResourceUnavailableError: no capacity"
        with self.assertRaises(Exception):
            worker_module.ensure_worker(self.plan, self.view, infra=infra)
        self.assertEqual(ledger.pending_creates("TR-0007"), [])

    def test_price_above_ceiling_after_create_is_destroyed(self):
        infra = FakeInfra()
        infra.create_worker_record = self.ready_record(hourly="0.99")
        with self.assertRaises(CostError):
            worker_module.ensure_worker(self.plan, self.view, infra=infra)
        self.assertEqual(infra.destroy_calls, ["w-1"])

    def test_unknown_price_after_create_is_destroyed(self):
        infra = FakeInfra()
        infra.create_worker_record = self.ready_record(hourly=None)
        with self.assertRaises(CostError):
            worker_module.ensure_worker(self.plan, self.view, infra=infra)
        self.assertEqual(infra.destroy_calls, ["w-1"])

    def test_second_live_worker_is_not_created_under_a_limit_of_one(self):
        infra = FakeInfra(workers=[self.ready_record()])
        worker_module.ensure_worker(self.plan, self.view, infra=infra)
        self.assertEqual(infra.count("worker_create"), 0)

    def test_stopped_worker_is_restarted_before_reuse(self):
        infra = FakeInfra(workers=[self.ready_record(state="STOPPED")])
        worker_module.ensure_worker(self.plan, self.view, infra=infra)
        self.assertEqual(infra.count("worker_start"), 1)
        self.assertEqual(infra.count("worker_create"), 0)

    def test_volume_selector_with_no_candidate_never_creates_a_volume(self):
        plan = jobspec.load_plan(self.write_plan(
            sample_plan(worker={
                "gpu_type": "NVIDIA GeForce RTX 4090", "cloud": "SECURE",
                "gpu_count": 1, "image": "runpod/pytorch:example",
                "container_disk_gb": 60,
                "network_volume": {"min_size_gb": 500},
            }),
            name="plan-volume.json",
        ))
        infra = FakeInfra(volumes=[{"id": "vol-1", "size_gb": 100,
                                    "datacenter": "US-KS-2"}])
        with self.assertRaises(CapacityError):
            worker_module.ensure_worker(plan, self.view, infra=infra)
        self.assertEqual(infra.count("worker_create"), 0)

    def test_adopted_worker_is_not_destroyed(self):
        infra = FakeInfra()
        ledger.adopt_worker(worker_record(), scope="TR-0007",
                            deadline="2999-01-01T00:00:00+00:00")
        with self.assertRaises(CapacityError):
            worker_module.destroy_worker(infra, "w-1", reason="cleanup")
        self.assertEqual(infra.destroy_calls, [])

    def test_stop_freezes_the_accrued_cost(self):
        infra = FakeInfra(workers=[self.ready_record()])
        ledger.redeem_create("TR-0007", worker_id="w-1", purpose="a",
                             envelope_digest="d", deadline="2999-01-01T00:00:00+00:00")
        worker_module.stop_worker(infra, "w-1", reason="test")
        leases = ledger.leases_for("TR-0007")
        self.assertEqual(leases[0]["state"], "stopped")
        self.assertIsNotNone(leases[0]["closed_cost_usd"])


if __name__ == "__main__":
    unittest.main()
