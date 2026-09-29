"""Cost accounting and ownership: derived spend, unknown prices, restart safety."""

import unittest
from decimal import Decimal

from improvements.compute import ledger
from improvements.compute.errors import ConfigurationError
from improvements.compute.tests.fakes import ComputeTestCase, worker_record


class ScopeMatchingTests(ComputeTestCase):
    def test_scope_prefix_is_the_ownership_tag(self):
        self.assertTrue(ledger.worker_belongs_to("wavcse-tr-0007-abc123", "TR-0007"))
        self.assertFalse(ledger.worker_belongs_to("wavcse-tr-0008-abc123", "TR-0007"))
        self.assertFalse(ledger.worker_belongs_to("wavcse-other-abc123", "TR-0007"))
        self.assertEqual(ledger.name_prefix_for("TR-0007"), "wavcse-tr-0007-")


class SpendTests(ComputeTestCase):
    def test_spend_is_derived_from_provider_facts(self):
        self.make_repo()
        workers = [worker_record(hourly="0.50", created="2026-09-29T00:00:00+00:00")]
        now = ledger.state_module.parse_timestamp("2026-09-29T02:00:00+00:00")
        report = ledger.derive_spend(workers, "TR-0007", leases=[], now=now)
        self.assertTrue(report.bounded)
        self.assertEqual(report.estimated_hourly_exposure_usd, Decimal("0.50"))
        self.assertEqual(report.estimated_wall_clock_hours, Decimal("2"))
        self.assertEqual(report.estimated_spend_usd, Decimal("1.00"))

    def test_unknown_price_is_unbounded(self):
        self.make_repo()
        report = ledger.derive_spend(
            [worker_record(hourly=None)], "TR-0007", leases=[]
        )
        self.assertFalse(report.bounded)
        self.assertTrue(report.unknowns)
        self.assertEqual(report.estimated_spend_usd, Decimal("0"))

    def test_unknown_creation_time_is_unbounded(self):
        self.make_repo()
        report = ledger.derive_spend(
            [worker_record(created=None)], "TR-0007", leases=[]
        )
        self.assertFalse(report.bounded)

    def test_stopped_worker_costs_nothing_more(self):
        self.make_repo()
        report = ledger.derive_spend(
            [worker_record(state="STOPPED")], "TR-0007", leases=[]
        )
        self.assertEqual(report.estimated_hourly_exposure_usd, Decimal("0"))
        self.assertEqual(report.billable, [])

    def test_other_scopes_are_not_counted(self):
        self.make_repo()
        workers = [worker_record(name="wavcse-tr-0008-xyz", hourly="9.99")]
        report = ledger.derive_spend(workers, "TR-0007", leases=[])
        self.assertEqual(report.estimated_spend_usd, Decimal("0"))
        self.assertEqual(report.live, [])

    def test_closed_lease_cost_survives_the_provider_record(self):
        self.make_repo()
        ledger.redeem_create(
            "TR-0007", worker_id="w-1", purpose="test",
            envelope_digest="digest", deadline="2999-01-01T00:00:00+00:00",
        )
        ledger.close_lease("w-1", cost_usd=Decimal("3.25"),
                           wall_clock_hours=Decimal("6.5"), state="destroyed")
        report = ledger.derive_spend([], "TR-0007")
        self.assertEqual(report.estimated_spend_usd, Decimal("3.25"))
        self.assertEqual(report.estimated_wall_clock_hours, Decimal("6.5"))


class OwnershipTests(ComputeTestCase):
    def test_create_intent_blocks_a_second_paid_request(self):
        self.make_repo()
        ledger.begin_create("TR-0007", purpose="a", envelope_digest="d",
                            request={}, deadline="2999-01-01T00:00:00+00:00")
        with self.assertRaises(ConfigurationError):
            ledger.begin_create("TR-0007", purpose="b", envelope_digest="d",
                                request={}, deadline="2999-01-01T00:00:00+00:00")

    def test_intent_is_reconciled_by_generated_identity(self):
        self.make_repo()
        intent = ledger.begin_create(
            "TR-0007", purpose="a", envelope_digest="d", request={},
            deadline="2999-01-01T00:00:00+00:00",
        )
        found = ledger.find_worker_for_intent(intent, [worker_record()])
        self.assertIsNotNone(found)
        self.assertIsNone(
            ledger.find_worker_for_intent(intent, [worker_record(name="wavcse-other-x")])
        )

    def test_abandoning_an_intent_allows_a_later_attempt(self):
        self.make_repo()
        ledger.begin_create("TR-0007", purpose="a", envelope_digest="d",
                            request={}, deadline="2999-01-01T00:00:00+00:00")
        ledger.abandon_create("TR-0007", "create provably did not happen")
        self.assertEqual(ledger.pending_creates("TR-0007"), [])

    def test_adopted_workers_are_marked(self):
        self.make_repo()
        lease = ledger.adopt_worker(
            worker_record(), scope="TR-0007",
            deadline="2999-01-01T00:00:00+00:00",
        )
        self.assertEqual(lease["provenance"], "adopted")

    def test_state_survives_a_restart(self):
        self.make_repo()
        ledger.redeem_create("TR-0007", worker_id="w-1", purpose="a",
                             envelope_digest="d", deadline="2999-01-01T00:00:00+00:00")
        # A "restart" is just a fresh read of the same directory.
        reloaded = ledger.leases_for("TR-0007")
        self.assertEqual(len(reloaded), 1)
        self.assertEqual(reloaded[0]["worker_id"], "w-1")

    def test_scope_is_busy_with_an_unresolved_intent(self):
        self.make_repo()
        self.assertFalse(ledger.scope_is_busy("TR-0007", workers=[]))
        ledger.begin_create("TR-0007", purpose="a", envelope_digest="d",
                            request={}, deadline="2999-01-01T00:00:00+00:00")
        self.assertTrue(ledger.scope_is_busy("TR-0007", workers=[]))


if __name__ == "__main__":
    unittest.main()
