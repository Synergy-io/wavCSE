"""Authorization envelope: strict schema, cost limits, tamper and expiry."""

import os
import unittest

from improvements.compute import envelope as envelope_module
from improvements.compute import ledger
from improvements.compute.errors import AuthorizationError, ConfigurationError, CostError
from improvements.compute.tests.fakes import ComputeTestCase, worker_record


class SchemaTests(ComputeTestCase):
    def test_valid_envelope_loads(self):
        self.make_repo()
        self.write_envelope("TR-0007")
        view = envelope_module.load("TR-0007")
        self.assertEqual(view.envelope["scope"], "TR-0007")
        self.assertFalse(view.modified_in_tree)
        self.assertFalse(view.uncommitted)

    def test_unknown_key_is_rejected(self):
        self.make_repo()
        path = self.write_envelope("TR-0007")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write("max_total_usd: 5\n")
        with self.assertRaises(ConfigurationError):
            envelope_module.load("TR-0007")

    def test_secrets_and_identifiers_are_rejected(self):
        for bad in ("X-Amz-Signature=abc", "Bearer abc", "AKIAIOSFODNN7EXAMPLE"):
            with self.subTest(bad=bad):
                self.make_repo()
                path = self.write_envelope("TR-0007")
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write("granted_by: \"{}\"\n".format(bad))
                with self.assertRaises(ConfigurationError):
                    envelope_module.load("TR-0007")

    def test_scope_must_match_the_filename(self):
        self.make_repo()
        self.write_envelope("TR-0007", overrides={"scope": "TR-0009"})
        with self.assertRaises(ConfigurationError):
            envelope_module.load("TR-0007")

    def test_missing_envelope_is_an_authorization_error(self):
        self.make_repo()
        with self.assertRaises(AuthorizationError):
            envelope_module.load("TR-0007")


class DecisionTests(ComputeTestCase):
    def setUp(self):
        super(DecisionTests, self).setUp()
        self.make_repo()
        self.write_envelope("TR-0007")
        self.view = envelope_module.load("TR-0007")

    def facts(self, *, spend="0", wall="0", workers=()):
        return {
            "live_workers": list(workers),
            "estimated_spend_usd": spend,
            "estimated_wall_clock_hours": wall,
        }

    def test_absent_authorization_stops_before_spend(self):
        with self.assertRaises(AuthorizationError):
            envelope_module.load("TR-9999")

    def test_hourly_price_above_ceiling_is_refused(self):
        decision = envelope_module.check(
            self.view, envelope_module.ACTION_CREATE_WORKER, self.facts(),
            requested={"hourly_usd": "0.75", "container_disk_gb": 40},
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.klass, "COST")
        with self.assertRaises(CostError):
            decision.require()

    def test_projected_total_above_budget_is_refused(self):
        decision = envelope_module.check(
            self.view, envelope_module.ACTION_CREATE_WORKER,
            self.facts(spend="7.9"),
            requested={"hourly_usd": "0.5", "projected_hours": "4",
                       "container_disk_gb": 40},
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.klass, "COST")

    def test_unknown_price_fails_closed(self):
        decision = envelope_module.check(
            self.view, envelope_module.ACTION_CREATE_WORKER, self.facts(),
            requested={"container_disk_gb": 40},
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.klass, "COST")

    def test_concurrency_limit_is_enforced(self):
        decision = envelope_module.check(
            self.view, envelope_module.ACTION_CREATE_WORKER,
            self.facts(workers=[{"id": "w-1", "name": "wavcse-tr-0007-x"}]),
            requested={"hourly_usd": "0.5", "container_disk_gb": 40},
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.klass, "CAPACITY")
        self.assertIn("replacement", decision.reason)

    def test_container_disk_limit_is_enforced(self):
        decision = envelope_module.check(
            self.view, envelope_module.ACTION_CREATE_WORKER, self.facts(),
            requested={"hourly_usd": "0.5", "container_disk_gb": 500},
        )
        self.assertFalse(decision.allowed)

    def test_volume_creation_is_refused_when_not_granted(self):
        decision = envelope_module.check(
            self.view, envelope_module.ACTION_CREATE_VOLUME, self.facts()
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.klass, "AUTHORIZATION")

    def test_within_authorization_is_autonomous(self):
        decision = envelope_module.check(
            self.view, envelope_module.ACTION_CREATE_WORKER, self.facts(),
            requested={"hourly_usd": "0.5", "projected_hours": "1",
                       "container_disk_gb": 40},
        )
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.klass, "AUTONOMOUS")


class IntegrityTests(ComputeTestCase):
    def test_expired_envelope_cannot_self_renew(self):
        self.make_repo()
        self.write_envelope("TR-0007", overrides={
            "expires_at": "2000-01-01T00:00:00+00:00"})
        view = envelope_module.load("TR-0007")
        decision = envelope_module.check(
            view, envelope_module.ACTION_SUBMIT_JOB, {"live_workers": []}
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.klass, "AUTHORIZATION")
        self.assertIn("expired", decision.reason)

    def test_modified_envelope_is_refused(self):
        self.make_repo()
        path = self.write_envelope("TR-0007")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write("\n# widened after the fact\n")
        view = envelope_module.load("TR-0007")
        self.assertTrue(view.modified_in_tree)
        decision = envelope_module.check(
            view, envelope_module.ACTION_CREATE_WORKER,
            {"live_workers": [], "estimated_spend_usd": "0"},
            requested={"hourly_usd": "0.5", "container_disk_gb": 40},
        )
        self.assertFalse(decision.allowed)

    def test_uncommitted_envelope_is_refused(self):
        self.make_repo()
        self.write_envelope("TR-0007", commit=False)
        view = envelope_module.load("TR-0007")
        self.assertTrue(view.uncommitted)
        decision = envelope_module.check(
            view, envelope_module.ACTION_SUBMIT_JOB, {"live_workers": []}
        )
        self.assertFalse(decision.allowed)

    def test_snapshot_refuses_rotation_while_the_scope_is_busy(self):
        self.make_repo()
        path = self.write_envelope("TR-0007")
        view = envelope_module.load("TR-0007")
        envelope_module.record_snapshot(view, busy=False)
        old = view.digest
        with open(path, "a", encoding="utf-8") as handle:
            handle.write("\n# a new grant\n")
        import subprocess

        subprocess.run(["git", "-C", os.environ["WAVCSE_REPO_ROOT"], "add", "."],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        subprocess.run(
            ["git", "-C", os.environ["WAVCSE_REPO_ROOT"], "-c", "user.name=t",
             "-c", "user.email=t@example.invalid", "commit", "--quiet", "-m", "regrant"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        new_view = envelope_module.load("TR-0007")
        self.assertNotEqual(old, new_view.digest)
        with self.assertRaises(AuthorizationError):
            envelope_module.record_snapshot(new_view, busy=True)
        # Not busy: the deliberate re-grant is accepted and the old one kept.
        envelope_module.record_snapshot(new_view, busy=False)
        active = envelope_module.active_snapshot("TR-0007")
        self.assertEqual(active["digest"], new_view.digest)

    def test_orchestrator_has_no_way_to_grant_or_widen(self):
        grants = [
            name for name in dir(envelope_module)
            if name in ("grant", "renew", "widen", "create_envelope", "write_envelope")
        ]
        self.assertEqual(grants, [])


if __name__ == "__main__":
    unittest.main()
