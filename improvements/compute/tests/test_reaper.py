"""Crash independence: the reaper must end spend when no orchestrator is alive.

Every test here is written against the property that actually matters — *a dead process
must not imply unlimited billing* — rather than against the reaper's internals: the
lease on disk is the only input, the provider is the only authority, and the reaper is a
separate command that no session has to keep alive.
"""

import os
import subprocess
import sys
import unittest

from improvements.compute import ledger, reaper, state as state_module
from improvements.compute.errors import BusyError
from improvements.compute.tests.fakes import (
    REPO_ROOT,
    ComputeTestCase,
    FakeInfra,
    worker_record,
)

DEADLINE_PAST = "2000-01-01T00:00:00+00:00"
DEADLINE_FUTURE = "2999-01-01T00:00:00+00:00"


class ReaperTestCase(ComputeTestCase):
    def setUp(self):
        super(ReaperTestCase, self).setUp()
        self.make_repo()

    def lease(self, worker_id, *, scope="TR-0007", deadline=DEADLINE_FUTURE,
              provenance="arc", created="2026-09-01T00:00:00+00:00"):
        """An ARC lease as it exists on disk after `worker-ensure` created the Pod."""

        if provenance == "pending-intent":
            ledger.begin_create(scope, purpose="job execution", envelope_digest="d",
                                request={"gpu": "gpu"}, deadline=deadline)
        lease = ledger.redeem_create(
            scope, worker_id=worker_id, purpose="test", envelope_digest="d",
            deadline=deadline,
        )
        if provenance not in ("arc", "pending-intent"):
            lease["provenance"] = provenance
            ledger._save(ledger.leases_path(), ledger._leases_document()[1])
        if created:
            # Backdate the lease so an observation has a billing period to measure from.
            document = ledger._leases_document()[1]
            for item in document["leases"]:
                if item.get("worker_id") == worker_id:
                    item["created_at"] = created
            ledger._save(ledger.leases_path(), document)
        return lease

    def observe(self, worker_id, *, at, state="RUNNING", hourly="0.50",
                started="2026-09-01T00:00:00+00:00"):
        """Persist an observation exactly as a live pass would have recorded it."""

        workers = [worker_record(worker_id=worker_id, state=state, hourly=hourly,
                                 created=started)]
        ledger.observe_workers(workers, now=state_module.parse_timestamp(at))
        return workers


class ReaperEnforcementTests(ReaperTestCase):
    def test_a_dead_controller_does_not_leave_a_pod_billing(self):
        """The headline property: worker created, controller gone, deadline passes."""

        self.lease("w-1", deadline=DEADLINE_PAST)
        self.observe("w-1", at="2026-09-01T00:30:00+00:00")
        infra = FakeInfra(workers=[worker_record(worker_id="w-1")])

        report = reaper.reap(infra, execute=True,
                             now=state_module.parse_timestamp("2026-09-01T01:00:00+00:00"))

        self.assertEqual(infra.stop_calls, ["w-1"])
        self.assertEqual(report["scopes"][0]["stopped"], ["w-1"])
        lease = ledger.all_leases()[0]
        self.assertEqual(lease["state"], "stopped")
        self.assertIsNotNone(lease["closed_cost_usd"])

    def test_an_active_job_inside_its_deadline_is_left_running(self):
        self.lease("w-1", deadline=DEADLINE_FUTURE)
        infra = FakeInfra(
            workers=[worker_record(worker_id="w-1")],
            jobs=[{"job_id": "job-1", "state": "RUNNING", "worker_id": "w-1",
                   "spec": {"tracking": {"metadata": {"scope": "TR-0007"}}}}],
        )

        report = reaper.reap(infra, execute=True)

        self.assertEqual(infra.stop_calls, [])
        self.assertEqual(report["scopes"][0]["classifications"][0]["classification"],
                         "job_running")

    def test_an_unknown_worker_is_reported_and_never_touched(self):
        self.lease("w-1", deadline=DEADLINE_PAST)
        infra = FakeInfra(workers=[
            worker_record(worker_id="w-1"),
            worker_record(worker_id="w-x", name="wavcse-someone-else-abc"),
        ])

        report = reaper.reap(infra, execute=True)

        self.assertEqual(infra.stop_calls, ["w-1"])
        unowned = report["scopes"][0]["unowned_workers"]
        self.assertEqual([item["worker_id"] for item in unowned], ["w-x"])

    def test_a_worker_belonging_to_a_foreign_scope_is_untouched(self):
        """A scope this controller has no lease for is not even inspected."""

        self.lease("w-1", scope="TR-0007", deadline=DEADLINE_PAST)
        foreign = worker_record(worker_id="w-f", name="wavcse-tr-9999-abc")
        infra = FakeInfra(workers=[worker_record(worker_id="w-1"), foreign])

        report = reaper.reap(infra, execute=True)

        self.assertEqual(infra.stop_calls, ["w-1"])
        self.assertEqual([item["scope"] for item in report["scopes"]], ["TR-0007"])

    def test_a_persistent_volume_is_never_in_the_plan(self):
        self.lease("w-1", deadline=DEADLINE_PAST)
        infra = FakeInfra(
            workers=[worker_record(worker_id="w-1")],
            volumes=[{"id": "vol-1", "size_gb": 100}],
        )

        report = reaper.reap(infra, execute=True)

        self.assertEqual(report["scopes"][0]["volumes_touched"], [])
        for call in infra.calls:
            self.assertNotIn("volume", str(call))
        self.assertNotIn("worker_destroy", infra.verbs())
        self.assertNotIn("worker_create", infra.verbs())

    def test_an_expired_authorization_still_ends_the_spend(self):
        """Stopping needs no envelope: the lease deadline is itself the authority."""

        self.lease("w-1", deadline=DEADLINE_PAST)
        infra = FakeInfra(workers=[worker_record(worker_id="w-1")])

        reaper.reap(infra, execute=True)

        self.assertEqual(infra.stop_calls, ["w-1"])

    def test_a_live_orchestrator_holding_the_scope_lock_is_never_raced(self):
        self.lease("w-1", deadline=DEADLINE_PAST)
        infra = FakeInfra(workers=[worker_record(worker_id="w-1")])

        with state_module.scope_lock("TR-0007"):
            report = reaper.reap(infra, execute=True)

        self.assertEqual(infra.stop_calls, [])
        self.assertIn("skipped", report["scopes"][0])
        self.assertEqual(reaper.exit_code(report), 3)

    def test_a_live_controller_process_blocks_the_whole_pass(self):
        """The host guard is taken before any scope: a mid-mutation host is off limits."""

        self.lease("w-1", deadline=DEADLINE_PAST)
        infra = FakeInfra(workers=[worker_record(worker_id="w-1")])

        with state_module.controller_guard():
            report = reaper.reap(infra, execute=True)

        self.assertEqual(infra.stop_calls, [])
        self.assertEqual(report["scopes"], [])
        self.assertIn("skipped", report)
        self.assertEqual(reaper.exit_code(report), 3)
        # The refusal is recorded, and nothing was written into the ledger.
        self.assertIsNone(ledger.all_leases()[0].get("observation"))
        kinds = [event.get("kind") for event in state_module.read_events()]
        self.assertIn("reaper-skipped-controller-guard", kinds)

    def test_a_dry_run_writes_nothing_at_all(self):
        self.lease("w-1", deadline=DEADLINE_PAST)
        infra = FakeInfra(workers=[worker_record(worker_id="w-1")])
        before = ledger.all_leases()[0].get("observation")

        report = reaper.reap(infra, execute=False)

        self.assertTrue(report["dry_run"])
        self.assertEqual(infra.stop_calls, [])
        self.assertIsNone(ledger.all_leases()[0].get("observation"))
        self.assertEqual(before, None)
        self.assertFalse(os.path.exists(reaper.report_path()))

    def test_a_second_pass_performs_nothing_again(self):
        self.lease("w-1", deadline=DEADLINE_PAST)
        infra = FakeInfra(workers=[worker_record(worker_id="w-1")])

        reaper.reap(infra, execute=True)
        second = reaper.reap(infra, execute=True)

        self.assertEqual(infra.stop_calls, ["w-1"])
        self.assertEqual(second["scopes"][0]["stopped"], [])
        self.assertEqual(os.path.exists(reaper.report_path()), True)

    def test_a_create_intent_left_by_a_crash_is_resolved_and_enforced(self):
        """A controller that died between the intent and the lease must not leak.

        The crash state is exactly one pending intent and no lease: the intent is
        written before the billable request, so that is all the record a dead
        controller leaves behind. The reaper must recognise the Pod by the generated
        name and then enforce the deadline that intent carried.
        """

        intent = ledger.begin_create("TR-0007", purpose="job execution",
                                     envelope_digest="d", request={"gpu": "gpu"},
                                     deadline=DEADLINE_PAST)
        document = ledger._leases_document()[1]
        self.assertEqual(len(document["leases"]), 0)
        document["pending_creates"][0]["created_at"] = "2026-09-01T00:00:00+00:00"
        ledger._save(ledger.leases_path(), document)
        infra = FakeInfra(workers=[worker_record(
            worker_id="w-1", name=intent["name_prefix"] + "fixture")])

        report = reaper.reap(infra, execute=True)

        self.assertEqual(report["scopes"][0]["create_intents"]["redeemed"], ["w-1"])
        self.assertEqual(infra.stop_calls, ["w-1"])
        self.assertEqual(ledger.all_leases()[0]["state"], "stopped")

    def test_an_unmatched_intent_is_abandoned_only_after_it_cannot_be_in_flight(self):
        ledger.begin_create("TR-0007", purpose="job", envelope_digest="d",
                            request={"gpu": "gpu"}, deadline=DEADLINE_FUTURE)
        document = ledger._leases_document()[1]
        document["pending_creates"][0]["created_at"] = "2026-09-01T00:00:00+00:00"
        ledger._save(ledger.leases_path(), document)
        infra = FakeInfra(workers=[])

        report = reaper.reap(infra, execute=True,
                             now=state_module.parse_timestamp("2026-09-01T02:00:00+00:00"))

        self.assertEqual(len(report["scopes"][0]["create_intents"]["abandoned"]), 1)
        self.assertEqual(ledger.pending_creates("TR-0007"), [])

    def test_a_recent_unmatched_intent_is_left_alone(self):
        ledger.begin_create("TR-0007", purpose="job", envelope_digest="d",
                            request={"gpu": "gpu"}, deadline=DEADLINE_FUTURE)
        infra = FakeInfra(workers=[])

        reaper.reap(infra, execute=True,
                    now=state_module.parse_timestamp("2026-09-01T00:05:00+00:00"))

        self.assertEqual(len(ledger.pending_creates("TR-0007")), 1)


class ReaperIndependenceTests(ReaperTestCase):
    def test_a_separate_process_enforces_the_deadline_with_no_session_alive(self):
        """The real crash-independence proof: a new process, state read from disk only.

        Nothing in this test process holds the lease, and no OMP session exists. The
        reaper is launched as its own OS process against an on-disk lease document and a
        recording control plane, and the control-plane log — not the child's word for it —
        says the Pod was stopped.
        """

        self.lease("w-1", deadline=DEADLINE_PAST)
        self.observe("w-1", at="2026-09-01T00:30:00+00:00")
        responses = {
            "worker list": [worker_record(worker_id="w-1")],
            "job list": [],
        }
        checkout, _cli, log_path = self.fake_control_plane(responses=responses)
        environment = dict(os.environ)
        environment.update({
            "WAVCSE_RESEARCH_STATE": os.environ["WAVCSE_RESEARCH_STATE"],
            "WAVCSE_INFRA_CHECKOUT": checkout,
            "WAVCSE_REPO_ROOT": os.environ["WAVCSE_REPO_ROOT"],
            "PATH": os.path.dirname(sys.executable) + os.pathsep +
                    environment.get("PATH", ""),
        })
        environment.pop("WAVCSE_INFRA_CLI", None)

        completed = subprocess.run(
            [sys.executable, "-m", "improvements.compute", "reap", "--execute", "--json"],
            cwd=REPO_ROOT, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, universal_newlines=True, timeout=120,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        with open(log_path, "r", encoding="utf-8") as handle:
            logged = handle.read()
        self.assertIn("worker stop w-1", logged)
        self.assertNotIn("worker create", logged)
        self.assertNotIn("worker destroy", logged)
        self.assertNotIn("volume", logged)
        lease = ledger.all_leases()[0]
        self.assertEqual(lease["state"], "stopped")


class ReaperReportingTests(ReaperTestCase):
    def test_the_report_records_what_was_done(self):
        self.lease("w-1", deadline=DEADLINE_PAST)
        infra = FakeInfra(workers=[worker_record(worker_id="w-1")])

        reaper.reap(infra, execute=True)

        document = state_module.read_json(reaper.report_path())
        self.assertTrue(document["executed"])
        self.assertEqual(document["scopes"][0]["stopped"], ["w-1"])
        kinds = [event.get("kind") for event in state_module.read_events()]
        self.assertIn("reaper-run", kinds)
        self.assertIn("worker-stopped", kinds)

    def test_summary_names_every_scope_and_the_stop(self):
        self.lease("w-1", deadline=DEADLINE_PAST)
        infra = FakeInfra(workers=[worker_record(worker_id="w-1")])

        lines = reaper.summarize(reaper.reap(infra, execute=True))

        self.assertTrue(any("TR-0007" in line and "w-1" in line for line in lines))

    def test_the_scope_lock_refuses_a_second_holder(self):
        with state_module.scope_lock("TR-0007"):
            with self.assertRaises(BusyError):
                with state_module.scope_lock("TR-0007"):
                    pass


class ReaperUnitTests(ComputeTestCase):
    def setUp(self):
        super(ReaperUnitTests, self).setUp()
        from improvements.compute import reaper_units

        self.units_module = reaper_units
        self.units = reaper_units.build_units(
            repo_root="/srv/wavCSE",
            command_argv=["/usr/bin/uv", "run", "--locked", "python", "-m",
                          "improvements.compute", "reap", "--execute"],
            state_root="/state/arc",
            infra_checkout="/srv/wavcse-infra",
            interval_seconds=300,
        )

    def test_the_service_stops_and_never_destroys(self):
        service = self.units[self.units_module.SERVICE_NAME]
        self.assertIn("ExecStart=/usr/bin/uv run --locked python -m "
                      "improvements.compute reap --execute", service)
        self.assertNotIn("destroy", service)
        self.assertNotIn("--scope", service)
        self.assertIn("Environment=WAVCSE_RESEARCH_STATE=/state/arc", service)
        self.assertIn("Environment=WAVCSE_INFRA_CHECKOUT=/srv/wavcse-infra", service)

    def test_the_timer_survives_a_reboot(self):
        timer = self.units[self.units_module.TIMER_NAME]
        self.assertIn("Persistent=true", timer)
        self.assertIn("OnBootSec=", timer)
        self.assertIn("OnUnitActiveSec=300", timer)
        self.assertIn("WantedBy=timers.target", timer)

    def test_installing_twice_writes_the_same_bytes_and_reports_no_change(self):
        target = os.path.join(self.home, "units")
        first = self.units_module.install(self.units, target)
        second = self.units_module.install(self.units, target)

        self.assertTrue(first["changed"])
        self.assertFalse(second["changed"])
        self.assertEqual(second["written"], [])
        self.assertEqual(sorted(os.listdir(target)),
                         [self.units_module.SERVICE_NAME, self.units_module.TIMER_NAME])

    def test_a_dry_run_install_writes_nothing(self):
        target = os.path.join(self.home, "units")

        result = self.units_module.install(self.units, target, dry_run=True)

        self.assertTrue(result["changed"])
        self.assertFalse(os.path.exists(target))

    def test_installing_never_invokes_systemctl(self):
        """Describing the timer must not be able to start it."""

        target = os.path.join(self.home, "units")
        calls = []

        def recorder(argv):
            calls.append(argv)
            raise AssertionError("install must never reach systemd")

        original = self.units_module._run
        self.units_module._run = recorder
        try:
            self.units_module.install(self.units, target)
        finally:
            self.units_module._run = original
        self.assertEqual(calls, [])

    def test_enabling_is_a_separate_explicit_step(self):
        """`enable` is the only path that touches systemd, and it says what it ran."""

        class Completed(object):
            returncode = 0
            stdout = ""
            stderr = ""

        calls = []
        result = self.units_module.enable(
            user=True, runner=lambda argv: (calls.append(argv), Completed())[1]
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1:], ["--user", "enable", "--now",
                                        self.units_module.TIMER_NAME])
        self.assertTrue(result["enabled"])

    def test_enabling_reports_a_systemctl_failure_instead_of_assuming_success(self):
        class Failed(object):
            returncode = 1
            stdout = ""
            stderr = "Unit not found"

        with self.assertRaises(Exception) as caught:
            self.units_module.enable(runner=lambda argv: Failed())

        self.assertIn("Unit not found", str(caught.exception))
