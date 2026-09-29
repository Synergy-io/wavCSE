"""Read-only guarantees: status reconciles without ever writing or spending."""

import os
import unittest

from improvements.compute import state as state_module
from improvements.compute import status as status_module
from improvements.compute.tests.fakes import ComputeTestCase, FakeInfra, worker_record


def tree(root):
    """Every path under a directory, relative to it."""

    found = set()
    for directory, dirs, files in os.walk(root):
        for name in dirs + files:
            found.add(os.path.relpath(os.path.join(directory, name), root))
    return found


class StatusTests(ComputeTestCase):
    def test_status_reports_an_absent_authorization_without_crashing(self):
        self.make_repo()
        picture = status_module.build("TR-0007", infra=FakeInfra())
        self.assertFalse(picture["envelope"]["present"])
        self.assertEqual(picture["blocked"]["class"], "AUTHORIZATION")
        action = status_module.next_action(picture)
        self.assertEqual(action["action"], "stop")

    def test_status_reports_the_scope_picture_from_fixtures(self):
        self.make_repo()
        self.write_envelope("TR-0007")
        infra = FakeInfra(workers=[worker_record()])
        picture = status_module.build("TR-0007", infra=infra, jobs=[])
        self.assertTrue(picture["envelope"]["present"])
        self.assertTrue(picture["spend"]["accounting_bounded"])
        self.assertEqual(len(picture["spend"]["live_workers"]), 1)
        self.assertIn("estimated_spend_usd", picture["spend"])

    def test_status_is_read_only(self):
        self.make_repo()
        self.write_envelope("TR-0007")
        infra = FakeInfra(workers=[worker_record()])
        before_state = tree(state_module.state_root())
        before_repo = tree(os.environ["WAVCSE_REPO_ROOT"])
        for _ in range(2):
            status_module.build("TR-0007", infra=infra, jobs=[])
        self.assertEqual(tree(state_module.state_root()), before_state)
        self.assertEqual(tree(os.environ["WAVCSE_REPO_ROOT"]), before_repo)

    def test_status_does_not_provision_submit_or_destroy(self):
        self.make_repo()
        self.write_envelope("TR-0007")
        infra = FakeInfra(workers=[worker_record()])
        status_module.build("TR-0007", infra=infra, jobs=[])
        for forbidden in ("worker_create", "job_submit", "worker_stop",
                          "worker_destroy", "worker_bootstrap"):
            self.assertEqual(infra.count(forbidden), 0, forbidden)

    def test_open_jobs_put_the_scope_in_monitoring(self):
        self.make_repo()
        self.write_envelope("TR-0007")
        infra = FakeInfra(workers=[worker_record()])
        jobs = [{
            "job_id": "job-1", "state": "RUNNING", "worker_id": "w-1",
            "name": "TR-0007__screen__mssl__s42",
            "spec": {"tracking": {"metadata": {"scope": "TR-0007"}}},
        }]
        picture = status_module.build("TR-0007", infra=infra, jobs=jobs)
        self.assertEqual(len(picture["open_jobs"]), 1)
        self.assertEqual(picture["blocked"]["class"], "MONITORING")

    def test_unknown_spend_blocks_new_spend(self):
        self.make_repo()
        self.write_envelope("TR-0007")
        infra = FakeInfra(workers=[worker_record(hourly=None)])
        picture = status_module.build("TR-0007", infra=infra, jobs=[])
        self.assertFalse(picture["spend"]["accounting_bounded"])
        self.assertEqual(picture["blocked"]["class"], "COST")


if __name__ == "__main__":
    unittest.main()
