"""Supervisor reaping and adoption, driven by stub processes.

The interesting logic is not "does subprocess.run work" but what the supervisor
records: a run that exits zero without writing an identity is *not* evidence, a
run whose process vanished is requeued rather than assumed either way, and a
child that is still alive is left alone. All of that is testable with a stub
that answers ``poll()``.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

from improvements.sweep import ledger, manifest, supervisor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layer_sweep_fixtures import make_study  # noqa: E402


class FakePopen(object):
    def __init__(self, code, pid=4242):
        self._code = code
        self.pid = pid

    def poll(self):
        return self._code


def _write_identity(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


class ReapTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.state = os.path.join(self.tmp, "sweep_state")
        os.makedirs(self.state)
        self.results = os.path.join(self.tmp, "results")
        self.checkpoints = os.path.join(self.tmp, "checkpoints")
        for directory in (self.results, self.checkpoints):
            os.makedirs(directory)
        with open(os.path.join(self.results, "ks_si_er.txt"), "w") as handle:
            handle.write("metrics\n")
        with open(os.path.join(self.checkpoints, "best.pth"), "wb") as handle:
            handle.write(b"weights")
        with open(os.path.join(self.checkpoints, "ignore.bin"), "wb") as handle:
            handle.write(b"nope")

    def _child(self, key, code, *, identity=True):
        identity_path = os.path.join(self.state, "identity", key + ".jsonl")
        if identity:
            _write_identity(identity_path, {
                "run_id": "run-" + key, "results_dir": self.results,
                "checkpoints_dir": self.checkpoints, "seed": 42,
            })
        return supervisor.Child(key, {"id": "all", "layers": "all",
                                      "group": "top layer dropping", "k": 2},
                                42, FakePopen(code),
                                os.path.join(self.state, "logs", key + ".log"),
                                identity_path, 0)

    def test_success_records_identity_and_hashes_evidence(self):
        child = self._child("ok", 0)
        running = supervisor._reap({"ok": child}, self.state, annotate=False)
        self.assertEqual(running, {})
        record = ledger.index(ledger.read(ledger.path_for(self.state)))["ok"]
        self.assertEqual(record["state"], ledger.SUCCEEDED)
        self.assertEqual(record["exit_code"], 0)
        self.assertEqual(record["identity"]["run_id"], "run-ok")
        names = {entry["path"].rsplit("/", 1)[-1] for entry in record["outputs"]}
        self.assertEqual(names, {"ks_si_er.txt", "best.pth"})
        self.assertTrue(all(len(entry["sha256"]) == 64 for entry in record["outputs"]))

    def test_group_and_k_are_recorded_for_each_run(self):
        child = self._child("grouped", 0)
        supervisor._reap({"grouped": child}, self.state, annotate=False)
        record = ledger.index(ledger.read(ledger.path_for(self.state)))["grouped"]
        self.assertEqual(record["combo"], "all")
        self.assertEqual(record["group"], "top layer dropping")
        self.assertEqual(record["k"], 2)

    def test_annotation_failure_does_not_invalidate_the_run(self):
        def exploding_factory():
            raise RuntimeError("tracking server unavailable")

        child = self._child("unreachable", 0)
        supervisor._reap({"unreachable": child}, self.state,
                         annotate=True, client_factory=exploding_factory)
        record = ledger.index(ledger.read(ledger.path_for(self.state)))["unreachable"]
        self.assertEqual(record["state"], ledger.SUCCEEDED)
        self.assertFalse(record["annotation"]["annotated"])
        self.assertIn("tracking server unavailable",
                      record["annotation"]["reason"])

    def test_zero_exit_without_identity_is_a_failure(self):
        child = self._child("silent", 0, identity=False)
        supervisor._reap({"silent": child}, self.state)
        record = ledger.index(ledger.read(ledger.path_for(self.state)))["silent"]
        self.assertEqual(record["state"], ledger.FAILED)
        self.assertNotEqual(record["exit_code"], 0)

    def test_nonzero_exit_is_a_failure(self):
        child = self._child("bad", 7)
        supervisor._reap({"bad": child}, self.state)
        record = ledger.index(ledger.read(ledger.path_for(self.state)))["bad"]
        self.assertEqual(record["state"], ledger.FAILED)
        self.assertEqual(record["exit_code"], 7)

    def test_live_child_is_kept_and_not_recorded(self):
        child = self._child("live", None)
        running = supervisor._reap({"live": child}, self.state)
        self.assertIn("live", running)
        self.assertEqual(ledger.read(ledger.path_for(self.state)), [])

    def test_hashing_can_be_switched_off(self):
        child = self._child("nohash", 0)
        supervisor._reap({"nohash": child}, self.state, hash_outputs=False,
                         annotate=False)
        record = ledger.index(ledger.read(ledger.path_for(self.state)))["nohash"]
        self.assertNotIn("outputs", record)

    def test_liveness_check_is_local_and_conservative(self):
        self.assertTrue(supervisor._alive(os.getpid()))
        self.assertFalse(supervisor._alive(999999999))
        self.assertFalse(supervisor._alive(None))


class AnnotationTest(unittest.TestCase):
    """The tag vocabulary a reader groups by, without a tracking server."""

    def test_tags_name_group_k_and_combo(self):
        tags = supervisor.annotation_tags({"id": "top-k2", "layers": "0-22",
                                           "group": "top layer dropping", "k": 2})
        self.assertEqual(tags["sweep_group"], "top layer dropping")
        self.assertEqual(tags["sweep_k"], "2")
        self.assertEqual(tags["sweep_combo"], "top-k2")
        self.assertEqual(tags["sweep_layers"], "0-22")

    def test_a_bare_combo_still_gets_a_combo_tag(self):
        tags = supervisor.annotation_tags({"id": None, "layers": "0-15",
                                           "group": None, "k": None})
        self.assertEqual(tags["sweep_combo"], "0-15")
        self.assertNotIn("sweep_group", tags)
        self.assertNotIn("sweep_k", tags)

    def test_missing_run_id_is_not_an_error(self):
        result = supervisor.annotate_run(None, {"id": "x", "layers": "0-15"})
        self.assertFalse(result["annotated"])


class AdoptTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.state = os.path.join(self.tmp, "sweep_state")
        os.makedirs(self.state)
        self.store = ledger.path_for(self.state)

    def test_dead_pid_is_orphaned_and_requeued(self):
        ledger.append(self.store, {"job_key": "gone", "state": ledger.RUNNING,
                                   "pid": 999999999, "combo": "all", "seed": 42})
        self.assertEqual(supervisor._adopt(self.state), ["gone"])
        record = ledger.index(ledger.read(self.store))["gone"]
        self.assertEqual(record["state"], ledger.ORPHANED)

    def test_live_pid_is_left_alone(self):
        ledger.append(self.store, {"job_key": "mine", "state": ledger.RUNNING,
                                   "pid": os.getpid(), "combo": "all", "seed": 42})
        self.assertEqual(supervisor._adopt(self.state), [])
        self.assertEqual(ledger.index(ledger.read(self.store))["mine"]["state"],
                         ledger.RUNNING)

    def test_terminal_records_are_never_touched(self):
        ledger.append(self.store, {"job_key": "done", "state": ledger.SUCCEEDED,
                                   "pid": 999999999})
        self.assertEqual(supervisor._adopt(self.state), [])


class OutstandingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.study = make_study(self.tmp)
        self.spec = manifest.load_spec(self.study["spec_path"])
        self.state = os.path.join(self.tmp, "sweep_state")
        os.makedirs(self.state)

    def test_unscheduled_stage_is_entirely_outstanding(self):
        jobs = supervisor._outstanding(self.spec, "screen", "c" * 40, self.state)
        self.assertEqual(len(jobs), 3)

    def test_terminal_jobs_are_not_outstanding(self):
        jobs = supervisor._outstanding(self.spec, "screen", "c" * 40, self.state)
        key, combo, seed = jobs[0]
        ledger.append(ledger.path_for(self.state), {
            "job_key": key, "state": ledger.SUCCEEDED, "combo": combo, "seed": seed})
        remaining = supervisor._outstanding(self.spec, "screen", "c" * 40, self.state)
        self.assertEqual(len(remaining), 2)
        self.assertNotIn(key, [item[0] for item in remaining])


if __name__ == "__main__":
    unittest.main()
