"""The ledger: append-only, restart-safe, tolerant of a torn last line."""

import json
import os
import shutil
import sys
import tempfile
import unittest

from improvements.sweep import ledger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class LedgerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.path = ledger.path_for(self.tmp)

    def test_missing_file_reads_as_empty(self):
        self.assertEqual(ledger.read(self.path), [])

    def test_append_then_read_round_trips(self):
        ledger.append(self.path, {"job_key": "a", "state": ledger.LAUNCHING})
        ledger.append(self.path, {"job_key": "a", "state": ledger.SUCCEEDED})
        self.assertEqual(len(ledger.read(self.path)), 2)
        self.assertEqual(ledger.index(ledger.read(self.path))["a"]["state"],
                         ledger.SUCCEEDED)

    def test_index_keeps_the_latest_record_per_key(self):
        ledger.append(self.path, {"job_key": "a", "state": ledger.LAUNCHING, "n": 1})
        ledger.append(self.path, {"job_key": "b", "state": ledger.FAILED})
        ledger.append(self.path, {"job_key": "a", "state": ledger.ORPHANED, "n": 2})
        latest = ledger.index(ledger.read(self.path))
        self.assertEqual(latest["a"]["n"], 2)
        self.assertEqual(latest["b"]["state"], ledger.FAILED)

    def test_counts_use_the_latest_state_only(self):
        for state in (ledger.LAUNCHING, ledger.RUNNING, ledger.SUCCEEDED):
            ledger.append(self.path, {"job_key": "a", "state": state})
        ledger.append(self.path, {"job_key": "b", "state": ledger.FAILED})
        self.assertEqual(ledger.counts(ledger.read(self.path)),
                         {ledger.SUCCEEDED: 1, ledger.FAILED: 1})

    def test_unknown_state_is_refused_at_write_time(self):
        with self.assertRaises(ValueError):
            ledger.append(self.path, {"job_key": "a", "state": "vibes"})

    def test_torn_last_line_is_ignored_not_fatal(self):
        ledger.append(self.path, {"job_key": "a", "state": ledger.SUCCEEDED})
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write('{"job_key": "b", "state": "laun')
        records = ledger.read(self.path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["job_key"], "a")

    def test_pending_keys_excludes_terminal_states(self):
        ledger.append(self.path, {"job_key": "done", "state": ledger.SUCCEEDED})
        ledger.append(self.path, {"job_key": "failed", "state": ledger.FAILED})
        ledger.append(self.path, {"job_key": "live", "state": ledger.RUNNING})
        latest = ledger.index(ledger.read(self.path))
        self.assertEqual(ledger.pending_keys(latest, ["done", "failed", "live", "new"]),
                         ["live", "new"])

    def test_summary_is_written_as_json(self):
        path = ledger.write_summary(self.tmp, {"stage": "screen", "counts": {}})
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertEqual(document["stage"], "screen")


if __name__ == "__main__":
    unittest.main()
