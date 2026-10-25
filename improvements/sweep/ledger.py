"""Append-only run records, so a restarted supervisor resumes from proof.

A sweep is long enough that a supervisor process will be interrupted -- an ssh
session drops, a pod is preempted, someone hits Ctrl-C. The ledger is what makes
the answer to "what has already run?" a file read instead of an inference from
whatever directories happen to exist. It is append-only JSONL: the newest record
for a job key wins, a partially written last line is ignored rather than fatal,
and nothing is ever rewritten, so the history of one run (launched, orphaned,
relaunched, succeeded) stays visible.
"""

import json
import os

# A job's state moves forward only. The set is closed so a reader never has to
# guess whether an unknown state is good news or bad news.
PENDING = "pending"
LAUNCHING = "launching"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
ORPHANED = "orphaned"
SKIPPED = "skipped"

STATES = (PENDING, LAUNCHING, RUNNING, SUCCEEDED, FAILED, ORPHANED, SKIPPED)
TERMINAL = (SUCCEEDED, FAILED, ORPHANED, SKIPPED)


def path_for(state_dir):
    return os.path.join(state_dir, "runs.jsonl")


def read(path):
    """Every record, oldest first; unparsable lines are skipped, not fatal."""
    records = []
    if not path or not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    return records


def append(path, record):
    """Append one record and flush it to the platform before returning."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    if record.get("state") not in STATES:
        raise ValueError("unknown ledger state: {!r}".format(record.get("state")))
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return record


def index(records):
    """Latest record per job key (later records supersede earlier ones)."""
    latest = {}
    for record in records:
        key = record.get("job_key")
        if key:
            latest[key] = record
    return latest


def counts(records):
    """Counts by state, over the latest record of each job."""
    totals = {}
    for record in index(records).values():
        state = record.get("state") or "unknown"
        totals[state] = totals.get(state, 0) + 1
    return totals


def pending_keys(latest, keys):
    """The job keys that still need to run, given the ledger's view."""
    outstanding = []
    for key in keys:
        record = latest.get(key)
        if record is None or record.get("state") not in TERMINAL:
            outstanding.append(key)
    return outstanding


def write_summary(state_dir, summary):
    """Write the sweep summary beside the ledger (overwritten: it is a view)."""
    os.makedirs(state_dir, exist_ok=True)
    path = os.path.join(state_dir, "SWEEP.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, sort_keys=True, indent=2)
        handle.write("\n")
    return path
