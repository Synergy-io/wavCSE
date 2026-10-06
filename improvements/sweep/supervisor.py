"""The pod-side loop: probe, admit, launch, reap, record.

This is the only process in the sweep that makes a decision. It runs on the
machine with the GPU and keeps exactly as many trainings alive as the box can
hold, recomputing that number from live headroom on every pass
(``scheduler.admit``), never from a fixed "-P N". When pressure rises -- VRAM
fills, RAM thins out, the machine gets loaded -- it stops admitting without
killing anything already running, because a half-trained run that is interrupted
costs more than a queue that drains a little later.

Three properties it deliberately has:

* **Restart-safe.** Every state change is appended to the ledger before the
  next one happens, so a supervisor that dies is restarted by re-running the
  same command; completed jobs are skipped, and a job recorded RUNNING whose
  process is gone is requeued and marked ORPHANED rather than silently lost.
* **Adopting, not owning.** A child is only ever one this supervisor spawned;
  liveness is checked with ``os.kill(pid, 0)`` on a PID it recorded itself, so
  other users' work on a shared box is invisible to it.
* **Bounded by wall clock, not by an envelope.** There is no authorization to
  enforce here (this sweep left the paid-compute contract on purpose), so the
  stop condition is ``policy.max_wall_seconds`` plus an operator's DRAIN flag.
"""

import argparse
import datetime
import hashlib
import os
import signal
import subprocess
import sys
import time

from improvements.run_identity import read_identity_file
from improvements.sweep import config_gen, ledger, manifest, resources, scheduler

DRAIN_FLAG = "DRAIN"

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_INCOMPLETE = 3


def annotation_tags(combo):
    """The tags a finished run carries so the sweep is sliceable in MLflow.

    The config already publishes ``research.group``/``research.k`` as *params*
    (the whole config is flattened into params) and the layer list as the run
    note, but tags are what ``search_runs`` can filter on -- so the two fields a
    reader groups by are re-stated as tags here. Pure, so the vocabulary is
    testable without a tracking server.
    """
    group = combo.get("group")
    k = combo.get("k")
    return {
        key: value for key, value in {
            "sweep_group": group,
            "sweep_k": None if k is None else str(k),
            "sweep_combo": manifest.combo_id(combo),
            "sweep_layers": combo.get("layers"),
        }.items() if value is not None
    }


def annotate_run(run_id, combo, tracking_uri=None, client_factory=None):
    """Apply the sweep's tags to a finished run; never fail the run for it.

    ``client_factory`` is a test seam: the failure path must be exercisable
    without an actual tracking server, because a test that waits on a network
    timeout is both slow and brittle.
    """
    tags = annotation_tags(combo)
    if not run_id or not tags:
        return {"annotated": False, "reason": "no run id or nothing to tag"}
    if client_factory is None:
        try:
            from mlflow.tracking import MlflowClient
        except ImportError as exc:  # pragma: no cover - environment problem
            return {"annotated": False, "reason": "mlflow unavailable: {}".format(exc)}

        def client_factory():
            return MlflowClient(tracking_uri=tracking_uri) if tracking_uri else MlflowClient()
    try:
        client = client_factory()
        for key, value in tags.items():
            client.set_tag(run_id, key, value)
    except Exception as exc:  # noqa: BLE001 - tagging must not invalidate evidence
        return {"annotated": False, "reason": "{}: {}".format(type(exc).__name__, exc)}
    return {"annotated": True, "tags": tags}


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _log(message):
    print("{} | {}".format(_utc(), message), flush=True)


def _alive(pid):
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def _hash_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_outputs(identity):
    """Hash the small evidence a run leaves behind (checkpoints + metric files)."""
    hashed = []
    for root_key in ("checkpoints_dir", "results_dir"):
        root = identity.get(root_key)
        if not root or not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if not name.endswith((".pth", ".txt")):
                continue
            path = os.path.join(root, name)
            if not os.path.isfile(path):
                continue
            hashed.append({"path": path, "sha256": _hash_file(path),
                           "size_bytes": os.path.getsize(path)})
    return hashed


class Child(object):
    """One training process this supervisor started and is responsible for."""

    __slots__ = ("key", "combo", "seed", "popen", "log_path", "identity_path",
                 "started_at", "started_monotonic", "device_index")

    def __init__(self, key, combo, seed, popen, log_path, identity_path,
                 device_index):
        self.key = key
        self.combo = combo
        self.seed = seed
        self.popen = popen
        self.log_path = log_path
        self.identity_path = identity_path
        self.started_at = _utc()
        self.started_monotonic = time.monotonic()
        self.device_index = device_index


def _launch(spec, stage, checkout, state_dir, combo, seed, commit, device_index):
    """Start one training and return its Child record (already in the ledger)."""
    combo_key = manifest.combo_id(combo)
    key = manifest.job_key(spec["study_id"], combo_key, seed, commit)
    identity_path = os.path.join(state_dir, "identity", key + ".jsonl")
    log_path = os.path.join(state_dir, "logs", key + ".log")
    for directory in (os.path.dirname(identity_path), os.path.dirname(log_path)):
        os.makedirs(directory, exist_ok=True)
    if os.path.exists(identity_path):
        os.remove(identity_path)

    config_rel = config_gen.relative_config_path(spec, stage, combo)
    argv = [
        sys.executable, "-m", "improvements.run_improvements",
        "--model", spec["run_model"],
        "--task_type", spec["task_type"],
        "--config", config_rel,
        "--device_index", str(device_index),
        "--seed", str(seed),
    ]
    environment = dict(os.environ)
    environment["ARC_RUN_IDENTITY_FILE"] = identity_path
    environment["PYTHONUNBUFFERED"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = str(device_index)

    handle = open(log_path, "ab")
    try:
        popen = subprocess.Popen(argv, cwd=checkout, env=environment,
                                 stdout=handle, stderr=subprocess.STDOUT)
    finally:
        handle.close()

    child = Child(key, combo, seed, popen, log_path, identity_path, device_index)
    ledger.append(ledger.path_for(state_dir), {
        "job_key": key, "state": ledger.LAUNCHING, "combo": combo_key,
        "group": combo.get("group"), "k": combo.get("k"),
        "layers": combo.get("layers"), "seed": seed,
        "stage": stage, "commit": commit, "pid": popen.pid,
        "argv": argv, "log_path": log_path, "identity_path": identity_path,
        "study_id": spec["study_id"], "sweep_id": spec["sweep_id"],
        "started_at": child.started_at,
    })
    _log("launched {} combo={} group={} k={} seed={} pid={}".format(
        key, combo_key, combo.get("group"), combo.get("k"), seed, popen.pid))
    return child


def _reap(children, state_dir, *, hash_outputs=True, annotate=True,
          tracking_uri=None, client_factory=None):
    """Collect finished children into the ledger; return the still-running map."""
    running = {}
    store = ledger.path_for(state_dir)
    for key, child in children.items():
        code = child.popen.poll()
        if code is None:
            running[key] = child
            continue
        records = read_identity_file(child.identity_path)
        identity = records[-1] if records else None
        wall = max(time.monotonic() - child.started_monotonic, 0.0)
        state = ledger.SUCCEEDED if code == 0 else ledger.FAILED
        if code == 0 and identity is None:
            # A zero exit with no identity means the run cannot be attributed to
            # a results directory, so it is not evidence and must not count as
            # success. The compute backend treats this the same way.
            state = ledger.FAILED
            code = 1
            _log("{} exited 0 but wrote no run identity: treating as failed"
                 .format(key))
        record = {
            "job_key": key, "state": state, "combo": manifest.combo_id(child.combo),
            "group": child.combo.get("group"), "k": child.combo.get("k"),
            "layers": child.combo.get("layers"),
            "seed": child.seed, "exit_code": code, "pid": child.popen.pid,
            "finished_at": _utc(), "wall_seconds": round(wall, 1),
            "identity": identity, "log_path": child.log_path,
        }
        if identity and hash_outputs:
            record["outputs"] = _hash_outputs(identity)
        if annotate and identity:
            record["annotation"] = annotate_run(
                identity.get("run_id"), child.combo, tracking_uri, client_factory)
        ledger.append(store, record)
        _log("{} {} (exit {}, {:.0f}s)".format(key, state, code, wall))
    return running


def _adopt(state_dir):
    """Requeue jobs a previous supervisor left RUNNING with a dead process."""
    store = ledger.path_for(state_dir)
    latest = ledger.index(ledger.read(store))
    adopted = []
    for key, record in latest.items():
        if record.get("state") not in (ledger.LAUNCHING, ledger.RUNNING):
            continue
        if _alive(record.get("pid")):
            continue
        ledger.append(store, {
            "job_key": key, "state": ledger.ORPHANED, "combo": record.get("combo"),
            "seed": record.get("seed"), "pid": record.get("pid"),
            "reason": "recorded pid is not running; requeued rather than assumed",
            "finished_at": _utc(),
        })
        adopted.append(key)
    if adopted:
        _log("orphaned and requeued: {}".format(", ".join(adopted)))
    return adopted


def _outstanding(spec, stage, commit, state_dir):
    """(key, combo, seed) still owed by this stage, in declaration order."""
    store = ledger.path_for(state_dir)
    latest = ledger.index(ledger.read(store))
    jobs = []
    for combo, seed in manifest.stage_jobs(spec, stage):
        key = manifest.job_key(spec["study_id"], manifest.combo_id(combo), seed, commit)
        record = latest.get(key)
        if record and record.get("state") in ledger.TERMINAL:
            continue
        jobs.append((key, combo, seed))
    return jobs


def run(spec, stage, checkout, state_dir, *, commit, dry_run=False, once=False,
        hash_outputs=True, annotate=True, tracking_uri=None, client_factory=None,
        max_seconds=None, interval=None, should_stop=None):
    """Drive one stage to completion (or one pass, when ``once``)."""
    policy = spec["policy"]
    interval = float(interval if interval is not None else policy["probe_interval_s"])
    budget = float(max_seconds if max_seconds is not None else policy["max_wall_seconds"])
    drain_path = os.path.join(state_dir, "DRAIN")
    os.makedirs(state_dir, exist_ok=True)

    _adopt(state_dir)
    queue = _outstanding(spec, stage, commit, state_dir)
    if not queue:
        _log("stage {} has nothing outstanding".format(stage))
        _write_summary(spec, stage, commit, state_dir, started=time.monotonic())
        return EXIT_OK

    if dry_run:
        snapshot = resources.probe(checkout)
        decision = scheduler.admit(snapshot, 0, policy)
        _log("dry-run: {} job(s) owed; {}".format(len(queue), resources.describe(snapshot)))
        _log("dry-run: first admission would {} ({})".format(
            "proceed" if decision.admit else "hold", decision.reason))
        for key, combo, seed in queue:
            _log("  would run {} combo={} seed={} config={}".format(
                key, combo, seed, config_gen.relative_config_path(spec, stage, combo)))
        return EXIT_OK

    started = time.monotonic()
    children = {}
    draining = False
    last_reason = None
    while True:
        children = _reap(children, state_dir, hash_outputs=hash_outputs,
                         annotate=annotate, tracking_uri=tracking_uri,
                         client_factory=client_factory)

        if os.path.exists(drain_path):
            if not draining:
                _log("DRAIN flag present: admitting nothing further")
            draining = True
        if should_stop is not None and should_stop() and not draining:
            _log("stop requested: draining (children are left to finish)")
            draining = True
        elapsed = time.monotonic() - started
        if elapsed > budget:
            if not draining:
                _log("wall-clock budget {:.0f}s reached: draining".format(budget))
            draining = True

        if queue and not draining:
            snapshot = resources.probe(checkout)
            decision = scheduler.admit(snapshot, len(children), policy)
            if decision.reason != last_reason:
                _log("admission: {} ({})".format(
                    "admit" if decision.admit else "hold", decision.reason))
                last_reason = decision.reason
            if decision.admit:
                key, combo, seed = queue.pop(0)
                children[key] = _launch(spec, stage, checkout, state_dir, combo,
                                        seed, commit, policy["device_index"])
        if not queue and not children:
            break
        if draining and not children:
            # Nothing admits any more and nothing is running: the stage is as
            # finished as it is going to get, so stop rather than sleep forever.
            break
        if once:
            break
        time.sleep(interval)

    if once:
        _log("one pass complete: {} running, {} queued".format(len(children), len(queue)))
        _write_summary(spec, stage, commit, state_dir, started=started)
        return EXIT_OK

    latest = ledger.index(ledger.read(ledger.path_for(state_dir)))
    outstanding = _outstanding(spec, stage, commit, state_dir)
    summary = _write_summary(spec, stage, commit, state_dir, started=started)
    if outstanding:
        _log("stopped with {} job(s) outstanding: {}".format(
            len(outstanding), ", ".join(key for key, _c, _s in outstanding)))
        return EXIT_INCOMPLETE
    failed = [record["job_key"] for record in latest.values()
              if record.get("state") in (ledger.FAILED, ledger.ORPHANED)
              and record.get("combo") in manifest.stage_combos(spec, stage)]
    if failed:
        _log("stage finished with failures: {}".format(", ".join(sorted(failed))))
        return EXIT_INCOMPLETE
    _log("stage {} complete: {} succeeded".format(
        stage, summary["counts"].get(ledger.SUCCEEDED, 0)))
    return EXIT_OK


def _write_summary(spec, stage, commit, state_dir, *, started):
    records = ledger.read(ledger.path_for(state_dir))
    latest = ledger.index(records)
    stage_keys = {manifest.job_key(spec["study_id"], manifest.combo_id(combo), seed,
                                   commit)
                  for combo, seed in manifest.stage_jobs(spec, stage)}
    stage_records = [latest[key] for key in sorted(stage_keys) if key in latest]
    summary = {
        "schema_version": 1,
        "study_id": spec["study_id"],
        "sweep_id": spec["sweep_id"],
        "stage": stage,
        "commit": commit,
        "policy": spec["policy"],
        "written_at": _utc(),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "counts": ledger.counts([record for record in stage_records]),
        "runs": stage_records,
    }
    path = ledger.write_summary(state_dir, summary)
    _log("summary written to {}".format(path))
    return summary


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run a layer-combination sweep on this machine, as many at once as fit")
    parser.add_argument("--spec", required=True, help="path of the sweep.json")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--checkout", default=None,
                        help="repository checkout to run from (default: cwd)")
    parser.add_argument("--state-dir", default=None,
                        help="default: <study>/sweep_state")
    parser.add_argument("--commit", default=None,
                        help="exact commit to bind (default: HEAD of the checkout)")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="override policy.max_wall_seconds")
    parser.add_argument("--interval", type=float, default=None,
                        help="override policy.probe_interval_s")
    parser.add_argument("--once", action="store_true",
                        help="one admission pass, then return with children still running")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would run and why, launching nothing")
    parser.add_argument("--no-hash", action="store_true",
                        help="do not hash each run's checkpoints and metric files")
    parser.add_argument("--no-annotate", action="store_true",
                        help="do not tag finished runs with group/k/combo in MLflow")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="permit a dirty checkout (smoke runs only)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    checkout = os.path.abspath(args.checkout or os.getcwd())
    try:
        spec = manifest.load_spec(args.spec)
    except manifest.SpecError as exc:
        print("sweep spec error: {}".format(exc), file=sys.stderr)
        return EXIT_USAGE

    state_dir = os.path.abspath(args.state_dir or manifest.state_dir(spec))
    if args.commit:
        commit = args.commit.strip().lower()
    else:
        state = manifest.git_state(checkout, spec["_study_dir"])
        if state["dirty"] and not args.allow_dirty:
            print("refusing to run while the sweep's own inputs are uncommitted ({} "
                  "entr{}); commit the generated configs first, or pass "
                  "--allow-dirty for a smoke run".format(len(state["dirty_entries"]),
                                                          "y" if len(state["dirty_entries"]) == 1 else "ies"),
                  file=sys.stderr)
            for line in state["dirty_entries"][:10]:
                print("  {}".format(line), file=sys.stderr)
            return EXIT_USAGE
        commit = state["head"]
    if args.stage not in spec["stages"]:
        print("unknown stage {!r}; the spec declares {}".format(
            args.stage, ", ".join(sorted(spec["stages"]))), file=sys.stderr)
        return EXIT_USAGE

    stopping = {"flag": False}

    def _stop(signum, _frame):
        # Drain rather than kill: a half-trained run costs more than a queue
        # that empties a little later, and the DRAIN flag survives this process
        # so a restart does not silently resume admitting.
        stopping["flag"] = True
        try:
            os.makedirs(state_dir, exist_ok=True)
            with open(os.path.join(state_dir, "DRAIN"), "w", encoding="utf-8") as handle:
                handle.write("stop requested by signal {}\n".format(signum))
        except OSError:
            pass
        _log("signal {} received: draining (children are left to finish)".format(signum))

    for signal_name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, signal_name):
            signal.signal(getattr(signal, signal_name), _stop)

    _log("sweep {} stage {} at commit {} on {}".format(
        spec["sweep_id"], args.stage, commit[:12], checkout))
    try:
        return run(spec, args.stage, checkout, state_dir, commit=commit,
                   dry_run=args.dry_run, once=args.once,
                   hash_outputs=not args.no_hash,
                   annotate=not args.no_annotate,
                   tracking_uri=(spec.get("mlflow") or {}).get("tracking_uri"),
                   max_seconds=args.max_seconds,
                   interval=args.interval,
                   should_stop=lambda: stopping["flag"])
    finally:
        if stopping["flag"]:
            print("{} | drained by signal; re-run the same command to resume"
                  .format(_utc()), flush=True)


if __name__ == "__main__":
    sys.exit(main())
