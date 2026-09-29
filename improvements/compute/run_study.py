"""Stage orchestration: an explicit, resumable state machine for one stage.

The cycle is a sequence of named transitions rather than one long function,
because every transition has to be safe to repeat after a crash:

    reconcile  -> adopt any provider job this controller already submitted
    submit     -> one deterministic spec per (arm, seed), envelope-checked
    monitor    -> a bounded read of provider state; never a resubmission
    collect    -> outputs verified from the job record, never from a log line
    finish     -> stop or destroy scope compute per the envelope's stop policy

Everything the machine needs lives in the controller-local run ledger, so
restarting the agent resumes from proven state rather than conversation memory.
Submission is keyed by a deterministic job identity, so a lost acknowledgement
is reconciled by matching the committed spec against the provider's job list
instead of submitting a second job.
"""

import os
import re
import subprocess
from decimal import Decimal

from improvements.compute import envelope as envelope_module
from improvements.compute import failures, jobspec, ledger, worker as worker_module
from improvements.compute import state as state_module
from improvements.compute.errors import (
    ArtifactIntegrityError,
    AuthorizationError,
    CostError,
    ImplementationBugError,
    ReconcilableError,
    RepositoryConflictError,
    UsageError,
)

OPEN_STATES = ("PENDING", "PREPARING", "RUNNING")
TERMINAL_STATES = ("SUCCEEDED", "FAILED", "CANCELLED")

PENDING = "pending"
SUBMITTED = "submitted"
MONITORING = "monitoring"
SUCCEEDED = "succeeded"
FAILED = "failed"
COLLECTED = "collected"


def record_path(scope):
    return state_module.path_for("runs", scope)


def load_record(scope):
    return state_module.read_json(record_path(scope), default=None) or {
        "schema_version": 1,
        "scope": scope,
        "entries": {},
    }


def save_record(scope, record):
    record["updated_at"] = state_module.isoformat(state_module.utc_now())
    state_module.write_json_atomically(record_path(scope), record)
    return record


def spec_path(scope, spec):
    return state_module.path_for(
        "specs", "{}-{}.json".format(scope, spec["name"])
    )


def plan_jobs(plan, stage):
    """The pre-registered (arm, seed) pairs for one stage."""

    seeds = jobspec.stage_seeds(plan, stage)
    return [(arm["arm"], seed) for arm in plan["arms"] for seed in seeds]


def _entry(record, key, *, arm, seed, plan, stage):
    entries = record.setdefault("entries", {})
    entry = entries.get(key)
    if entry is None:
        entry = {
            "job_key": key,
            "arm": arm,
            "seed": seed,
            "study": plan["study"],
            "stage": stage,
            "state": PENDING,
            "attempts": 0,
            "job_id": None,
            "worker_id": None,
            "spec_digest": None,
            "failure_class": None,
            "outputs_verified": False,
            "created_at": state_module.isoformat(state_module.utc_now()),
        }
        entries[key] = entry
    return entry


def _viewport(scope, *, repo_root=None, infra=None):
    """Load the envelope and record the digest this run acts under."""

    view = envelope_module.load(scope, repo_root)
    busy = ledger.scope_is_busy(scope)
    if infra is not None:
        busy = busy or any(
            ((job.get("spec") or {}).get("tracking") or {}).get("metadata", {}).get("scope") == scope
            and str(job.get("state") or "").upper() in OPEN_STATES
            for job in infra.job_list())
    envelope_module.record_snapshot(view, busy=busy)
    return view


def _match_provider_job(jobs, spec, excluded=()):
    for job in jobs:
        if job.get("job_id") not in excluded and jobspec.looks_like_duplicate(job, spec):
            return job
    return None


def reconcile(scope, plan, stage, *, infra, record=None):
    """Adopt provider jobs this scope already submitted, then refresh each entry."""

    record = record if record is not None else load_record(scope)
    commit = jobspec.git_state()["head"]
    jobs = infra.job_list()
    adopted = []

    for arm, seed in plan_jobs(plan, stage):
        key = jobspec.job_key(scope, plan["study"], stage, arm, seed, commit)
        entry = _entry(record, key, arm=arm, seed=seed, plan=plan, stage=stage)
        if entry.get("job_id"):
            continue
        spec = _spec_for(plan, stage, arm, seed, scope, commit)
        match = _match_provider_job(jobs, spec, entry.get("previous_job_ids") or ())
        if match is not None:
            entry["job_id"] = match.get("job_id")
            entry["submission_pending"] = False
            entry["worker_id"] = match.get("worker_id")
            entry["state"] = SUBMITTED
            entry["adopted_at"] = state_module.isoformat(state_module.utc_now())
            entry["spec_digest"] = jobspec.verification_digest(spec)
            adopted.append(entry["job_id"])
            state_module.append_event(
                {"scope": scope, "action": "job-adopted", "job_key": key,
                 "job_id": entry["job_id"],
                 "reason": "provider already holds this deterministic job"},
                kind="job-adopted",
            )

    for entry in record["entries"].values():
        if not entry.get("job_id") or entry.get("state") == COLLECTED:
            continue
        if (entry.get("stage") != stage or
                entry.get("job_key", "").rsplit("::", 1)[-1] != commit):
            if entry.get("state") not in (COLLECTED, FAILED):
                raise RepositoryConflictError(
                    "an earlier commit or stage still has an open job; reconcile it before "
                    "advancing the current plan")
            continue
        record_view = infra.job_status(entry["job_id"])
        if record_view.get("job_id") != entry["job_id"]:
            raise ArtifactIntegrityError("infra returned a different job identity for {}"
                                         .format(entry["job_id"]))
        expected = _spec_for(plan, entry["stage"], entry["arm"], entry["seed"],
                             scope, entry["job_key"].rsplit("::", 1)[-1])
        if not jobspec.looks_like_duplicate(record_view, expected):
            raise ArtifactIntegrityError("job {} no longer matches its committed study identity"
                                         .format(entry["job_id"]))
        if str(record_view.get("state") or "").upper() == SUCCEEDED.upper() and \
                record_view.get("exit_code") != 0:
            raise ArtifactIntegrityError("job {} claims success without exit code zero"
                                         .format(entry["job_id"]))
        entry["state"] = str(record_view.get("state") or entry["state"]).lower()
        entry["worker_id"] = record_view.get("worker_id") or entry.get("worker_id")
        entry["exit_code"] = record_view.get("exit_code")
        entry["worker_absent"] = bool(record_view.get("worker_absent"))
        entry["remote_status"] = record_view.get("remote_status")
        entry["state_reason"] = record_view.get("state_reason")
        entry["outputs"] = record_view.get("outputs")
        entry["finished_at"] = record_view.get("finished_at") or entry.get("finished_at")
    save_record(scope, record)
    return record, adopted


def _spec_for(plan, stage, arm, seed, scope, commit, inputs=()):
    """A spec used only to match an already-submitted job.

    The envelope digest is a marker, not a real grant: this spec is never
    written or submitted, and duplicate matching does not compare digests.
    """

    return jobspec.build_spec(
        plan, stage=stage, arm=arm, seed=seed, commit=commit, scope=scope,
        envelope_digest="match-only", inputs=inputs,
    )


def _inputs_for(plan, repo_root=None):
    inputs_file = plan.get("inputs_file")
    if not inputs_file:
        return []
    from improvements.compute import artifacts

    root = repo_root or jobspec.repo_root()
    root = os.path.realpath(root)
    path = os.path.realpath(os.path.join(root, inputs_file))
    if not path.startswith(root + os.sep):
        raise ArtifactIntegrityError("inputs_file must be inside the committed checkout")
    relative = os.path.relpath(path, root)
    tracked = subprocess.run(
        ["git", "-C", root, "ls-files", "--error-unmatch", "--", relative],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        universal_newlines=True,
    )
    if tracked.returncode != 0:
        raise ArtifactIntegrityError("inputs_file {} is not tracked by Git".format(relative))
    if not os.path.exists(path):
        raise ArtifactIntegrityError(
            "the plan declares inputs_file {} but it does not exist".format(inputs_file)
        )
    return artifacts.to_job_inputs(artifacts.load(path))


def submit_pending(scope, plan, stage, *, infra, view, record, dry_run=False,
                   require_clean=True):
    """Submit every unsubmitted (arm, seed) in this stage, inside the envelope.

    A real submission requires a commit-clean tree, because the worker checks out
    the commit and anything uncommitted would silently not be what ran. A dry run
    instead reports the tree state, so a plan can be previewed before committing.
    """

    commit = jobspec.git_state()["head"]
    tree = jobspec.git_state()
    if require_clean and not dry_run:
        jobspec.require_committed_experiment(expected_commit=commit)
    inputs = _inputs_for(plan)
    submitted = []
    planned = []
    warnings = []
    if not tree["clean"]:
        warnings.append(
            "the working tree has uncommitted changes ({}); a recorded job cannot "
            "be submitted from this state".format(", ".join(tree["dirty"][:5]))
        )

    for arm, seed in plan_jobs(plan, stage):
        key = jobspec.job_key(scope, plan["study"], stage, arm, seed, commit)
        entry = _entry(record, key, arm=arm, seed=seed, plan=plan, stage=stage)
        if entry.get("job_id") or entry.get("state") == COLLECTED:
            continue

        if entry.get("submission_pending"):
            raise ReconcilableError(
                "submission for {} remains ambiguous; reconcile the infra job record "
                "before issuing another request".format(key), action="submit-job")

        decision = envelope_module.check(
            view, envelope_module.ACTION_SUBMIT_JOB,
            ledger.derive_spend(infra.worker_list(), scope).facts(),
        )
        if not decision.allowed:
            decision.require()

        spec = jobspec.build_spec(
            plan, stage=stage, arm=arm, seed=seed, commit=commit, scope=scope,
            envelope_digest=view.digest, inputs=inputs,
        )
        planned.append({
            "job_key": key,
            "name": spec["name"],
            "arm": arm,
            "seed": seed,
            "spec_digest": jobspec.verification_digest(spec),
            "envelope_digest": view.digest,
        })
        if dry_run:
            continue

        worker_id = entry.get("worker_id") or _worker_id_for(infra, scope)
        known_jobs = infra.job_list()
        match = _match_provider_job(known_jobs, spec, entry.get("previous_job_ids") or ())
        if match is not None:
            entry["job_id"] = match["job_id"]
            entry["worker_id"] = match.get("worker_id")
            entry["state"] = SUBMITTED
            entry["submission_pending"] = False
            entry["spec_digest"] = jobspec.verification_digest(spec)
            save_record(scope, record)
            submitted.append(entry["job_id"])
            break
        open_jobs = [job for job in known_jobs
                     if str(job.get("state") or "").upper() in OPEN_STATES
                     and (job.get("worker_id") == worker_id or
                          ((job.get("spec") or {}).get("tracking") or {}).get("metadata", {}).get("scope") == scope)]
        if open_jobs:
            raise ReconcilableError(
                "worker {} or scope {} already has an open job; reconcile it before "
                "submitting another seed".format(worker_id, scope), action="submit-job")

        path = spec_path(scope, spec)
        jobspec.write_spec(spec, path)
        entry["spec_digest"] = jobspec.verification_digest(spec)
        entry["worker_id"] = worker_id
        entry["submission_pending"] = True
        save_record(scope, record)
        result = infra.job_submit(path, entry["worker_id"])
        record_view = result.payload if isinstance(result.payload, dict) else None
        if record_view and record_view.get("job_id"):
            entry["job_id"] = record_view["job_id"]
            entry["submission_pending"] = False
            entry["state"] = SUBMITTED
            entry["attempts"] = int(entry.get("attempts", 0)) + 1
            entry["submitted_at"] = state_module.isoformat(state_module.utc_now())
            ledger.record_job(entry["worker_id"], job_key=key, job_id=entry["job_id"])
            state_module.append_event(
                {"scope": scope, "action": "job-submitted", "job_key": key,
                 "job_id": entry["job_id"], "worker_id": entry["worker_id"],
                 "spec_digest": entry["spec_digest"],
                 "envelope_digest": view.digest, "attempt": entry["attempts"]},
                kind="job-submitted",
            )
            submitted.append(entry["job_id"])
            break

        # Submission returned no usable record: the request may or may not have
        # landed. Never re-submit blindly; reconcile by the deterministic identity.
        jobs = infra.job_list()
        match = _match_provider_job(jobs, spec, entry.get("previous_job_ids") or ())
        if match is not None:
            entry["job_id"] = match.get("job_id")
            entry["submission_pending"] = False
            entry["worker_id"] = match.get("worker_id")
            entry["state"] = SUBMITTED
            submitted.append(entry["job_id"])
            break
        entry["failure_class"] = failures.TRANSIENT_INFRA
        save_record(scope, record)
        raise ReconcilableError(
            "job submit for {} returned no record and no provider job matches the "
            "deterministic identity; the spec is on disk at {} and the same job is "
            "never submitted twice. Re-run this transition to reconcile.".format(
                key, path
            ),
            action="submit-job",
        )

    if not dry_run:
        save_record(scope, record)
    return {"planned": planned, "submitted": submitted, "warnings": warnings}


def _worker_id_for(infra, scope):
    workers = infra.worker_list()
    candidates = [
        worker for worker in workers
        if ledger.worker_belongs_to(worker.get("name"), scope)
        and str(worker.get("state") or "").upper() not in ("DESTROYED", "TERMINATING")
    ]
    candidates.sort(key=lambda worker: str(worker.get("created_at") or ""), reverse=True)
    if not candidates:
        raise UsageError(
            "no worker is available for {}; run the worker-ensure transition first".format(
                scope
            )
        )
    return candidates[0].get("id")


def collect(scope, plan, stage, *, record, require_all=True):
    """Verify outputs from the job records and mark entries collected."""

    collected = []
    unverified = []
    commit = jobspec.git_state()["head"]
    for arm, seed in plan_jobs(plan, stage):
        key = jobspec.job_key(scope, plan["study"], stage, arm, seed, commit)
        for entry in record["entries"].values():
            if entry.get("job_key") != key:
                continue
            if entry.get("state") != SUCCEEDED:
                continue
            outputs = entry.get("outputs") or []
            expected = jobspec.declared_outputs(plan, jobspec.arm_by_name(plan, arm), seed)
            by_path = {}
            for output in outputs:
                by_path.setdefault(output.get("path"), []).append(output)
            missing = []
            for declared in expected:
                if not declared["required"]:
                    continue
                matches = by_path.get(declared["path"], [])
                if len(matches) != 1:
                    missing.append(declared)
                    continue
                output = matches[0]
                if (output.get("artifact") != declared["artifact"]
                        or output.get("persisted") is not True
                        or not isinstance(output.get("verified_size_bytes"), int)
                        or output["verified_size_bytes"] <= 0
                        or not re.fullmatch(r"[0-9a-f]{64}", str(output.get("sha256") or ""))):
                    missing.append(declared)
            if missing:
                entry["state"] = FAILED
                entry["failure_class"] = failures.ARTIFACT_INTEGRITY
                entry["note"] = "required outputs not verified: {}".format(
                    ", ".join(str(item.get("path")) for item in missing)
                )
                unverified.append(entry["job_key"])
                continue
            entry["state"] = COLLECTED
            entry["outputs_verified"] = True
            entry["collected_at"] = state_module.isoformat(state_module.utc_now())
            collected.append(entry["job_key"])
    save_record(scope, record)
    return {"collected": collected, "unverified": unverified}


def assess(record, *, stage=None, logs=None):
    """Classify terminal-but-uncollected entries and set retry dispositions."""

    outcomes = []
    for entry in record["entries"].values():
        if stage is not None and entry.get("stage") != stage:
            continue
        state = str(entry.get("state") or "")
        if state == COLLECTED:
            continue
        if state == SUCCEEDED:
            outcomes.append({"job_key": entry["job_key"], "state": state,
                             "class": failures.SCIENTIFIC_FAILURE, "retry": False,
                             "reason": "completed execution; the outcome is evidence"})
            continue
        if state in (SUBMITTED, MONITORING, PENDING, "preparing", "running") and entry.get("job_id"):
            outcomes.append({"job_key": entry["job_key"], "state": state,
                             "class": None, "retry": None})
            continue
        if state == PENDING and not entry.get("job_id"):
            outcomes.append({"job_key": entry["job_key"], "state": state,
                             "class": None, "retry": None})
            continue
        log_text = (logs or {}).get(entry.get("job_id"), "")
        classification = failures.assess_job_outcome(
            {"state": "FAILED", "exit_code": entry.get("exit_code"),
             "worker_absent": entry.get("worker_absent"),
             "remote_status": entry.get("remote_status")},
            log_text=log_text,
            outputs_verified=bool(entry.get("outputs_verified")),
        )
        entry["failure_class"] = classification.klass
        allowed, reason = failures.decide_retry(
            classification, int(entry.get("attempts", 0)),
            reconciled=bool(entry.get("adopted_at") or entry.get("job_id")),
        )
        entry["retry_allowed"] = allowed
        entry["retry_reason"] = reason
        if allowed:
            previous = list(entry.get("previous_job_ids") or [])
            if entry.get("job_id"):
                previous.append(entry["job_id"])
            entry["previous_job_ids"] = previous
            entry["state"] = PENDING
            entry["job_id"] = None
            state_module.append_event(
                {"scope": record.get("scope"), "action": "job-retry-scheduled",
                 "job_key": entry["job_key"], "class": classification.klass,
                 "attempts": entry.get("attempts"), "reason": reason},
                kind="job-retry-scheduled",
            )
        outcomes.append({
            "job_key": entry["job_key"], "state": entry["state"],
            "class": classification.klass, "retry": allowed, "reason": reason,
        })
    save_record(record.get("scope"), record)
    return outcomes


def finish(scope, plan, *, infra, view, record):
    """Stop or destroy scope compute once nothing useful is pending."""

    if not record["entries"]:
        return {"finished": False, "reason": "no verified study work is recorded"}
    pending = [
        entry for entry in record["entries"].values()
        if entry.get("state") != COLLECTED
    ]
    if pending:
        return {"finished": False, "reason": "{} entries are not collected".format(len(pending))}
    leased_ids = {lease.get("worker_id") for lease in ledger.active_leases(scope)}
    open_jobs = [job for job in infra.job_list()
                 if (job.get("worker_id") in leased_ids or
                     ((job.get("spec") or {}).get("tracking") or {}).get("metadata", {}).get("scope") == scope)
                 and str(job.get("state") or "").upper() in OPEN_STATES]
    if open_jobs:
        return {"finished": False, "reason": "{} scope jobs are still open".format(len(open_jobs))}
    stop_policy = view.envelope["stop_policy"]
    actions = []
    for lease in ledger.active_leases(scope):
        worker_id = lease.get("worker_id")
        if stop_policy.get("destroy_on_completion"):
            result = worker_module.destroy_worker(infra, worker_id, reason="stage complete")
            if result.returncode != 0:
                raise ReconcilableError("worker {} destroy failed; cleanup remains pending"
                                        .format(worker_id), action="destroy-worker")
            actions.append({"worker_id": worker_id, "action": "destroy"})
        elif int(stop_policy.get("retain_for_reuse_hours", 0)) == 0:
            result = worker_module.stop_worker(infra, worker_id, reason="stage complete")
            if result.returncode != 0:
                raise ReconcilableError("worker {} stop failed; cleanup remains pending"
                                        .format(worker_id), action="stop-worker")
            actions.append({"worker_id": worker_id, "action": "stop"})
    return {"finished": True, "actions": actions}


def advance(scope, plan, stage, *, infra, record=None, view=None, dry_run=False):
    """One explicit transition forward.

    Order matters and is deliberate: adopt/refresh first, then decide whether a
    retry is permitted, then submit. **Nothing is submitted while a job of this
    scope is in flight**, so a stage can never accidentally exceed the
    authorization's concurrent-worker limit, and a job whose acknowledgement was
    lost is reconciled (adopted) rather than duplicated.
    """

    record = record if record is not None else load_record(scope)
    if dry_run:
        view = view or envelope_module.load(scope)
        preview = submit_pending(scope, plan, stage, infra=infra, view=view,
                                 record=record, dry_run=True, require_clean=False)
        return {"step": "dry-run", "adopted": [], "outcomes": [],
                "submit": preview, "in_flight": []}
    view = view or _viewport(scope, infra=infra)
    if not dry_run:
        spend = ledger.derive_spend(infra.worker_list(), scope)
        budget = view.envelope["budget"]
        violation = None
        if envelope_module.is_expired(view.envelope):
            violation = "authorization expired"
        elif not spend.bounded:
            violation = "provider cost facts are unbounded"
        elif spend.estimated_spend_usd >= Decimal(str(budget["max_total_gpu_usd"])):
            violation = "total GPU budget exhausted"
        elif spend.estimated_wall_clock_hours >= Decimal(str(budget["max_wall_clock_hours"])):
            violation = "paid wall-clock budget exhausted"
        elif any(item["hourly_cost"] is None or item["hourly_cost"] >
                 Decimal(str(budget["max_gpu_hourly_usd"])) for item in spend.billable):
            violation = "provider price is unknown or above the hourly ceiling"
        if violation:
            for lease in ledger.active_leases(scope):
                if any(item["id"] == lease.get("worker_id") and item["billable"]
                       for item in spend.billable):
                    stopped = worker_module.stop_worker(
                        infra, lease["worker_id"], reason=violation)
                    if stopped.returncode != 0:
                        raise ReconcilableError(
                            "{}; worker {} could not be stopped".format(
                                violation, lease["worker_id"]), action="stop-worker")
            raise CostError("scope {} stopped before job reconciliation: {}".format(
                scope, violation))
    record, adopted = reconcile(scope, plan, stage, infra=infra, record=record)
    outcomes = assess(record, stage=stage)

    entries = [entry for entry in record["entries"].values()
               if entry.get("stage") == stage]
    in_flight = [entry for entry in entries
                 if entry.get("state") in (SUBMITTED, MONITORING, "preparing", "running")
                 and entry.get("job_id")]
    submittable = [
        entry for entry in entries
        if entry.get("state") == PENDING
        and not entry.get("job_id")
        and (int(entry.get("attempts", 0)) == 0 or entry.get("retry_allowed"))
    ]

    submitted = {"planned": [], "submitted": []}
    step = "monitor"
    if in_flight:
        step = "monitor"
    elif submittable and not dry_run:
        submitted = submit_pending(
            scope, plan, stage, infra=infra, view=view, record=record
        )
        step = "submitted"
    elif submittable:
        submitted = submit_pending(
            scope, plan, stage, infra=infra, view=view, record=record, dry_run=True
        )
        step = "dry-run"

    terminal = bool(entries) and all(
        entry.get("state") in (SUCCEEDED, FAILED, COLLECTED) for entry in entries
    )
    if terminal:
        step = "collect"
    return {"step": step, "adopted": adopted, "outcomes": outcomes,
            "submit": submitted, "in_flight": [entry["job_id"] for entry in in_flight]}


def summarize(record, plan=None, stage=None):
    entries = list(record.get("entries", {}).values())
    counts = {}
    for entry in entries:
        counts[entry.get("state")] = counts.get(entry.get("state"), 0) + 1
    return {
        "scope": record.get("scope"),
        "entries": entries,
        "counts": counts,
        "attempts": sum(int(entry.get("attempts", 0)) for entry in entries),
    }


def require_envelope(scope, *, repo_root=None):
    view = envelope_module.load(scope, repo_root)
    if not view.envelope:
        raise AuthorizationError("no envelope for {}".format(scope))
    return view


def implementation_bug(message):
    return ImplementationBugError(message)
