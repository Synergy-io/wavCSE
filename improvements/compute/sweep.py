"""Scope compute reconciliation: classification, deadlines, and idempotent sweep.

Nothing in the control plane expires a worker, so the orchestrator is the
reaper — but a reaper is dangerous unless it is scrupulous about what it may
touch. The rules here:

* **Dry-run by default.** A sweep that is not explicitly asked to act returns
  the plan it would execute and changes nothing.
* **Unknown workers are reported, never touched.** A worker that does not match
  a known scope prefix has no owner this backend can explain, so it is listed.
* **Adopted workers are stopped, never destroyed.** A worker that matched a
  scope prefix but has no creation record may have its spend ended (reversible)
  but is never destroyed (irreversible) by automation.
* **Network volumes are a different resource class.** No sweep ever emits a
  volume verb; a volume keeps billing storage after compute stops and is an
  operator/scope decision, not cleanup.
"""

from improvements.compute import ledger
from improvements.compute import state as state_module
from improvements.compute.errors import CapacityError

# Classification of one worker relative to a scope.
ACTIVE_HEALTHY = "active_healthy"
JOB_RUNNING = "job_running"
AWAITING_OUTPUTS = "awaiting_verified_outputs"
STOPPED = "stopped"
STALE = "stale_deadline_passed"
UNKNOWN_WORKER = "unknown_worker"

_ACTIVE_STATES = ("PROVISIONING", "STARTING", "RUNNING")
_ENDED_STATES = ("STOPPED", "DESTROYED", "ERROR", "UNKNOWN")


def classify(worker, *, scope, lease, jobs=(), now=None,
             envelope_digest=None, deadline=None):
    """Classify one provider worker for one scope."""

    now = now or state_module.utc_now()
    name = str(worker.get("name") or "")
    if not ledger.worker_belongs_to(name, scope):
        return UNKNOWN_WORKER, "name does not match any known scope"
    state = str(worker.get("state") or "UNKNOWN").upper()
    if state in ("DESTROYED",):
        return STOPPED, "provider reports the worker destroyed"
    if state in ("STOPPED",):
        return STOPPED, "provider reports the worker stopped"
    open_jobs = [
        job for job in jobs
        if str(job.get("state") or "").upper() in ("PENDING", "PREPARING", "RUNNING")
    ]
    if deadline is not None and now >= deadline:
        return STALE, "lease deadline passed at {}".format(
            state_module.isoformat(deadline)
        )
    if open_jobs:
        return JOB_RUNNING, "{} job(s) still open".format(len(open_jobs))
    if lease is not None and lease.get("pending_outputs"):
        return AWAITING_OUTPUTS, "outputs not yet verified"
    if state in _ACTIVE_STATES:
        return ACTIVE_HEALTHY, "active and inside its deadline"
    return STOPPED, "provider state {} does not bill compute".format(state)


def plan_actions(scope, *, workers, leases, jobs_by_worker, now=None,
                 envelope_digest=None):
    """Build the ordered, idempotent action plan for one scope."""

    now = now or state_module.utc_now()
    lease_by_worker = {
        lease.get("worker_id"): lease for lease in leases if lease.get("worker_id")
    }
    actions = []
    inactive = []
    unknown = []
    for worker in workers:
        worker_id = worker.get("id")
        name = str(worker.get("name") or "")
        lease = lease_by_worker.get(worker_id)
        deadline = None
        if lease and lease.get("deadline"):
            deadline = state_module.parse_timestamp(lease["deadline"])
        klass, reason = classify(
            worker, scope=scope, lease=lease,
            jobs=jobs_by_worker.get(worker_id, ()), now=now,
            envelope_digest=envelope_digest, deadline=deadline,
        )
        if klass == UNKNOWN_WORKER:
            unknown.append({
                "worker_id": worker_id,
                "name": name,
                "state": str(worker.get("state") or "UNKNOWN"),
                "reason": reason,
            })
            continue
        entry = {
            "worker_id": worker_id,
            "name": name,
            "state": str(worker.get("state") or "UNKNOWN"),
            "classification": klass,
            "reason": reason,
            "provenance": (lease or {}).get("provenance", "unrecorded"),
            "deadline": (lease or {}).get("deadline"),
        }
        if klass == STALE and lease is not None:
            entry["action"] = "stop"
            actions.append(entry)
        elif klass == STOPPED and lease is not None and lease.get("state") == "active" \
                and str(worker.get("state") or "").upper() == "STOPPED":
            # Bookkeeping only: the provider already stopped it.
            entry["action"] = "reconcile-lease"
            actions.append(entry)
        elif klass == STOPPED:
            # Already stopped and nothing to record: report it, do nothing.
            entry["action"] = "none"
            inactive.append(entry)
        else:
            entry["action"] = "none"
            actions.append(entry)
    return {
        "scope": scope,
        "employees": actions,
        "inactive": inactive,
        "unowned_workers": unknown,
        "volumes_touched": [],
    }


def sweep(infra, scope, *, view=None, execute=False, jobs=None, now=None):
    """Reconcile one scope's compute. Dry-run unless ``execute`` is true."""

    workers = infra.worker_list()
    leases = ledger.leases_for(scope)
    jobs = infra.job_list() if jobs is None else jobs
    jobs_by_worker = {}
    for job in jobs:
        metadata = ((job.get("spec") or {}).get("tracking") or {}).get("metadata") or {}
        if metadata.get("scope") != scope:
            continue
        jobs_by_worker.setdefault(job.get("worker_id"), []).append(job)

    plan = plan_actions(
        scope, workers=workers, leases=leases, jobs_by_worker=jobs_by_worker,
        now=now, envelope_digest=(view.digest if view else None),
    )

    performed = []
    for entry in plan["employees"]:
        if not execute or entry["action"] in ("none", "reconcile-lease"):
            continue
        if entry["action"] == "stop":
            from improvements.compute import worker as worker_module

            result = worker_module.stop_worker(
                infra, entry["worker_id"], reason=entry["reason"]
            )
            if result.returncode != 0:
                raise CapacityError("worker {} could not be stopped at its deadline"
                                    .format(entry["worker_id"]))
            performed.append(dict(entry, performed=True))
        elif entry["action"] == "destroy":
            raise CapacityError(
                "the sweep never destroys automatically; destruction requires the "
                "explicit `stop --destroy` path with a verified output set"
            )
    plan["dry_run"] = not execute
    plan["performed"] = performed
    return plan
