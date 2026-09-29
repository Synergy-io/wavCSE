"""One read-only picture of a scope's compute.

This is the call a cycle makes first and last, and the call ``wav-status`` makes
to reconcile runtime reality. It is strictly read-only: it reads the envelope,
the leases, the run ledger and provider state, and it never records a snapshot,
writes a lease, submits anything, or ends any spend. A status command that
mutated state would be a trap, so the module contains no write path at all.
"""

import os

from improvements.compute import ledger
from improvements.compute import state as state_module
from improvements.compute.errors import ComputeError, ConfigurationError


def _envelope_section(scope, *, repo_root=None):
    from improvements.compute import envelope as envelope_module

    try:
        view = envelope_module.load(scope, repo_root)
    except (ConfigurationError, ComputeError) as exc:
        return {"present": False, "reason": str(exc)}
    section = view.as_dict()
    section["present"] = True
    section["expired"] = envelope_module.is_expired(view.envelope)
    section["problems"] = list(view.problems)
    return section


def _infra_section(infra):
    if infra is None:
        return {"available": False, "reason": "the control plane is not resolvable"}
    try:
        workers = infra.worker_list()
    except ComputeError as exc:
        return {"available": False, "reason": str(exc)}
    return {"available": True, "workers": len(workers), "_workers": workers}


def build(scope, *, infra=None, repo_root=None, now=None, jobs=None):
    """Assemble the read-only compute picture for one scope."""

    now = now or state_module.utc_now()
    envelope_section = _envelope_section(scope, repo_root=repo_root)

    if infra is None:
        from improvements.compute import infra_cli, resolve as resolve_module

        try:
            infra = infra_cli.InfraCli(resolve_module.resolve())
        except ConfigurationError as exc:
            infra = None
            infra_reason = str(exc)
        else:
            infra_reason = None
    else:
        infra_reason = None

    infra_section = _infra_section(infra)
    workers = infra_section.pop("_workers", []) if infra_section.get("available") else []

    spend = ledger.derive_spend(workers, scope, now=now)
    leases = ledger.leases_for(scope)
    pending = ledger.pending_creates(scope)

    open_jobs = []
    if jobs is None and infra is not None and infra_section.get("available"):
        try:
            jobs = infra.job_list()
        except ComputeError:
            jobs = []
    for job in jobs or []:
        metadata = ((job.get("spec") or {}).get("tracking") or {}).get("metadata") or {}
        if metadata.get("scope") != scope:
            continue
        if str(job.get("state") or "").upper() in ("PENDING", "PREPARING", "RUNNING"):
            open_jobs.append({
                "job_id": job.get("job_id"),
                "name": job.get("name"),
                "state": job.get("state"),
                "worker_id": job.get("worker_id"),
                "arm": metadata.get("arm"),
                "seed": metadata.get("seed"),
            })

    blocked = None
    if not envelope_section.get("present"):
        blocked = {"class": "AUTHORIZATION", "reason": envelope_section.get("reason")}
    elif envelope_section.get("expired"):
        blocked = {"class": "AUTHORIZATION", "reason": "the authorization expired"}
    elif envelope_section.get("modified_in_tree"):
        blocked = {"class": "AUTHORIZATION",
                   "reason": "the envelope differs from its committed revision"}
    elif not infra_section.get("available"):
        blocked = {"class": "TRANSIENT_INFRA",
                   "reason": infra_reason or infra_section.get("reason")}
    elif not spend.bounded:
        blocked = {"class": "COST", "reason": "; ".join(spend.unknowns)}
    elif open_jobs:
        blocked = {"class": "MONITORING", "reason": "jobs are in flight"}

    return {
        "scope": scope,
        "checked_at": state_module.isoformat(now),
        "envelope": envelope_section,
        "infra": infra_section,
        "spend": spend.as_dict(),
        "leases": {
            "active": [lease for lease in leases if lease.get("state") == "active"],
            "pending_creates": pending,
        },
        "open_jobs": open_jobs,
        "state_root": state_module.state_root(),
        "blocked": blocked,
    }


def next_action(picture):
    """The single next action a cycle should take, from the picture alone."""

    blocked = picture.get("blocked") or {}
    if blocked:
        return {"action": "stop", "class": blocked.get("class"),
                "reason": blocked.get("reason")}
    if picture["open_jobs"]:
        return {"action": "collect", "reason": "jobs are in flight"}
    if picture["leases"]["active"]:
        return {"action": "sweep", "reason": "a lease is active with no open job"}
    return {"action": "ensure-worker", "reason": "no live compute for this scope"}
