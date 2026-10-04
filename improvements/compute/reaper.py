"""Crash-independent enforcement of the cost envelope.

A deadline stored in a lease is a *record*, not an enforcement mechanism. The
orchestrator enforces it while it runs, so the moment the orchestrating process dies —
an OMP session terminated, a research runner killed, a crashed controller — nothing is
left that ever executes the sweep, and a paid Pod keeps billing against an authorization
nobody is watching. "A dead OMP process must not imply unlimited GPU billing" cannot be
satisfied by code that only runs inside OMP.

This module is the mechanism that closes that gap. It is an ordinary command — the same
backend, the same authoritative lease document, the same stop path — with **no
dependency on the session, the runner, or the agent** that created the resource, so a
controller-local timer (see :mod:`improvements.compute.reaper_units`) can execute it
after every other process has died. It is deliberately the smallest thing that can be
correct:

* it **reconciles before it acts** — provider inventory is read once, immutable
  observations are persisted, and unresolved create intents are resolved by their
  generated identity;
* it **stops, never destroys**. Stopping is reversible and ends the spend; destruction
  is irreversible and needs the deliberate `stop --destroy` path with a verified output
  set, so no automatic process can ever take it;
* it **never touches a resource it cannot attribute**. A worker outside every known
  scope prefix, or under a scope with no ARC lease, is reported and left alone; a network
  volume is never a cleanup step for compute and is never in the plan;
* it **takes the scope lock**, non-blocking, so it can never interleave with a live
  `/wav-cycle`; when the lock is held it refuses and the next tick retries;
* it is **idempotent** — a second run over the same provider state performs nothing,
  because the first one already recorded what it did;
* it **records what it did**, to the append-only event log and to a report document, so
  an unattended action is auditable after the fact.

What it cannot do is spend. It never creates a worker, a volume, or a job, and it never
writes, renews or widens an authorization; the only provider verb it can reach is the
reversible stop.
"""

import json
from decimal import Decimal

from improvements.compute import ledger, providers, state as state_module
from improvements.compute import sweep as sweep_module
from improvements.compute.errors import BusyError

# A create intent whose billable request never produced a worker and is older than this
# is abandoned. The threshold is far longer than any provider create round trip, so an
# in-flight request can never be mistaken for one that never happened.
INTENT_ABANDON_AFTER_HOURS = Decimal("1")

REPORT_SCHEMA_VERSION = 1


def report_path():
    """Where the reaper's last executed pass is recorded."""

    return state_module.path_for("reaper", "last-run.json")


def _age_hours(created_at, now):
    created = state_module.parse_timestamp(created_at)
    if created is None:
        return None
    seconds = (now - created).total_seconds()
    return Decimal(str(max(seconds, 0.0))) / Decimal(3600)


def reconcile_intents(scope, workers, *, now=None):
    """Resolve a create intent left behind by a controller that died mid-create.

    The intent is persisted *before* the billable request, so it is the authoritative
    record of a resource this backend asked for. If a worker with the generated name
    exists, it is redeemed into a lease — the same identity reconciliation the normal
    start path performs, and the reason a lost acknowledgement never becomes a duplicate
    Pod. If no such worker exists and the intent is old enough that the request cannot
    still be in flight, the intent is abandoned so it cannot block the scope forever.
    Nothing here touches the provider.
    """

    now = now or state_module.utc_now()
    redeemed = []
    abandoned = []
    for intent in ledger.pending_creates(scope):
        worker = ledger.find_worker_for_intent(intent, workers)
        if worker is not None:
            ledger.redeem_create(
                scope,
                worker_id=worker.get("id"),
                purpose=intent.get("purpose") or "recovered create intent",
                envelope_digest=intent.get("envelope_digest"),
                deadline=intent.get("deadline"),
                request=intent.get("request"),
                provider=providers.normalize_provider(intent.get("provider")),
            )
            redeemed.append(worker.get("id"))
            state_module.append_event(
                {"scope": scope, "action": "reaper-redeemed-create-intent",
                 "worker_id": worker.get("id"), "deadline": intent.get("deadline")},
                kind="reaper-redeemed-create-intent",
            )
            continue
        age = _age_hours(intent.get("created_at"), now)
        if age is not None and age >= INTENT_ABANDON_AFTER_HOURS:
            provider = providers.normalize_provider(intent.get("provider"))
            ledger.abandon_create(
                scope,
                "reaper: no worker ever appeared for a create intent older than {} "
                "hours".format(INTENT_ABANDON_AFTER_HOURS),
                provider=provider,
            )
            abandoned.append(intent.get("request_name") or
                             "{}:{}".format(provider, intent.get("created_at")))
            state_module.append_event(
                {"scope": scope, "action": "reaper-abandoned-create-intent",
                 "request_name": intent.get("request_name"), "age_hours": str(age)},
                kind="reaper-abandoned-create-intent",
            )
    return {"redeemed": redeemed, "abandoned": abandoned}


def _classifications(plan):
    return [
        {"worker_id": item["worker_id"], "classification": item["classification"],
         "action": item["action"]}
        for item in plan["employees"]
    ]


def _finalized(leases):
    return [
        {"worker_id": lease.get("worker_id"), "state": lease.get("state"),
         "cost_usd": lease.get("closed_cost_usd"),
         "wall_clock_hours": lease.get("closed_wall_clock_hours"),
         "precision": lease.get("finalization"),
         "reason": lease.get("finalization_reason")}
        for lease in leases
    ]


def reap(infra, *, execute=False, scopes=None, now=None):
    """Reconcile every known scope; stop what is past its deadline when executing.

    Dry-run by default, exactly like the sweep it delegates to: a reaper that acted
    without being asked would be a bigger hazard than the leak it exists to close. A
    dry run writes nothing at all — no observation, no lease, no report.

    An executing pass takes the host-level controller guard for its whole run, so it can
    never interleave with a live orchestrator mid-mutation, and then each scope's own
    lock. Refusing on either is a successful outcome for a timer: the next tick retries.
    """

    now = now or state_module.utc_now()
    scopes = list(scopes) if scopes else ledger.known_scopes()
    workers = infra.worker_list()
    jobs = infra.job_list()
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": state_module.isoformat(now),
        "executed": bool(execute),
        "dry_run": not execute,
        "state_root": state_module.state_root(),
        "provider_workers_seen": len(workers),
        "scopes": [],
    }
    if not execute:
        for scope in scopes:
            report["scopes"].append(
                _inspect(infra, scope, workers=workers, jobs=jobs, now=now))
        return report

    try:
        with state_module.controller_guard():
            for scope in scopes:
                report["scopes"].append(
                    _reap_scope(infra, scope, workers=workers, jobs=jobs, now=now))
    except BusyError as exc:
        # A live orchestrator is mutating this host right now. Its own scope lock would
        # catch its scope; the host guard catches every root it could be using.
        report["skipped"] = "another ARC controller process is mutating this host"
        report["reason"] = str(exc)
        state_module.append_event(
            {"action": "reaper-skipped-controller-guard", "reason": str(exc)},
            kind="reaper-skipped-controller-guard",
        )
    state_module.write_json_atomically(report_path(), report)
    state_module.append_event(
        {"action": "reaper-run", "scopes": scopes,
         "stopped": sum(len(item.get("stopped") or []) for item in report["scopes"]),
         "unbounded": sum(len(item.get("unbounded_leases") or [])
                          for item in report["scopes"])},
        kind="reaper-run",
    )
    return report


def _inspect(infra, scope, *, workers, jobs, now):
    """What an executing pass would consider for one scope, without writing anything."""

    absent = [
        lease.get("worker_id")
        for lease in ledger.active_leases(scope)
        if lease.get("worker_id") not in {worker.get("id") for worker in workers}
    ]
    plan = sweep_module.sweep(infra, scope, execute=False, jobs=jobs, now=now,
                              workers=workers)
    return {
        "scope": scope,
        "pending_creates": ledger.pending_creates(scope),
        "absent_leases": absent,
        "classifications": _classifications(plan),
        "unowned_workers": plan["unowned_workers"],
        "volumes_touched": plan["volumes_touched"],
    }


def _reap_scope(infra, scope, *, workers, jobs, now):
    """One scope's pass: observe, reconcile, bound, then stop what is past its deadline."""

    try:
        with state_module.scope_lock(scope):
            observed = ledger.observe_workers(workers, scope=scope, now=now)
            intents = reconcile_intents(scope, workers, now=now)
            cost = ledger.reconcile_absent_leases(scope, workers, now=now)
            plan = sweep_module.sweep(infra, scope, execute=True, jobs=jobs, now=now,
                                      workers=workers)
            return {
                "scope": scope,
                "observed_workers": observed,
                "create_intents": intents,
                "finalized_leases": _finalized(cost["finalized"]),
                "unbounded_leases": cost["unbounded"],
                "stopped": [entry["worker_id"] for entry in plan["performed"]],
                "classifications": _classifications(plan),
                "unowned_workers": plan["unowned_workers"],
                "volumes_touched": plan["volumes_touched"],
            }
    except BusyError as exc:
        # Another orchestrator owns this scope right now. Refusing is the whole point:
        # the live cycle is already enforcing the envelope, and the next tick reconciles
        # whatever it leaves behind.
        state_module.append_event(
            {"scope": scope, "action": "reaper-skipped-lock-held"},
            kind="reaper-skipped-lock-held",
        )
        return {
            "scope": scope,
            "skipped": "another orchestrator holds the scope lock",
            "reason": str(exc),
        }


def summarize(report):
    """One line per scope, for a timer's journal."""

    lines = ["reaper {} at {}".format(
        "executed" if report["executed"] else "dry-run", report["generated_at"])]
    for item in report["scopes"]:
        if item.get("skipped"):
            lines.append("  {}: skipped ({})".format(item["scope"], item["skipped"]))
            continue
        if item.get("stopped") is None:
            lines.append("  {}: {} classified, {} absent lease(s), {} unowned".format(
                item["scope"], len(item.get("classifications") or ()),
                len(item.get("absent_leases") or ()),
                len(item.get("unowned_workers") or ())))
            continue
        lines.append("  {}: stopped {}, finalized {}, unbounded {}, unowned {}".format(
            item["scope"], item["stopped"] or "none",
            len(item.get("finalized_leases") or ()),
            len(item.get("unbounded_leases") or ()),
            len(item.get("unowned_workers") or ())))
    return lines


def as_json(report):
    return json.dumps(report, indent=2, sort_keys=True)


def exit_code(report):
    """0 for a completed pass; 3 when a scope or the host could not be reconciled."""

    if report.get("skipped"):
        return 3
    return 3 if any(item.get("skipped") for item in report["scopes"]) else 0
