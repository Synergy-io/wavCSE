"""Cost accounting and worker ownership, derived from provider state.

Two facts drive this module:

* the provider is authoritative, and the control plane already reports what it
  observed — ``infra worker list --json`` carries each worker's ``hourly_cost``
  and ``created_at``, so spend does not need a counter the backend maintains for
  itself;
* a paid resource must always have an explainable owner. Ownership is derived
  from the worker name (the control plane prefixes every worker this tool
  creates with ``wavcse-`` and appends a generated suffix, so a scope-prefixed
  name *is* the ownership tag), and the lease adds only what the provider cannot
  know: the purpose, the deadline, the authorization digest, and the jobs run on
  it.

Accounting is honest about its own precision: every figure is an estimate from
the provider's own hourly rate and observed lifetime, never a billing statement.
An unknown price or an unknown creation time makes the figure unbounded, which
fails closed for new spend instead of under-reporting it.
"""

import os
import uuid
from functools import wraps
from decimal import Decimal

from improvements.compute import state as state_module
from improvements.compute.errors import ConfigurationError

# The control plane's own worker-name prefix (`_infra_worker_name`).
_PROVIDER_PREFIX = "wavcse"

SCHEMA_VERSION = 1

# Provider states, split by whether the resource can still bill for compute.
# Conservative on purpose: anything not proven stopped keeps counting.
_NON_BILLING_STATES = ("STOPPED", "DESTROYED")
_TERMINAL_STATES = ("DESTROYED",)

# Lease states from which no further compute cost can accrue. ``stopped`` stays in
# :func:`active_leases` for the restart path, so this tuple is used only where the
# question is "can this lease still bill?" rather than "is this lease finished?".
_CLOSED_LEASE_STATES = ("stopped", "destroyed", "vanished")

# How a closed lease's cost was established. ``provider`` means the figure came from a
# live provider read at the moment this backend ended the spend; the upper-bound marker
# means the resource left the provider inventory before that could happen and the figure
# is a provable ceiling derived from the last observation, never a measurement.
FINALIZATION_PROVIDER = "measured_from_provider_at_close"
FINALIZATION_UPPER_BOUND = "upper_bound_from_last_observation"

PRECISION = "estimate_from_provider_hourly_rate"


def normalize_scope(scope):
    """The control plane's name normalization for a human-supplied prefix."""

    normalized = "".join(
        character if (character.isalnum() or character == "-") else "-"
        for character in str(scope).strip().lower()
    )
    return normalized.strip("-")


def name_prefix_for(scope):
    """The worker-name prefix that marks ownership by ``scope``."""

    return "{}-{}-".format(_PROVIDER_PREFIX, normalize_scope(scope))


def worker_belongs_to(worker_name, scope):
    return bool(worker_name) and str(worker_name).startswith(name_prefix_for(scope))


def leases_path():
    return os.path.join(state_module.state_root(), "leases.json")


def _leases_document():
    path = leases_path()
    document = state_module.read_json(path, default=None)
    if document is None:
        document = {
            "schema_version": SCHEMA_VERSION,
            "pending_creates": [],
            "leases": [],
        }
    document.setdefault("pending_creates", [])
    document.setdefault("leases", [])
    return path, document


def _save(path, document):
    state_module.write_json_atomically(path, document)


def _serialized_lease_write(operation):
    """Protect the shared ledger across different scope controllers."""

    @wraps(operation)
    def locked(*args, **kwargs):
        with state_module.MutationLock("leases"):
            return operation(*args, **kwargs)

    return locked


def all_leases():
    return list(_leases_document()[1]["leases"])


def leases_for(scope):
    return [lease for lease in all_leases() if lease.get("scope") == scope]


def active_leases(scope=None):
    leases = all_leases() if scope is None else leases_for(scope)
    return [
        lease for lease in leases
        if lease.get("state") not in ("destroyed", "vanished")
    ]


def pending_creates(scope=None):
    intents = list(_leases_document()[1]["pending_creates"])
    if scope is None:
        return intents
    return [intent for intent in intents if intent.get("scope") == scope]


def known_scopes():
    """Every scope this controller has a lease or an unresolved intent for.

    This is the enforcement universe of the reaper: a scope with no recorded lease and no
    pending create has no ARC resource to reconcile, so nothing about it is touched.
    """

    scopes = set()
    for lease in all_leases():
        if lease.get("scope"):
            scopes.add(lease["scope"])
    for intent in pending_creates():
        if intent.get("scope"):
            scopes.add(intent["scope"])
    return sorted(scopes)


@_serialized_lease_write
def begin_create(scope, *, purpose, envelope_digest, request, deadline):
    """Record a creation intent *before* the billable request.

    Refuses while an earlier intent for the same prefix is unresolved: two
    overlapping creates would otherwise both reach the provider and open two
    billable Pods, which is exactly the failure this record exists to prevent.
    """

    path, document = _leases_document()
    prefix = name_prefix_for(scope)
    for intent in document["pending_creates"]:
        if str(intent.get("name_prefix") or "").startswith(prefix) and intent.get("status") == "pending":
            raise ConfigurationError(
                "an earlier worker create for {} is still unresolved (intent "
                "started {}); reconcile it with `sweep` before requesting another "
                "paid resource".format(scope, intent.get("created_at"))
            )
    nonce = uuid.uuid4().hex[:12]
    intent = {
        "name_prefix": prefix + nonce + "-",
        "request_name": scope + "-" + nonce,
        "scope": scope,
        "purpose": purpose,
        "envelope_digest": envelope_digest,
        "request": dict(request),
        "deadline": deadline,
        "status": "pending",
        "created_at": state_module.isoformat(state_module.utc_now()),
    }
    document["pending_creates"].append(intent)
    _save(path, document)
    state_module.append_event(
        {"scope": scope, "action": "worker-create-intent", "purpose": purpose,
         "envelope_digest": envelope_digest, "deadline": deadline},
        kind="worker-create-intent",
    )
    return intent


def find_worker_for_intent(intent, workers):
    """Reconcile a create intent against provider state by generated identity."""

    prefix = intent["name_prefix"]
    matches = [
        worker for worker in workers
        if str(worker.get("name") or "").startswith(prefix)
    ]
    if not matches:
        return None
    # Newest wins if an earlier stray exists; the caller records the ambiguity.
    matches.sort(key=lambda worker: str(worker.get("created_at") or ""), reverse=True)
    return matches[0]


@_serialized_lease_write
def redeem_create(scope, *, worker_id, purpose, envelope_digest, deadline, request=None):
    """Convert a resolved intent into an active lease."""

    path, document = _leases_document()
    prefix = name_prefix_for(scope)
    document["pending_creates"] = [
        intent for intent in document["pending_creates"]
        if not (str(intent.get("name_prefix") or "").startswith(prefix)
                and intent.get("status") == "pending")
    ]
    existing = [lease for lease in document["leases"]
                if lease.get("worker_id") == worker_id]
    if existing:
        _save(path, document)
        return existing[0]
    lease = {
        "scope": scope,
        "name_prefix": prefix,
        "worker_id": worker_id,
        "purpose": purpose,
        "envelope_digest": envelope_digest,
        "deadline": deadline,
        "request": dict(request or {}),
        "jobs": [],
        "state": "active",
        "provenance": "arc",
        "created_at": state_module.isoformat(state_module.utc_now()),
        "closed_at": None,
        "closed_cost_usd": None,
        "closed_wall_clock_hours": None,
    }
    document["leases"].append(lease)
    _save(path, document)
    state_module.append_event(
        {"scope": scope, "action": "worker-leased", "worker_id": worker_id,
         "purpose": purpose, "deadline": deadline, "envelope_digest": envelope_digest},
        kind="worker-leased",
    )
    return lease


@_serialized_lease_write
def reopen_lease(worker_id):
    """Mark an already accounted stopped worker active before restarting it."""

    path, document = _leases_document()
    for lease in document["leases"]:
        if lease.get("worker_id") == worker_id:
            lease["state"] = "active"
            _save(path, document)
            return lease
    raise ConfigurationError("worker {} has no ARC lease".format(worker_id))


@_serialized_lease_write
def abandon_create(scope, reason):
    """Drop an intent whose create provably did not happen."""

    path, document = _leases_document()
    prefix = name_prefix_for(scope)
    kept = []
    dropped = []
    for intent in document["pending_creates"]:
        if str(intent.get("name_prefix") or "").startswith(prefix) and intent.get("status") == "pending":
            dropped.append(intent)
        else:
            kept.append(intent)
    document["pending_creates"] = kept
    _save(path, document)
    for intent in dropped:
        state_module.append_event(
            {"scope": scope, "action": "worker-create-abandoned", "reason": reason},
            kind="worker-create-abandoned",
        )
    return dropped


@_serialized_lease_write
def adopt_worker(worker, *, scope, deadline, envelope_digest=None):
    """Record a matching but unleased worker without claiming authority over it.

    An adopted lease is a bookkeeping fact, not a claim that this session
    created the resource: it may be stopped (reversible, and it ends the spend)
    but never destroyed by the sweep.
    """

    path, document = _leases_document()
    for lease in document["leases"]:
        if lease.get("worker_id") == worker.get("id"):
            return lease
    lease = {
        "scope": scope,
        "name_prefix": name_prefix_for(scope),
        "worker_id": worker.get("id"),
        "purpose": "adopted: matched the scope name prefix with no creation record",
        "envelope_digest": envelope_digest,
        "deadline": deadline,
        "request": {},
        "jobs": [],
        "state": "active",
        "provenance": "adopted",
        "created_at": state_module.isoformat(state_module.utc_now()),
        "closed_at": None,
        "closed_cost_usd": None,
        "closed_wall_clock_hours": None,
    }
    document["leases"].append(lease)
    _save(path, document)
    state_module.append_event(
        {"scope": scope, "action": "worker-adopted", "worker_id": worker.get("id"),
         "reason": "scope prefix without a creation record"},
        kind="worker-adopted",
    )
    return lease


@_serialized_lease_write
def record_job(worker_id, *, job_key, job_id):
    """Attach a submitted job to the lease of the worker that runs it."""

    path, document = _leases_document()
    for lease in document["leases"]:
        if lease.get("worker_id") == worker_id:
            jobs = list(lease.get("jobs") or [])
            jobs = [job for job in jobs if job.get("job_key") != job_key]
            jobs.append({"job_key": job_key, "job_id": job_id})
            lease["jobs"] = jobs
            _save(path, document)
            return lease
    return None


@_serialized_lease_write
def close_lease(worker_id, *, cost_usd=None, wall_clock_hours=None, state="stopped"):
    """Freeze a lease's accrued cost so it survives the provider record.

    Called when this backend stops or destroys a worker: after that point the
    provider may stop reporting the worker at all, so the accrued figure is
    preserved here rather than recomputed from nothing.
    """

    path, document = _leases_document()
    updated = None
    for lease in document["leases"]:
        if lease.get("worker_id") == worker_id:
            was_active = lease.get("state") == "active"
            lease["state"] = state
            lease["closed_at"] = state_module.isoformat(state_module.utc_now())
            if cost_usd is not None:
                lease["closed_cost_usd"] = str(cost_usd)
            elif was_active:
                lease["closed_cost_usd"] = None
            if wall_clock_hours is not None:
                lease["closed_wall_clock_hours"] = str(wall_clock_hours)
            elif was_active:
                lease["closed_wall_clock_hours"] = None
            updated = lease
            break
    _save(path, document)
    return updated


@_serialized_lease_write
def observe_workers(workers, *, scope=None, now=None):
    """Record the provider facts about each lease's worker while it is still visible.

    A Pod that disappears takes its billing facts with it: the provider exposes only a
    current hourly rate and the start of the current billing period, and nothing at all
    once the Pod is gone from ``worker list``. Recording those two facts here, with the
    moment they were seen, is what later lets a vanished worker's cost be bounded from an
    authoritative observation instead of guessed.

    Only workers that match their own lease's scope prefix are recorded, so a lease can
    never absorb facts about a resource it does not own.
    """

    now = now or state_module.utc_now()
    path, document = _leases_document()
    by_id = {worker.get("id"): worker for worker in workers if worker.get("id")}
    observed = []
    for lease in document["leases"]:
        if scope is not None and lease.get("scope") != scope:
            continue
        if lease.get("state") not in ("active",):
            continue
        worker = by_id.get(lease.get("worker_id"))
        if worker is None:
            continue
        if not worker_belongs_to(worker.get("name"), lease.get("scope")):
            continue
        hourly = _decimal_or_none(worker.get("hourly_cost"))
        previous = lease.get("observation") or {}
        highest = _decimal_or_none(previous.get("max_hourly_cost"))
        if hourly is not None and (highest is None or hourly > highest):
            highest = hourly
        started = _latest_of(
            state_module.parse_timestamp(previous.get("billing_started_at")),
            state_module.parse_timestamp(worker.get("last_started_at")),
            state_module.parse_timestamp(worker.get("created_at")),
        )
        lease["observation"] = {
            "observed_at": state_module.isoformat(now),
            "state": str(worker.get("state") or "UNKNOWN").upper(),
            "hourly_cost": None if hourly is None else str(hourly),
            "max_hourly_cost": None if highest is None else str(highest),
            "billing_started_at": (
                None if started is None else state_module.isoformat(started)
            ),
        }
        observed.append(lease.get("worker_id"))
    if observed:
        _save(path, document)
    return observed


def _latest_of(*moments):
    known = [moment for moment in moments if moment is not None]
    return max(known) if known else None


@_serialized_lease_write
def reconcile_absent_leases(scope, workers, *, now=None):
    """Finalize the cost of leases whose provider worker can no longer bill.

    Two cases reach here, and both leave the ledger unbounded until they are resolved:

    * the worker is **absent** from provider inventory — it was destroyed, evicted, or
      terminated while nothing was watching;
    * the worker is present but reported in a non-billing state, and no earlier read
      recorded when its spend ended.

    Either way the resource *did* bill, and the honest answer is a provable ceiling rather
    than a measurement: the highest hourly rate ever observed for it, times the time from
    the start of its last observed billing period to the moment the absence was observed.
    A resource that vanished billed until it vanished and vanished no later than this
    observation, so its true cost cannot exceed that figure. A lease with no observation
    carrying both a rate and a start is left open and reported instead — the scope stays
    fail-closed rather than guessing a cost it cannot bound.

    The figure accumulates onto any cost already frozen on the lease, so a restart or an
    earlier reap never resets a predecessor's spend.
    """

    now = now or state_module.utc_now()
    path, document = _leases_document()
    present = {worker.get("id"): worker for worker in workers if worker.get("id")}
    finalized = []
    unbounded = []
    changed = False
    for lease in document["leases"]:
        if lease.get("scope") != scope:
            continue
        if lease.get("state") not in ("active",):
            continue
        worker_id = lease.get("worker_id")
        worker = present.get(worker_id)
        if worker is not None:
            state = str(worker.get("state") or "UNKNOWN").upper()
            if state not in _NON_BILLING_STATES:
                continue
            if _decimal_or_none(lease.get("closed_cost_usd")) is not None:
                continue
            target_state = "destroyed" if state == "DESTROYED" else "stopped"
            reason = "the provider reports this worker {}".format(state)
        else:
            target_state = "vanished"
            reason = "the worker is absent from provider inventory"
        entry = _finalize_from_observation(lease, state=target_state, now=now,
                                           reason=reason)
        if entry is None:
            observation = lease.get("observation") or {}
            missing = []
            if _decimal_or_none(observation.get("max_hourly_cost")) is None and \
                    _decimal_or_none(observation.get("hourly_cost")) is None:
                missing.append("no observed hourly price")
            if state_module.parse_timestamp(observation.get("billing_started_at")) is None:
                missing.append("no observed billing start")
            if not observation:
                missing.append("the worker was never observed while it was visible")
            unbounded.append({
                "worker_id": worker_id,
                "scope": scope,
                "state": target_state,
                "reason": reason,
                "missing": missing,
            })
            state_module.append_event(
                {"scope": scope, "action": "lease-cost-unbounded", "worker_id": worker_id,
                 "reason": "{}; {}".format(reason, "; ".join(missing))},
                kind="lease-cost-unbounded",
            )
            changed = True
            continue
        finalized.append(entry)
        changed = True
    if changed:
        _save(path, document)
    return {"finalized": finalized, "unbounded": unbounded}


def _finalize_from_observation(lease, *, state, now, reason):
    """Freeze a bounded final cost on one lease, or return None when none is provable."""

    observation = lease.get("observation") or {}
    rate = (_decimal_or_none(observation.get("max_hourly_cost"))
            or _decimal_or_none(observation.get("hourly_cost")))
    started = state_module.parse_timestamp(observation.get("billing_started_at"))
    if rate is None or rate <= 0 or started is None:
        return None
    hours = _elapsed_hours(started, now)
    prior_cost = _decimal_or_none(lease.get("closed_cost_usd")) or Decimal(0)
    prior_wall = _decimal_or_none(lease.get("closed_wall_clock_hours")) or Decimal(0)
    lease["state"] = state
    lease["closed_at"] = state_module.isoformat(now)
    lease["closed_cost_usd"] = str(prior_cost + rate * hours)
    lease["closed_wall_clock_hours"] = str(prior_wall + hours)
    lease["finalization"] = FINALIZATION_UPPER_BOUND
    lease["finalization_basis"] = {
        "hourly_cost_usd": str(rate),
        "billing_started_at": observation.get("billing_started_at"),
        "last_observed_at": observation.get("observed_at"),
        "last_observed_state": observation.get("state"),
        "absent_observed_at": state_module.isoformat(now),
    }
    lease["finalization_reason"] = reason
    state_module.append_event(
        {"scope": lease.get("scope"), "action": "lease-cost-finalized",
         "worker_id": lease.get("worker_id"), "state": state, "reason": reason,
         "cost_usd": lease["closed_cost_usd"],
         "wall_clock_hours": lease["closed_wall_clock_hours"],
         "precision": FINALIZATION_UPPER_BOUND},
        kind="lease-cost-finalized",
    )
    return lease


class SpendReport(object):
    """Estimated spend and exposure for one scope."""

    def __init__(self, scope, live, closed_cost, closed_wall_clock,
                 unknowns, now):
        self.scope = scope
        self.live = live
        self.closed_cost = closed_cost
        self.closed_wall_clock = closed_wall_clock
        self.unknowns = unknowns
        self.now = now

    @property
    def billable(self):
        return [entry for entry in self.live if entry["billable"]]

    @property
    def estimated_hourly_exposure_usd(self):
        total = Decimal(0)
        for entry in self.live:
            if not entry["billable"] or entry["hourly_cost"] is None:
                continue
            total += entry["hourly_cost"]
        return total

    @property
    def estimated_spend_usd(self):
        total = self.closed_cost
        for entry in self.live:
            if entry["accrued_usd"] is None:
                continue
            total += entry["accrued_usd"]
        return total

    @property
    def estimated_wall_clock_hours(self):
        total = self.closed_wall_clock
        for entry in self.live:
            if entry["elapsed_hours"] is None:
                continue
            total += entry["elapsed_hours"]
        return total

    @property
    def bounded(self):
        """False when some billing fact is unknown, so totals are lower bounds."""

        return not self.unknowns

    def facts(self):
        """The fact bundle :func:`improvements.compute.envelope.check` consumes."""

        return {
            "live_workers": [
                {
                    "id": entry["id"],
                    "name": entry["name"],
                    "state": entry["state"],
                    "hourly_cost": (
                        None if entry["hourly_cost"] is None
                        else str(entry["hourly_cost"])
                    ),
                }
                for entry in self.billable
            ],
            "estimated_spend_usd": str(self.estimated_spend_usd),
            "estimated_wall_clock_hours": str(self.estimated_wall_clock_hours),
            "estimated_hourly_exposure_usd": str(self.estimated_hourly_exposure_usd),
            "accounting_bounded": self.bounded,
            "unknowns": list(self.unknowns),
            "precision": PRECISION,
        }

    def as_dict(self):
        return {
            "scope": self.scope,
            "precision": PRECISION,
            "accounting_bounded": self.bounded,
            "unknowns": list(self.unknowns),
            "estimated_spend_usd": str(self.estimated_spend_usd),
            "estimated_wall_clock_hours": str(self.estimated_wall_clock_hours),
            "estimated_hourly_exposure_usd": str(self.estimated_hourly_exposure_usd),
            "closed_cost_usd": str(self.closed_cost),
            "live_workers": [
                {
                    "id": entry["id"],
                    "name": entry["name"],
                    "state": entry["state"],
                    "billable": entry["billable"],
                    "hourly_cost": (
                        None if entry["hourly_cost"] is None
                        else str(entry["hourly_cost"])
                    ),
                    "elapsed_hours": (
                        None if entry["elapsed_hours"] is None
                        else str(entry["elapsed_hours"])
                    ),
                    "accrued_usd": (
                        None if entry["accrued_usd"] is None
                        else str(entry["accrued_usd"])
                    ),
                    "deadline": entry["deadline"],
                    "lease_state": entry["lease_state"],
                    "provenance": entry["provenance"],
                }
                for entry in self.live
            ],
        }


def _decimal_or_none(value):
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except Exception:  # pragma: no cover - provider text we cannot parse
        return None


def _elapsed_hours(start, end):
    seconds = (end - start).total_seconds()
    if seconds < 0:
        seconds = 0.0
    return Decimal(str(seconds)) / Decimal(3600)


def derive_spend(workers, scope, leases=None, now=None):
    """Estimate what this scope has consumed, from provider facts."""

    now = now or state_module.utc_now()
    leases = all_leases() if leases is None else leases
    lease_by_worker = {
        lease.get("worker_id"): lease for lease in leases if lease.get("worker_id")
    }

    closed_cost = Decimal(0)
    closed_wall_clock = Decimal(0)
    unknowns = []
    for lease in leases:
        if lease.get("scope") != scope:
            continue
        if lease.get("state") in _CLOSED_LEASE_STATES and \
                (lease.get("closed_cost_usd") is None or
                 lease.get("closed_wall_clock_hours") is None):
            unknowns.append("closed lease {} has no verified accrued cost or time"
                            .format(lease.get("worker_id")))
        if lease.get("closed_cost_usd") is not None:
            closed_cost += _decimal_or_none(lease["closed_cost_usd"]) or Decimal(0)
        if lease.get("closed_wall_clock_hours") is not None:
            closed_wall_clock += (
                _decimal_or_none(lease["closed_wall_clock_hours"]) or Decimal(0)
            )

    entries = []
    seen = set()
    for worker in workers:
        name = str(worker.get("name") or "")
        if not worker_belongs_to(name, scope):
            continue
        worker_id = worker.get("id")
        seen.add(worker_id)
        lease = lease_by_worker.get(worker_id)
        state = str(worker.get("state") or "UNKNOWN").upper()
        billable = state not in _NON_BILLING_STATES
        hourly = _decimal_or_none(worker.get("hourly_cost"))
        # The provider's own most-recent start is the honest basis for the
        # current billing period: a worker stopped and started again does not
        # bill from its original creation.
        if lease and lease.get("closed_cost_usd") is not None:
            created = state_module.parse_timestamp(worker.get("last_started_at"))
            if billable and created is None:
                unknowns.append("restarted worker {} has no provider last-start time"
                                .format(worker_id))
        else:
            created = state_module.parse_timestamp(worker.get("created_at"))
        end = now
        if not billable and lease and lease.get("closed_at"):
            end = state_module.parse_timestamp(lease.get("closed_at")) or now
        elapsed = _elapsed_hours(created, end) if created else None
        accrued = None
        if billable:
            if hourly is None:
                unknowns.append(
                    "worker {} has no provider-reported hourly price".format(worker_id)
                )
            if elapsed is None:
                unknowns.append(
                    "worker {} has no provider-reported creation time".format(worker_id)
                )
            if hourly is not None and elapsed is not None:
                accrued = hourly * elapsed
        elif not (lease and lease.get("closed_cost_usd") is not None):
            # It billed until it stopped, and nothing here recorded when that
            # was, so the scope's total is a lower bound rather than a guess.
            unknowns.append(
                "worker {} is {} without a recorded close time; its accrued cost "
                "is unknown".format(worker_id, state)
            )
        entries.append({
            "id": worker_id,
            "name": name,
            "state": state,
            "billable": billable,
            "hourly_cost": hourly,
            "elapsed_hours": elapsed,
            "accrued_usd": accrued,
            "deadline": (lease or {}).get("deadline"),
            "lease_state": (lease or {}).get("state"),
            "provenance": (lease or {}).get("provenance", "unrecorded"),
        })

    # Leases whose worker the provider no longer lists, but which were never
    # closed by this backend: the spend happened, so keep counting it.
    for lease in leases:
        if lease.get("scope") != scope:
            continue
        if lease.get("state") in ("destroyed", "vanished"):
            continue
        if lease.get("worker_id") in seen:
            continue
        unknowns.append(
            "lease {} has no matching provider worker; its accrued cost is "
            "unknown".format(lease.get("worker_id"))
        )

    return SpendReport(scope, entries, closed_cost, closed_wall_clock, unknowns, now)


def scope_is_busy(scope, workers=None):
    """True while the scope has live compute or an unresolved create intent."""

    if pending_creates(scope):
        return True
    leases = [lease for lease in leases_for(scope) if lease.get("state") == "active"]
    if not leases:
        return False
    if workers is None:
        return True
    for worker in workers:
        if worker_belongs_to(worker.get("name"), scope):
            if str(worker.get("state") or "").upper() not in _TERMINAL_STATES:
                return True
    return False
