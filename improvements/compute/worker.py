"""Worker acquisition and teardown inside an authorization envelope.

The control plane owns worker lifecycle; this module decides *whether* to ask
for one, and records the ownership that the provider cannot express.

Two safety properties are load-bearing:

* **A create is never blind-retried.** The intent is persisted before the
  billable request; if the response is lost, the worker is reconciled by the
  generated name it would have had. A second billable request is not issued
  while that intent is unresolved.
* **The envelope is checked before and after.** ``--max-price`` makes the
  control plane refuse an offer above the ceiling, and the observed price is
  re-checked afterwards, so a Pod that somehow costs more than authorized is
  destroyed rather than left billing.
"""

from decimal import Decimal

from improvements.compute import ledger, providers, state as state_module
from improvements.compute import envelope as envelope_module
from improvements.compute import jobspec, remote_commit
from improvements.compute.errors import (
    AuthorizationError,
    CapacityError,
    ConfigurationError,
    CostError,
    ReconcilableError,
)

DEFAULT_DEADLINE_HOURS = Decimal("6")


def _deadline_for(view, horizon_hours=None, spend=None):
    """Bounded lease lifetime: envelope expiry, wall-clock budget, and a horizon.

    The tightest of the three wins when the controller runs a transition or
    sweep. A separate reaper is still required to enforce it after a crash.
    """

    from datetime import timedelta

    now = state_module.utc_now()
    candidates = [envelope_module.expiry(view.envelope)]
    wall_clock_limit = Decimal(str(view.envelope["budget"]["max_wall_clock_hours"]))
    consumed = Decimal(0) if spend is None else spend.estimated_wall_clock_hours
    remaining_wall = wall_clock_limit - consumed
    if remaining_wall < 0:
        remaining_wall = Decimal(0)
    candidates.append(now + timedelta(seconds=float(remaining_wall * 3600)))
    total_limit = Decimal(str(view.envelope["budget"]["max_total_gpu_usd"]))
    consumed_usd = Decimal(0) if spend is None else spend.estimated_spend_usd
    remaining_usd = max(Decimal(0), total_limit - consumed_usd)
    hourly_ceiling = Decimal(str(view.envelope["budget"]["max_gpu_hourly_usd"]))
    candidates.append(now + timedelta(seconds=float(
        remaining_usd / hourly_ceiling * 3600)))
    candidates.append(
        now + timedelta(seconds=float(
            Decimal(str(horizon_hours or DEFAULT_DEADLINE_HOURS)) * 3600
        ))
    )
    return min(candidates)


def _colab_deadline_for(view, horizon_hours=None, spend=None):
    """Bound a Colab lease without pretending CU has an USD exchange rate."""

    from datetime import timedelta

    now = state_module.utc_now()
    wall_clock_limit = Decimal(str(view.envelope["budget"]["max_wall_clock_hours"]))
    consumed = Decimal(0) if spend is None else spend.estimated_wall_clock_hours
    remaining_wall = max(Decimal(0), wall_clock_limit - consumed)
    return min(
        envelope_module.expiry(view.envelope),
        now + timedelta(seconds=float(remaining_wall * 3600)),
        now + timedelta(seconds=float(
            Decimal(str(horizon_hours or DEFAULT_DEADLINE_HOURS)) * 3600
        )),
    )


def _facts(scope, workers):
    spend = ledger.derive_spend(workers, scope)
    return spend, spend.facts()


def preflight(view, scope, *, workers, container_disk_gb, projected_hours=None):
    """Decide whether a create may proceed, before anything is submitted."""

    spend, facts = _facts(scope, workers)
    if not spend.bounded:
        return envelope_module.Decision(
            False, "COST",
            "the scope's spend cannot be bounded from provider facts ({}); new "
            "spend fails closed. Resolve with `status` before provisioning.".format(
                "; ".join(spend.unknowns)),
            action=envelope_module.ACTION_CREATE_WORKER,
        )
    requested = {
        "container_disk_gb": int(container_disk_gb),
        "hourly_usd": str(view.envelope["budget"]["max_gpu_hourly_usd"]),
        "projected_hours": str(projected_hours or 1),
    }
    return envelope_module.check(
        view, envelope_module.ACTION_CREATE_WORKER, facts, requested=requested
    )


def resolve_network_volume(view, volumes, selector):
    """Pick an existing network volume inside the envelope's selector.

    Never creates one: a new persistent resource needs an explicit grant, and a
    volume keeps billing storage after compute stops.
    """

    decision = envelope_module.check(
        view, envelope_module.ACTION_ATTACH_VOLUME, {"live_workers": []}
    )
    if not decision.allowed:
        return None, decision
    selector = selector or {}
    candidates = []
    for volume in volumes:
        if int(volume.get("size_gb") or 0) < int(selector.get("min_size_gb") or 0):
            continue
        if selector.get("datacenter") and volume.get("datacenter") != selector["datacenter"]:
            continue
        if selector.get("volume_type") and volume.get("volume_type") != selector["volume_type"]:
            continue
        candidates.append(volume)
    if not candidates:
        return None, envelope_module.Decision(
            False, "CAPACITY",
            "no existing network volume matches the plan's selector; creating one "
            "would add a new persistent-billing resource, which the envelope "
            "controls separately",
            action=envelope_module.ACTION_ATTACH_VOLUME,
        )
    candidates.sort(key=lambda volume: (-int(volume.get("size_gb") or 0),
                                        str(volume.get("id"))))
    return candidates[0], envelope_module.Decision(
        True, "AUTONOMOUS", "existing network volume selected",
        action=envelope_module.ACTION_ATTACH_VOLUME,
    )


def _worker_request(plan, scope):
    """Provider-neutral resource intent derived from one validated plan."""

    provider = jobspec.plan_provider(plan)
    worker = plan["worker"]
    if provider == providers.COLAB:
        return {
            "provider": provider,
            "name": None,
            "gpu": worker.get("gpu_type"),
            "cloud": None,
            "image": None,
            "template": None,
            "gpu_count": 1,
            "container_disk_gb": None,
            "volume_gb": 0,
            "data_centers": (),
            "network_volume_selector": None,
        }
    return {
        "provider": provider,
        "name": scope,
        "gpu": worker["gpu_type"],
        "cloud": worker["cloud"],
        "image": worker.get("image"),
        "template": worker.get("template"),
        "gpu_count": int(worker.get("gpu_count", 1)),
        "container_disk_gb": int(worker.get("container_disk_gb", 20)),
        "volume_gb": int(worker.get("volume_gb", 0)),
        "volume_mount_path": worker.get("volume_mount_path"),
        "data_centers": tuple(worker.get("data_centers") or ()),
        "network_volume_selector": worker.get("network_volume"),
    }


def ensure_worker(plan, view, *, infra, purpose=None, projected_hours=None):
    """Acquire one provider resource without leaking provider mechanics upward."""

    provider = jobspec.plan_provider(plan)
    authorized = envelope_module.envelope_provider(view.envelope)
    if provider != authorized:
        raise CostError(
            "compute plan provider {} does not match authorization provider {}".format(
                provider, authorized
            )
        )
    # An authorization is isolated by scope identity *and* scope kind. A plan
    # that names another scope, or the right scope of the wrong kind — a Study
    # plan under an infrastructure-validation grant, or the reverse — is
    # refused before any provider call, because no amount of later care makes
    # that pairing mean what either side intended.
    plan_scope = jobspec.plan_scope(plan)
    if plan_scope != view.scope:
        raise AuthorizationError(
            "compute plan scope {} does not match authorization scope {}; a plan may "
            "only spend the authorization it is named by".format(plan_scope, view.scope)
        )
    plan_kind = jobspec.plan_scope_kind(plan)
    authorized_kind = envelope_module.envelope_scope_kind(view.envelope)
    if plan_kind != authorized_kind:
        raise AuthorizationError(
            "compute plan scope {}({}) is a {} scope but the authorization for {} is a "
            "{} scope; an authorization is isolated by both scope identity and scope "
            "kind".format(plan_scope, view.scope, plan_kind, view.scope, authorized_kind)
        )
    if provider == providers.COLAB:
        return _ensure_colab_worker(
            plan, view, infra=infra, purpose=purpose, projected_hours=projected_hours)
    return _ensure_runpod_worker(
        plan, view, infra=infra, purpose=purpose, projected_hours=projected_hours)


def _ensure_colab_worker(plan, view, *, infra, purpose=None, projected_hours=None):
    """Return one owned READY Colab session, creating it exactly once if needed.

    wavcse-infra owns allocation identity, CU guards, bootstrap, readiness and
    release. This layer owns the human authorization, the scope lease, exact-
    commit availability, attempt accounting and cleanup attribution.
    """

    scope = jobspec.plan_scope(plan)
    actions = []
    infra.job_list()
    workers = infra.worker_list(provider=providers.COLAB)
    ledger.observe_workers(workers, scope=scope)
    ledger.reconcile_absent_leases(scope, workers)
    spend = ledger.derive_spend(workers, scope)
    if view.modified_in_tree or view.uncommitted:
        raise CostError("compute authorization must match a committed grant")
    commit = jobspec.git_state()["head"]
    remote_commit.require_plan_commit(plan, commit)
    actions.append("proved commit {} is available on the worker's remote".format(commit[:12]))
    for lease in ledger.active_leases(scope):
        if lease.get("state") == "active" and lease.get("envelope_digest") != view.digest:
            raise CostError("active worker {} belongs to a different authorization digest"
                            .format(lease.get("worker_id")))
        if ledger.lease_provider(lease) != providers.COLAB:
            raise CostError("scope {} already owns a non-Colab lease".format(scope))
    for intent in ledger.pending_creates(scope):
        if intent.get("envelope_digest") != view.digest:
            raise CostError("unresolved create intent belongs to a different authorization digest")
        if providers.normalize_provider(intent.get("provider")) != providers.COLAB:
            raise CostError("scope {} has an unresolved non-Colab create intent".format(scope))
    envelope_module.record_snapshot(view, busy=ledger.scope_is_busy(scope, workers))

    intents = ledger.pending_creates(scope)
    recovered = None
    if intents:
        if len(intents) != 1:
            raise ReconcilableError("multiple unresolved create intents for {}".format(scope),
                                    action="create-worker")
        intent = intents[0]
        recovered = ledger.find_worker_for_intent(intent, workers)
        if recovered is None:
            raise ReconcilableError(
                "Colab allocation for {} remains unresolved; no second provider "
                "request is safe".format(scope), action="create-worker")
        ledger.redeem_create(
            scope, worker_id=recovered["id"], purpose=intent["purpose"],
            envelope_digest=intent["envelope_digest"], deadline=intent["deadline"],
            request=intent["request"], provider=providers.COLAB,
        )
        actions.append("reconciled Colab create intent for {}".format(recovered["id"]))

    existing = recovered or _select_existing_colab(workers, scope)
    if existing is not None:
        lease = _lease_for(existing["id"])
        if lease is None or lease.get("scope") != scope or lease.get("provenance") != "arc":
            raise ReconcilableError(
                "Colab session {} has no wavCSE creation lease".format(existing["id"]),
                action="worker-ensure",
            )
        decision = envelope_module.check(
            view, envelope_module.ACTION_SUBMIT_JOB, spend.facts(),
            requested={"provider": providers.COLAB,
                       "projected_hours": str(projected_hours or 1)},
        )
        if not decision.allowed:
            release = destroy_worker(
                infra, existing["id"], reason=decision.reason, worker=existing)
            if release.returncode != 0:
                raise ReconcilableError(
                    "Colab session {} could not be released after authorization refusal"
                    .format(existing["id"]), action="destroy-worker")
            decision.require()
        actions.append("reused Colab session {}".format(existing["id"]))
        try:
            _prepare_colab(infra, existing["id"], existing)
        except ReconcilableError:
            _release_after_prepare_failure(infra, existing)
            raise
        actions.append("Colab readiness satisfied without SSH")
        return existing, actions

    if not spend.bounded:
        raise CostError(
            "the scope's Colab allocation accounting is not bounded ({}); "
            "new allocation fails closed".format("; ".join(spend.unknowns)))
    decision = envelope_module.check(
        view, envelope_module.ACTION_CREATE_WORKER, spend.facts(),
        requested={"provider": providers.COLAB,
                   "projected_hours": str(projected_hours or 1)},
    )
    if not decision.allowed:
        decision.require()

    request = _worker_request(plan, scope)
    before_ids = {worker.get("id") for worker in workers if worker.get("id")}
    deadline = state_module.isoformat(_colab_deadline_for(view, projected_hours, spend))
    intent = ledger.begin_create(
        scope,
        provider=providers.COLAB,
        purpose=purpose or "job execution",
        envelope_digest=view.digest,
        request={"provider": providers.COLAB, "gpu": request.get("gpu")},
        deadline=deadline,
        existing_worker_ids=before_ids,
    )
    actions.append("recorded a Colab create intent before the allocation request")

    result = infra.worker_create(
        provider=providers.COLAB,
        name=None,
        gpu=request.get("gpu"),
        cloud=None,
        start_ssh=False,
        require_direct_ssh=False,
        max_price=None,
        timeout=None,
    )
    created = ledger.find_worker_for_intent(
        intent, infra.worker_list(provider=providers.COLAB))
    if created is None:
        raise ReconcilableError(
            "the Colab allocation for {} returned {} but no single new session "
            "identity is visible. The intent stays open and the request is never "
            "repeated; reconcile with `sweep` or `worker ensure`. Provider said: {}"
            .format(scope, result.returncode, _failure_text(result)),
            action="create-worker",
        )

    worker_id = created["id"]
    ledger.redeem_create(
        scope, worker_id=worker_id, purpose=purpose or "job execution",
        envelope_digest=view.digest, deadline=deadline,
        request=intent["request"], provider=providers.COLAB,
    )
    actions.append("allocated Colab session {}".format(worker_id))
    try:
        _prepare_colab(infra, worker_id, created)
    except ReconcilableError:
        _release_after_prepare_failure(infra, created)
        raise
    actions.append("Colab readiness satisfied without SSH")
    return created, actions


def _select_existing_colab(workers, scope):
    active = {
        lease.get("worker_id"): lease
        for lease in ledger.leases_for(scope)
        if lease.get("state") == "active"
        and ledger.lease_provider(lease) == providers.COLAB
    }
    candidates = [
        worker for worker in workers
        if worker.get("id") in active
        and ledger.worker_matches_lease(worker, active[worker.get("id")])
        and str(worker.get("state") or "").upper() not in ("DESTROYED", "TERMINATING")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda worker: str(worker.get("id")))
    return candidates[-1]


def _ensure_runpod_worker(plan, view, *, infra, purpose=None, projected_hours=None):
    """Return a ready scope worker, creating one inside the envelope if needed.

    Returns ``(worker, actions)`` where ``actions`` is the ordered list of steps
    taken, so a caller can report exactly what was done. An existing worker is
    reused and re-walked through the (idempotent) readiness ladder rather than
    assumed ready.
    """

    scope = jobspec.plan_scope(plan)
    actions = []
    # A paid Pod is pointless if this checkout cannot discover and reconcile
    # jobs after a lost submit acknowledgement. Check that contract first.
    infra.job_list()
    workers = infra.worker_list()
    # A predecessor that vanished while nobody was watching must be given its bounded
    # final cost here, or the scope stays unbounded and no replacement can be created.
    ledger.observe_workers(workers, scope=scope)
    ledger.reconcile_absent_leases(scope, workers)
    spend = ledger.derive_spend(workers, scope)
    if view.modified_in_tree or view.uncommitted:
        raise CostError("compute authorization must match a committed grant")
    # A paid Pod — or a restart that resumes billing — is money spent on an impossible
    # job unless the worker's remote can serve the exact commit. Prove that against the
    # remote the worker clones, before any reuse decision touches the provider.
    commit = jobspec.git_state()["head"]
    remote_commit.require_plan_commit(plan, commit)
    actions.append("proved commit {} is available on the worker's remote".format(commit[:12]))
    for lease in ledger.active_leases(scope):
        if lease.get("state") == "active" and lease.get("envelope_digest") != view.digest:
            raise CostError("active worker {} belongs to a different authorization digest"
                            .format(lease.get("worker_id")))
    for intent in ledger.pending_creates(scope):
        if intent.get("envelope_digest") != view.digest:
            raise CostError("unresolved create intent belongs to a different authorization digest")
    envelope_module.record_snapshot(view, busy=ledger.scope_is_busy(scope, workers))

    intents = ledger.pending_creates(scope)
    recovered = None
    if intents:
        if len(intents) != 1:
            raise ReconcilableError("multiple unresolved create intents for {}".format(scope),
                                    action="create-worker")
        intent = intents[0]
        recovered = ledger.find_worker_for_intent(intent, workers)
        if recovered is None:
            raise ReconcilableError(
                "worker create for {} remains unresolved; no second paid request is safe"
                .format(scope), action="create-worker")
        ledger.redeem_create(
            scope, worker_id=recovered["id"], purpose=intent["purpose"],
            envelope_digest=intent["envelope_digest"], deadline=intent["deadline"],
            request=intent["request"],
        )
        actions.append("reconciled create intent for {}".format(recovered["id"]))

    existing = recovered or _select_existing(workers, scope)
    if existing is not None:
        lease = _lease_for(existing["id"])
        if lease is None or lease.get("scope") != scope or lease.get("provenance") != "arc":
            raise ReconcilableError(
                "scope-named worker {} has no ARC creation lease; refusing to start "
                "or use a worker with unproven ownership".format(existing["id"]),
                action="worker-ensure",
            )
        if lease.get("envelope_digest") != view.digest and lease.get("state") == "active":
            raise CostError("worker {} was created under a different authorization digest"
                            .format(existing["id"]))
        hourly = _decimal_or_none(existing.get("hourly_cost"))
        if hourly is None or hourly > Decimal(str(view.envelope["budget"]["max_gpu_hourly_usd"])):
            if str(existing.get("state") or "").upper() != "STOPPED":
                stopped = stop_worker(infra, existing["id"], reason="price is unknown or above ceiling")
                if stopped.returncode != 0:
                    raise ReconcilableError("worker {} has an unauthorized price and could not be stopped"
                                            .format(existing["id"]), action="stop-worker")
            raise CostError("worker {} has unknown or unauthorized provider price".format(existing["id"]))
        requested = {"hourly_usd": str(hourly),
                     "projected_hours": str(projected_hours or 1)}
        decision = envelope_module.check(
            view, envelope_module.ACTION_SUBMIT_JOB, spend.facts(), requested=requested)
        if not decision.allowed:
            if str(existing.get("state") or "").upper() != "STOPPED":
                stopped = stop_worker(infra, existing["id"], reason=decision.reason)
                if stopped.returncode != 0:
                    raise ReconcilableError("worker {} could not be stopped after authorization "
                                            "refusal".format(existing["id"]),
                                            action="stop-worker")
            decision.require()
        if str(existing.get("state") or "").upper() == "STOPPED":
            if not view.envelope["concurrency"]["replacement_workers_allowed"]:
                raise CapacityError("restarting a stopped worker requires replacement authority")
            ledger.reopen_lease(existing["id"])
        actions.append("reused worker {}".format(existing.get("id")))
        try:
            _prepare(infra, existing.get("id"), existing)
        except ReconcilableError:
            _stop_after_prepare_failure(infra, existing["id"])
            raise
        actions.append("readiness ladder satisfied")
        return existing, actions

    decision = preflight(
        view, scope, workers=workers,
        container_disk_gb=plan["worker"].get("container_disk_gb", 20),
        projected_hours=projected_hours,
    )
    if not decision.allowed:
        decision.require()

    volume = None
    request = _worker_request(plan, scope)
    selector = request.get("network_volume_selector")
    if selector:
        volume, volume_decision = resolve_network_volume(
            view, infra.volume_list(), selector
        )
        if volume is None:
            volume_decision.require()
        actions.append("attached existing network volume")

    intent = ledger.begin_create(
        scope,
        purpose=purpose or "job execution",
        envelope_digest=view.digest,
        request={
            "gpu": request["gpu"], "cloud": request["cloud"],
            "gpu_count": request["gpu_count"],
            "container_disk_gb": request["container_disk_gb"],
        },
        deadline=state_module.isoformat(
            _deadline_for(view, projected_hours, spend)
        ),
    )
    actions.append("recorded a create intent before the billable request")

    result = infra.worker_create(
        name=intent["request_name"],
        gpu=request["gpu"],
        cloud=request["cloud"],
        image=request["image"],
        template=request["template"],
        gpu_count=request["gpu_count"],
        container_disk_gb=request["container_disk_gb"],
        volume_gb=request["volume_gb"],
        network_volume_id=(volume or {}).get("id"),
        data_centers=request["data_centers"],
        max_price=view.envelope["budget"]["max_gpu_hourly_usd"],
        timeout=None,
    )

    created = ledger.find_worker_for_intent(intent, infra.worker_list())
    if created is None:
        raise ReconcilableError(
            "the worker create for {} returned {} but no worker with the "
            "expected generated name is visible yet. The intent stays open and "
            "the paid request is never repeated; reconcile with `sweep` or "
            "`worker ensure` once the provider lists it. Provider said: {}".format(
                scope, result.returncode, _failure_text(result)),
            action="create-worker",
        )

    worker_id = created.get("id")
    actions.append("created worker {}".format(worker_id))
    lease = ledger.redeem_create(
        scope, worker_id=worker_id, purpose=purpose or "job execution",
        envelope_digest=view.digest,
        deadline=state_module.isoformat(
            _deadline_for(view, projected_hours, spend)
        ),
        request=intent["request"],
    )

    actual = _decimal_or_none(created.get("hourly_cost"))
    ceiling = Decimal(str(view.envelope["budget"]["max_gpu_hourly_usd"]))
    if actual is not None and actual > ceiling:
        destruction = destroy_worker(infra, worker_id, reason="price above ceiling")
        if destruction.returncode != 0:
            raise ReconcilableError("worker {} has an unauthorized price and its destroy "
                                    "request failed; it still needs cleanup".format(worker_id),
                                    action="destroy-worker")
        raise CostError(
            "the created worker reports ${}/hour, above the authorized ceiling "
            "${}/hour; it was destroyed immediately rather than left billing".format(
                actual, ceiling
            )
        )
    if actual is None:
        destruction = destroy_worker(infra, worker_id, reason="price unknown")
        if destruction.returncode != 0:
            raise ReconcilableError("worker {} has an unknown price and its destroy "
                                    "request failed; it still needs cleanup".format(worker_id),
                                    action="destroy-worker")
        raise CostError(
            "the provider did not report a price for the created worker, so the "
            "authorization cannot be enforced; it was destroyed immediately"
        )

    try:
        _prepare(infra, worker_id)
    except ReconcilableError:
        _stop_after_prepare_failure(infra, worker_id)
        raise
    actions.append("readiness ladder satisfied")
    return created, actions


def _select_existing(workers, scope):
    """A scope worker that is worth reusing, newest first.

    Any non-terminal state is a candidate: the readiness ladder is idempotent,
    so reusing and re-walking it is safer than assuming a worker is ready.
    """

    candidates = [
        worker for worker in workers
        if ledger.worker_belongs_to(worker.get("name"), scope)
        and str(worker.get("state") or "").upper() not in ("DESTROYED", "TERMINATING")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda worker: str(worker.get("created_at") or ""), reverse=True)
    return candidates[0]


def _prepare_colab(infra, worker_id, worker):
    """Use Colab's native bootstrap/readiness path — never start or wait for SSH."""

    try:
        provider = providers.worker_provider(worker)
        transport = providers.worker_transport(worker)
    except providers.ProviderValueError as exc:
        raise ReconcilableError(str(exc), action="worker-ensure") from exc
    if provider != providers.COLAB or transport != providers.COLAB_EXEC:
        raise ReconcilableError(
            "worker {} is not a Colab/COLAB_EXEC resource".format(worker_id),
            action="worker-ensure",
        )
    bootstrap = infra.worker_bootstrap(worker_id)
    if bootstrap.returncode != 0:
        raise ReconcilableError(
            "Colab worker {} bootstrap failed: {}".format(
                worker_id, _failure_text(bootstrap)),
            action="bootstrap",
        )
    health = infra.worker_health(worker_id, json_output=False)
    if health.returncode != 0:
        raise ReconcilableError(
            "Colab worker {} health check failed: {}".format(
                worker_id, _failure_text(health)),
            action="health",
        )


def _release_after_prepare_failure(infra, worker):
    result = destroy_worker(
        infra, worker["id"], reason="Colab readiness failed", worker=worker)
    if result.returncode != 0:
        raise ReconcilableError(
            "Colab worker {} failed readiness and could not be released; cleanup "
            "remains pending".format(worker["id"]),
            action="destroy-worker",
        )


def _prepare(infra, worker_id, worker=None):
    """Walk the RunPod readiness ladder explicitly: start, SSH, bootstrap, health.

    Every step is idempotent in the control plane, so this is the recovery path
    after a partial failure as well as the normal path after a create. A worker
    the provider reports as stopped is started first — reuse is cheaper than a
    new Pod, and its cost is re-billed from the moment it starts.
    """

    try:
        provider = providers.worker_provider(worker or {})
        transport = providers.worker_transport(worker or {})
    except providers.ProviderValueError as exc:
        raise ReconcilableError(str(exc), action="worker-ensure") from exc
    if provider != providers.RUNPOD or transport != providers.SSH:
        raise ReconcilableError(
            "worker {} is not a RunPod/SSH resource".format(worker_id),
            action="worker-ensure",
        )
    state = str((worker or {}).get("state") or "").upper()
    if state == "STOPPED":
        start = infra.worker_start(worker_id)
        if start.returncode != 0:
            raise ReconcilableError(
                "worker {} could not be restarted: {}".format(
                    worker_id, _failure_text(start)
                ),
                action="start-worker",
            )
    wait = infra.worker_wait_ssh(worker_id)
    if wait.returncode != 0:
        raise ReconcilableError(
            "worker {} did not become SSH-ready: {}".format(
                worker_id, _failure_text(wait)
            ),
            action="wait-ssh",
        )
    bootstrap = infra.worker_bootstrap(worker_id)
    if bootstrap.returncode != 0:
        raise ReconcilableError(
            "worker {} bootstrap failed: {}".format(worker_id, _failure_text(bootstrap)),
            action="bootstrap",
        )
    health = infra.worker_health(worker_id)
    if health.returncode != 0:
        raise ReconcilableError(
            "worker {} health check failed: {}".format(worker_id, _failure_text(health)),
            action="health",
        )
    payload = health.payload if isinstance(health.payload, dict) else {}
    if payload.get("ready") is False:
        raise ReconcilableError(
            "worker {} reports readiness {} rather than READY: {}".format(
                worker_id,
                payload.get("readiness_state") or "unknown",
                "; ".join(
                    "{}: {}".format(check.get("name"), check.get("detail"))
                    for check in payload.get("checks", [])
                    if check.get("status") != "PASS"
                ) or "no failing check was reported",
            ),
            action="health",
        )
    return None


def _stop_after_prepare_failure(infra, worker_id):
    result = stop_worker(infra, worker_id, reason="readiness failed")
    if result.returncode != 0:
        raise ReconcilableError(
            "worker {} failed readiness and could not be stopped; cleanup remains pending"
            .format(worker_id), action="stop-worker")


def stop_worker(infra, worker_id, *, reason, worker=None):
    """End one owned resource in the provider's own lifecycle.

    RunPod stop is reversible. Colab has no stop/resume state, so ending the
    allocation means terminal release (`worker destroy`).
    """

    lease = _lease_for(worker_id)
    if lease is None:
        raise CapacityError("worker {} has no scope lease; refusing to stop it".format(worker_id))
    workers = infra.worker_list()
    matching = worker or next((item for item in workers if item.get("id") == worker_id), None)
    if matching is None or not ledger.worker_matches_lease(matching, lease):
        raise CapacityError("worker {} no longer matches its recorded lease".format(worker_id))
    if ledger.lease_provider(lease) == providers.COLAB:
        return destroy_worker(infra, worker_id, reason=reason, worker=matching)

    spend_before = _accrued_for(workers, worker_id)
    result = infra.worker_stop(worker_id)
    if result.returncode == 0:
        ledger.close_lease(
            worker_id,
            cost_usd=spend_before,
            wall_clock_hours=_elapsed_hours(worker_id, workers),
            state="stopped",
        )
        state_module.append_event(
            {"worker_id": worker_id, "provider": providers.RUNPOD,
             "action": "worker-stopped", "reason": reason},
            kind="worker-stopped",
        )
    return result


def destroy_worker(infra, worker_id, *, reason, allow_adopted=False, worker=None):
    """Destroy/release one exact provider identity this backend created."""

    lease = _lease_for(worker_id)
    if lease is None:
        raise CapacityError("worker {} has no ARC creation lease".format(worker_id))
    if lease.get("provenance") != "arc" and not allow_adopted:
        raise CapacityError(
            "worker {} has no creation record; it may be stopped but not destroyed "
            "by automation".format(worker_id)
        )
    workers = infra.worker_list()
    matching = worker or next((item for item in workers if item.get("id") == worker_id), None)
    if matching is None or not ledger.worker_matches_lease(matching, lease):
        raise CapacityError("worker {} no longer matches its recorded lease".format(worker_id))
    provider = ledger.lease_provider(lease)
    spend_before = _accrued_for(workers, worker_id) if provider == providers.RUNPOD else None
    wall_clock = _elapsed_hours(worker_id, workers)
    result = infra.worker_destroy(worker_id)
    if result.returncode == 0:
        ledger.close_lease(
            worker_id, cost_usd=spend_before, wall_clock_hours=wall_clock,
            state="destroyed",
        )
        state_module.append_event(
            {"worker_id": worker_id, "provider": provider,
             "action": "worker-destroyed", "reason": reason},
            kind="worker-destroyed",
        )
    return result


def _lease_for(worker_id):
    for lease in ledger.all_leases():
        if lease.get("worker_id") == worker_id:
            return lease
    return None


def _accrued_for(workers, worker_id):
    for worker in workers:
        if worker.get("id") == worker_id:
            if providers.worker_provider(worker) != providers.RUNPOD:
                return None
            hourly = _decimal_or_none(worker.get("hourly_cost"))
            lease = _lease_for(worker_id) or {}
            created = (state_module.parse_timestamp(worker.get("last_started_at"))
                       if lease.get("closed_cost_usd") is not None else
                       state_module.parse_timestamp(worker.get("created_at")))
            if hourly is None or created is None:
                return None
            elapsed = state_module.utc_now() - created
            hours = Decimal(str(max(elapsed.total_seconds(), 0.0))) / Decimal(3600)
            prior = _decimal_or_none(lease.get("closed_cost_usd")) or Decimal(0)
            return prior + hourly * hours
    return None


def _elapsed_hours(worker_id, workers=None):
    lease = _lease_for(worker_id)
    if not lease:
        return None
    record = next((item for item in (workers or ()) if item.get("id") == worker_id), {})
    created = (state_module.parse_timestamp(record.get("last_started_at"))
               if lease.get("closed_wall_clock_hours") is not None else
               (state_module.parse_timestamp(record.get("created_at"))
                or state_module.parse_timestamp(lease.get("created_at"))))
    if created is None:
        return None
    elapsed = state_module.utc_now() - created
    prior = _decimal_or_none(lease.get("closed_wall_clock_hours")) or Decimal(0)
    return prior + Decimal(str(max(elapsed.total_seconds(), 0.0))) / Decimal(3600)


def _decimal_or_none(value):
    if value in (None, ""):
        return None
    try:
        return Decimal(str(value))
    except Exception:  # pragma: no cover - provider text we cannot parse
        return None


def _failure_text(result):
    text = (result.stderr or result.stdout or "").strip()
    return text[:400] if text else "no diagnostic text"


def request_from_plan(plan):
    """The non-secret provider/resource intent one plan declares."""

    if "worker" not in plan:
        raise ConfigurationError("the plan declares no worker request")
    request = dict(plan["worker"])
    request["provider"] = jobspec.plan_provider(plan)
    return request
