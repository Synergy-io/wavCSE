# Compute authorizations

One file per scope — `<SCOPE>.yaml`, named by the execution scope it grants
(a Study id such as `TR-0007`, or an infrastructure-validation scope such as
`IN-0001`) — is the human's written authority to spend money on that scope. It is
the *only* authority: the orchestrator consumes it and can never create, renew,
widen or lift it.

Nothing in this directory exists for a scope that has not been granted, and the
backend fails closed when it is absent: a scope without an envelope cannot
provision, and reports the block rather than improvising.

## Rules

- **Strict schema.** Unknown keys are rejected at every level, because a
  misspelled limit would silently not bind.
- **Policy only.** No worker or volume identifiers, no prices, no URLs, no
  credentials, no secrets. `network_volume_selector` chooses among volumes that
  are already tracked, by properties, at run time.
- **Committed.** The envelope is read from the committed revision. An envelope
  that differs from its commit, or that is not committed at all, is refused.
- **Frozen while busy.** While a scope still has live compute or jobs in
  flight, the digest the run started under cannot change: a new envelope for a
  busy scope is refused until the human re-grants it deliberately. That is what
  makes "the authorization cannot drift underneath a running job" enforceable.
- **Expiring.** `expires_at` is required, and an expired envelope is a stop —
  never a renewal.
- **One action at a time.** Every paid action is checked against the envelope
  immediately before it runs, with totals derived from provider state.

## Schema

```yaml
schema_version: 1
scope: TR-0007                     # scope id; the unit of authorization
scope_kind: study                   # optional; "study" (default) or "infrastructure_validation"
granted_by: <human>                # who is accountable for this grant
granted_at: 2026-09-29T12:00:00+00:00
expires_at: 2026-10-06T00:00:00+00:00

budget:
  max_gpu_hourly_usd: 0.50         # ceiling passed to the control plane's price guard
  max_total_gpu_usd: 8.00          # cumulative for the scope, from provider facts
  max_wall_clock_hours: 12         # cumulative paid wall clock

concurrency:
  max_simultaneous_workers: 1      # workers that still bill, not workers ever created
  replacement_workers_allowed: true

resources:
  existing_network_volume_allowed: true
  network_volume_selector:         # optional; picks among existing volumes
    min_size_gb: 100
  new_persistent_resources: false  # a new volume bills storage after compute stops
  container_disk_gb_max: 100

stop_policy:
  destroy_on_completion: true
  retain_for_reuse_hours: 0
```

An authorization is isolated by **both** the scope identity and the scope kind.
A plan may only spend the authorization it is named by, and only when the two
kinds agree: a Study authorization cannot be spent by an infrastructure-validation
plan, and an infrastructure-validation authorization confers nothing on any
Study — `IN-0001` never authorizes `DG-0008`. An unrecognised `scope_kind` is
refused, and a missing one is a legacy Study grant.

## How a grant is made
1. A human writes `<SCOPE>.yaml` from the schema above and commits it in the
   same change as the study's `PLAN.md` where possible.
2. The first action under that envelope records its digest in the
   controller-local state (`~/.local/state/wavcse-research/envelopes/`), so the
   run can be audited against exactly the grant it used.
3. Changing any field is a new grant: edit, commit, and re-run. Nothing about
   the change is automatic, and the backend will not act under a digest that
   differs from the recorded one while the scope is busy.

## Sizing

Take the study's own `compute_estimate` from `STUDIES.jsonl`, multiply by the
discovered hourly price for the resource the plan names, and add headroom for a
re-run. The point of the envelope is a bounded maximum, not a forecast: an
autonomous run may spend up to it and never past it.
