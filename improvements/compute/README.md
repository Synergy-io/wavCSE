# Compute backend

The seam between wavCSE's research orchestration and the infrastructure control
plane. It exists so that a research cycle can acquire authorized compute,
submit exact-commit jobs, verify their outputs and stop paying for them without
a human changing directory — and without wavCSE learning anything about
providers, SSH, S3 or job state beyond the control plane's CLI contract.

```
/wav-cycle  ->  improvements.compute  ->  (subprocess, --json)  ->  infra CLI
```

## What it does not do

- It never imports the infrastructure package as a library, and never calls a
  provider or an AWS API directly.
- It never implements worker lifecycle, storage transfer, caching, or job state;
  it drives the control plane's CLI and reads its JSON.
- It never moves artifact bytes. It declares required inputs (digest-verified)
  and required outputs; the control plane materializes and persists them.
- It never writes to Git. Research intent is committed by the cycle; runtime
  facts stay under the controller's state directory.

## Verbs

```
python -m improvements.compute status          --scope TR-0007 [--json]
python -m improvements.compute resolve         [--json]
python -m improvements.compute envelope-check  --scope TR-0007 --action submit-job
python -m improvements.compute sweep           --scope TR-0007 [--execute]
python -m improvements.compute plan            --scope TR-0007 --plan P --stage screen
python -m improvements.compute worker-ensure   --scope TR-0007 --plan P
python -m improvements.compute advance         --scope TR-0007 --plan P --stage screen [--dry-run]
python -m improvements.compute collect         --scope TR-0007 --plan P --stage screen
python -m improvements.compute finish          --scope TR-0007 --plan P
python -m improvements.compute stop            --scope TR-0007 [--destroy]
```

Exit codes: `0` proceed, `2` usage or configuration failure, `3` refused by
policy. `status`, `resolve`, `envelope-check`, `plan` and `--dry-run` are
read-only and never write runtime state.

## Where the control plane is found

Resolution order, first validated candidate wins: an explicit checkout
(`WAVCSE_INFRA_CHECKOUT`) or CLI (`WAVCSE_INFRA_CLI`), then the `infra`
executable on `PATH` resolved through its symlink, then a sibling checkout. When
nothing validates, compute steps stay blocked with an explicit reason — never
silently skipped, never substituted.

## Runtime state (never committed)

Everything under `~/.local/state/wavcse-research/` (override with
`WAVCSE_RESEARCH_STATE` for tests):

| Path | Contents |
|---|---|
| `envelopes/<SCOPE>.json` | the digest each run acted under, with history |
| `runs/<SCOPE>.json` | per-stage job entries: identity, attempts, state, verification |
| `leases.json` | creation intents and worker leases, including deadlines |
| `specs/` | the exact job specs submitted, for auditing a run |
| `compute-events.jsonl` | append-only audit trail of every action and refusal |

Worker discovery uses a scope prefix plus a unique create-intent nonce. A
matching name alone is insufficient to start or destroy a worker: ARC also
requires its recorded creation lease.

## Accounting

Spend is derived, never self-reported: each scope worker's provider-reported
hourly price and observed lifetime are summed from `infra worker list --read-only --json`,
and a lease that this backend closed contributes its frozen cost. Every figure
is labelled an estimate. An unknown price or an unknown creation time makes the
total unbounded, and unbounded totals fail closed for new spend.

## Safety properties

- A create intent is persisted **before** the billable request; an ambiguous
  outcome is reconciled by the generated name, never re-requested.
- `--max-price` carries the envelope's hourly ceiling into the control plane's
  own guard, and the observed price is re-checked afterwards.
- A job is submitted only when nothing else for that scope is in flight, and is
  keyed by a deterministic identity (scope, name, commit, arm, seed) so a lost
  acknowledgement is adopted rather than duplicated.
- A recorded job is submitted only from a commit-clean tree.
- Retry budgets and attempt counts live in the run ledger, so a restart cannot
  reset them.
- The sweep is dry-run by default, never destroys, never touches a network
  volume, and never touches a worker it cannot attribute to a scope.
- Failure classes, retry bounds and the "OOM never changes the science" rule are
  implemented in `failures.py` and stated in `.agents/policies/autonomy.md`.

## Integration limits found in adversarial review

Do not treat this backend as approval for a paid autonomous cycle yet. Its
deadlines are enforced when the controller runs a transition or sweep; there
is no independent reaper after a controller crash. The declared input files
are placed under the job workspace, while current embedding loaders expect an
unpacked tree at the configured embedding root. Output records prove stored
bytes, but ARC does not yet parse the stored manifest or metrics before
collection and destructive cleanup. These gaps need a zero-cost integration
design and tests before an authorization is granted.

## Study compute plan

Each study that will run on compute carries `studies/<ID>/compute/plan.json`:
the study and stage seeds, the arms with their argv, the required inputs file
and outputs, the worker request, and the timeout. It is committed with the
study, so the job spec is derived from committed state and is byte-identical
when regenerated.
