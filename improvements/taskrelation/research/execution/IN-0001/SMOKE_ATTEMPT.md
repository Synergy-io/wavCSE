# IN-0001 smoke attempt — 2026-10-04: blocked by a wavcse-infra reconciliation deadlock

**Outcome: `IN0001_SOURCE_DEFECT`.** The authorized smoke did not reach allocation.
No Colab session was created, no paid CU was consumed, no job was submitted, and
no live resource exists to release. The blocker is in the canonical
`wavcse-infra` control plane, not in the IN-0001 workload.

## What was verified before the attempt

| Item | Value |
| --- | --- |
| wavCSE commit attempted | `95dbb871299fe5942693bbb04f2c527795dae6a7` (published; also `379173c…` and `6c99692…`) |
| Authorization | `authorizations/IN-0001.yaml`, digest `5f91a87670e79c166d071d2907c6d6b6ac72d779ed3eabc4360a7462327a27a7`, expires `2026-10-04T19:22:01+00:00` |
| wavcse-infra binding | `main` @ `540b617f66d4fc8c11419fb606a64047f085d529`, `resolved_by: environment` |
| Provider state before | `worker list --read-only` = `[]`; `doctor` "balance 0.00 CU, assignments 0" |
| `plan` dry-run | exactly one job, `IN-0001__smoke__smoke__s00`, spec digest `d6cb89f92799ce0a87db3756b52c1800b5c9612ba309c1ac952566e263846a03` |

## The exact reproduction

```text
$ export WAVCSE_INFRA_CHECKOUT=~/projects/wavcse-infra
$ uv run --locked python -m improvements.compute worker-ensure \
      --scope IN-0001 \
      --plan improvements/taskrelation/research/execution/IN-0001/smoke_plan.json --json
{
  "error": "ReconcilableError",
  "reason": "the Colab allocation for IN-0001 returned 1 but no single new session
             identity is visible. The intent stays open and the request is never
             repeated; reconcile with `sweep` or `worker ensure`. Provider said:
             Infrastructure error: Colab has an owned lease or pending allocation:
             wavcse-7f8888cdffb84331"
}
```

The refusal originates in `wavcse-infra` **before any provider call**:
`ColabLifecycle.create` refuses whenever any Colab record exists that is not
`provider_absent` (`src/wavcse_infra/workers/colab.py:71-79`), and
`WorkerStateStore.record_colab_intent` refuses for the same reason
(`src/wavcse_infra/state.py:264-270`).

## The blocking record

`~/.local/state/wavcse-infra/workers.json` → `wavcse-7f8888cdffb84331`:

```json
{"provider": "colab", "infra_identity": "wavcse-7f8888cdffb84331",
 "create_pending": true, "provider_absent": false, "billing_mode": null,
 "readiness_state": "NOT_READY", "last_observed_state": "PROVISIONING",
 "last_observed_at": "2026-10-01T11:34:09.502028Z"}
```

It is an abandoned creation intent from **2026-10-01**, and the provider holds no
such session (`worker list --read-only` = `[]`; `doctor` reports `assignments 0`).

## Why no supported verb can retire it

Every path that could reconcile it explicitly refuses a `create_pending` record:

| Command | Behaviour |
| --- | --- |
| `infra worker list` (reconcile) | `reconcile()` skips a `create_pending` record missing from a *successful* provider listing (`state.py:508-513`): "A missing read is not proof that a pending paid allocation never happened." |
| `infra worker show <id> --read-only` | "Colab session `wavcse-7f8888cdffb84331` is absent from provider sessions" — it *knows* the session is gone |
| `infra worker destroy <id> --yes` | `ConfigurationError: Colab session … lacks confirmed ownership; refusing terminal release` (`cli.py:1014-1023` — `record.create_pending` short-circuits before the provider check) |
| `infra worker bootstrap <id>` | `WorkerBootstrapError: Colab … has no confirmed owned lease` (`workers/colab.py:199-203`) |
| `ColabLifecycle._release_confirmed` | `ProviderOperationAmbiguousError: … no confirmed owned identity for terminal release` |
| `ColabLifecycle.create` | `CostGuardError: Colab has an owned lease or pending allocation` |

There is no `worker reconcile` / `worker forget` verb, and `docs/OPERATIONS.md`
and `docs/COLAB.md` document no procedure for resolving an unresolved
allocation intent whose exact identity a successful provider read shows absent.
The record therefore blocks **every** future `worker create --provider colab`
permanently.

The defect is not in the safety rule (never assume a pending paid allocation
vanished) but in its having no bounded escape for the case the CLI itself
already establishes — a *successful* listing that does not contain the exact
identity, three days old. The read-only verbs observe absence; none may act on
it.

## Residual state (nothing live)

- Provider: zero Colab sessions, `assignments 0`, paid balance `0.00 CU` —
  confirmed before and after the attempt. Nothing was allocated, so nothing was
  billed and nothing needed releasing.
- wavCSE ledger: one non-ambiguous pending create intent for `IN-0001`
  (created `2026-10-04T13:22:59Z`, deadline `15:22:59Z`). It is bounded by
  `improvements.compute.reap --execute`, which abandons an intent older than
  `INTENT_ABANDON_AFTER_HOURS = 1` once the provider shows no matching worker.
- No run record entries, no job, no lease, no artifact, no S3 object.
- `authorizations/IN-0001.yaml` remains committed and expires unused at
  `2026-10-04T19:22:01+00:00`; only the human may withdraw it.

## What must happen before the smoke can run

1. A `wavcse-infra` maintenance increment must make an unresolved Colab
   allocation intent resolvable — for example, let `reconcile()` retire a
   `create_pending` record when a *successful, repeated* provider listing omits
   its exact identity beyond a bounded age, or expose an explicit
   `worker reconcile`/`worker forget` for an exact infra-owned identity, with a
   regression test. That is a change in the owning repository, not here.
2. After that fix and its validation, re-run the same committed IN-0001 workload
   unchanged (the plan, envelope and this study scope are all still valid).
