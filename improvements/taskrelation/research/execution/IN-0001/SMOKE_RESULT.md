# IN-0001 smoke retry — 2026-10-04: FAILED at Colab readiness, session released

**Outcome: `IN0001_EXECUTION_FAILED`.** The retry passed every prerequisite, was
granted exactly one free-tier T4, and then failed during bootstrap readiness on a
Colab `exec` timeout. The session was released and its absence confirmed; no job
was ever submitted, no artifact was produced, and no scientific or benchmark work
was touched.

This file is the append-only record of the retry. It is not a rewrite of
`SMOKE_ATTEMPT.md` (the earlier, differently-caused failure).

## Prerequisites, all satisfied

| Item | Value |
| --- | --- |
| wavCSE commit (published, tree clean) | `1c48e5c2730d95fd6a03e52a5837d7d2f593c9ca` |
| wavcse-infra binding | `~/projects/wavcse-infra`; `resolve --json` → `resolved_by: environment` |
| wavcse-infra commit | `fc85441d020386b7e18bb3df2e60585b19b4be48` (HEAD = origin/main = remote main, tree clean) |
| Authorization | `IN-0001.yaml`, digest `5f91a87670e79c166d071d2907c6d6b6ac72d779ed3eabc4360a7462327a27a7` = its committed digest, unmodified, unexpired (`2026-10-04T19:22:01Z`) |
| Plan | unchanged: `IN-0001`, kind `infrastructure_validation`, provider `colab`, `T4 × 1`, one stage `smoke`, one arm, seed 0 |
| Stale intent | abandoned by the bounded reaper at `14:23:27Z` at age 1.0065 h (threshold 1 h) |

## What happened

```text
14:23:27Z  reaper: worker-create-abandoned (age 1.0065 h)  -> IN-0001 unblocked
14:24:17Z  worker-create-intent recorded (retry)
           ~ the first worker-ensure call was interrupted after the provider request
14:30:33Z  worker-lease: adopted the exact new session wavcse-fc5dbf40f42e4a66
           (single new identity, reconciled by exact identity - the create was never repeated)
14:30:33Z..14:36:26Z  bootstrap/readiness on the T4
14:36:26Z  bootstrap failed: "Colab exec for wavcse-fc5dbf40f42e4a66 timed out;
           no provider state was inferred"
14:36:26Z  session released ("Colab readiness failed"); absence confirmed
```

The failing step is `ColabLifecycle` readiness: the non-SSH health `exec` did not
return within the CLI's `command_timeout_seconds` (300 s), and the control plane
correctly refuses to infer provider state from a timed-out exec. The prepared
session was then released through the ownership-guarded path, which confirms
absence before marking the record destroyed.

Nothing was submitted: `job list` shows zero IN-0001 jobs and no run record was
created, so the `colab_exec` job-transport dispatch was not reached this time.

## Final state (nothing live, nothing open)

| Fact | Value |
| --- | --- |
| Provider sessions (authoritative `sessions` listing) | `[]` |
| Colab `usage` aggregate | paid balance `0.00 CU`, observed rate `1.07 CU/hour`, `assignments 1` — still 1 at `14:37:53Z` and `14:39:01Z` (~2.5 min after release) while the authoritative `sessions` listing stayed empty |
| Lease | `destroyed`, `closed_at 2026-10-04T14:36:26.832303Z`, `closed_wall_clock_hours 0.0944`, `closed_cost_usd null` (CU domain — no USD invented) |
| Pending create intents | none |
| Open jobs / attempts | none (`runs/IN-0001.json` does not exist) |
| Billing | FREE_TIER throughout; balance never left `0.00 CU`; no paid CU was consumed or required |

The single lease consumed ~5.7 minutes of free-tier session time, recorded as
provider metering, never as a paid cost.

## Why no second session was attempted

The authorization bounds this run to **one worker, no replacements, no second
Colab session**, and `concurrency.replacement_workers_allowed: false`. The single
permitted session existed and was released, so a retry would require a new grant
decision rather than a silent re-allocation. Recorded here instead of acted on.

## Observations worth carrying into any next attempt

1. **Free-tier readiness budget.** The health `exec` (which imports torch and may
   install `uv` via pip) did not complete inside the 300 s command timeout on a
   cold free-tier runtime. That is a provider/runtime timing condition, not a
   guard defect, and it is not evidence of a source defect — the command timeout
   is a deliberate bound, and the refusal to infer state from a timed-out exec is
   correct. If a next attempt is authorized, the timeout is the first parameter to
   review against the observed cold-start cost.
2. **Usage-counter lag.** The aggregate `usage` report still shows `assignments 1`
   and rate `1.07 CU/hour` while the authoritative `sessions` listing is empty.
   The create path attributes ownership by `usage_after - usage_before == +1`, so a
   next attempt should wait for that counter to settle before allocating; an
   unsettled counter can make a correct allocation look unattributed.
3. The bounded recovery added in `fc85441` was not exercised again here; the
   reaper, not `worker reconcile`, cleared the stale intent, which is the intended
   ordering (the reaper owns age-based abandonment).

## What would be required to retry

A fresh human decision, because the remaining blocker is authorization scope
(one worker, no replacements), not capability: the repaired control plane
allocated a T4 successfully, and the failure was in readiness timing.

---

# Retry 2 — 2026-10-04 14:45–14:50Z: PASSED

The researcher authorized one further bounded free-tier session. Everything below
is the second attempt; the sections above are the first attempt and are unchanged.

## Preconditions and the one change made

| Item | Value |
| --- | --- |
| wavCSE commit executed | `4522ee2c1e78f5556294f8078feaf63639dfbbd3` (requested == executed) |
| wavcse-infra commit | `fc85441d020386b7e18bb3df2e60585b19b4be48` (HEAD = origin/main, clean) |
| Previous session | absent from the authoritative listing; aggregate counter had settled to `assignments 0`, rate `0.00 CU/hour` |
| Authorization | **retry grant**, digest `a743ab5747ff93bf2ed733809806347b8194503cb1255d1ccbe7f6f1a913d13a`, commit `4522ee2`, window `14:45:17Z–20:45:17Z`. The first grant (`5f91a876…27a7`, commit `95dbb87`, `13:22:01Z–19:22:01Z`) was consumed by the first attempt and is preserved here and in Git history, not extended. |
| Readiness timeout | `colab.command_timeout_seconds` **300 s → 600 s** in the controller TOML (operator configuration surface; no source change). Reason: four earlier free-tier T4 sessions reached READY under the same code with the 300 s default, so the failure was cold-start latency; 600 s doubles the observed ceiling and stays far inside the 2 h wall-clock envelope. |

## Result

```text
14:46:44Z  session wavcse-e14e1496396e4df8 allocated (T4, free tier); readiness READY in ~83 s
           observed in-environment: gpu "Tesla T4", billing FREE_TIER, rate 1.07 CU/hour (metering)
14:48:28Z  job job-8d23d42086ea497c submitted to that session (provider colab, colab_exec)
14:48:28Z  job SUCCEEDED, exit 0; requested_commit == executed_commit
14:49:24Z  collected: outputs persisted and independently read back
14:49:42Z  session terminally released; absence confirmed
```

| Evidence | Value |
| --- | --- |
| Physical GPU (from inside the environment) | `Tesla T4` |
| Billing | `FREE_TIER`; paid balance never left `0.00 CU`; `closed_cost_usd` `null` (CU domain) |
| Readiness | `READY`, `bootstrap_version colab-1`, `/content` 75 GB free |
| Job identity | `job-8d23d42086ea497c`, name `IN-0001__smoke__smoke__s00`, `SUCCEEDED`, exit `0` |
| Commit | requested == executed == `4522ee2c1e78f5556294f8078feaf63639dfbbd3` |
| Transport | job directory `/content/.wavcse/jobs/job-8d23d42086ea497c`, provenance `provider: colab`, `worker_id wavcse-e14e1496396e4df8` — the non-SSH `colab_exec` dispatch path, exercised for real |
| Job-budget gate | `_require_colab_job_budget` ran on the submit path: record READY, no active job, baseline assignments `0` vs observed `1`, FREE_TIER + `allow_free_tier` → permitted |
| Artifact | `IN-0001/smoke_s00/smoke_probe.txt`, sha256 `2a7de24fb7bb2d55c87f753f3bcbc648d3f5dafa2f62b917e01bc5c3d68e8540`, 295 bytes |
| S3 persistence + read-back | the control plane verified the stored object; an independent digest-bound `storage read` returned the same digest, and the bytes name the executed commit, GPU, provider, worker and job |
| Lease | `wavcse-e14e1496396e4df8` `destroyed`, closed `14:49:42Z`, wall clock `0.0465 h`, `jobs: [job-8d23d42086ea497c]` |
| Provider afterwards | sessions `[]`, `assignments 0`, rate `0.00 CU/hour` (settled immediately this time) |
| Open state | none: no pending create intents, no open jobs, one attempt, entry `collected` |

Artifact bytes recovered from S3 (independent of the job record):

```json
{"commit": "4522ee2c1e78f5556294f8078feaf63639dfbbd3", "gpu": "Tesla T4",
 "job_id": "job-8d23d42086ea497c", "kind": "infrastructure_validation_probe",
 "protocol": "in0001-smoke-v1", "provider": "colab", "schema_version": 1,
 "seed": 0, "worker_id": "wavcse-e14e1496396e4df8"}
```

**IN-0001 is successfully executed.** The smoke exercised the real
`improvements.compute → wavcse-infra @ fc85441 → Colab → colab_exec → S3`
integration seam end to end, including both previously unvalidated paths, with no
research data, no embedding, no training, no benchmark, no RunPod and no paid CU.
The retry grant is spent; it remains committed and unexpired until
`2026-10-04T20:45:17Z` but must not be drawn on again, and no further session may
be allocated for it.

### Post-release counter settlement (bounded read-only window)

The aggregate `usage` counter lagged again after release while the authoritative
listing was empty, so it was re-read on the documented bounded procedure rather
than treated as proof of a live worker:

| Sample (UTC) | Authoritative `sessions` | observed rate | assignments |
| --- | --- | --- | --- |
| 14:52:12Z | `[]` | 1.07 CU/hour | 1 |
| 14:53:39Z | `[]` | 1.07 CU/hour | 1 |
| 14:55:04Z | `[]` | 0.00 CU/hour | **0** |
| 14:56:31Z | `[]` | 0.00 CU/hour | 0 |
| 14:57:59Z | `[]` | 0.00 CU/hour | 0 |

It settled to `assignments 0` ~5.4 min after release — the same order as the
first attempt's lag — and held for three consecutive samples with the listing
empty throughout. Nothing was allocated, and no release was needed, in that
window; the lag is aggregate-report latency, not a live session.


