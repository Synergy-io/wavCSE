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
