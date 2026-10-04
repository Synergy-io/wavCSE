---
description: Preflight one approved wavCSE proposal through the bounded Executor and Infrastructure Engineer loop
---

# Approved proposal execution preflight

Run Execution Plane V1 from one human-approved proposal to a durable final
preflight assessment. Stop before live compute.

Skills: wavcse-execution-plane
Boundaries: writes-reports, may-commit, no-paid-compute

## Entry gates

- proposal status is `APPROVED`, reviewer `PASS`, and human identity/timestamp exist;
- the proposal's exact SHA-256 can be bound;
- no scientific intent is supplied only in conversation.

## Steps

Follow `wavcse-execution-plane`. Persist artifacts under
`improvements/taskrelation/research/execution/preflights/<proposal-id>/`, validate
with:

```text
python -m improvements.taskrelation.research.execution_contract check <directory>
```

A `BLOCKED` final assessment is a valid preflight outcome. Do not make it pass by
registering a Study, creating authority, changing science or provisioning a
worker. Report the exact remaining blockers and stop.

## Never

Never run paid GPU compute, create a worker, execute a non-dry-run job, build a
Result Analyst, or use specialist transcript prose as the handoff.
