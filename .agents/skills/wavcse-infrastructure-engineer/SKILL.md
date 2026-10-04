---
name: wavcse-infrastructure-engineer
description: Assess or maintain wavCSE execution infrastructure while deterministic compute and infra operations remain authoritative.
---

# wavCSE Infrastructure Engineer

Use this skill for the infrastructure half of Execution Plane V1. Read
`infra/AGENTS.md` before consequential infrastructure work. The normative plane
contract is `improvements/taskrelation/research/execution/README.md`.

## Providers and the canonical control plane

`wavcse-infra` — the separate infrastructure repository — owns the deterministic
implementation: provider integrations, worker/session lifecycle, execution
transport, artifact movement and infrastructure operations. wavCSE owns only the
consumption contract (`improvements/compute` and this plane). Never reimplement
provider mechanics here. Never assume the `infra/` copy committed inside wavCSE
represents the current `wavcse-infra`: establish the canonical repository, branch
and commit first, and read exact commands, options and version-dependent
behaviour from the compatible installed `wavcse-infra` interface rather than from
prose.

Provider priority is a project decision:

- **PRIMARY: Colab.**
- **SECONDARY: RunPod.**

Provider-neutral semantics — allocation, bootstrap, execution, status, artifact
handling, reconciliation, release — belong in the common contract. Provider-
specific semantics stay provider-specific: do not force Colab through RunPod
concepts such as a network-volume requirement, RunPod offer discovery or direct
SSH, and do not remove RunPod behaviour merely because Colab is primary.

**Known discrepancy (recorded 2026-10-04).** The canonical `wavcse-infra`
checkout inspected on this machine (`main`) exposes only the RunPod provider; no
Colab provider, runtime or transport was present. Do not assert Colab
availability from this skill. Treat "which repository and commit carries Colab"
as a controller-side preflight check, and if it cannot be established, a human
decision — not something to invent.

## OPERATE

1. Validate the workload, proposal digest and negotiation state from durable
   files. Never depend on Executor transcript prose.
2. Resolve the control plane with `python -m improvements.compute resolve --json`
   and identify its exact commit/tree state/version.
3. Inspect deterministic status, provider/resource facts, authorization and
   existing plan. Read-only or dry-run operations first.
4. Assess feasibility, resource shape, runtime/cost bounds, bottlenecks,
   deterministic telemetry and cleanup policy.
5. Return one typed decision. A flexible request must name the workload choice
   id and allowed replacement value. An invariant target is scientific change.
6. After `ACCEPT`, any future paid action still goes only through
   `improvements.compute`; the assessment itself grants nothing.

Never call provider APIs directly or use ad-hoc SSH/shell as infrastructure
state. `improvements.compute` owns the envelope, spend, deterministic identity,
ledger, reconciliation and evidence gate; `infra` owns provider/worker/storage/job
mechanics.

## MAINTAIN

1. Persist `BLOCKED`/paused state and failure evidence.
2. Reproduce the infrastructure defect outside live scientific execution.
3. Fix only the owning infra seam, add a regression test and run its gate.
4. Commit the fix and record its full commit in a MAINTAIN assessment.
5. Re-evaluate runtime state explicitly. Reconcile/reprovision/resume is a new
   OPERATE decision; it is never implicit continuation.

A dirty tree cannot identify the infrastructure version that operated a Study.

## Monitoring and cleanup

Observation is deterministic; reasoning is agentic. Use job state, liveness,
heartbeat/progress, logs, worker readiness, GPU capacity, disk, artifact state,
runtime and derived cost. Missing utilization samples are unknown. Request the
smallest deterministic telemetry addition only when a decision needs it.

The agent may release early, but safety does not depend on memory: leases,
deadlines, envelope ceilings, finish/sweep and the crash-independent reaper own
cleanup. Never destroy unowned resources or canonical storage.

## Escalate

Escalate scientific changes, new/wider authority, budget/policy decisions,
persistent disagreement, blockers and the convergence limit. Ordinary flexible
optimization returns directly to the Research Executor.
