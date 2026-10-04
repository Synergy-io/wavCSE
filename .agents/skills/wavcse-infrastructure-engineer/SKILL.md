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

**Colab capability is canonical (corrected 2026-10-04).** An earlier record on
this skill stated that the canonical `wavcse-infra` `main` exposed only the
RunPod provider. That was true of the history visible at the time
(`63c61af1ed70cd283b03438ec9ec11cf4166541b`): controller-side commits had not
been pushed. They have since been pushed and pulled, and the canonical `main` is
now `2d7640c7c6454b662ab92c6744beff946bc111fa` — version `0.1.0`, unchanged —
with the Colab provider, the `colab_exec` execution transport and Colab
session/job support reachable from the canonical branch. Do not repeat the
earlier "Colab absent" claim as current state. Because a capability change need
not carry a version bump, inspect the exact bound checkout for commands, options
and version-dependent behaviour rather than reading them from this skill.

**Known wavCSE-side integration gap (recorded 2026-10-04).** The consumption
seam (`improvements/compute`) is RunPod-shaped and cannot express or address a
Colab worker: `validate_plan` requires `worker.{gpu_type,cloud,image}` and has no
provider field; `infra_cli.worker_create` always sends RunPod options
(`--cloud`, `--gpu-count`, `--container-disk`, `--start-ssh`,
`--require-direct-ssh`); worker ownership is attributed by a
`wavcse-<scope>-<nonce>` name while the Colab provider allocates its own
`wavcse-<hex>` identity and rejects `--name`; and the envelope budget is
USD/hour while Colab free tier bills no CU and paid CU is never converted to
USD. The job, storage, status and list surfaces are provider-neutral and match.
Do not assert that wavCSE can drive Colab until a separate bounded increment
extends this seam; that gap is wavCSE's, not a wavcse-infra defect.

## Provider semantics (stable concepts, not CLI flags)

- **Selection.** Provider selection follows configured placement (Colab primary,
  RunPod secondary); a job may be restricted to one provider or pinned to one
  exact worker. Never hardcode a provider into a wavCSE consumer.
- **Allocation / session lifecycle.** Colab is an ephemeral, non-resumable
  session (allocate → observe → bootstrap → run → release); RunPod is a Pod with
  a reversible stop and network-volume persistence. `worker stop/start` are
  unsupported for Colab; release is terminal.
- **Bootstrap.** Readiness is proven by inspection, not by lifecycle state:
  Python, Git, uv, physical GPU model, GPU library visibility, scratch disk and
  network are checked before a worker is READY.
- **Execution transport.** Transport is chosen per worker identity, independently
  of provider: RunPod uses SSH, Colab uses `colab_exec` (upload, execute, collect).
- **Status.** Worker and job state are read from the provider as authoritative;
  local records are reconciled against it, never trusted alone.
- **Artifact handling.** Inputs are digest-verified before launch; outputs are
  persisted through the storage layer and read back before success.
- **Cleanup / reconciliation.** Ownership is tracked; an ambiguous create is
  reconciled by exact identity, never blindly repeated; absence is marked only
  after provider confirmation.
- **Authorization.** The envelope and `authorizations/<SCOPE>.yaml` gate spend;
  the compute backend is the only agent-facing paid-operation seam.
- **DEV vs CONTROLLER.** The development machine holds no provider credential and
  performs no live provider action; controller-only state (credentials, config,
  account/runtime facts) is bound there and never copied into Git.

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
