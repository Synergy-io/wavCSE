# DEV → CONTROLLER handoff — DG-0008 / Execution Plane V1

Deterministic handoff from the development computer to the controller. It
records only identities and unresolved bindings; it copies no artifact, carries
no credential, and grants no authority. Rewritten on the development machine on
2026-10-04 in the provider-neutral integration increment (the Colab seam is now
implemented and validated; see *wavCSE ↔ wavcse-infra integration status*); the controller
must check out the commit that contains this file — that commit or a descendant
— resolved with
`git log -1 --format=%H -- improvements/taskrelation/research/execution/CONTROLLER_HANDOFF.md`.

Nothing here authorizes compute. A preflight acceptance is not an authorization,
and an authorization is not a submission.

## wavCSE side

| Item | Value |
| --- | --- |
| Repository | `https://github.com/Synergy-io/wavCSE.git` |
| Branch | `feature/mssl-task-relation-study` |
| Approved proposal | `DP-0008` — `research/proposals/DG-0008_exact_matched_directed_transfer.md`, commit `83694cd965a3045ddf2f0bc8c8c9949a5abfaba2`, SHA-256 `c3548d7f9bae2881dcd833c9ddb19e5abb0df174ae76692dac79ab6b566572bc`, review `PASS`, approved by Kevin Sanjula `<kevinxsanjula@gmail.com>` 2026-10-03T16:18:45+00:00 |
| Registered Study | `DG-0008` — `research/studies/DG-0008/PLAN.md`, `NOTE.md`; registration commit `fa326fc39ca1bc2977d30a0964b80c493571e02b` |
| Implementation identity | `research/dg0008/` plus `studies/DG-0008/{run_dg0008.py,generate_opportunity_manifest.py,evaluate_dg0008.py,check_validity_gate.py,analyze_dg0008.py,configs/dg0008.yaml}` — commit `d9675688cdb3759778214509d3f6e7fa3348f429` |
| Compute inputs | `research/studies/DG-0008/compute/inputs.json` — 15 digest-pinned requirements (5 speechcommand, 9 voxceleb, 1 iemocap), validated by `improvements.compute.artifacts` |
| Compute plan | **Not authored — no longer for a schema reason.** The schema can now express a Colab-first plan (`provider: colab`, accelerator-only `worker` block) and the seam drives it. The residual blockers are authorization/benchmark inputs, not gaps: the pinned accelerator is an authorization-time decision under DG-0008's one-GPU-model invariant, `timeout_seconds` needs the compatible-accelerator benchmark, and the `outputs` staging names need one run of the study entrypoint to confirm. Authoring now would invent those values. |
| JobSpec schema | `improvements/compute/jobspec.py`, `SCHEMA_VERSION = 1`; stages may select an arm subset (200 + 1 topology), and a plan states its `provider` (missing = legacy RunPod) (`tests/test_stage_arms.py`, `tests/test_provider_neutral.py`) |
| Provider contract | `improvements/compute/providers.py` — provider ids, execution transports (`ssh`, `colab_exec`) and cost units (`USD/hour`, `CU`) shared with `wavcse-infra`; no CU→USD conversion exists |
| Scientific invariants | The sealed workload `execution/preflights/DP-0008/workload.v2.json` (26 invariants, one flexible choice `IF-MANIFEST-IO=STREAM_CANONICAL_BYTES`). Not duplicated here. |
| Execution contract | `improvements/taskrelation/research/execution/README.md`; validator `python -m improvements.taskrelation.research.execution_contract check <dir>` |
| Authorization | **None.** `research/authorizations/DG-0008.yaml` does not exist and must not be created by an agent. |
| Capability boundary | `AGENTS.md` "Capability boundary"; `.omp/config.yml` `task.isolation.{enabled:true,apply:false}`; gate `scripts/agents/candidate_gate.py` |

### Compute-plan fields: intent vs binding

| Field | Kind |
| --- | --- |
| `provider`, `study`, `arms[].{arm,method,argv,config}`, `stages[].{seeds,arms}`, `outputs`, `inputs_file`, `embedding_layout`, `repository`, `task_type`, `environment_secrets` | static scientific / execution intent |
| `worker.gpu_type` | accelerator requirement — bounded execution intent; for Colab it is the only worker field a plan may carry, and DG-0008's one-GPU-model invariant makes the choice an authorization decision |
| `worker.{cloud,image,template,container_disk_gb,volume_gb,data_centers,network_volume}` | RunPod-only placement/storage binding (offer, cloud tier and volume are discovered or chosen at authorization time) |
| `timeout_seconds` | controller-time binding (schema default until a compatible-GPU benchmark bounds it) |
| `device_index` | execution intent with a safe default |
| authorization `provider`, `budget.cost_unit`, `budget.{max_gpu_hourly_usd,max_total_gpu_usd}` (RunPod) / `budget.{allow_free_tier,max_incremental_rate_cu_per_hour,max_job_cu}` (Colab) | human authorization: the provider and its native unit; provider-specific budget fields are mutually exclusive |

### Required before any DG-0008 job

- Lawful restoration of the Speech Commands, VoxCeleb1 and licence-gated IEMOCAP
  corpora, and the frozen `dg0008.example-identity.v1` manifest + its digest.
- Every identity/opportunity manifest, run-level digest and the matrix digest
  (§5–§14 of the approved proposal), generated after lawful restoration. None
  exists and none may be fabricated.
- A compatible-GPU five-epoch benchmark bounding runtime for all four arm
  classes and the repeat.
- A human-bounded authorization envelope, then a fresh accepted preflight.

## wavcse-infra side

| Item | Value |
| --- | --- |
| Canonical repository | `https://github.com/Ke-vin-S/wavcse-infra.git` |
| Branch / commit | `main` @ `540b617f66d4fc8c11419fb606a64047f085d529` (verified `origin/main`, clean tree). `2d7640c7c6454b662ab92c6744beff946bc111fa` was the earlier verified binding and remains an ancestor of it; the record advanced on the controller when `540b617` was verified against the same compatibility checks. |
| Version | `0.1.0` — **unchanged** across the Colab commits; do not judge capability from the version string |
| Required compatibility | that exact commit, or a later verified `main` for which the compatibility checks there pass; bind it explicitly (see below) |
| Provider capability | **Colab (primary)** and **RunPod (secondary)** both implemented. Colab: ephemeral session, `colab_exec` transport, FREE_TIER / PAID_CU modes, guarded allocate/bootstrap/release, recorded jobs, GPU-driver `LD_LIBRARY_PATH` seeding. RunPod (REST v2): worker lifecycle, network volumes, S3 artifact storage, exact-commit jobs, direct SSH — unchanged. |

Zero-cost validation on the development machine was at `2d7640c`: `make check`
green (1100 unit tests, ruff format/lint, cloud-init schema, agent assets) and
`git diff --check` clean. The controller re-validated the advanced binding at
`540b617` the same way: `make check` green (1141 tests) plus ruff, cloud-init and
agent assets. No live provider call was made in either case.

### Colab semantics verified at this commit

Read from the implementation, not from prose:

- **One provider, two modes.** Colab is `ProviderKind.COLAB` over
  `ExecutionTransport.COLAB_EXEC`; `ColabBillingMode` is derived from the paid
  balance (`> 0` → `PAID_CU`, `== 0` → `FREE_TIER`), never a second provider.
- **Zero paid balance is not zero entitlement.** `balance == 0` selects
  best-effort free tier; allocation is permitted when `colab.allow_free_tier` is
  true (default). `allow_free_tier = false` rejects a zero balance before
  allocation. The observed CU/hour in free tier is recorded as metering, never
  treated as a paid cost or a rate ceiling.
- **Cost guards.** `PAID_CU` enforces `minimum_balance_cu`, an observed
  incremental rate in `(0, max_incremental_rate_cu_per_hour]`, and
  `max_job_cu` projected from declared runtime; `FREE_TIER` applies no rate or
  paid-balance gate. No CU→USD conversion is invented.
- **Allocation.** Pinned `google-colab-cli==0.7.4` over ADC; a unique
  `wavcse-<hex>` identity is recorded before `new`, then a
  `usage_after − usage_before` assignment-count check; ambiguous creation is
  reconciled by exact identity and never repeated; failed cost/bootstrap checks
  release the confirmed session.
- **Bootstrap / readiness.** Non-SSH health check (Python, Git, uv, physical GPU
  model, CUDA, scratch disk, network) plus installation of the SHA-verified
  runner; RUNNING is not READY.
- **Execution / jobs.** `colab_exec` runs the same reviewed `worker/job_runner.py`
  and Phase 5 transfer module: exact-commit detached checkout, SHA-256 input
  verification, presigned PUT outputs, controller-side S3 read-back and digest.
  `LD_LIBRARY_PATH` is seeded with `/usr/lib64-nvidia`; a spec may override it.
- **Placement.** `colab` before `runpod` unless `--provider` restricts or
  `--worker` pins. A research failure never triggers a cross-provider rerun.
- **Boundaries.** Colab has no SSH, no network volume, no data-center selection;
  `worker stop/start` are unsupported and release is terminal. RunPod keeps its
  USD/hour ceiling, SSH, volumes and lifecycle.

### Interface assumptions wavCSE makes

`improvements/compute/infra_cli.py` shells out to the resolved `infra` binary and
parses `--json` output. It assumes:

- `doctor`;
- `worker list --read-only --json`, `worker show <id> --read-only --json`,
  `worker create … --yes`, `worker wait-ssh`, `worker bootstrap`, `worker health`,
  `worker start/stop/destroy`;
- `volume list --read-only --json`, `volume cache stats --worker <id> --json`;
- `provider list`;
- `worker list [--provider {runpod,colab}] --read-only --json`,
  `worker show <id> --read-only --json`, `worker create [--provider …] …`,
  `worker bootstrap/health <id>`, `worker start/stop/destroy`;
- `job submit <spec> [--worker <id>] [--provider {runpod,colab}] --json`,
  `job status <id> --json`, `job list [--state] [--worker] --json`,
  `job logs <id> --tail-bytes N`, `job cancel <id> --json`;
- `storage read <artifact> --expected-sha256 … --json`, `storage verify`,
  `storage download`, `storage list`;
- the version-1 JobSpec JSON: `source{repository,commit}`, `command{argv}`,
  `setup{argv}`, `runtime{timeout_seconds,environment,environment_secrets}`,
  `inputs[{artifact,destination,required,sha256,size_bytes}]`,
  `outputs[{path,artifact,required,overwrite}]`, `tracking{metadata}`.

Resolution order is `WAVCSE_INFRA_CLI` → `WAVCSE_INFRA_CHECKOUT` → `<repo>/infra`
→ `infra` on `PATH` → legacy sibling `wavcse-infra`. Because `<repo>/infra`
precedes both `PATH` and the sibling fallback, a stale embedded copy can win
silently. Two hazards therefore remain:

- the `infra/` **committed inside wavCSE is a superseded fork** at v0.1.1 that
  implements only Phases 0–4 and cannot serve a job — and it is recognised as an
  ordinary monorepo subsystem, so `resolve()` selects it whenever its `.venv` is
  present. It must never be the resolved control plane for DG-0008. This is
  architectural debt to report, not to repair by mirroring it (no approved
  deprecation/removal path exists in this increment).
- the control plane must be **bound explicitly** to the standalone `wavcse-infra`
  checkout at the commit above; do not rely on `PATH` or a sibling fallback.
  Verified here: `resolve` with `WAVCSE_INFRA_CHECKOUT=<standalone checkout>`
  reports `resolved_by: environment` and the Colab-capable CLI, whereas without
  it the embedded fork is selected (and currently fails only because its `.venv`
  is absent).

### wavCSE ↔ wavcse-infra integration status

**Classification: `RESOLVED` in the provider-neutral integration increment** (the
independent `STALE_RESOLUTION_ONLY` concern — the embedded-fork preference — is
unchanged and remains architectural debt).

Implemented and validated on the development machine, with no provider call:

- `jobspec.plan_provider` reads an explicit plan `provider`; a missing value is
  deterministically RunPod, so legacy plans are byte-for-byte unaffected. A
  Colab plan carries `worker: {gpu_type?, gpu_count}`, must not carry RunPod
  fields (`cloud`, `image`, `template`, `container_disk_gb`, `volume_gb`,
  `network_volume`, `data_centers`), and the reverse holds for RunPod.
- `InfraCli.worker_create(provider=…)` sends only that provider's options:
  `--provider colab --gpu <accelerator> --yes` for Colab, and the unchanged
  RunPod argv (no `--provider`, since RunPod is the CLI default) otherwise.
  `worker_list(provider=…)`, `job_submit(…, provider=…)` and `provider_list()`
  complete the seam.
- `worker.ensure_worker` dispatches by provider: RunPod keeps its
  start → wait-ssh → bootstrap → health ladder and USD price guard; Colab
  allocates once, reconciles the new `wavcse-<hex>` identity by listing diff,
  records it on the lease, then runs bootstrap → health with **no** start/SSH.
  Colab readiness is read from exit status (`worker_health(..., json_output=
  False)`), because at the pinned commit the Colab branch ignores `--json`.
- Ownership is no longer name-only: `ledger.worker_matches_lease` requires the
  exact provider worker id **and** provider kind, so a Colab session is
  attributed by the lease wavCSE created, never by a scope prefix. Sweep,
  reaper, status and `available_worker_id` all use it.
- The envelope declares `provider` and `budget.cost_unit`; RunPod keeps
  `USD/hour` with hourly/total USD ceilings, Colab uses `CU` with
  `allow_free_tier`, `max_incremental_rate_cu_per_hour` and `max_job_cu`, and
  cross-provider budget fields are rejected. CU enforcement itself stays in
  `wavcse-infra`; the ledger never writes a fake USD value for CU and closes a
  Colab lease on wall-clock time in the CU domain.
- Cleanup: a Colab lease ends by terminal release (`worker destroy`), because
  Colab has no stop/resume state and no SSH.

Job identity, the arm argv, the spec schema and the stage topology are
unaffected by provider — placement is a submission-time choice
(`job submit --provider`/`--worker`).

Old status, preserved:

The provider-neutral surfaces match: `job submit/status/logs/cancel/list`,
`storage read/verify/list/download`, `worker list --read-only`, and the
version-1 JobSpec schema are satisfied by the Colab-capable CLI.

The **worker-acquisition path is RunPod-shaped and cannot address a Colab
worker**. This is wavCSE's seam, not a wavcse-infra defect, and it is not
repaired in this increment:

- `improvements/compute/jobspec.py::validate_plan` requires non-empty
  `worker.{gpu_type,cloud,image}` and has no provider field, so a plan cannot
  express a Colab worker request; `cloud`/`image` are RunPod-only and Colab
  rejects them;
- `improvements/compute/infra_cli.py::worker_create` always sends RunPod options
  (`--cloud`, `--gpu-count`, `--container-disk`, `--start-ssh`,
  `--require-direct-ssh`) and never `--provider colab`; the Colab provider raises
  `ConfigurationError` on any of those options;
- worker ownership is attributed by a `wavcse-<scope>-<nonce>` name
  (`improvements/compute/ledger.py`), but the Colab provider allocates its own
  `wavcse-<hex>` identity and rejects `--name`, so `available_worker_id` /
  `worker_belongs_to` never match a Colab session and `advance` cannot pick one;
- the readiness ladder calls `worker start` / `wait-ssh` / `bootstrap`, which
  Colab does not support, and the envelope budget is USD/hour while Colab free
  tier bills no CU and paid CU is deliberately never converted to USD.

Consequence, as of the previous increment: the RunPod path through
`improvements.compute` was intact and the Colab path was not drivable. That is
now superseded — the Colab path is drivable through the compute backend, and the
RunPod path retains its previous argv, ladder, price guard and tests.

## Controller-only checks

None of these can be performed on the development machine.

1. Check out the exact wavCSE commit that contains this handoff, and confirm
   `make check` is green there. (Development machine, on the commit that adds
   this note: green — 780 tests, 2 skipped, `agents-check` and
   `candidate-gate-check` included; `pdftotext`/`poppler-utils` must be
   installed for `test_literature_primary_text`.)
2. Confirm the project-local Research Computer resolves from the repository —
   agents `.omp/agents/*`, skills `.agents/skills/*`, commands
   `.agents/commands/*`, tools `.omp/tools/*`, policy `.agents/policies/*` — and
   that `task.isolation.enabled` is in place **before** OMP starts (a mid-session
   change does not reach an already-built task schema).
3. Resolve and verify the canonical `wavcse-infra` checkout by **exact commit**
   `540b617f66d4fc8c11419fb606a64047f085d529` (version `0.1.0`), not by version
   string and not by "`63c61af` or later": confirm `git -C <checkout> rev-parse
   HEAD` and a clean tree. (The earlier required commit was
   `2d7640c7c6454b662ab92c6744beff946bc111fa`; it is an ancestor of this one, and
   the controller record advanced to `540b617` after it passed the same checks.)
4. Bind the control plane explicitly — set `WAVCSE_INFRA_CHECKOUT` (or
   `WAVCSE_INFRA_CLI`) to that checkout so `python -m improvements.compute
   resolve --json` reports `resolved_by: environment` and never selects the
   embedded `wavCSE/infra` fork; then confirm `infra doctor`, `infra worker list
   --read-only --json` and `infra job list --json` against it.
5. Colab account/runtime state: ADC authentication present; `infra doctor`
   reports pinned CLI 0.7.4, session access, the current paid-CU balance and
   observed usage rate, and the current billing mode (FREE_TIER at balance 0
   when `colab.allow_free_tier`, otherwise the PAID_CU policy); confirm whether a
   free T4 is currently allocatable and the current paid cost if any. **Allocate
   nothing in preflight.**
6. Private S3 availability and canonical embedded-object verification against
   `studies/DG-0008/compute/inputs.json`.
7. Lawful raw-corpus availability (Speech Commands, VoxCeleb1, licence-gated
   IEMOCAP) and the independent raw-metadata identity extraction.
8. MLflow/DagsHub credentials; current RunPod price/availability only if the
   secondary provider is used.
9. Authorization envelope, compatible-GPU benchmark, and a fresh accepted
   execution preflight before any submission.
10. A bounded **live Colab smoke test** (see below), which cannot be performed on
    the development machine.

## What the controller must not need to copy

The controller obtains from wavCSE Git alone: the Research Executor,
Infrastructure Engineer, Research Designer, Research Reviewer and Literature
Agent definitions, every project skill and command, the project tools, the
capability policy, the candidate gate, and the execution contract. Machine-local
state stays machine-local: credentials, `wavcse-infra` configuration,
authentication and provider/SSH state.

## Superseded finding (chronology preserved)

The previous version of this handoff recorded, correctly for the history visible
at the time, that the canonical `wavcse-infra` `main` at
`63c61af1ed70cd283b03438ec9ec11cf4166541b` contained no Colab provider, and left
the Colab question open. That finding is superseded, not erased:

- **then (previous audit):** `main @ 63c61af`; Colab absent from the visible
  history, because controller-side commits were unpushed.
- **event:** those controller commits were pushed and pulled.
- **now (this audit):** `main @
  2d7640c7c6454b662ab92c6744beff946bc111fa`; Colab is implemented and reachable
  from the canonical branch (commits `ff3125e` … `2d7640c`). The open decision is
  therefore **resolved by evidence**: Colab is implemented and now canonical.

The remaining question is not whether Colab exists but whether wavCSE can drive
it — see *wavCSE ↔ wavcse-infra integration status* (`WAVCSE_ADAPTER_MISMATCH`).

## Controller-side record — execution-scope increment (2026-10-04, no live action)

Performed on the controller, in this order, with no provider call:

1. **Bindings verified.** wavCSE HEAD `6c99692d06fdeed8bdf12b6e07b0909102f2a032`
   on `feature/mssl-task-relation-study` (pushed; the handoff-bearing commit),
   clean tree. `wavcse-infra` `main` @
   `540b617f66d4fc8c11419fb606a64047f085d529`, clean, version `0.1.0`, a verified
   descendant of `2d7640c`. Bound with `WAVCSE_INFRA_CHECKOUT`; `resolve --json`
   reports `resolved_by: environment`.
2. **Preflight.** `make check` green at both repositories (wavCSE 780 tests;
   wavcse-infra 1141). All 15 digest-pinned DG-0008 inputs exist in S3 with the
   declared sizes. `doctor` reports Colab `READY` (ADC, CLI 0.7.4, free tier,
   balance 0.00 CU, assignments 0). `worker list --read-only` shows no workers
   and `reap` (dry-run) no attributable compute. `DP-0008-N01` still validates as
   `BLOCKED`; DG-0008 still has no plan and no authorization.
3. **Two contract facts confirmed.**
   - `WAVCSE_INFRA_CHECKOUT` is **not persisted** on the controller, and the
     embedded `infra/` (v0.1.1) wins resolution ahead of `PATH` whenever its
     virtual environment exists. Without the explicit binding, resolution
     currently fails closed rather than silently selecting the fork — and the
     smoke procedure must not depend on that accident.
   - Every authorized execution scope was assumed to be a Study id. That is now
     corrected by the execution-scope-kind contract (below).
4. **Execution-scope kinds implemented (wavCSE side only).** A scope is an
   identity plus a kind: `study` (default, legacy-compatible) or
   `infrastructure_validation`. Plans carry `scope` + `scope_kind` for the
   non-Study kind; envelopes may carry `scope_kind`; identity and kind are
   compared together and a mismatch fails closed. Provider-mutating verbs
   (`worker-ensure`, non-dry `advance`) now require an explicitly bound control
   plane; cleanup verbs are deliberately never gated this way. Full contract:
   `../../../compute/README.md` §"Execution scope".
5. **IN-0001 added.** `execution/IN-0001/{README.md,smoke_plan.json}` — an
   infrastructure-validation scope that exercises
   `improvements.compute → external wavcse-infra → provider=colab → one T4 →
   bootstrap → one trivial recorded exact-commit job → one tiny artifact → S3
   read-back → release`. It uses no DG-0008 data, no embeddings, no corpus and no
   MLflow credential, and it confers no scientific authority. Run it only after a
   human commits `authorizations/IN-0001.yaml`.

Nothing here authorizes compute; no Colab or RunPod resource, job or
authorization was created, and no DG-0008 work was run.

## Controller-side record — IN-0001 smoke attempt (2026-10-04): BLOCKED, wavcse-infra defect

The authorized IN-0001 smoke was attempted and **did not reach allocation**
(`IN0001_SOURCE_DEFECT`). Full attempt record:
`execution/IN-0001/SMOKE_ATTEMPT.md`. In brief:

- Prerequisites succeeded: `379173c…`/`95dbb87…` published and anonymously
  reachable; `WAVCSE_INFRA_CHECKOUT` bound to
  `540b617f66d4fc8c11419fb606a64047f085d529` (`resolved_by: environment`); the
  human's `authorizations/IN-0001.yaml` committed and published (digest
  `5f91a876…27a7`, expiring `2026-10-04T19:22:01Z`).
- `improvements.compute worker-ensure --scope IN-0001` was refused by
  `wavcse-infra` **before any provider call**: an abandoned Colab creation
  intent from **2026-10-01** (`wavcse-7f8888cdffb84331`, `create_pending: true`,
  `provider_absent: false`, `PROVISIONING`) blocks every Colab allocation.
- No live resource existed before or after: zero provider sessions,
  `assignments 0`, paid balance `0.00 CU`. Nothing was allocated, billed,
  submitted or released; no artifact or job was produced.
- The blocker is a `wavcse-infra` reconciliation deadlock, not a wavCSE defect:
  `worker list` (reconcile) skips a `create_pending` record missing from a
  *successful* provider listing, while `worker destroy`/`bootstrap`/`show` and
  `ColabLifecycle._release_confirmed` all refuse a `create_pending` record — so
  an intent whose exact identity a successful read shows absent can never be
  retired by any supported verb, and permanently blocks Colab. Fixing it is a
  `wavcse-infra` maintenance increment; nothing was patched during the live run.
- Residual wavCSE state: one bounded, non-ambiguous pending create intent for
  `IN-0001`, self-cleaned by `improvements.compute reap --execute` after
  `INTENT_ABANDON_AFTER_HOURS = 1`. No DG-0008 work was performed and the
  DG-0008 benchmark was not started.

## Live-validation requirement

Deterministic, zero-cost coverage of the Colab path is strong (`make check`,
1100 tests: provider selection and placement order, free-tier vs paid-CU gates,
zero-balance behaviour, allocation identity/usage-delta/ambiguity, bootstrap,
release, failure cleanup, `colab_exec` recorded jobs, `LD_LIBRARY_PATH` seeding,
RunPod non-regression). Two residual coverage gaps were identified: the CLI
`_job_context_for` → `colab_exec` dispatch is exercised only with that selection
monkeypatched away, and the `_require_colab_job_budget` pre-cost gates lack a
direct test. No test allocates a live session.

A bounded **controller-side live Colab smoke test** is therefore still required
before any Colab-first execution claim: allocate one ephemeral T4 under the
configured policy, bootstrap to READY, run one trivial recorded job through the
`infra` CLI, verify the S3 read-back, and release — recording the exact
`wavcse-infra` commit and the observed billing mode. It is **not** performed in
this increment, and no live provider work is authorized by this handoff.

That smoke test can now be driven through either the `infra` CLI directly or
`improvements.compute` (`worker-ensure` → `advance`), which owns Colab
acquisition, attribution and release; the direct CLI path remains the smaller
first step, and it validates the same `wavcse-infra` contract the backend calls.

A compatible-GPU five-epoch benchmark remains the next paid/live scientific
action; it is neither authorized nor performed here.
