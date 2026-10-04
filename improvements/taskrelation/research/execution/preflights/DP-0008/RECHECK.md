# DP-0008 / DG-0008 — execution-preflight re-check (2026-10-04)

Status: **BLOCKED — re-checked, not accepted.** No `ACCEPT` was forced, no workload
revision was faked, no provider was called and no paid action occurred.

This file is a durable re-check record and is not part of the typed contract
directory (`check_directory` reads `*.json` only). The sealed artifacts
(`workload.v1.json`, `workload.v2.json`, `assessment.r1.json`, `assessment.r2.json`,
`negotiation.json`) are unchanged by this increment.

## Commits re-checked

| What | Commit |
|---|---|
| Study registration (`STUDIES.jsonl`, `studies/DG-0008/{PLAN.md,NOTE.md}`) | `fa326fc39ca1bc2977d30a0964b80c493571e02b` |
| Approved experiment implementation and Study records | `d9675688cdb3759778214509d3f6e7fa3348f429` |
| Embedded-infrastructure CLI contract repair, `infra` v0.1.1 | `a7ee886a8c0947f82b1079802b53c0507d7e6ec8` |

Approved proposal binding re-verified by the validator itself
(`_validate_proposal` re-derives the reference from Git at the recorded commit):
`83694cd965a3045ddf2f0bc8c8c9949a5abfaba2`, sha256
`c3548d7f9bae2881dcd833c9ddb19e5abb0df174ae76692dac79ab6b566572bc`, unchanged.

```
uv run --locked python -m improvements.taskrelation.research.execution_contract \
  check improvements/taskrelation/research/execution/preflights/DP-0008
{"assessments": 2, "negotiation_status": "BLOCKED", "transcript_required": false, "valid": true, "workloads": 2}
```

## Blocker re-derivation against the re-checked tree

| Blocker (assessment r2) | 2026-10-04 state |
|---|---|
| `B-REGISTRATION` — DG-0008 unregistered | **RESOLVED** (`fa326fc`): `STUDIES.jsonl` row + `studies/DG-0008/{PLAN.md,NOTE.md}` |
| Implementation `PLANNED`, no commit/entrypoint | **RESOLVED** (`d967568`): `dg0008` package + five entrypoints + `configs/dg0008.yaml`; 63 focused tests green; the accepted `IF-MANIFEST-IO=STREAM_CANONICAL_BYTES` seam verified byte- and digest-identical to the canonical oracle at `L = 140000` (5.1 MB vs 18.0 MB peak) |
| Run provenance absent | **RESOLVED** (`d967568`): every member opens a real MLflow run through `mlflow_utils`; no fallback |
| `B-COMPUTE-PLAN` — no committed plan/inputs | **PARTIAL**: `compute/inputs.json` written and validating; `compute/plan.json` impossible to write truthfully — `jobspec.validate_plan` needs a concrete `worker.gpu_type` and a benchmark-derived `timeout_seconds`, and `expand_argv` offers only `{task_type}`,`{config}`,`{device_index}`,`{seed}`, so DG-0008's folds require 40 literal arms plus a distinct repeat arm while the gate members' one-at-a-time ordering is an authorization concurrency property |
| `B-INFRA-SURFACE` — compute adapter's command surface missing | **PARTIAL — decision required**: the option-level contract is repaired in the owning layer (`infra` v0.1.1, `a7ee886`; `make infra-check` green, 154 tests) so `worker list/show --read-only` and `worker create --require-direct-ssh` now behave as the adapter and the proposal require. The whole `volume`/`storage`/`job` verb families are absent from this subsystem **and** from the standalone `wavcse-infra` checkout, so no job spec can be submitted. That is a feature program, not a maintenance repair |
| `B-CORPUS` — lawful raw corpus absent | **UNCHANGED**: `~/voice_dataset` absent, sources unrecorded in-tree, IEMOCAP LDC licence-gated |
| `B-EMBEDDING-OBJECTS` — canonical embeddings absent locally | **UNCHANGED**: declarations inherited from the committed canonical set; private S3 unreachable from this workstation |
| `B-IDENTITY-MANIFEST` — identity/opportunity artifacts absent | **UNCHANGED**: generator outputs after lawful restoration; `L`, `S`, `n_ER` must not be fabricated |
| `B-BENCHMARK`, `B-COMPATIBLE-GPU` — runtime/cost unbounded | **UNCHANGED**: no benchmark authorized; five-epoch duration, GPU-hours, offer price, storage lifetime and worker lifetime remain unknown |
| `B-AUTHORIZATION` — no envelope | **UNCHANGED**: `authorizations/DG-0008.yaml` deliberately absent; unknown price or lifetime still fails closed |

## Why no fresh assessment round was persisted

`execution_contract.apply_assessment` requires the negotiation to be in
`PENDING_INFRA`, and `apply_revision` requires `CHANGES_REQUESTED`. The
INC-021 negotiation is terminal at `BLOCKED` (round 2 of at most 3), and
`check_directory` requires exactly one negotiation per preflight directory, so
a new round can only follow a legitimately requested workload revision — which
itself requires a `REQUEST_IMPLEMENTATION_CHANGE`. Nothing in this increment
produced such a request: the accepted flexible choice is satisfied and the
remaining blockers are lifecycle/authority/data/scope gates, not implementation
defects. A new round would therefore have to restate the same `BLOCKED` verdict,
and the next legal round belongs to the increment that produces a real revision
(committed implementation binding + compute plan) or a genuine new request.

## Next human action required

The next required action is external, not computational: decide the
infrastructure scope question for the `volume`/`storage`/`job` feature program
(see `.agents/policies/autonomy.md` and `infra/docs/SPEC.md`), decide the lawful
corpus/embedding restoration question, and then, when a compatible-GPU benchmark
and a bounded price/lifetime envelope exist, authorize them explicitly. A
concrete-GPU benchmark authorization is the smallest next spend decision; the
201-job Stage-1 authorization follows only after the benchmark bounds it.
