# DG-0008 Run Note

Status: **REGISTERED / BLOCKED — zero-cost preparation active; no run; no MLflow run; no compute authorization**
Type: diagnostic (exact-opportunity ER-target residual)
Created: 2026-10-04T05:38:12Z
Proposal: DP-0008
Plan: `PLAN.md`
Execution preflight: `../../execution/preflights/DP-0008/`

## Registration authority

The researcher explicitly authorized registration of DG-0008 as the Study allocated by approved proposal DP-0008, plus zero-cost repository implementation, local tests, deterministic identity/artifact preparation, compute-plan construction, and necessary infrastructure maintenance. The same instruction explicitly withheld paid compute, billable provisioning, a paid GPU benchmark, and experiment submission.

The scientific design remains the exact approved proposal at commit `83694cd965a3045ddf2f0bc8c8c9949a5abfaba2`, SHA-256 `c3548d7f9bae2881dcd833c9ddb19e5abb0df174ae76692dac79ab6b566572bc`, approved by Kevin Sanjula `<kevinxsanjula@gmail.com>` at `2026-10-03T16:18:45+00:00` after `research-reviewer: PASS`.

## Current execution state

Registration creates no spend authority. `authorizations/DG-0008.yaml` is intentionally absent. No worker, volume, provider query, job, benchmark, scientific endpoint, or result exists for DG-0008.

The INC-021 workload revision accepted `IF-MANIFEST-IO: STREAM_CANONICAL_BYTES` as implementation-flexible while preserving canonical bytes and digest/object semantics. Its final assessment was valid and `BLOCKED`; registration resolves only its registration blocker. Implementation, lawful inputs and dual identity, compute artifacts, deterministic infra capability, benchmark/runtime/cost bounds, a human authorization envelope, and a fresh accepted preflight remain separate gates.

## Execution ledger

- 2026-10-04T05:38:12Z — DG-0008 registered from approved DP-0008. Zero-cost preparation authorized. Paid compute and benchmark not authorized.
- 2026-10-04 — zero-cost preparation increment (no compute, no provider mutation, no cost). Repository HEAD at the start: `fa326fc39ca1bc2977d30a0964b80c493571e02b`.

**Implemented.** `improvements/taskrelation/research/dg0008/` (canonical identity/opportunity manifest contract, manifest-driven sampler, fixed-LR five-epoch trainer with the default-off auxiliary switch, Stage-1 validity gate, frozen seed-level analysis) plus the Study-local entrypoints `generate_opportunity_manifest.py`, `run_dg0008.py`, `evaluate_dg0008.py`, `check_validity_gate.py`, `analyze_dg0008.py`, `configs/dg0008.yaml` and `compute/inputs.json`. Focused tests: `test_dg0008_identity_manifest.py`, `test_dg0008_opportunity_manifest.py`, `test_dg0008_pinned_determinism.py`, `test_dg0008_run_identity.py` — 63 tests, green.

**Accepted flexible seam implemented for real.** The INC-021 negotiation accepted `IF-MANIFEST-IO=STREAM_CANONICAL_BYTES`; the first implementation still materialized the whole `steps × 2048` `step_keys` matrix, so the choice was nominal. `manifest.iter_opportunity_manifest_bytes` now recomputes each batch from the frozen clause-9 permutation and streams small chunks in canonical key order; the materializing builder is retained only as a documented test oracle. Verified at a realistic `si_er` scale (`L = 140000`, `steps = 68`): streamed peak 5.1 MB against the oracle's 18.0 MB, largest chunk 271 B, and the streamed bytes and per-epoch digest byte-identical to `canonical_json_bytes`/`sha256(J(x))` of the oracle. The generator hashes each per-epoch file as it writes and re-hashes it by streaming re-read, failing closed on mismatch.

**Provenance.** Every member now opens a real MLflow run using the existing `improvements/mlflow_utils.py` helpers (`setup_mlflow` → `build_research_run_name` → `start_run`, then standard tags, flattened config params, per-epoch history, the fixed-final validation ER accuracy and the manifest/identity digest artifacts), with no fallback: a tracking failure aborts the member rather than training anonymously. Run names follow `DG-0008__stage1_screen__<method>__<cell>__smp25__s0N`; the note resolves to this file. The run identity is not exercised end-to-end here (no GPU, no embeddings, no frozen artifacts); the helper call shape and the offline file-backed logging path are covered by tests.

**Compute artifacts.** `compute/inputs.json` is written and passes `improvements.compute.artifacts.validate` (15 requirements reusing the committed canonical embedding declarations). `compute/plan.json` is deliberately not written: the schema requires a concrete `worker.gpu_type` and a benchmark-bounded `timeout_seconds`, and its argv expansion supports only `{task_type}`, `{config}`, `{device_index}`, `{seed}`, so DG-0008's folds require 40 literal arms plus a distinct repeat arm while the one-at-a-time gate ordering is an authorization/concurrency property rather than a plan field.

**Infrastructure.** The Infrastructure Engineer reproduced the INC-021 command-surface mismatch deterministically and found two distinct classes. Option-level mismatches on verbs that do exist were repaired in the owning layer (`infra/` v0.1.1): `worker list/show --read-only` now suppress the tracked-state writes the compute adapter's read-only discipline requires, and `worker create --require-direct-ssh` refuses a worker whose provider record carries only the command-only SSH proxy. Regression tests were added in `infra/tests/unit/test_cli.py` and `improvements/compute/tests/test_cli.py`; `make infra-check` is green (154 tests). The whole `volume`, `storage` and `job` verb families are absent from the embedded subsystem *and* from the standalone `wavcse-infra` checkout, so the mismatch is a missing feature program rather than a defect: the RunPod network-volume API version, an exact-commit remote-execution engine and S3 transport cannot be built or verified without a live worker, and are escalated as an explicit scope decision. The engineer reported a collision rather than editing `infra/AGENTS.md`/`infra/README.md`, which carry uncommitted user changes, so the two new options are documented by their CLI help text only.

**Data audit (verified locally).** `~/voice_dataset` and `~/embedding` are absent; no `.env`, no `~/.config/wavcse-infra/config.toml`, no `MLFLOW_TRACKING_*`, no `RUNPOD_API_KEY`; the AWS profiles present are unrelated to wavCSE; the DVC pointer is unresolved. Speech Commands v0.01 and VoxCeleb1 are public but not restored; IEMOCAP is licence-gated; the canonical embedding objects live in private S3. No identity or opportunity artifact exists, and none may be fabricated.

**Not done, deliberately.** No compute, provider mutation, worker, volume, job, benchmark, authorization, result interpretation or observability layer. No change to the approved proposal, to any scientific invariant, to another Study, or to `downstream/`.
