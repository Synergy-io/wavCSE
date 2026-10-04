# DG-0008 — Exact-opportunity ER-target residual diagnostic

Status: **REGISTERED / BLOCKED — zero-cost implementation and preflight only; no compute authorized**
Type: diagnostic (behavioral transfer characterization; no mechanism)
Created: 2026-10-04T05:38:12Z
Proposal: `DP-0008`, `../../proposals/DG-0008_exact_matched_directed_transfer.md`
Approved proposal commit: `83694cd965a3045ddf2f0bc8c8c9949a5abfaba2`
Approved proposal SHA-256: `c3548d7f9bae2881dcd833c9ddb19e5abb0df174ae76692dac79ab6b566572bc`
Approval: Kevin Sanjula `<kevinxsanjula@gmail.com>`, 2026-10-03T16:18:45+00:00
Reviewer: `research-reviewer`, `PASS`
Authorization: **NONE** — registration is not spend authority; `authorizations/DG-0008.yaml` does not exist and must not be created without a later bounded human grant.
Execution preflight: `../../execution/preflights/DP-0008/`

## Binding and interpretation

This Study registers the exact scientific design approved as DP-0008. The approved proposal bytes at the immutable commit and digest above are incorporated as the normative scientific specification. This plan records the Study lifecycle and executable contract without widening or changing that specification. If a summary below is ever incomplete or ambiguous, the approved proposal at that immutable commit controls; no implementation may infer a new arm, endpoint, tolerance, threshold, dataset, or continuation stage.

Registration authorizes bounded repository implementation, deterministic artifact preparation, local zero-cost checks, compute-plan construction, and necessary infrastructure maintenance only. It does not authorize a GPU benchmark, worker or volume creation, experiment submission, paid storage, or any other billable action.

## Observation and research question

DG-0001/F8 left a negative ER←SI residual and unresolved ER←KS residual after approximate optimizer-exposure matching, but its controls changed batch size and gradient noise and used one initialization. DG-0008 asks whether, for those two preselected ER-target cells, enabling the auxiliary loss under an otherwise identical five-epoch, fixed-LR, speaker-independent LOSO training opportunity produces a reproducible harmful ER←SI residual but a practically null ER←KS residual against `δ_DG = 0.010` ER accuracy.

The work is a behavioral diagnostic. It cannot identify semantic causation, establish full directionality, authorize a relation mechanism, or increment the failed-mechanism plateau counter.

## Hypotheses and competing explanations

For auxiliary `c ∈ {SI, KS}`, seed `s ∈ {0,1,2,3,4}` and LOSO fold `f ∈ {0,…,9}`:

- `d[c,s,f] = ER_accuracy(pair[c],s,f) − ER_accuracy(control[c],s,f)`;
- `T[c,s] = mean_f d[c,s,f]`;
- `D[s] = T[SI,s] − T[KS,s]`.

The five seeds are the independent units; folds are nested and are never treated as 50 independent observations.

- **H1 — pair-selective residual:** ER←SI is harmfully negative, ER←KS is practically null, and `D` is negative beyond `δ_DG`.
- **H2 — generic auxiliary-loss interference:** both auxiliaries harm ER beyond `δ_DG`, while `D` is practically equivalent to zero.
- **H3 — no residual after exact matching:** both ER-target residuals are practically equivalent to zero.
- **INCONCLUSIVE:** every other valid pattern, including opposite signs, a non-null KS cell outside H1, or intervals too wide for harm/equivalence.

All four classes are screening evidence only. DG-0008 stops after this screen for every class. Confirmation would require a different proposal, Study, design review, registration, and authorization.

## Independent variable and matched controls

The sole intended arm difference within each cell is whether the auxiliary cross-entropy term contributes to the loss:

- pair: `w_ER = 0.5`, `w_aux = 0.5`;
- control: `w_ER = 0.5`, `w_aux = 0`.

Both arms execute the same two-head forward graph over the same combined input stream before the loss-term switch. Within a cell they share dataset membership; canonical ordered opportunity manifests; epoch, step, and per-step ER counts; serialized initialization; Python, NumPy, CPU-Torch, CUDA, DataLoader-worker, and dropout RNG states; target coefficient; fixed LR sequence; precision; checkpoint point; and evaluation code. Weight decay is `5e-8`, L1 `1e-7`, L2 `1e-5`, and the optimizer implementation is common.

Exactness is within a cell only. `si_er` and `ks_er` have different streams and step counts; `D` contrasts two within-cell effects and is not an exposure-identical cross-cell intervention.

## Fixed scientific protocol

- Fixed `wavlm_large` embeddings, mean frame pooling, and `smp(0.5)` over all 25 layers.
- Cells exactly `ks_er` and `si_er`; arms exactly `pair` and `control`.
- Seeds exactly `0,1,2,3,4`; LOSO folds exactly `0…9` using `improvements/base/kfold_iemocap.py::build_loso_fold`.
- Five zero-based training epochs `e=0…4`; endpoint after `e=4`.
- Global batch size `2048`, `drop_last_train: true`.
- AdamW with LR fixed at `0.0025` for every step; no validation-driven scheduler in either arm.
- Fixed-final-epoch ER accuracy is the sole scientific endpoint. KS and SI outcomes are not endpoints.
- Exactly `2 cells × 2 arms × 5 seeds × 10 folds = 200` matrix trainings plus one same-arm repeat of `si_er/control/fold=0/seed=0`: 201 five-epoch trainings total.
- The four fold-0/seed-0 gate members count within the 200 and are not rerun.
- No reverse cell, KS↔SI cell, 30-epoch scale experiment, gradient diagnostic, mechanism, confirmation, or outcome-based early stop.

## Identity and opportunity-manifest contract

The generator, sampler, and execution path must implement the approved proposal's clauses 1–14 exactly. In particular:

1. Canonical relative paths replace `\\` with `/`; NUL, absolute, drive-qualified, empty, `.`, and `..` segments are rejected; there is no filesystem resolution, symlink dereference, case folding, percent decoding, or Unicode normalization. Keys are unique per component and ordered by UTF-8 bytes.
2. Labels and memberships come independently from lawful raw metadata: Speech Commands official lists/base metadata, VoxCeleb1 `iden_split.txt`/base metadata, and IEMOCAP sessions plus annotation-derived mapping. `fru` is dropped; `exc` maps to `hap`; IEMOCAP speakers and folds follow the approved loader-matching rules.
3. Canonical JSON is exactly `json.dumps(x, sort_keys=True, separators=(",",":"), ensure_ascii=False, allow_nan=False).encode("utf-8")`, with no BOM, indentation, spaces, or trailing newline.
4. The identity object schema is exactly `dg0008.example-identity.v1` and is frozen with its lowercase SHA-256 before training.
5. Ordered vectors are derived from identity records, never torchaudio enumeration indices. Cell component order is `[speechcommand, iemocap]` or `[voxceleb, iemocap]`.
6. `L[cell,f]`, `S[cell,f]=floor(L/2048)`, and epoch-indexed `n_ER[cell,s,f,e,k]` are lawful generator outputs and must not be fabricated.
7. The manifest seed, keyed Fisher–Yates permutation, batching, `dg0008.opportunity-manifest.v2` schema, per-epoch digest, run-level tuple-associated digest, and matrix-level tuple-associated digest are exactly clauses 8–12 of the approved proposal.
8. `IF-MANIFEST-IO=STREAM_CANONICAL_BYTES` is the accepted implementation-flexible strategy. It must preserve the exact canonical object bytes, object files, errors, tuple associations, and all per-epoch/run/matrix digests; only bounded-memory construction differs from materialization.
9. Admission requires both the declared embedding-object SHA-256 set and byte-identical fresh raw-metadata identity extraction. Neither substitutes for the other.
10. The freeze point records identity bytes/digest, every opportunity object/digest, every run digest, matrix digest, and exact `L`, `S`, and `n_ER` values before training or held-out endpoint reads.

## Reproducibility and validity gates

Every run sets `torch.use_deterministic_algorithms(True, warn_only=False)`, `torch.backends.cudnn.deterministic=True`, `torch.backends.cudnn.benchmark=False`, `torch.backends.cuda.matmul.allow_tf32=False`, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and uses one pinned software stack and one GPU model.

The repeat must be bitwise identical for the per-step loss trace, fixed-final validation ER accuracy, and serialized checkpoint. Any deterministic-algorithm error or disagreement is fatal. No tolerance may be invented.

Before held-out endpoints are read, the execution must show zero embedding-digest mismatches; zero identity byte/digest mismatches; zero opportunity/aggregate-digest mismatches; zero mismatched pair batch entries, ER order/counts, steps, target coefficient, LR sequence, initialization hash, or pre-forward RNG-state sequence; `S` within the coarse DG-0001 sanity envelope; successful checkpoint creation/read-back; a bitwise-identical repeat; and zero premature held-out test reads. All 200 matrix jobs and the repeat must finish validly before classification.

An infrastructure-failed job may be repeated only with the same identity digest, run-manifest digest, initialization, config, and commit, while retaining the failed attempt.

## Classification rule

For each `X ∈ {T[SI], T[KS], D}`, compute mean and sample SD over five seed values and the two-sided one-sample 95% t interval with 4 degrees of freedom. Let `δ = 0.010`. Boundary equality is not success.

- **H1:** `I_T[SI]` is wholly below `−δ`; `I_T[KS]` is wholly inside `(−δ,+δ)`; `I_D` is wholly below `−δ`.
- **H2:** both `I_T[SI]` and `I_T[KS]` are wholly below `−δ`; `I_D` is wholly inside `(−δ,+δ)`.
- **H3:** both cell intervals are wholly inside `(−δ,+δ)`.
- **INCONCLUSIVE:** otherwise.

Report seed means, SDs, intervals, nested fold residuals, held-out-speaker ranges, and fold sign counts. A wide interval is never equivalence.

## Information gain, evidence tier, and category boundary

This is the cheapest approved experiment that removes the known within-cell opportunity, initialization, LR-path, and single-seed confounds while retaining honest speaker-independent ER evaluation. Its information gain is whether the F8 residual survives exact opportunity matching and whether the two preselected auxiliaries differ behaviorally under that controlled regime.

Evidence remains screening-tier even with five seeds because this Study is pre-declared as a diagnostic screen with no confirmation stage. It stays within Task Relation Learning as behavioral transfer characterization and does not introduce loss weighting as a method, gradient surgery, low-rank structure, task clustering, or decomposition.

The design is motivated by internal evidence (F1, F3, F8–F10, DG-0001, FRAMEWORK R4/R6/R7), not by a new literature claim.

## Inputs, outputs, and checkpoint/resume semantics

Required inputs and their exact identities belong in `compute/inputs.json`: lawful raw-corpus metadata/layout, canonical embedding archives and SHA-256s, the frozen example-identity artifact, all per-epoch opportunity manifests/digests, aggregate digest indexes, configuration, lockfile/software identity, and the exact implementation commit.

Required outputs include identity/admission reports; manifests and digest indexes; per-job run identity; initialization and pre-forward RNG hashes; per-step loss trace; LR, target coefficient, step and ER-count traces; validation-only gate output; fixed-final checkpoints with read-back digest; repeat-comparison report; delayed held-out ER endpoints; and deterministic analysis inputs/outputs. Output schemas and paths must be committed before execution.

No scientific member may resume from a checkpoint because the comparison requires one uninterrupted deterministic five-epoch trajectory from the shared serialized initialization/RNG snapshot. An infrastructure failure restarts that member from the same immutable inputs and identity while preserving the failed attempt. Completed valid members may be collected idempotently and are not rerun. Held-out evaluation is a separate delayed phase after every training and validity gate succeeds.

## Compute and resource boundary

Stage 0 is CPU-only and zero-cost. Stage 1 is 201 five-epoch GPU trainings. Total runtime and cost remain unknown until a compatible-GPU benchmark measures training-plus-validation duration for every arm class and the repeat path. Historical 30-epoch three-task timing does not bound this workload.

Known constraints: at most two simultaneous GPU jobs; one live workspace per worker; approximately 45–50 GB measured workspace, so container disk must be at least 100 GB; a raw/embedding cache was previously about 200 GB; the pinned image supports at most `sm_90`, excluding Blackwell; the raw corpus must be lawfully restored and verified; a volume must be attached at worker creation in its data center. Offer price, storage price/lifetime, worker lifetime, retry allowance, and total spend are unresolved and must fail closed.

A later benchmark authorization, if granted, must bind the committed implementation, exact command/config, compatible GPU capability, required input identities, maximum duration, outputs, and narrow runtime-bounding purpose. It is not granted here.

## Current blockers and stop rule

Registration itself is complete. Execution remains blocked until all of these hold:

1. lawful Speech Commands, VoxCeleb1, and licence-gated IEMOCAP raw metadata/corpus availability is established;
2. canonical embeddings and their declared digests are available and independently admitted;
3. identity and opportunity artifacts are generated and frozen;
4. the approved experiment implementation and focused local validation are committed;
5. canonical `compute/plan.json` and `compute/inputs.json` validate and bind the implementation and inputs;
6. the deterministic infra command surface required by `improvements.compute` is committed and passes its checks;
7. a compatible-GPU five-epoch benchmark lawfully bounds runtime;
8. current provider/storage prices, lifetimes, retry allowance, and total cost are bounded;
9. the human commits a separate `authorizations/DG-0008.yaml` envelope covering the exact intended paid action; and
10. a fresh deterministic Execution Plane assessment reaches `PREFLIGHT_ACCEPTED` without forcing acceptance.

Missing or restricted data, identity mismatch, deterministic failure, unbounded cost/lifetime, absent authority, or any requested scientific change stops execution. No unavailable value may be invented and no dataset may be substituted.
