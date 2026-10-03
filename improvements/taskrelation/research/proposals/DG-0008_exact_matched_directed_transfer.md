---
proposal_schema: 1
proposal_id: DP-0008
allocated_study_id: DG-0008
status: CHANGES_REQUESTED
created_at: 2026-10-03T00:00:00+00:00
review: CHANGES_REQUIRED
reviewed_by: research-reviewer
approved_by: ""
approved_at: ""
superseded_by: ""
---

# DG-0008 — Exact-opportunity ER-target residual diagnostic

## Question

For the two ER-target cells preselected from F8, does enabling the auxiliary loss under an otherwise identical five-epoch LOSO training opportunity produce a reproducible harmful residual for ER←SI but a practically null residual for ER←KS, where “harmful” and “practically null” are judged against one pre-registered diagnostic SESOI that must be fixed before scientific execution?

## Evidence basis

- **F8 / DG-0001** establishes the motivating behavioral pattern and its limitation: approximate update matching left ER←SI negative and ER←KS unresolved, but the controls changed batch size and gradient noise, and the evidence came from one initialization. F8 therefore motivates these two ER-target cells only; it does not establish semantic transfer, full directionality, or a KS↔SI conclusion.
- **F1** requires seeds `0,1,2,3,4` for a sub-1pp claim and bars a one-seed result from promotion.
- **F3** requires speaker-independent ten-fold LOSO for an ER claim. The fold definition is the existing deterministic IEMOCAP protocol: test speaker `i`, validation speaker `(i+1) mod 10`, and the other eight speakers for training.
- **FRAMEWORK.md R4 and R6** require optimizer-exposure control and LOSO, respectively. **R7** keeps the programme in explicit Task Relation Learning; this diagnostic characterizes a transfer target but introduces no loss-weighting or gradient-surgery mechanism.
- **F9/F10** are evidence not to overread endpoint gradient geometry: gradient scale is conditioned on the training mixture, and near-zero gradient cosines did not establish transfer utility. Those findings do not supply a verdict threshold for this five-epoch residual study.
- Binding scope context: **DEC-0013** deferred diagnostic-first sequencing, and **DEC-0014** activated a separate project-original directed-relation arm. DG-0008 would therefore be optional characterization and would neither gate nor authorize that mechanism.
- Provenance to carry forward: `studies/DG-0001/{STUDY.md,analysis.md,result.json}` and `task_relations/{loso_transfer.json,optimization_control.json}`.
- Operational prerequisites: `WORKER_ENVIRONMENT.md` records that the prior 200 GB cache volume and raw-corpus farm were destroyed, that the raw corpus sources are not fully named in the repository, and that the existing L4 timing is for a different 30-epoch workload. It cannot be linearly divided to estimate this study.

## Hypotheses

Define, for auxiliary `c ∈ {SI, KS}`, seed `s`, and LOSO fold `f`,

`d[c,s,f] = ER_accuracy(pair[c], s, f) − ER_accuracy(control[c], s, f)`.

The independent seed-level estimate is

`T[c,s] = mean over the 10 nested folds of d[c,s,f]`,

and the preselected pair-selectivity contrast is

`D[s] = T[SI,s] − T[KS,s]`.

The five seeds—not the 50 seed-fold observations—are the independent units. Let `δ_DG > 0` be one diagnostic SESOI in ER accuracy percentage points. Its numerical value is not established by the repository and remains a blocking prerequisite.

- **H1 (primary — behavioral pair-selective residual).** Under exact within-cell opportunity matching, ER←SI is harmfully negative, ER←KS is practically null, and `D` is negative beyond `δ_DG`. This is a behavioral auxiliary-loss effect under the specified streams; it is not a claim that SI semantics caused the effect and not a claim of full transfer directionality.
- **H2 (competing — generic auxiliary-loss interference).** Both auxiliaries harm ER beyond `δ_DG`, while the seed-level contrast `D` is practically equivalent to zero. This would indicate a generic consequence of enabling a second task loss under these regimes rather than SI-selective behavior.
- **H3 (third outcome — no residual after exact matching).** Both ER-target residuals are practically equivalent to zero. This would support the explanation that F8’s remaining pattern did not survive exact opportunity matching.
- Any pattern outside these three predeclared regions—including opposite signs, a non-null KS cell without the H1 pattern, or intervals too wide to establish harm or equivalence—is **INCONCLUSIVE**, not a post-hoc fourth success condition.

## Proposed study

**Category.** `DG-xxxx` diagnostic. It tests a controlled behavioral transfer target relevant to Task Relation Learning, proposes no mechanism, does not increment the mechanism plateau counter, and cannot authorize a relation architecture.

**Regime.** Frozen `wavlm_large` embeddings; mean frame pooling; `smp(0.5)` over all 25 layers; the DG-0001 five-epoch LOSO architecture and regularization settings; batch size 2,048 with `drop_last_train: true`; fixed-final-epoch ER accuracy as the sole scientific endpoint. KS and SI outcomes are not study endpoints. The validation speaker remains separate from the held-out test speaker in every fold.

**Independent variable.** Whether the auxiliary task’s cross-entropy term is enabled. Both arms in a cell use the same two-head model and the same combined input stream. The pair arm uses the repository’s two-task coefficients, `w_ER = 0.5` and `w_aux = 0.5`; the control keeps `w_ER = 0.5` and sets `w_aux = 0`. This default-off diagnostic hook is a control instrument, not a proposed loss-weighting method.

**Matched controls and exact-opportunity rule.** Each pair arm has its own within-cell control. A control is not an `er`-only loader with a smaller batch; it is the same pair-task loader and model with the auxiliary loss disabled. Pair and control must consume the identical ordered batch manifest, including the same ER example IDs and order, the same auxiliary inputs, the same total batch size, the same number of optimizer steps, and the same per-step ER count. They must start from the same serialized model state and reset to the same Python, NumPy, CPU-Torch, CUDA, DataLoader-worker, and dropout RNG states. Both execute the same forward graph before the loss-term switch.

The validation-driven scheduler is removed symmetrically from all four arms. Every optimizer step uses AdamW learning rate `0.0025`; no arm-dependent validation statistic may alter the LR. Weight decay `5e-8`, L1 `1e-7`, L2 `1e-5`, optimizer implementation, precision, checkpoint point, and evaluation code are common. This fixed-LR regime is deliberately more controlled than DG-0001 and must be described as a new exact-opportunity regime, not an exact reproduction of DG-0001.

**Required exact per-cell table — incomplete and therefore blocking.** Approximate values such as 27 or 69 steps may not be substituted. Before `READY_FOR_HUMAN`, deterministic loader semantics and a no-training manifest generator must replace every unresolved entry with exact integers or an immutable per-step vector for every LOSO fold. Runtime equality is proved later by Stage 1, after registration and a human-granted Stage 1 envelope.

| Cell | Pair arm | Matched control | Shared input stream | Exact steps `S[c,f]` per epoch | Exact ER count vector `(n_ER[c,s,f,k])` | Pair weights `(ER, aux)` | Control weights `(ER, aux)` | LR path | Initialization/RNG rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ER←SI, decisive cell | `si_er`, SI loss enabled | same `si_er` model/stream, SI loss disabled | one immutable `si_er` batch manifest per seed/fold | **UNRESOLVED** | **UNRESOLVED** | `(0.5, 0.5)` | `(0.5, 0)` | `0.0025` at every step | one serialized init plus one reset RNG snapshot shared by the pair |
| ER←KS, candidate-null comparator | `ks_er`, KS loss enabled | same `ks_er` model/stream, KS loss disabled | one immutable `ks_er` batch manifest per seed/fold | **UNRESOLVED** | **UNRESOLVED** | `(0.5, 0.5)` | `(0.5, 0)` | `0.0025` at every step | one serialized init plus one reset RNG snapshot shared by the pair |

The trace must additionally record batch example IDs, valid-label counts, optimizer-step index, epoch, LR, initialization hash, and RNG-state hashes. Exactness is asserted only within a cell. SI and KS corpora produce different streams and potentially different `S[c,f]`; that cross-cell difference is a residual design limitation, not something the proposal may call exactly matched.

**Staged plan.**

1. **Design-prerequisite closure, before `READY_FOR_HUMAN`.** Complete the declarative opportunity table without training; freeze `δ_DG` and the seed-level precision calculation; either freeze the exploratory gradient estimator or delete it; document lawful corpus-access prerequisites; and derive a conservative, bounded Stage 1 ceiling from static workload facts. No scientific test result or paid execution belongs here.
2. **Stage 1 — implementation and exposure validation only.** Generate manifests for both cells, all five seeds, and all ten LOSO folds. Run one predeclared fold as four full five-epoch train/validation jobs—pair and control for each cell—without invoking or reading the held-out test evaluator. Verify zero manifest, ER-example, step-count, target-weight, LR-path, initialization-hash, and RNG-stream mismatches. Exercise checkpoint loading and the exploratory endpoint diagnostic. Stage 1 cannot support, reject, screen, select, or stop H1; a failure means the implementation is invalid and must be corrected under the unchanged design.
3. **Stage 2 — scientific stage, separately approved.** Run the two preselected cells only, with both arms, seeds `0,1,2,3,4`, and all ten LOSO folds: `2 cells × 2 arms × 5 seeds × 10 folds = 200` five-epoch training jobs. No cell is selected from Stage 1 and no test endpoint is read until its fixed run matrix is complete. Report the five seed-level `T[SI,s]`, `T[KS,s]`, and `D[s]` values with mean, SD, paired 95% t intervals, and sign counts. Fold values and speaker ranges are nested descriptive results.

No reverse cell, KS↔SI cell, or 30-epoch A0/A1 scale experiment belongs to DG-0008.

## Discriminating measurements

- **Primary behavioral measurements:** the five seed-level `T[SI,s]`, `T[KS,s]`, and `D[s]`. Together with the frozen `δ_DG`, these separate H1, H2, and H3 as behavioral outcomes.
- **Validity measurements:** exact equality of the paired manifests, ER IDs/order, per-step ER counts, step counts, target-loss coefficient, LR sequence, serialized initialization, and reset RNG sequence. These determine whether a contrast is admissible; they are not scientific evidence for a residual.
- **Nested descriptive measurements:** fold-level residuals, held-out-speaker ranges, and fold sign counts. They describe speaker heterogeneity but cannot be treated as 50 independent observations or used to narrow the seed-level interval.
- **Exploratory endpoint geometry:** if its estimator and cost are frozen before execution, compute at the fixed five-epoch checkpoint (a) the full-training-pool mean gradient for ER and the relevant auxiliary on shared parameters, and (b) a batch-gradient variance summary using one predeclared batch size, batch manifest, and number of draws. To limit cost, this is restricted in advance to seed `0`, all ten folds, all four arms: 40 frozen-checkpoint diagnostic suites. The exact variance estimator, batch size, and draw count are currently unresolved.
- Endpoint norm, variance, and cosine measurements may describe where the behavioral arms differ. They **do not** distinguish semantic interference from generic auxiliary-gradient effects, do not adjudicate H1 against a semantic H2/H3 story, and do not authorize a mechanism. A validated semantic-negative control would be required for that causal claim; KS is a preselected behavioral null-control cell, not a semantic-negative control.
- Ω magnitude, per-batch gradient norm, aggregate accuracy, best-checkpoint accuracy, and ordinary-split ER accuracy are not decision measurements.

## Success and stop conditions

**Current status:** the scientific conditions below are structurally specified but not numerically pre-registered because `δ_DG`, the seed-level variance basis, and the resulting precision/power calculation are unavailable. Stage 2 must not be authorized while those fields remain unresolved.

**Stage 1 validity gate:**

- exactly `0` mismatched batch-manifest entries within every exercised pair;
- exactly `0` mismatches in ER example IDs/order and per-step ER valid counts;
- exactly `0` differences in optimizer-step count, ER coefficient, LR sequence, initialization hash, and pre-forward RNG-state sequence;
- exactly `0` held-out test endpoint reads;
- successful creation and read-back of the fixed-final-epoch checkpoint and, if retained, the exploratory diagnostic artifact.

Failure of any gate invalidates the implementation. It is not evidence for or against a hypothesis and creates no scientific stop or promotion.

**Required precision prerequisite:** before Stage 2, name a defensible source for the between-seed SD of the seed-level paired residual. DG-0001’s ten folds cannot supply this because folds are nested within one seed. Using five independent seeds, calculate the expected 95% interval width and power for the proposed `δ_DG`, and state the required precision criterion numerically. If five seeds cannot resolve the chosen SESOI, the design must be revised before compute; it may not rely on fold pseudo-replication. No variance, interval width, or power value is presently supported by the record.

**Stage 2 verdicts, after `δ_DG` is numerically frozen:**

- **H1 supported:** the 95% interval for mean `T[SI]` lies wholly below `−δ_DG`, the 95% interval for mean `T[KS]` lies wholly inside `[−δ_DG,+δ_DG]`, and the 95% interval for mean `D` lies wholly below `−δ_DG`.
- **H2 supported:** both residual intervals lie wholly below `−δ_DG`, while the interval for `D` lies wholly inside `[−δ_DG,+δ_DG]`.
- **H3 supported:** both residual intervals lie wholly inside `[−δ_DG,+δ_DG]`.
- **INCONCLUSIVE:** every other valid result, including insufficient precision. A wide interval is not evidence of equivalence.

All 200 predeclared scientific jobs must finish validly before a verdict; there is no outcome-based early stop. A failed infrastructure job may be repeated only with the same manifest, initialization, config, and commit, with the failure retained in the record. After Stage 2, stop DG-0008 regardless of outcome. It may characterize these two ER-target cells only; it cannot establish full directionality, close KS↔SI, support a cross-direction magnitude comparison, or authorize a directed mechanism.

## Expected compute

A defensible **Stage 2** GPU-hour or monetary estimate is not currently derivable. `WORKER_ENVIRONMENT.md` measures approximately 19–22 minutes for a different 30-epoch arm on an L4 with a warm cache; dividing that duration by six would ignore LOSO launch/materialization costs, the exact-opportunity control, endpoint diagnostics, and the destroyed corpus cache. Before `READY_FOR_HUMAN`, the proposal needs a conservative bounded **Stage 1** ceiling sufficient for the human to decide whether to register and authorize validation. The measured Stage 1 benchmark then prices Stage 2.

The run matrix that the benchmark must price is:

| Stage | Training jobs | Endpoint diagnostic suites | Other paid work |
| --- | ---: | ---: | --- |
| Stage 1 validation/benchmark | 4 full five-epoch train/validation jobs on one predeclared fold, plus 1 repeated matched-control job with diagnostics disabled to measure diagnostic overhead | 4 | cold/warm input materialization, checkpoint I/O, loader construction, and environment verification |
| Stage 2 science | 50 ER←SI pair + 50 ER←SI controls + 50 ER←KS pair + 50 ER←KS controls = **200** | **40**, restricted to seed `0` across four arms and ten folds | artifact collection, analysis, failure allowance, and worker idle/setup time |

Let the measured per-job training times and per-suite diagnostic times be recorded separately for all four arm classes. Stage 2 GPU-hours must then be computed as the sum of the 200 measured-class training projections plus the 40 diagnostic projections; paid wall-clock must additionally include corpus preparation, materialization, collection, and a declared failure allowance. No values are inserted here because none were measured for this workload.

Stage 1 should use one worker and one live job at a time so its timing and cache state are interpretable. Stage 2 concurrency remains a human/resource decision after the benchmark and may not exceed the repository protocol’s two-job cap. The estimate must state the chosen concurrency rather than assuming perfect parallel scaling.

After registration and a human-granted Stage 1 envelope, but before any benchmark job, the raw tree must be restored and loader-verified: Speech Commands, VoxCeleb1, and licence-gated IEMOCAP in the paths recorded by `WORKER_ENVIRONMENT.md`. The prior 200 GB volume no longer exists, and the repository does not identify all raw-corpus sources. A worker also needs the recorded container-disk allowance sufficient for one approximately 45–50 GB workspace; this proposal provisions no resource.

## Risks and confounds

1. **No semantic-negative control.** The design identifies the effect of enabling a real auxiliary loss under a fixed stream. It cannot separate label semantics from auxiliary gradient distribution, corpus identity, class count, or other task-specific properties. Endpoint geometry cannot repair this.
2. **Only within-cell exactness is possible.** ER←SI and ER←KS use different combined streams and may have different exact step counts. `D` is a contrast of two within-cell effects, not a claim that the SI and KS interventions themselves are exposure-identical.
3. **No full directionality.** Reverse directions are intentionally absent, their DG-0001 references were not independent refits, and KS↔SI is intentionally untested. The study measures ER pair-selectivity only.
4. **New fixed-LR regime.** Removing ReduceLROnPlateau prevents arm-dependent LR paths but means the result characterizes this controlled five-epoch regime, not every detail of DG-0001’s historical scheduler trajectory.
5. **Seed-level precision is unknown.** The only relevant controlled residual is single-seed; fold variation cannot estimate initialization variance. The planned five-seed design may be underpowered for the eventual SESOI.
6. **Exact RNG equality requires implementation proof.** `set_seed` covers Python, NumPy, Torch, CUDA, and worker derivation, but paired runs also need immutable manifests, serialized initialization, reset RNG snapshots, the same forward graph, and deterministic-operation checks. A seed number alone is insufficient.
7. **Exploratory variance is estimator-dependent.** Its batch size, manifest, and draw count alter the quantity. Until frozen and benchmarked, it is not reproducible and has no decision role.
8. **Environment availability.** The cache volume and raw farm are gone; IEMOCAP is licence-gated; raw source locations are not fully recorded. This can block even a non-scientific benchmark.
9. **Test leakage through workflow.** Stage 1 must not invoke the held-out test evaluator. Stage 2 cells, thresholds, manifests, and analysis code must be frozen before any test result is read.
10. **Residual assumption.** Even with exact observable streams and deterministic RNG, GPU kernels may not be bitwise deterministic on the selected hardware. The implementation must either enforce deterministic kernels or predeclare a reproducibility tolerance and demonstrate it without test endpoints; that tolerance is currently unresolved.

## Repository effects

If later approved and registered, the study would create `studies/DG-0008/{PLAN.md,NOTE.md,configs/,compute/,analysis.md,result.json}` and append one `DG-0008` record to `STUDIES.jsonl`. It would add a default-off exact-opportunity batch-manifest/control hook, a fixed-LR diagnostic mode, initialization/RNG/exposure trace artifacts, scoped tests for equality gates, and a frozen analysis script implementing seed-level aggregation with folds nested.

If the exploratory endpoint measurement is retained, its estimator and artifact schema would also be added with tests, but it would remain explicitly exploratory in `analysis.md` and `result.json`.

A human-granted authorization would be a separate `authorizations/DG-0008.yaml`; this proposal creates none. Post-result updates could touch `FINDINGS.md`, `FAILURES.md`, `FRAMEWORK.md`, and `STATE.md` only to the extent licensed by the bounded ER-target result. It would not modify DEC-0014, register or authorize TR-0008, or claim KS↔SI/full-direction closure.

For the current revision, the proposal frontmatter should move to `status: CHANGES_REQUESTED`, `review: CHANGES_REQUIRED`, and `reviewed_by: research-reviewer`; no study, authorization, compute plan, or project state should be created.

## Human decisions required

1. Whether to reopen this deferred diagnostic at all, given DEC-0013/DEC-0014 and the fact that it does not gate the active directed mechanism.
2. Whether the narrowed deliverable—behavioral ER pair-selectivity only, with no full-directionality or semantic-causation claim—is worth the required LOSO matrix.
3. Approve the exact-opportunity intervention: same two-head stream in both arms, `(0.5,0.5)` versus `(0.5,0)`, and a common constant `0.0025` LR with no validation-driven scheduler.
4. Choose or approve the scientifically justified numeric `δ_DG` after the seed-level precision/power calculation. The programme’s `0.20pp` no-regression bar and DG-0001’s `1.0pp` screening band are not adopted as diagnostic SESOIs.
5. Decide whether a semantic claim is desired. If yes, require a separately validated semantic-negative control and revise the design; if no, accept that DG-0008 can produce only the behavioral H1/H2/H3 verdicts above.
6. Decide whether the exploratory pool-mean/variance suite is retained after its estimator and incremental workload are specified. It cannot affect the scientific verdict.
7. Once the declarative table, SESOI/precision calculation, optional-estimator decision, lawful corpus prerequisite and bounded Stage 1 ceiling are complete and independently re-reviewed, decide whether to register DG-0008 and grant a bounded Stage 1 envelope covering corpus restoration, implementation/exposure validation and the workload benchmark.
8. Grant or refuse a separate Stage 2 envelope only after Stage 1 passes and its measured benchmark produces a bounded full-matrix estimate. Stage 1 creates no automatic promotion or authorization.

## Review

The prior independent review returned **CHANGES_REQUIRED**. This revision clears the scope and inference errors by restricting the study to the preselected ER←SI decisive cell and ER←KS candidate-null comparator; removing reverse and KS↔SI conclusions; making Stage 1 implementation/exposure validation only; making Stage 2 the separately approved five-seed, ten-fold LOSO scientific stage; treating seeds as independent and folds as nested; removing the `0.20pp`/`1.0pp` diagnostic verdict bands; removing DEC-0011/DEC-0012 and the unrelated 30-epoch A0/A1 add-on; and demoting pool-mean/variance geometry to non-causal exploration.

The proposal is still **CHANGES_REQUESTED**, not `READY_FOR_HUMAN`. The pre-decision blockers are: (1) exact declarative per-fold `S` and per-step ER count vectors; (2) a justified numeric diagnostic SESOI plus a seed-level precision/power calculation that does not treat folds as independent; (3) either a frozen estimator/workload for the optional exploratory variance measurement or its removal; and (4) a conservative bounded Stage 1 ceiling with lawful corpus-access prerequisites. After human approval, registration and a Stage 1 envelope, Stage 1—not this proposal—proves shared-stream, fixed-LR, initialization/RNG and reproducibility gates, restores and verifies the corpus environment, and measures the five-epoch workload; its measured estimate is required before any separate Stage 2 authorization. A semantic-negative control is not required for the narrowed behavioral question, but remains mandatory before any semantic H2/H3 causal claim.
