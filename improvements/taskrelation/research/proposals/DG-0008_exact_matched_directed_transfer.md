---
proposal_schema: 1
proposal_id: DP-0008
allocated_study_id: DG-0008
status: APPROVED
created_at: 2026-10-03T00:00:00+00:00
review: PASS
reviewed_by: research-reviewer
approved_by: Kevin Sanjula <kevinxsanjula@gmail.com>
approved_at: 2026-10-03T16:18:45+00:00
superseded_by: ""
---

# DG-0008 — Exact-opportunity ER-target residual diagnostic

**Written:** 2026-10-03 (revision 5), against canonical `a1794dd3fb0e4e54f6fa8d728f9ba3dc215acf0e`.
**Predecessor:** F8 / DG-0001 (approximate exposure matching); DP-0008 revisions 1–4; the revision-4 re-review returned CHANGES_REQUIRED with four blockers.
**Human inputs incorporated (2026-10-03):** commit the design to n=5 independent seeds (0–4) with no n=15 branch and no claim of power; treat corpus/embedding availability and restoration as a fail-closed prerequisite; derive any Stage-1 ceiling only from repository-supported or measured quantities, leaving unknown restoration/storage/idle costs explicit and authorization-blocking.

## Question

For the two ER-target cells preselected from F8, does enabling the auxiliary loss under an otherwise identical five-epoch, fixed-LR, speaker-independent LOSO training opportunity produce a reproducible harmful residual for ER←SI but a practically null residual for ER←KS, judged against the pre-registered diagnostic SESOI `δ_DG = 0.010` ER-accuracy?

## Evidence basis

- **F8 / `studies/DG-0001/analysis.md`** motivates these two ER-target cells only: approximate update matching left ER←SI negative and ER←KS unresolved, but the controls changed batch size and gradient noise and used one initialization.
- **F1** requires seeds `0,1,2,3,4` for a sub-1pp claim and bars a one-seed result from promotion. It is the basis of the fixed `n=5` commitment and also warns that five seeds may not resolve very small effects (F4, alternative explanations). DG-0008 nevertheless remains screening evidence only.
- **F3** requires speaker-independent ten-fold LOSO for an ER claim. The fold definition is `improvements/base/kfold_iemocap.py::build_loso_fold`: with speakers sorted, fold `f` uses `speakers[f]` for test, `speakers[(f+1) mod 10]` for validation and the other eight speakers for training.
- **FRAMEWORK.md R4/R6** require optimizer-exposure control and LOSO; **R7** keeps the programme in explicit Task Relation Learning. DG-0008 is a behavioral transfer diagnostic and introduces no mechanism.
- **F9/F10** provide no verdict threshold for this question and motivate deletion of the optional gradient-geometry measurement.
- **DEC-0013** deferred diagnostic-first sequencing; **DEC-0014** activated a separate project-original directed-relation arm and kept deferred diagnostics deferred. Neither decision defines a numerical transfer gate. The `δ_DG` justification comes from DG-0001's pre-registered transfer-materiality bands.
- **Design-closure evidence:** loader and exposure semantics are verified in `downstream/dataset/load_embedding.py`, `downstream/dataset/preprocess_embedding.py`, `downstream/dataset/custom_emb_dataloader.py`, `downstream/trainer/trainer_model.py` and `improvements/base/kfold_iemocap.py`; the coarse historical step envelope comes from `studies/DG-0001/analysis.md`; resource, compatibility and timing facts come from `WORKER_ENVIRONMENT.md`; the spend fail-closed rule comes from `.agents/policies/autonomy.md`.
- **Corpus state:** `WORKER_ENVIRONMENT.md` records that the raw-corpus farm and cache volume were destroyed, the raw source locations are not recorded in-tree and IEMOCAP is licence-gated. Corpus, label/split identity and embedding availability are therefore blocking prerequisites, not assumptions.

## Hypotheses

For auxiliary `c ∈ {SI, KS}`, seed `s ∈ {0,1,2,3,4}` and fold `f ∈ {0,…,9}`:

`d[c,s,f] = ER_accuracy(pair[c],s,f) − ER_accuracy(control[c],s,f)`,

`T[c,s] = mean_f d[c,s,f]`, and

`D[s] = T[SI,s] − T[KS,s]`.

The five seeds are the independent units. The 50 seed-fold observations per cell are nested; they are never treated as 50 independent observations. The sole diagnostic SESOI is `δ_DG = 0.010` ER-accuracy.

- **H1 (primary — behavioral pair-selective residual).** Under exact within-cell opportunity matching, ER←SI is harmfully negative, ER←KS is practically null and `D` is negative beyond `δ_DG`. This is behavioral only: it does not claim that SI semantics caused the effect and does not establish full directionality.
- **H2 (competing — generic auxiliary-loss interference).** Both auxiliaries harm ER beyond `δ_DG`, while `D` is practically equivalent to zero.
- **H3 (third outcome — no residual after exact matching).** Both ER-target residuals are practically equivalent to zero.
- **INCONCLUSIVE.** Every valid pattern outside the three regions above, including opposite signs, a non-null KS cell without the H1 pattern or intervals too wide to establish harm or equivalence.

`H1`, `H2`, `H3` and `INCONCLUSIVE` are the complete, mutually exclusive screening vocabulary used throughout DG-0008. Every classification is screening evidence only. DG-0008 ends after screening regardless of class. Any confirmation is deferred to a future, separate proposal and separate Study that must be independently designed, independently reviewed and separately authorized; its seed set, protocol details and cost are not part of DG-0008.

## Proposed study

**Category.** `DG-xxxx` diagnostic: behavioral transfer characterization only. It proposes no mechanism, does not increment the mechanism plateau counter and cannot authorize a relation architecture.

**Regime.** Frozen `wavlm_large` embeddings; mean frame pooling; `smp(0.5)` over all 25 layers; the DG-0001 five-epoch LOSO architecture and regularization settings; batch size `2048` with `drop_last_train: true`; AdamW with fixed LR `0.0025` at every step; fixed-final-epoch ER accuracy as the sole scientific endpoint. KS and SI outcomes are not endpoints. Validation and test speakers are disjoint in every fold.

**Independent variable.** Whether the auxiliary task's cross-entropy term is enabled. Both arms use the same two-head model and the same combined input stream. Pair: `w_ER = 0.5`, `w_aux = 0.5`; control: `w_ER = 0.5`, `w_aux = 0`. This is a default-off diagnostic hook, not a proposed loss-weighting method.

**Matched controls.** Each pair arm has a within-cell control using the same pair-task loader and model with only the auxiliary loss disabled. Pair and control therefore share dataset membership, ordered manifests, epoch and step counts, per-step ER counts, serialized initialization, Python/NumPy/CPU-Torch/CUDA/DataLoader-worker/dropout RNG states, forward graph, target-loss coefficient, LR sequence, precision, checkpoint point and evaluation code. Both execute the forward graph before the loss-term switch. The validation-driven scheduler is removed symmetrically. Weight decay `5e-8`, L1 `1e-7`, L2 `1e-5` and the optimizer implementation are common. This fixed-LR regime is a new exact-opportunity regime; it is not represented as a reproduction of DG-0001's scheduler trajectory.

### Opportunity and identity manifests — fully declarative specification

A committed CPU-only, no-training generator, `generate_opportunity_manifest.py`, must implement the following contract exactly. A mismatch is fatal; an implementation may not choose a different normalization, serializer, ordering or RNG.

1. **Indices and epoch convention.** Cells are exactly `ks_er` and `si_er`; folds are `f ∈ {0,…,9}`; seeds are `s ∈ {0,1,2,3,4}`. The five training epochs are zero-based training passes `e ∈ {0,1,2,3,4}`: `e=0` is the first complete pass beginning from the serialized initial state and `e=4` is the fifth and final pass. Within each epoch, steps are zero-based `k ∈ {0,…,S[cell,f]−1}`. The scientific endpoint is measured after `e=4`.

2. **Canonical path normalization.** Given a dataset reader's relative path string, canonicalize it by replacing every backslash with `/`; reject NUL, an absolute path, a drive-qualified path, and any empty, `.` or `..` segment; then join the unchanged segments with `/`. Do not call filesystem resolution, dereference symlinks, case-fold, percent-decode or apply Unicode normalization. The resulting Unicode string is encoded as UTF-8 wherever bytes are required. Canonical keys must be unique within a component; a duplicate after normalization is fatal. Ordering is ascending lexicographic order of the canonical keys' UTF-8 byte strings, not filesystem order, reader order, locale order or hash order.

3. **Dataset-specific canonical keys and labels, matching the code.** The generator reads raw-dataset metadata without opening an embedding file and without calling an embedding wrapper's `__getitem__`:
   - **Speech Commands:** instantiate the torchaudio base reader for each official subset. For base `get_metadata(n)`, canonical key = normalized `metadata[0]`, the path relative to the dataset root and the same `wav_relpath` that `SPEECHCOMMANDSEmbedding.get_metadata` transforms with `.replace('.wav', ...)`; source label = `str(metadata[2])`. Effective KS class = that source label if it is a key in `LabelKeywordMapping.LABEL2INDEX_SPEECHCOMMANDv1`, otherwise `_unknown_`; record the corresponding effective integer index. Official membership is exactly one of `training`, `validation`, `testing`, as established by the raw distribution's official validation/testing lists and the base reader.
   - **VoxCeleb1:** instantiate the torchaudio base reader for each official subset. For base `get_metadata(n)`, canonical key = normalized `metadata[0]`, the relative wav path used by `VoxCeleb1Embedding.get_metadata`; effective SI speaker id = `str(metadata[2])`, with its `LabelKeywordMapping.LABEL2INDEX_VOXCELEB1` index also recorded. Official membership is exactly one of `train`, `dev`, `test`, derived from the restored corpus's `iden_split.txt` through the base reader.
   - **IEMOCAP:** instantiate the torchaudio base `IEMOCAP` reader for sessions 1–5. Iterate its `self.data` entries, each a `wav_stem`; obtain the raw relative wav path and source emotion from `self.mapping[wav_stem]["path"]` and `["label"]`. Canonical key = the normalized mapping path, which is the base `get_metadata(n)[0]` path used to locate the embedding; source label is the mapping label, corresponding to base `get_metadata(n)[3]`. Drop `fru` exactly as `IEMOCAPEmbedding.__init__` does. Effective ER label = `hap` when the source label is `exc`, otherwise the source label; record the corresponding `LabelKeywordMapping.LABEL2INDEX_IEMOCAP` index. Speaker id is parsed from canonical-path component 4 with `.split("_")[0]`, matching `IEMOCAPEmbedding` and `build_loso_fold`. Any retained label absent from the effective mapping, any path with fewer than four components or any speaker set other than the ten required by the fold builder is fatal.

4. **Digest-addressed example/split identity manifest.** The independent source of labels and membership is the restored raw corpus's official metadata: Speech Commands' official split lists and base metadata, VoxCeleb's `iden_split.txt` and base metadata, and IEMOCAP's annotation-derived base `data`/`mapping`. Embedding filenames, embedding tensors, model predictions and embedding SHA-256s are not label or split evidence. The generator emits exactly this object shape:

   `{"schema":"dg0008.example-identity.v1","components":[{"component":"speechcommand","records":[SC_RECORD,...]},{"component":"voxceleb","records":[VOX_RECORD,...]},{"component":"iemocap","records":[ER_RECORD,...]}],"iemocap_folds":[FOLD,...]}`

   Components occur in the displayed order. Records in each component are sorted by canonical-key UTF-8 bytes. `SC_RECORD` is exactly `{"key":str,"source_label":str,"effective_label":str,"effective_label_index":int,"official_split":str}`. `VOX_RECORD` is exactly `{"key":str,"effective_speaker_id":str,"effective_label_index":int,"official_split":str}`. `ER_RECORD` is exactly `{"key":str,"source_label":str,"effective_label":str,"effective_label_index":int,"speaker_id":str,"fold_membership":[{"fold":int,"split":str},...]}`; its ten membership entries are ordered by fold `0…9` and use only `train`, `validation`, `test`. `FOLD` is exactly `{"fold":int,"test_speaker":str,"validation_speaker":str}`, ordered by fold `0…9`; speakers are the UTF-8-byte-sorted unique speaker ids, with test and validation assigned by `build_loso_fold` and all other speakers assigned to training. Every loader-visible example in all official KS/SI splits and every retained IEMOCAP example must occur exactly once in its component records.

5. **Canonical JSON bytes and identity digest.** For every object in this specification, `J(x)` means exactly the bytes from Python `json.dumps(x, sort_keys=True, separators=(",",":"), ensure_ascii=False, allow_nan=False).encode("utf-8")`: no BOM, indentation, spaces or trailing newline. The identity-manifest digest is lowercase `sha256(J(identity_object)).hexdigest()`. The complete canonical identity JSON and its digest must be recorded in the registered Study's compute inputs before training.

6. **Version-independent ordered training vectors.** The Speech Commands vector is the identity records whose `official_split` is `training`, sorted by canonical-key bytes. The VoxCeleb vector is the records whose `official_split` is `train`, sorted the same way. For fold `f`, the IEMOCAP vector is the records whose fold-`f` membership is `train`, sorted the same way. No torchaudio enumeration index survives into these vectors. For `ks_er`, component order is `[speechcommand, iemocap]`; for `si_er`, it is `[voxceleb, iemocap]`. Global index `i` is the concatenation of component 0 followed by component 1. Thus component identity, labels and fold membership come from the identity manifest, while loader-version enumeration order cannot affect the schedule.

7. **Lengths and steps.** `L[cell,f] = N_0 + N_1`; `S[cell,f] = floor(L[cell,f] / 2048)`; the trailing partial slice is dropped. `L`, `S` and the per-step ER counts are unknown until the lawful corpus is restored and must not be fabricated.

8. **Manifest seed.** For every epoch, use the UTF-8 bytes of `"DG-0008|manifest|" + cell + "|f=" + str(f) + "|s=" + str(s) + "|e=" + str(e)` verbatim.

9. **Permutation algorithm.** Apply counter-based keyed Fisher–Yates to `IN = list(range(L))`. For `i` from `L−1` down to `1`, compute `key_i = SHA-256(("perm|" + manifest_seed + "|i=" + str(i)).encode("utf-8")).digest()`, set `j = int.from_bytes(key_i[:8], "big") % (i+1)`, and swap `IN[i]` and `IN[j]`. This is the only permitted permutation algorithm.

10. **Batching and epoch-indexed ER counts.** Step `k` is the ordered slice `IN[2048k:2048(k+1)]`, mapped to `[component_index, canonical_key]`. Both arms of a cell consume that exact step. Define

    `n_ER[cell,s,f,e,k] = #{entry in step k whose component is iemocap}`.

    The epoch index is mandatory because each epoch has a different frozen permutation. No occurrence of `n_ER` denotes an epoch-free scalar or vector.

11. **Per-epoch manifest schema and digest.** For each `(cell,f,s,e)`, emit exactly:

    `{"schema":"dg0008.opportunity-manifest.v2","cell":str,"fold":int,"seed":int,"epoch":int,"length":int,"steps":int,"step_keys":[[[int,str],...],...],"n_er":[int,...]}`.

    `step_keys` has exactly `S[cell,f]` batches of exactly 2048 entries; `n_er` has exactly `S[cell,f]` entries and `n_er[k] = n_ER[cell,s,f,e,k]`. The per-epoch digest is lowercase `sha256(J(manifest_object)).hexdigest()`.

12. **Exact tuple-to-digest association and aggregate bytes.** A five-epoch training run is identified for scheduling by `(cell,f,s)`; arm is deliberately absent because pair and control must share the schedule. For that run, construct exactly the JSON array

    `[[cell,f,s,0,digest_0],[cell,f,s,1,digest_1],...,[cell,f,s,4,digest_4]]`

    in ascending integer epoch order, where each `digest_e` is the lowercase per-epoch digest for that exact tuple. The run-level aggregate digest is lowercase `sha256(J(run_array)).hexdigest()`. That same run-level digest is attached to the pair job, control job and, where applicable, the determinism repeat. No delimiter-joined text or bare sorted digest list is permitted. The Study also records one matrix-level digest over the exact array of all `[cell,f,s,e,digest]` entries, sorted by `(cell UTF-8 bytes, f, s, e)` ascending and hashed as `sha256(J(matrix_array)).hexdigest()`. These arrays preserve the tuple-to-digest association explicitly.

13. **Dual identity admission, fail closed.** Before any training, a lawfully restored source corpus is used to produce and independently review the canonical identity manifest; its bytes and digest are then frozen in the registered Study's compute inputs. An execution environment is admitted only if both conditions hold: (a) every declared canonical embedding object matches the compute plan's SHA-256 set; and (b) a fresh extraction from the raw corpus produces byte-identical `dg0008.example-identity.v1` JSON and the same identity digest. Embedding equality cannot substitute for condition (b), because labels can change while embedding keys and fold membership remain unchanged. A mismatch in either identity is a stop. The DG-0001 `S` envelope is only a coarse sanity check and never proof of identity.

14. **Freeze point.** Immediately after lawful restoration and before any training or held-out endpoint read, record the identity object/digest, every per-epoch object/digest, every run-level digest, the matrix-level digest and the exact emitted `L`, `S` and `n_ER[cell,s,f,e,k]` values. No scientific job starts until these artifacts are frozen and all strict-schema checks pass.

**Opportunity table.** Exact integers remain generator outputs rather than invented values.

| Cell | Pair arm | Matched control | Shared stream | Manifest | `S[cell,f]` | `n_ER[cell,s,f,e,k]` | Pair weights | Control weights | LR | Init/RNG |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ER←SI, decisive | `si_er`, SI loss on | same `si_er` dataset/model, SI loss off | canonical `[voxceleb, iemocap]` | one immutable manifest per `(si_er,f,s,e)`; one aggregate digest per `(si_er,f,s)` | `floor((N_vox+N_er[f])/2048)`; DG-0001 envelope is sanity-only | generator-emitted epoch-indexed vector | `(0.5,0.5)` | `(0.5,0)` | `0.0025` every step | one serialized init and reset RNG snapshot shared by the pair |
| ER←KS, candidate-null | `ks_er`, KS loss on | same `ks_er` dataset/model, KS loss off | canonical `[speechcommand, iemocap]` | one immutable manifest per `(ks_er,f,s,e)`; one aggregate digest per `(ks_er,f,s)` | `floor((N_sc+N_er[f])/2048)`; DG-0001 envelope is sanity-only | generator-emitted epoch-indexed vector | `(0.5,0.5)` | `(0.5,0)` | `0.0025` every step | one serialized init and reset RNG snapshot shared by the pair |

Exactness is within a cell only. SI and KS produce different streams and different `S`; that cross-cell difference is a residual limitation, not exact matching.

### Reproducibility rule — pinned and fail closed

- Every training run sets `torch.use_deterministic_algorithms(True, warn_only=False)`, `torch.backends.cudnn.deterministic=True`, `torch.backends.cudnn.benchmark=False`, `torch.backends.cuda.matmul.allow_tf32=False` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and uses the Study's committed lockfile, pinned CUDA/cuDNN stack, one GPU model for all comparisons and the same worker environment.
- The screening validity gate includes one same-arm repeat: the `si_er` control at fold `0`, seed `0` is trained a second time using the same five run-manifest epochs, run-level digest, initialization and RNG snapshot. The two executions must agree bitwise on the per-step loss trace, fixed-final validation ER accuracy and serialized checkpoint. No held-out test result from the repeat is used.
- Any deterministic-algorithm error or bitwise disagreement is fatal to DG-0008: no tolerance is invented, no held-out endpoint is read and the design must be revised and independently reviewed before any new proposal can proceed. A later tolerance would require evidence of a named nondeterminism source and a new pre-registered metric.

### Staged plan

- **Stage 0 — design closure (non-scientific, zero GPU).** Revision 5 freezes the design contract. Corpus restoration, independent raw-metadata extraction, manifest generation and digest recording are execution prerequisites, not scientific results.
- **Stage 1 — screening (the only scientific stage in DG-0008).** Use seeds `0–4`, all ten LOSO folds and the two fixed cells.
  - **Environment and implementation gate:** restore the lawful corpus; verify the embedding digest set and example/split identity manifest; generate and freeze every opportunity manifest; then run the four fold-0/seed-0 matrix members—pair and control for each cell—sequentially, plus the one same-arm determinism repeat. Do not invoke or read held-out test evaluation. Verify every equality and bitwise gate. The four matrix members are retained for the full matrix; they are not additional jobs.
  - **Screening measurement:** complete the fixed `2 cells × 2 arms × 5 seeds × 10 folds = 200` five-epoch matrix. The four validated fold-0/seed-0 members count inside these 200 and are not rerun. Only after all 200 matrix trainings and the repeat are valid may the frozen checkpoints' held-out endpoints be evaluated and the pre-registered analysis run. No cell is selected or dropped based on validation behavior.

Successful DG-0008 execution therefore consists of exactly **200 five-epoch matrix training jobs plus one same-arm five-epoch determinism repeat = 201 five-epoch training jobs**, plus CPU-only identity/opportunity manifest generation and analysis. No reverse cell, KS↔SI cell, 30-epoch A0/A1 scale experiment or confirmation work belongs to DG-0008. DG-0008 stops after Stage 1 for `H1`, `H2`, `H3` and `INCONCLUSIVE` alike.

## Discriminating measurements

- **Primary:** the five seed-level values `T[SI,s]`, `T[KS,s]` and `D[s]`, each summarized by its mean, SD and two-sided one-sample 95% t interval over seeds. The H1/H2/H3/INCONCLUSIVE interval rules below are the only scientific classification rules.
- **Validity:** exact equality within each pair of the identity digest, per-epoch and run-level manifest digests, ordered batch entries, `n_ER[cell,s,f,e,k]`, step counts, target-loss coefficient, LR sequence, initialization hash and pre-forward RNG sequence, plus bitwise agreement of the repeat. These decide admissibility; they are not evidence for a transfer residual.
- **Nested description:** fold-level residuals, held-out-speaker ranges and fold sign counts. They describe heterogeneity but are never analyzed as 50 independent observations.
- **What does not discriminate:** aggregate accuracy, KS/SI outcomes, manifest sizes, relation-object magnitude and the deleted gradient-geometry suite cannot distinguish H1 from H2/H3. No gradient diagnostic job, schema or endpoint is part of the matrix.

## Success and stop conditions

**SESOI.** `δ = δ_DG = 0.010` ER-accuracy, fixed before execution. DG-0001 pre-registered this as the transfer-materiality boundary. It is not the champion no-regression bar and is not derived from DEC-0014.

**Seed-level intervals.** For each quantity `X ∈ {T[SI], T[KS], D}`, compute `mean_X`, sample SD `sd_X` over the five seed values and half-width `h_X = t_.025,4 × sd_X / sqrt(5)`. Let `I_X = [mean_X−h_X, mean_X+h_X]`. The paired construction for `D` handles covariance between the two cells; no zero-covariance assumption enters the realized decision. Boundary equality is not success: all region inequalities below are strict.

**Precision scenarios, not power or bounds.** The record does not contain the between-seed SD of these paired LOSO residuals, so no power or decision probability is claimed. The revision-4 scenarios are retained only to illustrate interval precision:

| Scenario (assumption, not a bound) | `σ_T` | expected `h` | directional harm can be established only if | equivalence can be established only if |
| --- | ---: | ---: | --- | --- |
| E1 — pooled single-run aggregate seed SD (F1), zero-covariance illustration, optimistic | 0.0016 | 0.20pp | `mean + h < −δ`, i.e. `mean < −1.20pp` | `mean−h > −δ` and `mean+h < +δ`, i.e. `−0.80pp < mean < +0.80pp` |
| E2 — ER paired ordinary-split SD (DG-0005), speaker-leaky and a different intervention, pessimistic | 0.0064 | 0.79pp | `mean + h < −δ`, i.e. `mean < −1.79pp` | `mean−h > −δ` and `mean+h < +δ`, i.e. `−0.21pp < mean < +0.21pp` |

E1 is not an ER-LOSO residual estimate, and a paired difference's SD is not its zero-covariance combination. E2 uses a speaker-leaky split and another intervention. Neither scenario bounds `σ_T`; neither supplies a power claim.

**Exhaustive screening classification.** After all validity gates pass:

- **H1:** `mean_T[SI] + h_T[SI] < −δ`; `mean_T[KS] − h_T[KS] > −δ` **and** `mean_T[KS] + h_T[KS] < +δ`; and `mean_D + h_D < −δ`.
- **H2:** `mean_T[SI] + h_T[SI] < −δ`; `mean_T[KS] + h_T[KS] < −δ`; and `mean_D − h_D > −δ` **and** `mean_D + h_D < +δ`.
- **H3:** for each `c ∈ {SI,KS}`, `mean_T[c] − h_T[c] > −δ` **and** `mean_T[c] + h_T[c] < +δ`.
- **INCONCLUSIVE:** every other valid result, including any interval touching or crossing a required boundary, any positive or mixed-sign pattern outside H1–H3 and any interval too wide to establish the required harm or equivalence. A wide interval is never evidence of equivalence.

These four classes are exhaustive and mutually exclusive because harm requires an interval wholly below `−δ`, while equivalence requires an interval wholly inside `(−δ,+δ)`. The prose, hypotheses and classification use the same regions. Every class is passed only to a future, separately designed and authorized confirmation proposal; none promotes a finding or continues DG-0008.

**Stage-1 validity gate.** Required counts are: `0` embedding-digest mismatches; `0` identity-manifest byte/digest mismatches; `0` opportunity-manifest or aggregate-digest mismatches; `0` mismatched batch entries within a pair; `0` mismatches in ER key order or `n_ER[cell,s,f,e,k]`; `0` differences in step count, ER coefficient, LR sequence, initialization hash or pre-forward RNG-state sequence; the emitted `S` values inside the coarse DG-0001 sanity envelope; a bitwise-identical same-arm repeat; `0` held-out test reads before all gates and all 200 matrix trainings pass; successful checkpoint creation and read-back. Any failure invalidates the execution and is not evidence for or against H1–H3.

All 200 matrix jobs and the repeat must finish validly before classification; there is no outcome-based early stop. An infrastructure-failed job may be repeated only with the same identity digest, run-manifest digest, initialization, config and commit, with the failed attempt retained. After classification, stop DG-0008 regardless of outcome. It cannot establish full directionality, close KS↔SI, support a cross-direction magnitude claim or authorize a directed mechanism.

## Expected compute

**Stage 0:** zero GPU-hours. Design closure is documentary. The required identity and opportunity manifests are generated CPU-only after lawful corpus restoration.

**Stage 1:** exactly 201 five-epoch training jobs: 200 fixed matrix members and one same-arm repeat. The four fold-0/seed-0 validation-gate jobs are members of the 200 and are reused. Total GPU-hours are **UNKNOWN** until a compatible-GPU benchmark measures or safely bounds five-epoch training-plus-validation time for all four arm classes; no DG-0008 GPU-hour or dollar ceiling is claimed now. Stage-1 gate jobs run one at a time so timing, cache state and determinism are interpretable. Remaining matrix jobs may use no more than the protocol's two-job concurrency cap (`VARIANT_BENCHMARK_PROTOCOL.md` §9); an authorization must price the chosen concurrency and lifetime rather than assume ideal scaling.

Repository-supported resource constraints are not a cost estimate: one live job workspace at a time, approximately the recorded per-job workspace requirement, container disk at or above the recorded minimum, the raw corpus on an attached network volume of the recorded size class and the volume attached when the worker is created (`WORKER_ENVIRONMENT.md` §§4–5). The recorded warm-cache timing in `WORKER_ENVIRONMENT.md` §7 is for a different 30-epoch three-task workload and is not transferred to this five-epoch exact-opportunity workload.

The following UNKNOWNs each block registration for spend and block compute authorization until measured or bounded in a human-owned envelope:

1. lawful corpus acquisition/restoration time and cost, including access to licence-gated IEMOCAP;
2. network-volume creation/storage hourly rate, retention lifetime and cold-cache embedding materialization/transfer time;
3. five-epoch training-plus-validation duration on the selected GPU for each of the four arm classes and the repeat;
4. worker setup, idle and teardown time and a bounded infrastructure-failure/retry allowance;
5. the selected compatible GPU offer's current hourly rate, the worker's maximum paid lifetime/lease deadline and the resulting total-cost ceiling.

**GPU compatibility and price/lifetime guard.** `WORKER_ENVIRONMENT.md` §6 records that the pinned image tops out at `sm_90`; Blackwell parts are therefore hard-filtered before price. At the authorization decision, discover offers with `infra worker gpu-types --data-center <dc> --require-direct-ssh`, where `<dc>` is the network volume's data center, and select the cheapest compatible offer in that data center. Record its then-current rate; do not reuse a historical rate. The human authorization envelope must bound hourly price, total spend and lifetime, and worker creation must pass the envelope's hourly ceiling through `--max-price`. Under `.agents/policies/autonomy.md`, unknown price or unknown lifetime fails closed for new spend. If the offer rate, maximum paid lifetime, volume lifetime or complete ledger bound remains unknown—or exceeds the envelope—DG-0008 stops before provisioning.

The raw tree must also satisfy `WORKER_ENVIRONMENT.md` §§2–4 and the loader-construction check, then pass the dual identity admission above. If any source cannot be lawfully obtained, the required layout cannot be reconstructed, the independent raw-metadata identity cannot be established or either digest identity fails, Stage 1 cannot run. No substitute corpus, labels, split or bytes may be fabricated.

No compute, seed set or cost for future confirmation is estimated here. Those belong to a future proposal and Study after DG-0008 has stopped.

## Risks and confounds

1. **No semantic-negative control.** Enabling a real auxiliary loss under a fixed stream identifies a behavioral loss-term effect. It cannot separate task semantics from auxiliary gradient distribution, corpus identity or class count. No semantic claim is licensed.
2. **Only within-cell exactness.** ER←SI and ER←KS have different streams and step counts. `D` contrasts two within-cell effects; it is not an exposure-identical cross-cell intervention.
3. **No full directionality.** Reverse cells and KS↔SI are intentionally absent.
4. **New fixed-LR regime.** Removing the scheduler prevents arm-dependent LR paths but characterizes this controlled regime, not DG-0001's historical scheduler path.
5. **Unknown `n=5` precision.** No multi-seed paired ER-LOSO residual exists. The precision examples are scenarios, not bounds or power, and borderline outcomes become INCONCLUSIVE.
6. **Corpus restoration is fail closed.** Exact `L`, `S` and `n_ER[cell,s,f,e,k]` do not exist until the corpus is lawfully restored. Counts alone do not establish identity. Both the embedding-digest set and independently sourced label/split identity manifest must match.
7. **Determinism is stack-dependent.** Bitwise agreement is required. If the pinned operation set cannot execute deterministically, DG-0008 fails rather than receiving an invented tolerance.
8. **Environment availability.** The cache volume and raw farm are gone, IEMOCAP is licence-gated and raw source locations are unrecorded. This may prevent Stage 1 entirely.
9. **Test-workflow leakage.** Cells, thresholds, labels, splits, manifests, digests and analysis code are frozen before any held-out endpoint is read; no held-out result is read until all 200 matrix trainings and validity gates pass.
10. **Cost and lifetime are unknown.** Unknown corpus/storage costs, selected-GPU rate, job duration, worker lifetime or ledger ceiling are stops, not estimates. Spend remains bounded by a future human envelope.
11. **A screening class is not confirmation.** Even H1 or H2 remains screening evidence. Conversely, H3 is not an established null and INCONCLUSIVE is not evidence for equivalence. Every class requires a future independent proposal before any confirmation work.

The residual assumption is therefore narrow and explicit: within each cell, after byte-identical data identity, schedule, initialization and deterministic execution are established, the enabled auxiliary loss is the only intended arm difference; across cells, semantic and stream differences remain inseparable.

## Repository effects

If later approved and registered, DG-0008 would create `studies/DG-0008/{PLAN.md,NOTE.md,configs/,compute/,analysis.md,result.json}` and append one DG-0008 record to `STUDIES.jsonl`. It would add a committed CPU-only identity/opportunity-manifest generator implementing the strict schemas and digest contracts above; a manifest-driven sampler; a default-off auxiliary-loss control hook; a fixed-LR diagnostic mode; deterministic pinning and same-arm repeat checks; initialization/RNG/identity/manifest trace artifacts; scoped tests for canonical keys, labels, split/fold membership, exact JSON bytes, tuple-associated digests and pair equality; and a frozen seed-level analysis with folds nested.

DG-0008 would not add gradient-diagnostic code or a confirmation result. It would create no future confirmation proposal or Study automatically. Any such work receives a different proposal and Study ID after independent design, review and human authorization.

A human-granted Stage-1 authorization would be a separate `authorizations/DG-0008.yaml`; this proposal creates none and claims no spend. Post-result updates could touch `FINDINGS.md`, `FAILURES.md`, `FRAMEWORK.md` and `STATE.md` only to the extent licensed by the bounded ER-target screening result. They would not modify DEC-0014, register or authorize TR-0008, or claim KS↔SI/full-direction closure.

## Human decisions required

1. Decide whether to reopen this deferred diagnostic at all, given DEC-0013/DEC-0014 and that DG-0008 does not gate the active directed mechanism.
2. Decide whether behavioral ER pair-selectivity only, `n=5` screening and no confirmed claim are worth the 201-job five-epoch workload.
3. Approve or refuse the exact-opportunity intervention: the same two-head stream in both arms, `(0.5,0.5)` versus `(0.5,0)`, fixed LR `0.0025`, strict identity/opportunity manifests and pin-and-fail-closed reproducibility.
4. Approve or refuse `δ_DG = 0.010`, the exhaustive H1/H2/H3/INCONCLUSIVE interval map and the rule that DG-0008 stops after screening for every class.
5. Decide whether a semantic claim is required. If yes, DG-0008 must be rejected or redesigned with a separately validated semantic-negative control; if no, accept the behavioral-only scope.
6. Confirm deletion of optional gradient diagnostics; that question would require a separate proposal.
7. After lawful corpus restoration establishes both identities and after all five cost/rate/lifetime unknown classes are measured or bounded, decide whether to register DG-0008 and grant a bounded human authorization envelope. Until then, provisioning and compute remain blocked.
8. After DG-0008 has stopped and reported its screening class, decide whether to request a new confirmation proposal. That future proposal must choose and justify its own seed set, precision target, compute, cost and stop rule; DG-0008 grants it no automatic promotion, registration or authorization.

## Review

**Revision 5.** The independent re-review of revision 4 returned **CHANGES_REQUIRED** on 2026-10-03 with four blockers. Revision 5 makes only the required design-closure changes; the fresh independent review of revision 5 returned **PASS** (below).

1. **Manifest executable degrees of freedom — resolved in the design.** Epochs are explicitly zero-based `e ∈ {0,1,2,3,4}` and every occurrence and schema now uses `n_ER[cell,s,f,e,k]`. Canonical JSON bytes are defined once; per-epoch, per-run and matrix digests have exact tuple-associated arrays and sort orders. Dataset-specific key extraction follows the verified loader code for Speech Commands, VoxCeleb and IEMOCAP, with a version-independent path normalization and UTF-8 ordering rule. A strict digest-addressed example/split identity schema records effective KS class, SI speaker id, ER label after `exc→hap`, official membership and every LOSO fold membership from raw-corpus metadata independent of embeddings. Admission requires both the embedding SHA-256 set and a byte-identical identity manifest.
2. **Contradictory outcome map — resolved in the design.** The only classifications are `H1`, `H2`, `H3` and `INCONCLUSIVE`, exactly matching the hypothesis regions plus a catch-all. Harm is directional (`mean+h < −δ`); equivalence requires the complete interval strictly inside `(−δ,+δ)`. The scenario table, prose and class rules use those same inequalities.
3. **Confirmation lifecycle — resolved by scope separation.** DG-0008 has one scientific stage, Stage 1 screening, and stops after it for every class. Any confirmation is a future, separately designed, independently reviewed, separately registered and separately authorized proposal/Study. Its seeds and cost are outside DG-0008.
4. **Resource count and price guard — resolved in the design.** The four fold-0/seed-0 jobs are explicitly reused members of the 200-job matrix, so successful execution is exactly 200 matrix trainings plus one repeat, 201 five-epoch training jobs total, plus CPU-only manifest work. Authorization is additionally blocked on the selected compatible GPU's current hourly rate and a human envelope bounding price, total spend and lifetime. The pinned image's `sm_90` limit hard-filters Blackwell before price; offers are discovered in the volume's data center at decision time, the cheapest compatible offer is selected and the envelope ceiling is passed through `--max-price`. Unknown price or lifetime is an explicit stop.


### Independent review of revision 5 — verdict: PASS

The independent `research-reviewer` returned **PASS** (2026-10-03). It checked each of the four revision-4 blockers against the code and the records and found all four resolved, with **zero blocking findings**, and confirmed the authority check (no registration, authorization, provisioning or compute).

- **Blocker 1 (manifest):** resolved — epoch convention and epoch-indexed `n_ER`; canonical-JSON bytes; per-epoch/run/matrix digest contracts with exact tuple-to-digest association and sort orders; dataset-specific key extraction verified against `downstream/dataset/preprocess_embedding.py`, `downstream/utils/constant_mapping.py` and the torchaudio readers; and a digest-addressed identity manifest with an independent raw-corpus label/split source and dual (embedding-digest + identity) admission.
- **Blocker 2 (outcome map):** resolved — the vocabulary is exactly `H1/H2/H3/INCONCLUSIVE`; the inequalities are directional and identical across the hypothesis prose, the scenario table and the class rules; the regions are disjoint and exhaustive.
- **Blocker 3 (confirmation lifecycle):** resolved — DG-0008 has one scientific stage and stops for every class; confirmation is a separate future proposal/Study; no `Stage 2` language remains.
- **Blocker 4 (resource/cost):** resolved as specified — 200 matrix trainings + 1 repeat = 201; the selected compatible GPU's then-current rate and a human price/spend/lifetime envelope are authorization-blocking, with `--max-price` and fail-closed unknown price/lifetime.

**Non-blocking findings** (recorded, not blocking; the reviewer states none affects the validity of the design or its decision rules):

1. The 200 held-out endpoint evaluations are GPU work not itemized in the job/compute accounting, while the summary calls the trailing work `CPU-only`. Total cost is UNKNOWN and envelope-bounded, so the ceiling is not misstated, but the itemization is incomplete.
2. Scenario E1's `sigma_T = 0.0016` is attributed to F1, which records 0.00130 / 0.00091; E1 is illustrative only and no rule consumes `sigma_T`.
3. The equivalence band uses `delta_DG = 0.010` (DG-0001's materiality boundary) rather than its 0.002 near-zero boundary, so DG-0001's 0.002-0.010 `ambiguous` residuals classify as null. Pre-registered, numeric and consistently applied.
4. How the per-seed serialized initialization/RNG snapshot is derived from `s` is left implicit; the pair comparison is within-seed and paired, so the estimand is unaffected.
5. Clause 9's concatenation mixes a `str` literal with the clause-8 byte string; the intended bytes are unambiguous, so implementations still converge.
6. Whether the full per-epoch manifest objects or only their digests are persisted is not stated; UNKNOWN item 2 covers storage rate and lifetime.

These six are optional text/accounting refinements; applying any of them would change the reviewed text and require a fresh independent review. They are surfaced for the human to accept as-is or request.

**Reviewer uncertainties:** the reviewer could not verify commit `a1794dd` (no shell); `L`, `S` and `n_ER` are generator outputs that do not exist until lawful corpus restoration; and whether the raw corpus can be reconstructed byte-for-byte is the fail-closed prerequisite the proposal names.
