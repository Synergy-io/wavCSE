## Verification of repository state

- **HEAD** `8a0878231c6cff6acbe22481856549c572d9ad8d` — *"Record the INC-V2-4 synthesis-layer evaluation"* (adds 74 lines to `LITERATURE_AGENT_V1_EVAL.md`). Branch `feature/mssl-task-relation-study`, pushed 23 commits ahead of origin; working tree has unrelated uncommitted infrastructure/`AGENTS.md` edits (`M`/`??` — not research state). The V2 literature record is committed and is the "latest literature investigation/evaluation record".
- **No research state modified by this session.** Everything below is read-only reconstruction and design.

---

# A. The exact unresolved scientific question

> After the optimizer and estimator confounds are controlled **exactly** (not approximately), does any KS/SI/ER task pair exhibit a **direction-specific, pair-selective, sign-stable** transfer residual — and if so, is that residual a property of task semantics/parameter interference rather than the per-task gradient estimator's variance or scale?

Equivalently: DG-0001/Stage-D left a controlled pattern (ER←KS ≈ 0; SI↔ER negative in both directions) measured with *approximate* exposure matching; DG-0005/F10 showed the ER gradient-scale signal is a training-mixture property whose late-phase mean-gradient component is unmeasured. Whether these two are the same or different phenomena, and whether either reflects genuine task interference, is the decision this study exists to make.

---

# B. Authoritative evidence reconstructed

### DG-0001 raw directed matrix (Stage A, seed 42, 30 epochs, `smp` 0.5, 25L, commit `4c0f08a5`)
Source `task_relations/empirical_transfer.json`:

| Target ← aux | KS | SI | ER |
|---|---:|---:|---:|
| KS | — | −0.000585 | +0.000146 |
| SI | −0.001697 | — | −0.008847 |
| ER | +0.057866 | +0.028933 | — |

`T(A←B) = acc(A jointly trained with B) − acc(A alone)`. Screening class: `candidate_asymmetry`; ER cells use the speaker-leaky split.

### DG-0001 Stage C/D — the exposure-controlled residuals (F8)
Source `studies/DG-0001/analysis.md`, `task_relations/{loso_transfer,optimization_control}.json`, commit `330d0f78` / `3f4752af`:

| Quantity | Value |
|---|---|
| ER-only batch 2048 | mean ER LOSO acc 0.33685, ≈2 updates/fold/epoch |
| ER-only batch 160 (targets KS+ER's 27) | 0.67267, 26–28 updates |
| ER-only batch 64 (targets SI+ER's 69) | 0.67518, 67–70 updates |
| ER←KS raw LOSO | **+0.34111** [+0.28469, +0.39754], 10/10 positive |
| ER←SI raw LOSO | **+0.27941** [+0.21117, +0.34766], 10/10 positive |
| ER←KS exposure gain / residual | +0.33582 / **+0.00529** [−0.02886, +0.03943], 5+/5− |
| ER←SI exposure gain / residual | +0.33833 / **−0.05892** [−0.08831, −0.02953], 0+/10− |
| KS←ER reverse (5-epoch) | −0.00380 [−0.00500, −0.00260], 10/10 negative |
| SI←ER reverse (5-epoch) | −0.02310 [−0.02748, −0.01873], 10/10 negative |

**Exact exposure mismatch remaining (F8, "Alternative explanations"):**
1. Batches 160/64 only *approximate* the pair arms' update counts (26–28 vs 27; 67–70 vs 69) and simultaneously change gradient noise and the ER effective batch size (160/64 vs the pair arm's ≈47 ER/batch).
2. Reverse cells compare pair folds against **one** seed-42 target-only reference fit, not ten refits.
3. No KS↔SI exposure control exists at all.
4. Stage-A KS/SI cells are one seed, exposure-uncontrolled.
5. Loss scaling / scheduler trajectory depend on task count.

### F9 — gradient instrumentation (DG-0002, seeds 0–4, commit `7f6d5248`)
- 142 of 2,820 optimizer steps sampled (first, final, every 20th); shared parameters = **1,550,800** (all trainable excluding `classifiers.`).
- Baseline max/min norm ratio: middle `7.243 ± 0.272`, late `8.962 ± 0.456` (5/5 ≥3); MTRL `7.129 ± 0.421`, `9.004 ± 1.059`; paired MTRL−baseline CIs include zero.
- Late mean cosines: KS↔SI +0.002, KS↔ER +0.001, SI↔ER +0.022; no seed meets the persistent-conflict criterion.
- Late ER mean norm `6.972` vs KS `0.828`, SI `0.804`.

### F10 / DG-0005 (seeds 0–4, commit `8032a937`)
- ER sampled share `47/2048 → 435/2048`; late ratio `8.962 ± 0.456 → 2.352 ± 0.270` (paired −6.610 [−7.353, −5.867]); middle `7.243 → 2.818`; ER late norm `6.972 → 1.608` (KS +0.066, SI −0.117); ER/KS realized count ratio 0.990–1.008; 2,820 steps / 142 samples every arm.
- Pre-registered post-hoc bound (`analyze_noise_shape.py` @ `2ac7f3d`): late estimator share ≤ `0.758` [0.715, 0.801] ⇒ **≥20 % (mean 24 %) of the late log-drop is not estimator size**; middle share `1.028` [0.975, 1.082]; per-step dispersion `CV ≈ 0.32` vs isotropic ceiling `≈0.00057`, so dispersion cannot proxy variance.
- ER distinct training pool ≈ 4.4 k (A1 drew ~43 k ER examples/epoch at ~9× repetition); ER/KS "one-knob" confound = estimator variance **and** ~9× optimizer updates ⇒ head saturates (train ≈0.99 / val ≈0.82).

### Protocol constants (all runs)
30 epochs · global batch 2048 · 94 batches/epoch · **2,820 steps** · 193,874 train samples · AdamW lr 0.0025, wd 5e-8, l1 1e-7, l2 1e-5 · effective scheduler patience **1** (`patience: 5` inert) · `smp` 0.5, all 25 layers · shared trunk 1024→512→2000 · standard composition ≈539 KS / 1462 SI / 47 ER per 2048. ER authoritative protocol = 10-fold speaker-independent LOSO, 5 epochs/fold (DG-0001 Stage C). DG-0001 = seed 42; DG-0002/DG-0005 = seeds 0–4.

### What is measured vs unmeasured
- **Measured:** per-task **per-batch** masked-CE gradient norm on shared params, pairwise batch-gradient cosines, negative-frequency, per-step valid-example counts, phase aggregates (`improvements/gradient_diagnostics.py`, logged per run to MLflow/DagsHub as `results/gradient_diagnostics.json`).
- **Unmeasured:** `‖E g‖` (pool-mean gradient); estimator variance `Var(g)` at a **fixed checkpoint**; mean-gradient cosines `cos(E g_A, E g_B)`; loss-weighted gradient scale.

### Reuse inventory
| Asset | Reusable? |
|---|---|
| Frozen WavLM embeddings (shared store) | **Yes** |
| Evaluation + ER-LOSO harness, `gradient_diagnostics.py`, DG-0001 configs/harness | **Yes** |
| DG-0002/0005/DG-0001 result JSONs (summaries), DagsHub per-run `gradient_diagnostics.json` | **Yes** (analysis-only) |
| **Checkpoints / trained states** | **No** — `saved_checkpoint_count: 1`, `outputs/` and `checkpoints/` are gitignored and lived on ephemeral workers; compute backend "never moves artifact bytes"; the shared 200 GB volume was destroyed 2026-09-30 (`WORKER_ENVIRONMENT.md`). No `.pth` exists in this tree. |
| Kaggle/DagsHub raw embedding artifacts | Yes (via `root_emb_path`) |

---

# C. H1/H2/H3 operational definitions

**H1 — exposure artifact.**
- *Mechanism:* the residual is produced by incomplete optimizer-opportunity matching — different update counts, different per-step target batch size, different effective loss scale, or a single-reference reverse arm.
- *Expected observable:* under **exact** target-opportunity matching every ER cell's residual moves inside the materiality band (‖residual‖ ≤ 0.20 pp = 0.002 accuracy), with its CI excluding the DG-0001 magnitudes (−0.059, +0.005).
- *Weakens H1:* a residual of the DG-0001 sign, magnitude outside the band, sign-consistent across seeds/folds under exact matching.
- *Does NOT distinguish H1:* any comparison without exact matching; raw pair-minus-single; per-batch gradient norms; Ω/precision magnitudes; the pool-mean gradient alone (not a transfer quantity).

**H2 — estimator variance / task reliability / gradient-scale.**
- *Mechanism:* per-task gradients estimated from few examples (ER ≈47/batch) have inflated norm (`E‖g‖ ≥ ‖E g‖`) and high variance; the residual and the ER norm dominance are this optimization-noise/scale effect, not semantics.
- *Expected observable (conjunctive):* at a fixed checkpoint, `‖E g_ER‖` is comparable across arms while the batch-gradient variance at matched `n` is large; the inflation factor `E‖g_n‖/‖E g‖` is ≫1 for ER and ≈1 for KS/SI and tracks `1/√n`; mean-gradient cosines ≈ 0; and a noise/scale-matched auxiliary (matched per-step count, semantic content removed) reproduces the residual.
- *Weakens H2:* a systematic mean-gradient asymmetry between arms (`‖E g_ER‖` differing beyond resampling error), or a nonzero mean-gradient cosine aligned with the residual's sign, or a residual that survives variance equalization.
- *Does NOT distinguish H2:* exact-step transfer alone; the already-conditioned raw per-batch norms; middle-phase data.

**H3 — genuine task-specific semantic/optimization interference.**
- *Mechanism:* shared parameters carry incompatible task-specific structure (e.g. speaker-identity features harmful to emotion discrimination; conflicting loss geometries), producing pair-selective sign-consistent interference.
- *Expected observable (conjunctive):* residual survives exact matching **and** the H2 control; sign stable across ≥5 seeds / 10 folds; pair-selective (SI/ER negative while KS/ER ≈0, KS/SI ≈0); ideally a systematic mean-gradient conflict.
- *Weakens H3:* residual inside the band; sign inconsistency; H1 or H2 reproduces it.
- *Does NOT distinguish H3:* object-side asymmetry (`B_st ≠ B_ts`) with no behavioural evidence; single seed; saturated Ω.

---

# D. Candidate diagnostic measurements

**A — pool-mean gradient and fixed-checkpoint variance decomposition.**
For task `t` at checkpoint `θ_c`:
- `E‖g‖` = mean of sampled batch-gradient norms — **already recorded**.
- `‖E g‖` = `‖ (1/N_t) Σ_i ∇_θ ℓ_t(x_i) ‖` — one deterministic pass over the task's full training pool (frozen embeddings ⇒ reproducible).
- `Var(g)` = `E‖g_n − E g‖²` — estimated at the **fixed** checkpoint by drawing `K` independent batches of size `n_t` from the pool (with replacement, fixed RNG seed) and averaging squared deviation from the pool mean. This is legitimate where DEC-0012's prohibition was not: DEC-0012 barred per-step dispersion **along a drifting trajectory**; at a frozen checkpoint with resampling from the pool the drift is gone.
- `cos(E g_A, E g_B)` — mean-gradient cross-task cosines (the current instrumentation records only batch-gradient cosines, which F9 showed are ≈0).
- Inflation curve `E‖g_n‖/‖E g‖` vs `n`.

**B — exact target-opportunity-matched directed transfer.**
`T(A←B) = Y_A(pair A+B) − Y_A(exactly matched A-only)`, with per-step target count, step count, seed, initialization, loss weight and evaluation point identical between the two arms.

---

# E. Information gained by each measurement

| Measurement | Resolves |
|---|---|
| B (exact-matched residual) | H1 vs {H2,H3}: whether the DG-0001 residual is an exposure artifact. Gives the sign-consistent, pair-selective transfer estimate the programme lacks. |
| A `‖E g‖`, inflation, variance | H2's scale/estimator branch: whether ER's dominance is estimator variance (⇒ H2) or a mean-gradient component; converts F10's "≥20 % unexplained" bound from a bound into an identification. |
| A `cos(E g_A, E g_B)` | H3 vs H2: a systematic conflicting mean gradient supports interference; near-zero mean-gradient cosines with a surviving residual points to variance/noise. |
| **A+B together** | The full H1/H2/H3 separation: B gives the behavioural residual; A explains its mechanism (noise vs systematic conflict). Neither alone separates all three. |

# F. Confounds controlled by each

| Confound | B | A |
|---|---|---|
| Optimizer update count | exact (same `S`) | n/a (inference-only) |
| Target examples/step | exact (same `n_A`) | matched `n` used for the variance estimate |
| Loss scaling (`1/num_tasks`) | explicit equal per-task weights | reports weighted and unweighted |
| Initialization / checkpoint / eval point | same seed, same step `S`, same tag | same checkpoint |
| Reverse-direction reference | both directions refit, same schedule (fixes DG-0001) | n/a |
| Gradient-estimate variance | matched `n_A`; aux-noise manipulation optional | **directly measured** |
| Gradient scale | reported | **directly measured** |
| ER split leakage | LOSO where ER is the target | n/a |

---

# G. Recommended sequencing and the §3 choice

**Choice: one combined diagnostic study**, staged. Reasoning:

- A and B target different hypotheses (H2 vs H1/H3) but must be measured **in the same regime** for joint interpretation: the H2 explanation for the residual must come from the checkpoints of the very arms that produced the residual.
- Checkpoints are **not** retained, so a standalone A study would re-train the arms anyway; embedding the pool pass in B's runs costs ≈ one extra epoch-equivalent per arm (forward+backward, no optimizer step) on top of runs that must happen.
- "One measurement added to existing artifacts" is unavailable: the retained artifacts are JSON summaries, not model states. (`E‖g‖` can be re-derived from DagsHub records; `‖E g‖` and `Var(g)` cannot.)
- The alternative — **two sequential studies** — is preferable only if the human separately wants DEC-0012's 30-epoch DG-0005 late-phase scale question answered on its own (a narrower H2 item; ~10 short 30-epoch runs ≈ 6 GPU-h) *before* the transfer study. That item is **not required** to separate H1/H2/H3 for the residual and should be treated as an optional add-on, not a gate.

**Staging (cheap result gates the expensive one):**
`Stage 0 (0 GPU)` → `Stage 1 screen (1 seed, short regime)` → `Stage 2 confirmation (seeds 0–4 + ER LOSO)`.

---

# H. Exact proposed study design

**Regime:** DG-0001 Stage-C/D regime — the regime the residual came from: `ks_si_er`-family pairwise/single arms, `smp` 0.5 all 25 layers, frozen WavLM embeddings, standard composition, AdamW as protocol; **5 epochs** for ER-target LOSO cells (10 speaker folds, 8 train / 1 val / 1 test speakers as in `run_base_er_kfold.py`), matched fixed-epoch for KS/SI targets. (Rationale: the residual is a 5-epoch measurement; reproducing it under exact matching must stay in the same regime.)

### H.1 Exact matching (fixes the batch-size approximation)
For a pair (target A, auxiliary B) at seed `s`, fix target-opportunity budget `n_A` = the pair arm's per-step A-example count, and `S` = optimizer steps, at the pair arm's own values.

- **Pair arm P(A|B):** each step draws exactly `n_A` A-examples and `n_B` B-examples; loss `w_A·L_A + w_B·L_B`.
- **Control arm C(A):** each step draws exactly `n_A` A-examples; `S` steps; loss `w_A·L_A`.
- **`w_A` identical in both arms** — the trainer's implicit `1/num_tasks` scaling (0.5 for the pair, 1.0 for the single) is overridden by a diagnostic-only, default-off per-task weight knob, so the target's effective loss scale is matched (F8's "loss scaling" control). Precedent: DG-0005's additive, default-off composition knob (its A0 was byte-equivalent when unused).
- ⇒ Target update opportunity (count, per-step count, loss scale) and the target's estimator-variance distribution are **exactly** matched; the only difference is the presence of the B loss term. `T(A←B)` = that difference. Global batch differs (`n_A` vs `n_A+n_B`), which under AdamW (no batch-size-dependent LR) enters only through the B gradient's noise — the independent variable — and is stated as a pre-registered residual assumption.
- **Reverse cells refit** under the same seed/fold schedule (removes DG-0001's single-reference flaw).

### H.2 Arms (screen, one seed)
Decisive ER cells + null control:
`ER-alone(matched)`, `KS-alone(matched)`, `SI-alone(matched)`, `ER+KS`, `KS+ER`, `ER+SI`, `SI+ER`, `KS+SI`, `SI+KS` (7–9 arms; single-task references shared where their matched `n_A` coincides, otherwise one per cell). ER-target cells use 10-fold LOSO.

### H.3 Embedded measurement A
In each arm, at the late-phase checkpoint (final epoch) — and optionally at the 2/3 boundary — run:
1. one full-pool gradient pass per task ⇒ `‖E g_t‖`;
2. `K ≥ 32` iid batches of size `n_t` from the pool at the same checkpoint ⇒ `Var(g_t)` and the inflation curve;
3. `cos(E g_A, E g_B)` and `cos(E g_A, E g_B)` at matched `n`;
4. report alongside the recorded `E‖g‖`.

### H.4 Endpoints
- Primary: per-cell `T(A←B)` with a paired 95% t interval over seeds/folds, sign counts, and target-standardized comparison of `T(ER←SI)` vs `T(SI←ER)`.
- Secondary: `‖E g_t‖` per arm, `Var(g_t)`, inflation factor, mean-gradient cosines, pair selectivity (ER-pairs vs KS/SI).
- ER claims only from LOSO folds; ordinary-split ER is screening context (F3).

---

# I. Reuse plan

| Measurement | Class |
|---|---|
| Reconstruct Stage-D exposure mismatch from committed JSONs/configs; read exact per-task counts; confirm checkpoint non-retention | **analysis-only, 0 GPU** |
| `‖E g‖`, `Var(g)`, mean-gradient cosines | **short GPU** (embedded in B's runs; ≈1 epoch-equivalent per arm) or, if run standalone at 30 epochs, **short GPU diagnostic** (10 runs ≈6 GPU-h) |
| `T(A←B)` exact-matched | **full training** (short regime: 5-epoch LOSO folds) |
| Confirmation seeds 0–4 + LOSO | **full training ×5** |
| Checkpoints from DG-0002/0005 | **not reusable** |

---

# J. Success / falsification criteria (fixed before execution)

| Result | Verdict |
|---|---|
| Every ER cell's residual CI inside ±0.20 pp under exact matching | **H1 supported** → exposure artifact; directed work not interpretable |
| Residual present, but `‖E g_ER‖` ratio ≈1 across arms, inflation ≫1 for ER, mean-gradient cosines ≈0, and a noise/scale-matched aux reproduces it | **H2 supported** → estimator/scale artifact |
| Residual survives exact matching with DG-0001 sign and sign consistency, but the H2 test is inconclusive | **H3 remains plausible** (not yet supported) |
| Residual survives exact matching **and** is not reproduced by the H2 control, **and** mean-gradient cosine conflicts or semantic-aux ≠ permuted-aux in the residual's direction | **H3 positively supported** |
| No cell exceeds the band under exact matching, or every cell is explained by H1/H2 | **stop investigating directed relations** |
| H3 positively supported for one pair, sign stable | **a directed-method study is scientifically justified** (D1 if positive, D2 if negative) |

Thresholds fixed: materiality band **0.20 pp** (`VARIANT_BENCHMARK_PROTOCOL.md` §4); screening band 0.010 (DG-0001's own). ER target ⇒ LOSO.

# K. Conditions for progressing to directed-relation methods

- **D1 (non-negative / loss-scaled directed B, AMTL-like):** justified only by a **positive** exposure-matched one-way benefit (some `T(A←B) > +0.20 pp`, reverse absent/adverse, replicated, ER LOSO). The current sign pattern (SI↔ER negative both ways) is **not representable** by non-negative B ⇒ **D1 unjustified by the present evidence.**
- **D2 (signed / no-loss-scaling directed R, AutoTR-like):** justified only if (a) a signed residual survives exact matching and the H2 control; (b) sign stable across seeds/folds; (c) pair-selective; (d) it is framed as **representing** direction, not predicting it, with the object→behaviour sign-agreement test pre-registered (§8.4 of `ASYMMETRIC_RELATIONS.md`). Because signed R does **not predict the sign**, an H3-supported negative residual is necessary but not sufficient — the object must also track the residual.
- Any directed implementation remains **project-original** under DEC-0014, with the mandatory labelling.

# L. Conditions for stopping the line

- All residuals inside the band under exact matching (H1), **or**
- the surviving residual is reproduced by the estimator-variance / scale control (H2), **or**
- no pair shows sign-stable, pair-selective structure across ≥5 seeds / 10 folds.
Then record the negative evidence (FL-xxxx) and close the directed-relation question as "no genuine asymmetric transfer at this representation."

---

# M. Estimated compute / run burden (estimates; protocol timings from DG-0002/DG-0005)

| Stage | Runs | GPU-hours |
|---|---|---|
| Stage 0 (reconstruction, counts, threshold fixing) | 0 | **0** |
| Stage 1 screen, 1 seed, 7–9 arms (5-epoch LOSO for ER cells; matched fixed-epoch for KS/SI) | 7–9 | **≈2–3** (≤2 concurrent ⇒ wall ≈1.5 h) |
| Stage 2 confirmation, seeds 0–4 + ER LOSO | ×5 | **≈12–15** |
| Optional DEC-0012 30-epoch pool-mean add-on (A0/A1, seeds 0–4) | 10 (30-epoch) | **≈6** |
| Relative burden | | Stage 0 ≪ Stage 1 < Stage 2; the optional add-on is independent |

No compute provisioned or authorized.

# N. Proposed Study classification

- **New `DG-*` diagnostic study** (not a confirmation of DG-0001 — new exact-matching independent variable, multi-seed, pre-registered thresholds — and not of DG-0005 — different object and regime).
- **Proposed ID:** the next free diagnostic number is **`DG-0008`** (`STUDIES.jsonl` holds DG-0001, DG-0002, DG-0005, DG-0007; DG-0003/0004/0006 are backlog Qs, DG-0007 is registered/blocked). Allocation follows the registry convention (append to `STUDIES.jsonl` + `studies/DG-0008/`); **the ID must be allocated by the registration machinery, not assumed** — this document proposes, it does not allocate.
- Evidence tier: Stage 1 = screening only; a claim requires Stage 2 (F1) and LOSO for ER (F3).

# O. Decisions that require human approval

1. Whether to open this diagnostic at all (it reopens the deferred diagnostic line; DEC-0013 deferred, did not cancel, diagnostics).
2. **Compute authorization** (`authorizations/DG-0008.yaml`) and the worker/volume question — the shared 200 GB volume was destroyed (2026-09-30); the raw-corpus prerequisite must be re-declared.
3. Registration of `DG-0008` in `STUDIES.jsonl` (allocation convention).
4. The diagnostic-only **explicit per-task loss-weight knob** (a protocol-touching, default-off addition needed to match loss scaling) and the exact-matching definition.
5. The **materiality threshold applied to a negative directional residual** (the 0.20 pp band is written as a regression threshold; applying it to a signed residual needs signoff).
6. Whether the optional DEC-0012 30-epoch pool-mean add-on is in scope now or deferred.
7. Whether a negative H3 result may motivate D2 given signed R represents but does not predict sign (scope question).

# P. Repository artifacts created/updated IF approved

- `studies/DG-0008/{PLAN.md, NOTE.md, configs/*.yml, run_*.py, analyze*.py, analysis.md, result.json, confirmation_result.json}`
- `STUDIES.jsonl` — one appended `DG-0008` entry (ISO 8601 dates, commit SHAs).
- `authorizations/DG-0008.yaml` — human-granted envelope only.
- Additive, default-off code: a diagnostic sampler/weight hook and a `‖E g‖`/variance mode in `improvements/gradient_diagnostics.py` (or a new `improvements/pool_mean_gradient.py`), plus tests under `research/tests/`.
- Post-result only: `FINDINGS.md` (F11/refinement), `FAILURES.md` (if negative), `DECISIONS.md`, `FRAMEWORK.md` §7, `STATE.md`, weekly report.

---

**Stopping here as instructed: design only.** No experiment executed, no compute provisioned or authorized, no research state (`DECISIONS`/`FINDINGS`/`BACKLOG`/`STUDIES.jsonl`) written, no INC-V2-5 work. The single most important design caveat to carry into review: **the exact-matched design cannot hold global batch and target share simultaneously when adding an auxiliary task** — it holds *target* opportunity exactly (count, per-step count, loss scale, steps), which is the scientifically correct quantity, and states the global-batch difference as the independent variable under AdamW.