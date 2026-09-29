# TR-0007 screen — analysis

**Classification: `REJECTED`** (pre-registered rule, applied unchanged after the runs).
Stage: `screen`, seed 42, one run per arm, protocol checkpoint `epoch` (the protocol's primary).
lambda_2 = 0.01 **researcher-fixed** (DEC-0016); lambda_0 = 1.0; all 25 wavLM layers; `smp` 0.5;
30 epochs; batch 2048; AdamW lr 0.0025. Machine-readable result: `screen_result.json`.

## Result

| arm | commit | `test_epoch_acc_all` | ks | si | er | evidence |
|---|---|---|---|---|---|---|
| **p-MSSL** (candidate) | `d13e82e` | **0.9662** | 0.9867 | 0.9615 | 0.7848 | collected, verified |
| classical MTRL (in-category control) | `664c572` | 0.9752 | 0.9855 | 0.9794 | 0.7848 | collected, verified |
| matched wavCSE baseline (control) | `664c572` | 0.9748 | 0.9861 | 0.9782 | 0.7848 | collected, verified |

Paired deltas (candidate − control, percentage points) at the protocol checkpoint:

| vs | all | ks | si | er |
|---|---|---|---|---|
| classical MTRL | **−0.90** | +0.12 | **−1.79** | 0.00 |
| matched baseline | **−0.86** | +0.06 | **−1.67** | 0.00 |

The pre-registered rule required the candidate to exceed **both** controls in the aggregate
with no per-task regression beyond `0.20pp`. It exceeds neither, and SI regresses by ~1.7pp, so
the screen is `REJECTED`. Nothing was re-tuned, no threshold was moved.

Reproducibility: the candidate arm was executed twice (see *Execution record*); the two
executions produced identical checkpoint bytes and metrics to five decimal places
(`0.966238` both times), so the outcome is not a run-to-run artefact.

## Why it failed — the mechanism, measured

The Ω step is live but **mis-scaled for this representation at λ₂ = 0.01**, exactly the caveat
`PLAN.md` pre-registered before any run. Evidence from the candidate run's per-epoch relation
artefact (`omega_history.json`, MLflow) and the implied task-row alignment `C ≈ d·Ω⁻¹`
(unit-norm rows give `S = C/d`, so the unpenalised precision is `d·C⁻¹`):

| epoch | Ω trace | implied cosines (ks·si, ks·er, si·er) | mean abs off-diagonal | relative duality gap |
|---|---|---|---|---|
| 3 (before the coupling) | 6 308 | +0.26, +0.04, +0.01 | 211 | −4.3e−05 |
| 4 – 30 (after) | 113 438 (frozen) | **+0.981, +0.981, +0.981** | 18 571 | −0.0535 |

1. As soon as the coupling term engages (warmup ends at epoch 3), the three task summaries
   align to cos ≈ 0.981.
2. Aligned unit-norm rows make `C` nearly singular, so Ω jumps ~20× to a nearly uniform
   precision — the same *uninformative* object classical MTRL saturates into (F7), reached
   differently.
3. The coupling gradient `2·lambda_0·Ω·W` is then `~1e5` against task gradients of `~1e-2`, so
   the regularizer, not the tasks, drives the update: the reported training loss is dominated by
   the coupling term (`≈2015` against `≈0.5` of task losses) and validation plateaus from
   epoch ≈10 (`val_acc_all` 0.9609 for the last five epochs). The learning-rate scheduler, fed
   by the coupling-dominated validation loss, decays to its floor.
4. The Ω snapshots also show the ADMM's own limit: from epoch 4 the relative duality gap stays
   at `−0.054`, i.e. the 2000-iteration cap was reached at that conditioning. The solver reports
   this per snapshot rather than hiding it; it does not change the verdict (the arm was already
   regularizer-dominated), but it is a limit of the current solver settings at this scale.

This is *not* a solver or protocol defect: the Ω step is verified against closed forms and the
problem's own optimality certificate (`test_mssl_omega_solver.py`), and every frozen protocol
field was checked (`test_tr0007_protocol.py`). It is the published estimator behaving as
published: Eq. (8) is not scale free, and the paper selects λ₂ on data precisely because its
magnitude is representation-dependent. With λ₂ = 0.01 fixed by the researcher, the arm is
`REJECTED` **at that fixed λ₂** — the screen cannot say that p-MSSL "cannot work", only that it
does not work here, at this scale, under this protocol.

## What this screen does and does not establish

* It establishes a **screening-level negative result** (single seed, single split): the
  faithful p-MSSL implementation, with the shared mean-head summary adapter and
  λ₂ = 0.01, does not beat either control on this protocol.
* It does **not** establish anything about ER as a result: the ordinary split is speaker-leaky
  (F3), and all three arms are within 0.004pp of each other on ER (0.7848). Any ER claim needs
  LOSO.
* It does **not** promote p-MSSL, and it authorizes no confirmation run (confirmation is a
  separate human grant).
* It leaves the open question it exposed, for a future decision rather than an autonomous one:
  whether a scale-calibrated λ₂ (validation-selected, or expressed relative to `d`) would make
  the mechanism informative, and whether the coupling term needs a magnitude control
  independent of Ω's scale. Both are protocol-level questions, not something to tune after
  seeing this screen.

## Execution record (infrastructure, recorded because it shaped the run)

* Raw-dataset provisioning (a gap ARC does not close): the loader's dataset classes read the
  *raw* voice corpora at `paths.root_data_path` (`~/voice_dataset/<dataset>`), which no study
  plan declares — the plan materialises embeddings only — while the network volume mounts at
  `/workspace/cache`. Before submitting, the worker's raw tree was verified and the loader's
  paths were pointed at the cache with the project's own workstation convention, a symlink per
  dataset (`~/voice_dataset/speechcommand/SpeechCommands/speech_commands_v0.01`,
  `~/voice_dataset/voxceleb/{wav,iden_split.txt}`, `~/voice_dataset/iemocap`), verified by
  file counts (64,727 / 153,516 / 10,039 wavs). No scientific parameter changed; a resumed or
  transferred attempt needs the same step.
* One worker (`8jy5oamc…`, `NVIDIA RTX A4500`, `$0.25/h`, EU-RO-1 with the pre-existing
  network volume), reused for all three arms; **paid wall clock 2.96 h, cost `$0.7404`**, well
  inside the screen envelope (0.80/h, 5.00 total, 6 h).
* Three jobs failed *before* producing science and were re-run under the bounded retry policy:
  two job-preparation failures caused by the worker's 60 GB container disk filling during
  checkout/transfer (the container disk, not the NFS cache volume, holds each job's ~45 GB
  scratch), and one duplicate candidate execution whose `MANIFEST.json` could not be persisted
  because ARC's output keys are per `(study, arm, seed)` and the first attempt's object already
  existed. The duplicate's metrics are retained as corroboration; it is excluded from the
  classification.
* The two control arms therefore ran at commit `664c572` and the candidate's collected run at
  `d13e82e`. The diff between those commits is **control-plane only** (job-preparation failure
  classification in `improvements/compute/failures.py` and its test); no scientific
  configuration or training code differs. `screen_commits.json` records this per arm, and the
  analysis binds each arm to the commit its evidence came from instead of to one SHA.
* The four historical `TR-0007__screen__*` / `scale-corrected` runs in
  `taskrelation-variant-benchmark` (commits `10aaaea3…`, `3df542d…`, seed 0) were **not**
  touched, relabelled or substituted; they remain historical evidence only.

---

# 2026-09-29 — Post-hoc scale reading of the *same* artifacts (appended; verdict unchanged)

**Why this is appended rather than folded in.** The screen's classification, its numbers and its
caveats above are untouched: this screen ran at the researcher-fixed `lambda_2 = 0.01` and it is
`REJECTED` at that value. What follows is a *derived reading* of artifacts the screen already
published — the three arms' checkpoints and the candidate's `omega_history.json` — taken
afterwards, on the same evidence tier (one seed, one split, screening). It adds no run, no
compute and no metric, and it does not reopen the λ question (`DEC-0016`).

Script and machine-readable output: `analyze_coupling_scale.py`, `coupling_scale_result.json`
(read-only, deterministic; it fetches the arms' `checkpoints/train_ks_si_er_best.pth` from the
tracking store and records each artifact's SHA-256).

## 1. What a checkpoint can be asked

`W` — the arm's `[3, 2001]` class-mean summary — is a *pure function* of the classifier heads
(`[mean(weight, dim=0); mean(bias)]` per task, unit-normalised because `normalize_w` is on in both
relation arms), so it can be reconstructed from any stored checkpoint. `R = W Wᵀ` is then the
summary Gram whose off-diagonals are the pairwise cosines of the three task summaries, and the
checkpoints also carry the arm's final `omega`.

| arm | summary cosines (ks·si, ks·er, si·er) | eigenvalues of `R` | condition |
|---|---|---|---|
| **p-MSSL** (candidate) | **+0.99989, +0.99991, +0.99990** | −0.0, 0.0, 2.99970 | rank-1 (numerically) |
| **classical MTRL** (control) | +0.99822, +0.99932, +0.99827 | −0.0, 0.0, 2.99581 | rank-1 (numerically) |
| **matched wavCSE baseline** (control) | **+0.21067, +0.03061, +0.02339** | 0.78725, 0.99273, 1.21547 | 1.54 |

`OBSERVED` (this reading). Both *coupled* arms end with numerically rank-one summaries; the
*uncoupled* baseline ends with the summaries nearly orthogonal. The screen's own record reports the
candidate's *pre-activation* alignment as cosines +0.261, +0.035, +0.009 at epoch 3 — the same
order as the baseline's final geometry. So the collapse is not a property of the class-mean adapter
or of these three tasks' heads: it is **produced by the coupling term, within one epoch of it
engaging**.

The candidate's stored `omega` reproduces the screen's recorded mechanism from the artifact side:
trace `113 437.98` (the record's `1.13e5`), off-diagonal `−18 570.4` against diagonal `37 812.7`
(the record's mean `|off-diagonal|` `18 571`), **zero exact off-diagonal zeros** (dense support),
partial correlations `+0.4911, +0.4911, +0.4911` — the three identical, and at the `m = 3`
equicorrelation ceiling `ρ = r/(1+r) → 1/2` that `literature_survey/MSSL_SPARSITY_ANALYSIS.md`
§3.2 derives. `DERIVED`: `Ω = S⁻¹` with `S = R/d` is what makes the two readings one object, so a
rank-one `R` and a trace-`1.1e5`, equicorrelated `Ω` are two views of the same degenerate geometry.

## 2. Why the coupling dominates — and why λ cannot fix it

The trainer adds the published coupling `lambda_0 * tr(W Ω Wᵀ)` to the **batch-mean** task loss.
The coupling's value is `tr(Ω R)`, which at the unpenalized optimum (`Ω = S⁻¹`, `S = R/d`) is
exactly `3d` and which the penalized fits below keep at `≈ 5993` for a well-conditioned `R`
(measured at the epoch-3 geometry). So at the published `lambda_0 = 1` the term is `O(d)` — the
recorded `≈ 2015` is this quantity at the collapsed geometry, where the `ℓ₁` holds the fit below the
unpenalized optimum — against `≈ 0.5` of task losses. `DERIVED`: the coupling's **gradient**
`2 λ₀ Ω W` is `O(d)` as well — `≈ 6.9e3` even for `Ω = d·I`, i.e. ~`7e5`× the recorded
task-gradient scale (`~1e-2`).

Both published axes were then swept with this repository's own `graphical_lasso_admm` at each
measured Gram (exact Eq. (8) solves, not runs):

| `lambda_2` | support (exact zeros, of 3) | `tr Ω` | coupling value | coupling gradient | partial correlations |
|---|---|---|---|---|---|
| 0.01 (screen) | 0 | 113 438 | 2 014.5 | 159 481 | +0.491, +0.491, +0.491 |
| 0.1 | 0 | 40 626 | 2 143.9 | 56 462 | +0.474, +0.474, +0.474 |
| 1 | 3 | 6 004 | 6 003.6 | 6 932 | 0, 0, 0 |
| 10, 100 | 3 | 6 004 | 6 003.6 | 6 932 | 0, 0, 0 |

(Late geometry; the uncoupled baseline's Gram gives 0 zeros / gradient `7 240` at `0.01`, 2 zeros /
`7 029` at `0.1`, 3 zeros / `6 942` at `≥1`; the epoch-3 recorded geometry gives 1 / 0 / 3 zeros
and `7 407 / 7 117 / 6 932`.) `DERIVED` from those solves: **every value of the published `λ₂` grid
leaves the coupling term of order `d`** — the values that retain any conditional edge are the
loudest, and the value that removes all three still carries `tr(Ω R) = 3d` on its diagonal. A
`lambda_0` sweep behaves the same way: at `lambda_0 = 1e-4` the coupling gradient is still `21.7`
(≈`2e3`× the recorded task scale) and **all three edges are already zero**, because lowering
`lambda_0` raises the ℓ₁'s relative weight instead of fixing the scale.

So `λ₂` moves the *support*; it does not move the coupling's magnitude. The scale is set by the
objective's normalisation — the penalty is a sum over `d = 2001` summary coordinates added to a
batch-mean loss — not by a hyperparameter. This refines, and does not contradict, the mechanism
section above: that section is right that the arm is coupling-dominated and right that `Eq. (8)`
is not scale free; this reading adds that the fixed `λ₂` is not the quantity that could have
rescued it.

## 3. What this does not change, and what it does not establish

* The verdict stands: `REJECTED` at `lambda_2 = 0.01`, one seed, one split, `FL-0005` unchanged.
* One seed, one split, `criterion = epoch`. The collapse and the `Ω` values are single-run
  observations; the axis table is arithmetic at those observed Grams, not a re-run.
* The epoch-3 cosines in the axis table are the **record's** implied cosines, re-used as an input
  Gram; they are not a re-measurement of epoch 3 (no epoch-3 checkpoint exists).
* It does not establish that p-MSSL cannot help. It establishes that this arm, at every
  hyperparameter the paper publishes, is dominated by its own regularizer, so the screen's negative
  result is not yet readable as a statement about the relation object.
* Any successor that changes the coupling's *scale* changes the objective's normalisation, which is
  a **declared deviation** — a `HUMAN_DECISION` class question this file does not answer. Proposals:
  `../../proposals/TR-0012_scale_commensurate_coupling.md` and
  `../../proposals/TR-0013_published_lambda2_axis.md`; synthesis:
  `literature_survey/POST_TR0007_SYNTHESIS.md`.
