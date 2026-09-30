# TR-0013 screen — analysis

**Classification: `REJECTED`** (the pre-registered rule, applied unchanged after the runs).
Stage: `screen`, seed 42, one run per arm, protocol checkpoint `epoch` (the protocol's primary).
`lambda_2` **validation-selected over the paper's two smallest published grid values `{0.01, 0.1}`**
(TR-0013's registered policy); all 25 wavLM layers; `smp` 0.5; 30 epochs; batch 2048; AdamW
lr 0.0025. Machine-readable result and the mechanism trajectories: `result.json`; the runs' own
in-run artifacts: `mechanism/`.

## 1. Result and classification

| arm | `test_epoch_acc_all` | ks | si | er |
|---|---|---|---|---|
| **p-MSSL `lambda_2 = 0.01`** (selected representative) | **0.965727** | 0.986540 | 0.959884 | 0.795660 |
| p-MSSL `lambda_2 = 0.1` | 0.965727 | 0.987125 | 0.960005 | 0.786618 |
| classical MTRL (in-category control) | 0.974679 | 0.985955 | 0.977700 | 0.790235 |
| matched wavCSE baseline (control) | 0.975190 | 0.986101 | 0.978427 | 0.792043 |

Paired deltas of the selected candidate (percentage points, protocol checkpoint):

| vs | all | ks | si | er |
|---|---|---|---|---|
| classical MTRL | **−0.8952** | +0.0585 | **−1.7816** | +0.5425 |
| matched wavCSE baseline | **−0.9464** | +0.0439 | **−1.8543** | +0.3617 |

The pre-registered rule requires the selected candidate to exceed **both** controls in the aggregate
with no per-task regression beyond `0.20pp`. It exceeds neither and SI regresses by ~1.8pp, so the
screen is **`REJECTED`** — the same verdict and the same SI regression as TR-0007, now with the
`lambda_2` axis selected the way the source paper's own procedure would select it.

### The validation selection is indiscriminate

`val_acc_all` at the protocol checkpoint: `0.963311` (`0.01`) vs `0.964013` (`0.1`) — a spread of
**0.0702pp**, inside the pre-registered `0.20pp` tie band. The registered tie rule therefore selects
`lambda_2 = 0.01` as the representative and records the selection as **indiscriminate**. That is the
pre-declared *third outcome* on the selection side: **the paper's own cross-validation signal cannot
distinguish the two published penalty values at this representation and scale**, and both give the
same aggregate accuracy to six decimals.

## 2. What `lambda_2` did change — and what it did not

All measurements below are the runs' own per-epoch records (`mechanism/omega_history.json`,
`mechanism/coupling_scale.json`) and the collected checkpoints; recipes and derivations in `result.json`.

### 2.1 The relation object: support changes, magnitude does not (materially)

| `lambda_2` | support (edges of 3) | Ω trace (final) | partial correlations (ks·si, ks·er, si·er) | coupling value | ADMM relative gap |
|---|---|---|---|---|---|
| 0.01 | **3** (dense, no exact zeros) | 113 437.9 | +0.4911, +0.4911, +0.4911 | 2 014.4 | −0.0535 (cap) |
| 0.1 | **1** (both ER edges exactly zero) | 127 864.4 | +0.8997, −0.0, −0.0 | 4 107.3 | −0.00011 (converged) |

So `lambda_2` **materially alters the support** — two of the three conditional edges vanish at
`0.1`, exactly the "2 of 3 edges gone" the pre-registration predicted from a healthy Gram — and it
leaves the precision's magnitude in the same order (`1.1e5 → 1.3e5`, coupling `2e3 → 4e3`). It also
materially improves the Ω subproblem's own conditioning: the `0.01` solve sits at the ADMM's
2000-iteration cap (relative duality gap `−0.054`, as TR-0007 recorded), while `0.1` converges
(`−0.00011`).

### 2.2 Representation geometry: collapse persists at both values

Summary cosines at the `best` checkpoint (all four arms, same measurement):

| arm | cosines (ks·si, ks·er, si·er) | summary Gram eigenvalues | raw row norms |
|---|---|---|---|
| p-MSSL `0.01` | **+0.999893, +0.999906, +0.999904** | 0.0, 0.0, 2.9997 (rank-1) | 1.918, 1.830, 2.399 |
| p-MSSL `0.1` | **+0.999892, +0.000014, +0.000014** | 0.0, 0.0276, 1.9998 | 1.894, 1.803, **0.00002** |
| classical MTRL | +0.998679, +0.999327, +0.998713 | 0.0, 0.0, 2.9967 (rank-1) | 0.283, 0.103, 0.312 |
| matched baseline | +0.190186, +0.048268, +0.036224 | 0.807, 0.982, 1.206 | 0.279, 0.061, 0.306 |

Both coupled arms collapse; the uncoupled baseline ends near-orthogonal. At `0.1` the collapse is
**pair-selective**: ks and si fuse (cosine 0.9999) while the ER summary row collapses to norm
`2e-5` — the ER head's summary effectively vanishes, and the ER-involving edges are exactly the two
the Ω fit drops. At `0.01` all three fuse, replicating TR-0007's recorded object almost exactly
(trace 113 437.9 vs its 113 438, partials +0.4911 vs its +0.4911) on different hardware — a
reproduction of that mechanism, not a new one.

### 2.3 Gradient scales: the value/step distinction the previous diagnosis blurred

Measured on the first training batch of every epoch, over the same parameter set the optimizer
sees (all trainable parameters and the classifier heads), by `torch.autograd.grad` (no `.grad`
writes; probe proven read-only by `research/tests/test_mssl_mechanism_probe.py`):

| `lambda_2` | coupling value | ‖∂coupling/∂θ_heads‖ | ‖∂L_task/∂θ_heads‖ | ratio trajectory | epochs with ratio > 10 |
|---|---|---|---|---|---|
| 0.01 | ≈2 014 (frozen) | 0.11 → 2.8 | ≈2.6 → 3.0 | 437 (ep 4), 17 (ep 5), then **0.73 → 0.015** | 4 and 5 only |
| 0.1 | ≈4 107 (frozen) | 1.1e3 → 1e10 | ≈2.3 → 2.9 | rises to **8.3e9** (ep 24), ends **8.4e7** | 26 of 28 |

Two facts follow, and both refine the earlier reading:

1. **The coupling's *value* is large at both values** (≈2–4e3 against a batch-mean task loss ≈0.5),
   so "the relation penalty outweighs the task loss" survives the axis unchanged.
2. **Its *gradient* does not dominate at `lambda_2 = 0.01`**: measured like-for-like, the relation
   gradient is *smaller* than the task gradient from epoch 6 on (ratio 0.015 at the end = 60×
   smaller), and only exceeds it transiently at epochs 4–5. TR-0007's "coupling gradient ≈1e5 vs task
   gradients ≈1e-2" compared an analytic W-space gradient (`2λ₀ΩW`, ~`O(d)` by construction) against a
   task-gradient figure of a different provenance and parameter set; the like-for-like measurement
   does not reproduce a permanent 1e5× domination at `λ₂ = 0.01`. At `λ₂ = 0.1` the ratio instead
   explodes (up to 8e9), because the ER diagonal of Ω stays enormous while the ER summary row
   vanishes (`2e-5`), so the coupling's gradient is dominated by that one coordinate.
3. Both arms plateau anyway (**flat from epoch 11 / 12 for 20 / 19 epochs**, tail spread 0.014pp),
   with identical aggregate accuracy — so neither a task-dominated (0.01) nor a relation-dominated
   (0.1) gradient regime produced a better run. Under AdamW the per-parameter step is
   scale-normalised, so a gradient-magnitude argument alone cannot explain the outcome; the
   *directions* the coupling adds still can, and that is what a scale-commensurate test (TR-0012)
   would isolate.

## 3. Reconciling the pre-registered prediction and its falsifiers

| Pre-registered prediction (H-scale) | Record |
|---|---|
| at `λ₂ = 0.1` the summary cosines still exceed `0.9` within two epochs of the coupling engaging | **Held**: cosines reach 0.9999 for ks·si; the ER pairs go to ~0 |
| the coupling value stays `O(2e3)` per task | **Held**: 4 107 total ≈ 1 369 per task at `0.1`; 2 014 ≈ 671 at `0.01` |
| training still plateaus | **Held**: flat from epoch 12 (`0.1`) / 11 (`0.01`) to epoch 30 |
| the support shrinks relative to `0.01` (predicted 2 of 3 edges gone on a healthy Gram) | **Held**: support 3 → 1, i.e. exactly 2 of 3 edges gone |
| the outcome stays `REJECTED` with the same SI regression | **Held**: −1.78pp vs the in-category control at both values |
| **Falsifier**: a run whose coupling-to-task gradient ratio stays below `10` through training | **Not met**: ratio exceeds 10 at epochs 4–5 (`0.01`) and at 26 of 28 epochs (`0.1`) |
| **Falsifier**: cosines staying below `0.5` | **Not met**: 0.9999 |
| **Falsifier**: a validation curve without a plateau | **Not met**: both plateau |
| **Falsifier**: an outcome that separates from both controls | **Not met**: both values sit ~0.90pp below the in-category control |
| **Falsification of H-λ**: both values reproduce the collapse and the plateau with the same support | **Partly met**: collapse and plateau reproduce at both values, but the *support differs* (3 vs 1) — so H-λ's own falsifier is not literally satisfied either |

**Verdict on the hypothesis.** H-scale (the synthesis's prediction) **survived**: the `λ₂` axis buys
support and Ω conditioning, not a resolvable run; the representation collapse, the gradient regime
and the non-beneficial outcome all persist. H-λ (the faithful explanation that a data-selected `λ₂`
would rescue the arm) is **not supported**: the validation signal cannot discriminate the two
published values, and the value it would have picked delivers the same regression as the
researcher-fixed one. The relation object *is* different (support, partial correlations, Ω
conditioning), so the axis is not inert — it is **outcome-irrelevant at this representation and
scale**, which is a sharper statement than the draft's "inert" alternative.

## 4. What this screen does and does not establish

* It establishes a **screening-level negative result** (single seed, single split) for the faithful
  published p-MSSL arm under the paper's own `λ₂` selection procedure: neither published value beats
  either control; both regress SI by ~1.8pp.
* It **closes TR-0007's `λ₂` caveat**: that the arm failed only because `λ₂` was researcher-fixed is
  now measured to be false at this representation — selecting `λ₂` on validation cannot help when the
  validation differences (0.07pp) are far inside the noise the arm's own task differences show.
* It does **not** establish anything about ER as a result: the ordinary split is speaker-leaky (F3);
  the ER differences here (+0.36 to +0.54pp vs the controls, and 0.9pp between the two `λ₂` values)
  are single-split context, well inside the documented ~±1.3pp single-split ER noise.
* It does **not** test the scale-convention question. The coupling's *value* scale is unchanged along
  this axis, and the measured gradient regime differs between the values; a commensurate-imposition
  test (TR-0012) remains a deviation-class question the researcher has not authorized.
* It does not promote anything and authorizes no confirmation, LOSO or ablation.

## 5. Defects found and repaired (recorded, not smoothed over)

1. **`support_edges` / `zero_offdiagonals` were wrong in the recorded runs.** The in-run count was
   taken over the output of `torch.triu`, which returns the *whole* matrix with its lower part
   zeroed, so the zeroed region was counted too: a 3-task snapshot with two absent edges recorded
   `zero_offdiagonals = 8` and `support_edges = −5`. The **raw Ω matrix was always recorded
   correctly**, so the support in this analysis is recomputed from it (`result.json` carries both the
   recorded and the recomputed values). Fixed in `04-mssl/mssl_trainer.py` after the screen with a
   regression test (`test_mssl_mechanism_probe.py::test_support_counts_only_the_strict_upper_edges`).
2. **Job-launch failures (6 pre-science attempts).** Of the twelve job submissions, six failed before
   any training step, all at job start and none producing science, an MLflow run or an output: one
   wrong raw-dataset link (IEMOCAP), three container-disk exhaustion (a stale unreferenced job
   workspace coexisting with a new one on a 100 GB disk), and two "process no longer running, no
   outcome" failures whose only correlate was a large `rm -rf` of a predecessor's workspace while the
   next job started. Each was repaired at the environment level (link; scratch wiped; sweeping made
   idle-only), and each re-run used the identical registered configuration. `logs/` holds the
   preserved evidence; the ARC audit trail has a `pre-science-failure-repaired` event per attempt.
   **Cost of the lesson: ~2 h of the paid worker time, recorded here rather than hidden.**

## 6. Execution record

* One worker, `al5b4m763ci2bu`, `NVIDIA L4`, `$0.49/h`, EU-RO-1, the existing 200 GB network volume;
  all four arms sequentially on it; container disk 100 GB (± the envelope's 120 GB ceiling).
  **Destroyed** at the end (`finish` → `destroy`), so billing stopped when the science did. The
  volume that the cache and the raw-corpus farm lived on was **deprovisioned the same day** on the
  researcher's instruction; nothing canonical was lost, and the rebuild recipe — raw tree, counts,
  disk rules, GPU filter, cost shape — is kept in `../../WORKER_ENVIRONMENT.md`.
* Runs (DagsHub MLflow, experiment per arm as the naming convention requires):
  `mssl-l2-0p01` `03e9b2d26bdf4691b455bb324a11a82c`, `mssl-l2-0p1` `237f38800a05409da185bcd0454652a7`,
  `classical-mtrl` `ff9f3b138d2c4452acc5dcefcc72f2ac`, `wavcse-baseline` `09f78efc2d2940d4a0e028804702f7a7`.
* All four ran at the registered commit `f2d746a86f7cd191b6e1608c9ff1a8629dded25b`; every run carries
  `study_id = TR-0013`, `stage`, `arm`, `method`, `seed`, `lambda_2`, `lambda_2_selection`, the exact
  git SHA, `layers = all`, `layer_count = 25`, the GPU type and the protocol fields.
* **All-25-layer proof, per run:** the arms' configs resolve `all` → `[0…24]` via
  `parse_transformer_layers`, enforced by `test_tr0013_protocol.py`; the run log of the first job
  prints `Selected transformer layers: [0, 1, …, 24]`, and every run's MLflow tags carry
  `layers=all`, `layer_count=25`.
* Paid wall clock and cost: see the ledger (`improvements.compute status --scope TR-0013`); the
  screen closed well inside its envelope (0.80 USD/h, 3.00 USD, 4 h, one worker).
* Original artifacts of every run were collected and digest-verified by the backend's evidence
  validator before this analysis read them; the two candidate runs' in-run mechanism artifacts are
  copied into `mechanism/` with their SHA-256 recorded in `result.json`.
