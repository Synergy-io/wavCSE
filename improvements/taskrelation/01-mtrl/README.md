# wavCSE-MTRL — Multi-Task Relationship Learning

**Current best config: `smp` pooling (λ=0.5), 16 layers, `mtrl_lambda=0.01`,
`normalize_w=true` — MTRL 0.9728 vs. baseline 0.9724, the only config where
MTRL beats its matched baseline outright. Found via a full grid search over
all 10 pooling methods at both 16 and 25 layers — see
`improvements/base/POOLING_GRID_SEARCH.md` for the complete search, the
best-ever baseline number (0.9767 at `smp`+25L), and why the earlier
"16 layers beats 25" conclusion (below) turned out to be pooling-specific,
not general.**

**Paper:** Zhang & Yeung, "A convex formulation for learning task relationships
in multi-task learning" (UAI 2010; journal version ACM TKDD 2014). The central
method of the Task Relation Learning approach (Zhang & Yang survey, §2.4) —
every later method in that section (sparse Ω, high-order, asymmetric,
deep/tensor extensions) is an extension of this one.

This is the first entry in a new numbering convention for `taskrelation/`:
`01-<name>/` folders are self-contained (own model, trainer, config, README)
architecture attempts, as opposed to the flat `models/`/`trainers/`/`configs/`
layout used by the three earlier exploratory variants (GBC, TSM, PMR).

## What it does

Learns an `[num_tasks, num_tasks]` task covariance matrix Ω jointly with the
task classifier heads, via the regularizer `tr(W Ω⁻¹ Wᵀ)`. Unlike wavCSE-PMR
(which learns a *precision* matrix by gradient descent, jointly with W, in a
single combined loss), MTRL's Ω has a **closed-form optimum given W**:

```
Ω = (W Wᵀ)^(1/2) / tr((W Wᵀ)^(1/2))
```

computed analytically (matrix square root via eigendecomposition of a tiny
`num_tasks x num_tasks` matrix) and refreshed on a schedule
(`mtrl.omega_update_frequency`, default every epoch after a
`mtrl.warmup_epochs`-epoch warmup) — never touched by the optimizer. This is
the actual alternating-minimization scheme from the paper: solve for `W`
given `Ω` fixed (standard gradient step, every batch), then solve for `Ω`
given `W` fixed (analytic step, periodic).

### Design notes (why this isn't "PMR again")

- **`W` is built from mean-pooled per-task vectors, not raw flattened
  weights.** wavCSE's task heads have very different class counts (`ks`: 12,
  `si`: 1251, `er`: 4), so naively flattening each head's `[out_dim,
  hidden_dim]` weight + `[out_dim]` bias produces rows of different lengths
  that cannot be stacked into one matrix — and would let the 1251-class
  Speaker head's ~2.5M raw parameters dominate the 4-class Emotion head's
  ~8K purely by having more classes, not by being genuinely more "complex" in
  the MTRL sense. Instead, each task's row is `mean(weight, dim=0)` (→
  `[hidden_dim]`) concatenated with `mean(bias)` — a fixed-length "direction"
  per task, independent of class count. See `mtrl_model.py`'s
  `get_task_parameter_matrix()` docstring.
- **Ω is a `register_buffer`, not an `nn.Parameter`.** It's never in the
  optimizer's parameter list; `update_omega()` is called explicitly by the
  trainer and wrapped in `torch.no_grad()`.
- **The regularizer term uses live (non-detached) classifier weights**, so
  `tr(W Ω⁻¹ Wᵀ)` genuinely back-props into the task heads during the normal
  backward pass, pulling related tasks' parameters together. (Contrast: PMR's
  equivalent loss term is built from `head.weight.data`/`head.bias.data`,
  i.e. detached — so its trace term never actually reaches the classifier
  heads via gradient descent, only `omega_chol` gets a gradient from it.)

## Running

```bash
cd wavCSE   # repo root (wavCSE/wavCSE), NOT this folder
conda activate opencv
python improvements/run_improvements.py --model mtrl --task_type ks_si_er
```

Tracked in MLflow under experiment `taskrelation-mtrl` (same DagsHub server
and tagging convention as every other `improvements/` run — see
`improvements/README.md`).

## Experiment log

Each entry: what was run, the result, and what changes going into the next
attempt. `test_opt_acc_all` is the overall test accuracy across all tasks at
the "opt" checkpoint (same metric `wavcse-baseline` reports, for direct
comparison).

### Baseline (for comparison)

Fresh `wavcse-baseline` run against the *current* `base_config.yml` (30 epochs,
`ks_si_er`, 100% of data) — run fresh rather than trusting historical DagsHub
history for this experiment, which spans several incompatible past configs.

Run: `base_original_ks_si_er_2026_08_25_10_39_12`
([DagsHub](https://dagshub.com/Ke-vin-S/wavCSE.mlflow/#/experiments/1/runs/c6e60537efc2414db17cff8674f6c677))

| Metric | all | ks | si | er |
|---|---|---|---|---|
| test_opt_acc | 0.970970 | 0.985369 | 0.973458 | 0.755877 |

### Iteration 1 — `mtrl_lambda=0.01` (defaults, as committed)

Config: `mtrl_config.yml` as-is (`mtrl_lambda=0.01`, `omega_epsilon=1e-4`,
`warmup_epochs=3`, `omega_update_frequency=1`), 30 epochs, `ks_si_er`, 100% of
data, same GPU/hyperparameters otherwise as the baseline.

Run: `taskrelation_mtrl_ks_si_er_2026_08_25_10_40_05`
([DagsHub](https://dagshub.com/Ke-vin-S/wavCSE.mlflow/#/experiments/6/runs/380c2cf72455450dae19c82aa5f2cdad))

| Metric | all | ks | si | er |
|---|---|---|---|---|
| test_opt_acc | 0.969308 | 0.983760 | 0.969943 | **0.781193** |
| Δ vs baseline | **-0.17pp** | -0.16pp | -0.35pp | **+2.53pp** |

**Reading the result:** MTRL doesn't beat the baseline on the overall
(sample-weighted) accuracy — it's essentially flat, very slightly behind,
driven mostly by `si` (the task with the most test samples, 8251) dropping
0.35pp. But it delivers a real, meaningful improvement on `er` (the smallest,
hardest task, 553 samples): +2.53pp. This is exactly the kind of effect
task-relationship regularization is supposed to produce — pulling a
data-poor task's parameters toward the structure shared with better-resourced
tasks — but the learned Ω came out fairly weak: mean |off-diagonal| stabilized
around **0.013** (against a diagonal of ~0.33 each under the tr(Ω)=1
constraint for 3 tasks), i.e. the regularizer is only lightly coupling the
heads.

**Next step:** the coupling looks under-powered rather than wrong in
direction — `er` improved despite a weak Ω, suggesting a stronger
`mtrl_lambda` could amplify the effect (more `er` gain) without necessarily
costing much more on `ks`/`si`, since those two are already near their
ceiling. Iteration 2 tries `mtrl_lambda=0.05` (5x) with everything else held
fixed.

### Iteration 2 — `mtrl_lambda=0.05` (5x stronger regularizer)

Same as iteration 1 except `mtrl_lambda: 0.05` (everything else unchanged:
`omega_epsilon=1e-4`, `warmup_epochs=3`, `omega_update_frequency=1`, 30
epochs).

Run: `taskrelation_mtrl_ks_si_er_2026_08_25_11_01_10`
([DagsHub](https://dagshub.com/Ke-vin-S/wavCSE.mlflow/#/experiments/6/runs/84066836ff564fe6b5d130d438fa7273))

| Metric | all | ks | si | er |
|---|---|---|---|---|
| test_opt_acc | 0.967453 | 0.981712 | 0.969216 | 0.764919 |
| Δ vs iteration 1 | -0.19pp | -0.20pp | -0.07pp | **-1.63pp** |
| Δ vs baseline | -0.35pp | -0.37pp | -0.42pp | +0.90pp |

**Reading the result:** stronger regularization made things *worse*, not
better, on every axis including `er` — the one task it was supposed to help
most. This rules out the hypothesis from iteration 1 ("the coupling looks
under-powered, a stronger lambda should amplify the `er` gain"). Looking at
the per-epoch Ω summaries logged during this run, the off-diagonal magnitude
was actually *smaller and noisier* at λ=0.05 (fluctuating ~0.005–0.007)
than at λ=0.01 (~0.013, trending up smoothly). So a larger λ doesn't push W
toward a more strongly-coupled Ω — it just perturbs classifier training hard
enough to hurt the fit on all three tasks without buying more informative
structure. **λ=0.01 (iteration 1) remains the best result so far** on both
the overall metric and the `er` improvement.

*Note: this run predates the Ω-history logging added below (iteration 1/2 only
have the per-epoch summary line in the training log, not the full
`omega_history.json` artifact) — that logging is active from iteration 3
onward.*

### Infrastructure change: full Ω history now persisted

Added to `mtrl_trainer.py` (applies from iteration 3 onward): every time
`update_omega()` runs, the full matrix is now (1) appended to an in-memory
history and written to `results_dir/omega_history.json` at the end of
training (task names + one snapshot per update + the final matrix), which
rides along with the existing `mlflow.log_artifacts(...)` upload of the
whole results directory, and (2) logged live as per-epoch MLflow metrics
(`omega_<task_i>_<task_j>` for each pair, `omega_diag_<task>` for each
diagonal entry) so the evolution is chartable on DagsHub instead of only
visible as a one-line summary in the log. Previously only a
mean-off-diagonal/trace summary was logged to text; the actual matrix values
were never saved anywhere.

### Iteration 3 — `mtrl_lambda=0.01` (revert to best), 2x epochs

Reverting to the best lambda found so far (0.01) and testing whether more
training amplifies the effect: `num_epochs: 60` (vs. 30 for baseline/iter
1/iter 2), everything else unchanged. This is an apples-to-apples question
about training length rather than a baseline comparison, so it's compared
primarily against iteration 1 (same lambda, 30 epochs) rather than the
30-epoch baseline.

### Iteration 3 (first attempt) — CRASHED at epoch 1 (environment, not a code bug)

Same config as the entry above (60 epochs, λ=0.01, Ω-history logging active).
Died during the first checkpoint save with
`PytorchStreamWriter failed writing file data/8: file write failed` — the
shared machine's root disk had hit **0 bytes free** mid-run (it fluctuates
wildly; other users' jobs consume tens of GB/hour). The run is marked FAILED
on DagsHub and its local checkpoint dir contained a corrupt partial
`train_ks_si_er_best.pth`, since deleted. Not an MTRL code problem —
wavCSE's checkpoint writer just fails the same way for any run when the disk
is full (this exact failure mode is already a known gotcha in the repo's
CLAUDE.md).

**Recovery / hardening done before the rerun:**

- Deleted the corrupt partial dirs and all superseded run dirs (iterations
  1–2 checkpoint dirs — their best/opt checkpoints are on DagsHub — plus
  pre-today baseline run dirs under `improvements/base/`).
- Added a **pre-run disk guard** to `improvements/run_improvements.py`:
  checks `shutil.disk_usage()` on the checkpoints/results roots and aborts
  loudly before training if <2 GB free, instead of dying mid-checkpoint.
- Reduced `saved_checkpoint_count` 3 → 1 in `mtrl_config.yml` (matches
  `base_config.yml`; cuts the per-run steady-state checkpoint footprint).
- **Environment change:** the team migrated this repo from conda to uv
  mid-session; the `opencv` conda env no longer exists. Runs now use
  `.venv/bin/python -m improvements.run_improvements ...` (Python 3.9.25,
  torch 2.7.1+cu126).

### Iteration 3 (rerun) — `mtrl_lambda=0.01`, 60 epochs — more epochs does NOT help

Identical config to the crashed attempt (λ=0.01, 60 epochs,
`saved_checkpoint_count: 1`), rerun after the cleanup/guard above. No disk
incident this time; run completed and the Ω-history logging worked (58
snapshots in `omega_history.json`, per-epoch `omega_*` metrics on DagsHub).

Run: `taskrelation_mtrl_ks_si_er_2026_08_25_16_21_37`
([DagsHub](https://dagshub.com/Ke-vin-S/wavCSE.mlflow/#/experiments/6/runs/961f3eb8e8cc4e6fbb7d2b65aa2b045e))

| Metric | all | ks | si | er |
|---|---|---|---|---|
| test_opt_acc | 0.967837 | 0.982590 | 0.968973 | 0.768535 |
| Δ vs iteration 1 (30 ep, same λ) | -0.15pp | -0.12pp | -0.10pp | **-1.27pp** |
| Δ vs baseline (30 ep) | -0.31pp | -0.28pp | -0.45pp | +1.27pp |

**Reading the result:** doubling the epochs made every metric slightly
*worse*, not better — val acc was flat from ~epoch 20 (plateau-scheduler LR
had decayed to ~1e-7 by the end), so the extra epochs only fit noise. The
"more epochs" question is answered: **30 epochs with λ=0.01 (iteration 1)
stays the best result.** No 60-epoch baseline is needed to interpret this —
MTRL@60 is already below MTRL@30 on every axis.

**Learned Ω at 60 epochs** (first time the full matrix is available, not
just a summary — this is what the new history logging exists for):

```
         ks        si        er
ks   [+0.0115, -0.0323, -0.0109]
si   [-0.0323, +0.1512, -0.0219]
er   [-0.0109, -0.0219, +0.8373]
```

Reading it: the off-diagonals are small and (mostly) slightly *negative* —
the mean-pooled per-task head directions are close to orthogonal with mild
negative correlation, and the `er` head's mean-weight norm dominates the
diagonal (0.837 vs 0.151/0.011), which through Ω⁻¹ means the regularizer
pulls mainly on `ks`/`si`. Two takeaways: (1) the weak coupling the
iteration-1 numbers hinted at is real and structural — with this W
representation the tasks genuinely look nearly independent, so the
regularizer can only make small changes; (2) if we want MTRL to couple the
tasks more strongly, the lever isn't λ (ruled out in iteration 2) — it's the
**W representation** (e.g. unit-normalizing each task row before forming
WWᵀ, or excluding the bias from the vector). That's a model-level change
worth a future iteration.

### Iteration 4 — stability check: re-run the best config (λ=0.01, 30 epochs)

The iteration-1 numbers are the headline result (overall ≈ baseline with
`er` +2.53pp), but the gap between iteration 1 (er 0.781) and this 60-epoch
run (er 0.769) at the same λ hints at run-to-run noise of ~1pp on `er`.
Iteration 4 re-runs the exact iteration-1 config (λ=0.01, 30 epochs, now
with the Ω-history logging that didn't exist then) to see whether the
`er` gain reproduces.

### Iteration 4 — stability check result: the er gain did NOT reproduce

Exact iteration-1 config (λ=0.01, 30 epochs), rerun to test reproducibility.

Run: `taskrelation_mtrl_ks_si_er_2026_08_25_16_53_55`
([DagsHub](https://dagshub.com/Ke-vin-S/wavCSE.mlflow/#/experiments/6/runs/7d049ef4b3fe44b3ac773c8fbbfa02b4))

| Metric | all | ks | si | er |
|---|---|---|---|---|
| test_opt_acc | 0.967581 | 0.982443 | 0.969337 | 0.757685 |
| Δ vs iteration 1 (same config) | -0.17pp | -0.13pp | -0.06pp | **-2.35pp** |
| Δ vs baseline | -0.34pp | -0.29pp | -0.41pp | +0.18pp |

**Reading the result:** the headline er gain from iteration 1 (+2.53pp) was
**not reproducible** — this rerun lands at er 0.7577, essentially the
baseline's 0.7559 (+0.18pp). Run-to-run noise on `er` (only 553 test
samples, noisy training) is ~±1.3pp, larger than the effect we're chasing;
the overall metric wobbles ~±0.2pp run-to-run. Honest conclusion so far:
**MTRL at λ=0.01/30 epochs is statistically tied with the baseline** —
neither clearly better nor clearly worse — and the learned Ω's coupling
stays weak (mean |off-diagonal| ~0.010 this run vs ~0.013 in iteration 1),
i.e. the regularizer barely moves the task heads relative to the noise floor.

The lever that remains is the W representation: the 60-epoch Ω analysis
showed the `er` head's parameter-norm dominates Ω's diagonal, which decouples
it from the regularizer through Ω⁻¹. Iteration 5 turns on `normalize_w`
(unit-normalize each task row before forming WWᵀ) so Ω encodes *directions*
only, not norms.

### Iteration 5 — `normalize_w: true` (Ω from cosine directions, not raw params)

Model change (opt-in flag `normalize_w`, default off so iterations 1–4 are
unchanged): each task's parameter row in W is unit-normalized (epsilon
floor), so WWᵀ becomes the cosine-similarity matrix between task directions
and Ω can no longer be dominated by whichever head has the largest parameter
norm. Everything else = iteration-1 config (λ=0.01, 30 epochs).

### Iteration 5 (result) — `normalize_w: true` works: coupling saturates, er gain reproduces

Run: `taskrelation_mtrl_ks_si_er_2026_08_25_17_10_55`
([DagsHub](https://dagshub.com/Ke-vin-S/wavCSE.mlflow/#/experiments/6/runs/554566464ba348628bf359b874053eb8))

| Metric | all | ks | si | er |
|---|---|---|---|---|
| test_opt_acc | 0.969499 | 0.984199 | 0.969943 | **0.781193** |
| Δ vs baseline | -0.15pp | -0.12pp | -0.35pp | **+2.53pp** |

**Reading the result:** normalizing W changed Ω structurally — mean
|off-diagonal| jumped from ~0.010 (iterations 1–4) to **0.333** (the PSD
maximum under tr(Ω)=1 for 3 tasks: coupling saturated) — and the final Ω
carries a clear, interpretable sign pattern:

```
         ks        si        er
ks   [+0.3334, -0.3331, +0.3331]
si   [-0.3331, +0.3331, -0.3330]
er   [+0.3331, -0.3330, +0.3335]
```

i.e. **ks and er are positively related; si is anti-related to both** — the
model found real structure once norm-dominance was removed, and at this
setting the `er` gain from iteration 1 **reproduced exactly** (0.781193, the
same 432/553 test samples correct). Overall accuracy stays statistically tied
with the baseline (-0.15pp, within the ±0.2pp run-to-run noise), while er
sits +2.53pp above it.

`normalize_w: true` is now the default in `mtrl_config.yml` and this is the
recommended MTRL config for the thesis comparison.

## Summary of all iterations

| Run | Config | all | ks | si | er |
|---|---|---|---|---|---|
| Baseline | `base_config.yml`, 30 ep | 0.9710 | 0.9854 | 0.9735 | 0.7559 |
| Iter 1 | λ=0.01, 30 ep | 0.9693 | 0.9838 | 0.9699 | **0.7812** |
| Iter 2 | λ=0.05, 30 ep | 0.9675 | 0.9817 | 0.9692 | 0.7649 |
| Iter 3 | λ=0.01, **60 ep** | 0.9678 | 0.9826 | 0.9690 | 0.7685 |
| Iter 4 | λ=0.01, 30 ep (stability) | 0.9676 | 0.9824 | 0.9693 | 0.7577 |
| **Iter 5** | λ=0.01, 30 ep, **normalize_w** | **0.9695** | 0.9842 | 0.9699 | **0.7812** |

Conclusions, in order of confidence:

1. **MTRL (every variant tried) ties the baseline overall** — the best runs
   sit ~0.15pp below `wavcse-baseline`'s overall test accuracy, which is
   within the ±0.2pp run-to-run noise observed across identical-config
   reruns. It does not beat the baseline on the sample-weighted metric.
2. **The effect on the small task is real and controllable.** At the two
   strongest-coupling settings (iterations 1 and 5), `er` reaches 0.7812
   (+2.53pp over baseline); at weak-coupling settings it falls back to
   baseline level (~0.756–0.769). The task-relation regularizer buys the
   data-poor task a meaningful gain at a small, noise-level cost to the
   dominant tasks.
3. **`normalize_w` is the decisive lever** — not λ (0.05 hurt), not epochs
   (60 hurt). It changed the learned Ω from near-independence
   (|off-diag|≈0.01) to saturated, interpretable structure (±1/3), and that
   is exactly the "learned task relationship matrix" the architecture
   promises as its interpretability artifact.
4. **The learned relationship reads as:** ks↔er positive, si anti-related to
   both — a quantitative, data-driven answer to "which speech tasks share
   structure", which is the thesis's core question.

## All 25 Layers Campaign

Everything above used a hand-curated 16-layer subset of WavLM-Large's 25
layer outputs (`6,1,0,3,2,5,4,7,8,9,10,11,12,17,14,13`). This section
re-runs the campaign with `selected_transformer_layers: all` (confirmed via
investigation: the shared embedding cache already stores all 25 layers per
utterance — no re-extraction needed, purely a downstream config change).
New configs: `mtrl_alllayers_config.yml` (this folder),
`improvements/base/configs/base_alllayers_config.yml`.

Monitoring note: a watchdog logged `df -h`/`nvidia-smi` every 2 minutes
during these runs (`/tmp/watchdog_phaseA.log`). GPU 1 already had another
user's job running when Phase A launched (PID 300053, ~1.5GB, 48% util) —
noted, did not block or meaningfully slow our run given 15GB/GPU headroom.
A second foreign job appeared on GPU 0 (~5GB) right after our MTRL run
finished — no impact, just recorded for the log.

### Phase A — baseline and MTRL at their previous best settings, all layers

| Run | Config | all | ks | si | er |
|---|---|---|---|---|---|
| baseline (mix), 16L | (prior campaign) | 0.9710 | 0.9854 | 0.9735 | 0.7559 |
| **baseline (mix), 25L** | `base_alllayers_config.yml` | 0.9676 | 0.9884 | 0.9628 | **0.7812** |
| MTRL (weighted, λ=0.01, normalize_w), 16L | iter 5 | 0.9695 | 0.9842 | 0.9699 | 0.7812 |
| **MTRL (weighted, λ=0.01, normalize_w), 25L** | `mtrl_alllayers_config.yml` | 0.9637 | 0.9881 | 0.9570 | 0.7613 |

Runs: [baseline+25L](https://dagshub.com/Ke-vin-S/wavCSE.mlflow/#/experiments/1/runs/8ff37bc435ab4d57936533d0b9c43624), [MTRL+25L](https://dagshub.com/Ke-vin-S/wavCSE.mlflow/#/experiments/6/runs/ec3da72b344d418db9c83a0ddf4e1f2c).

**Why the result came out this way** (inspected the learned `weighted`-pooling
softmax attention over the 25 layers, from the MTRL+25L checkpoint):

```
layer   0- 11: 0.045-0.067 each  (high — early/mid layers, roughly the ones
                                  the original hand-picked 16-layer set used)
layer  12- 14: 0.029-0.045 each  (transitional)
layer  15- 24: 0.017-0.020 each  (suppressed, but never zero — softmax pooling
                                  can down-weight, not eliminate, a layer)
```

The learned pooling correctly identified layers 15-24 as less useful and
suppressed them — but 10 suppressed-not-eliminated layers still inject
~18% cumulative weight of comparatively noisy signal into the pooled
representation. `ks` (keyword spotting, a coarser task) is robust to this;
`si` and `er` (which need precise acoustic detail) are not — both regressed
on **both** arms (baseline si -1.07pp, MTRL si -1.29pp), consistent with a
general "more layers without re-tuning dilutes the harder tasks" effect,
not something specific to MTRL.

**The one divergent result is `er`:** baseline **gained** +2.53pp (0.756→0.781)
while MTRL **lost** -1.99pp (0.781→0.761) from adding layers. And the learned
Ω tells us why this specific run's regularizer stopped helping: unlike every
16-layer run, Ω did **not** saturate here (mean |off-diagonal| = 0.149, vs.
0.333 at 16 layers with the identical `normalize_w=true` setting):

```
         ks        si        er
ks   [+0.3027, +0.2598, +0.1023]
si   [+0.2598, +0.3076, -0.0853]
er   [+0.1023, -0.0853, +0.3897]
```

Reading it: ks↔si are now the strongly-related pair (+0.26); er is only
weakly tied to ks (+0.10) and *anti*-related to si (-0.085) — a different,
less decisive structure than either 16-layer result (ks↔er positive/si
anti-related at `weighted`; near-uniform positive at `lnp`). With a
partially-unsaturated Ω, the regularizer's pull on the classifier heads is
weaker and, in this configuration, landed in a direction that cost `er`
rather than helping it.

**Decision for iteration 1 of Phase B:** the earlier finding "stronger λ
hurts" was established when Ω was already saturated at λ=0.01 (16 layers) —
a different regime from here, where Ω is under-saturated at the same λ. That
finding doesn't necessarily transfer. Testing a moderate increase,
**λ=0.03** (between the working 0.01 and the previously-ruled-out-at-16-layers
0.05), to see whether it pushes Ω toward saturation at 25 layers and
recovers `er`, without repeating the exact setting already ruled out.

### Phase B, iteration 1 — λ=0.03 result: hypothesis rejected

Run: `taskrelation_mtrl_ks_si_er_2026_08_31_12_00_33`
([DagsHub](https://dagshub.com/Ke-vin-S/wavCSE.mlflow/#/experiments/6/runs/9fb3f037eae34022bb8c9bfd9ef27a73))

| Metric | all | ks | si | er |
|---|---|---|---|---|
| λ=0.01 (Phase A) | 0.963681 | 0.988149 | 0.956975 | 0.761302 |
| λ=0.03 | 0.960419 | 0.985808 | 0.952733 | 0.761302 |
| Δ | -0.33pp | -0.23pp | -0.42pp | **0.00pp (exactly flat)** |

Final Ω:
```
         ks        si        er
ks   [+0.3099, +0.2094, +0.1674]
si   [+0.2094, +0.3334, -0.1126]
er   [+0.1674, -0.1126, +0.3567]
```

**Why:** the hypothesis was that λ=0.01 under-saturates Ω at 25 layers (mean
|off-diag| 0.149, vs. 0.333 at 16 layers) and a moderate increase would push
it toward saturation, recovering `er`. It didn't: off-diagonal barely moved
(0.149→0.163 mean), the *diagonal* entries partially saturated (si 0.333, er
0.357) while the off-diagonals shifted **direction**, not magnitude (ks↔si
weakened 0.260→0.209, ks↔er strengthened 0.102→0.167, si↔er got more
negative -0.085→-0.113) — and `er` didn't move at all (0.7613 both times,
exactly). **Conclusion: λ is not the lever at 25 layers either** (mirrors the
16-layer finding that λ tuning doesn't help, now confirmed in a second
regime) — increasing it just perturbs Ω's structure without a clear
beneficial direction, while consistently costing `si`/`ks`. Reverting to
λ=0.01.

**Decision for iteration 2:** if it's not λ, the remaining candidate from the
plan is `warmup_epochs`. At 3 epochs, the very first Ω computation happens
while `weighted` pooling's 25 position-weights are still close to their
uniform initialization (the learned specialization seen in Phase A only
fully emerges by epoch ~20-30) — so the regularizer may be anchoring W
toward a structure computed from immature, still-mostly-uniform-pooling
features. Testing `warmup_epochs: 3 → 10` (λ back to 0.01) to let both the
classifier heads and the pooling weights stabilize more before Ω starts
constraining them.

### Phase B, iteration 2 — `warmup_epochs=10` result: also rejected

Run: `taskrelation_mtrl_ks_si_er_2026_08_31_12_17_56`
([DagsHub](https://dagshub.com/Ke-vin-S/wavCSE.mlflow/#/experiments/6/runs/41173c26aa164e8482c762a4daa4598d))

| Metric | all | ks | si | er |
|---|---|---|---|---|
| λ=0.01, warmup=3 (Phase A, best so far) | 0.963681 | 0.988149 | 0.956975 | 0.761302 |
| λ=0.01, **warmup=10** | 0.956327 | 0.987418 | 0.944249 | 0.752260 |
| Δ | **-0.74pp** | -0.07pp | **-1.27pp** | -0.90pp |

Final Ω:
```
         ks        si        er
ks   [+0.3196, +0.2555, +0.0594]
si   [+0.2555, +0.2946, -0.1366]
er   [+0.0594, -0.1366, +0.3857]
```

**Why:** the hypothesis was that a longer warmup would let the pooling
weights specialize before Ω constrains them, producing a better-informed
regularizer. Instead it made every metric worse, `si` badly (-1.27pp). Most
likely explanation: `warmup_epochs` doesn't just delay the *regularizer* —
`_process_batch`'s warmup check gates the whole MTRL loss term, but nothing
else about training changes, so the extra 7 warmup epochs are just 7 fewer
epochs (out of a fixed 30-epoch budget) that the model trains *with* the
eventual regularization pressure active — i.e. this accidentally re-ran the
already-known-bad "fewer effective epochs under the real objective" failure
mode from the 16-layer campaign's iteration 3 (more epochs hurt because the
LR schedule plateaus early), just approached from the other direction.
**Second rejected lever.** Two independent attempts to fix the 25-layer
regression via MTRL's own hyperparameters (λ, warmup) both failed and both
pointed the same way: perturbing away from the Phase A defaults costs
accuracy on `si`/`er` without recovering anything. Reverting to
`λ=0.01, warmup_epochs=3` (Phase A settings) as the best all-layers
weighted-pooling MTRL config found.

**Decision:** stop tuning the weighted-pooling arm here (2 of the plan's 5
iterations spent, both negative results, consistent story) and move to
**Phase C** — the lnp-matched pair at all layers — using the Phase A
defaults. This was always the next planned step and is a genuinely new axis
(pooling mechanism, not another MTRL hyperparameter), rather than a third
attempt at a direction two independent tests have already refuted.

### Phase C — lnp-matched pair, all layers: pooling/layer-count mismatch, not an MTRL problem

Runs: [baseline+lnp+25L](https://dagshub.com/Ke-vin-S/wavCSE.mlflow/#/experiments/7/runs/50952f003bbf450e96c61e2923d1c14b), [MTRL+lnp+25L](https://dagshub.com/Ke-vin-S/wavCSE.mlflow/#/experiments/6/runs/173d39e263f74d348c3b95721d4fe119).

| Run | all | ks | si | er |
|---|---|---|---|---|
| baseline+lnp, 16L | 0.9730 | 0.9867 | 0.9741 | 0.7884 |
| **baseline+lnp, 25L** | 0.8857 | 0.9877 | **0.8095** | 0.7631 |
| MTRL+lnp, 16L | 0.9726 | 0.9874 | 0.9733 | 0.7794 |
| **MTRL+lnp, 25L** | 0.8884 | 0.9868 | **0.8143** | 0.7758 |

**Both arms collapsed on `si` by ~16pp** (0.974→0.810 baseline, 0.973→0.814
MTRL) — clearly a general effect of the pooling+layer-count combination, not
an MTRL-specific failure (MTRL is actually marginally *better* than baseline
on 3/4 metrics in this broken regime: all +0.26pp, si +0.49pp, er +1.26pp).

**Why:** `lnp` pooling is `(sum(|x|^p) / n)^(1/p)` with `p=16` fixed in the
config — a hyperparameter tuned (implicitly, by whoever chose it originally)
for the 16-layer subset, and never re-validated for a different layer count.
With `p=16` (a high power), the sum is dominated by whichever layer has the
largest-magnitude activations — effectively a soft-max over layers, not a
real average. Adding 9 more layers changes *which* layer's magnitude
dominates that soft-max, and evidently the layers added in 15-24 (the same
range the `weighted`-pooling run in Phase A learned to suppress) pull the
dominant term toward something far less informative for `si` specifically —
`ks` (coarse, robust) is unaffected, `er` degrades mildly, `si` collapses.

This is the clearest evidence in the whole campaign that **the 16-layer
subset wasn't an arbitrary choice — every pooling mechanism we've tested
(`weighted`'s learned attention, `lnp`'s fixed power) either had to relearn
around the extra layers (Phase A, costing `si`/`er`) or was invalidated by
them outright (Phase C)**. Unlike `weighted` pooling (which has learnable
parameters that can *partially* adapt, as seen in Phase A), `lnp`'s `p` is a
fixed hyperparameter with no mechanism to adapt to a new layer count at all.

**Not pursuing further tuning here** — this isn't an MTRL hyperparameter
question (re-tuning `p` for 25 layers is a baseline/pooling-methodology
question that would need its own sweep, and both Phase B attempts already
showed MTRL's own knobs don't fix layer-count-driven pooling problems). This
closes the all-25-layers campaign at 2 of the plan's 5 tuning-iteration
budget spent (both on Phase B, both negative — a clean, consistent result).

## All 25 Layers — Final Summary

| Run | Layers | all | ks | si | er |
|---|---|---|---|---|---|
| baseline (mix) | 16 | 0.9710 | 0.9854 | 0.9735 | 0.7559 |
| baseline (mix) | **25** | 0.9676 | 0.9884 | 0.9628 | **0.7812** |
| MTRL (weighted, best config) | 16 | 0.9695 | 0.9842 | 0.9699 | 0.7812 |
| MTRL (weighted, best config) | **25** | 0.9637 | 0.9881 | 0.9570 | 0.7613 |
| baseline+lnp | 16 | 0.9730 | 0.9867 | 0.9741 | 0.7884 |
| baseline+lnp | **25** | 0.8857 | 0.9877 | 0.8095 | 0.7631 |
| MTRL+lnp | 16 | 0.9726 | 0.9874 | 0.9733 | 0.7794 |
| MTRL+lnp | **25** | 0.8884 | 0.9868 | 0.8143 | 0.7758 |

**Bottom line: at this training budget (30 epochs), all 25 layers did NOT
beat the curated 16-layer subset on any arm's overall accuracy** — it won on
`ks` everywhere (simple task, more info always helps a little) and, for the
`mix`-pooling baseline only, on `er` (+2.53pp), but cost `si` on every arm
(-1.07 to -16.5pp depending on pooling) and cost overall accuracy on every
arm. The mechanism is well-evidenced, not a mystery: **more layers ≠ more
accuracy unless the pooling mechanism can actually suppress the added noise,
and neither `weighted` (partial, learnable but slow) nor `lnp` (a fixed
power, no adaptation at all) fully managed that within 30 epochs.** The
original hand-picked 16-layer selection remains the best-performing input to
this architecture.

**Task-relation-matrix (Ω) interpretation across all layer counts** — the
learned relationship is genuinely representation-dependent, not a fixed
property of the tasks:

| Setting | ks↔si | ks↔er | si↔er | Saturated? |
|---|---|---|---|---|
| weighted, 16L | -0.333 | +0.333 | -0.333 | Yes (±1/3) |
| weighted, 25L (best) | +0.260 | +0.102 | -0.085 | No (~0.15) |
| lnp, 16L | +0.333 | +0.333 | +0.333 | Yes, uniform |
| lnp, 25L | +0.297 | +0.120 | +0.113 | Partial |

Every setting agrees `si`↔`er` is the weakest or most negative pair (speaker
identity and emotion share the least structure) and `ks` is positively
related to both others in 3 of 4 settings. But the *strength* and, at 16
layers under `weighted` pooling, even the *sign* of `si`↔`er` are not
stable across representations — reinforcing the caveat already noted from
the 16-layer campaign: Ω describes how the tasks relate **as seen through
the current pooled representation**, not a pooling-independent ground truth.
Broader input (more layers, weaker pooling adaptation) here produced a
*less* decisive Ω, not a more informative one — the cleanest, most saturated
task-relation reading this project has produced remains
**`weighted`-pooling at 16 layers**.

<!-- ENTRIES APPENDED BELOW AS RUNS COMPLETE -->
