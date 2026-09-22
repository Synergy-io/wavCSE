# DG-0005 analysis — ER data-regime gradient-scale control

## Decision

**CONFIRMED diagnostic.** The pre-registered support condition holds: A0's late max/min task-norm ratio was ≥ 3.0 and A1's was < 3.0 in all five matched seeds, the exposure gate passed for every pair, and every run carries the same implementation commit.

This confirms a *diagnostic* about where the F9 gradient-scale signal comes from. It authorizes no Task Relation Learning mechanism: the `TR-xxxx` gate stays closed under DEC-0009.

## Protocol validity

| Check | Result |
| --- | --- |
| Same implementation commit, all ten runs | `8032a937050d8bbd3114b172cb813a8fc7370b37` |
| Exposure gate, all five seed pairs | passed |
| Optimizer steps / diagnostic samples | 2,820 / 142 in every arm |
| A1 realized ER÷KS sampled-batch count | `0.990`, `0.991`, `0.990`, `0.990`, `1.008` |
| DagsHub run note present | yes, all ten runs |

Both arms used `smp(0.5)`, all 25 layers, 30 epochs, global batch 2048, `ks_si_er`, and an identical evaluation protocol with an explicit seed. Only the training split's composition differed: A1 reweights ER by `11.5` with `num_samples` unchanged, so validation and test data are untouched and the step count is preserved (F8 satisfied by construction).

Cross-check: the A0 arms reproduce the DG-0002 matched-baseline phase summaries exactly — middle `7.243 ± 0.272`, late `8.962 ± 0.456`. An unused sampling knob therefore leaves baseline training numerically identical, as the additive/default-off requirement demands.

## Primary result

| Phase | A0 | A1 | Paired A1−A0 | 95% t interval | Seeds in pre-registered direction |
| --- | ---: | ---: | ---: | --- | ---: |
| middle | `7.243 ± 0.272` | `2.818 ± 0.272` | `−4.425` | `[−4.980, −3.869]` | 5/5 (late rule) |
| late | `8.962 ± 0.456` | `2.352 ± 0.270` | `−6.610` | `[−7.353, −5.867]` | 5/5 |

Per-seed late ratios: A0 `9.382, 9.220, 9.205, 8.730, 8.272`; A1 `2.305, 2.266, 2.250, 2.819, 2.119`.

The pre-registered rule names the late phase, and it holds in every seed. The middle phase is borderline: A1 stayed below 3.0 in `4/5` seeds and was `3.054` in the remaining one. The honest statement is "late dominance is removed in 5/5 seeds; middle is at the threshold", not "the imbalance is gone everywhere".

## Where the change came from

Late mean shared-parameter norms:

| Task | A0 | A1 | Change |
| --- | ---: | ---: | ---: |
| KS | `0.828` | `0.894` | `+0.066` |
| SI | `0.804` | `0.687` | `−0.117` |
| ER | `6.972` | `1.608` | `−5.364` |

The ratio collapse is localized to ER; KS and SI barely move. A composition that merely redistributed mass between KS and SI could not produce it. ER's own late norm falls by about 77%.

Pairwise late cosines remain near zero in both arms (A0 `+0.0022 / +0.0013 / +0.0221`; A1 `+0.0020 / +0.0019 / +0.0027`). The result is a scale change, not a conflict change, so it neither explains nor contradicts DG-0002's rejection of persistent pairwise conflict (FL-0002).

## Outcome context (no ER claim)

Fixed-final-epoch paired A1−A0 deltas: aggregate `+0.00106 [+0.00026, +0.00186]`, KS `+0.00064 [−0.00118, +0.00246]`, SI `+0.00078 [+0.00016, +0.00139]`, ER `+0.01049 [+0.00249, +0.01849]`.

These are speaker-leaky ordinary-split numbers (F3) and support no ER performance claim. The ER direction is in fact better explained by memorization than by generalization: A1 trains ER to ≈0.99 train accuracy while validation plateaus near 0.82, and ER's final train−val gap grows from `0.134` (A0) to `0.174` (A1) at the screening seed. A genuine ER benefit would require LOSO, which this diagnostic deliberately does not run.

## Interpretation

F9's reproducible ER norm dominance is **not task-intrinsic at this representation**. It tracks the training mixture: with ER contributing ~435 rather than ~47 of every 2,048 sampled examples, ER's late gradient norm drops by ~77% and both phase ratios fall below the dominance threshold in 5/5 seeds. The earlier "data scarcity, task difficulty, label noise or gradient-estimate variance" alternatives listed in F9 are therefore narrowed — the dominance is a property of how the tasks are sampled, not of the tasks themselves at this representation.

Framework consequence: **gradient scale is a training-mixture quantity**, like relation magnitude and relation confidence before it. A mechanism justified by "ER has intrinsically larger gradients" now rests on a refuted premise.

## Remaining alternative explanation

The knob changes two things at once. Raising ER's share removes gradient-estimate variance *and* multiplies ER's optimizer updates per epoch (the same ~43k ER examples are drawn ~9× more often). Part of the norm drop may therefore reflect faster ER convergence/overfitting — a smaller residual gradient on a saturated head — rather than purely reduced estimator variance. The training-accuracy evidence is consistent with that reading.

The single-knob design cannot separate these. A2 (the pre-registered reverse direction) was reserved for an *ambiguous* A1 and is not triggered, and it would not separate them either: it changes ER's share by reducing KS/SI. This is the highest-information gap left by DG-0005, and it is a candidate for a future diagnostic rather than a conclusion.

## What this does not establish

* **Not** a causal explanation of MTRL's outcome null. F9 remains correlational with respect to accuracy; DG-0005 changes where the scale signal comes from, not whether it costs accuracy.
* **Not** an ER performance effect — no LOSO, leaky split, larger overfit gap.
* **Not** a mechanism justification. Neither the confirmed result nor the remaining confound authorizes a `TR-xxxx` Study; that gate is a human decision under DEC-0009.
* **Not** a revision of F4/F8/FL-0001–0003, which are untouched.

Provenance: `confirmation_result.json` (ten runs, seeds 0–4), MLflow experiment `taskrelation-diagnostics`, stage `confirm`, commit `8032a937050d8bbd3114b172cb813a8fc7370b37`.

---

## Post-hoc: how much of the ER drop is estimator size?

**Status: post-hoc exploratory analysis, pre-registered in `analyze_noise_shape.py` at commit `2ac7f3d622610624b5609815c91a93f3a70ebb3a` (committed before any statistic was computed). It carries no `CONFIRMED` claim and creates no new finding.**

The question left open above is whether the ER norm drop is a smaller *gradient estimate* (fewer samples → noisier → `E‖g‖ ≥ ‖E g‖`) or a smaller *mean gradient* (ER head state after ~9× more updates). Two things were fixed before computing anything.

**A shape test is not usable here.** Under isotropic gradient noise, the relative dispersion of `‖g‖` over 1,550,800 shared parameters is bounded by `0.707/√d ≈ 0.00057`. The observed late-phase ER dispersion is `CV = 0.316` `[0.290, 0.342]` across the ten runs — about `560×` the ceiling. Within-phase dispersion is therefore dominated by step-to-step variation of the mean gradient, not by estimator noise, so mean/median, CV and skew cannot proxy estimator variance with this instrumentation. Shape statistics are reported in `noise_shape_result.json` for completeness and deliberately not used as a test.

**What is identifiable is a bound.** With `σ ∝ 1/√n`, a noise-dominated norm has `E‖g‖ ∝ 1/√n`, so the mean-norm ratio between arms is predicted to be `√(n_A1/n_A0)`. Any larger observed drop cannot be estimator size.

| Phase | A0 mean norm | A1 mean norm | Observed ratio | Noise-scaling prediction | Estimator share bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| late | `6.288 – 7.299` | `1.556 – 1.772` | `3.99 – 4.69` | `2.99 – 3.10` | `0.758` `[0.715, 0.801]` |
| middle | `6.929 – 8.000` | `2.272 – 2.744` | `2.85 – 3.23` | `3.02 – 3.09` | `1.028` `[0.975, 1.082]` |

Per-seed late shares: `0.797, 0.746, 0.716, 0.790, 0.739` (seeds 0–4). Sample counts moved `44.9–48.8 → 428.4–436.4`.

**Result.** In the late phase, estimator-size scaling explains **at most 76%** of the ER log-drop in every seed (95% CI `[0.72, 0.80]`), leaving **at least ~20%** (mean 24%) that requires either a smaller mean gradient — ER head state under ~9× more updates — or noise growing faster than `1/√n`. In the middle phase the estimator account is sufficient on its own (share ≈ `1.03`, CI spanning 1.0). The phase difference is the informative part: early in training the ER norm behaves like a sampling artifact, and late in training it acquires a component that sampling cannot explain.

Descriptively, and consistent with that reading, ER's late *median* falls by the same factor as its mean (`5.911–6.698 → 1.475–1.674`, ratios `3.85–4.45` against mean ratios `4.00–4.61`): the distribution is rescaled rather than reshaped.

**Independent replication.** DG-0002's matched baseline confirmation (standard composition, same seeds, commit `7f6d5248`) gives identical A0 statistics — late ER mean/median `1.080 ± 0.020`, CV `0.341 ± 0.033`, mean norm `6.972 ± 0.407`, matching this Study's A0 arm value for value. This is a further check that the sampling knob is inert when unused.

**Bound, not identification.** The estimator share lies in `[0, bound]`: a signal-dominated norm would make the share 0, so this analysis cannot prove a mean-gradient change — it only rules out "estimator size alone" and, more usefully, *locates the residual in the late phase*. It also does not separate "ER head saturates faster" from "noise scaling steeper than `1/√n`", and one-knob confounding (optimizer path, LR schedule) remains as stated above. KS/SI bounds are unstable by construction — their counts move only ~1.2×, so the ratio of two small log-changes is dominated by noise — and are not evidence either way.

Provenance: `noise_shape_result.json`; script `analyze_noise_shape.py` at `2ac7f3d`; inputs are the ten confirmation artifacts plus DG-0002's five baseline confirmation artifacts (run IDs in `STUDIES.jsonl`).
