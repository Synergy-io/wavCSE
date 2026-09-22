# DG-0002 Analysis — Exposure-controlled gradient compatibility

Status: **CONFIRMED — matched seeds 0–4; norm dominance established, persistent conflict rejected**

## Protocol and execution

Two matched 30-epoch `ks_si_er` runs used seed 42, fixed WavLM-Large embeddings, frame-mean pooling, `smp(0.5)` layer pooling, all 25 layers, full data, batch size 2048, and identical optimizer/scheduler/checkpoint/evaluation configuration. The only method difference was classical MTRL (`lambda=0.01`, normalized task-head summaries, warmup 3, Ω updated each epoch).

- baseline: MLflow `71b89be472284af9a855f46871eb9f38`
- MTRL: MLflow `12a92107a6a34f26a705061f0e373239`
- implementation/config commit: `75e31b81e860d54f6125dd4a623d85e93718b7ec`

Both runs finished. Their standard Study tags and DagsHub notes are complete. Runtime was 18.3 minutes for baseline and 19.2 minutes for MTRL.

The diagnostic sampled the first, final, and every twentieth optimizer step. Both arms produced 142 samples over 2,820 steps with no missing-task samples. At every sampled step, valid-example counts matched exactly between methods. Mean sampled batch composition was KS 539.1 (range 477–586), SI 1461.7 (1409–1515), and ER 47.2 (32–64). Shared-parameter definition and count also matched: `projector_layer` plus `hidden_layer`, 1,550,800 parameters. The exposure-pairing check therefore passed.

## Outcome context

Fixed final epoch was pre-registered as primary.

| Checkpoint | Metric | baseline | MTRL | Δ MTRL − baseline |
| --- | --- | ---: | ---: | ---: |
| epoch | aggregate | 0.975446 | 0.974551 | −0.000895 |
| epoch | KS | 0.986247 | 0.986979 | +0.000732 |
| epoch | SI | 0.979639 | 0.976609 | **−0.003030** |
| epoch | ER | 0.779385 | 0.790235 | +0.010850 |
| opt | aggregate | 0.971418 | 0.972760 | +0.001343 |
| opt | KS | 0.985516 | 0.987125 | +0.001609 |
| opt | SI | 0.972731 | 0.975276 | +0.002545 |
| opt | ER | 0.777577 | 0.757685 | −0.019892 |
| best | aggregate | 0.975446 | 0.974231 | −0.001215 |
| best | KS | 0.986247 | 0.986686 | +0.000439 |
| best | SI | 0.979639 | 0.976488 | **−0.003151** |
| best | ER | 0.779385 | 0.786618 | +0.007233 |

The fixed-epoch SI regression exceeds the project's 0.20 percentage-point materiality threshold. It is one seed and conflicts with F4's matched five-seed 25-layer mean (which did not show a stable SI regression), so it is screening evidence—not a new outcome claim. Ordinary-split ER changes sign across checkpoint policies and the split is speaker-leaky; no ER performance interpretation is admissible.

Training behavior was stable, not a crash or failed fit. Both methods were identical through the pre-MTRL warmup. Baseline validation aggregate accuracy ended at its maximum, 0.972641 at epoch 30; MTRL peaked at 0.972852 at epoch 28 and ended at 0.972361. Baseline final train/validation aggregate accuracy was 0.991414/0.972641; MTRL was 0.990827/0.972361. The shared ReduceLROnPlateau policy produced different realized learning-rate paths after the MTRL regularizer changed validation loss (final LR `7.8125e-5` baseline versus `3.90625e-5` MTRL). This is an operational consequence of the canonical method under the same scheduler policy, but any later causal ablation of the regularizer must decide whether to fix the realized LR path.

## Gradient norms

Each task norm is from its unweighted mean masked cross-entropy over the same forward pass. Equal `1/3` task-loss weighting in the actual objective preserves these norm ratios.

| Method | Phase | KS norm | SI norm | ER norm | max/min mean-norm ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | early | 4.759 | 4.682 | 12.194 | 2.605 |
| baseline | middle | 1.117 | 1.389 | 8.218 | **7.357** |
| baseline | late | 0.864 | 0.831 | 6.363 | **7.661** |
| MTRL | early | 4.631 | 4.013 | 10.897 | 2.715 |
| MTRL | middle | 1.148 | 1.123 | 7.576 | **6.746** |
| MTRL | late | 0.862 | 0.651 | 6.038 | **9.276** |

The pre-registered norm-dominance threshold (ratio at least 3 in at least two training thirds) is met in the middle and late thirds. Baseline ER gradients remain about 7.4–7.7 times the smallest task gradient after the unstable early transient. MTRL does not remove this imbalance. It reduces the late SI norm more than ER, increasing the late ratio to 9.28.

The early all-run means contain very large first-step transients (KS max 127.5, SI 61.5, ER 69.0), but the middle/late result does not depend on those outliers. Median late norms tell the same story: baseline KS/SI/ER 0.800/0.857/6.384; MTRL 0.842/0.654/5.920.

## Pairwise gradient compatibility

Entries are `mean cosine / negative-conflict frequency` over sampled steps.

| Method | Phase | KS↔SI | KS↔ER | SI↔ER |
| --- | --- | ---: | ---: | ---: |
| baseline | early | +0.051 / 0.208 | +0.043 / 0.333 | +0.136 / 0.104 |
| baseline | middle | +0.010 / 0.362 | +0.034 / 0.298 | +0.054 / 0.170 |
| baseline | late | +0.005 / 0.447 | +0.003 / 0.468 | +0.025 / 0.319 |
| MTRL | early | +0.056 / 0.188 | +0.044 / 0.250 | +0.125 / 0.188 |
| MTRL | middle | +0.010 / 0.383 | +0.013 / 0.426 | +0.058 / 0.234 |
| MTRL | late | +0.005 / 0.489 | −0.008 / 0.532 | +0.005 / 0.532 |

No pair meets the pre-registered persistent-conflict criterion (mean at most −0.05 and conflict frequency at least 0.60 in two thirds). No pair meets the material dynamic sign-change criterion. The pairwise-conflict branch of the hypothesis is therefore weakened for seed 42: baseline gradients begin weakly compatible and approach near-orthogonality, not persistent opposition.

MTRL does not improve compatibility. Mean-cosine changes are small; negative-conflict frequency increases for KS↔ER in the middle third by 0.128 and for SI↔ER in the late third by 0.213. Because those late means remain near zero, this is not evidence of strong negative transfer by itself. It is evidence against claiming that MTRL mitigated optimization conflict.

## Ω correspondence

MTRL Ω evolves from differentiated weak positive entries to uniform saturation:

| Epoch | KS↔SI | KS↔ER | SI↔ER |
| ---: | ---: | ---: | ---: |
| 3 | 0.0384 | 0.0223 | 0.0070 |
| 5 | 0.2854 | 0.0586 | 0.0732 |
| 10 | 0.3288 | 0.3178 | 0.3187 |
| 15 | 0.3321 | 0.3310 | 0.3312 |
| 30 | 0.33304 | 0.33311 | 0.33307 |

The final off-diagonal range is only `6.95e-5`. Ω has lost pair discrimination by roughly epoch 10 while gradient cosines remain pair- and phase-dependent and ER gradient norms remain much larger than KS/SI. This extends F7 observationally: saturated Ω coexists with a strong shared-gradient scale imbalance and does not correct it.

Ω and cosine have the same positive sign when averaged over the full run, but that coarse sign agreement is uninformative: all Ω edges are effectively identical, while cosine magnitudes and conflict frequencies differ and drift toward zero. Ω is a head-parameter covariance, not a task-weighting device; the result does not prove Ω is mathematically wrong. It shows that classical MTRL as implemented does not represent or regulate the dominant optimization-scale behavior measured here.

## Hypothesis and alternatives

### Supported at screening level

The norm-dominance branch of the hypothesis: ER produces persistently larger shared-parameter gradients in the middle and late thirds, and MTRL neither represents that scale asymmetry in Ω nor mitigates it. The fixed-epoch outcome is consistent with no aggregate improvement and material SI harm at this seed.

### Weakened

Persistent pairwise gradient conflict as the explanation for MTRL's null result. No baseline pair crosses the pre-registered conflict threshold; late interactions are near-orthogonal rather than strongly negative.

### Remaining alternatives

- ER has only about 47 examples per mixed batch versus 539 KS and 1462 SI. Its larger, noisier gradient may arise from sample scarcity, task difficulty, or label noise rather than task semantics. DG-0005-style data-regime controls become relevant only after the norm pattern survives seeds.
- Uniform Ω saturation may be the primary relation-estimation failure and norm dominance merely a co-occurring optimization property. This screen does not establish causality.
- The result may be seed-specific. The 142 within-run samples are correlated trajectory observations and are not substitutes for independent seeds.
- The realized scheduler paths diverge after warmup. That is part of the operational method comparison, but a later ablation may need a fixed LR path to isolate direct regularizer effects.

## Decision

**PROMISING**, not confirmed.

The exposure check passed and the pre-registered norm-dominance criterion was met in two training thirds. The pairwise-conflict mechanism is not supported, so the confirmation question is narrow:

> Does persistent ER shared-gradient norm dominance reproduce across seeds, and does classical MTRL consistently fail to reduce it while Ω saturates?

Do not start a new mechanism or targeted literature search from this one seed. Continue DG-0002 with matched baseline and MTRL seeds `0,1,2,3,4`, the same 30-epoch protocol, and seed-level phase summaries. Treat each seed—not each sampled step—as the independent unit. If norm dominance is reproducible, it identifies a concrete optimization-scale limitation and justifies targeted literature work. If it is not, reject the gradient-interaction explanation and return to Ω estimation/parameter-summary diagnostics.

## Matched-seed confirmation

### Execution and controls

Baseline and MTRL arms completed for seeds `0,1,2,3,4` using the unchanged 30-epoch `smp(0.5)` 25-layer protocol. All ten runs used commit `7f6d5248f40c0c1cbd30f15b8f7cd1fe2dbb04eb`. Every baseline/MTRL seed pair sampled the same 142 of 2,820 steps, with identical sampled valid-example counts and the same 1,550,800 shared parameters. The independent observations below are the five seed-level phase summaries, not within-run gradient samples.

### Reproducible gradient-scale imbalance

| Method | Phase | Mean max/min task-norm ratio | Seed SD | 95% t interval | Seeds ≥ 3 |
| --- | --- | ---: | ---: | --- | ---: |
| baseline | middle | 7.243 | 0.272 | [6.905, 7.581] | 5/5 |
| baseline | late | 8.962 | 0.456 | [8.396, 9.528] | 5/5 |
| MTRL | middle | 7.129 | 0.421 | [6.606, 7.652] | 5/5 |
| MTRL | late | 9.004 | 1.059 | [7.689, 10.319] | 5/5 |

ER was the largest-norm task in these phases. Baseline seed-level middle ratios ranged 6.921–7.484 and late ratios 8.272–9.382. MTRL-minus-baseline paired ratio differences were:

| Phase | Mean paired Δ | 95% paired t interval |
| --- | ---: | --- |
| early | +0.007 | [−0.136, +0.150] |
| middle | −0.114 | [−0.717, +0.490] |
| late | +0.042 | [−1.473, +1.556] |

MTRL therefore did not consistently reduce the scale imbalance. It remained above the pre-registered ratio-3 threshold in the middle and late thirds for every seed.

### Pairwise compatibility

No baseline seed produced a pair meeting the persistent-conflict criterion. Across seeds, baseline late mean cosines were KS↔SI `+0.002`, KS↔ER `+0.001`, and SI↔ER `+0.022`; corresponding negative-conflict frequencies were `0.464`, `0.426`, and `0.357`. MTRL late means were similarly near zero (`+0.002`, `+0.004`, `+0.020`). The persistent pairwise-conflict branch is rejected for this protocol: gradients become near-orthogonal, not persistently opposed.

### Ω correspondence

Every MTRL seed ended with saturated off-diagonal magnitudes near `1/3`. Seeds 0–3 were uniform positive. Seed 4 had KS↔SI `+0.33316`, KS↔ER `−0.33323`, and SI↔ER `−0.33316`, reproducing F6's joint ER-edge sign flip. Thus magnitude saturation is reproducible, but uniform positive coupling is not. Ω's sign structure varies while the gradient-norm imbalance remains present in every seed; Ω does not regulate that scale behavior.

### Outcome and validation context

Fixed-final-epoch outcomes—the pre-registered context checkpoint—showed no resolved MTRL effect:

| Metric | baseline mean | MTRL mean | paired Δ | 95% paired t interval |
| --- | ---: | ---: | ---: | --- |
| aggregate | 0.974730 | 0.974423 | −0.000307 | [−0.001213, +0.000599] |
| KS | 0.986042 | 0.985867 | −0.000176 | [−0.000628, +0.000277] |
| SI | 0.978209 | 0.977845 | −0.000364 | [−0.002069, +0.001342] |
| ER | 0.783002 | 0.781917 | −0.001085 | [−0.010423, +0.008253] |

MTRL's mean final/best validation aggregate accuracies were `0.972066/0.972922`, versus baseline `0.973076/0.973637`; final training accuracy was essentially unchanged (`0.991076` vs `0.990917`). The validation pattern and fixed-epoch tests provide no outcome improvement. The `opt` ordinary-split ER delta (`−0.01049`, interval excluding zero) reproduces F4's speaker-leaky result and is not an admissible ER generalization claim.

### Interpretation and limits

The narrow confirmed statement is:

> Under matched three-task `smp` 25-layer training, ER shared-gradient norms dominate KS/SI in the middle and late phases across seeds 0–4, while classical MTRL neither removes the imbalance nor produces a resolved outcome improvement.

This is a concrete optimization-scale limitation of the current mechanism, not causal evidence that scale imbalance explains all of MTRL's null outcome. ER contributes about 47 valid examples per sampled batch versus 539 KS and 1,462 SI; scarcity, task difficulty, label noise and gradient-estimate variance remain competing causes. A data-regime control is required before calling the imbalance task-intrinsic. No ER performance claim is made, so this diagnostic does not substitute for LOSO.

### Study decision

**CONFIRMED** for the norm-dominance diagnostic. **REJECTED** for persistent pairwise conflict. No architecture is promoted.

The next action is targeted literature research from the measured failure mode: explicit task-relation methods that account for unequal task-gradient scale or relation reliability. Generic gradient surgery or loss reweighting is not automatically in Task Relation Learning scope and must not be implemented without taxonomy verification and a published method mapping.
