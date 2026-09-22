# DG-0002 Analysis — Exposure-controlled gradient compatibility

Status: **PROMISING — one-seed diagnostic screen; confirmation required**

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
