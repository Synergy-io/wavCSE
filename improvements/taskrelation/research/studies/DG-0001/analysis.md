# DG-0001 analysis — empirical directed transfer

Date: 2026-09-21
Status: COMPLETE

## Question

Does empirical KS/SI/ER transfer show a directional or pair-selective structure that classical MTRL's symmetric dense Ω cannot represent?

## Protocol

Stage A used the plain wavCSE downstream model, `smp` 0.5, all 25 layers, seed 42 and 30 epochs. Three single-task, three pairwise and one triple-task arm were run at commit `4c0f08a57b2c63931fce75ecc4d7d7a2efe0eef9`. Fixed-final-epoch task accuracy was pre-registered as primary.

Because Stage A suggested ER-directed asymmetry, Stage C repeated `er`, `ks_er` and `si_er` under matched ten-fold speaker-independent LOSO, five epochs/fold, at commit `330d0f78e449ec07e60ddbd0713b61639f31ca24`.

Stage C exposed an optimization confound: adding a large auxiliary dataset changes the number of batches and effective ER minibatch size. Stage D therefore added:

- five-epoch single-task KS and SI reverse controls;
- ER-only LOSO with batch 160, approximately matching KS+ER's 27 updates/fold/epoch;
- ER-only LOSO with batch 64, approximately matching SI+ER's 69 updates/fold/epoch.

Stage D ran at commit `3f4752afc88aeb135931d32e6b3d50d330efb2dd`.

All new MLflow runs carry Study ID, stage, task set, seed, method, pooling, layers, commit SHA and a DagsHub run note. GBC and LNP were not used.

## Stage A — raw fixed-epoch transfer screen

`T(A <- B) = accuracy(A trained with B) - accuracy(A trained alone)`.

| Target \ auxiliary | KS | SI | ER |
| --- | ---: | ---: | ---: |
| KS | — | −0.00059 | +0.00015 |
| SI | −0.00170 | — | −0.00885 |
| ER | +0.05787 | +0.02893 | — |

At face value, ER appeared to benefit from KS and SI while neither auxiliary benefited from ER. This met the pre-registered candidate-asymmetry threshold. It remained Level C because it was one seed and ER used the leaky split.

## Stage C — raw LOSO transfer

Fixed-epoch mean accuracy across the same ten held-out speakers:

| Arm | Target | Mean accuracy |
| --- | --- | ---: |
| ER-only, batch 2048 | ER | 0.33685 |
| KS+ER | ER | 0.67796 |
| SI+ER | ER | 0.61626 |
| KS+ER | KS | 0.97996 |
| SI+ER | SI | 0.92963 |

Raw paired LOSO differences:

| Direction | Mean Δ | Paired 95% CI | Sign |
| --- | ---: | --- | --- |
| ER <- KS | +0.34111 | [+0.28469, +0.39754] | positive in 10/10 folds |
| ER <- SI | +0.27941 | [+0.21117, +0.34766] | positive in 10/10 folds |

These values do not establish semantic transfer. ER-only at batch 2048 receives only about two optimizer updates per epoch. KS+ER receives about 27 total updates and SI+ER about 69; shuffled combined batches also split ER examples into much smaller effective minibatches. The intervention changed information, update count and optimization granularity together.

## Stage D — optimization-exposure decomposition

Step-matched ER-only fixed-epoch means:

| ER-only control | Approximate updates/fold/epoch | Mean ER LOSO accuracy |
| --- | ---: | ---: |
| batch 2048 | 2 | 0.33685 |
| batch 160 | 26–28, matching KS+ER's 27 | 0.67267 |
| batch 64 | 67–70, matching SI+ER's 69 | 0.67518 |

For each fold:

`raw pair gain = optimizer-exposure gain + pair-minus-step-matched residual`.

| Auxiliary | Raw ER gain | Exposure gain | Residual pair − step-matched ER | Residual 95% CI | Residual signs |
| --- | ---: | ---: | ---: | --- | --- |
| KS | +0.34111 | +0.33582 | +0.00529 | [−0.02886, +0.03943] | 5 positive / 5 negative |
| SI | +0.27941 | +0.33833 | −0.05892 | [−0.08831, −0.02953] | 0 positive / 10 negative |

The KS-directed apparent benefit disappears after approximate update matching. The SI-directed apparent benefit reverses: SI+ER underperforms step-matched ER-only by 5.89 points in every fold.

This is not a claim that batch 160 or 64 is an optimized ER model. Batch size was chosen only to approximate the pair arm's update count; it changes gradient noise and does not exactly equalize every optimization detail. It is nevertheless sufficient to reject the interpretation that the raw 28–34 point gains identify auxiliary-task semantics.

## Reverse directions at the matched five-epoch budget

Five-epoch single-task references were KS 0.98376 and SI 0.95273. Pairwise fold means were compared with those references:

| Direction | Mean Δ | 95% interval over ER folds | Sign |
| --- | ---: | --- | --- |
| KS <- ER | −0.00380 | [−0.00500, −0.00260] | negative in 10/10 folds |
| SI <- ER | −0.02310 | [−0.02748, −0.01873] | negative in 10/10 folds |

These intervals describe variation over ER folds while each single-task reference is one seed-42 fit, not ten independent single-task refits. They are sensitivity evidence, not a multi-seed confirmation. Together with the step-matched ER residuals, the controlled pattern is neutral KS↔ER and mutually harmful SI↔ER—not demonstrated beneficial asymmetry.

## Comparison with learned Ω

The relevant final MTRL Ω values come from matched 25L `smp` evidence:

- across five seeds, KS↔SI is saturated positive in 5/5;
- KS↔ER and SI↔ER are saturated positive in 4/5 and negative in 1/5;
- across ten LOSO folds, KS↔ER is positive in 9/10 and SI↔ER positive in 7/10.

Against controlled LOSO residuals:

| Pair | LOSO Ω mean ± SD | Controlled direction | Mean transfer | Ω/transfer sign agreement |
| --- | ---: | --- | ---: | ---: |
| KS↔ER | +0.098 ± 0.072 | ER <- KS | +0.0053, unresolved | 6/10 |
| KS↔ER | +0.098 ± 0.072 | KS <- ER | −0.0038 | 1/10 |
| SI↔ER | +0.033 ± 0.103 | ER <- SI | −0.0589 | 3/10 |
| SI↔ER | +0.033 ± 0.103 | SI <- ER | −0.0231 | 3/10 |

Exploratory fold correlations are not predictive evidence: KS↔ER Ω versus ER residual `r=0.195`; SI↔ER Ω versus ER residual `r=0.469`, both `n=10` with no held-out validation.

The exact-protocol Stage-A KS/SI screen supplies a second discrepancy: 25L Ω is stably near +1/3, yet both directed KS/SI cells are within 0.17 percentage points of zero. This is one-seed transfer evidence, so it cannot establish absence of a sub-1-point effect, but it shows that saturated Ω magnitude is not sufficient evidence of useful transfer.

## Hypothesis assessment

**Primary hypothesis: useful transfer is directional and pair-selective.**

**Weakened / not supported by the controlled evidence.** Raw training outcomes were strongly directional, but almost all apparent ER benefit was reproduced by ER-only update matching. No positive semantic-transfer residual was resolved. SI shows negative residual transfer in both directions; KS/ER is neutral-to-small-negative within current uncertainty.

**Classical MTRL symmetry as the cause of failure.**

**Insufficient evidence.** DG-0001 does not justify an asymmetric replacement method. A symmetric matrix may still be inadequate for negative/selective structure, but that claim needs replicated controlled transfer rather than the raw matrix.

**Learned Ω corresponds to useful transfer.**

**Weakened.** Stable/saturated positive Ω does not align with the controlled direction signs or with observed MTRL benefit. However Ω comes from triple-task MTRL while the controlled transfer arms are baseline pairwise runs; this comparison identifies a discrepancy, not its cause.

## Evidence level

- **Strong for this protocol:** raw pair-versus-single ER gains are dominated by task-count-dependent optimizer exposure; the original directed matrix is not a semantic transfer matrix.
- **Moderate:** SI and ER interfere after approximate step matching; Ω sign/magnitude does not reliably track controlled transfer.
- **Preliminary:** KS/SI near-zero transfer, reverse-direction magnitudes and any pair taxonomy beyond the ten-fold ER controls. Multi-seed confirmation is absent.

## Decision

DG-0001 is complete as a diagnostic negative result.

1. Do not select or search for an asymmetric Task Relation Learning method from the raw matrix.
2. Do not treat same-epoch single/pairwise comparisons as empirical transfer unless optimizer exposure, effective task batch size and checkpoint policy are controlled.
3. Classical MTRL's negative outcome remains unexplained by symmetry. The stronger supported limitation is that Ω is not a validated proxy for useful transfer.
4. The next highest-information diagnostic is DG-0002: task-specific gradient norms, cosine and conflict frequency under an exposure-controlled sampling protocol. This tests whether optimization interaction—not covariance parameterization—explains the MTRL null result.
5. Targeted next-method literature search is not yet justified. It becomes justified only if DG-0002 or a replicated controlled-transfer study identifies a concrete mechanism-level failure.
