# Classical MTRL diagnostic synthesis

Date: 2026-09-22

This synthesis combines legacy evidence with registered diagnostic Studies DG-0001 and DG-0002. It is an analysis record, not a mechanism proposal. Historical runs before DG-0001 remain identified as `LEGACY-PRE-ID`.

## Analysis question

Why has classical MTRL failed to produce a significant, reproducible improvement over the matched wavCSE baseline for KS, SI and speaker-independent ER, and does the evidence already identify which mathematical assumption fails?

The decision criterion is deliberately strict: distinguish an observed property of learned Ω from evidence that the property causes transfer or downstream performance.

## What classical MTRL assumes

Classical MTRL places a matrix-normal prior on task parameters and optimizes

`loss(W) + λ1 ||W||²_F + λ2 tr(W Ω⁻¹ Wᵀ)`

subject to symmetric positive-semidefinite Ω and a trace constraint. Given task parameters, the relation update is analytic:

`Ω = sqrt(WᵀW) / tr(sqrt(WᵀW))`.

The wavCSE implementation stores tasks as rows, so it uses `WWᵀ` and evaluates `tr(Wᵀ Ω⁻¹ W)`. The implementation therefore makes these concrete assumptions:

1. **Symmetric, undirected relation.** Ω cannot represent `T(A <- B) != T(B <- A)`.
2. **One dense global structure.** One matrix couples all task heads; there is no learned edge selection, direction, layer-specific structure or confidence per edge.
3. **Parameter covariance is relevant to transfer.** Similarity of task-head parameter summaries is assumed to be a useful regularization target. It is not empirical transfer and need not have the same sign.
4. **Task summaries are comparable.** Because KS, SI and ER have 12, 1251 and 4 output classes, respectively, each task is represented by the mean classifier-weight row plus mean bias, rather than by the full task parameters. With `normalize_w=true`, these summaries are unit-normalized.
5. **The relation is global at each update.** Ω is refreshed once per epoch after warmup, so the optimization path can be dynamic, but each update still applies one symmetric matrix to all represented head dimensions.
6. **The analytic subproblem remains meaningful.** The closed-form Ω update is exact conditional on the current summary matrix. The original joint-convexity statement does not extend to the full deep wavCSE network; only the Ω subproblem is analytic here.

These are mathematical/implementation facts. Existing outcomes do not yet tell us which assumption causes the negative result.

## Controlled evidence table

Accuracy is shown as mean ± population standard deviation for the five-seed and ten-fold groups, matching the existing project summaries. `Δ` is candidate minus its matched baseline. ER in ordinary runs uses the speaker-leaky historical split and is not evidence of speaker-independent ER transfer.

| Evidence unit | Method / stage | Seeds or folds | Tasks | Pooling / layers | Evaluation | KS | SI | ER | Aggregate | Matched delta (KS / SI / ER / all) | Relation diagnostic | Commit | Caveat / level |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| `LEGACY-PRE-ID`, 16L control | wavCSE baseline, confirmation | seeds 0–4 | KS/SI/ER | `smp` 0.5 / selected 16 | 30 epochs, ordinary split, `opt` | 0.98376 ± 0.00111 | 0.97462 ± 0.00254 | 0.76890 ± 0.00787 | 0.97134 ± 0.00130 | reference | none | `bfb1ad44` | ER leaky; otherwise matched. Level A for KS/SI/aggregate baseline variability |
| `LEGACY-PRE-ID`, 16L MTRL | MTRL λ=0.01, `normalize_w`, confirmation | seeds 0–4 | KS/SI/ER | `smp` 0.5 / selected 16 | same | 0.98435 ± 0.00093 | 0.97387 ± 0.00181 | 0.76998 ± 0.00787 | 0.97124 ± 0.00091 | +0.00059 / −0.00075 / +0.00108 / −0.00010 | final Ω varies strongly by seed | `bfb1ad44` | all paired 95% CIs include zero; ER leaky. Level A: no meaningful aggregate effect |
| `LEGACY-PRE-ID`, 25L control | wavCSE baseline, confirmation | seeds 0–4 | KS/SI/ER | `smp` 0.5 / all 25 | 30 epochs, ordinary split, `opt` | 0.98657 ± 0.00087 | 0.97758 ± 0.00138 | 0.78662 ± 0.00647 | 0.97476 ± 0.00108 | reference | none | `bfb1ad44` | ER leaky; otherwise matched. Level A for KS/SI/aggregate baseline variability |
| `LEGACY-PRE-ID`, 25L MTRL | MTRL λ=0.01, `normalize_w`, confirmation | seeds 0–4 | KS/SI/ER | `smp` 0.5 / all 25 | same | 0.98578 ± 0.00129 | 0.97787 ± 0.00071 | 0.77613 ± 0.00932 | 0.97419 ± 0.00071 | −0.00079 / +0.00029 / −0.01049 / −0.00056 | KS↔SI saturates positive in 5/5; ER edges flip together in 1/5 | `bfb1ad44` | ER loss is consistent on the leaky split but is contradicted as a general ER effect by LOSO. Level A: no meaningful aggregate gain; Level C for ER generalization |
| `LEGACY-PRE-ID`, LOSO control | wavCSE baseline | 10 held-out speakers | joint KS/SI/ER training | `smp` 0.5 / all 25 | 5 epochs/fold, ER LOSO, `opt` | not summarized | not summarized | 0.63907 ± 0.05065 | not summarized | reference | none | `bfb1ad44` | one training realization per fold; speaker-independent. Level A for mean ER protocol result |
| `LEGACY-PRE-ID`, LOSO MTRL | MTRL λ=0.01, `normalize_w` | same 10 folds | joint KS/SI/ER training | same | same | not summarized | not summarized | 0.63797 ± 0.04837 | not summarized | ER −0.00110; paired fold 95% CI [−0.03070, +0.02850] | full five-epoch Ω history per fold | `bfb1ad44` | MTRL warmup scaled from 3/30 to 1/5. Level A: no resolved ER effect |
| `LEGACY-PRE-ID`, LNP control | wavCSE baseline, control | unseeded single run | KS/SI/ER | `lnp` p=16 / selected 16 | 30 epochs, ordinary split, `opt` | 0.98669 | 0.97406 | 0.78843 | 0.97302 | reference | none | `ae0b15ef` | single seed, leaky ER. Level C |
| `LEGACY-PRE-ID`, LNP MTRL | MTRL λ=0.01, `normalize_w`, control | unseeded single run | KS/SI/ER | same | same | 0.98742 | 0.97334 | 0.77939 | 0.97263 | +0.00073 / −0.00073 / −0.00904 / −0.00038 | uniform saturated Ω ≈ +1/3 | `ae0b15ef` | single seed, leaky ER. Comparing MTRL final epoch against baseline `opt` changes checkpoint policy and is not a controlled architecture claim. Level C |
| `DG-0002`, gradient confirmation | wavCSE baseline vs MTRL, confirmation | seeds 0–4 | KS/SI/ER | `smp` 0.5 / all 25 | 30 epochs; fixed-epoch outcome context | 0.98604 / 0.98587 | 0.97821 / 0.97785 | 0.78300 / 0.78192 | 0.97473 / 0.97442 | −0.00018 / −0.00036 / −0.00109 / −0.00031 | baseline norm ratio 7.24 middle, 8.96 late; MTRL 7.13, 9.00; no persistent conflict | `7f6d5248` | all outcome CIs include zero; ER outcome leaky. Level A for the matched gradient diagnostic |
| `DG-0005`, data-regime control | wavCSE baseline A0 vs A1 ER-weighted composition | seeds 0–4 | KS/SI/ER | `smp` 0.5 / all 25 | 30 epochs; fixed-epoch outcome context | 0.98604 / 0.98669 | 0.97821 / 0.97898 | 0.78300 / 0.79349 | 0.97473 / 0.97579 | +0.00064 / +0.00078 / +0.01049 / +0.00106 | A0 7.24 middle, 8.96 late; A1 2.82, 2.35; change ER-localized; no conflict signal | `8032a937` | exposure gate passed 5/5; only composition changed. ER leaky, no performance claim. Level A for the mixture control |

MLflow metadata confirms the five-seed and LOSO comparisons share commit `bfb1ad44ef987e6484183eda7d782f28cec5c667`. The legacy runs have no Study ID or DagsHub run note; the LOSO fold children also omit full pooling/layer parameters, which are present on their parent runs. Two abandoned duplicate seed runs remain marked `RUNNING` in MLflow but have no test metrics and were excluded.

### Paired five-seed effects

| Setting | Task | Mean Δ | Paired 95% CI | Seed signs (MTRL better / worse / tie) | Auxiliary relative error reduction |
| --- | --- | ---: | --- | --- | ---: |
| 16L | KS | +0.00059 | [−0.00019, +0.00136] | 4 / 1 / 0 | +3.60% |
| 16L | SI | −0.00075 | [−0.00435, +0.00285] | 1 / 4 / 0 | −2.96% |
| 16L | ER | +0.00108 | [−0.00557, +0.00774] | 3 / 2 / 0 | +0.47% |
| 16L | aggregate | −0.00010 | [−0.00183, +0.00163] | 1 / 3 / 1 | not used |
| 25L | KS | −0.00079 | [−0.00191, +0.00033] | 0 / 4 / 1 | −5.88% |
| 25L | SI | +0.00029 | [−0.00235, +0.00293] | 2 / 3 / 0 | +1.30% |
| 25L | ER | −0.01049 | [−0.01910, −0.00188] | 0 / 5 / 0 | −4.92% |
| 25L | aggregate | −0.00056 | [−0.00243, +0.00130] | 2 / 3 / 0 | not used |

The 25L ordinary-split ER interval excludes zero, but this does not survive the more appropriate speaker-independent protocol: LOSO `opt` is −0.00110 with paired 95% CI [−0.03070, +0.02850]. Relative error reduction is therefore auxiliary only and does not rescue an ER claim.

## Evidence levels

### Level A — strong within the stated protocol

- MTRL has no meaningful reproducible aggregate advantage at either 16L or 25L under matched five-seed evaluation.
- MTRL has no resolved speaker-independent ER effect under matched ten-fold LOSO.
- The historical ordinary ER split is not suitable for a generalization claim; it inflates the best baseline from 0.6391 LOSO to 0.7902 ordinary-split accuracy.
- Within the 25L LOSO condition, KS↔SI is fold-stable while ER-involving Ω entries are fold-sensitive and can change sign.
- At fixed `smp` pooling, final Ω stability changes materially between the five-seed 16L and 25L settings. This is strong evidence for those configurations, not a universal ranking of task pairs.
- DG-0002 establishes ER shared-gradient norm dominance under matched 25L training: every seed 0–4 exceeds ratio 3 in the middle/late phases, and MTRL does not consistently mitigate it. No seed supports persistent pairwise conflict.
- DG-0005 establishes that this dominance is mixture-controlled: changing only ER's per-batch sampling weight (≈47 → ≈435 examples of 2,048) removed the late-phase ratio (8.962 → 2.352) in 5/5 seeds with exposure, data and evaluation held fixed. Gradient scale is therefore a training-mixture property, not a task-intrinsic one (F10).

### Level B — moderate

- Pooling and layer representation materially change Ω. The effect is repeated across settings, but most pooling-specific Ω cells are legacy single-run observations.
- During five-epoch LOSO, KS↔SI strengthens and becomes consistent across folds while ER-edge dispersion grows. This is replicated descriptive dynamics, but its predictive consequence is untested.
- Increasing MTRL λ from 0.01 to 0.05 worsened all three tasks in the historical matched run. This rejects the specific “under-powered coupling” explanation but is not a general regularization law.

### Level C — preliminary or confounded

- The LNP final-epoch apparent improvement used a different checkpoint-selection comparison and one unseeded run.
- Historical weighted-MTRL versus mix-baseline claims changed pooling and cannot identify an architecture effect.
- Correlations between final ER edges and LOSO fold transfer are exploratory (`r=0.64` for KS↔ER versus paired `opt` delta; `n=10` non-independent folds, no held-out validation). They do not establish prediction or causality.
- DG-0001's directed-transfer interpretation remains confounded after exposure controls; useful sparsity and causal Ω/transfer correspondence remain untested.

## Per-task behavior

### KS

MTRL is effectively neutral. The 16L mean is +0.0585 percentage points and the 25L mean is −0.0790 points; both paired intervals cross zero. KS cannot support a benefit claim. The sample-weighted aggregate can obscure such small changes, but no hidden material KS effect appears after task-wise inspection.

### SI

MTRL is also effectively neutral: −0.0751 points at 16L and +0.0291 points at 25L, with wide paired intervals around zero. Because SI has the largest test set, small SI movements disproportionately affect aggregate accuracy. This explains why aggregate ranking can move even when ER changes more in percentage points; it does not show SI transfer.

### ER

The 16L ordinary split is neutral. At 25L MTRL is lower in all five seeds by 1.05 points on average, but that protocol shares speakers between training and test. The matched LOSO result is indistinguishable from zero. Therefore the only defensible ER conclusion is “no resolved effect”; neither improvement nor harm has been shown for unseen speakers.

### Aggregate

At 16L the aggregate difference is −0.010 points; at 25L it is −0.056 points. Both are below the project’s 0.20-point material-regression threshold and both paired intervals include zero. MTRL neither wins nor materially regresses the aggregate under the confirmed settings.

## Relations are four different objects

| Object | Available evidence | What can be concluded |
| --- | --- | --- |
| Empirical directed transfer `T(A <- B)` | DG-0001 single/pairwise screen, ten-fold ER LOSO and optimizer-exposure controls | The raw matrix is not a semantic transfer target: update count and effective task batch size dominate it. Controlled KS/ER residual is unresolved; SI/ER is negative in both directions. |
| Gradient interaction | DG-0002 exposure-matched baseline/MTRL confirmation, seeds 0–4, 142 sampled steps per arm; DG-0005 composition control, same seeds | ER shared-gradient norm dominance is reproducible **under the standard training mixture** and is removed by matching ER's per-batch share to KS scale (DG-0005, F10). MTRL does not consistently mitigate the standard-mixture signal. No seed supports persistent pairwise conflict. |
| Learned relation Ω | five-seed/fold/epoch diagnostics plus DG-0001 controlled transfer | Ω is symmetric, representation-sensitive and conditionally unstable. Its signs/magnitudes show moderate disagreement with controlled transfer, but protocols are not identical. |
| Downstream outcome | matched five-seed aggregate/task metrics, ER LOSO and DG-0001 controls | MTRL has no reproducible meaningful advantage; raw transfer asymmetry is an optimizer-exposure artifact. |

DG-0001 now supplies cases where learned similarity is not useful transfer.
25L KS↔SI stays near +1/3 while the one-seed 30-epoch directed cells are near
zero. Across LOSO folds, SI↔ER Ω is positive in 7/10 while step-controlled
transfer is negative in both directions. This weakens Ω as a utility proxy, but
does not yet prove whether its estimation or downstream regularization causes
MTRL's performance null.

DG-0002 adds a different Ω mismatch. Across seeds 0–4, final Ω off-diagonal
magnitudes saturate near `1/3` while ER shared-gradient norm ratios remain
7.24×/8.96× in baseline middle/late training. Four seeds are uniform positive;
seed 4 flips both ER-edge signs. This does not make Ω a task-weighting matrix or
establish causality; it shows that classical MTRL does not represent or regulate
the dominant optimization-scale signal measured under this protocol.

## Relation stability

Final Ω across matched seeds:

| Setting | Pair | Mean | SD | Sign consistency | Range | Interpretation |
| --- | --- | ---: | ---: | --- | --- | --- |
| `smp` 16L, seeds 0–4 | KS↔SI | +0.110 | 0.236 | 4/5 positive | −0.333 to +0.333 | unstable; one saturated sign flip |
| `smp` 16L, seeds 0–4 | KS↔ER | +0.035 | 0.213 | 4/5 positive | −0.333 to +0.333 | unstable; one saturated sign flip |
| `smp` 16L, seeds 0–4 | SI↔ER | −0.256 | 0.076 | 5/5 negative | −0.333 to −0.138 | weaker magnitude variation and stable sign |
| `smp` 25L, seeds 0–4 | KS↔SI | +0.33308 | 0.00017 | 5/5 positive | +0.33276 to +0.33324 | extremely stable but saturated |
| `smp` 25L, seeds 0–4 | KS↔ER | +0.200 | 0.266 | 4/5 positive | −0.333 to +0.333 | unstable sign because one seed flips |
| `smp` 25L, seeds 0–4 | SI↔ER | +0.200 | 0.266 | 4/5 positive | −0.333 to +0.333 | unstable sign because one seed flips |
| `smp` 25L, 10 LOSO folds | KS↔SI | +0.286 | 0.012 | 10/10 positive | +0.259 to +0.305 | fold-stable within this condition |
| `smp` 25L, 10 LOSO folds | KS↔ER | +0.098 | 0.072 | 9/10 positive | −0.079 to +0.201 | fold-sensitive |
| `smp` 25L, 10 LOSO folds | SI↔ER | +0.033 | 0.103 | 7/10 positive | −0.202 to +0.205 | fold-sensitive |

The previous shorthand “KS↔SI is stable; ER relations are unstable” is valid for the 25L LOSO folds, but not as a task-intrinsic statement. At 16L, KS↔SI is seed-unstable while SI↔ER has the only fully consistent sign. Stability is conditional on representation, layer selection and perturbation axis.

A saturated stable edge is not automatically useful. At 25L, KS↔SI is maximally stable but MTRL does not improve either task materially. Relation strength and confidence must therefore be reported separately from outcome utility.

## Dynamic behavior

The ten LOSO artifacts contain five Ω updates per fold:

| Epoch | KS↔SI mean ± SD | Positive folds | KS↔ER mean ± SD | Positive folds | SI↔ER mean ± SD | Positive folds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | +0.0347 ± 0.0065 | 10/10 | +0.0182 ± 0.0130 | 9/10 | −0.0040 ± 0.0068 | 2/10 |
| 2 | +0.1050 ± 0.0176 | 10/10 | +0.0285 ± 0.0192 | 9/10 | −0.0084 ± 0.0210 | 2/10 |
| 3 | +0.2540 ± 0.0201 | 10/10 | +0.0537 ± 0.0333 | 9/10 | −0.0119 ± 0.0558 | 4/10 |
| 4 | +0.2805 ± 0.0110 | 10/10 | +0.0728 ± 0.0510 | 9/10 | +0.0054 ± 0.0772 | 7/10 |
| 5 | +0.2856 ± 0.0120 | 10/10 | +0.0981 ± 0.0721 | 9/10 | +0.0332 ± 0.1026 | 7/10 |

KS↔SI strengthens consistently and largely stabilizes by epochs 4–5. ER-edge dispersion increases throughout training; SI↔ER also changes majority sign. This establishes dynamics, but not that a dynamic relation mechanism would help. The artifacts do not align Ω changes with task-specific gradient conflict, and a five-epoch budget is too short to claim a general stabilization time relative to validation convergence.

## Failure clusters

1. **No meaningful architecture effect.** Matched five-seed aggregate comparisons and ER LOSO all resolve to no effect.
2. **Relation estimate conditionality.** Ω changes with pooling, layer selection, seed and held-out speaker. A global pair ranking does not survive all axes.
3. **Mechanism collapse to uninformative coupling.** Several normalized-W settings produce rank-near-one, ±1/3 saturation. Uniform saturation removes pair-specific discrimination; causality for accuracy remains untested.
4. **Parameter-summary sensitivity.** Unit normalization moves Ω from weak norm-dominated coupling to saturation; mean-pooled heads are a lossy substitute for the classical comparable task-parameter columns. Whether this is the failure source is unknown.
5. **Excessive regularization.** Increasing λ from 0.01 to 0.05 worsened all tasks and did not produce more informative Ω, weakening the “coupling is merely too weak” hypothesis.
6. **Protocol confounding.** Unmatched pooling, seed noise, checkpoint policy, ER leakage and task-count-dependent optimizer exposure each produced false architecture/relation narratives.
7. **Optimizer-exposure dominance.** DG-0001's raw ER gains of 28–34 points were reproduced by ER-only step controls; KS left no resolved residual and SI became negative.
8. **Aggregate masking.** SI’s larger test set dominates aggregate movement; task-wise trade-offs must be checked even when aggregate differences are small.

9. **Optimization-scale mismatch** (cause resolved 2026-09-22). ER shared-gradient norms dominate KS/SI under the standard training mixture; MTRL leaves the ratio essentially unchanged while Ω magnitude saturates. DG-0005 then showed the signal is mixture-controlled: matching ER's per-batch sample count to KS scale removes it in 5/5 seeds with exposure fixed (F10). A pre-registered post-hoc bound then localized what remains: estimator-size scaling covers the middle phase entirely (share `1.03`) but at most `0.758` of the late drop, so ≥20% of the late change is a smaller mean gradient or steeper-than-`1/√n` noise growth, and per-step dispersion cannot proxy estimator variance at all (DEC-0012). The mismatch therefore no longer licenses a mechanism motivated by a task-intrinsic scale property (DEC-0010).

There is established evidence against persistent pairwise gradient conflict and
no supported beneficial asymmetry. SI/ER negative interaction remains moderate
evidence under approximate step matching; useful sparsity and data-size
causation remain untested.

## Active hypothesis assessment

| Hypothesis | Status | Evidence for | Evidence against / missing |
| --- | --- | --- | --- |
| MTRL meaningfully beats wavCSE | contradicted | isolated single-run/checkpoint gains | matched five-seed and LOSO results show no reproducible advantage |
| Useful transfer is asymmetric | weakened / not supported | raw ER-directed outcomes were asymmetric | F8 shows optimizer exposure explained them; no positive controlled residual remained |
| Useful relations are sparse/selective | insufficient evidence | controlled KS/ER is neutral and SI/ER negative | KS/SI is one seed; no replicated beneficial edge pattern |
| Relation confidence should be distinct from magnitude | supported as a diagnostic principle; mechanism value unknown | stable saturated KS↔SI coexists with no gain; ER-edge variance is high | no confidence-aware intervention has been tested |
| ER instability is caused by smaller/noisier data | **supported for gradient scale**, unmeasured for relation estimates | DG-0005's matched-composition control removes ER's gradient-norm dominance in 5/5 seeds (F10) | ER-edge Ω instability under matched composition has not been measured; the residual drop may be convergence rather than estimator variance |
| Relations vary by representation depth | supported descriptively, not causally | matched `smp` 16L and 25L final Ω stability patterns differ | layer selection also changes represented information; no layer-wise diagnostic |
| Relations change during training | supported descriptively, predictive value unknown | LOSO Ω trajectories differ by pair | no gradient alignment or prospective outcome prediction |
| One global structure creates negative transfer | insufficient / moderate for SI↔ER | step-controlled SI/ER residual is negative in both directions | batch-size matching is approximate and MTRL edge ablation is absent |
| Current task summaries are inadequate | plausible, strengthened but unproven | Ω sign/magnitude disagrees with controlled transfer; normalization changes Ω radically | triple-task Ω and pairwise transfer protocols differ |
| Ω saturation causes MTRL failure | insufficient evidence | saturation co-occurs with neutral/below-baseline outcomes | no intervention that changes saturation while holding the rest fixed; direction of causality unknown |
| ER gradient-scale dominance is a limitation of current MTRL | reframed: MTRL does not mitigate a signal that is itself mixture-dependent | ratio exceeds 3 in middle/late for 5/5 baseline and 5/5 MTRL seeds under the standard mixture; paired mitigation intervals include zero | the dominance is not task-intrinsic (F10), so it cannot motivate a relation mechanism as a task property (DEC-0010); only the estimator-vs-convergence split remains open |
| Stronger λ fixes under-coupling | weakened / contradicted for tested setting | initial weak-Ω run motivated it | λ=0.05 worsened KS, SI and ER and reduced off-diagonal magnitude |

## Emerging selection framework

This is an evidence boundary, not a completed method-selection framework. The cross-study version of these rules, and the conditioned-quantity table they belong to, is `research/FRAMEWORK.md`.

| Observable | Current KS/SI/ER evidence | Assumption it could test | Selection implication now |
| --- | --- | --- | --- |
| Transfer symmetry and sign | beneficial asymmetry not supported after exposure controls | symmetric vs directional | do not select an asymmetric method from DG-0001 |
| Pair selectivity | unknown empirically | dense vs sparse/selective | unknown; Ω weakness alone is insufficient |
| Gradient cosine/conflict | no persistent conflict in 5/5 seeds; late gradients near-orthogonal | conflict-aware relation vs covariance relation | conflict-only surgery is not indicated |
| Gradient norm scale | ER dominates KS/SI by roughly 7–9× under the standard mixture, and returns to ≈2.4× once ER's per-batch share reaches KS scale (F10) | scale-aware or reliability-aware explicit relations | the search was run and found no eligible method (LT-0001/FL-0003); a mixture-controlled scale signal cannot justify a relation mechanism |
| Relation stability | condition-specific; exact tables above | deterministic dense vs confidence-aware | confidence must be reported; it remains unmeasured under matched composition, so it selects no mechanism |
| Data-size imbalance | norm dominance tracks per-batch composition (DG-0005); relation-estimate noise under matched composition is still unmeasured | sample-aware/uncertainty-aware | a sample-aware *relation* mechanism must show relation-estimate noise, not gradient scale; task-intrinsic claims are now excluded (F10) |
| Representation sensitivity | strong | global vs representation- or layer-specific | any claimed relation must name pooling/layers; mechanism choice still unknown |
| Temporal dynamics | present in LOSO Ω | static vs dynamic | existence alone does not justify a dynamic method |
| Stable relation with no task gain | observed for 25L KS↔SI | learned covariance vs useful transfer | Ω magnitude cannot be used as a utility proxy |
| Higher-order structure | unmeasured | pairwise vs higher-order | unknown |

## Highest-value missing evidence

### What we know strongly

- Classical MTRL provides no meaningful reproducible advantage under matched five-seed aggregate evaluation or speaker-independent ER LOSO.
- Ordinary-split ER, single seeds, unmatched pooling and unmatched optimizer exposure can each create false architecture narratives.
- Learned Ω is not task-intrinsic: its strength and stability depend on representation and perturbation axis.
- DG-0001's same-epoch raw transfer matrix is dominated by task-count-dependent optimizer exposure (F8).
- DG-0002 confirms ER gradient-scale dominance across seeds and rejects persistent pairwise conflict under the matched protocol (F9).
- DG-0005 shows that dominance is mixture-controlled, so gradient scale is not a task-intrinsic property (F10).
- LT-0001 (eight primary sources) found no published method that is both an explicit Task Relation Learning method and a direct mechanism for that scale signal, so no mechanism is authorized (FL-0003, DEC-0010).

### What appears likely but remains uncertain

- Ω encodes head-parameter geometry rather than useful transfer.
- SI and ER interfere under approximate exposure matching.
- The confirmed scale imbalance may contribute to MTRL's null outcome, but no causal intervention has isolated it.
- The residual ER norm drop under matched composition is either reduced gradient-estimate variance or ER convergence/overfitting; DG-0005 cannot separate them.
- ER's *relation-estimate* noise (Ω edge dispersion) under matched composition is unmeasured, so "reliability-aware relations" remains an untested assumption rather than a motivated mechanism.

### What is contradicted

- MTRL has a confirmed performance win.
- Stronger λ simply amplifies a useful relation.
- Raw pair-minus-single accuracy at equal epochs is a valid transfer matrix when task count changes batching.
- DG-0001 justifies an asymmetric replacement method.
- Persistent pairwise gradient conflict is the DG-0002-supported explanation.

### Status of the DEC-0007 mandate (closed)

The literature sequence this section previously prescribed **has been executed**:

1. **Targeted literature** — `LT-0001` screened eight primary sources against pre-registered eligibility gates.
2. **Taxonomy and assumption gate** — applied; explicit relation methods either lacked a direct scale mechanism or required aligned Gaussian/mean-estimation assumptions, while the direct scale methods (uncertainty weighting, GradNorm) learn per-task scalars with no relation object.
3. **Candidate `LT-xxxx` Study** — not registered: no source satisfied the gates (FL-0003, `literature/INDEX.md`).
4. **Outcome** — `NEEDS-HUMAN-REVIEW`; DEC-0009 retained strict scope, DEC-0010 then withdrew the task-intrinsic-scale rationale once DG-0005 showed the signal is mixture-controlled.

Re-running this search against F9 is therefore prohibited without new evidence or an explicit scope change. The remaining open questions are not literature questions; they are the diagnostic ones listed below.

### Highest-information remaining evidence

1. **The late-phase mean-gradient component** of ER's drop under DG-0005's matched composition (DEC-0012) — the middle phase is already accounted for by estimator size, and the late residual is not. Discriminating measurement: `‖E g‖`, the pool-mean gradient, which is estimator-noise-free; per-batch dispersion is unusable here. Bounded: instrument the existing arms, no new architecture.
2. **Relation-estimate noise under matched composition** — would require an MTRL arm under the A1 composition, which DG-0005 deliberately excluded; it is the only route that would make a confidence/reliability-aware relation assumption checkable rather than assumed.
3. **Layer-wise gradient relation** (TR-0005) and **parameter-summary adequacy** (TR-0006/DG-0003) — instrumentation-level, no architecture change.

The cross-study view of these priorities, with the selection rules they feed, is `FRAMEWORK.md`.

## Concise synthesis

1. **Strongest conclusion:** MTRL is outcome-neutral; raw directed transfer is dominated by optimizer exposure; ER gradient-scale dominance is reproducible only under the standard training mixture and is not mitigated by MTRL.
2. **Most important resolved question:** the dominance is a training-mixture property, not a task-intrinsic one (F10) — so it cannot justify a scale- or reliability-aware *relation* mechanism (DEC-0010).
3. **Most important open question:** is the residual ER norm drop reduced gradient-estimate variance or ER convergence/overfitting?
4. **Literature:** the F9-era search is complete and negative (`LT-0001`, FL-0003); do not repeat it without new evidence or an explicit scope change.
5. **Framework:** the emerging selection rules and the conditioned-quantity table are in `FRAMEWORK.md`; no mechanism is authorized today.
