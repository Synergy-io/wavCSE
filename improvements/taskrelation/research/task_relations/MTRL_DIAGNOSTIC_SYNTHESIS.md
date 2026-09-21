# Classical MTRL diagnostic synthesis

Date: 2026-09-21

This is an analysis of legacy evidence, not a new Study and not a mechanism proposal. No training was launched. Historical runs predate Study-ID tagging (FINDINGS.md R2), so their Study ID is recorded as `LEGACY-PRE-ID`.

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

### Level B — moderate

- Pooling and layer representation materially change Ω. The effect is repeated across settings, but most pooling-specific Ω cells are legacy single-run observations.
- During five-epoch LOSO, KS↔SI strengthens and becomes consistent across folds while ER-edge dispersion grows. This is replicated descriptive dynamics, but its predictive consequence is untested.
- Increasing MTRL λ from 0.01 to 0.05 worsened all three tasks in the historical matched run. This rejects the specific “under-powered coupling” explanation but is not a general regularization law.

### Level C — preliminary or confounded

- The LNP final-epoch apparent improvement used a different checkpoint-selection comparison and one unseeded run.
- Historical weighted-MTRL versus mix-baseline claims changed pooling and cannot identify an architecture effect.
- Correlations between final ER edges and LOSO fold transfer are exploratory (`r=0.64` for KS↔ER versus paired `opt` delta; `n=10` non-independent folds, no held-out validation). They do not establish prediction or causality.
- No empirical single-task/pairwise transfer or task-specific gradient diagnostic exists. Claims about direction, sparsity, conflict or negative transfer remain untested.

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
| Empirical directed transfer `T(A <- B)` | none; every recorded run trains all three tasks | Unknown. Symmetry, sign and pair selectivity cannot be assessed. |
| Gradient interaction | no per-task gradient cosine, conflict rate or norm history | Unknown. Optimization conflict cannot explain the negative outcome yet. |
| Learned relation Ω | final values across five seeds at 16L/25L; full epoch histories across ten LOSO folds; legacy pooling variants | Ω is symmetric, representation-sensitive and conditionally unstable. It is not yet validated against transfer. |
| Downstream outcome | matched five-seed aggregate/task metrics and ten-fold ER LOSO | MTRL has no reproducible meaningful advantage. |

No available case can yet demonstrate “high learned similarity but negative empirical transfer” because empirical transfer has never been measured. The 25L setting does show a related discrepancy: KS↔SI is almost uniformly +1/3 while SI accuracy is neutral, so a stable strong learned relation is not sufficient for an observed performance benefit. This is an outcome discrepancy, not proof of negative transfer.

Ω is symmetric by construction, so any future asymmetric empirical transfer would immediately falsify its representational adequacy. That asymmetry has not yet been observed.

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
6. **Protocol confounding.** Unmatched pooling, seed noise, checkpoint-policy changes and speaker-leaky ER each produced apparent improvements that did not survive stronger controls.
7. **Aggregate masking.** SI’s larger test set dominates aggregate movement; task-wise trade-offs must be checked even when aggregate differences are small.

There is no evidence yet for gradient conflict, directed negative transfer, useful sparsity or data-size causation. Those must not be used as failure labels until measured.

## Active hypothesis assessment

| Hypothesis | Status | Evidence for | Evidence against / missing |
| --- | --- | --- | --- |
| MTRL meaningfully beats wavCSE | contradicted | isolated single-run/checkpoint gains | matched five-seed and LOSO results show no reproducible advantage |
| Useful transfer is asymmetric | insufficient evidence | Ω cannot represent it; asymmetry is plausible in principle | no single-task or pairwise runs exist |
| Useful relations are sparse/selective | insufficient evidence | some Ω edges are weak or unstable | weak Ω is not zero empirical transfer; no directed transfer matrix |
| Relation confidence should be distinct from magnitude | supported as a diagnostic principle; mechanism value unknown | stable saturated KS↔SI coexists with no gain; ER-edge variance is high | no confidence-aware intervention has been tested |
| ER instability is caused by smaller/noisier data | insufficient evidence | ER edges vary across LOSO folds and ER accuracy has high fold variance | no matched-data/bootstrap control |
| Relations vary by representation depth | supported descriptively, not causally | matched `smp` 16L and 25L final Ω stability patterns differ | layer selection also changes represented information; no layer-wise diagnostic |
| Relations change during training | supported descriptively, predictive value unknown | LOSO Ω trajectories differ by pair | no gradient alignment or prospective outcome prediction |
| One global structure creates negative transfer | insufficient evidence | no MTRL advantage; 25L leaky ER is lower | no empirical pairwise transfer or ablation isolating each edge |
| Current task summaries are inadequate | plausible but unproven | normalization radically changes Ω; saturation is common | no validated external relation target, class-count-invariant summary comparison or transfer correspondence |
| Ω saturation causes MTRL failure | insufficient evidence | saturation co-occurs with neutral/below-baseline outcomes | no intervention that changes saturation while holding the rest fixed; direction of causality unknown |
| Stronger λ fixes under-coupling | weakened / contradicted for tested setting | initial weak-Ω run motivated it | λ=0.05 worsened KS, SI and ER and reduced off-diagonal magnitude |

## Emerging selection framework

This is an evidence boundary, not a completed method-selection framework.

| Observable | Current KS/SI/ER evidence | Assumption it could test | Selection implication now |
| --- | --- | --- | --- |
| Transfer symmetry and sign | unknown | symmetric vs directional | unknown; measure first |
| Pair selectivity | unknown empirically | dense vs sparse/selective | unknown; Ω weakness alone is insufficient |
| Gradient cosine/conflict | unknown | parameter relation vs optimization intervention | unknown; measure after/alongside transfer |
| Relation stability | condition-specific; exact tables above | deterministic dense vs confidence-aware | confidence must be reported, but no mechanism is selected |
| Data-size imbalance | ER is smaller and noisier; causation untested | sample-aware/uncertainty-aware | unknown pending matched-data control |
| Representation sensitivity | strong | global vs representation- or layer-specific | any claimed relation must name pooling/layers; mechanism choice still unknown |
| Temporal dynamics | present in LOSO Ω | static vs dynamic | existence alone does not justify a dynamic method |
| Stable relation with no task gain | observed for 25L KS↔SI | learned covariance vs useful transfer | Ω magnitude cannot be used as a utility proxy |
| Higher-order structure | unmeasured | pairwise vs higher-order | unknown |

## Highest-value missing evidence

### What we know strongly

- Classical MTRL provides no meaningful reproducible advantage under matched five-seed aggregate evaluation or speaker-independent ER LOSO.
- Ordinary-split ER, single seeds and unmatched pooling can each create false architecture narratives.
- Learned Ω is not task-intrinsic: its strength and stability depend on representation and perturbation axis.
- Within 25L LOSO, KS↔SI is fold-stable while ER edges are fold-sensitive; this ordering does not generalize to all seed/layer settings.

### What appears likely but remains uncertain

- The mean-pooled, normalized task-head summary is contributing to Ω saturation and instability.
- Stable Ω can encode parameter geometry without encoding useful transfer.
- ER’s sample regime contributes to relation uncertainty.

### What is contradicted

- MTRL has a confirmed performance win.
- Stronger λ simply amplifies a useful relation.
- A single learned Ω can be interpreted as a pooling-, layer-, seed- and split-independent truth about KS, SI and ER.

### Single highest-information experiment

**DG-0001 Stage A: a matched single-task/pairwise baseline matrix at one explicit screening seed, with a matching triple-task control.** It directly measures all six directed cells `T(A <- B)` and tests the central unresolved assumption—whether useful transfer is symmetric and dense—without introducing another architecture. Run three single-task, three pairwise and, unless an exact Study-tagged/current-commit triple arm exists, one triple-task control. ER cells remain screening-only until decisive entries are repeated under LOSO.

This has higher information gain per GPU-hour than more MTRL tuning because every possible next explanation branches on its result:

- all cells near zero → investigate saturation/regularization rather than relation representation;
- symmetric dense transfer → symmetric Ω is representationally adequate; diagnose learning/optimization;
- asymmetric or selective transfer → classical Ω is structurally inadequate and targeted literature research becomes justified;
- transfer exists but Ω disagrees → validate the parameter-summary/update path before selecting a replacement.

### Minimum diagnostic sequence before searching for the next method

1. **Prerequisite smoke test, not a Study result:** exercise single-task and pairwise `task_type` paths on tiny data.
2. **DG-0001 Stage A:** measure the directed empirical transfer matrix under exact matched controls.
3. **Immediate analysis against existing five-seed triple-task Ω.** Do not automatically run all pairwise MTRL arms. If transfer is asymmetric, symmetry is already falsified mathematically. If transfer is selective or symmetric, run only the minimum DG-0003 arms needed to determine whether Ω fails to represent it.
4. **DG-0002 only on the unresolved branch:** measure task-specific gradient norms/cosines/conflict when Stage A shows no useful transfer or when Ω tracks transfer but performance remains neutral.
5. **Targeted literature search only after one branch establishes a concrete limitation.** Search terms must name that measured limitation.

No new method search is justified yet. Existing evidence proves conditionality and no performance gain, but not whether the decisive failure is symmetry, density, parameter summarization or optimization.

## Concise synthesis

1. **Strongest conclusion:** classical MTRL is outcome-neutral and its Ω is a condition-dependent parameter-geometry artifact, not yet a validated transfer map.
2. **Most important uncertainty:** the project has never measured directed empirical transfer or task-gradient interaction, so it cannot identify which MTRL assumption fails.
3. **Next scientific question:** is KS/SI/ER transfer symmetric and dense, and does existing Ω agree with the six directed transfer cells?
4. **Literature:** a search for the next method is premature; it becomes mandatory once DG-0001/DG-0003 or the conditional DG-0002 identifies the concrete failure mode.
