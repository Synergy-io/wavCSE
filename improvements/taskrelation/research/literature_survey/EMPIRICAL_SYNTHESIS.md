# Empirical synthesis — DG-0001 and DG-0002

> **Status note (2026-09-29, integration).** This document was written against checkout
> `664c572` / `9955166`, before the TR-0007 screen closed and before DG-0007 was registered. Its
> reasoning, derivations and predictions are unchanged and none of its claims is withdrawn; only
> sentences that report a *corrected seed-42 TR-0007 result*, a *DG-0007 record*, or an *edge-support
> reading* as absent **in this checkout** are stale, and they are preserved as that checkout's
> history. Since then: the screen closed `REJECTED` at the researcher-fixed `λ₂ = 0.01` (dense
> support, all three partial correlations `+0.4911`), DG-0007 is registered and pre-registered with
> **no result**, and the post-TR-0007 reconciliation — including the exact-solve reading that makes
> the published `λ₂` axis empty of a scale fix — is
> [`POST_TR0007_SYNTHESIS.md`](./POST_TR0007_SYNTHESIS.md) (§5 lists every stale sentence by file and
> line). Source: `research/taskrelation-literature` `9955166`, `a53f29d`, integrated here.


**Evidence labels:** **OBSERVED** = number read from existing JSON; **DERIVED** = explicit calculation from those numbers; **HYPOTHESIZED** = prediction not measured. This is an analysis of completed records, not a new study or result. Data sources: [DG-0001 screen](../../../../improvements/taskrelation/research/task_relations/empirical_transfer.json), [raw LOSO](../../../../improvements/taskrelation/research/task_relations/loso_transfer.json), [Stage-D optimizer controls](../../../../improvements/taskrelation/research/task_relations/optimization_control.json), [DG-0001 full result](../../../../improvements/taskrelation/research/studies/DG-0001/result.json); [DG-0002 seed-level confirmation](../../../../improvements/taskrelation/research/studies/DG-0002/confirmation_result.json), [DG-0002 initial screen](../../../../improvements/taskrelation/research/studies/DG-0002/result.json). Interpretive checks: [DG-0001 analysis](../../../../improvements/taskrelation/research/studies/DG-0001/analysis.md), [DG-0002 analysis](../../../../improvements/taskrelation/research/studies/DG-0002/analysis.md), [FINDINGS F8–F10](../../../../improvements/taskrelation/research/FINDINGS.md). `SI` in records = `SID` here. All performance deltas are **accuracy fractions**, not percentage-point numerals.

## Recorded study conditions and limits

- **OBSERVED DG-0001 Stage A:** seed 42, fixed-final epoch, `smp(0.5)`, all 25 layers, **one** fit per task combination (commit `4c0f08a…`). Ordinary ER split is speaker-leaky. Raw `T(target←auxiliary)` changes task count, training steps, task sampling and scheduler path; it is *operational* pair-minus-single accuracy, not causal task information.
- **OBSERVED Stage C/D:** one seed 42, ten ER speaker-held-out folds, five epochs/fold. Raw Stage C (`330d0f7…`) contrasted ER-only batch 2048 (~2 steps/fold/epoch) with KS+ER (~27) and SI+ER (~69). Stage D (`3f4752a…`) added ER-only batch 160 (~26–28) and 64 (~67–70) update-count controls plus five-epoch KS/SI single-task references. Batches are only **approximately** exposure matched: changed noise and effective per-task batch size are not perfectly equalized. Reverse KS/SID references are *one seed-42 fit each*, not ten independent single-task fits.
- **OBSERVED DG-0002:** five paired baseline/MTRL seeds 0–4 (commit `7f6d5248…`), three-task 25-layer `smp(0.5)` 30 epochs, same 142 sampled steps of 2820, same 1,550,800 shared parameters and valid-example counts within each seed pair; seed-level *phase* summaries are the units, not 142 independent observations. Ordinary ER split: no generalization claim. This measures gradients of each task's loss on shared projector/hidden parameters **in a joint training arm**, not directed interventions or conditional parameter dependence.

## All six operational directed cells and controls

**OBSERVED** Stage-A epoch data from `empirical_transfer.json`; Stage-C raw LOSO from `loso_transfer.json`; Stage-D column from `optimization_control.json`. `—` means *not measured*, not zero. Intervals describe fold variation, not seed replication. Seed throughout = 42. Pair identity and checkpoint matter: Stage-A ordinary ER numbers are not directly comparable to Stage-D LOSO/five-epoch numbers.

| Target ← auxiliary | Stage-A raw (30 epochs, ordinary) | Stage-C raw ER LOSO (5 epochs) | Stage-D exposed-matched or reverse sensitivity (5 epochs) | Evidence tier |
| --- | ---: | ---: | ---: | --- |
| KS ← SID | −0.000585 | — | — | one-seed raw; exposure uncontrolled |
| SID ← KS | −0.001697 | — | — | one-seed raw; exposure uncontrolled |
| KS ← ER | +0.000146 | −0.005999 raw vs earlier single reference | −0.003804 vs **new** five-epoch KS reference, 10/10 folds negative; CI [−0.005003, −0.002605] | reverse sensitivity; one single-task KS ref, not replicated matched transfer |
| ER ← KS | +0.057866 (leaky) | +0.341113, 10/10 folds; CI [+0.284690,+0.397535] | **+0.005289** pair − step-matched ER-only; CI [−0.028855,+0.039433], 5+/5− | raw gain dominated by exposure; residual unresolved |
| SID ← ER | −0.008847 | −0.056308 raw vs earlier single reference | −0.023100 vs **new** five-epoch SID reference, 10/10 folds negative; CI [−0.027475,−0.018725] | reverse sensitivity; one single-task SID ref |
| ER ← SID | +0.028933 (leaky) | +0.279413, 10/10 folds; CI [+0.211167,+0.347659] | **−0.058921** pair − step-matched ER-only; CI [−0.088313,−0.029529], 0+/10− | moderate adverse residual with approximate matching, not multiple seeds |

**DERIVED** from recorded Stage-D components, to six decimals: ER←KS raw `+0.341113 = +0.335824 exposure +0.005289 residual`; ER←SID raw `+0.279413 = +0.338334 exposure −0.058921 residual`. These decompositions refute the original large positive ER-directed interpretation. Reverse controls do not match every feature of target-only optimization; neither negative reverse cell proves intrinsic task hostility. Critically, comparing an ER-target LOSO accuracy delta to a SID-target accuracy delta as if they shared an accuracy scale is not a causal direction estimate; paired, target-calibrated outcomes and fully matched references are needed. See [DIRECTIONAL_EVIDENCE.md](DIRECTIONAL_EVIDENCE.md).

## Shared-gradient observations by phase

**OBSERVED** `confirmation_result.json.cross_seed.gradients`; means across **five seed-level summaries**, not a pooled sample over steps. `c/f` = mean pairwise cosine / negative-cosine frequency for shared-parameter task gradients. Cosine is **symmetric**: `cos(g_A,g_B)=cos(g_B,g_A)`.

| Method / phase | KS--SID c/f | KS--ER c/f | SID--ER c/f | mean norms KS / SID / ER | max/min norm ratio |
| --- | --- | --- | --- | --- | ---: |
| baseline early | +0.0482 / .2250 | +0.0261 / .3083 | +0.1218 / .1542 | 4.644 / 4.571 / 12.800 | 2.860 |
| baseline middle | +0.0109 / .3745 | +0.0143 / .3957 | +0.0500 / .2383 | 1.046 / 1.277 / 7.578 | 7.243 |
| baseline late | +0.0022 / .4638 | +0.0013 / .4255 | +0.0221 / .3574 | .828 / .804 / 6.972 | 8.962 |
| MTRL early | +0.0484 / .2667 | +0.0281 / .3083 | +0.1284 / .1083 | 4.665 / 4.493 / 12.622 | 2.867 |
| MTRL middle | +0.0113 / .3830 | +0.0154 / .3660 | +0.0547 / .2553 | 1.134 / 1.360 / 8.023 | 7.129 |
| MTRL late | +0.0024 / .4468 | +0.0039 / .4851 | +0.0196 / .3745 | .840 / .811 / 6.834 | 9.004 |

**OBSERVED:** no seed meets the pre-registered persistent pair-conflict criterion in the baseline. Baseline **late seed cosine values** (order seeds 0–4) are KS--SID `[+.0009,−.0038,+.0039,+.0096,+.0005]`, KS--ER `[−.0101,+.0082,+.0064,+.0116,−.0096]`, SID--ER `[+.0162,+.0343,+.0349,+.0115,+.0137]`. Late norms are near-orthogonal interactions, not persistent antiparallel gradients. Middle/late ER norm dominance is genuine for the *standard* sampler, but [DG-0005](../../../../improvements/taskrelation/research/studies/DG-0005/analysis.md) then showed it is mixture-conditioned: ER share ~47→~435/2048 reduced baseline late ratio 8.962→2.352 in all five matched seeds, without a conflict signal. Thus norm dominance neither measures pairwise affinity nor identifies a relation method.

## Classical Ω versus the three pairs

**OBSERVED** `confirmation_result.json.cross_seed.omega.final_pair_values` at 25L (seeds 0–4): KS--SID `[+.33324,+.33320,+.33303,+.33276,+.33316]`; KS--ER `[+.33317,+.33301,+.33328,+.33259,−.33323]`; SID--ER `[+.33307,+.33304,+.33308,+.33259,−.33316]`. All *magnitudes* near one-third; seed 4 flips both ER edges. This is the historical normalized MTRL **head-parameter covariance**, not p-MSSL precision or empirical transfer, and saturation is not proven to cause the accuracy null. Different pooling/layer axes produce different Ω stability ([MTRL synthesis](../../../../improvements/taskrelation/research/task_relations/MTRL_DIAGNOSTIC_SYNTHESIS.md) §§Relation stability, Dynamic behavior).

| Undirected pair | Marginal *parameter* association evidence | Useful transfer evidence | Conditional dependence evidence | Optimization compatibility |
| --- | --- | --- | --- | --- |
| KS--SID | **OBSERVED** 25L MTRL Ω ~+.3331 in 5/5 seeds, saturated; no independent population covariance estimated. | Stage-A directions −.000585 / −.001697 raw, seed42, no controlled residual. **Not established either way**. | **Absent:** no validated precision, partial correlation, or edge support from current TR-0007 screen in this checkout. | **OBSERVED** baseline early +.0482/.2250 → late +.0022/.4638. Near orthogonal late, not a directional finding. |
| KS--ER | **OBSERVED** 25L Ω positive 4/5 and negative 1/5; 25L LOSO MTRL edge mean +.098±.072 across folds (different condition). | ER←KS step-matched residual +.005289 CI spans zero; KS←ER −.003804 reverse sensitivity. No confirmed helpful pair. | **Absent** as above. | **OBSERVED** baseline early +.0261/.3083 → late +.0013/.4255; neither sign indicates transfer direction. |
| SID--ER | **OBSERVED** 25L Ω positive 4/5, negative 1/5; LOSO MTRL edge +.033±.103, fold-sensitive. | ER←SID residual −.058921 (10/10 negative with approximate exposure control); SID←ER −.023100 reverse sensitivity; suggest adverse coupling, not a multi-seed result. | **Absent** as above. | **OBSERVED** baseline early +.1218/.1542 → late +.0221/.3574; weakly positive cosine coexists with adverse performance residual under **different protocols**. |

**Scientific bound:** An Ω sign/magnitude, a precision edge, a cosine, and a target-accuracy delta describe *different estimands*. Triple-task covariance vs pairwise single-task intervention also differ in training distribution; do not correlate their three entries as if they were independent task samples. [RELATION_SEMANTICS.md](RELATION_SEMANTICS.md), [IDENTIFIABILITY.md](IDENTIFIABILITY.md), [EXPERIMENT_ROADMAP.md](EXPERIMENT_ROADMAP.md).
