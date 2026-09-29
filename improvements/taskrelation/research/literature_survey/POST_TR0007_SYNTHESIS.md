# Post-TR-0007 scientific synthesis — what to test next for useful task transfer

**Status:** synthesis and proposal, written *after* the TR-0007 screen closed. It registers no
study, creates no authorization, requests no compute, and changes no pre-registration. TR-0007's
pre-registration (`studies/TR-0007/PLAN.md`, `DEC-0015`/`DEC-0016`) and its `REJECTED`
classification are untouched.

**Written:** 2026-09-29, against canonical `feature/mssl-task-relation-study` at `90b73b2`, with
the literature survey integrated from `research/taskrelation-literature` (`9955166`, `a53f29d`)
and the MTRL theory audit already in canonical (`e695167`).

**Evidence labels** (the convention the survey documents use): `OBSERVED` = a number read from a
recorded artifact or a measurement of one; `DERIVED` = algebra or an exact solve performed here,
shown; `HYPOTHESIZED` = a falsifiable prediction, unmeasured; `RECORD` = a statement about the
record itself.

**Inputs.**

| Source | What it contributes |
|---|---|
| `studies/TR-0007/{analysis.md,result.json,screen_result.json,screen_commits.json}` | the screen's result, caveats and recorded mechanism |
| `studies/TR-0007/{analyze_coupling_scale.py,coupling_scale_result.json}` | post-hoc, read-only scale reading of the same artifacts (appended to `analysis.md` 2026-09-29) |
| `literature_survey/` (18 documents) | the literature survey, estimand semantics, identifiability preconditions and the pre-registered predictions `P1–P8` |
| `FINDINGS.md` F1–F10, `FAILURES.md` FL-0001–FL-0005 | established evidence and negative evidence |
| `task_relations/MTRL_DIAGNOSTIC_SYNTHESIS.md`, `FRAMEWORK.md`, `VARIANT_BENCHMARK_PROTOCOL.md` | MTRL diagnostics, the selection framework, the frozen protocol |
| `DECISIONS.md` `DEC-0001…DEC-0019`, `STUDIES.jsonl`, `STATE.md` | binding decisions and registry state |
| DagsHub runs `a82144d2` (p-mssl), `f7550613` (p-mssl duplicate), `1e406945` (classical-mtrl), `641b25ab` (baseline) | verified metrics and the checkpoints the scale reading uses |

---

## 1. The answer in one paragraph

TR-0007 did not show that sparse precision is a poor relation object. It showed that at the
researcher-fixed `lambda_2 = 0.01` the published p-MSSL arm is **dominated by its own relation
penalty**: the coupling term is a sum over the 2001 summary coordinates (`≈ 3d` in value, `≈ 1e5`
in gradient) while the adapted task loss is a batch mean (`≈ 0.5`; task gradients `≈ 1e-2`), and
the coupling's magnitude is **invariant along every hyperparameter the paper publishes** — `λ₂`
moves the support, not the scale, and lowering `λ₀` only raises the ℓ₁'s relative weight while the
gradient stays `≥ 20`. The summary-geometry collapse the screen recorded (`cos ≈ 0.981`, `S`
near-singular, `Ω` trace `1.13e5`) is therefore a **consequence** of that domination, not its
cause: the *uncoupled* matched baseline ends the same protocol with the three head summaries
nearly orthogonal (cosines `+0.211, +0.031, +0.023`), and the candidate's own pre-activation
geometry was in that same healthy band one epoch earlier. The next step is consequently **not**
another relation matrix, another λ, or a better summary: it is one pre-registered test of whether
the *estimated* relation helps when it is imposed at a magnitude the optimizer can weigh against
the task loss — with the faithful alternative (`λ₂`) first eliminated as a cheap protocol
correction, and with the deviation-class intervention gated on a human faithfulness decision.

---

## 2. Consolidated empirical evidence (what we actually know)

### 2.1 The screen (`OBSERVED`, screening tier: one seed, one split)

| arm | commit | `test_epoch_acc_all` | ks | si | er |
|---|---|---|---|---|---|
| p-MSSL (candidate) | `d13e82e` | **0.9662** | 0.9867 | 0.9615 | 0.7848 |
| classical MTRL (in-category control) | `664c572` | 0.9752 | 0.9855 | 0.9794 | 0.7848 |
| matched wavCSE baseline | `664c572` | 0.9748 | 0.9861 | 0.9782 | 0.7848 |

Paired deltas (candidate − control, pp): vs MTRL `−0.90` aggregate, `+0.12` ks, `−1.79` si, `0.00`
er; vs baseline `−0.86`, `+0.06`, `−1.67`, `0.00`. The pre-registered rule (`PROMISING` iff above
both controls with no task beyond `0.20pp`) gives **`REJECTED`**, applied unchanged. All three arms
sit at `0.7848` on ER, which is the speaker-leaky ordinary split (F3) and is not an ER result.
Compute: one worker, 2.96 paid hours, `$0.7404`. The candidate ran twice with identical checkpoint
bytes and metrics to five decimals.

`RECORD`: the identical numbers are on the tracking store (`a82144d2` = 0.966238, `1e406945` =
0.975190, `641b25ab` = 0.974807), so the repository's `result.json` is a faithful rendering of the
runs, not a transcription.

### 2.2 The recorded mechanism (`OBSERVED`, as recorded)

From the candidate's `omega_history.json` and the analysis: the coupling engages at epoch 3→4, the
task summaries align to `cos ≈ 0.981`, `S` becomes near-singular, `Ω` jumps to a near-uniform
precision of trace `1.13e5`, the reported loss is dominated by the coupling term (`≈ 2015` against
`≈ 0.5` of task losses), and training plateaus from epoch `≈ 10`. The Ω snapshots also report the
ADMM's own residual limit at that conditioning (relative duality gap `−0.054` at its 2000-iteration
cap) rather than hiding it. The Ω step is verified against closed forms and the problem's own
optimality certificate, and no protocol field moved: this is not a solver defect.

### 2.3 The post-hoc scale reading of the same artifacts (`OBSERVED` measurement + `DERIVED` algebra)

From `analyze_coupling_scale.py` / `coupling_scale_result.json` (appended to
`studies/TR-0007/analysis.md`):

* **Geometry by arm, at each arm's `best` checkpoint.** p-MSSL cosines `+0.99989, +0.99991,
  +0.99990` (Gram numerically rank-1); classical MTRL `+0.99822, +0.99932, +0.99827` (rank-1);
  matched baseline `+0.21067, +0.03061, +0.02339` (Gram eigenvalues `0.787, 0.993, 1.215`,
  condition `1.54`). The record's own pre-activation cosines for the candidate are `+0.261,
  +0.035, +0.009`.
* **The stored `Ω` reproduces the record exactly**: trace `113 437.98` (`1.13e5`), off-diagonal
  `−18 570.4` vs diagonal `37 812.7` (mean `|off| 18 571`), **0 exact off-diagonal zeros**, partial
  correlations `+0.4911, +0.4911, +0.4911` — all three identical, at the `m = 3` equicorrelation
  ceiling `ρ = r/(1+r) ≤ 1/2` derived in `MSSL_SPARSITY_ANALYSIS.md` §3.2.
* **The coupling's scale.** Its value is `tr(ΩR)`, exactly `3d` at the unpenalized optimum
  (`Ω = S⁻¹`) and `≈ 5993` for the penalized fit at the healthy (epoch-3) geometry, so at the
  published `λ₀ = 1` the coupling is `O(d)` against a batch-mean task loss `≈ 0.5` (the record's
  `≈ 2015` is this quantity at the collapsed geometry, where the ℓ₁ holds the fit below the
  unpenalized optimum); its gradient `2λ₀ΩW` is `≈ 6.9e3` even for `Ω = d·I` — `≈ 7e5`× the
  recorded task-gradient scale — rising to `1.6e5` at the collapsed geometry.
* **The published `λ₂` axis cannot move it.** Exact Eq. (8) solves with this repository's own
  `graphical_lasso_admm` at each measured Gram: `λ₂ = 0.01` → 0 zeros, gradient `1.6e5`; `0.1` → 0
  zeros, `5.6e4`; `≥ 1` → 3 zeros (all conditional edges gone), `6.9e3`. On the uncoupled
  baseline's Gram: `0.01` → 0 zeros / `7 240`; `0.1` → 2 zeros / `7 029`; `≥ 1` → 3 zeros /
  `6 942`. On the recorded epoch-3 geometry: `0.01` → 1 zero / `7 407`; `0.1` → 2 zeros / `7 117`.
  Every value of the published grid leaves the coupling `≥ 7e3`, i.e. `≥ 7e5`× the task scale, and
  the value that removes all coupling still carries `3d` on its diagonal.
* **The `λ₀` axis.** `λ₀ = 1e-4` lowers the coupling gradient to `21.7` (still `≈ 2e3`× the task
  scale) while **all three edges are already zero** — the reduction comes from the ℓ₁'s relative
  weight growing, not from a scale fix.

### 2.4 The controls' own evidence (unchanged, for context)

F1 single-seed improvements are not trustworthy; F4/F7 classical MTRL has no reproducible advantage
and its `Ω` saturates to the rank-1 `±1/3` boundary in 5/5 seeds at 25L `smp` (the measured
classical-MTRL checkpoint here, cosines `≈ 0.9983`, is that boundary seen from the parameter side);
F3 the ordinary ER split inflates ER by `≈ 15pp`; F8 optimizer exposure dominates the raw DG-0001
transfer matrix; F10 gradient scale is a training-mixture property, so F9 cannot justify a
scale-aware relation mechanism.

### 2.5 Registry state of the neighbouring studies (`RECORD`)

* **TR-0008** (directed relation, project-original): named and activated by `DEC-0014` §5, **not
  registered**; its exact relation rule is design work the autonomy policy classes as introducing a
  project-original mechanism, so it needs either an explicit authorization to design it
  autonomously or a design review. Unchanged by this synthesis.
* **DG-0007** (normalization-corrected MTRL): registered, pre-registered, **no result exists**.
  A read-only search of every experiment on the tracking store finds **no DG-0007 run**, and
  `STATE.md`, `STUDIES.jsonl` and the study's `NOTE.md` all say the screen did not execute. Commit
  `2e17c21`'s message says a narrowed two-arm screen was run (`DEC-0019`) and no result of it is in
  the repository. **This is a record conflict and is left on record rather than smoothed over:**
  DG-0007 has no evidence, and nothing in this synthesis treats it as having any.
* **`DECISIONS.md`** was restored from `8fb395b` in this integration after `2e17c21` truncated it
  to one blank line, with `DEC-0019` appended from that commit's message (provenance stated in the
  entry).

---

## 3. Three questions that must not be conflated

The survey insists that an `Ω` sign, a precision edge, a partial correlation, a cosine and an
accuracy delta are different estimands. The same separation applies to *quality*, *stability* and
*usefulness*.

**(1) Relation-estimation quality — did the estimator return a well-posed relation object?**
Partly yes, partly unresolvable. `OBSERVED`: the object is positive definite, dense at `λ₂ = 0.01`,
its three partial correlations are identical to four decimals (`+0.4911`), and the support is
`{ks·si, ks·er, si·er}`. `DERIVED`: the object is not a *discriminating* fit — `Ω ≈ d·R⁻¹` with a
rank-1 `R` means the fit is the equicorrelated family, whose members differ only by a scale that
the `−log|Ω|` barrier and the ℓ₁ set; and the three identical partial correlations sit exactly at
the `m = 3` ceiling. So the estimator did what the objective says; there is no pair information in
its input to return, because the input geometry was destroyed by the coupling. `RECORD`: at the
`m = 3` equicorrelation ceiling the *identifiability* preconditions `P-i…P-iii` of
`IDENTIFIABILITY.md` were never reachable here — the run never got past the scale failure.

**(2) Optimization stability — was the run a training of the tasks or of the regularizer?**
No: `DERIVED` and `OBSERVED` agree that the relation penalty outweighed the task loss by `≈ 4e3`
in value and `≈ 1e7` in gradient, that this is invariant along the published axes, and that it is
what produced the plateau. The screen is therefore *not* a test of the relation object's
usefulness, and its `−0.90pp` cannot be attributed to the estimator's semantics.

**(3) Downstream usefulness — is there any evidence that a learned relation improves useful
transfer?** No, at any tier. `OBSERVED`: MTRL at 25L `smp` is a matched null (F4); its `Ω` is
rank-1/uninformative (F7); p-MSSL's screen is `REJECTED` under a scale failure (FL-0005 plus this
reading); DG-0001's apparent directed gains are optimizer exposure (F8). `HYPOTHESIZED`: nothing in
the record yet shows a learned relation improving a *controlled* target metric; and per
`HYPOTHESIS_GRAPH.md`, no traversal from a symmetric cosine or a precision zero to a transfer claim
is licensed. The honest programme-level statement is that the *link* between relation estimation
and useful transfer remains unmeasured — and now has a demonstrated prerequisite: the penalty must
be imposable at a magnitude that does not swamp the task loss.

---

## 4. Where TR-0007's failure actually came from — the A–E question

| Candidate cause | Verdict | Evidence |
|---|---|---|
| **A. relation-estimator semantics** (the precision is the wrong object; sparse/conditional is wrong for `m=3`) | **Not implicated.** The estimator solved the published problem; the failure is outside the estimator. | `OBSERVED`: the stored `Ω` satisfies the recorded trace/off-diagonal magnitudes and the interior optimum; `DERIVED`: the ℓ₁ at `λ₂ = 0.01` is inert (off-diagonals `1.9e4` against an effective threshold `λ₂/d = 5e-6`), so the "sparse precision" reading never got a chance to matter. Falsification of the earlier `H-sparse` reading is not even reached: the run never tested sparse edges, because none were selected at this `λ₂` at this scale. |
| **B. summary-representation geometry** (the class-mean adapter is degenerate by construction) | **Refuted as the primary cause; it is a consequence.** The uncoupled baseline ends the identical protocol with nearly orthogonal summaries; the candidate's pre-activation geometry was equally healthy. | `OBSERVED` §2.3: cosines `0.211/0.031/0.023` (baseline) vs `0.9999` (coupled arms); the record's epoch-3 cosines `0.261/0.035/0.009`. |
| **C. relation-loss scale** | **Primary.** The coupling's magnitude is `O(d)` against a batch-mean loss and is invariant along `λ₀` and `λ₂`; the domination is what drives the alignment/conditioning loop. | `DERIVED` §2.3: coupling `≈ 3d` in value against `≈ 0.5`; gradient `≥ 6.9e3` at every published `λ` against `≈ 1e-2`. |
| **D. coupling schedule** | **Not a fix; at most a delay.** The loop starts when the coupling engages, and the coupling is already `≈ 7e5`× the task gradient at the healthy pre-activation geometry, so any activation epoch starts it. | `DERIVED` §2.3 (epoch-3 geometry row); `OBSERVED` (collapse within one epoch of activation: epoch 3 healthy → epoch 4 rank-1). |
| **E. combination** | **C is the root; B is its first consequence; D is timing.** Only C is an independent variable worth changing first. | As above. |

`RECORD`, stated plainly because it is a refinement of the study's own wording: the analysis file
attributes the mis-scale to the researcher-fixed `λ₂ = 0.01` and says the arm is `REJECTED` "at
that fixed `λ₂`". That remains true, and the `λ₂` caveat was pre-registered. What the reading adds
is that `λ₂` is **not the quantity that could have rescued the arm**, so the open question the
screen leaves should be re-scoped from "which `λ₂`" to "which normalisation of the coupling term
against the loss" — a deviation-class question, not a hyperparameter question.

---

## 5. Reconciliation with the survey's pre-registered predictions

The survey was written against checkout `664c572`/`9955166`, i.e. **before** the screen ran, and it
says so. Nothing below rewrites it; each row is what the integration can now say about it.

| Pre-registered item | Prediction as written | What the record now shows |
|---|---|---|
| `MSSL_PREDICTIONS.md` P1 (support) | at `λ₂ = 0.01` the precision may be dense | **Confirmed** (`OBSERVED`): 0 exact zeros, dense support. The falsifier (any exact zero) did not occur. |
| P2 (scale diagnostic) | read conditional Gram residual, raw `Ω`, partial correlations, ordering | **Partially available** (`OBSERVED`): `Ω`, partial correlations and exact support are now readable from the stored artifact + checkpoint; the *per-epoch* `R` trajectory is not stored. |
| P3 (causal gate) | a zero edge alone cannot establish benefit | **Standing**; no zero edge occurred and no edge intervention exists. |
| P4 (estimator contrast) | dense precision need not behave like MTRL | **Refuted in the strong form** (`OBSERVED`): both coupled arms end rank-1 — `cos ≈ 0.9983` (MTRL) vs `≈ 0.9999` (p-MSSL) — so the two arms reached the *same* degenerate geometry by different routes, exactly as the `Ω = Σ⁻¹` bijection argument in `MSSL_SPARSITY_ANALYSIS.md` §1.2 implies. Their *outcomes* still differ (MTRL null, p-MSSL `−0.90pp`), which the scale difference between their coupling matrices can explain but does not yet establish. |
| P5 (solver validity) | finite-objective precision is PD, no exact rank-1 | **Confirmed** (`OBSERVED`): `Ω` finite, eigenvalues `671.5, 56 383, 56 383`. Note the ADMM's own 2000-iteration cap at that conditioning (gap `−0.054`), recorded per snapshot. |
| P6 (input convention) | inspect the Eq. (8) scaling identity with unit rows and `d = 2001` | **Confirmed and extended** (`DERIVED` §2.3): with unit rows the arm's objective is the paper's standardised graphical lasso at `λ₂` — and that same identity is why `Ω = O(d)` and the coupling is `O(d²)` in the unnormalised view. |
| P7 (outcome reporting) | report per-task effects beside the aggregate and both controls | **Done** (`OBSERVED`): SI `−1.79/−1.67pp` is the regression; the aggregate alone would have masked it. |
| P8 (historical context) | seed-0 historical runs cannot classify this screen | **Standing**; they were not touched or substituted. |
| `IDENTIFIABILITY.md` P-i (sampling model) | head-summary coordinates would need to support an i.i.d.-Gaussian reading | **Still unverified**; the collapse means the fitted object never answered the question. |
| P-ii (stable support) | support stability under independent reruns with a declared penalty | **Now partially answerable and negative for this run**: one run only, and at this scale the `λ₂` threshold is inert, so support "stability" would be trivially stable (all edges). |
| P-iii (replicated interpretation) | repeat fits across seeds; LOSO for ER | **Not measured**; one seed, one split. |
| `EXPERIMENT_ROADMAP.md` M1 (DG-0007) | observe the corrected-MTRL arm when available | **Blocked**: DG-0007 has no result (§2.5). |
| M2 (read the TR-0007 screen) | is the precision sparse-conditional or merely dense? | **Answered at the estimator level**: dense, and inert-ℓ₁; the scale failure makes it uninformative about the estimand question. |
| M3 (one controlled pair's semantic transfer) | only if candidates diverge on a pair | **Blocked by M1/M2 as originally scoped**; §8 proposes what replaces it. |
| `MSSL_SPARSITY_ANALYSIS.md` §5.2 | transition to exact zeros at `λ₂ ≈ 0.2–0.5`; `λ₂ = 0.01` shrinks only marginally | **Confirmed and sharpened** (`DERIVED`): at the measured Grams the transition is at `λ₂` between `0.1` and `1`, and `λ₂ = 1` already removes all three edges. |
| `HYPOTHESIS_GRAPH.md` `H_dense` | corrected normalization reveals meaningful dense structure | **Not tested**: DG-0007 has no result, and the p-MSSL arm's dense structure was uninformative. |
| `HYPOTHESIS_GRAPH.md` `H_sparse conditional` | a stable absent edge aligns with a harmful pair | **Not reached**: no absent edge, no intervention, and the "screen at `λ₂ = 0.01` cannot test edge suppression" caveat the survey itself stated is exactly what happened. |

**Stale checkout-specific statements corrected by addendum** (each affected document carries a
dated `Status note (2026-09-29, integration)` at its head; the sentences themselves are preserved,
because they were true of the checkout they describe):

| Document | Sentence that is now stale | Now |
|---|---|---|
| `MSSL_PREDICTIONS.md` §0 L37-44 | "no corrected screen result or analysis artifact in **this checkout**" | result, analysis and mechanism all on the record |
| `MSSL_PREDICTIONS.md` §3 L125 | "None of these has been produced by this checkout" | bundle rows 1-4, 6, 7, 8 now exist for the candidate run |
| `MSSL_PREDICTIONS.md` §9 L268 | "**No corrected seed-42 screen result is present in this checkout.**" | present, and additionally re-read from the stored checkpoint |
| `MTRL_PREDICTIONS.md` L3, L33 | "no `DG-0007` entry or result; no `studies/DG-0007/` directory is present" | DG-0007 is registered and pre-registered with a full study folder — and still has **no result** |
| `EMPIRICAL_SYNTHESIS.md` L47-49 | "**Absent:** no validated precision, partial correlation, or edge support from current TR-0007 screen in this checkout" | support and partial correlations are measured: dense, all three `+0.4911` |
| `IDENTIFIABILITY.md` L71-72 | "**not yet measured** for the corrected TR-0007 screen"; "**absent for corrected TR-0007 in this checkout**" | measured (one seed); still not *replicated* |
| `RELATION_SEMANTICS.md` L130 | "registered TR-0007; corrected screen result absent in this checkout" | screen closed `REJECTED`; the estimand itself was never discriminatively tested |
| `EXPERIMENT_ROADMAP.md` L3 | "The present checkout has **no DG-0007 result, no corrected TR-0007 screen result and no TR-0008 registered result**" | TR-0007 has a result; DG-0007 and TR-0008 still have none |
| `OPEN_QUESTIONS.md` Q8 | "What is the latest actual status of DG-0007 / TR-0007 / TR-0008?" | answered in §2.5 of this synthesis, including the DG-0007 conflict |
| `DIRECTIONAL_PREDICTIONS.md` L3, `HYPOTHESIS_GRAPH.md` | "`TR-0008` not registered here" | unchanged and still correct |

---

## 6. Candidate families considered, and how each is classified

The brief's six families and the survey's own candidates, each judged on this evidence. "Identifiable
now" means the recorded evidence already fixes what the variable is and what a result would mean.

| Candidate | Classification | Reason |
|---|---|---|
| **1. magnitude-controlled relation regularization** | **Identifiable now.** Primary successor; needs a human faithfulness call because it changes a published objective's normalisation. | `DERIVED` §2.3 shows the coupling's magnitude is the binding defect and is not accessible through `λ`; the intervention has a single, pre-registerable definition (§8, S1). |
| **2. better task-summary representation** | **Deferred — predicted ineffective here, and confounded.** | `OBSERVED` §2.3: the uncoupled baseline's summaries are already well-conditioned, so the collapse is coupling-induced; changing the adapter changes the input to *both* the candidate and the in-category control, i.e. it is a representation variable, not a relation variable (F2/F5 warn that representation effects dominate), and with the coupling's scale untouched the penalty would still swamp the loss. |
| **3. relation-confidence / conditioning gate** | **Deferred — predicted inert or self-defeating.** | `DERIVED`: the pre-activation geometry is healthy, so a conditioning-keyed gate would not fire at onset; a gate keyed on the coupling-to-task gradient ratio would fire immediately and leave the arm equal to the baseline (no relation test). Its *measurement* is worth having — and is included in S1 as instrumentation, not as a mechanism. |
| **4. phase-dependent coupling** | **Deferred — not a fix for the measured mechanism.** | `DERIVED` §2.3: the domination is present at the healthy pre-activation geometry too, so later activation delays the collapse rather than preventing it. Cheap to falsify as a *side observation* inside S1 (activate at the same epoch; record the ratio), but it must not be a second changed variable in the same arm. |
| **5. covariance vs precision** | **Deferred — redundant/not discriminating now.** | Both families couple through a matrix whose magnitude is set by the same `S` normalisation (`tr(Ω⁻¹,R)` for a trace-1 covariance, `tr(Ω,R)` for a precision), and `MSSL_SPARSITY_ANALYSIS.md` §1.2 shows `Ω = Σ⁻¹` is a bijection; the recorded evidence contains no pair-selective structure for either estimand to differ on. It becomes discriminating only after a scale-commensurate arm produces *some* pair-specific reading. |
| **6. directed relation (TR-0008)** | **Blocked by a human decision, unchanged.** | `DEC-0014` §5 names the project-original arm; its exact rule is still design work. The integration adds one relevant fact: the directed fit's identifiability concern (`IDENTIFIABILITY.md`, near-collinear summary columns) is measured — under coupling the columns are collinear to `cos ≈ 0.9999`, while *uncoupled* they are near-orthogonal (`cos ≤ 0.21`), so a `B`-fit is meaningful only on an uncoupled geometry. |
| Survey `CANDIDATE-A` (SPATS sparse covariance) | **Deferred — `NEEDS_EXISTING_STUDY_RESULTS` stands.** | As family 5: no discriminating prediction is available yet, and SPATS's own primary warns that sparsity loses at small task counts. |
| Survey `CANDIDATE-B` (MTHOL higher-order) | **Deferred — prerequisite diagnostic now *fails*.** | It required a nonsingular summary Gram with stable eigenvalues; the coupled arms' Grams are numerically rank-1 (`−0.0, 0.0, 2.9998`). |
| Survey `CANDIDATE-C` (phase-dependent, diagnostic-only) | **Deferred.** | As family 4; and its own document already declines to call it a mechanism. |
| Grouping / clustering arm | **Rejected for now.** | Task clustering is a different research branch (`DEC-0005` §3, `AGENTS.md` boundaries). |
| Gradient surgery / loss weighting (GradNorm, PCGrad, uncertainty weighting) | **Out of category.** | Not Task Relation Learning under Zhang & Yang §2.4; `DEC-0010` withdrew the F9 rationale these would have addressed. |
| Broad `λ` sweep, more seeds of the same arm, retuning epochs | **Rejected.** | `DEC-0005`/`DEC-0006`: no rerun merely because a result disappointed; `DERIVED` shows the axis is empty, so a sweep would buy support changes, not the scale fix. |

---

## 7. What the successor must change — and what it must hold

The one scientifically meaningful independent variable is **the magnitude at which the estimated
relation is imposed on the task parameters**. Everything else in this benchmark is frozen
(`VARIANT_BENCHMARK_PROTOCOL.md` §1) or is a control (§2): frozen WavLM Large, `smp(0.5)` over all
25 layers, `ks_si_er`, shared `1024→512→2000` trunk, 30 epochs, batch 2048, AdamW `2.5e-3`,
`≈ 2820` steps, epoch checkpoint, the shared class-mean summary adapter, `λ₀ = 1` and `λ₂ = 0.01`
**inside the Ω step**, the published ADMM with its certificate, the three-arm structure and the
mandatory classical-MTRL and matched-baseline controls.

Two successor hypotheses follow. They are **not ranked by preference**: S2 is the *faithful,
pre-registered, cheap* step whose elimination is what justifies a deviation-class intervention, and
S1 is the *identifiable* mechanism test that needs that justification. Both carry a full
pre-registration draft under `../../proposals/`. Neither is registered, and neither is authorized.

---

## 8. Two actionable successor hypotheses

### S1 — TR-0012 (proposed): the relation is imposed at a magnitude the optimizer can weigh

Pre-registration draft: `proposals/TR-0012_scale_commensurate_coupling.md`.

| Field | Value |
|---|---|
| **Observed problem** | The published coupling term carries `≈ 3d` in value and `≥ 6.9e3` in gradient against a batch-mean loss of `≈ 0.5` and task gradients of `≈ 1e-2`, at **every** `(λ₀, λ₂)` the paper publishes (`DERIVED` §2.3); the arm's own regularizer, not the relation, drives training, the summaries align within one epoch, and the run plateaus. |
| **Causal/mechanistic hypothesis** | The published `λ₀ = 1` presumes the paper's *sum*-normalised likelihood convention; the adaptation added the same penalty to a *batch-mean* deep loss, which raises its relative weight by the summary length (`d = 2001`) and by the summary Gram's conditioning. Once dominating, the coupling's gradient sits along the precision's large (difference-mode) eigenvalues, i.e. it pushes the three summaries together, which makes `S` more singular, which inflates `Ω`, which raises the gradient again — a self-reinforcing loop, and the reason the collapse is a jump rather than a drift. Competing explanation, stated in advance: the relation object carries no useful information at all, so at *any* imposition strength the arm's outcome stays at the controls. |
| **Proposed intervention** | One scalar, pre-registered, *measured not tuned*: the coupling's contribution is calibrated at the first post-warmup step so that `‖∂(coupling)/∂θ‖` equals `‖∂L_task/∂θ‖` over the shared parameters and the classifier heads (`ρ* = 1`, one value, recorded per run together with the measured norms). The Ω step is untouched (`λ₀ = 1`, `λ₂ = 0.01`), so **the estimator and the learned relation object are exactly the published ones**; only the strength with which the estimated relation is imposed changes. |
| **Predicted measurable effect** | Coupling value `O(1)` against `O(0.5)`; the summary cosines stay in the uncoupled band (`≤ ≈ 0.5`, against `0.9999`); `S` does not lose rank; no plateau; the Ω support and partial correlations become readable per epoch instead of pinned at `+0.4911`; the outcome either separates from both controls or is a **clean** null. |
| **Falsification criterion** | The geometry still collapses (cosines `> 0.9` by the second post-warmup epoch), or the coupling-to-task gradient ratio exceeds `10` after calibration, or the validation trajectory plateaus as before. Any of these refutes the scale hypothesis and sends the explanation back to the estimator or the representation. |
| **Changed variable** | the coupling term's magnitude policy (`1 → ρ*·‖∇L‖/‖∇coupling‖`), i.e. exactly one scalar, held fixed for the whole run. |
| **Variables held constant** | everything in §7: protocol, adapter, `Ω` step, `λ₂`, ADMM and certificate, arms, controls, seeds, checkpoint policy, evaluation. |
| **Implementation complexity** | Low: one coefficient in the W step plus the calibration measurement, which reuses the existing gradient-diagnostics tooling (DG-0002 measured exactly these quantities). No new architecture, no new estimator. |
| **Compute cost** | One screen (three arms × seed 42) plus a reduced confirmation (seeds 0-4) if the screen promotes: same order as TR-0007's screen, i.e. `≈ 1.8` GPU-h and well under `$1` at the `$0.25–0.28/h` compatible parts. |
| **Risk of confounding** | Stated plainly: imposing the penalty at a chosen magnitude is a **declared deviation**, so a null cannot be attributed to p-MSSL as published; the calibrated scalar is measured on training gradients, not on outcomes, so it is not tuning, but it is a *chosen convention*, and every report must carry it; and the `Ω` step's own `λ₀ = 1` with a scaled coupling coefficient means the two half-steps use different `λ₀` — a deviation the proposal must state in the arm README before any run. |
| **Relationship to published literature** | The intervention targets the JMLR Eq. (3)/(4b)/(8) normalisation the paper uses with `λ₀ = 1`; the paper's `λ₁`/`λ₂` are CV-selected for its own sum-likelihood, and it publishes no statement about a batch-mean deep loss. This is therefore a *scale-convention* correction to the adaptation, motivated by the literature's own mathematics, not a new estimator — the same class of finding as the MTRL audit's `normalize_w` deviation. |

### S2 — TR-0013 (proposed): the published `λ₂` axis, as the faithful alternative that must be eliminated

Pre-registration draft: `proposals/TR-0013_published_lambda2_axis.md`.

| Field | Value |
|---|---|
| **Observed problem** | The screen ran a researcher-fixed `λ₂ = 0.01`; the arm's own README (`deviation 3`) and `DEC-0015` §3 pre-registered a validation-selected `λ₂` over the paper's two smallest grid values as an open human item, and the screen's own caveat scopes its `REJECTED` to that fixed value. That open item is still open, and it is the *faithful* explanation of the failure. |
| **Causal/mechanistic hypothesis** | Competing pair, both pre-registered: **H-λ (live)** a validation-selected `λ₂` from the paper's grid restores a resolvable run (either by removing the harmful edge or by shrinking the precision enough that the coupling stops dominating); **H-scale (this synthesis)** `λ₂` cannot, because the coupling's magnitude is `λ₂`-invariant for the values that retain any edge and `λ₂ ≥ 1` removes all three edges while leaving `3d` on the diagonal (`DERIVED` §2.3). |
| **Proposed intervention** | `λ₂` becomes validation-selected over `{0.01, 0.1}` — the paper's two smallest grid values, budget-matched to the in-category control's own two-value selection — with the *screen* running both values and the selection made on validation, never test. One variable: the `λ₂` policy. |
| **Predicted measurable effect** | `H-scale`: at `λ₂ = 0.1` the summary cosines still reach `> 0.9`, the coupling stays `≈ 2e3` per task, the plateau persists, and the support shrinks (2 of 3 edges on a healthy Gram) — i.e. the axis buys support, not scale, and the arm closes as a *faithful* negative result. `H-λ`: a run in which the geometry stays healthy and the outcome separates from both controls. |
| **Falsification criterion** | `H-scale` is falsified by any of: a `λ₂ ∈ {0.01, 0.1}` run whose coupling-to-task gradient ratio stays below `10` through training; summary cosines staying below `0.5`; a validation curve without the epoch-`≈ 10` plateau. `H-λ` is falsified if both values reproduce the same collapse and the same plateau. |
| **Changed variable** | the `λ₂` selection rule (researcher-fixed → validation-selected over two published values). |
| **Variables held constant** | as §7; in particular the arm keeps `λ₀ = 1` and the published summary adapter, so the arm stays a faithful published-method implementation and the result is citable as such. |
| **Implementation complexity** | Very low: the config key exists, the README already documents the convention, and the comparison is two arms of the same code path with validation-selected reporting. |
| **Compute cost** | the cheapest possible screen: 5 runs (2 candidate values + 3 controls) at seed 42, `≈ 1.5–1.8` GPU-h, `<$1`. |
| **Risk of confounding** | Low, but it must not be sold as a mechanism test: at `λ₂ = 0.1` the support changes, so an outcome change would be *either* a scale effect *or* an edge-suppression effect, and at `λ₂ ≥ 1` all edges are gone (H-inert). It also cannot rescue the published arm against the *value*-scale defect if `H-scale` is right, and it must not be extended into a broad `λ` sweep. |
| **Relationship to published literature** | This *is* the paper's own procedure: Algorithm 1 declares `λ₂` "chosen by cross-validation" and the classification experiments use the grid `{0.01, 0.1, 1, 10, 100}` (JMLR §4.1), which `MSSL_SPARSITY_ANALYSIS.md` §5.1 reads as strictly more general than the frozen `(λ₀, λ₂) = (1, 0.01)`. No new method is introduced. |

**Sequencing, on evidence rather than preference:** S2 first (faithful, cheapest, and its null is
what makes S1's deviation *justified* rather than convenient), S1 gated on the human's faithfulness
decision and, if approved, run after S2's result is recorded. S1's design already contains the
instrumentation — per-epoch summary Gram, `Ω` support, partial correlations, and the
coupling-to-task gradient ratio — that would otherwise need a separate diagnostic study.

---

## 9. Prerequisite diagnostics

| Diagnostic | Status |
|---|---|
| Is the summary-geometry collapse coupling-induced or intrinsic to the class-mean adapter? | **Answered at zero cost** (`OBSERVED`/`DERIVED` §2.3): coupling-induced; the uncoupled baseline is well-conditioned. No GPU needed. |
| Does the published `λ₂` axis contain a scale-safe value? | **Answered at the estimator level** (`DERIVED` §2.3): no value in the grid is scale-safe. S2 is the *training-side* confirmation and is proposed as a study, not assumed. |
| Is the collapse reproducible across seeds and across independent reruns? | **Open, single-seed** (F1). S2 and S1 both record it per run; a dedicated multi-seed diagnostic would be a re-run of a disappointing result (`DEC-0005`) and is not proposed. |
| Does *any* learned relation object track controlled transfer? | **Open, programme-level** (the survey's largest uncertainty, restated in §3). S1 is the first design in which the answer would be readable. |
| DG-0007 (normalization-corrected MTRL) | **Not a prerequisite and not evidence**: no result exists. If it is ever run, its per-epoch raw-summary Gram (its `PLAN.md` already asks for it) would be a second, independent probe of the same geometry question. |
| The DG-0007 record conflict (§2.5) | **Needs a human or an orchestrator resolution**, not compute: either the narrowed screen ran (and its evidence must be collected and recorded, or its absence explained) or the commit message is wrong. |

---

## 10. Decision and authorization state

* No compute is authorized for anything in this document; no envelope covers S1 or S2;
  `authorizations/TR-0007.yaml` is the only envelope on the record and it excludes both by scope
  and has expired.
* S1's intervention is a **declared deviation on a published arm** — the class
  `.agents/policies/autonomy.md` reserves to the researcher ("decide a faithfulness/deviation
  question"; "choose between scientifically different fixes for the same failure"), and the class
  `DEC-0015` closed for that phase ("do not invent a new normalisation, do not tune the formulation
  in this phase"). Reopening it for a *measured* scale defect is the researcher's call, and the
  proposal states both candidate realisations (a calibrated coefficient, or the natural
  `1/d` normalisation whose residual gradient ratio is still `≈ 350–8000`) with their different
  confounds, so the choice is made with the numbers in hand.
* S2 needs no deviation; it needs a compute envelope and a `λ₂`-policy confirmation, because the
  arm's README already pre-registers it as an open human item.
* TR-0008 remains blocked on its own design decision, unchanged.
* Nothing here registers a study: `STUDIES.jsonl` is untouched. `TR-0012` and `TR-0013` are
  allocated identifiers for drafts only, chosen above the `TR-0009/0010/0011` range that the
  tracking store already carries from a code line absent from this repository's history.

---

## 11. Open questions this synthesis leaves

1. **Does the scale defect reproduce in an independent run?** Single seed, one split (F1).
2. **Is the value-scale or the gradient-scale the right commensuration target?** S1's calibration
   fixes the gradient; the `1/d` normalisation fixes the value; they differ by `≈ 6` orders and only
   one can be the pre-registered variable.
3. **Is the coupling-induced collapse specific to this adapter?** The measurement covers the three
   TR-0007 arms; classical MTRL's `Ω` saturation (F5/F7) is consistent with the same loop at a
   different scale, but its coupling is a different operator (`Ω⁻¹`, `ε = 1e-4`) and was not
   measured.
4. **Does the ℓ₁ ever matter at all here?** `DERIVED`: at the collapsed geometry it is inert; the
   only regime in which it selects is `λ₂ ≥ 1`, where it removes everything. Whether any faithful
   p-MSSL configuration on this adapter selects *some* edges and stays scale-safe is exactly S2's
   question.
5. **What would count as "useful transfer" for this programme?** The protocol's promotion bar is
   `> 0.20pp` on every task across seeds with LOSO for ER; the *mechanistic* bar in
   `HYPOTHESIS_GRAPH.md` is stricter (a controlled, exposure-matched, target-specific intervention
   with a pre-declared direction). No study has yet been designed against the second bar.

---

## 12. Related documents

* `studies/TR-0007/analysis.md` §"2026-09-29 — Post-hoc scale reading" and
  `studies/TR-0007/coupling_scale_result.json`
* `proposals/TR-0012_scale_commensurate_coupling.md`, `proposals/TR-0013_published_lambda2_axis.md`
* `MSSL_SPARSITY_ANALYSIS.md` (§1.2 bijection, §3.2 ceiling, §4.1 barrier, §5.1 scale identity),
  `MSSL_PREDICTIONS.md`, `IDENTIFIABILITY.md`, `RELATION_SEMANTICS.md`, `HYPOTHESIS_GRAPH.md`,
  `EXPERIMENT_ROADMAP.md`
* `FINDINGS.md` F1–F10, `FAILURES.md` FL-0005, `FRAMEWORK.md`, `VARIANT_BENCHMARK_PROTOCOL.md`
* `audits/2026-09-29-mtrl-theory-to-implementation-audit.md` (the same class of finding, for the
  in-category control) and `audits/2026-09-29-dg0007-canonical-integration.md`
