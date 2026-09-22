# Failures — Task Relation Learning

Negative evidence registry. Entries are grouped by explanation rather than by
architecture name. Historical runs predate Study-ID tagging and are retained as
`LEGACY-PRE-ID` evidence.

## FL-0001 — Classical MTRL has no meaningful reproducible outcome effect

**Status:** ESTABLISHED NEGATIVE RESULT

**Observation:** Classical MTRL does not significantly or materially improve
the matched wavCSE baseline at `smp` 16L or 25L, and it has no resolved effect
on speaker-independent ER.

**Evidence:**

* 16L, matched seeds 0–4: aggregate Δ = −0.00010; paired 95% CI
  [−0.00183, +0.00163].
* 25L, matched seeds 0–4: aggregate Δ = −0.00056; paired 95% CI
  [−0.00243, +0.00130].
* ER LOSO, ten matched folds: `opt` Δ = −0.00110; paired 95% CI
  [−0.03070, +0.02850].

**Rejected explanations:**

* “one successful seed proves an MTRL win” — rejected by matched five-seed
  confirmation;
* “stronger λ will amplify useful coupling” — λ=0.05 worsened KS, SI and ER
  relative to λ=0.01 and reduced the off-diagonal magnitude;
* “more epochs will reveal the effect” — 60 epochs underperformed the matched
  30-epoch MTRL run on every task;
* “ordinary-split ER is an adequate substitute for LOSO” — contradicted by the
  ~15-point leakage gap and the MTRL-vs-baseline LOSO null result.

**What this does not explain:** The evidence establishes failure to improve; it
does not establish whether the cause is symmetric/dense relation
parameterization, relation estimation, parameter summarization or optimization.

**Consequence:** No generic MTRL tuning. DG-0001 rejected raw asymmetry as an
optimizer-exposure artifact (F8); DG-0002 confirms gradient-scale imbalance but
rejects persistent pairwise conflict (F9).

Provenance: FINDINGS.md F1–F4; `01-mtrl/README.md`; MLflow experiments
`taskrelation-mtrl`, `wavcse-baseline`, `taskrelation-mtrl-er-kfold`,
`wavcse-baseline-er-kfold`; `task_relations/MTRL_DIAGNOSTIC_SYNTHESIS.md`.

---

## FL-0002 — Persistent pairwise gradient conflict does not explain the MTRL null

**Status:** ESTABLISHED NEGATIVE DIAGNOSTIC

**Observation:** Under the matched `smp` 25-layer protocol, no KS/SI/ER pair met the pre-registered persistent-conflict threshold in baseline training for any seed `0–4`.

**Evidence:** Baseline late mean cosines across seeds were KS↔SI `+0.002`, KS↔ER `+0.001`, and SI↔ER `+0.022`; negative-conflict frequencies were `0.464`, `0.426`, and `0.357`. MTRL late means were similarly near zero. The signal is near-orthogonality, not persistent opposition.

**Rejected explanation:** Classical MTRL fails because it ignores a stable pairwise gradient conflict among these three tasks. DG-0002 provides no such conflict target under its controlled protocol.

**What remains:** F9 establishes a different optimization signal: ER gradient norms dominate KS/SI by roughly `7–9×` in middle/late training, and MTRL does not consistently reduce it. This does not yet prove that scale imbalance causes the outcome null.

**Consequence:** Do not prioritize conflict-only gradient surgery or a dynamic-relation mechanism from DG-0002. Target literature at explicit task-relation methods handling unequal task scale or reliability, subject to the taxonomy gate.

Provenance: FINDINGS.md F9; `research/studies/DG-0002/{analysis.md,confirmation_result.json}`; MLflow `taskrelation-diagnostics`, stage `confirm`.

---

## FL-0003 — No literature-eligible Task Relation Learning mechanism for F9

**Status:** ESTABLISHED NEGATIVE LITERATURE RESULT

**Observation:** Eight primary sources were screened against pre-registered eligibility gates. None is simultaneously an explicit Task Relation Learning method and a direct mechanism for the F9 shared-gradient scale imbalance in heterogeneous deep classification.

**Evidence:**

* Bayesian covariance methods (MTGTP; GP-kronsum) learn explicit task relations but assume aligned Gaussian multi-output regression and do not regulate shared-gradient magnitude.

* Multi-task averaging and SPATS are legitimate relation methods with sample-variance or sparsity assumptions that do not match F9; SPATS's own evidence warns sparsity can hurt with few tasks.

* Uncertainty weighting (Kendall et al. 2018) and GradNorm (Chen et al. 2018) directly address task scale and gradient-magnitude dominance, but learn per-task scalars with no relation object, placing them in optimization/loss weighting under the binding taxonomy.

* The 2024 informative-relation method rests on `W = H + P` parameter decomposition and does not target F9.

**Rejected explanation:** “A published explicit Task Relation Learning method already exists that directly addresses the confirmed F9 limitation and can be implemented faithfully here.”

**What remains:** Relation magnitude, relation confidence/noise and optimization weight are distinct quantities. Rakitsch et al.'s separate signal/residual task covariances encode the right distinction but are not portable to disjoint heterogeneous multiclass heads; F9 does not establish that residual noise causes the scale signal.

**Consequence:** No mechanism Study is authorized. The programme state is `NEEDS-HUMAN-REVIEW` (DEC-0008): retain strict scope and diagnose ER's data regime, broaden scope to optimization-aware MTL, or authorize a project-original hybrid. Do not relitigate the same literature gate for F9.

Provenance: FINDINGS.md R3; `research/studies/LT-0001/{PLAN.md,analysis.md,result.json}`; `research/literature/INDEX.md`.

---

## Repeated failure clusters

### No meaningful architecture effect

The controlled MTRL comparisons converge on outcome neutrality. Small
task-specific signs reverse across layers or evaluation protocol, and aggregate
effects remain below the project’s 0.20-point material-regression threshold.

### Relation estimate conditionality

Ω changes with pooling, layers, seed and held-out speaker. The earlier broad
claim that KS↔SI is inherently stable is narrowed: it is stable within 25L LOSO
and 25L seeds, but not across 16L seeds.

### Mechanism collapse to uninformative coupling

Several normalized-W settings saturate toward rank-near-one ±1/3 structures.
Uniform +1/3 saturation removes pair-specific discrimination. Saturation is an
observed failure mode for interpretation, not yet a demonstrated cause of the
performance null.

### Excessive regularization

Increasing λ perturbed all three task fits without creating stronger or more
useful relation structure. This is a specific rejected tuning hypothesis, not a
license to infer that every relation regularizer is harmful.

### Pooling and layer confounding

Changing pooling moved ER more than any MTRL-vs-baseline delta in the historical
campaign and qualitatively changed Ω. Layer-count conclusions also reversed
under `smp`. Architecture claims with unmatched representation are invalid.

### Seed noise and checkpoint-policy artifacts

The original 16L win disappeared over five seeds. The LNP final-epoch advantage
compares a candidate final epoch with a baseline whose checkpoints happened to
tie; it cannot replace the protocol-selected `opt` comparison.

### ER leakage

The historical ordinary split shares speakers between train and test. It
inflates ER by about 15 points and even produces a five-seed 25L MTRL regression
that does not appear under LOSO. Leaky ER remains screening evidence only.

### Task-count-dependent optimizer exposure

DG-0001's raw ER transfer appeared enormous under both standard-split and LOSO
evaluation. Approximate update-matched ER-only controls reproduced the gain:
KS+ER left only +0.53pp residual (CI spans zero), while SI+ER was 5.89pp worse
than its matched ER control. Same-epoch pair-minus-single comparisons are
invalid semantic-transfer evidence when task sets change the number and
composition of optimizer steps.

### Aggregate task-size masking

SI’s larger test set dominates the sample-weighted aggregate. Aggregate
neutrality can coexist with larger ER percentage-point movement, so every
comparison must retain per-task results. Existing speaker-independent evidence
still resolves ER to no effect.

---

## Explanations not yet supported

Do not classify the MTRL result as any of the following until the named
diagnostic exists:

* beneficial asymmetric transfer — DG-0001's raw signal was rejected after
  optimizer-exposure controls (F8); replication under a fully controlled sampler
  would be required to reopen it;
* a causal gradient-scale explanation — F9 establishes reproducible ER
  gradient-norm dominance and MTRL non-mitigation, but not that the imbalance
  causes MTRL's outcome null;
* causal Ω/transfer mismatch — DG-0001 shows a moderate discrepancy, but the
  triple-task Ω and pairwise controlled-transfer protocols are not identical;
* ER data-size causation — **resolved by DG-0005 (CONFIRMED)**: ER's gradient-norm
  dominance is training-mixture dependent and disappears when ER contributes
  KS-scale examples per batch (F10). The 7–9× ratio is *not* a task-intrinsic
  property. What remains open is whether the residual norm drop is reduced
  gradient-estimate variance or ER convergence/overfitting — DG-0005's single
  knob cannot separate them;
* confidence-aware, dynamic, layer-specific or sparse mechanism failure — no
  such mechanism has been tested.

---

## FL-0004 — Task-intrinsic gradient-scale justification for a relation mechanism

**Status:** ESTABLISHED NEGATIVE JUSTIFICATION (2026-09-22)

**Observation:** The symptom that motivated the F9-era mechanism search — ER
gradient norms dominating KS/SI by 7–9× — is a sampling-regime effect, so it
cannot license a scale- or reliability-aware mechanism as a fix for a task
property.

**Evidence:** DG-0005 matched seeds 0–4 at commit `8032a937` removed the late
dominance in `5/5` seeds by changing only ER's per-batch sampling weight, with
every exposure gate passing, a localized ER norm drop (`6.972 → 1.608`), and no
change in pairwise cosine structure. F10.

**Rejected justification:** "ER is intrinsically a large-gradient task, so
classical MTRL's static covariance is inadequate for a scale/reliability
reason." That premise is withdrawn (DEC-0010).

**What this does not explain:** Why classical MTRL fails to improve outcomes
(F4, FL-0001) remains unexplained. F7 saturation, relation estimation error and
parameter-summary inadequacy are all still untested.

**Consequence:** Do not re-open the F9 literature gate (FL-0003), and do not
re-ablate sampling composition expecting a mechanism justification. A future
mechanism must be motivated from relation structure or from an explicitly
regime-conditioned claim.
