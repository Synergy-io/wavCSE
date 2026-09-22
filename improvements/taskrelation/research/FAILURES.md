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
optimizer-exposure artifact (F8); DG-0002 is next.

Provenance: FINDINGS.md F1–F4; `01-mtrl/README.md`; MLflow experiments
`taskrelation-mtrl`, `wavcse-baseline`, `taskrelation-mtrl-er-kfold`,
`wavcse-baseline-er-kfold`; `task_relations/MTRL_DIAGNOSTIC_SYNTHESIS.md`.

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
* persistent pairwise gradient conflict — not supported in DG-0002's seed-42
  screen; ER gradient-norm dominance is promising but requires matched seeds
  0–4 before it can explain MTRL's null result;
* causal Ω/transfer mismatch — DG-0001 shows a moderate discrepancy, but the
  triple-task Ω and pairwise controlled-transfer protocols are not identical;
* ER data-size causation — requires DG-0005;
* confidence-aware, dynamic, layer-specific or sparse mechanism failure — no
  such mechanism has been tested.
