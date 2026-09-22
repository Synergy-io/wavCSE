# Task Relation Learning — Research Backlog

This backlog contains scientific questions, not merely implementation tasks.

The agent may reprioritize entries when new evidence appears, but it must record
the reason in STATE.md / DECISIONS.md.

Priority meanings:

* **P0** — foundational; needed before interpreting later experiments
* **P1** — high-value research hypothesis
* **P2** — useful after stronger candidates emerge
* **P3** — exploratory / optional

Statuses:

* `READY`
* `BLOCKED`
* `ACTIVE`
* `DONE`
* `REJECTED`
* `SUPERSEDED`

---

# Scope of This Backlog

The formal progression is binding (DEC-0005). In practice, for every entry here:

* **diagnostic studies (`DG-xxxx`) are the only studies that may start now.**
* a mechanism study (`TR-xxxx`) requires **both** diagnostic evidence naming the
  MTRL assumption that fails **and** a published method (`LT-xxxx`) whose stated
  assumption addresses it. Until both exist its status is `BLOCKED` — "ready
  after diagnostics" is not an admissible status for a mechanism study.
* LNP is a diagnostic/control configuration, not a method (DEC-0002). GBC is
  archived and is not a backlog item (DEC-0003). TSM/PMR are quarantined pending
  the validity audit in DEC-0004 and are not backlog items.
* targeted literature mode is **mandatory** once a diagnostic identifies a
  concrete MTRL limitation — it is a stage of the progression, not a
  plateau-only fallback.

---

# P0 — Diagnostics

## DG-0001 — Empirical directed task-transfer matrix

**Status:** DONE — completed 2026-09-21

### Study record

`research/studies/DG-0001/` holds the protocol, configs, run registry,
`result.json` and `analysis.md`. Machine-readable outputs:
`task_relations/{empirical_transfer,loso_transfer,optimization_control}.json`.

### Primary hypothesis

Beneficial transfer among KS, SI and ER is **directional and pair-selective**,
so a single symmetric dense Ω cannot represent it: `T(A <- B)` is not symmetric
in A/B for at least one pair, and/or is indistinguishable from zero for at least
one pair.

### Design

Reference config: `smp` (λ=0.5) + 25 layers — the best-supported configuration
(F2). Every DG-0001 arm must use the same explicit seed, committed code/config,
pooling, layers, epoch budget and checkpoint policy.

Stage A (screening, one explicit seed):

| Task set | Arm | Purpose |
| --- | --- | --- |
| `ks`, `si`, `er` | baseline | T(A <- ·) references, one per task |
| `ks_si`, `ks_er`, `si_er` | baseline | pairwise arms |
| `ks_si_er` | baseline | matched triple-task control |

The legacy triple-task runs may be reused only if their seed, commit and config
exactly match the new arms. Otherwise rerun the triple: historical evidence
predates Study IDs and is not automatically a DG-0001 control.

Stage B — Ω correspondence (DG-0003 folded in) is adaptive. First compare the
Stage-A matrix with the existing five-seed triple-task Ω evidence. Add MTRL
pairwise/triple arms only where that comparison cannot distinguish relation
representation, relation estimation and downstream regularization failure.

Stage C — only if a Stage-A asymmetry or zero-transfer pair is decisive, and
only for the decisive pairs: multi-seed confirmation, seeds 0–4 (F1). Any
ER-related conclusion additionally requires the LOSO harness (F3): the leaky
split may suggest a direction, it never carries the claim.

### Falsification — each outcome decides the next stage

* **All pairwise transfers ~0** → the failure is not about *representing*
  useful relations. Pivot to F7 saturation and regularization/optimization;
  TR-0001/TR-0002 stay blocked.
* **Symmetric and dense transfer** → a symmetric dense Ω is representationally
  adequate; diagnose whether Ω estimates it and whether the regularizer uses it.
* **Replicated asymmetric transfer** → symmetric Ω is structurally inadequate
  by construction. This is a concrete limitation for targeted literature
  search; Ω cannot “track” both directions.
* **Selective but symmetric transfer, Ω tracks it** → representation is
  adequate; inspect regularization/optimization before considering replacement.
* **Selective transfer, Ω does not track it** → diagnose the mean-head summary,
  normalization and closed-form update; the literature target is relation
  estimation only after that implementation-specific explanation is separated.

### Cost

Stage A requires six new trainings if an exact matched triple-task control is
reusable, otherwise seven. Do not pre-allocate four Stage-B MTRL runs: its cost
is zero to four trainings depending on Stage A and the existing triple-task Ω
comparison. All full runs use one screening seed and at most two concurrent
jobs. Measure one run's wall-clock first.

### Known risks

* single-task and pairwise paths passed smoke tests and all registered arms;
* changing task count changed loss scaling, shared-backbone update count and
  effective per-task minibatch size; Stage D showed this dominated raw ER
  transfer (F8);
* leaky-split ER was treated as screening only and followed by ten-fold LOSO.

### Outcome

The raw seed-42 matrix suggested ER-directed asymmetry, including ER<-KS
+5.79pp and ER<-SI +2.89pp on the leaky split. Ten-fold LOSO amplified the raw
gains, but optimization-exposure controls rejected the semantic interpretation:

| Controlled residual, fixed epoch | Mean Δ | Paired 95% CI |
| --- | ---: | --- |
| KS+ER − step-matched ER | +0.0053 | [−0.0289, +0.0394] |
| SI+ER − step-matched ER | −0.0589 | [−0.0883, −0.0295] |

Same-epoch pairwise training changes the number and composition of optimizer
steps. The primary asymmetry hypothesis is weakened, not supported. No
asymmetric or sparse mechanism is justified. See F8 and the Study analysis.

### Deliverables

The raw matrix remains in `task_relations/empirical_transfer.json` with its
caveats. Controlled decomposition and Ω comparison are in
`task_relations/optimization_control.json` and the Study `analysis.md`.

---

## DG-0002 — Gradient compatibility baseline

**Status:** CONFIRMED — matched baseline/MTRL seeds 0–4 completed 2026-09-22

### Question

Do task gradients agree or conflict during ordinary wavCSE MTL training?

### Measure

For shared parameters:

* gradient norm per task;
* pairwise gradient cosine;
* proportion of sampled steps with cosine `< 0`;
* mean/median cosine;
* distribution across training;
* optional layer-wise measurements where computationally feasible.

### Sampling constraint from DG-0001

Hold optimizer exposure constant: define how task examples are batched, match
or model the number of shared-parameter updates, and report effective per-task
batch sizes. Otherwise gradient frequency and task semantics are confounded
(F8). Use the same exposure-controlled sampler for the baseline and any MTRL
comparison.

Pairs:

* KS/SI
* KS/ER
* SI/ER

### Compare

* early training;
* middle training;
* late training.

### Why

A static task-relation matrix may fail if optimization relationships are dynamic.

### Outcome

The exposure check passed for every matched seed pair. ER shared-gradient norms reproducibly dominated KS/SI:

| Method | Middle max/min ratio | Late max/min ratio | Seeds above ratio 3 in both phases |
| --- | ---: | ---: | ---: |
| baseline | 7.243 ± 0.272 | 8.962 ± 0.456 | 5/5 |
| MTRL | 7.129 ± 0.421 | 9.004 ± 1.059 | 5/5 |

MTRL-minus-baseline paired ratio intervals included zero in both phases, so MTRL did not consistently mitigate the imbalance. No pair met the persistent-conflict threshold in any seed; late gradient interactions were near-orthogonal. Final Ω magnitude saturated near `1/3` in every seed, but seed 4 flipped both ER-edge signs.

DG-0002 therefore confirms an optimization-scale limitation of classical MTRL under the matched `smp` 25-layer protocol and rejects persistent pairwise conflict as the supported explanation. It does not establish causation or an ER performance effect. Targeted literature research is now mandatory (DEC-0005); search from unequal task-gradient scale / relation reliability, not from a preselected mechanism. See F9 and `studies/DG-0002/{analysis.md,confirmation_result.json}`.

---

## DG-0003 — Compare learned Ω with empirical transfer

**Status:** PARTIALLY ESTABLISHED — DG-0001 shows moderate disagreement, but causal interpretation is blocked by triple-task Ω versus pairwise-transfer protocol mismatch

### Question

Does MTRL's learned Ω correspond to actual beneficial transfer?

### Analysis

Compare:

* Ω pair magnitude/sign;
* empirical directed transfer;
* gradient cosine;
* performance changes.

### Important

Do not assume covariance similarity implies positive transfer.

### Potential finding

Learned parameter similarity may measure something different from useful
cross-task transfer.

---

## DG-0004 — Relation stability analysis

**Status:** PARTIALLY ESTABLISHED — retrospective analysis complete; prospective
prediction/replication remains

### Existing evidence

F5/F6 and `task_relations/MTRL_DIAGNOSTIC_SYNTHESIS.md` now cover:

* seed — final Ω across matched seeds 0–4 at `smp` 16L and 25L;
* pooling — legacy `weighted`, `lnp` and `smp` settings;
* LOSO fold — ten 25L `smp` speaker folds;
* training epoch — five Ω updates in each LOSO fold;
* layer configuration — matched five-seed 16L versus 25L final Ω.

The result is conditional, not one pair ranking: 25L KS↔SI is stable across
seeds/folds, while 16L KS↔SI is seed-unstable and 16L SI↔ER is the only
sign-consistent edge. ER-edge dispersion grows over the five LOSO updates.

### Remaining question

Does relation stability predict empirical transfer or downstream outcome on
new runs? Retrospective Ω variance alone cannot select a mechanism.

### Required next use

Do not run another generic stability sweep. Pre-register a stability statistic
(mean, standard deviation, sign consistency and saturation state), then test it
against DG-0001 transfer or a prospective outcome. Coefficient of variation is
reported only when the mean is meaningfully away from zero; rank consistency is
reported only when unsaturated pair ranks exist.

### Goal

Separate `strong relation`, `stable estimate`, and `useful transfer`. Existing
evidence proves these are not interchangeable.

---

# P1 — Mechanism Research

## TR-0001 — Sparse/selective relation learning

**Status:** BLOCKED — gate: DG evidence of pair-selectivity + an LT-xxxx published method (DEC-0005)

### Hypothesis

Not every speech-task pair should be coupled strongly.

Dense symmetric MTRL may create unnecessary transfer.

### Motivation

Ω stability is pair- and condition-dependent, but no pair-selective empirical
transfer has been measured. F6 motivates the diagnostic only; DG-0001 must
establish selectivity before this hypothesis can motivate a method.

### Candidate mechanisms

Search literature for genuine Task Relation Learning methods using:

* sparse covariance;
* sparse precision;
* structured sparsity;
* pair selection.

### Falsification

If sparse relation learning repeatedly collapses to dense relations or does not
reduce negative transfer under matched evaluation, reject the hypothesis.

---

## TR-0002 — Asymmetric task relation learning

**Status:** BLOCKED — gate: DG-0001 must show usable empirical asymmetry + an LT-xxxx published method

### Hypothesis

Transfer between these tasks is directional.

Example possibility:

`SI → ER`

may differ from:

`ER → SI`.

### Requirement

DG-0001 must first show whether useful empirical asymmetry exists.

### Literature queries

* asymmetric multi-task relationship learning;
* directed multi-task transfer;
* asymmetric task covariance;
* directed task relation matrix.

### Evaluation

Compare against:

* wavCSE matched baseline;
* symmetric MTRL;
* directed empirical transfer matrix.

### Important

Do not implement asymmetry merely because it sounds plausible.

Require evidence first.

---

## TR-0003 — Confidence-aware task relations

**Status:** BLOCKED — gate: DG-0004 stability evidence + an LT-xxxx published method

### Hypothesis

The magnitude of a learned relation and the confidence in that estimate should
be treated separately.

A highly unstable ER relation should receive weaker influence than a stable
KS↔SI relation.

### Candidate ideas

* variance-aware relation regularization;
* relation shrinkage;
* uncertainty-weighted pairwise coupling;
* Bayesian relation estimates;
* bootstrap-based confidence;
* sample-size-aware relation strength.

### Scientific significance

This may explain why classical MTRL fails when tasks have very different data
regimes.

---

## TR-0004 — Dynamic relation learning over training

**Status:** BLOCKED on DG-0002, and then an LT-xxxx published method (DEC-0005)

### Hypothesis

Task compatibility changes during optimization.

A single relation matrix refreshed occasionally may not capture the relevant
dynamics.

### Questions

* Are relations different in early and late training?
* Does negative transfer occur only during particular phases?
* Does relation stabilization correlate with convergence?

### Candidate mechanisms

Only search/implement after DG-0002 demonstrates meaningful temporal behaviour.

---

## TR-0005 — Layer-dependent task relations

**Status:** BLOCKED — gate: DG evidence + an LT-xxxx published method; "pending evidence" is not an admissible status for a mechanism study

### Hypothesis

Tasks may share low-level speech information while diverging at higher
representation levels.

### Questions

* Are gradient relations different by layer?
* Does task relation inferred at one representation level predict transfer at
  another?
* Does pooling obscure this structure?

### Warning

Do not drift into the project's decomposition/low-rank research categories.

The primary contribution must remain explicit Task Relation Learning.

---

## TR-0006 — Better task parameter representation for MTRL

**Status:** BLOCKED on DG-0003 — this is a diagnostic of MTRL's own input representation; if it is pursued as a mechanism it additionally needs an LT-xxxx published method

### Problem

Current MTRL must construct comparable per-task vectors despite very different
classifier output dimensions:

* KS classes;
* large SI class count;
* ER classes.

Existing MTRL therefore summarizes task-head parameters.

### Question

Does the current summary destroy information required to learn useful task
relationships?

### Investigate

* sensitivity of Ω to parameter summarization;
* normalization schemes;
* class-count invariance;
* alternative fixed-dimensional task signatures.

### Constraint

Avoid methods whose main contribution becomes generic representation learning.

---

# P1 — ER-specific scientific controls

## DG-0005 — Data-regime hypothesis

**Status:** DONE — **CONFIRMED** 2026-09-22. Matched A0/A1 seeds 0–4 at commit `8032a937`; late max/min task-norm ratio `8.962 ± 0.456 → 2.352 ± 0.270` (paired `−6.610`, `[−7.353, −5.867]`), below the pre-registered `3.0` threshold in `5/5` seeds with every exposure gate passing. Established F10 and refined F9. Diagnostic only: it does not reopen the mechanism gate.

**Arms:** A0 matched baseline; A1 ER-weighted training composition (raise ER per-batch contribution to KS scale); A2 pre-registered reverse direction (reduce KS/SI contribution to ER scale), launched only if A1 is ambiguous.

**Gate:** the per-step realized valid-example counts must match each arm's design target, otherwise the arm is invalid, not merely noisy (F8).

**Not in this Study:** no MTRL arm, no Ω analysis, no GradNorm-style scale-normalization arm (unauthorized), no LOSO (no ER performance claim).

**Confirmation result:** ten runs, all exposure gates passed (A1 realized ER/KS sampled-batch count `0.990–1.008` vs A0's `0.089`), 2,820 steps and 142 diagnostic samples in every arm. A0 middle/late `7.243 ± 0.272` / `8.962 ± 0.456` (exactly reproducing DG-0002's baseline); A1 `2.818 ± 0.272` / `2.352 ± 0.270`. A1's late ratio was below `3.0` in `5/5` seeds; the middle ratio was below `3.0` in `4/5` (one seed `3.054`). The change is ER-localized (late ER norm `6.972 → 1.608`; KS `+0.066`, SI `−0.117`) with no conflict signal. A2 remains untriggered because A1 was unambiguous.

**Remaining confound:** A1 also multiplies ER's optimizer updates per epoch and saturates its head harder (train ≈0.99 vs validation ≈0.82), so estimator variance and convergence/overfitting are not separated. That split is the highest-information follow-up if a new diagnostic is opened.

**Outcome:** gradient scale is a training-mixture property (F10), not a task-intrinsic relation property; F9 is refined accordingly. No ER performance claim — ordinary-split deltas are speaker-leaky context and their ER direction is consistent with memorization.

### Reframed question for the F9 era

Does ER's much smaller effective batch (≈47 sampled examples per mixed batch versus 539 KS and 1,462 SI) produce the ER gradient-norm dominance, or does the dominance persist under matched data regimes?

### Hypothesis

ER relation instability and its gradient-norm dominance are primarily caused by
its smaller/noisier data regime rather than by task semantics.

### Potential experiment

Reduce KS/SI effective training data to matched regimes and measure whether
relation stability deteriorates correspondingly.

### Alternative

Bootstrap equal-sized subsets where feasible.

### Interpretation

If relation instability tracks sample size rather than task semantics, this is
important for the final method-selection framework.

---

## DG-0006 — ER relation consistency under LOSO

**Status:** PARTIALLY ESTABLISHED

### Existing evidence

ER-related Ω entries vary strongly across LOSO folds.

### Remaining question

Do other relation diagnostics show the same instability?

Compare across folds:

* learned Ω;
* gradient cosine;
* directed transfer where feasible;
* parameter similarity.

---

# P2 — Candidate combinations

Do not start these until at least one mechanism shows a credible effect.

## AB-0001 — Sparse + asymmetric

Question:

Are useful relations both selective and directional?

Blocked until:

TR-0001 and TR-0002 have independent evidence.

---

## AB-0002 — Confidence-aware + sparse

Question:

Can uncertain edges be shrunk while retaining strong stable edges?

Blocked until:

TR-0001 and TR-0003.

---

## AB-0003 — Dynamic + asymmetric

Blocked until:

TR-0002 and TR-0004 independently show value.

---

# P2 — Final framework analyses

These are paper-level analyses rather than initial architecture searches.

## FW-0001 — Predict architecture success from relation diagnostics

### Goal

Given pre-training/cheap diagnostics, predict which Task Relation Learning
assumption is appropriate.

Potential inputs:

* directed transfer;
* gradient cosine;
* conflict frequency;
* relation stability;
* task size imbalance;
* task difficulty;
* relation asymmetry.

Potential output:

decision framework for method selection.

---

## FW-0002 — Relation taxonomy for KS/SI/ER

Characterize each pair using terms such as:

* positive / negative transfer;
* symmetric / asymmetric;
* stable / unstable;
* dense / weak;
* representation-sensitive;
* sample-sensitive.

This should be empirically grounded.

---

## FW-0003 — Cost/benefit analysis

For each successful method record:

* parameter overhead;
* training overhead;
* inference overhead;
* performance gain;
* robustness;
* interpretability.

A method-selection framework should not optimize accuracy alone.

---

# Literature Mode

Targeted literature mode is a **required stage of the formal progression**
(DEC-0005), not a fallback — it is entered as soon as a diagnostic identifies a
concrete MTRL limitation, and it must be completed before a mechanism study is
implemented. The OBJECTIVE.md plateau criterion is a second, independent
trigger.

## Current targeted literature action — COMPLETE (`LT-0001`, 2026-09-22)

**Status:** DONE — REJECTED. Eight primary sources were screened against the gates below; none satisfied all of them. See `literature/INDEX.md` for the cards and `studies/LT-0001/analysis.md` for the eligibility matrix.

**Result:** Explicit Task Relation Learning methods either lacked a direct optimization-scale/reliability mechanism or required aligned Gaussian/mean-estimation assumptions incompatible with disjoint heterogeneous multiclass tasks. The methods that directly address scale (homoscedastic uncertainty weighting, GradNorm) learn per-task loss weights with no relation object and are optimization/loss weighting, not relation learning. The closest relation-level principle — Rakitsch et al.'s separate signal and residual task covariances — would require a novel deep-classification hybrid and presupposes residual-noise causation that F9 has not established.

**Consequence:** No mechanism Study is authorized. Programme state is `NEEDS-HUMAN-REVIEW` (DEC-0008) with three options: retain strict scope and diagnose ER's data regime; broaden scope to optimization-aware MTL; or authorize a project-original relation-plus-reliability hybrid. The conservative default is the first.

**Taxonomy gate (retained for any future cycle):** generic loss weighting, gradient surgery, mixture-of-experts, low-rank, clustering or decomposition methods are not admissible merely because they address imbalance.

**Remaining causal uncertainty:** ER's smaller effective batch may cause the imbalance. Any claim that the relation is task-intrinsic still requires a matched-data control, which is the conservative autonomous continuation.

Expected output per literature cycle: paper cards (`literature/INDEX.md`) and,
for each method worth pursuing, an `LT-xxxx` study entry whose assumption maps
onto a specific measured finding (F2, F5, F6, F7 or a DG result).

Do NOT search merely:

`best multi task learning model`

Instead construct queries from observations.

Examples:

* sparse task relationship learning negative transfer;
* asymmetric task relation multi-task learning;
* task covariance uncertainty multi-task learning;
* task relation stability multi-task learning;
* dynamic task relationship learning;
* layer-wise task relationships multi-task neural networks;
* multi-task relation learning unequal sample sizes;
* negative transfer speech multi-task learning.

Every useful paper should produce a structured paper card containing:

1. citation;
2. problem;
3. mathematical assumption;
4. relation representation;
5. optimization method;
6. evidence;
7. assumptions;
8. differences from our setting;
9. implementation difficulty;
10. candidate Study ID.

Downloading papers is not itself research progress.

A paper matters only when its assumption can be related to observed evidence.

---

# Agent Backlog Rules

The agent may add studies.

For every new study it must state:

* observation motivating it;
* falsifiable hypothesis;
* independent variable;
* matched control;
* cheapest adequate experiment;
* expected information gain;
* whether literature motivated it;
* category-boundary check.

The agent should prefer:

**high information gain per GPU-hour**

over:

**largest number of experiments.**
