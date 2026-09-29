# Task Relation Learning — Current Research State

> This file is the canonical restart point for a fresh research agent.
>
> Keep it concise. Do not turn this into an experiment log.
> Detailed history belongs in STUDIES.jsonl, individual study folders,
> FINDINGS.md, FAILURES.md, and architecture READMEs.

## Research Scope

We study **parameter-based Multi-Task Learning — Task Relation Learning**
for speech using the wavCSE experimental framework.

Tasks:

* `ks` — Keyword Spotting
* `si` — Speaker Identification
* `er` — Emotion Recognition

The upstream speech representation is fixed to the existing wavCSE /
WavLM embedding pipeline unless the human explicitly changes project scope.

The current primary objective is:

> Develop a Task Relation Learning approach that produces a reproducible,
> statistically credible improvement over the matched wavCSE baseline across
> KS, SI, and ER, without material negative transfer.

The larger scientific objective is:

> Understand how measurable task relationships and transfer behaviour predict
> which parameter-based MTL approaches are appropriate for a set of speech tasks.

Terminology — older docs are ambiguous about the word "baseline":

* **wavCSE baseline** — the matched reproduction of the feature-based paper
  approach (`improvements/base/`). The reference to beat, and the current
  champion.
* **MTRL** — the formal Task Relation Learning baseline *method* under study
  (`01-mtrl/`). The thing we diagnose and extend, not the thing we beat.

Binding scope decisions: `DECISIONS.md` (DEC-0001 … DEC-0014). Established
findings: `FINDINGS.md` — authoritative over the one-line summaries below.

---

# Current Research Phase

**Phase:** **variant benchmarking** (DEC-0013, human re-scope, 2026-09-22).
`DG-0005` stands (F10: gradient scale is a training-mixture property) and the
F9-era rationale stays withdrawn. By human decision the diagnostic-first
sequencing is **deferred**, and the programme now implements and compares
**published** relation-learning variants under one matched protocol, starting
with asymmetric/directed relations and a better relation estimator or
task-parameter representation. The published-method and category gates are
unchanged; only one relation method (classical MTRL) has ever been evaluated
here, so this space is unexplored rather than exhausted. DEC-0014 (human,
2026-09-22) then recorded that family A (asymmetric/directed) has no clean
published pass, so that arm is **project-original** and must be labelled as
such; family B keeps faithful-implementation status.

**Active artifact:** `FRAMEWORK.md` — the cross-study method-selection framework
(conditioned-quantity table, relational-vs-optimization classification, evidence
levels, selection rules R1–R8, open decision points, ranked missing evidence).
`FINDINGS.md` remains authoritative for numbers; `MTRL_DIAGNOSTIC_SYNTHESIS.md`
holds the per-quantity MTRL evidence and is kept consistent with it.

**Formal progression** (binding — DEC-0005; stages are not skipped):

```
reproduced wavCSE baseline
  -> classical MTRL
  -> diagnostic analysis of MTRL behaviour
  -> identify concrete limitations of MTRL
  -> targeted literature search for published Task Relation Learning methods
     addressing those limitations
  -> implement justified methods
  -> screen
  -> multi-seed confirmation
  -> LOSO where ER claims are involved
  -> ablation and behavioural analysis
  -> derive method-selection framework
```

A mechanism study (`TR-xxxx`) may only start once **both** hold:

1. a diagnostic study (`DG-xxxx`) has identified the specific MTRL assumption
   that fails, with evidence; and
2. a literature study (`LT-xxxx`) has produced a *published* method whose stated
   assumption addresses that failure.

No arbitrary architecture generation. No pre-committed method ordering. The
previous ranked ordering (sparse → asymmetric → confidence → dynamic → layer)
is removed for exactly this reason; the hypotheses survive as *gated* backlog
entries.

**Immediate research question:**

> For each published relation-learning variant in the two selected families:
> does it beat **both** classical symmetric MTRL (matched in-category control)
> and the matched **wavCSE baseline** under one shared protocol, without material
> regression on any task? First sub-question, answered by `LT-0002` before any
> code: which published methods in those families are formal Task Relation
> Learning (Zhang & Yang §2.4) and implementable faithfully against disjoint,
> heterogeneous KS/SI/ER heads?

---

# Existing Implementations

## 01 — MTRL — ACTIVE (formal starting method)

Location:

`improvements/taskrelation/01-mtrl/`

Method:

Classical Multi-Task Relationship Learning based on a learned task covariance
matrix Ω and relation regularization.

Classification:

**ACTIVE — the formal starting method of the Task Relation Learning programme
and its primary existing baseline method (DEC-0001).** Its evidence is retained
and remains the starting point for all future work in this branch.

Status of its evidence:

**Heavily evaluated (16L, 25L, 5-seed, LOSO) with no reproducible meaningful
advantage over the matched wavCSE baseline (F4).**

Consequences:

* the negative result is the programme's first input → diagnose *why* before
  proposing any replacement (DEC-0005);
* generic MTRL hyperparameter tuning stays forbidden — `mtrl_lambda` was already
  swept (0.05 was worse than 0.01 on every task) and no diagnostic implicates
  another parameter yet;
* per-task single-split deltas in the folder's iteration log are superseded by
  the multi-seed results at the same configs (F1): do not quote an MTRL
  task-level gain measured on one split.

---

## 02 — LNP — DIAGNOSTIC / CONTROL (not an active method)

Location:

`improvements/taskrelation/02-lnp/`

Purpose:

Controlled comparison demonstrating the importance of holding pooling constant.

Classification:

**DIAGNOSTIC / CONTROL study (DEC-0002).** LNP is *not* an active Task Relation
Learning method. It is removed from the method progression and from architecture
search; its configs are control configs only.

Purpose (historical, still binding on methodology):

Controlled comparison demonstrating the importance of holding pooling constant.

Status:

**Completed diagnostic study. Findings retained (F2, F5).**

Major lesson:

Pooling/representation choice can produce an effect larger than the Task
Relation Learning mechanism itself — +3.25pp on `er` from a pooling change
alone, larger than any MTRL-vs-baseline delta measured in the whole `01-mtrl`
campaign — and it changes the learned Ω qualitatively (`weighted` sign-structured
vs `lnp`/`smp`+25L saturated uniform).

---

## 03 — GBC — ARCHIVED / OUT OF CURRENT FORMAL SCOPE

Location:

`improvements/taskrelation/03-gbc/`

Classification:

**ARCHIVED / out of the current formal research programme (DEC-0003).**
Implementation, configs and historical notes are preserved as-is.

Why:

GBC is an original project design, and no defensible published Task Relation
Learning method matching its mechanism has been established (an earlier
attribution to "Zhang et al. 2010 / BOMC" was retracted in the folder README).
A method with no verified literature source cannot carry this project's
literature-derived contribution claim.

Rules:

* do not use GBC results to motivate, rank or gate literature-grounded Task
  Relation Learning approaches;
* spend no autonomous research cycles on GBC unless the human explicitly
  reactivates it;
* if it appears in a historical comparison table, label it "original project
  design, not literature-grounded".

---

## Legacy exploratory implementations — TSM and PMR — QUARANTINED

Older exploratory implementations (flat `models/`/`trainers/`/`configs/` layout,
predating the `0N-<name>/` convention):

* TSM — `models/tsm_model.py`, `trainers/tsm_trainer.py`, `configs/tsm_config.yml`
* PMR — `models/pmr_model.py`, `trainers/pmr_trainer.py`, `configs/pmr_config.yml`

Classification:

**Historical / exploratory, unvalidated, quarantined (DEC-0004).** Neither has
ever completed a training run — no `tsm` or `pmr` MLflow experiment exists.
Neither is the next research direction, and neither is a backlog item.

They may be relied on only after a written validity audit covering all four:
mathematical validity of the implemented objective; literature source verified
against the actual publication (GBC's attribution was wrong once already);
implementation correctness (PMR has a known `torch.stack` shape bug that fires
once its regularizer activates after warmup); and taxonomy placement (a genuine
Task Relation Learning method in the Zhang & Yang §2.4 sense, and distinct from
the low-rank / clustering / decomposition branches).

---

# Established Findings

Authoritative text, numbers and provenance: `FINDINGS.md`. The summaries below
exist so this restart file stands alone; when the two disagree, `FINDINGS.md`
wins.

These findings are already supported by existing experiments and should be
treated as prior evidence.

## F1 — Single-seed improvements are not trustworthy

An apparent MTRL improvement disappeared when matched baseline and MTRL were
repeated across multiple seeds.

Therefore:

* one seed = screening evidence only;
* a single high score must never be described as a confirmed improvement;
* promising candidates require multi-seed confirmation.

---

## F2 — Pooling is a major confound

Changing pooling produced effects comparable to or larger than Task Relation
Learning architecture changes.

Therefore architecture comparisons must use matched:

* pooling;
* layer selection;
* data;
* epochs;
* optimization;
* evaluation protocol;
* random-seed treatment.

A comparison with unmatched pooling does not establish an architectural effect.

---

## F3 — ER's historical single split is speaker-leaky

Single-split Emotion Recognition numbers are not sufficient for serious claims.

The authoritative evaluation for ER conclusions is speaker-independent
Leave-One-Speaker-Out evaluation.

Single-split ER may still be used for cheap screening, but must be clearly
labelled as screening evidence.

---

## F4 — Classical MTRL does not currently beat wavCSE meaningfully

Across the existing controlled and multi-seed investigations, classical
symmetric MTRL has not demonstrated a reproducible meaningful advantage over
the matched wavCSE baseline.

Therefore the next research stage should primarily investigate limitations of
the modelling assumption rather than performing arbitrary tuning of the same
method.

---

## F5 — Learned task relationships are representation-dependent

The learned Ω matrix changed substantially under different pooling methods.

Therefore:

> A learned task-relation matrix is not an intrinsic, representation-independent
> truth about KS, SI and ER.

It describes relationships between tasks as expressed through the current
representation, architecture, sample and optimization process.

This is potentially an important final-framework finding.

---

## F6 — Relation stability is conditional on evaluation axis

Within 25L `smp` LOSO, KS↔SI is fold-stable while ER-involving Ω entries are
variable and may change sign. Matched five-seed evidence narrows that claim:
at 16L KS↔SI is seed-unstable and SI↔ER is the only sign-consistent edge; at
25L KS↔SI is stably saturated while ER edges flip together in one seed.

Therefore no pair has a demonstrated task-intrinsic stability ranking. Relation
claims must name pooling, layers and whether variation is over seed or fold.
Stable magnitude also does not imply useful transfer: 25L KS↔SI is near +1/3
without a material KS or SI gain. See the revised F6 and the formal synthesis.

* **F7 — Ω saturation is a failure mode.** When Ω saturates to uniform (+1/3),
  the regularizer has no pair-specific information left; a plausible
  non-relational explanation for MTRL ≤ baseline at `smp`+25L. Not yet shown to
  be a cause — a diagnostic question.

* **F8 — raw pairwise transfer is confounded by optimizer exposure.** DG-0001's
  large ER-directed gains disappeared or reversed after approximate update
  matching. Equal epochs do not imply equal optimization opportunity when task
  sets change concatenated dataset size and effective task minibatches.

* **F9 — ER gradient-norm dominance is reproducible (ESTABLISHED, refined).**
  Across matched baseline/MTRL seeds 0–4, ER dominated the smallest task norm by
  7.24×/8.96× in baseline middle/late training. MTRL remained at 7.13×/9.00×
  with no consistent paired reduction. No seed showed persistent pairwise
  conflict. Ω magnitude saturated in all seeds; one seed flipped both ER edges.
  **Refined by F10: the dominance holds only under the standard training
  mixture.** Raising ER's per-batch share to KS scale removes it in 5/5 seeds,
  so it is a sampling-regime property, not task-intrinsic.

* **F10 — Gradient scale is a training-mixture property (ESTABLISHED,
  2026-09-22).** DG-0005 matched seeds 0–4: late max/min task-norm ratio
  `8.962 ± 0.456 → 2.352 ± 0.270` (paired `−6.610`, `[−7.353, −5.867]`) when ER's
  sampled share rose from ≈`47/2048` to ≈`435/2048` with exposure, data and
  evaluation held fixed. ER-localized (late ER norm `6.972 → 1.608`; KS `+0.066`,
  SI `−0.117`), no conflict signal. Gradient scale must therefore be reported
  with its sampling regime, like relation magnitude (F5) and confidence (F6).

* **R1 — DG-0001 created the first single/pairwise runs.** The dynamic task path
  is exercised; its raw matrix is preserved but is not a semantic transfer
  target because of F8.

---

# Candidate Failure Modes of Classical MTRL

These are the falsifiable sub-hypotheses behind the immediate research question
stated under "Current Research Phase". They form an inventory of what to test —
not a method plan, and not a commitment to any one mechanism.

Note also that the failure may be *non-relational*: F7 records that Ω can
saturate to uniform (+1/3 everywhere), at which point the regularizer degenerates
into an uninformative uniform coupling penalty. The first diagnostic study must
rule this in or out before relational explanations are accepted.

Candidate explanations to test include:

1. task relations may be asymmetric;
2. useful relations may be sparse rather than dense;
3. relation confidence may differ strongly by task pair;
4. ER's low-data/noisy setting may make directly learned relations unreliable;
5. task relationships may vary across representation depth;
6. task relationships may change during training;
7. forcing all tasks into one relation structure may create negative transfer;
8. parameter summaries used to construct the relation matrix may inadequately
   represent task behaviour.

DG-0001 weakens the asymmetry explanation: its raw directed matrix was dominated
by task-count-dependent optimizer exposure (F8), and no positive semantic
transfer residual survived step matching. DG-0002 rejects persistent pairwise
gradient conflict under the matched protocol but establishes a reproducible
optimization-scale limitation (F9). DG-0005 then resolved the data-regime
branch: the imbalance is mixture-dependent (F10), so it cannot motivate a
mechanism as a task-intrinsic property. Parameter-summary inadequacy and causal
Ω/transfer mismatch remain untested; the ER convergence-vs-estimator-variance
split remains unresolved.

Each candidate is only admissible as motivation for a mechanism after a
diagnostic study (`DG-xxxx`) has produced evidence for it, and the mechanism must
then come from a published method (`LT-xxxx`) — see DEC-0005 and `BACKLOG.md`'s
gating rules.

---

# Next Research Action

**DEC-0009 (human decision) is recorded.** Strict Task Relation Learning scope
is retained; Option 2 (optimization-aware MTL) is declined; Option 3 is deferred
behind two gates — environment support **and** explicit human authorization at
that time. The GradNorm-style diagnostic control arm is not authorized.

DG-0005's matched confirmation is complete: ten runs at commit `8032a937`, five
seed pairs, all exposure gates passed. A1's late ratio was below the
pre-registered `3.0` dominance threshold in `5/5` seeds while A0's stayed above
it (`8.962 ± 0.456 → 2.352 ± 0.270`, paired `−6.610 [−7.353, −5.867]`). Study
decision **CONFIRMED**; see F10 and `studies/DG-0005/analysis.md`.

No further autonomous GPU work is justified by this result. F9 no longer
supports a task-intrinsic-scale mechanism rationale, and the remaining
estimator-variance-versus-convergence question is a *new* diagnostic that needs
its own pre-registered Study and a human decision about whether it is worth
running under DEC-0009. No ER performance claim is made: this diagnostic used
the speaker-leaky split for context only, and its ER accuracy direction is
consistent with memorization (ER train ≈0.99 vs validation ≈0.82).

---

# Evaluation Strategy

## Screening

Purpose:

Reject weak ideas cheaply.

Typical protocol:

* one controlled seed;
* same pooling/configuration as matched baseline;
* KS/SI/ER metrics;
* task-relation diagnostics;
* cheap ER single-split result may be inspected but is not a final claim.

A screening result cannot establish a research conclusion.

Protocol requirements that bind *every* run, screening included:

* matched pooling / layer selection / data / epochs / optimizer / evaluation
  protocol against the arm it is compared with (F2) — `02-lnp` exists as the
  control precedent;
* an explicit `--seed` (unseeded runs are not comparable at all);
* Study ID, stage and git commit SHA recorded in MLflow (DEC-0006);
* `df -h /` and `nvidia-smi` checked before launch, and at most two training
  jobs running at once (project rule).

---

## Confirmation

For a promising method:

* matched baseline;
* same seed set;
* seeds `0,1,2,3,4` unless OBJECTIVE.md changes this;
* mean and standard deviation;
* per-task results;
* aggregate results;
* task-relation diagnostics;
* statistical comparison where appropriate.

---

## ER confirmation

Any serious claim involving ER must include appropriate speaker-independent
evaluation.

LOSO takes precedence over the speaker-leaky historical split when the two
disagree.

---

# Two-GPU Policy

The machine contains two physical GPUs.

Use them deliberately.

Preferred usage:

### Screening

* GPU 0 — candidate
* GPU 1 — matched control or directly comparable experiment

### Confirmation

Distribute seeds across GPUs:

* GPU 0: seeds 0, 2, 4
* GPU 1: seeds 1, 3

Do not launch unrelated experiments merely because a GPU is free.

Parallel experiments should belong to the same scientific question whenever
possible.

---

# Research Record Policy

Each scientific hypothesis receives a Study ID.

Study IDs:

* `BL-xxxx` — baseline / reproduction
* `DG-xxxx` — diagnostic study
* `TR-xxxx` — Task Relation Learning mechanism
* `AB-xxxx` — ablation
* `LT-xxxx` — experiment derived directly from literature investigation

A Study is not equivalent to one MLflow run.

For example:

`TR-0012`

may contain:

* screening seed 0;
* confirmation seeds 0–4;
* ablations;
* LOSO evaluation.

All runs must reference their Study ID.

---

# DagsHub / MLflow

DagsHub/MLflow is the canonical execution record.

Every training run must record:

* Study ID;
* stage;
* method;
* seed;
* task set;
* pooling;
* layer configuration;
* git commit;
* relevant hyperparameters;
* metrics;
* artifacts;
* concise run note.

The repository research records contain scientific interpretation.

DagsHub contains execution evidence.

The same important result should be traceable between both.

---

# Current Pending Work

Mechanism work is **authorized for benchmarking** under DEC-0013 (published methods
only), and DEC-0014 activates Option 3 for the directed-relation question:

* `LT-0002` — **complete** (2026-09-22); 15 papers verified across the two selected
  families, verdicts and eligibility gates in `studies/LT-0002/analysis.md`.
  Family B has a clean published pass (MSSL); family A has none.
* `TR-0007` (MSSL, published arm) — **pre-registered 2026-09-29; `DEC-0015` (human) chose
  Option A, the faithful published formulation.** The arm is now wired:
  `04-mssl/README.md`, `04-mssl/mssl_config.yml`, the `mssl` dispatch in
  `run_improvements.py` and `studies/TR-0007/PLAN.md` exist, and the Ω step implements
  Eq. (4b)/Eq. (8) with `λ₀ = 1`, the `1/d` applied inside the solver for an explicit `d`,
  off-diagonal ℓ₁ per the cited graphical lasso, and its own primal–dual optimality
  certificate. The gate's equation was a mis-transcription (Eq. 4b's barrier with Eq. 8's
  data term); the primary source governs, the literature card is corrected and the historical
  gate text is preserved in `studies/TR-0007/NOTE.md`. The three formerly red solver tests
  are fixed — the analytic gradient was transposed, the bit-exact-zero assertions tested the
  wrong variable, and the ADMM silently failed to converge at the summary's 1e-4 scale (it
  returned `Ω ≈ 63·I` where the optimum is `Ω ≈ 7e4·I`); the module is back inside
  `make check`. Two inputs remain the researcher's: the **λ₂ rule** (`PLAN.md` pre-registers
  validation selection from the paper's own grid `{0.01, 0.1}`, five-run screen, against a
  fixed value at three runs) and the **compute envelope** (not created). Plan and exact-commit
  preflight are prepared; nothing was submitted.
  * Record conflict: DagsHub already holds `TR-0007__screen__*` runs (seed 0, commits
    `10aaaea3…`/`3df542d…`, experiment `taskrelation-variant-benchmark`) from a code line
    absent from this repository's history — `p-mssl` 0.9607 vs `classical-mtrl` 0.9744 and
    `wavcse-baseline` 0.9737 (`test_epoch_acc_all`), all tagged `rejected`, plus a
    `scale-corrected` `p-mssl-correlation` run at 0.9644. Retained as evidence and listed in
    `studies/TR-0007/NOTE.md`; the human decides whether the corrected implementation still
    warrants its own screen.
* `TR-0008` (directed relation, project-original) — activated by DEC-0014 §5, **not
  registered**. Its exact relation rule is design work the policy classes as introducing a
  project-original mechanism; it needs either an explicit authorization to design it
  autonomously or a design review (Option C of the TR-0007 gate).
* `DG-0007` (MTRL implementation faithfulness) — **registered 2026-09-29, pre-registered;
  `BLOCKED`. Screen not executed; no compute authorized.** It is the successor to the
  independent theory-to-implementation audit of the classical MTRL arm
  (`audits/2026-09-29-mtrl-theory-to-implementation-audit.md`): every evidence-carrying
  MTRL config sets `model.normalize_w: true`, so the published Ω closed form (TKDD 2014
  Eq. (14)) and the published relation penalty were evaluated on a row-unit-normalized
  copy of the task parameter matrix rather than on `W` itself — the code's Ω sits ≈50 %
  above the published subproblem minimum for the parameters the model trains, and the
  penalty loses its degree-two scaling. The arm is therefore **normalization-corrected
  MTRL, not fully faithful Zhang & Yeung MTRL**: the correction removes that one
  deviation (the audit's `D2`) and leaves the declared mean-head adapter (`D1`), the
  absent `λ₁` term (`D3`), warmup (`D4`), the ε-regularized inverse (`D6`) and the
  per-epoch Ω cadence (`D7`) in place. Independent variable: `model.normalize_w`
  `true → false`, one boolean, same code path. Three arms — normalization-corrected
  MTRL, the historical MTRL control and the matched wavCSE baseline — at `smp` 0.5 over
  all 25 layers, screen seed 42 then confirmation seeds 0–4; LOSO is a conditional
  escalation with a separate, currently unmeasured budget. Predecessor: the historical
  MTRL campaign (`LEGACY-PRE-ID` evidence within `DG-0001`/`DG-0002`).
  `studies/DG-0007/` holds the pre-registration (`PLAN.md`, `NOTE.md`), the three-arm
  execution configs with their `research:` identity blocks (`configs/`, verified by
  `research/tests/test_dg0007_run_identity.py`) and the pre-registered runtime gate
  (`check_runtime_faithfulness.py`). It requests no compute and **may not be launched**
  until the authorization question is answered. The `VARIANT_BENCHMARK_PROTOCOL.md` §2
  question is **resolved** (`DEC-0017`, human, 2026-09-29): historical MTRL is **retained** as
  the §2 in-category control and reproducibility anchor, and normalization-corrected MTRL
  remains a distinct successor experimental arm — so `DG-0007` is now blocked on
  **authorization coverage alone**.
* the exact commit is **local-only**: `origin` publishes
  `feature/mssl-task-relation-study` at `05fa10c`, so any worker request is refused until
  the developer pushes the commit `preflight` resolves. Publication is a developer action,
  never performed by a cycle.
* `TR-xxxx` variant Studies — one shared matched protocol, screen then confirm.
* Deferred (not cancelled): the `DG-xxxx` diagnostics below.

* DG-0001 — complete; optimizer exposure dominates its raw transfer matrix (F8);
* DG-0002 — **CONFIRMED**; ER gradient-scale dominance reproduces across seeds,
  MTRL does not mitigate it, and persistent pairwise conflict is rejected (F9);
* LT-0001 — **REJECTED**; the F9 literature gate found no eligible published
  mechanism (DEC-0008, superseded by DEC-0009);
* DG-0007 — **registered, pre-registered, `BLOCKED`**; normalization-corrected MTRL successor to
the audit's D2 finding, screen not executed, no compute authorized;
* DG-0005 — **CONFIRMED** (2026-09-22); matched seeds 0–4 removed late ER
  norm dominance in 5/5 seeds under a passing exposure gate. Produced F10 and
  refined F9. It authorizes no mechanism by itself — the benchmark's authority
  comes from DEC-0013, which is a human decision;
* DG-0003 — partial Ω/transfer discrepancy only; causal interpretation remains
  blocked by protocol mismatch;
* DG-0004 — retrospective stability complete; prospective prediction remains;
* DG-0006 — cross-diagnostic ER fold controls remain.

Future transfer diagnostics must control optimizer steps, effective per-task
batch size, loss scaling, epoch budget and checkpoint policy (F8). No relation
mechanism may cite F9 — that rationale is withdrawn (DEC-0010). Mechanism work
now proceeds under DEC-0013, which requires a verified published source for
every variant, keeps the category boundaries (no loss weighting, gradient
surgery, low-rank, clustering or decomposition), and fixes one shared matched
protocol across arms.

---

# Current Champion

No Task Relation Learning method is currently considered a confirmed
significant replacement for the matched wavCSE baseline.

Until multi-seed evidence demonstrates otherwise:

**Champion: matched wavCSE baseline**

MTRL is the branch's formal baseline *method* (DEC-0001) — the thing under
diagnosis — not a champion. GBC is archived and not eligible (DEC-0003).

---

# Last State Update

Update this section after every completed research cycle.

Last fully completed Study:

`DG-0005 — ER data-regime gradient-scale control` (2026-09-22), decision
**CONFIRMED**. Ten matched runs at commit `8032a937`; F10 established and F9
refined. `LT-0002 — published relation-learning variants for two selected
families` (2026-09-22, decision `COMPLETE - CANDIDATES_FOUND`, 0 GPU-hours) is
the most recent completed literature Study; `LT-0001` remains the last rejected
one.

Most recent completed execution stage:

DG-0005 matched A0/A1 confirmation, seeds `0,1,2,3,4`, ten runs at commit
`8032a937050d8bbd3114b172cb813a8fc7370b37`, decision **CONFIRMED**.

Current active Study:

`NONE` RUNNING. `TR-0007` is registered (`BLOCKED`, pre-registration) and carries the
faithfulness gate described under *Current Pending Work*; `TR-0008` is named and activated
by DEC-0014 but not registered. `DG-0007` is registered (`BLOCKED`, pre-registration) with
no compute authorized for its scope. Every other `TR-xxxx` still requires its own explicit
human authorization under DEC-0009/DEC-0010.

Result (CONFIRMED, F10):

DG-0005 A1 removed ER's late norm dominance in `5/5` seeds under a passing
exposure gate: late ratio `8.962 ± 0.456 → 2.352 ± 0.270` (paired `−6.610`,
`[−7.353, −5.867]`); middle `7.243 ± 0.272 → 2.818 ± 0.272`, below `3.0` in
`4/5` seeds (one seed `3.054`). The reduction is ER-localized (late ER norm
`6.972 → 1.608`; KS `+0.066`, SI `−0.117`) and no conflict signal appears.
A0 reproduced DG-0002's baseline numbers exactly, confirming the knob is
default-off. Consequence: gradient scale is a training-mixture property (F10),
so F9 cannot justify a task-intrinsic-scale mechanism.

Unresolved questions:

1. Which mechanism carries the late-phase part of the ER norm drop? The
   pre-registered bound (DEC-0012) already excludes estimator size for ≥20% of
   the late log-drop, while the middle phase is fully estimator-consistent; what
   is not separated is a smaller mean gradient (ER head saturating under ~9× more
   updates: train ≈0.99 vs validation ≈0.82, final train−val gap `0.134 → 0.174`
   at the screening seed) versus noise growing faster than `1/√n`. Per-step
   dispersion cannot answer it, and the untriggered A2 arm would not either.
2. Whether any of this justifies Option-3 work was a human scope decision under
   DEC-0009/DEC-0010; DEC-0014 has since answered it for the directed-relation
   question (Option 3 activated; project-original labelling mandatory).

Next recommended action:

**TR-0007: answer the λ₂ rule, then authorize the screen.** Stage 1 (`LT-0002`) and the
faithfulness gate (`DEC-0015`, Option A) are closed. Controller-side work is complete: the arm
is wired, all 25 layers are enforced executably
(`research/tests/test_tr0007_protocol.py`), the published Eq. (8) convention is pinned by
tests that are back inside `make check`, MLflow/DagsHub credentials were verified with a
zero-cost read-only authentication check against the real endpoint, and the deterministic
screen plan plus the exact-commit preflight are prepared. Two human inputs remain: **(1)** the
λ₂ rule — the source paper publishes no default and selects λ₂ on data, so `PLAN.md`
pre-registers validation selection from the paper's grid restricted to `{0.01, 0.1}`
(five-run screen, budget-matched to the control's two-value λ selection) with a
researcher-fixed value as the three-run alternative; and **(2)**
`authorizations/TR-0007.yaml`, which does not exist. After those, the screen is
submittable as registered (seed 42, three arms, one run per arm under
`VARIANT_BENCHMARK_PROTOCOL.md`, screen then confirm, LOSO for any ER claim). `TR-0008`
(directed relation, project-original) still needs its own authorization to be designed, and
the deferred `DG-xxxx` diagnostics stay on the backlog.

Latest iteration (2026-09-29, no compute): **§2 control identity decided — RETAIN; `DG-0007`
stays `BLOCKED` on authorization alone.** The researcher resolved the open
`VARIANT_BENCHMARK_PROTOCOL.md` §2 question as `DEC-0017` (OPTION B): historical MTRL
(`mtrl_poolingwinner_25L_config.yml`, `normalize_w: true`) **remains** the in-category control
and reproducibility anchor, and normalization-corrected MTRL
(`mtrl_norm_corrected_25L_config.yml`, `normalize_w: false`) remains a **distinct `DG-0007`
successor experimental arm**. Historical MTRL is neither replaced nor redefined. §2.1's text is
unchanged and now carries a dated decision note; `DG-0007`'s three-arm design, seeds (screen 42;
confirmation `0–4`), all-25-layer policy and `smp` 0.5 are untouched, and the study stays
`BLOCKED` / `pre_registration` with `blocked_on` narrowed to authorization coverage. No compute
authorization was created, nothing was submitted or provisioned, and no metric of any run was
touched. Whether the corrected configuration should later *become* the standing control stays a
decision for `DG-0007`'s own evidence.

Previous iteration (2026-09-29, no compute): **DG-0007 accepted into canonical; still
`BLOCKED` on authorization.** The theory-to-implementation audit of the classical MTRL arm
and its remediation were integrated from the read-only audit branch
(`research/mtrl-theory-audit`, accepted through `46fc0f9`, verdict
`DG_0007_MICRO_VERIFY_PASSED`) into `feature/mssl-task-relation-study`; the audit's
stale-branch MSSL text was **not** taken — this repository's `DEC-0015`/`DEC-0016`
semantics and its corrected `literature/goncalves-2016-mssl.md` govern. What arrived is the
`DG-0007` registration above, its pre-registration (`studies/DG-0007/PLAN.md`, `NOTE.md`),
its six study configs, its runtime-faithfulness checker, the corrected MTRL config
`01-mtrl/mtrl_norm_corrected_25L_config.yml`, the audit report and three research test
modules. The arm's label is **normalization-corrected MTRL**, never "faithful Zhang & Yeung
MTRL" (`DEC-0015` governs MSSL; this governs the MTRL arm). No GPU run, no worker, no job,
no S3 or network-volume mutation, and no authorization was created for DG-0007:
`authorizations/TR-0007.yaml` is the only envelope on the record and it excludes DG-0007 by
name. Its runtime gate is a checker fix (malformed evidence renders a bounded structured
diagnostic and exits `2`; a scientific failure stays `1`); no scientific semantics changed,
and no historical evidence or metric was rewritten.

Previous iteration (2026-09-29, no compute): **TR-0007 Option A implemented and pre-registered.**
No GPU run, no worker, no job, no S3 or volume mutation, no authorization created. The human
gate on `TR-0007` was answered (`DEC-0015`: Option A, faithful published MSSL formulation) and
recorded append-only in `studies/TR-0007/NOTE.md`, with a primary-source correction of the
equation the gate had been stated against (the paper's Ω step is Eq. 4b
`λ₀ tr(WΩWᵀ) − d log|Ω| + λ₂‖Ω‖₁` = Eq. 8 `λ₀ tr(SΩ) − log|Ω| + (λ₂/d)‖Ω‖₁`, `S = (1/d)WᵀW`,
`λ₀ = 1` in all experiments; the card and `DEC-0015` carry the corrected text, the historical
gate text is preserved). The arm now implements that objective: Eq. (8) with the `1/d` applied
inside the solver for an explicit `d = W.shape[1]`, off-diagonal ℓ₁ per the graphical lasso the
paper cites, ADMM with Boyd et al. §3.4.1 residual balancing in float64, the split variable `Z`
returned so the ℓ₁ support is exact, and a primal–dual optimality certificate per solve. The
`mssl` arm was wired into `run_improvements.py` with its own config (`taskrelation-mssl`,
`study_id`/`stage`/`method` research tags) and README; all 25 layers are enforced executably,
not only in prose. The three formerly red solver tests were corrected: the analytic coupling
gradient was transposed (`summary @ omega` for `2λ₀ Ω W`), the sparsity assertions demanded
bit-exact float32 zeros from the variable that does not carry the ℓ₁ support, and the old ADMM
was shown to fail silently at the summary's 1e-4 scale — it returned `Ω ≈ 63·I` against an
optimum of `Ω ≈ 7e4·I`, so a scale-free initialization, residual balancing and float64 were
required. DagsHub authentication was verified zero-cost and read-only against the real
endpoint; the screen plan and preflight are prepared but **nothing was submitted**, and
`authorizations/TR-0007.yaml` still does not exist. A record conflict was found and is
recorded rather than smoothed over: the tracking repository already holds `TR-0007__screen__*`
runs from a code line absent from this repository's history (seed 0; `p-mssl` 0.9607 vs
0.9744/0.9737 for the controls; all `rejected`). Two inputs remain the researcher's: the λ₂
rule and the compute envelope. The screen's exact commit is now `8b40eede8ceddfd4209ca0c3ef503786461c7f21`, published to `origin/feature/mssl-task-relation-study` and verified worker-fetchable by the exact-commit preflight (`available: True`, mechanism `ls-remote-ref`); the deterministic plan renders three job specs (`TR-0007__screen__{p-mssl,classical-mtrl,wavcse-baseline}__s42`), each carrying its arm's method, layer policy and declared inputs, and nothing was submitted.

Previous iteration (2026-09-29, no compute): **ARC v1 zero-cost integration exercise,
research-side.** No GPU run, no worker, no job, no authorization. Established: the deployed
compute backend resolves the `wavcse-infra` checkout, its read-only verbs (`status`,
`resolve`, `preflight`, `reap` dry-run) reach the control plane, and the controller had no
ARC state root, no leases and no envelope. Installed and enabled the crash-independent
reaper timer on the controller (`reap-install --install --enable`; service
`wavcse-arc-reaper.service`, timer `wavcse-arc-reaper.timer`, 300 s, `Persistent=true`); its
first executing pass reconciled zero scopes and stopped nothing. Two unambiguous ARC
integration defects were found, fixed with regression tests and committed: a sharded
canonical artifact could not be declared (the loader-layout contract accepted exactly one
archive per dataset, while the artifact pipeline must shard above the 5 GB single-PUT
ceiling), and the layout extractor required dataset-prefixed TAR members while the
published archives are dataset-relative (`Session1/…`, `speech_commands_v0.01/…`), which
would have refused every canonical artifact before training; the installer also reported a
privilege boundary as a traceback instead of a refusal. The manifest's recorded
loader-root layout is now validated against the plan's declared inputs before an entry can
become COLLECTED. Input readiness was re-established read-only for all three datasets
(`speechcommand` 5 shards, `voxceleb` 9 shards, `iemocap` 1 tar: 15 objects, every shard
manifest digest cross-checked against its set manifest, every object verified at its
declared size), and `embedding_layout.loader_pooling` on the protocol config resolves
`wavlm_large/mean`, matching the manifests' recorded layout. `preflight` reports the exact
commit is **not** available to workers, so publication is required before any worker
request. `TR-0007` was registered and blocked at pre-registration on a faithfulness
decision (see *Current Pending Work*); `DG-0005`'s analysis and all earlier records are
unchanged, and no metric of any run was touched.

Previous iteration (2026-09-22, no compute): pre-registered post-hoc bound on
DG-0005's ER norm drop (`studies/DG-0005/analyze_noise_shape.py`, pre-registered
at `2ac7f3d`, results in `noise_shape_result.json`). Estimator-size scaling
explains the middle-phase drop on its own (share `1.03` `[0.975, 1.082]`) but at
most `0.758` `[0.715, 0.801]` of the late drop, so ≥20% is not estimator size;
per-step dispersion is proven unusable as a noise proxy (`CV ≈ 0.32` vs an
isotropic ceiling of `≈ 0.00057`); DG-0002's baseline reproduces the A0
statistics value for value. Recorded as DEC-0012 and inside F10 — post-hoc, no
new finding, no GPU work. No metric of any run was changed. Commits:
`2ac7f3d` (pre-registration), `27dee9f` (results and ledger updates). The ten
DG-0005 confirmation runs on DagsHub keep `study_decision=CONFIRMED` and their
run note was re-pushed with the post-hoc bound (DEC-0012).

Earlier iteration (2026-09-22, no compute): analysis/synthesis pass. Created
`FRAMEWORK.md`, refreshed `MTRL_DIAGNOSTIC_SYNTHESIS.md` so it no longer
prescribes the closed DEC-0007 sequence, annotated the TR-0002/TR-0003/TR-0004
gates, and recorded DEC-0011. No Study was created, no run was launched and no
metric was changed; every change is documentation, committed as
`be67b6a89e197f18e2fd28b7ceb15e64f795e1b0` (documentation only — it changes no
training code, so the DG-0005 confirmation runs still report `8032a937`).

GPU jobs still running: `NONE`. All DG-0005 queues finished; no tmux training
session remains. The queueing helper `wait_for_gpu1_confirmation.py` was
deleted after use; `run_confirmation.py` remains as the Study's stage runner.
Every confirmation run records commit `8032a937050d8bbd3114b172cb813a8fc7370b37`;
HEAD may advance again. The scientific-interpretation record (this file,
`FINDINGS.md`, `DECISIONS.md`, `FAILURES.md`, `BACKLOG.md`, `STUDIES.jsonl`,
`studies/DG-0005/*`) was committed as `b485be48f6bdb2ec6238b6ee7f5a5ce2cca0357f`
— a documentation commit that changes no training code, so it does not affect
the runs above.

Current consecutive unsuccessful mechanism studies:

`1` — classical MTRL (F4). Diagnostic and literature studies do not increment
the plateau counter.

Literature-search trigger:

**CLOSED for F9 (LT-0001, FL-0003, DEC-0011).** The mandate was executed and
returned a negative result; do not re-run it against the same evidence. Re-enter
literature mode only for a new, separately evidenced limitation, or if the human
reopens scope under DEC-0009.
