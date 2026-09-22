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

Binding scope decisions: `DECISIONS.md` (DEC-0001 … DEC-0012). Established
findings: `FINDINGS.md` — authoritative over the one-line summaries below.

---

# Current Research Phase

**Phase:** **framework synthesis** (DEC-0011). DG-0005 is CONFIRMED (2026-09-22)
and F9's gradient-norm dominance is a training-mixture property, not
task-intrinsic (F10). The F9-era literature gate is closed and negative
(`LT-0001`/FL-0003). No mechanism Study exists or is authorized; the programme
waits on a human decision about the Option-3 gate (DEC-0009/DEC-0010) and
consumes no compute meanwhile.

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

**Immediate research question (human-gated):**

> Which of the three documented options does the programme take — (a) a bounded
> diagnostic separating gradient-estimate variance from ER convergence under
> DG-0005's matched composition, (b) close-out with the framework as the
> characterisation result, or (c) explicit Option-3 authorization with a
> rationale that does not rest on F9? Only (a) or (c) would consume compute; the
> framework does not require either.

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

No mechanism work:

* DG-0001 — complete; optimizer exposure dominates its raw transfer matrix (F8);
* DG-0002 — **CONFIRMED**; ER gradient-scale dominance reproduces across seeds,
  MTRL does not mitigate it, and persistent pairwise conflict is rejected (F9);
* LT-0001 — **REJECTED**; the F9 literature gate found no eligible published
  mechanism (DEC-0008, superseded by DEC-0009);
* DG-0005 — **CONFIRMED** (2026-09-22); matched seeds 0–4 removed late ER
  norm dominance in 5/5 seeds under a passing exposure gate. Produced F10 and
  refined F9. No mechanism is authorized by it;
* DG-0003 — partial Ω/transfer discrepancy only; causal interpretation remains
  blocked by protocol mismatch;
* DG-0004 — retrospective stability complete; prospective prediction remains;
* DG-0006 — cross-diagnostic ER fold controls remain.

Future transfer diagnostics must control optimizer steps, effective per-task
batch size, loss scaling, epoch budget and checkpoint policy (F8). No new
relation mechanism may cite F9, and no mechanism may be implemented at all,
until a human records an explicit authorization under DEC-0009.

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
refined. `LT-0001` remains the last completed literature Study (REJECTED).

Most recent completed execution stage:

DG-0005 matched A0/A1 confirmation, seeds `0,1,2,3,4`, ten runs at commit
`8032a937050d8bbd3114b172cb813a8fc7370b37`, decision **CONFIRMED**.

Current active Study:

`NONE`. No `TR-xxxx` is open and none may open without explicit human
authorization under DEC-0009/DEC-0010.

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
2. Whether any of this justifies Option-3 work is a human scope decision under
   DEC-0009/DEC-0010, not an autonomous one.

Next recommended action:

**None requiring GPU work.** What remains is a human choice among the three
documented options in `FRAMEWORK.md` §6: (a) the bounded late-phase
mean-gradient diagnostic (narrowed by DEC-0012 to measuring `‖E g‖` on the
existing arms), (b) close-out with the framework as the characterisation result,
or (c) explicit Option-3 authorization with a rationale that does not rest on
F9. Option (a) would be pre-registered as its own `DG-xxxx` Study before any run,
and any ER performance claim afterwards would still require LOSO (F3).

Latest iteration (2026-09-22, no compute): pre-registered post-hoc bound on
DG-0005's ER norm drop (`studies/DG-0005/analyze_noise_shape.py`, pre-registered
at `2ac7f3d`, results in `noise_shape_result.json`). Estimator-size scaling
explains the middle-phase drop on its own (share `1.03` `[0.975, 1.082]`) but at
most `0.758` `[0.715, 0.801]` of the late drop, so ≥20% is not estimator size;
per-step dispersion is proven unusable as a noise proxy (`CV ≈ 0.32` vs an
isotropic ceiling of `≈ 0.00057`); DG-0002's baseline reproduces the A0
statistics value for value. Recorded as DEC-0012 and inside F10 — post-hoc, no
new finding, no GPU work. No metric of any run was changed.

Previous iteration (2026-09-22, no compute): analysis/synthesis pass. Created
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
