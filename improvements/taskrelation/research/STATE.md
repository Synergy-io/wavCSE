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

Binding scope decisions: `DECISIONS.md` (DEC-0001 … DEC-0006). Established
findings: `FINDINGS.md` — authoritative over the one-line summaries below.

---

# Current Research Phase

**Phase:** MTRL diagnosis — confirming whether DG-0002's seed-42
shared-gradient norm imbalance is reproducible before selecting any mechanism.

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

**Immediate research question** — this, not "which architecture do we try next":

> Why does classical MTRL fail to produce a significant reproducible improvement
> over wavCSE for KS, SI and ER, and what measurable task behaviour indicates
> which Task Relation Learning assumption should replace or extend it?

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

* **F9 — ER gradient-norm dominance is a confirmation candidate (SCREENING).**
  In DG-0002 seed 42, baseline ER shared-gradient norms were 7.36× and 7.66×
  the smallest task norm in the middle/late thirds. MTRL did not mitigate the
  imbalance and Ω saturated uniformly. No pair showed persistent conflict.
  Matched seeds 0–4 are required before treating this as a limitation.

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
transfer residual survived step matching. DG-0002's seed-42 screen weakens
persistent pairwise gradient conflict but identifies ER gradient-norm dominance
as a confirmation candidate (F9). Data-size causation and parameter-summary
inadequacy remain untested; the norm result is not yet a mechanism gate.

Each candidate is only admissible as motivation for a mechanism after a
diagnostic study (`DG-xxxx`) has produced evidence for it, and the mechanism must
then come from a published method (`LT-xxxx`) — see DEC-0005 and `BACKLOG.md`'s
gating rules.

---

# Next Research Action

Diagnostics only. DG-0002's exposure-matched seed-42 screen is **PROMISING**:
ER shared-gradient norms dominated KS/SI in the middle and late thirds, while
MTRL did not mitigate the imbalance and Ω saturated to uniform +1/3. Persistent
pairwise conflict was not supported.

Continue the same Study with matched baseline and MTRL seeds `0,1,2,3,4`,
preserving the 30-epoch `smp` 25-layer protocol and diagnostic schedule. Treat
seed-level phase summaries—not the correlated within-run samples—as the
independent observations.

Do not enter next-method literature mode yet. If the norm pattern confirms, it
identifies a concrete optimization-scale limitation and should drive targeted
literature queries. If it fails, reject gradient interaction as the explanation
and return to Ω estimation / parameter-summary diagnostics. DG-0003 remains
incomplete because its Ω/transfer protocols are unmatched.

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

No mechanism work. Completed and pending diagnostics:

* DG-0001 — complete. F8 shows optimizer exposure dominates its raw directed
  matrix; it cannot justify asymmetry;
* DG-0002 — **ACTIVE / PROMISING**. Seed-42 paired screen complete; matched
  baseline/MTRL seeds 0–4 pending to confirm or reject F9;
* DG-0003 — partial Ω/transfer discrepancy only; causal interpretation remains
  blocked by protocol mismatch;
* DG-0004 — retrospective stability complete; prospective prediction remains;
* DG-0005/DG-0006 — ER data-regime and cross-diagnostic fold controls remain.

Future transfer diagnostics must control optimizer steps, effective per-task
batch size, loss scaling, epoch budget and checkpoint policy (F8). DG-0002
additionally records exact per-task batch exposure for every sampled step.

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

`DG-0001 — empirical directed task-transfer matrix` (2026-09-21).

Most recent completed cycle/stage:

`DG-0002 — paired gradient-compatibility screen` (2026-09-22), status
**PROMISING**. Both seed-42 arms finished; no GPU jobs remain.

Current active Study:

`DG-0002` — confirmation pending.

Important new finding:

F9 (SCREENING): ER's shared-gradient norm dominated KS/SI by 7.36–7.66× in
baseline middle/late training; MTRL did not mitigate it and Ω saturated. No
pair met the persistent-conflict threshold.

Unresolved question:

Does the norm imbalance reproduce across independent seeds, or is it a
seed/data-regime artifact?

Next recommended experiment:

Continue DG-0002 with matched baseline and MTRL seeds `0,1,2,3,4` under the
same 30-epoch `smp` 25-layer protocol. Confirmation is pending; do not start a
new Study, mechanism, or literature cycle first.

GPU jobs still running:

`NONE`.

Current consecutive unsuccessful mechanism studies:

`1` — classical MTRL (F4). The counter drives OBJECTIVE.md's plateau trigger;
MTRL counts as the first, and the programme formally starts here. Diagnostic
studies (DG-xxxx) do not increment it.

Literature-search trigger:

Two triggers, the second taking precedence:

1. the plateau criterion in OBJECTIVE.md; or
2. **mandatory** as soon as a diagnostic identifies a concrete MTRL limitation —
   targeted literature mode is a required stage of the formal progression
   (DEC-0005), not a fallback. Build queries from the observed failure mode, not
   from "best multi task learning model".
