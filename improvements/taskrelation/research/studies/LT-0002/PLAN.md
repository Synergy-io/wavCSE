# LT-0002 — Published relation-learning variants for two selected families

Status: RUNNING — source verification, no code, no GPU
Type: literature (source verification + card production)
Created: 2026-09-22
Research family: Task Relation Learning
Authorization: DEC-0013 (human re-scope: benchmark published MTRL variants; diagnostic-first sequencing deferred)
Parent context: `STATE.md` §Current Research Phase; families selected by the human after the DG-0005 status review

## Observation

Exactly one relation-learning method has ever been implemented and evaluated in
this branch: classical symmetric, dense MTRL (35 runs at `smp` 0.5 / 25 layers,
seeds 0–4, ER LOSO). `LT-0001`'s negative literature result was scoped to one
*rationale* (F9 gradient-scale imbalance), not to the variant families
themselves: SPATS was rejected because it targets many-task sparsity, MTGTP on
implementation assumptions, Rakitsch as non-portable without a novel hybrid.
The relation-method space is therefore unexplored rather than exhausted, and the
human has re-scoped the programme to benchmarking published variants
(DEC-0013).

## Research question

Which **published** Task Relation Learning methods (Zhang & Yang 2021 §2.4)
belong to the two selected families — (A) asymmetric / directed relations, and
(B) a better relation estimator or task-parameter representation — and can be
implemented *faithfully* against our setting: disjoint datasets, heterogeneous
classifier heads (12 / 1251 / 4 classes), fixed upstream WavLM embeddings, `smp`
0.5 pooling over 25 layers, three tasks?

## Hypothesis

At least one published method in each selected family has a formal assumption
that (i) is explicit Task Relation Learning, (ii) survives disjoint datasets
with unequal head dimensionality, and (iii) can be implemented without
inventing a new mechanism — so it can be screened against classical MTRL and
the matched wavCSE baseline.

## Competing explanation

Every candidate in these families fails at least one gate: directed relations
may require aligned per-task architectures or shared parameter columns; better
estimators may assume Gaussian/regression outputs, task-parameter comparability
that heterogeneous heads cannot provide, or may cross into low-rank /
clustering / decomposition / loss-weighting territory. In that case the honest
outcome is a negative literature result and a return to the human.

## Falsification condition

- **Pass** for a family if ≥1 candidate satisfies every eligibility gate below
  and its card states, concretely, how the relation object and the update rule
  map onto our three-head model with a matched-pair control.
- **Fail** for a family if no candidate satisfies all gates; record the negative
  result, keep the family's backlog entry blocked, and report back rather than
  implementing something adjacent.
- Partial passes with a *documented* deviation (a hybrid) are recorded as
  "not faithfully implementable" — that is a Fail, not a Pass, because the
  project's contribution claim requires faithful published methods (DEC-0013 §3).

## Eligibility gates (each candidate must pass all)

1. **Explicit relation object** — learns or uses a task relation structure
   (covariance/precision, directed transfer graph, similarity graph, learned
   estimator of one of these). Not merely shared layers or loss weighting.
2. **Taxonomy** — Task Relation Learning under the project's binding taxonomy;
   explicitly **not** low-rank, clustering, decomposition, uncertainty/loss
   weighting, or gradient surgery.
3. **Heterogeneous-head compatibility** — the formal assumption must survive
   disjoint datasets and different classifier widths (12 / 1251 / 4), or the
   card must state precisely what has to change and why that is still faithful.
4. **Fixed-representation compatibility** — implementable on frozen WavLM
   embeddings; no requirement to re-train upstream.
5. **Faithful implementability** — the update rule can be implemented as
   published, with hyperparameters taken from the paper or its natural defaults.
6. **Source verified** — claim checked against the primary publication (venue,
   year, equations), not a secondary summary. Attribution errors are a known
   project failure mode (GBC retraction, DEC-0003).

## Independent variable

The candidate method identity (per arm). Nothing else: pooling, layers, epochs,
batch size, seeds, optimizer, checkpoint policy and evaluation protocol are
fixed by the shared benchmark protocol.

## Control

Two controls per variant arm: **classical symmetric MTRL** (matched in-category
control — the thing the variant must beat to be interesting) and the matched
**wavCSE baseline** (the reference to beat, F4). A variant that beats neither
has not contributed.

## Controlled variables (fixed in every future variant arm)

`smp` 0.5 pooling; all 25 layers; task set `ks_si_er`; 30 epochs; global batch
2048; split handling unchanged; seeds 0–4 for confirmation (single explicit seed
for screening); identical optimizer/regularization settings; identical
evaluation and checkpoint protocol; ER claims require LOSO (F3).

## Evaluation protocol (for the future variant Studies this Study enables)

Primary: per-task and aggregate test accuracy at the protocol checkpoint, with
paired per-seed differences against both controls and 95% intervals; no material
regression on any task (project threshold 0.20pp). Secondary: relation
diagnostics (Ω trajectory or the variant's relation object), gradient behaviour,
validation curves. A single seed screens; five seeds confirm; LOSO for ER.

## Diagnostics

No training diagnostics here. This Study's diagnostics are paper-level: stated
assumption, relation object, update rule, hyperparameters, evidence in the
source, and the concrete mapping onto our model.

## Screening protocol

Not applicable to this Study (no runs). It defines the screen that the later
`TR-xxxx` Studies must use: one explicit seed, matched controls, exposure and
protocol checks before any comparison is interpreted.

## Confirmation protocol

Not applicable here; inherited by the `TR-xxxx` Studies (seeds 0–4, matched
controls, paired intervals, LOSO for ER).

## Compute estimate

0 GPU-hours, 0 MLflow runs. Literature only.

## GPU allocation

None.

## Expected information gain

Decisive for the re-scoped programme: it determines whether the benchmark can
start at all, and with which arms. Either outcome is informative — a verified
candidate set (with implementation notes) or a documented negative result that
forces the human to choose between the deferred diagnostics and closing out.

## Deliverables

Paper cards in `research/literature/` (one per verified candidate, following the
existing card format), an updated `literature/INDEX.md`, and `analysis.md`
listing the ranked candidate set per family with the gates each passed/failed
and a recommended first implementation arm.

## Explicit non-goals

No architecture implementation; no hyperparameter search; no new method
invented here; no changes to `01-mtrl/`; no re-run of the F9 search (that
prohibition stands, DEC-0011).
