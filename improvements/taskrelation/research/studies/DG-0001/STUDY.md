# DG-0001 — Empirical directed task-transfer matrix

Status: ACTIVE — Stage A preparation
Started: 2026-09-21
Research family: Task Relation Learning
Method under diagnosis: classical MTRL
Execution method: matched plain wavCSE controls

## Question

Is useful transfer among KS, SI and ER directional or pair-selective, making classical MTRL's single symmetric dense covariance matrix structurally inadequate?

## Falsifiable hypothesis

At least one directed transfer cell

`T(A <- B) = accuracy(A trained with B) - accuracy(A trained alone)`

will either:

1. differ materially from the reverse cell `T(B <- A)`; or
2. remain practically indistinguishable from zero while another pair shows material transfer.

The null for Stage A is that the six cells are all near zero or form a symmetric dense pattern. Either null outcome weakens the claim that MTRL fails because Ω cannot represent task relations.

## Scope

Stage A changes only the task set. Every arm uses:

- fixed wavCSE/WavLM embeddings;
- plain wavCSE downstream model, not a new Task Relation Learning architecture;
- `smp` pooling, parameter 0.5;
- all 25 transformer layers;
- seed 42;
- 30 epochs;
- identical optimizer, scheduler, dataset split and evaluation settings;
- commit SHA recorded by MLflow;
- Study ID, stage and run note tags.

Arms:

- single-task: `ks`, `si`, `er`;
- pairwise: `ks_si`, `ks_er`, `si_er`;
- matched triple control: `ks_si_er`.

The triple is rerun rather than inherited from legacy evidence because the historical runs predate Study IDs and the registered protocol.

## Primary outcome and checkpoint policy

The primary transfer estimate uses `test_epoch_<task>_acc`, the fixed final-epoch checkpoint. This avoids selecting different epochs with an auxiliary-dependent aggregate validation criterion.

`opt` and `best` results are retained as sensitivity analyses only:

- `best` selects by sample-weighted validation accuracy across the arm's tasks;
- `opt` selects by macro-average validation accuracy across the arm's tasks;
- both criteria change when the auxiliary task changes and therefore cannot be the primary directed-transfer estimate.

## Screening interpretation

Stage A is one seed and cannot confirm an improvement.

- `|T| < 0.002` (0.2 percentage points): practically near zero at the project's materiality threshold;
- `0.002 <= |T| < 0.010`: ambiguous screening signal; do not interpret without matched multi-seed replication;
- `|T| >= 0.010`: material screening signal, still requiring seeds 0–4 before a finding;
- candidate asymmetry requires `|T(A <- B) - T(B <- A)| >= 0.010` and at least one direction with `|T| >= 0.010`.

These thresholds rank follow-up; they are not significance tests.

ER uses the historical speaker-leaky split in Stage A. Every ER cell is screening-only regardless of magnitude. A serious ER transfer claim requires matched LOSO confirmation.

## Known design limitations

1. One epoch exposes each target to its own dataset once in both single and pairwise arms, but pairwise arms contain additional auxiliary batches and therefore more optimizer updates. This is part of the operational transfer intervention, but compute and transfer are not separately identified.
2. The trainer scales active loss by `1 / num_tasks`; single and pairwise arms therefore differ in loss scale. AdamW is approximately but not perfectly scale-invariant, and regularization terms are not scaled identically. Record this as an alternative explanation.
3. Scheduler input is aggregate validation loss and therefore depends on the arm's task set. The fixed-epoch primary checkpoint removes selection bias but not learning-rate-path differences.
4. Pairwise training changes the number and ordering of batches. Multi-seed confirmation is required for any candidate effect below one percentage point and for every promoted conclusion.

These limitations prevent a causal claim that auxiliaries alone produce an observed difference. If Stage A is decisive, the confirmation design must add the minimum control needed for the implicated pair rather than generalizing from one seed.

## Prerequisite smoke test

Before full training:

1. run `ks` with `configs/smoke.yml`;
2. run `ks_si` with `configs/smoke.yml`;
3. confirm dynamic label width, classifier count, training, checkpointing and per-task evaluation;
4. exclude smoke metrics from the transfer matrix.

## Decision branches

- all cells near zero: symmetric/dense representation is not the primary issue; prioritize saturation or optimization diagnostics;
- symmetric dense transfer: Ω is representationally adequate; test whether it estimates and uses the structure;
- candidate asymmetry: confirm only the implicated directions over seeds 0–4, with LOSO for ER;
- candidate selective transfer: compare the pattern with existing MTRL Ω, then run only the minimum DG-0003 arm needed to locate estimation versus regularization failure.

No new architecture or literature-derived mechanism is selected in this Study.

## Deliverables

- `result.json` — run IDs, metrics, directed cells and decision;
- `../../task_relations/empirical_transfer.json` — machine-readable directed matrix;
- `analysis.md` — evidence-level interpretation and Ω comparison;
- updates to `STUDIES.jsonl`, `FINDINGS.md`, `STATE.md`, `BACKLOG.md` and `FAILURES.md` only when justified.
