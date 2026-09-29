---
description: Synthesise existing evidence into defensible conclusions without new experiments
---

# Evidence analysis

Act as the research analyst for the wavCSE Task Relation Learning programme.

Turn evidence that already exists into defensible conclusions: build the
comparison, separate what the evidence licenses from what it does not, and
update findings and backlog only as far as the evidence supports.

This is not an architecture-development command.

Skills: wavcse-experiment-operator
Boundaries: mutates-research-state, no-paid-compute

## Must not

- Never develop, redesign or tune model, trainer or mechanism architecture.
- Never launch a paid or GPU experiment to make the analysis look better; if a
  question cannot be answered from existing evidence, record the missing
  evidence as a backlog entry instead.
- Never compare runs as if they were controlled when pooling, layers, splits,
  epoch budget, checkpoint policy, seed treatment or evaluation protocol differ.
- Never let weaker evidence overwrite stronger evidence, and never retire an
  unattractive hypothesis that the evidence still supports.
- Never promote a single-seed, unmatched or leaky-split observation beyond what
  its evidence level licenses.

## Steps

1. Load the state: `improvements/taskrelation/research/STATE.md`,
   `improvements/taskrelation/research/OBJECTIVE.md`,
   `improvements/taskrelation/research/FINDINGS.md`,
   `improvements/taskrelation/research/FAILURES.md`,
   `improvements/taskrelation/research/DECISIONS.md`,
   `improvements/taskrelation/research/BACKLOG.md`,
   `improvements/taskrelation/research/STUDIES.jsonl`,
   `improvements/taskrelation/research/FRAMEWORK.md`, the relevant
   `improvements/taskrelation/research/studies/<ID>/` folders including each
   Study's `result.json` and `analysis.md`, and the diagnostics under
   `improvements/taskrelation/research/task_relations/`.
2. State the exact analysis question; if none was supplied, take the
   highest-value unresolved question in
   `improvements/taskrelation/research/STATE.md`.
3. Build a matched evidence table across the relevant Studies, recording
   method, stage, seeds, task set, pooling, layers, protocol, per-task endpoints
   and matched baseline, and mark every unmatched or confounded comparison
   explicitly.
4. Classify each observation by the project's evidence levels, defined in
   `improvements/taskrelation/research/FRAMEWORK.md` and the study semantics of
   `wavcse-experiment-operator`. Apply that rule; do not restate or extend the
   taxonomy here.
5. Analyse KS, SI and ER separately and then the aggregate, so a dominant task
   or hidden negative transfer is not mistaken for a mechanism effect.
6. Where relation evidence exists, compare empirical transfer, gradient
   interaction, the learned relation object and downstream outcome as distinct
   quantities that need not agree.
7. Examine negative results and cluster failures by explanation rather than by
   method name.
8. Evaluate each active hypothesis as supported, weakened, contradicted or
   insufficiently evidenced, and state the strongest evidence on both sides.

## Durable state

- Findings: `improvements/taskrelation/research/FINDINGS.md` — authoritative for
  numbers and provenance; update only when the evidence justifies it.
- Backlog: `improvements/taskrelation/research/BACKLOG.md` — reprioritise only
  with a recorded reason.
- State: `improvements/taskrelation/research/STATE.md`; binding scope:
  `improvements/taskrelation/research/DECISIONS.md`; negative evidence:
  `improvements/taskrelation/research/FAILURES.md`; study registry:
  `improvements/taskrelation/research/STUDIES.jsonl`.
- Per-quantity synthesis:
  `improvements/taskrelation/research/task_relations/MTRL_DIAGNOSTIC_SYNTHESIS.md`.

Report conclusions with the evidence that supports them, and leave anything the
evidence cannot decide clearly marked as unresolved.
