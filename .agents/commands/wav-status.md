---
description: Report the current state of the Task Relation Learning research programme
---

# Research Status Briefing

Act as a research briefing assistant for the wavCSE Task Relation Learning
programme and answer, in plain scientific language for the human researcher:

> Where are we, what have we learned, what is running, and what happens next?

Skills: wavcse-experiment-operator
Boundaries: read-only, no-commit, no-paid-compute

## Must not

- modify code, configs, study folders or any research document;
- launch, stop, provision or pay for anything, including compute;
- create, reprioritize or retire Studies, or edit the backlog;
- perform literature research, run training, or commit.

## Steps

1. Reconcile the durable record before saying anything:
   `.omp/AGENTS.md`,
   `improvements/taskrelation/research/OBJECTIVE.md`,
   `improvements/taskrelation/research/STATE.md`,
   `improvements/taskrelation/research/FINDINGS.md`,
   `improvements/taskrelation/research/DECISIONS.md`,
   `improvements/taskrelation/research/FAILURES.md`,
   `improvements/taskrelation/research/BACKLOG.md`,
   `improvements/taskrelation/research/STUDIES.jsonl`,
   `improvements/taskrelation/research/FRAMEWORK.md`,
   `improvements/taskrelation/research/VARIANT_BENCHMARK_PROTOCOL.md` and
   `improvements/taskrelation/research/task_relations/MTRL_DIAGNOSTIC_SYNTHESIS.md`.
2. Inspect only the most recent Study folders under
   `improvements/taskrelation/research/studies/` and the architecture READMEs
   needed to interpret them. Do not re-derive settled history.
3. Reconcile runtime reality with
   `python -m improvements.compute status --scope <SCOPE> --json` — it is
   strictly read-only and reports the envelope, spend, leases and jobs in
   flight. When it names an active scope you may additionally run
   `sweep --scope <SCOPE>` **without** `--execute`, which is a dry-run plan:
   report it, never act on it. This command provisions nothing, submits nothing,
   destroys nothing, writes no runtime state, and consumes no spend.
4. Report in this order: executive summary; current position, phase and
   champion; established findings with a confidence label each (strong /
   moderate / screening only); the most recent Studies and what changed because
   of them; the task-level picture for KS, SI and ER; the research funnel with
   the current location marked; the next expected step and why it is highest
   priority; tempting conclusions the evidence does not support; brief
   operational status; and a three-line bottom line (we know / we don't know /
   next).
5. When repository records disagree, report the disagreement explicitly and
   name the authoritative record rather than guessing.
   `improvements/taskrelation/research/FINDINGS.md` is authoritative for numbers
   and provenance; `improvements/taskrelation/research/STATE.md` is authoritative
   for the restart position.

## Evidence rules

- Keep single-seed observations separate from established findings (F1).
- Label ER evidence as screening unless it comes from the speaker-independent
  LOSO protocol (F3).
- Never present a plan, a hypothesis or speculation as a result.

## Durable state

The repository research records under `improvements/taskrelation/research/` are
the programme's memory. This command reads them and writes nothing anywhere.
