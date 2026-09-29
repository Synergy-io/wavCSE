---
description: Produce the supervisor-facing weekly research report from repository and run records
---

# Weekly research report

Act as the research lead writing the weekly progress report for supervisors,
collaborators and other researchers.

Build the report from repository records and run-tracking metadata only. Never
invent progress from conversational context, and never smooth over a
disagreement between records.

Skills: wavcse-experiment-operator
Boundaries: read-only, no-commit, no-paid-compute

## Weekly period

- A research week runs Wednesday 00:00 through Tuesday 23:59, Asia/Colombo.
- Unless another period is requested, report the research week ending on the
  most recent Tuesday: the current Tuesday if today is Tuesday, otherwise the
  preceding Tuesday.
- Deliver the report in the conversation. Generating a weekly briefing never
  writes a research record; a separately requested publication may do so.

## Must not

- Never report a screening, single-seed, unmatched or speaker-leaky observation
  as a confirmed conclusion.
- Never present a smoke test or an edited old file as substantive progress.
- Never omit or soften a negative result.
- Never claim significance, causation or a champion without the evidence.
- Never write any file or commit while answering this read-only command.
- Never commit, and never launch or buy compute.

## Steps

1. Fix the reporting period and state its start date, end date, generation date
   and timezone.
2. Reconstruct the week from records:
   `improvements/taskrelation/research/STATE.md`,
   `improvements/taskrelation/research/OBJECTIVE.md`,
   `improvements/taskrelation/research/FINDINGS.md`,
   `improvements/taskrelation/research/FAILURES.md`,
   `improvements/taskrelation/research/DECISIONS.md`,
   `improvements/taskrelation/research/BACKLOG.md`,
   `improvements/taskrelation/research/STUDIES.jsonl`,
   `improvements/taskrelation/research/FRAMEWORK.md`, the
   `improvements/taskrelation/research/studies/<ID>/` folders touched in the
   period, and run-tracking metadata; use Git history within the period where it
   helps.
3. Decide whether substantive research work actually occurred; do not count a
   Study merely because an old file was edited.
4. Classify each item by the project's study semantics and evidence levels,
   defined by `wavcse-experiment-operator` and
   `improvements/taskrelation/research/FRAMEWORK.md`. Apply them; do not restate
   the taxonomy here.
5. Write the report: executive summary, starting position, studies conducted,
   key findings, hypotheses weakened or rejected, current understanding of the
   method, performance status, infrastructure and reproducibility, blockers,
   direction changes, next-period objectives, research funnel, metrics at a
   glance, and one supervisor paragraph.
6. Label single-seed and speaker-leaky results as such, and state explicitly
   where records disagree instead of choosing a side.
7. Reply with the period, strongest
   conclusion, biggest unresolved question, next-period priority and the
   supervisor paragraph.

## Durable state

- Existing reports: `improvements/taskrelation/research/weekly/`.
- Existing index: `improvements/taskrelation/research/weekly/INDEX.md`.
- All records are read-only for this command; if a record looks wrong,
  disclose the disagreement in the report rather than editing the record.
