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
Boundaries: writes-reports, no-commit, no-paid-compute

## Weekly period

- A research week runs Wednesday 00:00 through Tuesday 23:59, Asia/Colombo.
- Unless another period is requested, report the research week ending on the
  most recent Tuesday: the current Tuesday if today is Tuesday, otherwise the
  preceding Tuesday.
- Name the report by its Tuesday end date and write it to
  `improvements/taskrelation/research/weekly/<YYYY-MM-DD>.md`, updating an
  existing report for that week rather than duplicating it.
- Maintain one row per report in
  `improvements/taskrelation/research/weekly/INDEX.md`.

## Must not

- Never report a screening, single-seed, unmatched or speaker-leaky observation
  as a confirmed conclusion.
- Never present a smoke test or an edited old file as substantive progress.
- Never omit or soften a negative result.
- Never claim significance, causation or a champion without the evidence.
- Never write outside the report file and the weekly index.
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
7. Update the weekly index, then reply with the period, report path, strongest
   conclusion, biggest unresolved question, next-period priority and the
   supervisor paragraph.

## Durable state

- Reports: `improvements/taskrelation/research/weekly/`.
- Index: `improvements/taskrelation/research/weekly/INDEX.md`.
- Source records are read-only for this command; if a record looks wrong,
  disclose the disagreement in the report rather than editing the record.
