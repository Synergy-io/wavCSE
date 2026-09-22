---
description: Generate the weekly wavCSE Task Relation Learning research progress report
---

# Weekly Task Relation Learning Research Report

Act as the research lead preparing a weekly progress report for supervisors,
collaborators, and other researchers.

Generate a scientifically accurate, human-readable summary of progress for the
most recently completed or currently requested research week.

The default research week is:

**Wednesday 00:00 → Tuesday 23:59**

Timezone:

**Asia/Colombo**

Unless the user specifies another reporting period, use the research week ending
on the current Tuesday.

If today is Wednesday through Monday, use the most recently completed Tuesday as
the end of the reporting period.

If today is Tuesday, include the current Tuesday.

Example:

```text
Wednesday 2026-09-16 00:00
→
Tuesday 2026-09-22 23:59
```

The repository and MLflow/DagsHub records are the source of truth.

Do not invent progress from conversational context.

---

# 1. Determine Reporting Period

Unless a specific date range is supplied:

1. determine the applicable Tuesday report-end date;
2. set the start date to the preceding Wednesday;
3. use the full Wednesday-through-Tuesday interval in Asia/Colombo.

State explicitly:

- report start date;
- report end date;
- report generation date;
- timezone.

Name the report using the Tuesday end date:

`research/weekly/YYYY-MM-DD.md`

Example:

`research/weekly/2026-09-22.md`

Create the directory if necessary.

If a report already exists for that Tuesday end date, update it rather than
creating a duplicate.

---

# 2. Reconstruct Weekly Activity

Read:

- `.omp/AGENTS.md`
- `research/OBJECTIVE.md`
- `research/STATE.md`
- `research/FINDINGS.md`
- `research/STUDIES.jsonl`
- `research/BACKLOG.md`
- `research/FAILURES.md`
- `research/DECISIONS.md`

Inspect Study folders that were:

- started;
- active;
- completed;
- confirmed;
- rejected;
- materially updated

during the reporting period.

Inspect relevant:

- `PLAN.md`
- `NOTE.md`
- `analysis.md`
- `result.json`

Use Git history within the Wednesday-through-Tuesday reporting period when
useful.

Use MLflow/DagsHub metadata where needed to verify experiment execution and
results.

Do not count a Study merely because an old file was edited during the reporting
period.

Determine whether substantive research work actually occurred.

---

# 3. Evidence Classification

Distinguish clearly between:

### Confirmed

Supported by the project's required confirmation protocol.

### Strong diagnostic evidence

Well-controlled evidence that answers a diagnostic question but may not imply
performance improvement.

### Preliminary / screening

Single-seed, incomplete, or otherwise provisional evidence.

### Rejected

Hypothesis or mechanism weakened/rejected by evidence.

### Inconclusive

Experiment did not provide enough evidence for a decision.

Never report a screening observation as a confirmed research conclusion.

---

# 4. Produce the Weekly Report

Write:

`research/weekly/<TUESDAY-END-DATE>.md`

using the following structure.

# Weekly Research Progress — Week Ending <Tuesday date>

**Period:** <Wednesday start date> – <Tuesday end date>

**Timezone:** Asia/Colombo

## 1. Executive Summary

Write 3–6 concise bullets covering:

- the main research question worked on this week;
- the most important experiment(s);
- the strongest thing learned;
- whether the baseline was beaten;
- whether the research direction changed;
- the primary next step.

This section should be understandable by someone who has not followed the daily
experiments.

---

## 2. Starting Position

Briefly state what was believed or unknown at the beginning of the reporting
period.

Keep this concise.

---

## 3. Studies Conducted

For every substantive Study worked on during the week:

### <STUDY_ID> — <title>

**Question:**  
What scientific question was tested?

**Method:**  
What was actually done?

**Scale:**  
Relevant number of runs, seeds, folds, controls, or GPU experiments.

**Result:**  
The important quantitative result.

**Decision:**  
CONFIRMED / PROMISING / REJECTED / INCONCLUSIVE / diagnostic conclusion.

**Meaning:**  
Why this changed or did not change our understanding.

Do not dump all individual run metrics.

Link Study/DagsHub references where they materially help collaborators inspect
the evidence.

---

## 4. Key Scientific Findings

List only findings materially strengthened, weakened, added, or revised during
the reporting period.

For each:

**Finding:** <plain-language conclusion>

**Evidence level:** Confirmed / Strong diagnostic / Preliminary

**Why it matters:** <research implication>

Separate observations from interpretations.

---

## 5. Hypotheses Rejected or Weakened

Explicitly report negative results.

For each:

- hypothesis;
- evidence that weakened/rejected it;
- consequence for future research.

Negative results count as progress when they eliminate plausible explanations.

---

## 6. Current Understanding of MTRL

Summarize the best current causal/mechanistic picture.

Use three subsections:

### Supported

What current evidence supports.

### Weakened

What explanations no longer look convincing.

### Unresolved

What still requires experimentation.

Do not make stronger causal claims than the evidence allows.

---

## 7. Performance Status

Report the current performance picture.

Include:

- current champion/reference;
- whether any method has reproducibly beaten it;
- relevant KS delta;
- relevant SI delta;
- relevant ER delta;
- aggregate delta where appropriate.

If a result is single-seed or speaker-leaky ER, label it clearly.

Do not select results simply because they are the largest observed numbers.

---

## 8. Research Infrastructure / Reproducibility

Briefly summarize meaningful infrastructure work completed during the week,
such as:

- MLflow/DagsHub metadata improvements;
- experiment runner changes;
- diagnostic instrumentation;
- reproducibility fixes;
- storage/checkpoint handling;
- automated analysis tooling.

Include infrastructure only when it enabled or protected research validity.

---

## 9. Problems / Blockers

Report:

- scientific blockers;
- compute/storage issues;
- implementation uncertainty;
- missing literature;
- evaluation limitations;
- unresolved reproducibility concerns.

If there are none, state that explicitly.

---

## 10. Research Direction Changes

If DECISIONS.md records a substantive change during the reporting period,
explain:

**Previous direction:** ...

**Evidence:** ...

**New direction:** ...

**Reason:** ...

If no major direction changed, say so.

---

## 11. Next Reporting Period

List no more than 3 primary objectives for the next Wednesday-through-Tuesday
research period.

For each objective state:

1. question;
2. planned Study/action;
3. decision the result will enable.

Prefer scientific questions over implementation tasks.

---

## 12. Research Funnel

Show a compact progression:

```text
wavCSE baseline
    ↓
MTRL
    ↓
<completed diagnostic>
    ↓
<completed diagnostic>
    ↓
[CURRENT POSITION]
    ↓
<next question>
```

This should make the research story visible at a glance.

---

## 13. Metrics at a Glance

Include a compact table such as:

| Item | This week |
|---|---:|
| Studies started | ... |
| Studies completed | ... |
| Confirmed studies | ... |
| Rejected hypotheses | ... |
| MLflow runs completed | ... |
| LOSO folds completed | ... |
| New findings | ... |
| Major research decisions | ... |

Use actual repository/MLflow evidence.

Do not optimize these counts as productivity metrics.

They are descriptive only.

---

## 14. One-Paragraph Supervisor Summary

End with one polished paragraph suitable for reading aloud or pasting into a
weekly meeting update.

It should state:

- what we investigated;
- what we learned;
- what was ruled out;
- what we are doing next.

Avoid implementation details unless essential.

---

# 5. Update Weekly Index

Maintain:

`research/weekly/INDEX.md`

with one row per report:

| Week ending | Period | Main question | Main conclusion | Next direction |
|---|---|---|---|---|

Example:

| 2026-09-22 | 2026-09-16 → 2026-09-22 | Gradient interaction | ... | ... |

Keep entries concise.

---

# 6. Accuracy Rules

Never:

- inflate preliminary evidence;
- hide negative results;
- claim significance without evidence;
- call a run a Study;
- count smoke tests as substantive scientific progress;
- use test-set results as evidence of hyperparameter-selection quality;
- describe speaker-leaky ER as authoritative;
- report an architecture as successful because of one seed.

When project records disagree, explicitly note the inconsistency.

---

# 7. Final Response

After generating/updating the weekly report, reply with:

- reporting period;
- report path;
- strongest conclusion;
- biggest unresolved question;
- next-period priority.

Then provide the **One-Paragraph Supervisor Summary** directly in the response
so the human can immediately use it in a meeting.
