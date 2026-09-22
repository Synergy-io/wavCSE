---
description: Explain the current Task Relation Learning research status in clear human-readable language
---

# Human-Readable Research Status

Act as a research briefing assistant.

Your task is to explain the current state of the wavCSE Task Relation Learning
research program to the human researcher in a concise, understandable format.

This command is READ-ONLY.

Do not:

- modify code;
- modify research files;
- launch experiments;
- create Studies;
- update the backlog;
- perform literature research;
- make commits.

The goal is to answer:

> Where are we, what have we learned, what is running, and what happens next?

---

# 1. Reconstruct Current State

Read:

- `.omp/AGENTS.md`
- `.omp/RULES.md`
- `improvements/taskrelation/research/OBJECTIVE.md`
- `improvements/taskrelation/research/STATE.md`
- `improvements/taskrelation/research/FINDINGS.md`
- `improvements/taskrelation/research/BACKLOG.md`
- `improvements/taskrelation/research/STUDIES.jsonl`

If present, also inspect:

- `FAILURES.md`
- `DECISIONS.md`
- `task_relations/MTRL_DIAGNOSTIC_SYNTHESIS.md`

Inspect the most recent Study directories as necessary.

Check runtime state with:

```bash
tmux ls
nvidia-smi
```

Do not change anything.

---

# 2. Produce a Human Research Briefing

Use the following structure.

## Executive summary

Explain in 3–6 sentences:

- where the research currently stands;
- whether anything has beaten the baseline;
- the most important thing learned recently;
- what question the research is currently trying to answer.

Avoid excessive implementation terminology.

---

## Current position

Report:

- current research phase;
- current champion/reference method;
- active Study, if any;
- last completed Study;
- whether anything is waiting for confirmation;
- whether any experiments are currently running.

If GPUs/jobs are active, explain what they are doing.

---

## What we have learned

List the most important **confirmed or reasonably strong scientific findings**.

For each finding explain:

**Finding:** plain-language statement.

**Meaning:** why it matters scientifically.

**Confidence:** strong / moderate / screening only.

Do not mix tentative single-seed observations with established findings.

---

## What changed recently

Summarize the most recent 1–3 Studies.

For each:

### <Study ID> — <title>

Explain:

- question asked;
- what was tested;
- main result;
- decision;
- what changed because of it.

Use actual numbers only where they help understanding.

Do not dump all metrics.

---

## Current MTRL explanation

Explain our current best understanding of why classical MTRL has not produced
a clear reproducible improvement.

Separate:

### Evidence supports

What is currently supported.

### Evidence weakens

What explanations have been tested and weakened/rejected.

### Still unknown

What remains unresolved.

Do not present speculation as fact.

---

## Task-level picture

Explain the current behaviour of:

- KS;
- SI;
- ER.

Mention important positive/negative transfer, instability, data-regime,
gradient or relation behaviour if supported.

Call out when ER evidence is only from speaker-leaky screening versus
speaker-independent LOSO.

---

## Research funnel

Show the current research path in a compact form, for example:

```text
wavCSE baseline
    ↓
MTRL
    ↓
DG-0001: ...
    ↓
DG-0002: ...
    ↓
CURRENT QUESTION
    ↓
NEXT EXPECTED STEP
```

Mark the current location clearly.

---

## Next step

Explain:

1. the next planned Study/action;
2. why it is the highest-priority question;
3. what result would change the research direction;
4. whether literature search is currently justified;
5. whether a new architecture is currently justified.

This should be understandable without reading BACKLOG.md.

---

## Things that are NOT conclusions

Explicitly list major tempting conclusions that current evidence does **not**
support.

Examples might include:

- MTRL definitely hurts SI;
- asymmetric relations are required;
- ER gradient dominance is confirmed;
- a particular new architecture should be implemented.

Only include items relevant to current state.

This section is important for preventing overinterpretation.

---

## Operational status

Report briefly:

- running tmux jobs;
- GPU state;
- most recent relevant Git commit(s);
- disk-space concern if material;
- MLflow/DagsHub tracking health;
- any failed/incomplete runs requiring attention.

Keep this short.

---

## Bottom line

Finish with exactly three concise statements:

**We know:** <strongest current conclusion>

**We don't know:** <most important unresolved question>

**Next:** <next action>

---

# Style

Write for the human researcher, not for another agent.

Prefer plain scientific language.

Explain jargon when needed.

Do not produce a giant metric dump.

Do not repeat every historical experiment.

Prioritize:

- scientific meaning;
- evidence strength;
- current direction;
- what changed;
- what happens next.

If a result is preliminary, say so explicitly.

If project files disagree, point out the inconsistency rather than guessing.
