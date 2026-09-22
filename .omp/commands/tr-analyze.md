# Task Relation Scientific Analysis

Act as a research analyst for the wavCSE Task Relation Learning project.

The purpose of this command is to synthesize existing experimental evidence and identify scientifically defensible conclusions.

This is **not** an architecture-development command.

Do not modify model/trainer architecture code unless a tiny analysis-only instrumentation fix is absolutely required.

Do not launch new expensive GPU experiments unless analysis is impossible without one and the missing evidence is clearly identified first.

---

# 1. Load Research State

Read:

- `.omp/AGENTS.md`
- `.omp/RULES.md`
- `improvements/taskrelation/research/OBJECTIVE.md`
- `improvements/taskrelation/research/STATE.md`
- `improvements/taskrelation/research/FINDINGS.md`
- `improvements/taskrelation/research/STUDIES.jsonl`
- `improvements/taskrelation/research/FAILURES.md` if present
- `improvements/taskrelation/research/DECISIONS.md` if present

Inspect relevant:

- Study `PLAN.md`
- Study `analysis.md`
- Study `result.json`
- architecture READMEs
- saved relation diagnostics
- available MLflow/DagsHub run metadata

Do not assume README claims are correct when more recent controlled evidence supersedes them.

Prefer the most rigorous evaluation protocol available.

---

# 2. Define the Analysis Question

State exactly what is being analyzed.

Examples:

- Why has MTRL failed to beat baseline?
- Does learned Ω predict empirical transfer?
- Are KS↔SI relations more stable than ER relations?
- Do gradient conflicts predict negative transfer?
- Does relation asymmetry correspond to directed empirical transfer?
- Which observables could support a method-selection framework?
- Which previous apparent improvements were confounded?

If no explicit question was supplied, choose the highest-value unresolved scientific question in STATE.md.

---

# 3. Build an Evidence Table

Construct a consistent comparison across relevant Studies.

For each Study/run include where available:

- Study ID
- method
- stage
- seed(s)
- task set
- pooling
- layers
- evaluation protocol
- KS
- SI
- ER
- aggregate
- matched baseline
- delta per task
- run variability
- relation diagnostics
- Git commit
- protocol caveats

Do not compare incomparable runs as though they were controlled.

Explicitly mark:

- different pooling;
- different layer selection;
- different data split;
- different checkpoint-selection policy;
- leaky ER evaluation;
- single-seed-only evidence;
- unmatched baselines.

---

# 4. Separate Evidence Levels

Classify observations using:

## Level A — Strong

Examples:

- matched multi-seed comparison;
- replicated finding;
- LOSO-supported ER finding;
- stable effect across reasonable protocol variants.

## Level B — Moderate

Examples:

- matched experiment with limited seeds;
- repeated qualitative behaviour;
- stable diagnostic without direct performance consequence.

## Level C — Preliminary

Examples:

- one seed;
- exploratory result;
- confounded comparison;
- leaky ER single split;
- mechanism behaviour not yet replicated.

Never let a Level C result overwrite a Level A finding.

---

# 5. Analyze Per-Task Behaviour

Treat KS, SI and ER separately.

For each relevant mechanism ask:

- Did KS improve or regress?
- Did SI improve or regress?
- Did ER improve or regress?
- Is one task dominating the aggregate metric?
- Does a mechanism trade one task against another?
- Does the aggregate score hide negative transfer?

Where useful calculate:

`raw_delta = candidate - matched_baseline`

and:

`relative_error_reduction = raw_delta / (1 - baseline_accuracy)`

Use relative error reduction as an auxiliary interpretation only.

Do not replace actual accuracy with it.

---

# 6. Analyze Task Relations

Where data exists, compare four distinct concepts:

## A. Empirical transfer

Example:

`T(KS <- SI)`

## B. Gradient interaction

Examples:

- cosine similarity;
- conflict frequency;
- norm dominance.

## C. Learned task relation

Examples:

- Ω;
- precision matrix;
- structure matrix.

## D. Outcome

Actual downstream performance.

Determine whether these agree.

Do not assume they measure the same thing.

Explicitly look for cases such as:

- high learned similarity but negative empirical transfer;
- positive gradient cosine but no performance benefit;
- asymmetric empirical transfer represented by a symmetric Ω;
- unstable relation estimate associated with unstable accuracy;
- stable relation without useful coupling.

These discrepancies are scientifically important.

---

# 7. Analyze Relation Stability

For each pair:

- KS↔SI
- KS↔ER
- SI↔ER

measure where possible:

- mean relation;
- standard deviation;
- sign consistency;
- seed variation;
- fold variation;
- pooling variation;
- epoch variation;
- layer variation.

Distinguish:

`relation strength`

from:

`confidence/stability of relation estimate`.

A large but unstable value may be less actionable than a weaker stable value.

---

# 8. Analyze Dynamic Behaviour

If epoch-level diagnostics exist, ask:

- do task relations change during training?
- do gradient conflicts concentrate early or late?
- does relation stabilization occur before or after validation convergence?
- does an architecture help by changing the final relation or by changing the optimization path?

Do not introduce a dynamic relation method merely because dynamics exist.

First establish whether the dynamic behaviour predicts useful outcomes.

---

# 9. Examine Negative Results

Read FAILURES.md and rejected Studies.

Cluster failures by explanation rather than by architecture name.

Possible clusters:

- no meaningful architecture effect;
- relation estimate unstable;
- negative transfer;
- mechanism collapses to baseline behaviour;
- optimization instability;
- pooling confound;
- seed noise;
- ER leakage;
- excessive regularization;
- task-size imbalance;
- relation representation inadequate.

Look for repeated failure modes across different models.

Repeated negative evidence may be more valuable than another architecture.

---

# 10. Evaluate Current Hypotheses

For each active hypothesis in STATE/BACKLOG classify:

- supported;
- weakened;
- contradicted;
- insufficient evidence.

State the strongest evidence on both sides.

Do not preserve an attractive hypothesis after evidence turns against it.

---

# 11. Update FINDINGS.md Carefully

Add or modify a Finding only when evidence justifies it.

Each Finding must contain:

- Observation
- Evidence
- Interpretation
- Alternative explanations
- Confidence
- Implications
- Required follow-up

Do not turn a hypothesis into a Finding.

When new evidence contradicts an existing Finding:

- do not silently delete the old statement;
- revise it;
- record why confidence changed.

---

# 12. Develop the Emerging Selection Framework

Attempt to map:

`observable task characteristics`

to:

`appropriate Task Relation Learning assumptions`.

Candidate observables include:

- transfer symmetry;
- transfer sign;
- gradient compatibility;
- conflict frequency;
- relation stability;
- data-size imbalance;
- relation representation sensitivity;
- task difficulty;
- pooling sensitivity;
- temporal relation dynamics.

Candidate mechanism assumptions include:

- dense symmetric;
- sparse/selective;
- asymmetric/directional;
- uncertainty/confidence-aware;
- dynamic;
- layer-specific;
- higher-order.

Do not force a framework before evidence exists.

Mark unsupported cells as unknown.

---

# 13. Determine the Highest-Value Missing Evidence

At the end identify:

## What we know strongly

## What appears likely but remains uncertain

## What is contradicted

## What single missing experiment would provide the most information?

Rank suggested next work primarily by:

**information gain per GPU-hour**

not expected score gain alone.

Update BACKLOG.md if priorities should change.

Record the reason.

---

# 14. Outputs

Update only the research artifacts justified by the analysis:

- `FINDINGS.md`
- `STATE.md`
- `BACKLOG.md`
- `FAILURES.md`
- `DECISIONS.md`
- relevant Study `analysis.md`
- task-relation summary files

Do not modify model architecture merely to make the analysis look cleaner.

End with a concise research synthesis:

1. strongest current conclusion;
2. most important uncertainty;
3. most promising next scientific question;
4. whether new literature research is justified.