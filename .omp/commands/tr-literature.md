# Targeted Task Relation Literature Research

Act as the literature and methods researcher for the wavCSE Task Relation Learning project.

The objective is not to collect papers.

The objective is to discover **scientifically relevant mechanisms whose assumptions address an observed limitation in our experiments**, then convert those mechanisms into justified research candidates.

Do not modify production/model code during this command.

Do not launch GPU training.

Literature research should end with evidence-backed candidate Studies, not implementations.

---

# 1. Load Existing Evidence First

Read:

- `.omp/AGENTS.md`
- `.omp/RULES.md`
- `improvements/taskrelation/research/OBJECTIVE.md`
- `improvements/taskrelation/research/STATE.md`
- `improvements/taskrelation/research/FINDINGS.md`
- `improvements/taskrelation/research/BACKLOG.md`
- `improvements/taskrelation/research/STUDIES.jsonl`
- `improvements/taskrelation/research/FAILURES.md` if present
- `improvements/taskrelation/research/literature/INDEX.md` if present

Also read relevant architecture READMEs.

Do not search literature until you understand the specific empirical problem.

---

# 2. Define the Literature Question

State:

## Observed problem

Example:

ER-related learned task relations change sign across folds.

## Current method assumption

Example:

MTRL assumes one symmetric global task covariance.

## Why that assumption may be inadequate

Example:

Relation reliability differs sharply by task pair.

## Literature question

Example:

Are there Task Relation Learning methods that model uncertainty or pair-specific confidence in learned relations?

This formulation must precede search.

---

# 3. Search Targeted Concepts

Construct search queries from the observed failure mode.

Good examples:

- sparse task relationship learning multi-task learning;
- asymmetric task relation learning;
- directed task transfer multi-task learning;
- uncertainty task relationship multi-task learning;
- Bayesian task covariance multi-task learning;
- robust task relationship learning noisy tasks;
- dynamic task relationship learning;
- layer-wise task relation neural multi-task learning;
- multi-task learning negative transfer relation matrix;
- unequal sample size task relationship learning;
- task covariance stability multi-task learning.

Also search speech-specific work when relevant:

- speech multi-task negative transfer;
- keyword spotting speaker identification multi-task;
- speech emotion auxiliary task transfer;
- speaker emotion task relationship;
- gradient conflict speech multi-task learning.

Do not use generic queries such as:

`best multi task learning architecture`

unless performing a deliberate broad survey.

---

# 4. Cover Both Foundational and Recent Work

For every topic, search for:

- foundational method papers;
- important extensions;
- recent work that changes the assumptions;
- surveys/reviews useful for taxonomy;
- speech-domain applications where available.

Do not prefer a recent paper merely because it is recent.

A foundational method may be more relevant to the mechanism being tested.

---

# 5. Verify the Method Category

For every candidate paper determine whether the primary contribution is genuinely relevant to **Task Relation Learning**.

Distinguish from:

- low-rank parameter methods;
- task clustering;
- parameter decomposition;
- generic loss weighting;
- gradient surgery;
- generic feature sharing;
- architecture search.

Methods from adjacent categories may be useful as:

- diagnostics;
- controls;
- conceptual inspiration.

But do not silently claim them as Task Relation Learning.

Record category ambiguity explicitly.

---

# 6. Retrieve and Store Papers

For highly relevant papers, obtain an accessible PDF or official manuscript when possible.

Store under:

`improvements/taskrelation/research/literature/papers/`

Use a stable descriptive filename, for example:

`2010_zhang_yeung_mtrl.pdf`

Do not download dozens of weakly related papers.

Prefer a small number of papers that directly address the observed problem.

---

# 7. Create a Paper Card

For every paper worth retaining, create:

`literature/<paper-slug>.md`

using:

# Citation

Full title, authors, venue, year.

# Link / local file

Source and local PDF path.

# Problem

What problem is the method solving?

# Core assumption

What does it assume about relationships among tasks?

# Mathematical representation

Examples:

- covariance matrix;
- precision matrix;
- directed matrix;
- graph;
- tensor;
- latent relation;
- uncertainty distribution.

# Optimization

How is the relation learned?

# Symmetry

Is the relation:

- symmetric;
- asymmetric;
- directed;
- unspecified?

# Sparsity

Does the method allow/select sparse relationships?

# Dynamics

Is the relation:

- static;
- training-dependent;
- sample-dependent;
- layer-dependent?

# Confidence / uncertainty

Does it model uncertainty in the relation estimate?

# Requirements

What data/parameter structure does it assume?

# Evidence reported by authors

What experiments support the method?

Do not reproduce claims without attribution.

# Limitations

What would make it unsuitable for KS/SI/ER?

# Connection to our evidence

Explicitly map the method to findings such as:

- unstable ER relations;
- stable KS↔SI relation;
- pooling sensitivity;
- asymmetric empirical transfer;
- gradient conflicts;
- task-size imbalance.

# Difference from existing implementation

Explain how it differs from:

- MTRL;
- PMR;
- TSM;
- GBC;
- already tested approaches.

# Implementation complexity

Estimate qualitatively:

- low;
- medium;
- high.

Identify required code changes.

# Expected information gain

What scientific question would implementing this method answer?

# Candidate Study

Propose a Study only if justified.

---

# 8. Compare Papers by Assumption, Not by Reported Accuracy

Create or update:

`improvements/taskrelation/research/literature/INDEX.md`

with a comparison table containing:

| Paper | Relation type | Symmetric? | Sparse? | Dynamic? | Uncertainty? | Deep-compatible? | Relevant finding | Candidate Study |
|---|---|---|---|---|---|---|---|---|

Do not rank papers solely by the benchmark accuracy reported in unrelated datasets.

The important question is:

> Does the method's assumption match the behaviour we measured?

---

# 9. Look for Contradictory Evidence

For promising mechanisms search for:

- later papers identifying limitations;
- failed reproductions;
- known instability;
- assumptions that do not hold in deep networks;
- computational issues;
- cases of negative transfer.

Do not search only for evidence supporting the mechanism.

---

# 10. Translate Literature into Hypotheses

A paper does not automatically become an experiment.

For each candidate mechanism state:

## Observation from our project

What motivates it?

## Literature mechanism

What specific method/idea addresses that observation?

## Hypothesis

What should happen in KS/SI/ER if the mechanism is relevant?

## Falsification

What result would show it does not solve our problem?

## Control

What must it be compared against?

## Minimum viable experiment

What is the cheapest implementation capable of testing the hypothesis?

## Category check

Why does this remain Task Relation Learning rather than another parameter-based MTL family?

---

# 11. Prioritize Candidate Studies

Prioritize primarily by:

1. fit to existing evidence;
2. expected information gain;
3. ability to falsify an important hypothesis;
4. implementation risk;
5. GPU cost;
6. novelty relative to existing project work.

Do not prioritize merely because a method is fashionable or complex.

A simple method tightly matched to an observed failure mode is preferable.

---

# 12. Add Literature-Derived Backlog Entries

For the strongest candidates, add entries to BACKLOG.md.

If a candidate is sufficiently concrete, reserve the next `LT-xxxx` Study ID.

Do not mark it ACTIVE.

It becomes ACTIVE only through `/tr-cycle`.

Each backlog entry must reference:

- paper card;
- motivating Finding;
- testable hypothesis;
- matched control;
- minimum experiment.

---

# 13. Determine Whether Literature Actually Changes the Plan

At the end answer:

## What did the literature add that we did not already know?

## Which current hypothesis gained support?

## Which current hypothesis weakened?

## Did we find a mechanism whose assumptions fit our measured behaviour?

## Is implementation justified?

Possible conclusions include:

- YES — create/prioritize candidate Study;
- NOT YET — diagnostics are missing first;
- NO — current literature does not justify another architecture.

"No implementation yet" is a valid successful outcome.

---

# 14. Update Research Memory

Update as justified:

- `literature/INDEX.md`
- individual paper cards
- `BACKLOG.md`
- `STATE.md`
- `FINDINGS.md` only when literature plus our empirical evidence supports a change
- `DECISIONS.md` if a method family is explicitly accepted/rejected for investigation

Do not place large paper summaries in STATE.md.

---

# 15. Final Output

End with:

## Observed research problem

One paragraph.

## Most relevant literature mechanisms

Only the strongest candidates.

## Best candidate for experimentation

State the mechanism and why it fits.

## Required diagnostic before implementation

If any.

## Proposed Study

Study ID/type, hypothesis and minimal experiment.

## Papers retained

List the paper cards created.

## What not to implement

Identify attractive but poorly matched methods and explain why.

The goal is to reduce the research search space, not expand it indefinitely.