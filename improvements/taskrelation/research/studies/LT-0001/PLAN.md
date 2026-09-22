# LT-0001 — Scale- and reliability-aware task relation literature gate

Status: REJECTED — no eligible published mechanism; completed 2026-09-22T09:15:21+00:00
Type: literature
Created: 2026-09-22T09:09:01+00:00
Started: 2026-09-22T09:09:01+00:00
Research family: Task Relation Learning
Parent Study: DG-0002
Motivating finding: F9

## Observation

DG-0002 establishes that, under matched `smp(0.5)` 25-layer KS/SI/ER training, ER shared-gradient norms dominate KS/SI by roughly 7–9× in the middle and late phases across seeds 0–4. Classical MTRL does not consistently reduce this imbalance. Persistent pairwise conflict is not supported, and Ω magnitude saturates in every seed.

## Research question

Does published Task Relation Learning literature contain an explicit learned-relation method whose stated mathematical assumption addresses unequal task optimization scale, task reliability, sample-size-dependent confidence, or separation of task signal relations from task noise—and which can be adapted to this project without becoming loss weighting, gradient surgery, low-rank learning, clustering, or parameter decomposition?

## Hypothesis

At least one published method in the formal Task Relation Learning category jointly models explicit task relations and task-specific reliability/noise in a way that maps directly to F9 and is implementable for heterogeneous deep classification heads with fixed wavCSE/WavLM embeddings.

## Competing explanation

The literature may split the problem across categories: explicit relation methods model parameter/function covariance but not optimization scale, while methods that directly balance gradient magnitude or uncertainty are optimization/loss-weighting methods rather than Task Relation Learning. Existing reliability-aware relation methods may also be restricted to aligned multi-output Gaussian regression, making them mathematically inapplicable to KS/SI/ER classification.

## Falsification condition

Reject the candidate-method hypothesis if no verified primary paper satisfies all of the following:

1. learns an explicit quantitative task relation matrix or graph;
2. explicitly models unequal task scale, reliability, uncertainty, sample-size confidence, or residual/noise structure;
3. states a mechanism that maps to F9 rather than merely to pairwise conflict or sparsity;
4. remains in Task Relation Learning under Zhang and Yang's taxonomy;
5. can be implemented without changing the fixed upstream representation or crossing into low-rank, clustering, decomposition, generic loss weighting, or gradient surgery;
6. has a mathematically defensible adaptation to heterogeneous multiclass heads and non-aligned task datasets.

## Independent variable

Literature mechanism assumption and taxonomy. No model or training variable changes in this Study.

## Control

Classical MTRL's matrix-normal task covariance assumption and the formal Task Relation Learning definition in Zhang and Yang's survey. Candidate papers are compared against F9 and the project's category boundaries using the same eligibility checklist.

## Controlled variables

No empirical protocol changes. Any eventual candidate must preserve fixed WavLM embeddings, `smp(0.5)` pooling, all 25 layers, KS/SI/ER data and splits, 30 epochs, optimizer, seed treatment, checkpoint policy, and diagnostics unless a later Study explicitly makes one of these the independent variable.

## Evaluation protocol

Read primary sources rather than abstracts alone. For each relevant paper record citation, problem, mathematical assumption, relation representation, optimization, evidence, assumptions, differences from this project, implementation difficulty, taxonomy decision, F9 mapping, and candidate Study decision. Separate directly eligible methods from adjacent diagnostic evidence and explicit rejections.

## Diagnostics

Literature-level checks:

- explicit relation object present;
- relation and reliability/noise are distinct quantities;
- gradient-scale or confidence mechanism stated, not inferred;
- classification and unequal-data applicability;
- heterogeneous task-head applicability;
- branch/category boundary;
- adaptation required versus faithful implementation;
- evidence strength and validation protocol.

## Screening protocol

Search from the observed limitation using combinations of `task relationship learning`, `task covariance`, `gradient magnitude`, `task uncertainty`, `heteroscedastic`, `residual covariance`, `unequal sample size`, and `reliability`. Read the Zhang–Yang taxonomy, foundational covariance methods, direct gradient-scale methods, and recent explicit-relation methods. A paper advances only if a primary source supports its assumption and taxonomy.

## Confirmation protocol

Before a mechanism Study is opened, independently verify the surviving paper's equations and category against its primary text and at least one authoritative taxonomy/source. Confirm that the proposed implementation is the paper's mechanism rather than an original hybrid. If no paper survives, record the negative result and enter `NEEDS-HUMAN-REVIEW`; do not substitute an adjacent category.

## Compute estimate

CPU/document work only; zero GPU-hours and no MLflow execution runs.

## GPU allocation

None.

## Expected information gain

High. The Study either identifies a defensible literature-derived successor to MTRL or prevents an invalid contribution claim by establishing that direct scale-balancing methods are outside the formal Task Relation Learning scope and that covariance/noise methods require missing diagnostics or incompatible assumptions.
