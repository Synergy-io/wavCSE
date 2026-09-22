# Chang et al. (2024) — Informative Relationship Multi-Task Learning

## Citation

Xiangchao Chang, Menghui Zhou, Xulong Wang, Yun Yang, and Po Yang. “Informative Relationship Multi-Task Learning: Exploring Pairwise Contribution Across Tasks’ Sharing Knowledge.” *Knowledge-Based Systems* 301:112187, 2024.

Primary source: https://eprints.whiterose.ac.uk/213770/1/KBS_clean.pdf

## Problem

Covariance methods can impose compulsory similarity on every task parameter and obscure task-specific characteristics. The paper aims to learn pairwise contribution only in the knowledge-sharing component.

## Mathematical assumption

Decompose task parameters as `W = H + P`. Learn covariance Ω over shared component `H` via `tr(H Ω^{-1} Hᵀ)` while an exclusive-Lasso penalty models task-specific component `P`. A sparse variant adds `||Ω||₁`.

## Relation representation

An explicit covariance Ω over the shared parameter component. The model interprets this as an “informative relationship”; its sparse variant selects contributing pairs.

## Optimization method

Alternating updates of `H`, `P`, and Ω using accelerated proximal-gradient methods. The paper derives estimation and sparsity-recovery bounds under Gaussian linear-model assumptions.

## Evidence

Synthetic covariance recovery and regression datasets including School, Parkinson, SmartFert, and longitudinal Alzheimer’s measures. Experiments compare against sparse relation, exclusive relation, robust/decomposition, and temporal methods.

## Assumptions

Linear regression/classification formulation; Gaussian-noise theory; a meaningful additive shared/task-specific parameter decomposition; often equal sample sizes in the main derivation, though the authors state extension is possible.

## Differences from our setting

The model addresses compulsory sharing and task-specific structure, not measured gradient-scale dominance. Its defining `W = H + P` mechanism is parameter decomposition, which the project must keep distinct from Task Relation Learning. Sparse pair selection also lacks support from DG-0001/F9 and three tasks are unlike the paper’s many-task motivation.

## Implementation difficulty

High for heterogeneous heads and deep shared parameters. More importantly, faithful implementation crosses the project’s decomposition boundary.

## Candidate Study ID

`LT-0001` — recent literature boundary check.

## LT-0001 decision

**EXCLUDE BY CATEGORY AND ASSUMPTION.** The paper learns an explicit relation matrix, but its contribution fundamentally relies on shared/specific parameter decomposition and does not target F9. It cannot be used to bypass the project’s decomposition boundary.
