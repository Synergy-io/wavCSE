# Zhang and Yang (2021) — A Survey on Multi-Task Learning

## Citation

Yu Zhang and Qiang Yang. “A Survey on Multi-Task Learning.” *IEEE Transactions on Knowledge and Data Engineering*; arXiv:1707.08114v3, 2021.

Primary source: https://arxiv.org/pdf/1707.08114

## Problem

Organize multi-task learning methods by what they share and how they model task relatedness.

## Mathematical assumption

Parameter-based MTL can encode relatedness through low rank, task clusters, explicit quantitative task relations, or parameter decomposition. These are distinct modelling categories.

## Relation representation

In §2.4, Task Relation Learning is defined by learning quantitative task similarity, correlation, covariance, or another explicit relation object from data. Canonical regularized methods use a task covariance/precision matrix in `tr(W Ω^{-1} Wᵀ)`; the survey separately classifies low-rank, clustering, and decomposition approaches.

## Optimization method

Survey, not an algorithm. It summarizes matrix-normal covariance learning, sparse covariance, high-order relations, local/asymmetric relations, tensor-normal deep extensions, and related Bayesian models.

## Evidence

Taxonomy and comparative review of foundational methods through 2020. It explicitly places GradNorm, PCGrad, and multi-objective optimization outside the §2.4 lineage: they are optimization methods, not covariance/task-relation models.

## Assumptions

The taxonomy treats the primary mathematical object and knowledge-sharing mechanism as category-defining. Merely using task-dependent scalars or gradients does not create a Task Relation Learning method.

## Differences from our setting

The survey is general and does not solve F9. It supplies the binding category test needed to avoid relabelling loss balancing, low-rank learning, clustering, or decomposition as Task Relation Learning.

## Implementation difficulty

None; taxonomy reference only.

## Candidate Study ID

`LT-0001` — anchor, not a mechanism candidate.

## LT-0001 decision

**RETAIN AS TAXONOMY AUTHORITY.** An eligible F9 method must learn an explicit relation object and cannot derive its category solely from balancing gradients or losses.
