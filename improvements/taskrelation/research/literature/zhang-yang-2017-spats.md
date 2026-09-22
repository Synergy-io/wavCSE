# Zhang and Yang (2017) — Learning Sparse Task Relations

## Citation

Yu Zhang and Qiang Yang. “Learning Sparse Task Relations in Multi-Task Learning.” *AAAI*, 2017.

Primary source: https://ojs.aaai.org/index.php/AAAI/article/download/10820/10679

## Problem

Dense pairwise covariance can overfit when there are many tasks and many task pairs are unrelated.

## Mathematical assumption

Task parameters follow covariance coupling `tr(W Ω^{-1} Wᵀ)`, but Ω is sparse because a task cannot benefit from every other task. An `ℓ₁` penalty on Ω makes zero covariance entries represent absent task transfer.

## Relation representation

A learned PSD sparse task covariance Ω. This is a direct extension of the classical MTRL relation object and remains in the Task Relation Learning category.

## Optimization method

Jointly convex objective for convex loss. Alternating optimization updates task parameters and solves the Ω subproblem with FISTA and soft thresholding.

## Evidence

Synthetic covariance recovery and five regression/classification datasets with 20–139 tasks. Benefits were strongest with many tasks; the paper reports weaker performance on a four-task sentiment dataset and explicitly attributes that to insufficient transfer under sparse coupling.

## Assumptions

Useful task relations are sparse; task parameter vectors are comparable; enough tasks exist for sparsity to reduce model complexity; zero Ω entries identify absent data contribution.

## Differences from our setting

DG-0001 did not establish beneficial pair selectivity, and KS/SI/ER contain only three tasks. F9 reports gradient-scale imbalance, not dense negative transfer. SPATS has no scale, uncertainty, sample-size, or noise-reliability term. Its own evidence warns that sparsity may hurt when task count is small.

## Implementation difficulty

Moderate, but irrelevant to the current evidence gate. It would require an Ω proximal update and an extra sparsity hyperparameter without a justified independent variable.

## Candidate Study ID

`LT-0001` — screened candidate; related backlog item `TR-0001` remains blocked.

## LT-0001 decision

**REJECT FOR F9.** Legitimate Task Relation Learning, but it addresses many-task pair selectivity rather than unequal optimization scale. Implementing it now would rediscover the pre-evidence sparse-method ordering that DEC-0005 removed.
