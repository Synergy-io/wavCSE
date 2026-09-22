# Chen et al. (2018) — GradNorm

## Citation

Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. “GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks.” *ICML*, PMLR 80, 2018.

Primary source: https://proceedings.mlr.press/v80/chen18a/chen18a.pdf

## Problem

Tasks with larger loss scales produce larger shared gradients and dominate deep multi-task training.

## Mathematical assumption

Task imbalance manifests as unequal backpropagated gradient magnitudes. Learn task loss weights so each weighted task-gradient norm approaches the average norm times a relative inverse-training-rate target `r_i(t)^α`.

## Relation representation

None. GradNorm learns per-task scalar loss weights from norm and training-rate signals. It does not learn pairwise task relations.

## Optimization method

At each step, compute weighted per-task gradient norms on selected shared parameters, minimize an `L1` gradient-normalization loss with respect to task weights, renormalize weights to sum to the number of tasks, then apply the standard network update.

## Evidence

Synthetic tasks with deliberately different scales and NYUv2 regression/classification tasks. The paper reports improvement over equal weights and uncertainty weighting, and shows that scale-dominant tasks can suppress other tasks without normalization.

## Assumptions

A target based on relative training rates is appropriate; one asymmetry hyperparameter controls the restoring force; selected shared-layer gradients represent training dominance.

## Differences from our setting

This paper is the most direct mechanistic match to F9: its motivating symptom is precisely dominant task gradient magnitude. However, it is an optimization/loss-balancing method and has no learned covariance, relation graph, or pairwise task object. F9 also found no persistent directional conflict, which supports norm balancing but does not change its taxonomy.

## Implementation difficulty

Low to moderate; the existing DG-0002 instrumentation already computes most required norms. Category eligibility—not feasibility—is the blocker.

## Candidate Study ID

`LT-0001` — direct symptom match, taxonomy exclusion.

## LT-0001 decision

**EXCLUDE BY TAXONOMY.** Implementing GradNorm would answer an optimization branch question, not produce a Task Relation Learning extension. It may be an informative external control only with explicit human scope approval.
