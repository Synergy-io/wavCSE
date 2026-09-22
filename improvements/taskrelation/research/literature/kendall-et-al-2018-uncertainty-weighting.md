# Kendall, Gal, and Cipolla (2018) — Uncertainty Weighting

## Citation

Alex Kendall, Yarin Gal, and Roberto Cipolla. “Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics.” *CVPR*, 2018.

Primary source: https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf

## Problem

Different tasks have different loss units, scales, and noise. Equal or manually tuned loss weights can make one task dominate shared training.

## Mathematical assumption

A task-dependent homoscedastic uncertainty scalar parameterizes each task likelihood. Maximizing joint likelihood yields learned weights approximately proportional to inverse task variance plus a log-variance regularizer, for combinations of regression and classification tasks.

## Relation representation

None. The method learns one independent reliability/scale scalar per task; it does not learn pairwise task covariance, similarity, precision, a relation graph, or a transfer structure.

## Optimization method

Joint backpropagation of network parameters and log variances through a weighted sum of task losses.

## Evidence

Cityscapes semantic segmentation, instance segmentation, and depth. Learned weighting outperformed equal weighting, approximate grid-selected weights, and single-task models in the reported setting.

## Assumptions

Factorized task likelihoods, task-level homoscedastic uncertainty, shared input examples, and specific approximate treatment of classification temperature.

## Differences from our setting

The motivating scale problem maps closely to F9 and the method handles mixed classification/regression. However, it has no explicit task relation object and therefore is a loss-weighting/optimization method, not Task Relation Learning under the binding taxonomy. Combining it with MTRL would be an original hybrid unless supported by another primary source.

## Implementation difficulty

Low, but category-ineligible.

## Candidate Study ID

`LT-0001` — boundary case.

## LT-0001 decision

**EXCLUDE BY TAXONOMY.** Strong evidence that reliability-weighted losses can address scale imbalance, but it cannot carry this branch’s explicit Task Relation Learning contribution. It is useful as a diagnostic/control candidate only if the human broadens scope.
