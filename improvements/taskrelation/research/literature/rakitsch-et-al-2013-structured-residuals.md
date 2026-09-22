# Rakitsch et al. (2013) — It Is All in the Noise

## Citation

Barbara Rakitsch, Christoph Lippert, Karsten Borgwardt, and Oliver Stegle. “It Is All in the Noise: Efficient Multi-Task Gaussian Process Inference with Structured Residuals.” *NeurIPS*, 2013.

Primary source: https://papers.nips.cc/paper_files/paper/2013/file/59c33016884a62116be975a9bb8257e3-Paper.pdf

## Problem

A conventional multi-task GP may falsely attribute residual correlation from hidden causes to useful task sharing. The paper seeks to identify true signal relationships by modelling residual task structure separately.

## Mathematical assumption

Observed multi-task outputs have covariance

`C_task ⊗ R_sample + Σ_noise ⊗ I`,

where `C_task` describes relations in the predictable signal and `Σ_noise` describes task correlations in residuals. These are distinct learned covariance matrices.

## Relation representation

Two explicit task covariance matrices: signal relation `C` and residual/noise relation `Σ`. Equality of the two causes task cancellation and eliminates joint-prediction benefit.

## Optimization method

Closed-form Gaussian marginal likelihood with gradients. Simultaneous diagonalization and Kronecker algebra reduce runtime from naive cubic complexity in the product dimension to `O(N³ + T³)`.

## Evidence

Synthetic hidden-process experiments and two genetic phenotype applications. Separating signal and noise covariance improved prediction when unobserved correlated causes were present; the paper visualized distinct learned signal and noise relations.

## Assumptions

Fully observed aligned multi-output Gaussian regression: every sample has outputs for all tasks. Tasks share one output space and sample covariance. Noise is Gaussian and input-independent.

## Differences from our setting

This is the closest conceptual match to F9 because it separates relation from reliability/noise. However, KS/SI/ER use disjoint datasets, different label spaces and multiclass cross-entropy. F9 measures shared-parameter gradient norms, not correlated output residuals. A faithful Kronecker-sum GP cannot be inserted into the current heterogeneous classifier heads. An MTRL-plus-uncertainty construction would be a new hybrid not specified by this paper.

## Implementation difficulty

Prohibitive under the current protocol. It would require reformulating all tasks as aligned Gaussian multi-output regression or deriving and validating a novel deep-classification analogue.

## Candidate Study ID

`LT-0001` — closest conceptual candidate.

## LT-0001 decision

**REJECT FOR CURRENT IMPLEMENTATION; RETAIN AS DIAGNOSTIC PRINCIPLE.** It establishes that task relation and task noise must be separated, but does not provide a faithful mechanism for F9 in this heterogeneous classification setting. Before revisiting this direction, measure whether ER label/data noise rather than task semantics drives the scale signal.
