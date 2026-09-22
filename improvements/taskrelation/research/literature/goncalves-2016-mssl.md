# Gonçalves, Von Zuben and Banerjee (2016) — Multi-task Sparse Structure Learning

## Citation

André R. Gonçalves, Fernando J. Von Zuben, and Arindam Banerjee. “Multi-task Sparse Structure Learning with Gaussian Copula Models.” *Journal of Machine Learning Research* 17(33):1–30, 2016.

Primary source: https://jmlr.org/papers/volume17/15-215/15-215.pdf (verified 2026-09-22; the `/v17/…` short path 404s)

## Problem

The task-relationship structure usually has to be estimated from data, not assumed. Earlier estimators either restrict the dependence structure (Zhang & Yeung’s convex relaxation) or are computationally prohibitive because they model a large feature covariance. The paper wants a sparse, interpretable *conditional* task graph, jointly learned with the task parameters, that works for regression and classification.

## Mathematical assumption

For linear models `y_k = X_k w_k + ε_k`, the rows `w^j` of the coefficient matrix `W ∈ ℝ^{d×m}` (features across tasks) are i.i.d. `N(0, Σ)`, with precision `Ω = Σ⁻¹`. Under this Gaussian graphical model, `Ω_ij = 0` iff tasks `i` and `j` are conditionally independent given the remaining tasks, so a sparse `Ω` is a sparse task graph. A semiparametric Gaussian-copula variant replaces the Gaussian marginal assumption with rank statistics (Spearman/Kendall mapped through `2 sin(πρ̂/6)`), preserving the same conditional-dependence graph.

## Relation representation

An explicit learned `m × m` **sparse task precision matrix `Ω`**, interpreted through partial correlations `−Ω_ij / √(Ω_ii Ω_jj)`. Two instantiations: `p`-MSSL estimates `Ω` from the task parameters `W` (rows of `W` assumed Gaussian); `r`-MSSL estimates it from the residuals `ε` (loosely-coupled tasks). `r`-MSSL is regression-only, because classification residuals are non-Gaussian (stated in the paper).

## Optimization method

Alternating minimization of a biconvex objective (Eq. 1/3): the `W` step is an `ℓ₁`-penalized quadratic solved by FISTA; the `Ω` step is exactly the graphical-lasso problem

`min_{Ω≻0} tr(S Ω) − d log|Ω| + λ₂‖Ω‖₁`, with `S = (1/d) WᵀW`,

solved by ADMM whose `Ω` update is an SVD and whose `Z` update is element-wise soft-thresholding. Alternation converges to a partial optimum. Each task’s empirical loss is scaled by `1/n_k` (Eq. 3) so a high-sample task cannot dominate the fit. Classification enters through the GLM link (Bernoulli/logistic; multinomial, Poisson, Gamma listed as supported conditional distributions).

## Evidence

Synthetic structure recovery (10 tasks, relevant plus irrelevant features), benchmark regression and classification data sets, and an Earth-System-Model temperature-combination problem (tasks = geographic locations). MSSL is competitive with or better than the compared MTL baselines, and the learned graph is reported to match domain knowledge (geographically nearby regions recovered as related tasks without spatial inputs).

## Assumptions

All tasks share one feature dimension `d`; rows of `W` are i.i.d. Gaussian (or Gaussian-copula); `W` is sparse; enough parameters/tasks exist for precision selection to be well-posed (`d` provides the effective sample size for the `m × m` precision); linear (or GLM) per-task model; the sparse-precision estimator is convex in each block.

## Differences from our setting

The paper’s model is a shallow linear/GLM classifier; ours are deep heads of widths 12 / 1251 / 4 hanging off a shared 512→2000 trunk, trained on disjoint datasets. MSSL supplies no `d × m` `W` for us — we must feed it the same **mean-head-summary matrix the classical MTRL control already uses** (each task contributes one summary vector, giving a common per-task dimension by construction). With `d ≈ 2000` summary rows and `m = 3` tasks the `3 × 3` precision problem is well-posed; disjoint datasets are handled because the loss is a per-task sum. The paper’s `1/n_k` re-weighting is sample-size-aware for the *loss*, not for the *relation estimate*. Saturation is addressed structurally rather than by a parameter: the `−d log|Ω|` barrier plus `ℓ₁` shrinkage on a full-rank `S` is a bounded, regularised precision parameterisation, unlike the closed-form `(WᵀW)^{1/2}` covariance, which collapses toward a rank-near-one, uniform (±1/3) `Ω`. This is the concrete mechanism family B was searching for.

## Implementation difficulty

Moderate. Requirements: a mean-head-summary matrix (already produced for the control), an FISTA `W` step with the `tr(W Ω Wᵀ)` coupling term instead of `tr(W Ω⁻¹ Wᵀ)`, an ADMM graphical-lasso solver for a `3 × 3` precision, and two hyperparameters (`λ₁` on `W`, `λ₂` on `Ω`; the paper uses stability selection to choose them). Cost is negligible at `m = 3`.

## Candidate Study ID

`LT-0002` — verified family-B candidate; proposed arm `TR-xxxx` (to be registered) with classical symmetric MTRL and the matched wavCSE baseline as the two controls under the shared protocol.

## LT-0002 assessment

1. **Explicit relation object — PASS.** A learned sparse task precision `Ω` with a partial-correlation interpretation.
2. **Taxonomy — PASS.** Explicit Task Relation Learning in the Zhang & Yang §2.4 sense (precision/covariance relation learned from data). Not low-rank, not clustering, not decomposition, not loss weighting.
3. **Heterogeneous-head compatibility — PASS.** The estimator operates on a `d × m` parameter matrix; the project’s existing mean-head-summary convention supplies exactly that, and it is the *same* adapter the control requires, so the variant does not change the input representation between arms. Disjoint datasets are native to the model. Caveat stated rather than hidden: the Gaussian-row premise is an approximation identical in kind to the control’s matrix-normal premise, and the summaries are rank-degenerate; MSSL does not remove that, it regularises around it.
4. **Fixed-representation compatibility — PASS.** `Ω` is learned from head parameters only; the regularizer `tr(W Ω Wᵀ) − d log|Ω|` back-props into the heads and never into frozen wavCSE/WavLM embeddings.
5. **Faithful implementability — PASS.** The published update rule (FISTA + ADMM graphical lasso, alternating) is implementable as written; hyperparameters come from the paper (or its natural defaults) via stability selection. Nothing is glued to an unrelated weighting scheme — the `1/n_k` loss scaling is part of the paper.
6. **Primary-source verification — PASS.** Authors, venue, year and equations checked against the JMLR PDF (JMLR 17(33), 2016); the `p`-MSSL/r-MSSL split, Eq. (3), Eq. (8) and the ADMM steps are from the source.

**Verdict: PASS — implementable as published.** Project note: the repo’s quarantined `PMR` (`models/pmr_model.py`, DEC-0004) is *not* this method — it learns a precision by gradient descent in a single combined loss, whereas published MSSL alternates FISTA with an ADMM graphical-lasso `Ω` step. A faithful MSSL arm is therefore new work; whether it un-quarantines the PMR line is a human decision, not a literature one.
