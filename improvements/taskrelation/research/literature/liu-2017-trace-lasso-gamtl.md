# Liu and Pan (2017) — Adaptive Group Sparse Multi-task Learning via Trace Lasso

## Citation

Sulin Liu and Sinno Jialin Pan. “Adaptive Group Sparse Multi-task Learning via Trace Lasso.” *Proceedings of the 26th International Joint Conference on Artificial Intelligence (IJCAI-17)*, pp. 2358–2364, 2017. DOI 10.24963/ijcai.2017/328.

Primary source: https://www.ijcai.org/Proceedings/2017/0328.pdf

Verified against that PDF (abstract; §3–§5, Eqs. (1)–(4); Theorem 1; Algorithm 1; Tables 1 and Figs. 1–3; reference list). Naming hazard: these authors call their method **GAMTL** (Group Adaptive MTL), as do Oliveira et al. (Group Asymmetric MTL) and Yu et al. (Graph Adjacency MTL) — three distinct methods share the acronym; always cite by author/year.

## Problem

Clustered/grouped MTL methods require the number of clusters or latent bases in advance, and existing sparse-relation methods (AMTL) recover a relation that is too sparse to reveal grouping: with highly correlated tasks, an `ℓ₁` penalty keeps one representative and zeroes the others. The paper wants an adaptive grouping mechanism without pre-specifying the group count.

## Mathematical assumption

Each task parameter is a linear combination of the other tasks’ parameters,

`w_i = W^{[i]} c_i` (`C ∈ R^{m×m}`, `C_ii = 0`), with `C` entries allowed to be positive, zero **or negative** and with no symmetry constraint, so the learned relation is signed and directed. Grouping is then read off the sparsity pattern of `C`: tasks in one group should share a sparsity pattern, making a reordered `C` block-diagonal.

## Relation representation

A learned `m × m` sign-free relation matrix `C` whose column `c_i` gives the coefficients by which the other task parameters reconstruct `w_i`; entry `C_ki` is “how much of task `k`’s model is transferred to task `i`”. Regularisation is a **trace-Lasso** penalty, `‖W^{[i]} Diag(c_i)‖_*`, which interpolates between `ℓ₁` (uncorrelated predictors → one representative task) and `ℓ₂` (highly correlated predictors → whole group selected), replacing AMTL’s `ℓ₁`. The objective is

`min_{W,C} Σ_i L(w_i; D_i) + (λ/2) Σ_i ‖w_i − W^{[i]} c_i‖₂² + γ Σ_i ‖W^{[i]} Diag(c_i)‖_*`.

The relation `C` *is* learned from data (not user-supplied), and it is directed: `C_ki ≠ C_ik` in general.

## Optimization method

Biconvex; not jointly convex (Theorem 1, with a counterexample). Alternating optimization (Algorithm 1): with `W` fixed, solve for `C` by ADMM (matrix soft-thresholding in the `J`-step, closed-form least-squares `C`-step); with `C` fixed, solve for `W` by ADMM with the same structure. Regularisation parameters `λ`, `γ` by cross-validation.

## Evidence

Synthetic: 100 classification tasks in 5 clusters (sizes 10/20/20/20/30), each cluster built from 3–5 correlated tasks; the learned `C` recovers the block structure that AMTL’s sparse `C` cannot, with test error `{7.55 %, 1.56 %}` vs AMTL `{17.43 %, 17.53 %}` on the positive-relation variant. Real: School (139 schools, regression), MDS (22 sentiment domains), CIFAR (45 one-vs-one binary tasks). GAMTL best on MDS (12.82 %) and CIFAR (16.63 %); GO-MTL best on School; MTRL close on MDS. Conclusions stress *grouping recovery*, not asymmetric transfer.

## Assumptions

All tasks have one predictor vector of the same length `d` (all experiments are binary/regression with a single weight vector per task); each task has its own design matrix but the same feature dimensionality; tasks are believed to fall into overlapping groups; the interesting structure is the grouping.

## Differences from our setting

* **Purpose.** The trace-Lasso machinery exists to recover a *grouping*; block structure in `C` is the paper’s headline result. Grouping/clustering is explicitly outside the project’s Task Relation Learning scope.
* **Head widths.** Binary tasks with equal-width predictors; our 12 / 1251 / 4-class heads again require the class-mean summary device before `W^{[i]}c_i` is even defined, and the summary would then be reconstructed by signed combinations of other summaries — the mechanism is defined, but it is the same deviation as the AMTL card, on top of a grouping objective we do not want.
* **Shallow only.** No deep extension; the reconstruction is on the raw linear predictor, not on a trunk.
* **Relation to family A.** Mathematically `C` is AMTL’s relation object with signs permitted and a non-convex (trace-norm) penalty; the paper is careful to state that it builds on Lee et al. (2016) and changes the penalty to expose groups. It is therefore a *related-object* near-miss whose contribution lies in the excluded category.

## Implementation difficulty

Moderate (ADMM for both blocks, trace-norm/matrix soft-thresholding, two hyperparameters), but the implementation would deliver a clustering mechanism, not a directed-relation transfer mechanism, and would not be a family-A arm.

## Candidate Study ID

`LT-0002` — screened candidate (family A), recorded as a taxonomy failure.

## LT-0002 assessment

* **Gate 1 — explicit relation object: PASS.** A learned, signed, directed relation matrix `C` (no user-supplied relation).
* **Gate 2 — taxonomy: FAIL.** The paper’s stated contribution is *adaptive group sparse* MTL — recovering task clusters/groups; the project excludes clustering/task-grouping from §2.4 and reserves the category for quantitative relation learning. The trace-Lasso penalty is a grouping device, not a task-relation parameterisation.
* **Gate 3 — heterogeneous heads: FAIL in the published form.** Requires equal-length task parameter vectors (binary tasks); the class-mean summary would be needed *and* the objective would then be a grouping objective on summaries.
* **Gate 4 — fixed representation: PASS.** Shallow, downstream-only; frozen embeddings untouched.
* **Gate 5 — faithful implementability: PASS mechanically, FAIL purposefully.** The update rule is implementable as published, but what it implements is a task-grouping method — an arm of the wrong category.
* **Gate 6 — source verified: PASS.** Venue, pages (2358–2364), DOI, equations and evidence checked against the official IJCAI PDF.

**Verdict: `FAIL — the contribution is adaptive task grouping via a trace-Lasso penalty, i.e. the clustering/task-grouping category the project excludes; the signed directed relation matrix is mathematically AMTL’s object but the method is not a directed-relation transfer mechanism.`**
