# Zhao, Stretcu, Smola and Gordon (2020) — Efficient Multitask Feature and Relationship Learning

## Citation

Han Zhao, Otilia Stretcu, Alexander J. Smola, and Geoffrey J. Gordon. “Efficient Multitask Feature and Relationship Learning.” *Proceedings of the 35th Conference on Uncertainty in Artificial Intelligence (UAI)*, PMLR 115:777–787, 2020.

Primary source: https://proceedings.mlr.press/v115/zhao20a.html

## Problem

The established multitask feature-and-relationship framework (MTFRL, Zhang & Schneider 2010) learns a task covariance and a feature covariance under a matrix-variate normal prior, but the paper shows its maximum-likelihood formulation is **ill-posed**: the flip-flop algorithm requires `n > max(d/m, m/d)` samples, whereas in MTL the parameter matrix `W` is a single unknown draw, i.e. `n = 1`. The covariance updates then produce rank-deficient matrices, the inverse is undefined, and published fixes add a fixed fudge `εI` that biases the estimate and whose convergence is unclear. The objective is even unbounded (`W = 0`, precisions scaled to infinity).

## Mathematical assumption

`W ∈ ℝ^{d×m}` with `vec(W) ∼ N(0, Ω₁⁻¹ ⊗ Ω₂⁻¹)`. The **FETR** variant constrains the precisions to a compact set `l I ⪯ Ω₁ ⪯ u I`, `l I ⪯ Ω₂ ⪯ u I`, which they interpret as a truncated matrix-normal prior; the feasible set is compact, so the objective attains finite bounds and the problem is well-defined. The objective is `‖Y − XW‖_F² + tr(Ω₁ W Ω₂ Wᵀ) − (m log|Ω₁| + d log|Ω₂|)`, multiconvex in `(W, Ω₁, Ω₂)`.

## Relation representation

Two explicit precision matrices: the task precision `Ω₂` (`m × m`) and the feature precision `Ω₁` (`d × d`). The task precision is the relation object; the feature precision is the feature-structure object. Their contribution is not a new relation *form* but a **bounded, well-posed** parameterisation of the same relation object — i.e. a covariance/precision parameterisation explicitly designed so the estimate cannot collapse or diverge.

## Optimization method

Block coordinate-wise minimisation, alternating `W`, `Ω₁`, `Ω₂`. The `W` step has three published options: closed form via an `md × md` system, gradient descent with a proven linear convergence rate, or a Sylvester equation solved by Bartels–Stewart. The `Ω₁`/`Ω₂` steps reduce to one SVD plus a hard-thresholding `T_{[l,u]}(x) = max{l, min{u, x}}` on the eigenvalues (via a min-weight perfect matching argument on a bipartite graph over the Birkhoff polytope), giving a closed-form solution per block.

## Evidence

Synthetic convergence/scalability studies and real regression/classification data sets. FETR is reported orders of magnitude faster than the flip-flop and projected-gradient baselines and often converges to better solutions. A **nonlinear extension** replaces the feature matrix `X` with a neural network `g(x; θ)` and keeps `W` as a linear layer on top, reporting significantly better generalisation.

## Assumptions

Linear regression per task (classification described as a straightforward extension); a feature matrix `X` shared by the tasks for the closed-form/Sylvester `W` steps (gradient descent is offered when tasks have different inputs); fixed `d` and `m`; bounded precision constants `l, u` chosen by the user; joint identifiability of the pair `(Ω₁, Ω₂)` (their product is what is identified).

## Differences from our setting

Our relations are among **per-task classifier heads of different widths**, not columns of one shared `d × m` weight matrix. FETR’s nonlinear extension is a *single shared* `p × m` layer on a shared neural map, which contradicts our per-task heterogeneous heads. Worse, the `d × d` feature precision must be estimated from `m` columns: with `m = 3` tasks it is radically under-determined (`d ≈ 2000 ≫ m`). The bounded constraint guarantees the estimate is *finite*, not that it is *identifiable*; the paper’s own ill-posedness argument (`n = 1 < max(d/m, m/d)`) applies with full force to a `2000 × 2000` feature precision. Fixing `Ω₁ = I` to escape this removes the paper’s defining joint feature-and-relation mechanism.

## Implementation difficulty

Moderate for the relation block itself (SVD + hard-thresholding on a `3 × 3` precision is trivial), but the published method also requires the feature-covariance machinery, which is not implementable meaningfully with three tasks. Adopting only the bounded task-precision step would be a hybrid.

## Candidate Study ID

`LT-0002` — screened family-B candidate; not advanced.

## LT-0002 assessment

1. **Explicit relation object — PASS.** Learned task precision `Ω₂`, with a conditional-independence interpretation.
2. **Taxonomy — PASS (as a whole).** Explicit Task Relation Learning in the §2.4 sense; the feature precision is feature-structure learning, but the relation object is the task precision, not a low-rank/clustering/decomposition object.
3. **Heterogeneous-head compatibility — FAIL.** The formal model requires one shared `d × m` parameter matrix (or a single shared head layer in the nonlinear extension); it does not survive disjoint data sets with head widths 12 / 1251 / 4.
4. **Fixed-representation compatibility — PASS in principle.** The nonlinear extension freezes/hosts the representation; nothing forces upstream retraining.
5. **Faithful implementability — FAIL.** The published method jointly estimates a `d × d` feature precision that is unidentifiable with `m = 3` tasks; the bounded variant makes the subproblem finite but not meaningful. Implementing only the bounded `Ω₂` step is a deviation from the published mechanism.
6. **Primary-source verification — PASS.** Authors, venue (UAI 2020, PMLR 115:777–787), and the ill-posedness argument (Section 3), the bounded formulation (Eq. 3–4) and the block algorithms (Algorithms 1–2) checked against the PMLR PDF.

**Verdict: FAIL — the published method’s joint feature-and-task covariance is unidentifiable with three tasks, and its parameterisation requires a shared `d × m` head incompatible with our heterogeneous per-task heads.** Its ill-posedness analysis is nevertheless the cleanest published statement of why unconstrained covariance estimation of this kind degenerates, which is directly relevant to our observed `Ω` saturation.
