# Yu, Alesiani, Shaker and Yin (2020) — Learning an Interpretable Graph Structure in MTL

## Citation

Shujian Yu, Francesco Alesiani, Ammar Shaker, and Wenzhe Yin. “Learning an Interpretable Graph Structure in Multi-Task Learning.” arXiv:2009.05618 [cs.LG], 11 September 2020 (v1; the v2 identifier returned 404 at the time of checking — cite v1).

Primary source: https://arxiv.org/abs/2009.05618 (full text: https://arxiv.org/pdf/2009.05618v1)

Verified against the v1 PDF (abstract; §II–§IV; Eqs. (1)–(14); Theorems 3.1–3.4; Algorithms 1–3; Tables I–III). Naming hazard: the authors name their method **GAMTL** (Graph Adjacency MTL), colliding with Oliveira et al.’s Group Asymmetric MTL and Liu & Pan’s Group Adaptive MTL.

## Problem

Task covariance/precision matrices are not guaranteed to be valid graph Laplacians, most graph-based MTL assumes the topology a priori or estimates it in a separate step, and covariance matrices are hard to interpret. The paper wants to learn a sparse, interpretable *graph* jointly with the task parameters.

## Mathematical assumption

Model parameters `W = [w_1 … w_T]` (one column per task) with a graph regulariser built on pairwise parameter distances `Z_ij = ‖w_i − w_j‖₂²`. The objective is

`min_{W, A} Σ_t ‖w_tᵀx_t − y_t‖₂² + ‖A ⊙ Z‖_{1,1} − 1ᵀlog(A1) + β‖A‖²_F`,

over `A ∈ 𝒜 = {A ∈ S^T : A_ij ≥ 0 ∀i≠j, diag(A) = 0}` — a weighted adjacency matrix with non-negative weights, a logarithmic barrier on node degrees to keep the graph connected, and a Frobenius term to control sparsity. The graph is learned jointly with `W` (no separate preprocessing), and the paper extends it to a multi-head RBF network (`RBF-GAMTL`, Eq. (11)).

## Relation representation

A learned `T × T` weighted adjacency matrix `A` — and it is **symmetric by construction**. The implementation stores only the upper part of `A` (“`w` is the upper part of `A`, thus enforcing `A` to be symmetric”), and the authors state plainly that they “aim to learn an **undirected** graph that is sparse and much easier to interpret”, contrasting it with a *directed* message-passing alternative they explicitly do not target. The relation is estimated from data; its direction is not.

## Optimization method

Biconvex in `(W, A)`; alternating optimization (Algorithm 1). The `W` step reduces to a linear system (`(B + C)V = D`, solved with a combinatorial-multigrid solver); the `A` step is solved by a primal-dual algorithm (`A`-update in Algorithm 2) with barrier and `ℓ₁` terms. Convergence to a first-order (and, in a proximal version, second-order) stationary point is argued from bi-convexity. Complexity `O(d³T³ + d²N + T³)`, reducible when `A` is sparse.

## Evidence

Synthetic Syn 1 (20 tasks, two groups plus two outliers) and Syn 2 (a circular neighbour relation): the learned graph matches the generating structure more faithfully than MTRL, MSSL, BMSL, TAT, GFL and CCMTL. Real: Parkinson’s telemonitoring (42 patient tasks, RMSE over train/test ratios), Birmingham parking occupancy (29 lots), and a distributed system-identification simulation (10 nodes). RBF-GAMTL helps on the nonlinear data but not on the linear synthetic sets.

## Assumptions

Regression only (`y ∈ R`, squared loss, one shared feature space per task); comparability of task parameter vectors (all tasks have a `d`-dimensional predictor, and `Z` is a parameter distance); the relation of interest is a *symmetric* similarity/adjacency; interpretability of the graph is a goal in itself.

## Differences from our setting

* **Family fit.** The family-A question is whether a method learns a *directed* relation. This one is explicitly undirected; `A_ij = A_ji` is not a learned outcome but an implementation constraint, and the authors position themselves against the directed-graph alternative.
* **Object and heads.** Parameter-distance regularisation needs comparable parameter vectors (`Z_ij` is a Euclidean distance between parameters), the exact construct our 12 / 1251 / 4-class heads cannot provide without the class-mean summary; and the method is regression-only.
* **Interpretation target.** The paper optimises for graph interpretability (outlier detection, visual task topology) rather than for asymmetric transfer.

## Implementation difficulty

Moderate (alternating scheme with a primal-dual `A`-step and a CMG linear solve), but it cannot serve as a directed-relation arm: the relation object is undirected by construction, so there is nothing to compare against a *directed* variant of the control.

## Candidate Study ID

`LT-0002` — screened candidate (family A), recorded as an out-of-family near-miss (with the GAMTL name collision).

## LT-0002 assessment

* **Gate 1 — explicit relation object: FAIL for this family.** An explicit learned relation object exists (weighted adjacency `A`), and it is estimated from data — but it is **symmetric/undirected**, and the paper states this as its aim. No directed transfer graph, no asymmetric parameterisation.
* **Gate 2 — taxonomy: PASS (as an undirected relation method).** Graph-structured task-relation learning; it is not low-rank, clustering (the graph is not a partition), decomposition, uncertainty or gradient surgery.
* **Gate 3 — heterogeneous heads: FAIL.** Requires comparable, equal-length parameter vectors for the pairwise distances; regression outputs only.
* **Gate 4 — fixed representation: PASS.** Downstream-only; frozen embeddings untouched.
* **Gate 5 — faithful implementability: PASS mechanically.** Algorithms 1–3 are implementable; the method is simply not a directed-relation mechanism.
* **Gate 6 — source verified: PASS.** Authors, equations, algorithms and evidence checked against the arXiv v1 PDF.

**Verdict: `FAIL — the learned relation object is an explicitly undirected adjacency matrix; the paper states the undirected graph as its design goal and contrasts itself with directed graph MTL. It supplies no directed relation-learning mechanism.`** Recorded mainly to prevent the three-way GAMTL name collision from producing a false family-A candidate.
