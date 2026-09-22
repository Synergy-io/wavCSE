# Yu, Tresp and Yu (2007) — Robust Multi-Task Learning with t-Processes

## Citation

Shipeng Yu, Volker Tresp, and Kai Yu. “Robust Multi-Task Learning with t-Processes.” *Proceedings of the 24th International Conference on Machine Learning (ICML 2007)*, pp. 1103–1110. DOI 10.1145/1273496.1273635.

Primary source (author copy): https://www.dbs.ifi.lmu.de/~tresp/papers/icml2007_tp.pdf

## Problem

Most multi-task frameworks assume all tasks are equally important and equally reliable. “Outlier” tasks — noisy, malicious or careless sources — can degrade the shared model. The paper wants a Bayesian multi-task model that distinguishes informative tasks from noisy ones and down-weights the latter.

## Mathematical assumption

A **t-process** (TP) is a heavy-tailed generalisation of a GP: for any finite input set, the function values follow a multivariate Student-`t` rather than a Gaussian, `f ∼ TP_ν(h, κ)`. Equivalently (Prop. 2.5), `f | τ ∼ GP(h, τ⁻¹κ)` with `τ ∼ Gamma(ν/2, ν/2)`, i.e. an infinite mixture of GPs. Multi-task learning is then: all `m` latent task functions share one TP prior, `f_ℓ | ν, h, κ ∼ t_ν(h_ℓ, K_{ℓ,ℓ})`, with Gaussian observations `y_ℓ | f_ℓ, σ² ∼ N(f_ℓ, σ²I)`. A conjugate Normal-Inverse-Wishart prior is placed on the finite prior `(h, K)`.

## Relation representation

**No explicit task-relation object.** Tasks are coupled only by sharing the prior `(h, κ)` over the *item* space; `K` is the covariance over items/inputs, not a task×task matrix. The per-task quantity learned is `τ_ℓ` (or its variational mean `α_ℓ/β_ℓ`), an **informativeness/scale weight** that shrinks toward zero when a task is poorly explained by the shared structure. Robustness is a per-task reliability weight, not a relation.

## Optimization method

Variational Bayes. E-step: iterate closed-form updates for the per-task posterior `(μ_ℓ, C_ℓ)` and the Gamma posterior `(α_ℓ, β_ℓ)` of `τ_ℓ`. M-step: update the shared mean `h`, covariance `K` and noise `σ²` by MAP/ML, where `h` and `K` become `τ`-weighted averages of the task posteriors. As `ν → ∞` every update reduces to the multi-task GP of Yu et al. (2005).

## Evidence

A 1-D synthetic study (15 “good” plus 5 “noisy” functions; TP recovers `h` and `K` while the GP is biased and down-weights functions 16–20), robust collaborative filtering on MovieLens (943 users, 50 synthetically injected noisy users of three kinds, all recovered; TP beats GP on RMSE/MAE/MZOE with `p = 0.01`), and indoor temperature prediction on a Berkeley sensor network.

## Assumptions

A shared item set `X` with task `ℓ` labelling a subset `I_ℓ ⊆ X` (in the fully-observed case all tasks label the same items); scalar real-valued latent functions; Gaussian observation noise by default (a `t` noise model is discussed as an extension); a fixed degrees-of-freedom `ν` (the paper deliberately does not estimate it, citing bad local minima).

## Differences from our setting

There is no `m × m` relation object to implement — the paper’s mechanism is a shared hierarchical prior plus per-task reliability scalars. Our three tasks have disjoint data sets rather than a shared item set, and heterogeneous multiclass heads. Even if adapted, the resulting object would be a per-task weight, which the project’s taxonomy already excludes (the same family as Kendall’s uncertainty weighting), not a task-relation structure.

## Implementation difficulty

High (variational Bayes over latent functions) and, more importantly, not the right object: adapting it would mean inventing a task-relation mechanism the paper does not contain.

## Candidate Study ID

`LT-0002` — family-B robustness lead, verified and closed.

## LT-0002 assessment

1. **Explicit relation object — FAIL.** The shared prior over items is not a task-relation structure; `τ_ℓ` is a per-task scalar. No covariance/precision/graph/estimator over tasks is learned.
2. **Taxonomy — FAIL.** Per-task reliability weighting is out of the §2.4 Task Relation Learning category (uncertainty/robustness weighting family).
3. **Heterogeneous-head compatibility — FAIL.** Requires a shared item set and scalar latent functions; disjoint multiclass heads do not fit.
4. **Fixed-representation compatibility — PASS in principle.** Nothing forces upstream retraining.
5. **Faithful implementability — FAIL.** There is no relation update rule to implement as published; a port would be a new mechanism.
6. **Primary-source verification — PASS.** Authors (Yu, Tresp, Yu — note Schwaighofer is *not* an author of this paper), venue (ICML 2007, pp. 1103–1110), Defs. 2.4/Props. 2.5–2.7, the VB updates (2)–(6) and the MovieLens/temperature results checked against the ICML PDF.

**Verdict: FAIL — heavy-tailed robustness in this line is a per-task reliability weight attached to a shared hierarchical prior, not a task-relation object.** Documented negative for the “robust/heavy-tailed covariance estimation for task relations” lead: the paper is robust, but its robustness object is not a relation.
