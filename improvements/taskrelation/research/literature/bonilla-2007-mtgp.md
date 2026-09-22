# Bonilla, Chai and Williams (2007) — Multi-task Gaussian Process Prediction

## Citation

Edwin V. Bonilla, Kian Ming A. Chai, and Christopher K. I. Williams. “Multi-task Gaussian Process Prediction.” *Advances in Neural Information Processing Systems 20 (NIPS 2007)*, pp. 153–160, 2008.

Primary source: https://proceedings.neurips.cc/paper/2007/hash/66368270ffd51418ec58bd793f2d9b1b-Abstract.html

## Problem

Learn inter-task dependencies from task identity and observed data alone, without task-descriptor features, so that related tasks transfer and unrelated tasks are not harmed. The target is a flexible relation model that needs little data per task.

## Mathematical assumption

Latent task functions `{f_l}` have a zero-mean GP prior whose covariance factorises as `⟨f_l(x) f_k(x′)⟩ = K^f_{lk} k_x(x, x′)` with `K^f` a PSD matrix of inter-task similarities and `k_x` a stationary input **correlation** function (unit variance, so variance is carried entirely by `K^f`). Observations are Gaussian: `y_{il} ∼ N(f_l(x_i), σ_l²)`. Stacking all tasks, `y ∼ N(0, K^f ⊗ K^x + D ⊗ I)` — not block-diagonal across tasks, so one task’s observations affect another’s predictions.

## Relation representation

An explicit learned `M × M` **free-form task covariance `K^f`**, parameterised as a Cholesky factor `K^f = LLᵀ` (guaranteeing positive semi-definiteness) or, for many tasks, a reduced-rank PPCA form `K^f ≈ ŨΛŨᵀ`. This is the canonical learned task-similarity object and is the direct ancestor of the covariance-relation family.

## Optimization method

Type-II maximum likelihood: maximise `p(y_o | X, θ_x, K^f)` by gradient descent on `L` (the Cholesky factor) and the input-kernel hyperparameters `θ_x`. An EM alternative exploits the Kronecker structure to give a closed-form `K^f` update, `K̂^f = (1/N) ⟨Fᵀ K^x(θ̂_x)⁻¹ F⟩`, with `K^f` guaranteed PSD; the authors report gradient descent usually outperformed EM in solution quality and speed.

## Evidence

Compiler performance prediction (11 tasks, 16/32/64/128 training points per task) and the ILEA school data set (139 tasks, 15,362 students). The free-form `K^f` beats no-transfer and is consistently (usually marginally) better than a parametric task-descriptor covariance; on school data a rank-2 `K^f` is best (29.2% variance explained vs 21.1% no-transfer, and 31.6% for the parametric task-descriptor model). The paper also proves a cancellation of inter-task transfer in the noise-free block design (autokrigeability), a real negative result for that regime.

## Assumptions

Gaussian likelihood with scalar real-valued outputs (regression); one shared input space `X` with tasks observed on (ideally block-designed) subsets of it; a single shared covariance function `k_x` across tasks; per-task noise `σ_l²`; `M(M+1)/2` parameters for a full `K^f`.

## Differences from our setting

Our tasks have disjoint data sets and heterogeneous multiclass heads (12 / 1251 / 4 classes); their model has one scalar latent function per task and a joint Gaussian likelihood over a shared input design. Frozen wavCSE/WavLM embeddings could play the role of `x`, and per-task `σ_l²` is a reliability term, but our likelihood is categorical cross-entropy, not Gaussian. Faithfully porting it would mean building a multi-class multi-task GP likelihood over head parameters — a new model, not a re-parameterisation of the existing `Ω`. Note also that the paper’s own free-form estimator becomes the sample covariance of decorrelated targets in the noise-free block design, i.e. it is not automatically immune to the rank collapse we observe.

## Implementation difficulty

High. Not a bolt-on regularizer: it requires marginal-likelihood optimisation (or EM) with a Cholesky-factorised `K^f`, an input kernel, and a multi-class likelihood replacement.

## Candidate Study ID

`LT-0002` — family-B candidate, screened and not advanced to a `TR-xxxx` arm as published.

## LT-0002 assessment

1. **Explicit relation object — PASS.** Learned `M × M` task covariance `K^f`, estimated from data by marginal likelihood.
2. **Taxonomy — PASS.** Task Relation Learning in the §2.4 sense; it is the origin of the covariance-relation lineage, not low-rank/clustering/decomposition/loss weighting. (Its own optional PPCA rank reduction is an approximation of `K^f`, not the method.)
3. **Heterogeneous-head compatibility — FAIL as published.** The formal model is scalar-output Gaussian regression over a shared input design; our disjoint multiclass heads with 12/1251/4 outputs require a different likelihood and a different parameterisation of the relation.
4. **Fixed-representation compatibility — PASS in principle.** Nothing in the estimator requires retraining the upstream representation; frozen embeddings can serve as `x`.
5. **Faithful implementability — FAIL.** The published update rule is a marginal-likelihood/EM procedure over latent function values, not a regularizer on classifier-head parameters. Implementing it here is a new hybrid.
6. **Primary-source verification — PASS.** Authors, venue, year, Eq. (1)–(5) and the experimental numbers checked against the NIPS 2007 proceedings PDF.

**Verdict: PASS WITH DOCUMENTED DEVIATION.** Deviation: the relation estimator would be applied to a deep multiclass model with disjoint data sets, replacing the Gaussian scalar-output likelihood. This counts as a fail for faithfulness (LT-0002 falsification condition), so the paper is eligible in category but not faithfully implementable in our setting.
