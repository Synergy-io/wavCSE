# Zhou and Yang (2023) — Automatic Temporal Relation in Multi-Task Learning

## Citation

Menghui Zhou and Po Yang. “Automatic Temporal Relation in Multi-Task Learning.” *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’23)*, pp. 3570–3580. ACM, 2023. DOI 10.1145/3580305.3599261.

Primary source (author-accepted version): https://eprints.whiterose.ac.uk/id/eprint/199291/8/AutoTR_KDD_Camera_Ready_v4.pdf

Verified against that PDF (abstract; Eqs. (1)–(13); Algorithm 1; Tables 1–5; Figs. 1–4; reference list). Code: https://github.com/menghui-zhou/AutoTR

## Problem

Multi-task learning with *temporal* relation (each time point = one task) uses predefined, symmetric relation structures — temporal smoothness (`w_k ≈ w_{k+1}`) or mean temporal relation (`w_k ≈ mean of the others`). Both are rigid and provably symmetric, so they cannot express unevenly spaced, direction-dependent temporal relations.

## Mathematical assumption

Each task’s parameter is a *free, signed* linear reconstruction of all other task parameters,

`w_k ≈ Σ_{x≠k} r_{x,k} w_x`, equivalently `W ≈ W R`, `r_{k,k} = 0`,

with no symmetry constraint (`r_{x,k} ≠ r_{k,x}`) and no non-negativity constraint (the learned matrix contains small negative entries). The directed structure is therefore *learned*, not assumed.

## Relation representation

A learned `m × m` directed relation matrix `R` — the adjacency matrix of a directed graph over tasks — where `r_{x,k}` is the relation from task `x` to task `k`. The objective is

`min_{W,R} L(W) + λ₁‖W − W R‖²_F + λ₂‖R ⊙ S‖_{1,1}`, `S = (s−1)I_m + 1_{m×m}`, `s = 10⁹`,

so `‖R ⊙ S‖_{1,1}` is `ℓ₁` sparsity with the diagonal suppressed (the paper notes `s` is a pseudo-hyperparameter, not tuned).

**Attachment in our model.** Identical in form to AMTL: the coupling parameter is `R ∈ R^{3×3}` (six free signed entries). It requires one parameter *column per task* (`W ∈ R^{d×m}`), which our 12 / 1251 / 4-wide heads do not provide as published; the same class-mean head summary used by the in-category control gives `W = [w̃_KS, w̃_SI, w̃_ER] ∈ R^{2000×3}` and aligns the columns. Because `T = 3`, both `R` and any symmetric counterpart are fully identified; a matched-pair control against the symmetric MTRL `Ω` is a direct comparison.

## Optimization method

Biconvex. Alternating optimization (Algorithm 1): fix `R`, update `W` by accelerated proximal gradient (APM) — both terms of the `W` subproblem are smooth, so no proximal operator is needed; fix `W`, update `R` by soft thresholding, which has the closed form `π(R) = max(|R| − λ₂S, 0) ⊙ sgn(R)` (cost `O(m²)`). A Gaussian-kernel warm start over time indices speeds convergence (up to 17.6×); the zero-initialised variant (`AutoTR-0`) is reported as the baseline algorithm and is the variant applicable to non-temporal task sets.

## Evidence

Six public datasets spanning motor/total UPDRS, MMSE, ADAS-Cog, RAVLT (Alzheimer’s progression, `m` = 12 time points, 314 features) and Weather; baselines Ridge, Lasso, TaskTS (task-level temporal smoothness), FeaTS (feature-level), MeanTR (mean temporal relation). AutoTR is best overall and on most individual tasks (t-tests, `p < 0.05`, one exception at `p = 0.057`). Learned `R` matrices are shown to be non-symmetric in every dataset, with a few small negative entries; adjacent tasks have the strongest relation. No experiment uses heterogeneous output widths or disjoint feature spaces.

## Assumptions

Each task is one column of a single shared coefficient matrix `W ∈ R^{d×m}` over a *shared* feature space; a linear (or kernelised) predictor per task; tasks are time points, so a decaying-similarity warm start is meaningful; sparsity on the directed relation is desirable; `d` identical across tasks.

## Differences from our setting

* **Non-temporal tasks.** Our KS/SI/ER tasks are not a time series: the Gaussian-kernel warm start is inapplicable (use `AutoTR-0`, which the paper evaluates), and the *interpretation* of `r_{x,k}` as past→future is lost, though the estimation mechanism is task-agnostic and unaffected.
* **No aligned columns.** As published, the method acts on the actual task coefficient columns; here it can act only on the class-mean head summary (same deviation as the AMTL card), with the same uniform-pull side effect across the classes of a head.
* **Disjoint data.** `L(W)` is a per-task empirical risk, so disjoint datasets are fine; there is no requirement of identical instances.
* **Attribution.** `W ≈ W R` with an `ℓ₁`-penalised relation matrix is formally AMTL’s relation object *without* the loss-scaled sparsity (i.e. AMTL-noLoss) and *without* non-negativity. AutoTR’s reference list does not cite Lee et al. (2016), and its “asymmetry not considered” claim is scoped to temporal smoothness / mean-relation baselines. Record this so the two are not conflated.

## Implementation difficulty

Moderate and closely comparable to AMTL: summaries → gradient step on `W` (including `λ₁‖W − WR‖²_F`) → periodic closed-form soft-thresholding update of `R`. The single difference from AMTL is that `R` is signed and solved in closed form rather than by a non-negative weighted LASSO, which makes AutoTR the *simpler* of the two to implement under one optimizer.

## Candidate Study ID

`LT-0002` — screened candidate (family A). If adopted, one `TR-xxxx` variant Study is pre-registered per DEC-0013 under the shared matched protocol.

## LT-0002 assessment

* **Gate 1 — explicit relation object: PASS.** `R` is a learned directed relation matrix (weighted adjacency over tasks), estimated jointly with the parameters. It is not a user-supplied relation.
* **Gate 2 — taxonomy: PASS.** Explicit relation learning in the §2.4 sense — parameter reconstruction via a relation matrix — with no low-rank, clustering, decomposition, uncertainty or gradient-surgery component.
* **Gate 3 — heterogeneous heads: PASS ONLY WITH STATED CHANGE.** The formal assumption needs one coefficient column per task; head widths 12 / 1251 / 4 break this. The stated change (class-mean head summary over the shared 2000-dim hidden layer) restores the required alignment but shifts the penalty’s effect onto the summary rather than the task models.
* **Gate 4 — fixed representation: PASS.** No upstream retraining; the relation regularises downstream parameters over frozen embeddings.
* **Gate 5 — faithful implementability: PASS.** Eqs. (2)–(3), (7)–(10) and Algorithms 1–2 are directly implementable; hyperparameters `λ₁`, `λ₂` are grid-searched in the paper and `s = 10⁹` is stated to be a pseudo-hyperparameter that needs only to be “large enough”.
* **Gate 6 — source verified: PASS.** Venue, authors, equations and evidence checked against the accepted-version PDF and the published DOI record.

**Verdict: `PASS WITH DOCUMENTED DEVIATION`** — a clean, closed-form-implementable signed directed relation matrix, but as in the AMTL card the published object (`W`’s per-task columns) does not exist in our architecture and must be replaced by the class-mean head summary. Per LT-0002 that documented deviation is not a faithful implementation.
