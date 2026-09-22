# Task Relation Learning Literature Registry

Primary-source literature cards for the formal Task Relation Learning programme. Cards record both eligible mechanisms and explicit taxonomy/assumption rejections; a paper’s presence here does not mean it is approved for implementation.

## Current literature Study

`LT-0002 — Published relation-learning variants for two selected families` (DEC-0013)

**Decision:** `COMPLETE — candidates found`. Family **B** (relation estimator / task-parameter representation) yields one clean pass: **Gonçalves et al. 2016 p-MSSL**, a sparse task *precision* learned by graphical lasso. Family **A** (asymmetric / directed relations) yields three methods whose relation object and update rule are published but which require a declared project-level deviation (the relation acts on class-mean head summaries rather than per-task model parameters): **Lee et al. 2016 AMTL**, **Zhou & Yang 2023 AutoTR**, **Oliveira et al. 2019 GAMTL**. Whether deviation-class arms are admissible is a human call, because LT-0002's falsification rule counted a documented deviation as a faithfulness fail.

`LT-0001 — Scale- and reliability-aware task relation literature gate` (superseded premise)

**Decision:** `REJECTED` candidate-method hypothesis. No reviewed published method simultaneously satisfies the explicit-relation, direct-F9, heterogeneous-classification, faithful-implementation, and category-boundary gates. Its rejections were scoped to the F9 gradient-scale rationale, which DEC-0010 later withdrew — see the closure note at the end of this file.

## Cards — LT-0002 (DEC-0013 benchmark candidates)

| Paper | Family | Relation object | Verdict |
| --- | --- | --- | --- |
| [Gonçalves et al. 2016 — p-MSSL](goncalves-2016-mssl.md) | B — estimator | Sparse task **precision** via graphical lasso (`−d log|Ω|` + `ℓ₁`) | **PASS** |
| [Lee et al. 2016 — AMTL](lee-2016-asymmetric-mtl.md) | A — directed | Sparse non-negative **directed** transfer graph, loss-scaled rows | PASS WITH DOCUMENTED DEVIATION |
| [Zhou & Yang 2023 — AutoTR](zhou-2023-autotr.md) | A — directed | Signed directed relation matrix `R`, closed-form update | PASS WITH DOCUMENTED DEVIATION |
| [Oliveira et al. 2019 — GAMTL](oliveira-2019-group-lasso-asymmetric.md) | A — directed | Per-covariate-group directed matrices | PASS WITH DOCUMENTED DEVIATION (needs an invented group partition) |
| [Bonilla et al. 2007 — multi-task GP](bonilla-2007-mtgp.md) | B — estimator | Free-form PSD task covariance (Cholesky/PPCA) | Deviation — Gaussian regression outputs |
| [Liu & Pan 2017 — trace-Lasso GAMTL](liu-2017-trace-lasso-gamtl.md) | A (name collision) | Signed directed matrix, but for grouping | FAIL — clustering category |
| [Yu et al. 2020 — graph-adjacency GAMTL](yu-2020-graph-adjacency-gamtl.md) | A (name collision) | Undirected adjacency by construction | FAIL — not directed |
| [Lee et al. 2018 — Deep-AMTFL](lee-2018-deep-asymmetric-mtfl.md) | A | Task-to-basis matrix inside a low-rank model | FAIL — low-rank/basis, out of taxonomy |
| [Nguyen et al. 2021 — TP-AMTL](nguyen-2021-tp-amtl.md) | A | Per-pair uncertainty attention, no relation matrix | FAIL — uncertainty weighting; temporal only |
| [Graffeuille et al. 2024 — SAAL](graffeuille-2024-self-auxiliaries.md) | A | Directed coefficients over auxiliary clones | FAIL — loss weighting; Eq. (4) undefined here |
| [Zhang & Yeung 2014 — MTRL, "asymmetric"](zhang-yeung-2014-mtrl-asymmetric.md) | A (attribution guard) | Symmetric covariance; "asymmetric" = target/source setting | FAIL — no directed relation object |
| [Zhang & Schneider 2010 — sparse matrix-normal](zhang-schneider-2010-sparse-matrix-normal.md) | B — estimator | Joint task+feature precision | FAIL — feature precision unidentifiable at m = 3 |
| [Zhao et al. 2020 — FETR](zhao-2020-fetr.md) | B — estimator | Bounded task precision + `d×d` feature precision | FAIL — needs a shared `d×m` head |
| [Yu et al. 2007 — t-processes](yu-2007-t-processes.md) | B — estimator | None (per-task reliability weights) | FAIL — no relation object |
| [Fifty et al. 2021 — TAG](fifty-2021-tag.md) | B — estimator | Directed affinity matrix for **grouping** | FAIL — clustering category |

## Cards — LT-0001 (F9 rationale, closed)

| Paper | Role | Explicit relation? | Direct F9 scale/reliability mechanism? | LT-0001 decision |
| --- | --- | --- | --- | --- |
| [Zhang & Yang 2021 — MTL survey](zhang-yang-2021-mtl-survey.md) | Binding taxonomy | Defines category | N/A | Retain as taxonomy authority |
| [Zhang & Yeung 2010 — MTGTP](zhang-yeung-2010-mtgtp.md) | Bayesian relation uncertainty / robust likelihood | Yes, covariance prior | No direct gradient-scale control | Reject for implementation |
| [Rakitsch et al. 2013 — structured residuals](rakitsch-et-al-2013-structured-residuals.md) | Separates signal relation from residual/noise relation | Yes, two task covariances | Conceptually closest; output-noise rather than gradient scale | Reject now; retain diagnostic principle |
| [Feldman et al. 2014 — multi-task averaging](feldman-et-al-2014-mta.md) | Sample/variance-aware relation strength | Yes, similarity graph | Yes for mean-estimation variance, not deep classification | Reject as mechanism; retain framework evidence |
| [Zhang & Yang 2017 — SPATS](zhang-yang-2017-spats.md) | Sparse covariance relations | Yes | No; targets many-task sparsity | Reject for F9 |
| [Kendall et al. 2018 — uncertainty weighting](kendall-et-al-2018-uncertainty-weighting.md) | Task reliability / loss scale | No | Yes | Exclude by taxonomy |
| [Chen et al. 2018 — GradNorm](chen-et-al-2018-gradnorm.md) | Gradient-magnitude balancing | No | Yes, direct symptom match | Exclude by taxonomy |
| [Chang et al. 2024 — informative relations](chang-et-al-2024-informative-relations.md) | Recent explicit relation over shared component | Yes | No | Exclude by decomposition boundary |

## LT-0001 synthesis

The literature separates the two properties F9 connects:

1. **Explicit Task Relation Learning** methods learn parameter/function covariance, sparse relations, or signal/noise covariance. The reviewed faithful formulations do not regulate shared-gradient magnitude in heterogeneous deep classifiers.
2. **Scale/reliability methods** such as uncertainty weighting and GradNorm directly address loss or gradient dominance, but learn per-task scalar weights rather than task relations and therefore belong to optimization/loss-balancing, not the binding Task Relation Learning category.

The closest explicit-relation paper, Rakitsch et al. (2013), learns separate signal and residual task covariance matrices. Its fully observed aligned Gaussian multi-output regression assumptions do not hold for disjoint KS/SI/ER datasets and heterogeneous multiclass heads. Porting only the idea would require a novel deep-classification hybrid, not a faithful published implementation.

No mechanism Study is authorized by LT-0001. The programme was left in `NEEDS-HUMAN-REVIEW`; that has since resolved as follows (2026-09-22):

* **DEC-0009 (human):** strict Task Relation Learning scope retained; broadening to optimization-aware MTL declined; a project-original hybrid deferred behind explicit authorization.
* **DG-0005 (CONFIRMED, F10):** the ER data-regime control was run. It showed the gradient-scale signal is a **training-mixture property**, so the F9-era rationale for a scale- or reliability-aware relation mechanism is withdrawn (DEC-0010).
* **DEC-0011:** this literature mandate is closed. Do not re-run this search against the same evidence.

Consequence for the cards above: the direct-scale methods (GradNorm, uncertainty weighting) remain excluded by taxonomy; the explicit-relation methods remain ineligible as published mechanisms; and Rakitsch et al.'s retained "diagnostic principle" is now less well supported than when LT-0001 closed it, because the measured signal turned out to be mixture-driven gradient-estimate scale rather than demonstrated residual/output noise. The open decisions are listed in `../FRAMEWORK.md` §6.
