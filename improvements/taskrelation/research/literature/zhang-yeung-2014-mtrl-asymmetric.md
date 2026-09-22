# Zhang and Yeung (2014) — MTRL in the symmetric and asymmetric settings

## Citation

Yu Zhang and Dit-Yan Yeung. “A Regularization Approach to Learning Task Relationships in Multi-Task Learning.” *ACM Transactions on Knowledge Discovery from Data* 8(3), Article 12, pp. 1–31, 2014. DOI 10.1145/2538028.

Primary source (author copy): https://yuzhanghk.github.io/papers/Zhang_Yeung_TKDD14.pdf

Verified against that PDF (Eqs. (1)–(21); Lemma 1; Theorems 1; footnotes; §2.3; §4.6; Table VII). Dedup note: this is the journal version of the UAI 2010 conference paper (“An abridged version of this paper was published in UAI 2010”, footnote 1) and is a **different paper** from the carded Zhang & Yeung AISTATS 2010 generalized-t process card (`zhang-yeung-2010-mtgtp.md`). Its *symmetric* relation object is the project’s existing in-category control.

## Problem

Learn the task relationship matrix jointly with the task parameters, under a convex objective, while allowing positive correlation, negative correlation and outlier (unrelated) tasks — and also support the *asymmetric* MTL setting in which a new target task arrives after source tasks have been learned.

## Mathematical assumption

A matrix-variate normal prior on the parameter matrix, `q(W) = MN_{d×m}(0, I_d ⊗ Ω)`, so `Ω` (the column covariance) models task relations. The MAP/ML problem is

`min_{W,b,Ω⪰0} Σ_i (1/n_i)Σ_j (y_j^i − w_iᵀx_j^i − b_i)²/2 + λ₁tr(WWᵀ)/2 + λ₂tr(WΩ⁻¹Wᵀ)/2`, `tr(Ω) ≤ 1`,

jointly convex; this is exactly the classical symmetric MTRL relation object used as the project’s control.

## Relation representation

A **symmetric PSD task covariance** `Ω` (plus, in the asymmetric *setting*, an augmented `Ω̃ = [[(1−σ)Ω, ω],[ωᵀ, σ]]`). The directed-family question is whether the asymmetric extension introduces a directed relation object: it does not. Following Xue et al. (2007), the paper defines “asymmetric multi-task learning” as the *learning setting* in which “the objective is to improve the performance of some target task using information from the source tasks”, and its §2.3/§4.6 instantiate this as: learn sources symmetrically, then incorporate one new target task by learning the covariance *vector* `ω` between the new task and the existing ones (`ωᵀΩ⁻¹ω ≤ σ − σ²`, Eq. (16)). `Ω̃` remains symmetric by construction; `ω` is a covariance sub-vector, not a directional transfer weight.

**Consequence for family A.** There is no directed relation object anywhere in the paper: `Ω` and `Ω̃` are symmetric PSD matrices throughout. Lee et al. (ICML 2016) make the same observation when distinguishing their learned asymmetry from “asymmetry” that “simply means either having a pre-defined set of main and auxiliary tasks ... or training models for new tasks along with the existing models (Zhang & Yeung 2010)”.

## Optimization method

Alternating optimization over convex subproblems: given `Ω`, solve a dual quadratic problem with a multi-task kernel defined by `Ω` using SMO (Appendix A); given `W`, the relation subproblem has the analytic solution `Ω = (WᵀW)^{1/2}/tr((WᵀW)^{1/2})` (Eq. (14)). For the new-task case the augmented problem is an SDP, reformulated as an SOCP for larger `m` (Appendix B); the new task’s parameters are pulled toward `u = W_m Ω⁻¹ ω`, a covariance-weighted combination of source parameters.

## Evidence

Toy problem plus benchmark datasets (multi-domain sentiment and others), reporting task-covariance interpretability (positive/negative correlations and outlier detection). §4.6 evaluates the *asymmetric setting* on multi-domain sentiment data at 10 %, 30 % and 50 % training sizes against STL and DP-MTL (Xue et al. 2007), which is the only baseline in that table. All tasks are regression/binary with a shared feature space; no unequal-width outputs are considered.

## Assumptions

Shared input space and shared feature mapping (kernel); one real-valued or binary output per task with a per-task weight vector of equal length `d`; a single symmetric task covariance summarises all relations; the asymmetric setting means target/source tasks with a sequential arrival, not directed transfer.

## Differences from our setting

* **Family fit.** The relation object is symmetric; nothing here learns or parameterises a directed/asymmetric task-relation structure. It is, however, precisely the classical MTRL formulation already implemented as the in-category control, including its mean-head-summary device for unequal head widths.
* **Deep/heterogeneous.** The method is a shallow kernel/linear model; porting it to our trunk and heads is what the control already does. There is no mechanism to evaluate under family A.
* **Purpose of recording it.** A literature search for “asymmetric MTRL” will surface this paper first, and its “asymmetric” keyword invites exactly the attribution error the project treats as a known failure mode (an architecture was archived earlier for a wrong citation). This card records the finding that the keyword does not denote a directed relation.

## Implementation difficulty

Not applicable as a family-A arm — its symmetric part is already implemented as the control; adding the asymmetric-setting machinery would mean re-learning a covariance vector for a target task, which is a different experiment (target-task incorporation), not a directed-relation variant.

## Candidate Study ID

`LT-0002` — screened candidate (family A), recorded as a keyword/attribution failure.

## LT-0002 assessment

* **Gate 1 — explicit relation object: FAIL.** The relation object is a symmetric covariance `Ω` (or a symmetrically augmented `Ω̃`); “asymmetric” denotes the target/source learning setting, not a directed relation. No directed transfer graph, no asymmetric precision/covariance parameterisation, and no learned direction.
* **Gate 2 — taxonomy: PASS.** It is canonical §2.4 Task Relation Learning (matrix-variate prior, `tr(WΩ⁻¹Wᵀ)`).
* **Gate 3 — heterogeneous heads: PASS (as the control).** Already solved in this project by mean-head parameter summaries; it is the matched in-category control for every variant arm.
* **Gate 4 — fixed representation: PASS.** Applies to downstream parameters over frozen embeddings (the implemented control does exactly this).
* **Gate 5 — faithful implementability: N/A for family A.** Implementable (and implemented) as the control, but it is not a directed-relation method, so it cannot serve as a family-A variant.
* **Gate 6 — source verified: PASS.** Venue, volume, article number, DOI, equations and the footnoted UAI-2010 lineage all checked against the author copy and the ACM record.

**Verdict: `FAIL — the relation object (Ω, and the augmented Ω̃ of the “asymmetric setting”) is symmetric by construction; the paper supplies a target/source learning setting, not a directed relation-learning mechanism.`** Its symmetric half is already the project’s in-category control, so it is retained as the matched control rather than as a variant.
