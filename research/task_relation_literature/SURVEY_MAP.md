# SURVEY_MAP — Zhang & Yang (2021) mapped onto the wavCSE task-relation problem

Status: **literature/theory only.** This file contains no code, configs, experiments or
registered studies. It does not authorise any study and does not select a winner.

Scope of this worktree: `research/task_relation_literature/`. Companion documents in the same
directory: `PRIMARY_LITERATURE.md`, `TASK_RELATION_TAXONOMY.md`, `THREE_TASK_ANALYSIS.md`,
`MSSL_SPARSITY_ANALYSIS.md`, `ASYMMETRIC_RELATIONS.md`, `CANDIDATES.md`,
`SUCCESSOR_STUDIES.md`, `OPEN_QUESTIONS.md`.

Evidence wording used throughout:

* **OBSERVED (survey)** — read in the survey text itself.
* **OBSERVED (primary)** — read in the primary paper (equation/section given).
* **PRIMARY_SOURCE_NOT_VERIFIED** — asserted here only at survey granularity; do not promote.
* **INFERRED** / **HYPOTHESIZED** — never stated as fact.

The binding benchmark conditions for anything downstream are
`improvements/taskrelation/research/VARIANT_BENCHMARK_PROTOCOL.md`; the binding scope decisions
are `improvements/taskrelation/research/DECISIONS.md` (DEC-0005 §3, DEC-0013 §5) and the binding
findings are `improvements/taskrelation/research/FINDINGS.md` + `FAILURES.md`. This file never
overrides them.

---

## 1. Source

* Yu Zhang and Qiang Yang, *A Survey on Multi-Task Learning*, arXiv:1707.08114v3
  (arXiv:1707.08114v3 [cs.LG] 29 Mar 2021); published version IEEE TKDE.
  Read directly from https://arxiv.org/pdf/1707.08114 (v3), sections §1–§7 and the reference
  list, on 2026-09-29. **OBSERVED (survey).**
* The survey is used as a **map**, not as evidence for any algorithmic claim
  (mandate: "Treat it as a MAP of the literature, not as sufficient evidence"). Every
  algorithmic claim below is either read in the survey with its section/equation, or verified in
  the primary paper (see §10). Where a primary source could not be opened it is labelled
  `PRIMARY_SOURCE_NOT_VERIFIED`.

---

## 2. The survey's own taxonomy (what it considers a distinct mechanism)

The survey classifies supervised MTL into **five approaches** (abstract; §2): **feature
learning** (→ §2.1, incl. feature transformation and feature selection), **low-rank** (§2.2),
**task clustering** (§2.3), **task relation learning** (§2.4), **decomposition** (§2.5).
**OBSERVED (survey §2, abstract).**

§2.8 gives a second, finer taxonomy for *regularized* MTL, splitting it into exactly two
families by which covariance sits in the regularizer. **OBSERVED (survey §2.8):**

* **learning with feature covariance** — `min L(W,b) + tr(Wᵀ Σ⁻¹ W) + f(Σ)` (Eq. 32); `Σ` is
  the **row** covariance, i.e. the *feature* covariance shared across tasks.
* **learning with task relations** — `min L(W,b) + tr(W Ω⁻¹ Wᵀ) + g(Ω)` (Eq. 33); `Ω` is the
  **column** covariance, i.e. the *task* covariance.

This is the survey's own statement that "feature covariance" (Eq. 32) and "task relation"
(Eq. 33) are different objects even though the two regularizers look similar. For this project
the second is the relevant family; the first is the multi-task-feature-learning line.

**The category-defining object is the mathematical relation object, not the presence of
task-dependent scalars or gradients.** §2.6 says the task-relation approach "can learn model
parameters and pairwise task relations simultaneously … The learned task relations can give us
insights about the relations between tasks and hence they improve the interpretability."
**OBSERVED (survey §2.6).**

---

## 3. §2.4 Task Relation Learning — the binding category test

Survey definition: "task relatedness can be quantitated via task similarity, task correlation,
task covariance and so on. Here we use task relations to include all the quantitative
relatedness." **OBSERVED (survey §2.4).** §2.4 is the *only* location in the survey where
covariance/precision/graph relation objects are the learned object; this is the category the
project is bound to (DEC-0005 §3; DEC-0013 §5; FL-0003).

### 3.1 What §2.4 contains, in the survey's own words

| Survey anchor | Object in the survey's words | Primary ref (survey numbering) |
|---|---|---|
| §2.4, a priori | known task similarities used to regularize (mean-parameter shrink, pairwise similarity, tree) | [69],[70] Evgeniou & Pontil / Evgeniou et al.; [71],[72]; [75] Görnitz et al. |
| §2.4, MTGP | learned `M×M` free-form task covariance `K^f`, PSD; `⟨f_i f_p⟩ = ω_ip k(x,x')` | [76] Bonilla, Chai & Williams 2007 |
| §2.4, MTGTP | inverse-Wishart prior on the task covariance (Bayesian uncertainty on the relation) | [78] Zhang & Yeung, AISTATS 2010 |
| §2.4, MTRL | matrix-variate normal prior `W ~ MN(0, I, Ω)`; Eq. (21) `min L(W,b) + λ₁‖W‖²_F + λ₂ tr(W Ω⁻¹ Wᵀ)` s.t. `Ω ⪰ 0, tr(Ω) = 1` | [79],[80] Zhang & Yeung, UAI 2010 / TKDD 2014 |
| §2.4, sparse relation | `ℓ₁` on `Ω` "when the number of tasks is large" | [87] Zhang & Yang, AAAI 2017 (SPATS) |
| §2.4, sparse matrix-normal | `W ~ MN(0, ·, ·)` with sparse `Σ₁`, `Σ₂` (i.e. sparse covariances) | [88] Zhang & Schneider, NIPS 2010 |
| §2.4, block sparse | matrix-variate generalized hyperbolic prior (block-sparse `W`) | [89] Archambeau et al. |
| §2.4, low-rank relation | matrix generalized inverse Gaussian prior (low-rank covariances) | [90] Yang, Li & Zhang |
| §2.4, high-order | `(WᵀW)^t ~ W(0, Ω)`, Eq. (22), "high-order task relationships" | [91] Zhang & Yeung, IJCAI 2013 |
| §2.4, deep tensor | tensor-variate normal prior on the fully-connected parameter tensor | [95] Long et al. (MRN) |
| §2.4, local kNN | `m×m` contribution matrix `Ω` where `Ω_ip` is "the similarity from `T_p` to `T_i`"; Eq. (23)–(24); regularizer "enforces `Ω` to be nearly symmetric" | [96] Zhang, NIPS 2013 |
| §2.4, asymmetric | `W ≈ WA` with `A` "asymmetric task relations"; Eq. (25); sparse + loss-scaled rows `(1 + λ₁‖â_i‖₁)` | [97] Lee, Yang & Hwang, ICML 2016 (AMTL) |

The survey explicitly states that the relations learned by the methods above are symmetric
"except [96]" (Zhang 2013), and that even Lee's `A` is regularizer-symmetric (§3.2).
**OBSERVED (survey §2.4).**

### 3.2 The survey's own caveat on the only "asymmetric" §2.4 object (critical)

Immediately after Eq. (25), the survey writes (OBSERVED, §2.4):

> "Though `A` is asymmetric, from the perspective of the regularizer, the task relations here
> are symmetric and act as the task precision matrix with a restrictive form."

The survey rewrites Lee's regularizer as `‖W − WA‖²_F = tr(W(I−A)(I−A)ᵀWᵀ)`, a special case of
Eq. (21) with `Ω⁻¹ = (I−A)(I−A)ᵀ`. **Consequence:** the single directed `A` in §2.4 is
*symmetrized* by the regularizer; the learned object used to couple parameters is `Ω⁻¹`, which
is symmetric PSD. This is a survey-level statement, not a claim about whether AMTL's own
`A` is "directional" — the primary paper keeps a directed graph and a row-wise loss scaling
(see `ASYMMETRIC_RELATIONS.md` and the card
`improvements/taskrelation/research/literature/lee-2016-asymmetric-mtl.md`), but the survey
classifies its *regularizer* as a precision form. Both readings are recorded; neither is
promoted to an improvement claim.

### 3.3 What §2.4 is not

§2.4 does not contain: low-rank parameter tensors (§2.2), task clustering/grouping (§2.3),
parameter decomposition (§2.5), gradient manipulation or loss weighting (§2.10), or
representation sharing (§2.1). The survey physically places them elsewhere. **OBSERVED
(survey §2–§2.5, §2.10).**

---

## 4. §2.9 statement on multi-class / unequal heads — the alignment fact

This is the survey's own statement of the dimensional obstacle that decides several
faithfulness verdicts here. **OBSERVED (survey §2.9):**

> "In many MTL classification problems, each task is explicitly or implicitly assumed to be a
> binary classification problem … It is not difficult to see that many methods in the feature
> learning approach, low-rank approach and decomposition approach can be directly extended to a
> general setting where each classification task can be a multi-class classification problem …
> **However, to the best of our knowledge, there is no theoretical or empirical study to
> investigate such direct extension.** For most methods in the task clustering and task
> relation learning approaches, **such direct extension does not work** since for multiple
> columns in `W` corresponding to one task, we do not know which one(s) can be used to represent
> this task."

**Mapping to wavCSE.** The three tasks have classifier widths 12 (KS), 1251 (SI) and 4 (ER) —
unequal, multi-class heads, not one column of `W` per task. Under §2.9 the covariance /
clustering / relation approaches are exactly the classes for which the one-column-per-task
convention is load-bearing. Any port therefore has to say *what plays the role of a task
column*. The project's answer is the class-mean head summary already used by the in-category
control (`MTRL_DIAGNOSTIC_SYNTHESIS.md` §"What classical MTRL assumes", item 4); the
consequences are quantified in `THREE_TASK_ANALYSIS.md` and `ALIGNMENT` in
`TASK_RELATION_TAXONOMY.md`. Survey §2.9 is the primary citable support for calling this a
real, stated limitation rather than a project quirk.

---

## 5. §2.10 Optimization Techniques — where gradient/weighting methods actually live

The survey puts **GradNorm, gradient surgery and multi-objective MTL in §2.10 "Optimization
Techniques in MTL"**, under "Gradient descent method and its variants", not in §2.4.
**OBSERVED (survey §2.10).** This is the survey-level basis for the project's rule that
gradient surgery and generic loss weighting are out of the Task Relation Learning category
(DEC-0005 §3; FL-0003; card `chen-et-al-2018-gradnorm.md`).

**Citation-numbering caution (OBSERVED).** The §2.10 sentence reads "the GradNorm [115] is
devised to normalize gradients … [116] proposes the gradient surgery … [117] studies MTL from
the perspective of multi-objective optimization", but the reference list assigns
**[115] = Sener & Koltun (multi-objective), [116] = Yu et al. (PCGrad / gradient surgery),
[117] = Chen et al. (GradNorm)**. The survey's in-text attributions appear shifted. Do not cite
survey numbering for these three; cite the primaries directly (verified in §10):
PCGrad = Yu et al., NeurIPS 2020, Algorithm 1; GradNorm = Chen et al., ICML 2018; CAGrad =
Liu et al., NeurIPS 2021, Alg. 1 / Eq. (3).

---

## 6. Mapping table — survey taxonomy → the mandate's families A–L

The mandate enumerates twelve candidate families A–L. They are **not** the survey's five
approaches; several A–L families cut across survey categories, and some are survey-external
(deep routing, bilevel/meta). The table states, per family, what the survey does with it, the
relation object, and the wavCSE status vocabulary defined in `TASK_RELATION_TAXONOMY.md`
(`IN_SCOPE_DIRECTLY` / `IN_SCOPE_WITH_JUSTIFIED_ADAPTATION` / `OUT_OF_SCOPE`).

| Family | Survey home | Relation object (what is learned) | Survey/primary anchor | wavCSE status |
|---|---|---|---|---|
| **A. covariance / precision task-relation learning** | §2.4 (+§2.8 Eq. 33) | task covariance `Ω` (PSD, trace-1 in MTRL) or precision `P` (positive definite in MSSL) | survey §2.4 Eq. (20)–(21); MTRL Eq. (7), §2.2; MSSL Eq. (3),(8) | `IN_SCOPE_WITH_JUSTIFIED_ADAPTATION` for unequal multiclass heads; direct category, adapted attachment |
| **B. sparse task-relation learning** | §2.4 | sparse covariance `‖Ω‖₁` (SPATS) **or** sparse precision `‖P‖₁` (MSSL) | SPATS Eq. (3), Thm 1; MSSL Eq. (3)/(8) | both require summary adapter here; MSSL is **already registered** TR-0007 |
| **C. task clustering / grouped MTL** | §2.3 | partition / soft assignment `Z`, cluster indicator `M=E(EᵀE)⁻¹Eᵀ`, or representative-task weights | CMTL Eq. (5)–(16); FCMTL Eq. (4); GO-MTL Eq. (1) | `OUT_OF_SCOPE` (clustering category; DEC-0013 §5) |
| **D. graph-structured task relations** | §2.4 (a-priori similarity) + §3 (graphical models) | graph Laplacian `L` as coupling `Ω⁻¹=L` (formally, singularity requires care), or learned adjacency `A`, or joint precision of GGMs | MTRL §3 Laplacian special case; GAMTL (Yu 2020) adjacency `A`; Honorio & Samaras [172]; Lin et al. [174] | learned undirected graph requires adaptation; a-priori graph is not task-*relation learning* |
| **E. asymmetric / directed task relations** | §2.4 (only [97]) | directed matrix `A` / `B`; survey converts to precision `(I−A)(I−A)ᵀ` | survey §2.4 Eq. (25); AMTL Eq. (1)–(3); AutoTR; Oliveira GAMTL | published forms `IN_SCOPE_WITH_JUSTIFIED_ADAPTATION` only; DEC-0014 arm is project-original |
| **F. latent task-relation models** | §2.3/§2.4 hybrid | latent bases `L` + sparse coefficients `S`; latent task model | GO-MTL `W=LS` Eq. (1); Passos et al.; NIPS'05 latent ICA [43] | `OUT_OF_SCOPE` (latent/low-rank/decomposition boundary) |
| **G. low-rank task structures** | §2.2 | shared subspace `Θ`, `W=U+ΘᵀV`; trace/capped-trace norm | survey §2.2 Eq. (9)–(14); Argyriou MTFL §2.1 Eq. (2) | `OUT_OF_SCOPE` (low-rank category; DEC-0013 §5) |
| **H. task-specific parameter sharing** | §2.1 (cross-stitch), §2.3 (overlap), §2.9 (tensor slice) | per-task mixing coefficients / soft subspace membership | cross-stitch Eq. (1)–(3); GO-MTL overlap; Yang & Hospedales tensor [67] | `OUT_OF_SCOPE` as a *relation* object; cross-stitch is feature routing |
| **I. learned weighting that represents relations** | §2.10 (as weighting), §2.3 (as grouping), §2.8 ([108]) | per-task scalars `w_i`; **no** pair object unless the weight is a covariance diagonal | GradNorm (Chen 2018, no pair object); Sener & Koltun [115]; "Learning to Multitask" [108] | `OUT_OF_SCOPE` — weights are loss/optimization quantities, not pair relations |
| **J. gradient-based task-relation mechanisms** | §2.10 | gradient cosine / projection / conflict-averse update direction; (TAG: affinity from losses) | PCGrad Alg. 1; CAGrad Alg. 1/Eq. (3); TAG Eq. (1) | `OUT_OF_SCOPE` as a regularizer; TAG endpoint is grouping; diagnostics only |
| **K. task routing / selective sharing** | §2.1 (cross-stitch, §2.1 deep sub-categories), §2.9 (routing nets) | activations routing / per-channel mixing; discrete paths | cross-stitch Eq. (1); routing networks (Rosenbaum et al., cited in CAGrad/PCGrad); Soft Modularization | `OUT_OF_SCOPE` (representation/architecture, not a learned pair relation) unless a param-routing form is defined |
| **L. bilevel / meta-learned task relations** | §2.8 ([108] "Learning to Multitask") | outer objective over a task-coupling parameter (weights, grouping, or relation) | survey §2.8 [108]; TAG's grouping-selection step over the affinity matrix | `OUT_OF_SCOPE` as published (meta/bilevel objective); a bilevel *estimator for Ω* would be project-original. (AutoTR's `R` update is an alternating/closed-form step, not bilevel — it belongs to family E.) |

Two survey-explicit boundary statements worth quoting for families C and I:
"the task clustering approach can capture positive correlations among tasks in the same cluster
but ignore negative correlations among tasks in different clusters" (§2.6, OBSERVED); and the
§2.8/§2.10 placement of weighting methods outside §2.4.

---

## 7. The wavCSE problem this map is for (fixed context)

Binding context, cited so the mapping is not read in the abstract
(`VARIANT_BENCHMARK_PROTOCOL.md`; `FRAMEWORK.md` §1; `MTRL_DIAGNOSTIC_SYNTHESIS.md`):

* Three tasks, disjoint data: KS (12-class), SI (1251-class), ER (4-class); `m = 3`.
* Frozen upstream: WavLM-large, frame pooling `mean`, layer pooling `smp` 0.5 over all 25 layers.
* Shared trunk 1024→512→2000; unchanged across arms; only the relation mechanism may vary.
* Classical symmetric dense MTRL is the in-category control; the matched wavCSE baseline is the
  reference (F4: nothing has displaced it).
* Measured relational facts already on record (OBSERVED in-repo, not re-derived here):
  Ω saturates to ±1/3 at `smp` 25L and `lnp` 16L (F5/F7); raw ER-directed asymmetry did not
  survive optimizer-exposure control (F8/DG-0001); ER gradient-norm dominance is a
  training-mixture property (F10/DG-0005); no persistent pairwise gradient conflict (F9/DG-0002).
* m=3 geometry: 3 unique symmetric off-diagonal relations vs 6 possible directed ones; a
  trace-1 `3×3` covariance has 5 free parameters before regularisation. Detail and consequences
  belong to `THREE_TASK_ANALYSIS.md`; this file only fixes the mapping.

---

## 8. Category boundaries the map must preserve

The mandate says "Do not force every family into our parameter-based MTL constraint. Reject
unsuitable families explicitly." The following are `OUT_OF_SCOPE` and are recorded as such, not
silently admitted (DEC-0005 §3; DEC-0013 §5; FL-0003):

* **Loss/uncertainty weighting** (Kendall et al. 2018; GradNorm, Chen et al. 2018) — no pair
  relation object. §2.10.
* **Gradient surgery / conflict-averse updates** (PCGrad, CAGrad) — model-agnostic update
  operators on the shared gradient; no parameter-space relation. §2.10.
* **Multi-objective MTL** (Sener & Koltun) — Pareto/weight optimisation. §2.10.
* **Low-rank** (survey §2.2) and **decomposition** `W = Σ W_k` (§2.5) — different parameter
  regularizers; the decomposition boundary in particular bars the `W=H+P` "informative
  relations" paper (`chang-et-al-2024-informative-relations.md`).
* **Clustering / grouping** (survey §2.3; TAG's endpoint) — a partition or discrete grouping,
  not a coupling matrix used in training.
* **Feature/representation sharing** (survey §2.1) — cross-stitch and routing belong here.

A family may be `IN_SCOPE_WITH_JUSTIFIED_ADAPTATION` only if the relation object and update rule
are published and the adaptation is declared (Vocabulary in `TASK_RELATION_TAXONOMY.md`).

---

## 9. What this map does **not** license

* It does not claim any family improves KS/SI/ER. No such measurement exists; the matched
  wavCSE baseline remains the champion (F4). Every cell above is a *hypothesis of mechanism*,
  not a result.
* It does not decide a winner, rank arms, or authorise a Study (mandate; DEC-0013).
* It does not re-open the F9 literature search (DEC-0011) — §2.4 is mapped to the
  variant-benchmark premise of DEC-0013, not to the withdrawn scale rationale (DEC-0010).
* Survey-level claims are not promoted; the §3 table marks which are primary-verified in §10.

---

## 10. Primary verification ledger for this file

Each row was read directly (arXiv/venue PDF) on 2026-09-29; equation/section is given. Anything
not listed is at survey granularity only.

| Primary | Verified detail | Location |
|---|---|---|
| Zhang & Yang 2021 (survey) | five categories; §2.8 Eq. (32)/(33); §2.4 Eq. (20)–(25) + the "symmetric … task precision matrix" caveat; §2.9 multi-class statement; §2.10 placement + numbering anomaly | arXiv:1707.08114v3 §2, §2.8, §2.4, §2.9, §2.10 |
| Zhang & Yeung 2010 (MTRL) | Eq. (7) `min Σ(1/n_i)ℓ + λ₁/2 tr(WWᵀ) + λ₂/2 tr(WΩ⁻¹Wᵀ)` s.t. `Ω⪰0, tr(Ω)=1`; Ω update `Ω=(WᵀW)^{1/2}/tr((WᵀW)^{1/2})`; §3 relates fixed graph-Laplacian coupling to `Ω⁻¹=L` formally (L is singular, so note pseudoinverse caveat); §2.3 asymmetric = new-target task incorporation | UAI 2010, §2.1 Eq. (1)–(7), §2.2, §2.3, §3 |
| Jacob, Bach & Vert 2008 (CMTL) | penalty Eq. (5); `Σ(M)⁻¹` Eq. (7); convex relaxation Eq. (15)–(16); §3.2 "convex relaxation of K-means" | arXiv:0809.2085 §2–§3 |
| Kumar & Daumé III 2012 (GO-MTL) | `W = LS`, objective Eq. (1); alternating `s_t` (Eq. 2) / `L` (Eq. 3); overlap via shared bases | ICML 2012, §3 Eq. (1)–(3) |
| Zhou & Zhao 2016 (FCMTL) | representative-task assignment `Z`, objective Eq. (3)/(4); row-sparsity `‖Z‖₁,₂`; Thm 1 self/common-representative regimes | TPAMI 38(2):266–278, §3.1–§3.3, Thm 1 |
| Zhang & Yang 2017 (SPATS) | framework Eq. (1); **sparse covariance** `+ λ₂‖Ω‖₁` Eq. (3) (contrast: MSSL penalizes its directly parameterized **precision** `P`, not a covariance); Thm 1 zero `ω_ij` ⇒ params not spanned by the other task's data; Thm 2 eigenvalue bound; §Experiments notes sparse relations suit **many** tasks (Sentiment, m=4, sparse loses) | AAAI 2017, Eq. (3), Thm 1–2, experiments |
| Lee, Yang & Hwang 2016 (AMTL) | independently verified in [PRIMARY_LITERATURE.md](PRIMARY_LITERATURE.md): directed sparse elementwise non-negative `B_st≥0`, loss-scaled outgoing rows, `B_tt=0`; Theorem 1 has a reconstruction-cost alternative | [ICML 2016 paper](https://proceedings.mlr.press/v48/leeb16.pdf) §3 Eqs. (1)–(3), Theorem 1 |
| Yu et al. 2020 (PCGrad) | Alg. 1 projection `g_i ← g_i − (g_i·g_j)/‖g_j‖² g_j` iff `cos<0`; Def. 1 conflict `cos<0`; model-agnostic, shared params only; no relation object | arXiv:2001.06782 §2.2–§2.3 |
| Liu et al. 2021 (CAGrad) | Eq. (3) `max_d min_i⟨g_i,d⟩` s.t. `‖d−g₀‖≤c‖g₀‖`; dual over simplex `w∈Δ^K`; Alg. 1; `c→0` recovers GD; no relation object | arXiv:2110.14048 §3.1 Eq. (3), Alg. 1 |
| Misra et al. 2016 (cross-stitch) | Eq. (1) `[x̃_A;x̃_B] = [[α_AA,α_AB],[α_BA,α_BB]][x_A;x_B]`; Eq. (2)/(3) gradients; two-task construction; `α_AB≠α_BA` allowed (not constrained symmetric) | arXiv:1604.03539 §3.3 Eq. (1)–(3) |
| Bonilla, Chai & Williams 2007 (MTGP) | free-form PSD `K^f=LLᵀ`; `⟨f_l f_k⟩=K^f_{lk}k_x`; Gaussian likelihood; EM closed form `K̂^f=(1/N)⟨FᵀK^x⁻¹F⟩` | card `bonilla-2007-mtgp.md` (NIPS 2007, Eq. 1–5) — **PRIMARY_SOURCE_NOT_VERIFIED in this pass** (proceedings PDF URL 404; card's own check stands) |
| Zhang 2013 local kNN relation | Eq. (23)–(24); `Ω_ip` = contribution `T_p → T_i`; regularizer "nearly symmetric" | survey §2.4 (survey-level only) |
| Zhou & Yang 2023 (AutoTR) | closed-form signed directed matrix `R` | card `zhou-2023-autotr.md` (KDD 2023) — not re-opened in this pass |

Gaps, stated explicitly: **Bonilla 2007 (MTGP)** was not re-opened as a PDF in this
pass and remains at repository-card/survey granularity. Lee 2016 AMTL was read
directly for the companion primary ledger after this survey map's initial pass.
Where an attribution differs, the primary paper governs.

---

## 11. Cross-links

* Taxonomy vocabularies and dimensions: [`TASK_RELATION_TAXONOMY.md`](./TASK_RELATION_TAXONOMY.md).
* m=3 structural consequences: [`THREE_TASK_ANALYSIS.md`](./THREE_TASK_ANALYSIS.md).
* Sparse-covariance vs sparse-precision: [`MSSL_SPARSITY_ANALYSIS.md`](./MSSL_SPARSITY_ANALYSIS.md).
* Directed methods in detail: [`ASYMMETRIC_RELATIONS.md`](./ASYMMETRIC_RELATIONS.md).
* Comparison matrix and shortlist: [`CANDIDATES.md`](./CANDIDATES.md).
* Primary-source ledger: [`PRIMARY_LITERATURE.md`](./PRIMARY_LITERATURE.md).
* Open questions: [`OPEN_QUESTIONS.md`](./OPEN_QUESTIONS.md).

Repository authorities: `improvements/taskrelation/research/{OBJECTIVE,FRAMEWORK,FINDINGS,FAILURES,DECISIONS,BACKLOG,STATE}.md`,
`.../VARIANT_BENCHMARK_PROTOCOL.md`, `.../literature/INDEX.md`,
`.../task_relations/MTRL_DIAGNOSTIC_SYNTHESIS.md`.
