# TASK_RELATION_TAXONOMY — what "task relation" means across the candidate families

Status: **literature/theory only.** No code, configs, experiments or registered studies; no
winner selection; no claim that any family improves KS/SI/ER.

Companion documents (same directory): [`SURVEY_MAP.md`](./SURVEY_MAP.md) (survey → families A–L),
[`PRIMARY_LITERATURE.md`](./PRIMARY_LITERATURE.md), [`THREE_TASK_ANALYSIS.md`](./THREE_TASK_ANALYSIS.md),
[`MSSL_SPARSITY_ANALYSIS.md`](./MSSL_SPARSITY_ANALYSIS.md),
[`ASYMMETRIC_RELATIONS.md`](./ASYMMETRIC_RELATIONS.md), [`CANDIDATES.md`](./CANDIDATES.md),
[`SUCCESSOR_STUDIES.md`](./SUCCESSOR_STUDIES.md), [`OPEN_QUESTIONS.md`](./OPEN_QUESTIONS.md).

Wording: **OBSERVED** = read in the cited source; **INFERRED** /**HYPOTHESIZED** = flagged; a
claim carrying `PRIMARY_SOURCE_NOT_VERIFIED` is at card/survey granularity only and must not be
promoted.

---

## 0. Vocabulary (used consistently below and in `CANDIDATES.md`)

**Faithfulness classes** — how a candidate relates to a *published* method, as required by
DEC-0013 §3 and the LT-0002 gates
(`improvements/taskrelation/research/studies/LT-0002/PLAN.md` §"Eligibility gates"):

* `IN_SCOPE_DIRECTLY` — explicit Task Relation Learning object *and* the published update rule
  is implementable against our model (possibly via the shared class-mean-summary adapter that
  the in-category control already uses). Example: MSSL on the summary matrix (TR-0007).
* `IN_SCOPE_WITH_JUSTIFIED_ADAPTATION` — explicit relation object, but a *declared* deviation
  (adapter, likelihood, or invented structure) is needed. Under the LT-0002 falsification rule a
  documented deviation counts as a faithfulness **fail**; these arms may only run if the human
  admits deviation-class arms, and must never be labelled a faithful reproduction.
* `PROJECT_ORIGINAL` — the mechanism is ours, motivated by literature but not a published
  method (DEC-0014 labelled the directed arm this way).
* `OUT_OF_SCOPE` — category boundary (low-rank, clustering, decomposition, loss weighting,
  gradient surgery, representation sharing). DEC-0005 §3; DEC-0013 §5; FL-0003.

**Relation vs optimization quantity** — `FRAMEWORK.md` §2 rule 1: a non-relational quantity
(loss/gradient scale, gradient variance, per-task weighting, sampling composition) cannot
justify a relation mechanism, and a relation mechanism cannot be smuggled in from a scale
method.

---

## Category test (Zhang and Yang 2.4)

A candidate is in the project's category iff it learns or uses an **explicit task relation
object** estimated from data — a covariance, precision, similarity/adjacency, directed transfer
matrix, or a learned estimator of one of these — and the object is the thing that couples the
tasks. Survey §2.4 defines task relations as "all the quantitative relatedness" (task
similarity, correlation, covariance, …). **OBSERVED (survey §2.4).**

The test is deliberately about the **object**, not about the *setting*:

* Merely mentioning "tasks help each other", or having task-dependent scalars, or operating on
  gradients, does **not** create a relation object.
* The survey's own negative examples: Lee's directed `A` yields a *regularizer* that is
  symmetric — "the task relations here are symmetric and act as the task precision matrix with a
  restrictive form" (§2.4, OBSERVED); Zhang & Yeung's "asymmetric MTRL" is symmetric and its
  asymmetry is the target/source *setting* (MTRL UAI 2010 §1, §2.3, OBSERVED);
  GradNorm/PCGrad/CAGrad/Sener have no pair object and sit in §2.10 (OBSERVED).
* A method that learns an explicit relation object but whose *published endpoint* is a discrete
  grouping (TAG) fails the category test on endpoint grounds even though its estimator is
  explicit (`fifty-2021-tag.md`).

Consequences of the §2.9 statement (survey, OBSERVED): for "most methods in the task clustering
and task relation learning approaches" the one-column-per-task convention is load-bearing and
"such direct extension does not work" for multi-class tasks. Unequal heads 12/1251/4 (KS/SI/ER)
therefore make the **alignment / dimensional requirement** dimension (D7) decisive rather than
cosmetic.

---

## Relation-object dimensions

Every candidate is described by these eight dimensions. They are orthogonal: two methods can
share an object type and differ in symmetry, sign or estimation.

### D1. Object type
What is literally stored and updated. Concrete inventory: **covariance** `Ω`; **precision**
`Ω⁻¹`; **adjacency** `A`; **graph Laplacian** `L`; **directed transfer matrix** `B`/`A`/`R`;
**cluster assignment** `Z`/`M=E(EᵀE)⁻¹Eᵀ`; **latent bases + coefficients** `(L,S)`;
**feature-mixing matrix** `α`; **per-task scalar weights** `w_i`. These are *not*
interchangeable — see §"Six encodings compared".

### D2. Symmetry
Whether the *learned object* is constrained to be symmetric: covariance,
precision and Laplacian are symmetric; cluster co-membership is symmetric but
representative-selection `Z` can be asymmetric without measuring directed
transfer. Zhang 2013's kNN relation is regularised toward symmetry (survey
§2.4); AMTL `B`, AutoTR `R` and cross-stitch `α` can be asymmetric.
AMTL's induced quadratic penalty is nevertheless symmetric. Distinguish
object asymmetry from a measured `T(A←B) ≠ T(B←A)`.

### D3. Sign
Whether off-diagonal entries may be negative and what a negative means. Values: **non-negative**
(adjacency entries `A_ij≥0`, AMTL `B_st≥0`, similarity kernels given a priori,
cluster memberships, cross-stitch initialised in [0,1] but not sign-constrained),
**signed-marginal** (covariance `Ω` off-diagonals; negative means anti-correlated
parameters/tasks), **signed-conditional** (precision `P` off-diagonals;
partial-correlation sign is **opposite** `P_ij`), **signed-directed** (AutoTR `R`).
A sign is meaningful only within its object: a negative covariance entry is not evidence of harmful transfer.

### D4. Sparsity
Whether the object is explicitly driven sparse and by which norm. **Sparse covariance**
(SPATS: `+λ₂‖Ω‖₁`), **sparse precision** (MSSL: `ℓ₁` on directly parameterised
precision `P`; Zhang & Schneider: inverse task covariance), **block/structured
sparsity** (Archambeau; clustered MTL's block structure), **dense** (MTGP
free-form `K^f`, MTRL without sparsity), **row-sparse assignment** (FCMTL
`‖Z‖₁,₂`). Zero in SPATS's covariance `Ω` removes a direct contribution
in its shallow representer theorem; zero in MSSL precision `P` is conditional
independence under its Gaussian coefficient model. See
[`MSSL_SPARSITY_ANALYSIS.md`](./MSSL_SPARSITY_ANALYSIS.md).

### D5. Global vs dynamic (and conditioned)
**Global-static** (one object for the whole run: MTGP, MTRL-with-one-update, Laplacian with given
`L`), **dynamic** (object refreshed during training: MTRL per epoch, AMTL `B` alternating, TAG
affinity averaged over training but itself time-varying), **input-conditioned** (relation
depends on the sample: cross-stitch is per-channel and feature-dependent; TAG affinity is
step/batch-dependent), **layer/parameter-block-specific** (cross-stitch per layer; Yang &
Hospedales tensor). The wavCSE protocol pins the relation object to one JSON shape per epoch
(`VARIANT_BENCHMARK_PROTOCOL.md` §4), so "dynamic" means an update schedule, not per-sample.

### D6. Direct vs conditional dependence
Whether the learned zero means "these tasks do not interact" (direct/marginal) or "these tasks do
not interact *given the others*" (conditional). SPATS's sparse **covariance** is the
marginal/direct reading (Thm 1). MSSL's sparse **precision** is the conditional reading: it is
the Gaussian graphical model of the tasks. With three tasks this is a substantive difference —
conditional independence of a pair is a statement about the 2×2 Schur complement, not about
`Ω_ij = 0`. Detail: §"Conditional vs direct dependence".

### D7. Alignment / dimensional requirements
What must line up: **one parameter column per task in a common `W`** (all of §2.4's covariance
and all of §2.3's clustering use `d×m` `W` with one column per task), **comparable per-task
parameter vectors for a distance** (GO-MTL, GAMTL, FCMTL, Jacob), **one shared `d×p` parameter
matrix with a shared feature dimension** (Zhang & Schneider, Zhao 2020 FETR), **aligned
single-output functions over a shared input design** (MTGP, Rakitsch), **parallel per-task
networks of identical shape** (cross-stitch), **only shared-parameter gradients and per-task
losses** (TAG, PCGrad, CAGrad). KS/SI/ER have heads 12/1251/4 on disjoint data, so only the last
class is available verbatim. See §"Alignment and dimensional requirements".

### D8. Estimability with m = 3
Whether the object has enough observations to be estimated non-degenerately at three tasks. This
is the m=3 geometry question (§"m = 3 geometry constraints"; full treatment in
[`THREE_TASK_ANALYSIS.md`](./THREE_TASK_ANALYSIS.md)). It is a *different* axis from D7: an
object can be perfectly aligned and still collapse at m=3 (the observed Ω saturation, F5/F7).

---

## Symmetry and sign

**Symmetric** objects share one number per pair; **directed** objects have two. For `m = 3` this
is 3 vs 6 parameters (all off-diagonal counts). Symmetry is a *representational* property, not a
claim about the world.

* Symmetric by construction: covariance `Ω`, precision `Ω⁻¹`, graph Laplacian `L`, GAMTL
  adjacency (`A_ij = A_ji` stated as a design goal — OBSERVED, primary),
  cluster co-membership (representative assignment itself may be asymmetric).
* "Nearly symmetric": Zhang 2013's local kNN contribution matrix regularises `Ω` toward symmetry
  (survey §2.4, OBSERVED at survey level).
* Symmetrised by the regularizer: AMTL's `A` (survey §2.4 rewrite `Ω⁻¹=(I−A)(I−A)ᵀ` — OBSERVED,
  survey-level; the primary paper keeps a directed non-negative `B` with loss-scaled rows and
  states `A→B ≠ B→A`; both readings recorded, neither promoted).
* Asymmetric-capable: AMTL `B` (`B_st≥0`, `B_tt=0`), AutoTR signed `R` (`r_kk=0`), cross-stitch
  `α` (`α_AB` vs `α_BA` are free).

**Sign semantics must travel with the object.** Examples of the trap this document exists to
prevent:

* A **negative covariance** `Ω_ij<0` means the task parameter vectors are *anti-correlated*; it
  is not by itself "negative transfer", and with `tr(Ω)=1` the PSD constraint bounds it.
* A **negative precision** `P_ij<0` corresponds to a **positive** conditional
  correlation `−P_ij/√(P_ii P_jj)>0` given the third task; it is not `Ω_ij<0`.
* A **non-negative adjacency** cannot express negative correlation at all — which is exactly the
  survey's criticism of clustering ("can capture positive correlations … but ignore negative
  correlations among tasks in different clusters", §2.6, OBSERVED).

OBSERVED in-repo, for context (do not re-derive as new): the MTRL relation object here is
symmetric, its ER-involving entries are conditionally unstable across folds, and the object
saturates (F5/F7; `MTRL_DIAGNOSTIC_SYNTHESIS.md`). These are properties of the learned object,
not evidence about the tasks.

---

## Alignment and dimensional requirements

This is the dimension that decides most faithfulness verdicts for wavCSE, because heads are
12/1251/4 and datasets are disjoint.

| Requirement class | Requires | Examples | Blocked by wavCSE heads? |
|---|---|---|---|
| one column of `W` per task, common `d` | `W=[w_1…w_m]`, one parametrization per task | MTRL Eq. (7); MTGP; SPATS; MSSL; CAGrad's `w∈Δ^K` needs only losses | Not blocked for `m` but blocked for *per-class* columns (§2.9) |
| comparable per-task parameter vectors | `‖w_i − w_j‖` meaningful | GO-MTL; FCMTL; Jacob CMTL; GAMTL | Yes — needs the class-mean summary |
| shared `d×p` matrix + shared feature dim `p` | joint row/column covariance | Zhang & Schneider 2010; Zhao 2020 FETR | Yes — `p×p` feature precision not estimable at `m=3` |
| aligned single-output functions on shared design | one scalar `f_l` per task, common inputs | MTGP; Rakitsch Kronecker GP | Yes — disjoint data, multiclass CE |
| parallel per-task networks of identical shape | duplicated trunk per task | cross-stitch Eq. (1) | Yes — single shared trunk |
| only shared-parameter gradients + per-task losses | no parameter alignment | TAG affinity Eq. (1); PCGrad Alg. 1; CAGrad Eq. (3) | No — available, but endpoint is grouping/update, not a relation regularizer |

The project's adapter is the **class-mean head summary** already used by the in-category control
(12/1251/4 → one mean classifier-weight row + mean bias per task; `normalize_w` unit-normalises
them) — OBSERVED, `MTRL_DIAGNOSTIC_SYNTHESIS.md`. Any covariance/clustering arm that needs
`W` uses this adapter; that is the declared deviation recorded per card, and it is held constant
across arms so it is not a differential confound (LT-0002 analysis §4) — but it is external
invalidity and must be stated.

Two alignment cases that are *not* the same:

* **Aligned by adapter**: the relation still consumes a per-task parameter summary; the adapter
  sits between the heads and the relation.
* **Aligned by mechanism**: the relation never touches parameters (gradients/losses/activations).
  This class is available verbatim in wavCSE, but it is where the *category* boundary bites
  (families J/K).

---

## Conditional vs direct dependence

With `Ω` the task covariance and `S = Ω⁻¹` the precision:

* `Ω_ij = 0` — **marginal / direct** non-interaction: the pair is uncorrelated in the parameter
  matrix. SPATS penalises `‖Ω‖₁`, so its zeros carry this reading (primary Thm 1: task `i`'s
  optimum is not spanned by task `j`'s data).
* `S_ij = 0` — **conditional** independence: tasks `i,j` are independent *given the other
  tasks*. MSSL and Zhang & Schneider penalise `‖Ω⁻¹‖₁` (graphical lasso), so their zeros carry
  this reading.

OBSERVED, primary: MTRL's own analytic update is `Ω = (WᵀW)^{1/2}/tr((WᵀW)^{1/2})` (UAI 2010
§2.2), a scale-fixed covariance; MTRL §2.5 shows a-priori graph-Laplacian regularizers are the
special case `Ω = L`. SPATS (AAAI 2017, primary) explicitly contrasts itself with
`‖Ω⁻¹‖₁` methods. So the same `3×3` object carries two different *meanings* depending on which
side is sparsified — this is the D6 distinction and the reason the two must not be compared as
if they were one mechanism.

**Conditional form with `m=3`.** A zero off-diagonal in a `3×3` precision says a pair is
independent *given the third task*; the implied *marginal* covariance entry is generally
non-zero. Whether that is desirable for KS/SI/ER is untested here and is a question for
[`MSSL_SPARSITY_ANALYSIS.md`](./MSSL_SPARSITY_ANALYSIS.md) and
[`THREE_TASK_ANALYSIS.md`](./THREE_TASK_ANALYSIS.md), not a result of this taxonomy.

---

## Six encodings compared

The mandate requires these to be kept apart. "Relation object" = the thing that couples tasks.

| Encoding | Object | Symmetry | Sign | Sparsity lever | Direct vs conditional | Alignment need | Category |
|---|---|---|---|---|---|---|---|
| **Covariance** | `Ω` (PSD, tr=1) | symmetric | signed (anti-correlation) | `ℓ₁` on `Ω` (SPATS) | direct/marginal | `W` columns | §2.4 in-scope |
| **Precision** | `Ω⁻¹` (PSD) | symmetric | signed (conditional) | `ℓ₁` on `Ω⁻¹` (graphical lasso; MSSL, Zhang & Schneider) | conditional | `W` columns (or summary adapter) | §2.4 in-scope |
| **Graph** | Laplacian `L` with MTRL coupling `Ω⁻¹=L` (where defined), or adjacency `A` | symmetric by construction (GAMTL) | non-negative adjacency | barrier + `ℓ₁` (GAMTL `‖A⊙Z‖₁,₁`) | direct | comparable params for distances | learned graph: adapted; given graph: not a learned relation |
| **Cluster** | assignment `Z`, indicator `M=E(EᵀE)⁻¹Eᵀ` | symmetric (membership) | non-negative | row-sparsity `‖Z‖₁,₂` (FCMTL); spectral set (Jacob) | direct within-cluster only | comparable params / distances | §2.3 — **out of category** |
| **Direction** | `B`/`A`/`R`, `diag=0` | asymmetric-capable | non-negative (AMTL) or signed (AutoTR) | `ℓ₁` rows (AMTL) | directed (one-way) | aligned per-task columns | §2.4 but adapted here |
| **Feature routing** | `α` mixing (per layer/channel) | asymmetric-capable (`α_AB≠α_BA`) | unconstrained | none intrinsic (dense) | direct (feature-level) | parallel identical networks | §2.1 — **out of category** |
| **Weighting** | per-task scalars `w_i` | N/A (no pairs) | non-negative (usually) | N/A | N/A (not a pair relation) | none | §2.10 — **out of category** |

Notes anchored in primaries: covariance/precision contrast — SPATS `‖Ω‖₁` vs
MSSL `‖P‖₁`; graph — MTRL §3 (formal `Ω⁻¹=L`, noting Laplacian singularity)
and GAMTL undirected-by-design; cluster — Jacob Eq. (5)/(7)/(15), GO-MTL
`W=LS` Eq. (1), FCMTL `Z` Eq. (3)/(4); direction — survey §2.4 Eq. (25)
and its symmetrised-regularizer caveat; routing — cross-stitch Eq. (1)–(3);
weighting — GradNorm (no pair object).

---

## Family matrix A–L (object semantics × dimensions × faithfulness class)

Mandate families A–L, described in taxonomy terms. This table is descriptive only; it does not
rank or select.

| Family | Object (D1) | Sym (D2) | Sign (D3) | Sparse (D4) | Global/dyn (D5) | Direct/cond (D6) | Align (D7) | m=3 (D8) | Faithfulness class here |
|---|---|---|---|---|---|---|---|---|---|
| A. covariance / precision | `Ω` or `P` | symmetric | signed | configurable | global; refreshed during training | marginal (Ω) / conditional (P) | `W` columns (adapter) | 5 free params only when trace-one | `IN_SCOPE_WITH_JUSTIFIED_ADAPTATION` for unequal multiclass heads |
| B. sparse task relations | `Ω` (SPATS) or `P` (MSSL) | symmetric | signed | `ℓ₁` on one object | global | marginal vs conditional | as A | only 3 possible off-diagonal zeros | both `IN_SCOPE_WITH_JUSTIFIED_ADAPTATION`; MSSL already TR-0007 |
| C. clustering / grouped | `Z`, `M`, representatives | symmetric | non-negative | row-sparsity | global (or hierarchical) | direct, within-cluster | comparable params | ≤3 binary partitions | `OUT_OF_SCOPE` |
| D. graph | `L` or `A` | symmetric | non-negative adjacency | `ℓ₁`/barrier | global | direct | params or given graph | only 3 edges | given `L`: `OUT_OF_SCOPE` as *learned* relation; learned `A`: adaptation required |
| E. asymmetric / directed | `B`/`A`/`R`, `diag=0` | asymmetric-capable | non-neg. / signed | `ℓ₁` rows | global; alternating | directed one-way | aligned columns | 6 params, weak evidence | published forms adapted (LT-0002 family A); project-original per DEC-0014 |
| F. latent task-relation | `(L,S)` latent bases | via overlap | signed coeffs | `ℓ₁` on `S` | global | indirect (via bases) | `W=LS` | `k≤2` bases | `OUT_OF_SCOPE` (low-rank/decomposition) |
| G. low-rank | shared subspace `Θ` | symmetric | n/a | trace/capped-trace | global | via subspace | `W` columns | rank ≤3 | `OUT_OF_SCOPE` |
| H. task-specific parameter sharing | mixing coeffs / soft membership | asymmetric-capable | unconstrained | none | dynamic (learned) | direct (params) | matched architectures | small | `OUT_OF_SCOPE` as a relation object |
| I. learned weighting that represents relations | per-task scalars | n/a | non-neg. | n/a | dynamic | n/a | none | n/a | `OUT_OF_SCOPE` (no pair object) |
| J. gradient-based mechanisms | gradient field / affinity | N/A | signed cosine | N/A | per-step | n/a | shared grad + losses | 3 tasks → 3 pairs | `OUT_OF_SCOPE` as regularizer; TAG endpoint is grouping |
| K. task routing / selective sharing | `α` / discrete paths | asymmetric-capable | unconstrained | path sparsity | per-layer, dynamic | feature-level | parallel networks | — | `OUT_OF_SCOPE` |
| L. bilevel / meta-learned | outer objective over a coupling parameter | inherits inner object | inherits | inherits | dynamic by construction | inherits | inherits | — | `OUT_OF_SCOPE` as published; a bilevel Ω estimator would be project-original |

---

## m = 3 geometry constraints

Structural facts (arithmetic, INFERRED from the definitions unless a source is given):

* **Pair counts.** Symmetric relation: 3 unique off-diagonal entries (KS–SI, KS–ER, SI–ER). Any
  directed system: 6 ordered entries. A symmetric dense model therefore *cannot* encode
  `T(A←B) ≠ T(B←A)` by construction (this is the category-level statement; whether such
  asymmetry exists here is separate and was weakened by F8/DG-0001).
* **Free parameters.** A `3×3` symmetric matrix has 6 independent entries; MTRL's
  `Ω ⪰ 0`, `tr(Ω)=1` leaves 5. MSSL's precision has 6 before its penalty and
  log-determinant barrier; it does **not** inherit MTRL's trace constraint.
* **Saturation (OBSERVED in-repo, not derived here).** At `smp` 25L and `lnp` 16L the learned Ω
  magnitude saturates to about ±1/3, leaving no pair-specific information (F5, F7; carried in
  `FRAMEWORK.md` §1 rows 3 and `MTRL_DIAGNOSTIC_SYNTHESIS.md`). A saturated relation object is a
  failure mode of *estimation/parameterisation*, distinct from any statement about the tasks
  (R3).
* **Clustering degenerates.** With `m=3` the number of two-cluster partitions is 3 and the
  cluster count `r` is effectively `∈{1,2,3}`; Jacob's cluster norm takes `r` as a
  hyperparameter (the paper reports choosing among 2 and 10 clusters even for 35 MHC-I tasks —
  OBSERVED, primary). A clustering mechanism at `m=3` carries very little structure to identify.
* **Overlap/latent bases degenerate.** GO-MTL needs `k < T` latent bases; with `T=3` the useful
  range is `k∈{1,2}`, so "groups with partial overlap" reduces to a near-trivial pattern.
* **Sparse relations are a many-task strategy.** SPATS (primary) states that with few tasks
  sparse relations can lose to dense ones and reports this on the 4-task Sentiment dataset;
  its good results are on 20–139-task data. This is a primary-source warning against assuming
  sparsity helps at `m=3`; it is *not* evidence that it hurts KS/SI/ER.
* **Directed models have ≥6 parameters fit to a 3-head summary.** INFERRED risk to be evaluated
  in `THREE_TASK_ANALYSIS.md` and [`ASYMMETRIC_RELATIONS.md`](./ASYMMETRIC_RELATIONS.md); no
  measurement exists here.
* **Observations for estimation.** The number of usable "task-pair observations" scales with
  training data and task count, not task count alone; at `m=3` the relation object is estimated
  from the parameter summary matrix and its epochs, which is why refreshes and saturation matter
  (`MTRL_DIAGNOSTIC_SYNTHESIS.md`, OBSERVED).

Full derivation and the m=3 algebraic consequences live in
[`THREE_TASK_ANALYSIS.md`](./THREE_TASK_ANALYSIS.md); the taxonomy only records that D8 is a
first-class dimension and that clustering/latent/directed sparsity all lose discriminating power
at `m=3`.

---

## Boundary exclusion checklist (must be applied before any candidate is recorded)

A candidate is `OUT_OF_SCOPE` and must be recorded as rejected if its contribution is one of
(DEC-0005 §3; DEC-0013 §5; FL-0003):

1. **Loss / uncertainty weighting** with no pair object (Kendall et al. 2018; GradNorm 2018).
2. **Gradient surgery / conflict-averse update** (PCGrad 2020; CAGrad 2021).
3. **Multi-objective optimisation** (Sener & Koltun 2018).
4. **Low-rank** regularisation of `W` (survey §2.2).
5. **Decomposition** `W = Σ W_k` (survey §2.5) — explicitly bars `W = H + P`
   (`chang-et-al-2024-informative-relations.md`).
6. **Clustering / grouping** as the endpoint (survey §2.3; TAG 2021).
7. **Representation / feature sharing** only (survey §2.1; cross-stitch; routing).

A candidate survives only if its *object* is a §2.4 relation object **and** it is not merely a
reformulation of an existing arm (classical MTRL, MSSL/TR-0007, the matched baseline).

---

## Relation-object inventory (methods → taxonomy slot)

Cross-reference; full verdicts live in `PRIMARY_LITERATURE.md`, `CANDIDATES.md`, and
`improvements/taskrelation/research/literature/INDEX.md`.

| Method | Object | Family | Sym | Sparse side | Faithfulness here |
|---|---|---|---|---|---|
| Bonilla et al. 2007 MTGP | free-form `K^f=LLᵀ` | A | sym | dense | adapted (Gaussian regression) |
| Zhang & Yeung 2010 MTRL | `Ω`, tr=1 | A | sym | dense | control (in-category) |
| Zhang & Yeung 2010 MTGTP | `Ω` with inverse-Wishart | A | sym | dense | adapted (Bayesian regression) |
| Zhang & Schneider 2010 | `Ω⁻¹`, `Σ⁻¹` | A/B | sym | **precision** | `OUT_OF_SCOPE` (feature precision at m=3) |
| Zhang & Yang 2017 SPATS | `Ω` | A/B | sym | **covariance** | adapted (many-task; covariance sparsity) |
| Gonçalves et al. 2016 MSSL | learned precision `P` | A/B | sym | **precision** | `IN_SCOPE_WITH_JUSTIFIED_ADAPTATION` (TR-0007 already registered, head-summary attachment) |
| Rakitsch et al. 2013 | signal `C` + noise `Σ` | A | sym | dense | adapted (aligned Gaussian) |
| Jacob et al. 2008 CMTL | cluster norm `Σ(M)` | C | sym | spectral | `OUT_OF_SCOPE` |
| Kumar & Daumé 2012 GO-MTL | `(L,S)` | C/F | via overlap | `ℓ₁` on `S` | `OUT_OF_SCOPE` |
| Zhou & Zhao 2016 FCMTL | `Z` row-sparse | C | sym | `‖Z‖₁,₂` | `OUT_OF_SCOPE` |
| Yu et al. 2020 GAMTL | adjacency `A` | D | **undirected** | barrier+`ℓ₁` | adapted; fails directed-family gate |
| Lee et al. 2016 AMTL | directed elementwise nonnegative `B` | E | directed | `ℓ₁` rows | adapted (aligned-column assumption) |
| Zhou & Yang 2023 AutoTR | signed `R` | E | directed | soft-threshold | adapted |
| Oliveira et al. 2019 GAMTL | per-group directed `B^g` | E | directed | group lasso | adapted + invented groups |
| Misra et al. 2016 cross-stitch | `α` mixing | K/H | asymmetric-capable | none | `OUT_OF_SCOPE` (routing) |
| Fifty et al. 2021 TAG | directed affinity `Ẑ` | J/C | directed | grouping solve | `OUT_OF_SCOPE` (endpoint = grouping) |
| Yu et al. 2020 PCGrad | gradient projections | J | n/a | n/a | `OUT_OF_SCOPE` (update) |
| Liu et al. 2021 CAGrad | conflict-averse direction | J | n/a | n/a | `OUT_OF_SCOPE` (update) |
| Chen et al. 2018 GradNorm | per-task weights | I | n/a | n/a | `OUT_OF_SCOPE` (weights) |

This inventory is for orientation across the sibling documents; nothing in it is a result about
KS/SI/ER.

---

## Cross-links

* Survey mapping and A–L → survey homes: [`SURVEY_MAP.md`](./SURVEY_MAP.md).
* Primary sources and exact equations: [`PRIMARY_LITERATURE.md`](./PRIMARY_LITERATURE.md).
* m=3 algebra (pair counts, rank/df, Schur complements): [`THREE_TASK_ANALYSIS.md`](./THREE_TASK_ANALYSIS.md).
* Sparse covariance vs sparse precision, conditional dependence: [`MSSL_SPARSITY_ANALYSIS.md`](./MSSL_SPARSITY_ANALYSIS.md).
* Directed/asymmetric applicability: [`ASYMMETRIC_RELATIONS.md`](./ASYMMETRIC_RELATIONS.md).
* Comparison matrix / shortlist: [`CANDIDATES.md`](./CANDIDATES.md).
* Unresolved scientific questions: [`OPEN_QUESTIONS.md`](./OPEN_QUESTIONS.md).

Repository authorities: `improvements/taskrelation/research/DECISIONS.md` (DEC-0005, DEC-0010,
DEC-0013, DEC-0014), `FINDINGS.md` (F4/F5/F7/F8/F9/F10), `FAILURES.md` (FL-0001–FL-0004),
`FRAMEWORK.md` §2 rules, `VARIANT_BENCHMARK_PROTOCOL.md`,
`task_relations/MTRL_DIAGNOSTIC_SYNTHESIS.md`, `literature/INDEX.md`.
