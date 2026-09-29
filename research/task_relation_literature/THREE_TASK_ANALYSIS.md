# Three-task analysis — `m = 3` geometry, parameter counts and statistical resolution

**Scope.** Theory only. This document collects the arithmetic and the resolution limits
that apply *because the benchmark fixes exactly three tasks* (`ks`, `si`, `er`), for use
by any future relation-learning write-up in this directory. It does not decide a winner
and does not authorise a study. The conditional-independence primitive it assumes is
derived in [`MSSL_SPARSITY_ANALYSIS.md`](./MSSL_SPARSITY_ANALYSIS.md) §2–3 (Schur
complement, and the exact `3 × 3` identity `Ω_ij = (Σ_ikΣ_jk − Σ_ijΣ_kk)/det Σ`).

**Sources.** Primary: Zhang & Yang, “A Survey on Multi-Task Learning”, arXiv:1707.08114**v3**
(2021), §2.4 (Eqs. 20–25) and §2.8 (Eqs. 32–33). Gonçalves, Von Zuben & Banerjee,
*JMLR* 17(33):1–30, 2016, §§3.1–3.3, 3.6, 4.1. Repository:
[`VARIANT_BENCHMARK_PROTOCOL.md`](../../improvements/taskrelation/research/VARIANT_BENCHMARK_PROTOCOL.md)
(frozen conditions), [`FINDINGS.md`](../../improvements/taskrelation/research/FINDINGS.md)
F1–F10, [`MTRL_DIAGNOSTIC_SYNTHESIS.md`](../../improvements/taskrelation/research/task_relations/MTRL_DIAGNOSTIC_SYNTHESIS.md),
[`studies/LT-0002/analysis.md`](../../improvements/taskrelation/research/studies/LT-0002/analysis.md),
[`studies/DG-0001/result.json`](../../improvements/taskrelation/research/studies/DG-0001/result.json),
the AMTL card
[`lee-2016-asymmetric-mtl.md`](../../improvements/taskrelation/research/literature/lee-2016-asymmetric-mtl.md).

**Epistemic key.** `OBSERVED` = primary source or repository artefact; `INFERRED` =
derived here, algebra shown; `HYPOTHESIZED` = predicted, falsifiable, unmeasured.
Counts marked *(verified)* were recomputed here in exact integer arithmetic.

---

## 1. The constraint: `m = 3` is fixed and nothing in the benchmark can change it

The protocol freezes the task set to `ks_si_er` (§1) and forbids varying it to improve a
number. Every relation object in the benchmark therefore lives in an `m × m` space with
`m = 3`. The consequence is not a matter of taste: **the hypothesis space of every
relation mechanism the benchmark can test is finite and small**, and the two mandatory
controls exist precisely because a single 3-edge comparison cannot resolve a mechanism on
its own. `m = 3` is also the smallest `m` for which a *non-trivial* conditional-independence
statement exists: for `m = 2` there is no third variable, so conditioning on the other
variable coincides with marginal independence (`Ω₁₂ = 0` in a `2 × 2` precision is just
“the two tasks are uncorrelated”), and the precision adds no edge-selection content.

---

## 2. Parameter and structure counts at `m = 3`

| Object | Count at `m = 3` | Derivation |
|---|---:|---|
| Full symmetric covariance `Σ` (or precision `Ω`) | 6 | `m(m+1)/2` |
| — with MTRL's constraint `Ω ⪰ 0`, `tr Ω = 1` (survey Eq. 21) | 5 | 6 − 1 trace constraint |
| Correlation-normalised `Σ` (or partial-correlation `Ω`) | 3 | `m(m−1)/2` — the three pairs |
| Undirected edges in a relation graph | 3 | `C(3,2)` |
| Undirected graphs (all supports of `Ω`) | **8** *(verified)* | `2³` |
| — of which connected | 4 *(verified)* | 3 spanning trees + 1 triangle |
| — of which forests | 7 *(verified)* | all except the triangle |
| — spanning trees | 3 *(verified)* | Cayley: `m^{m−2} = 3¹` |
| Directed edges | 6 | `m(m−1)` |
| Simple digraphs | 64 *(verified)* | `2⁶` |
| DAGs | **25** *(verified)* | OEIS A003024 at `n = 3` |
| Task partitions (“clustering”) | **5** *(verified)* | Bell number `B₃ = 5` |
| Possible ranks of a `3 × 3` relation matrix | `{1, 2, 3}` | rank ≤ `m` |

Read together: a “structure learning” claim at `m = 3` has a **3-bit** support and an
**8-element** hypothesis class. One edge flip is 12.5 % of the entire space. A
ranking- or grouping-based claim is finer still: 5 partitions, of which two are the trivial
extremes (all-in-one, all-singletons) and only three are the informative “one pair +
singleton” clusterings. The `tr Ω = 1` row is the count used in the taxonomy's
“m = 3 geometry constraints” (in
[`TASK_RELATION_TAXONOMY.md`](./TASK_RELATION_TAXONOMY.md));
this document additionally tracks the correlation-normalised (3) and unconstrained (6)
counts because the precision family does *not* impose `tr Ω = 1` (MSSL uses the log-det
barrier instead).

### 2.1 Grouping is not conditional independence  `INFERRED`

A three-task partition has five states, whereas an undirected precision support has
A "pair + singleton" partition can be represented by a **one-edge** precision
graph if the singleton is exactly independent of that pair; a two-edge chain instead
has a conditionally independent *pair of endpoints* connected through the middle task,
not an isolated singleton. Even the all-in-one group need not have a complete
precision graph. Group membership and conditional dependence are different estimands;
their coincidence in selected special cases cannot justify treating clustering as a
sparse-precision implementation
([`TASK_RELATION_TAXONOMY.md#category-test-zhang-and-yang-24`](./TASK_RELATION_TAXONOMY.md#category-test-zhang-and-yang-24)).

### 2.2 Rank geometry  `INFERRED` *(verified numerically)*

A `3 × 3` relation matrix can be rank 1, 2 or 3. Rank 1 is the degenerate “one shared
direction” case. The repository's classical MTRL trace-normalised covariance
**approaches** `Ω* = (1/3) **11**ᵀ` at `smp`+25L (F6 reports
`0.33308 ± 0.00017` for KS--SI; see
[`MSSL_SPARSITY_ANALYSIS.md`](./MSSL_SPARSITY_ANALYSIS.md) §3.3).
That limiting matrix has `rank Ω* = 1`, `tr Ω* = 1`; the *measured*
matrices need not have exact rank one. For an equicorrelation matrix with
off-diagonal `r`, `det = (1−r)²(1+2r) → 0` as `r → 1`, illustrating the
PSD-cone boundary approached under trace-one normalisation. Consequences:

* a low-rank relation arm has little room at `m = 3` — rank 2 removes one
  direction and rank 1 approaches the observed saturation pattern;
* a precision parameterisation with a log-det barrier cannot reach that boundary
  ([`MSSL_SPARSITY_ANALYSIS.md`](./MSSL_SPARSITY_ANALYSIS.md) §4.1), so
  “near-saturated vs interior” is a **testable distinction** (prediction P5 there).

### 2.3 Directed parameter counts, and what “directed” buys at `m = 3`  `OBSERVED`

The directed family (family A of LT-0002) is represented here by Lee et al. 2016 (AMTL),
survey Eq. (25): `min Σ_i (1 + λ₁‖â_i‖₁) Σ_j ℓ(y_ji; (w^i)ᵀx^{ij} + b_i) + λ₂‖W − WA‖²_F`,
`s.t. a_ij ≥ 0`, where `â_i` is row `i` of `A` with the diagonal removed and `A` is the
**directed** transfer matrix (`a_ii = 0`, non-negative). Parameter counts:

| | symmetric (MTRL/MSSL) | directed (AMTL Eq. 25) |
|---|---:|---:|
| free relation entries at `m = 3` | 3 | **6** |
| support patterns | 8 | 64 (25 acyclic) |
| asymmetry statistic | — | 3 differences `a_ij − a_ji` |
| constraint | PD/PSD ± ℓ₁ | `a_ij ≥ 0`, rows ℓ₁-scaled by the task’s own loss |

Two points the project must not lose:

1. **The asymmetry lives in 3 numbers.** With `m = 3` the only asymmetry content is the
   three pairwise differences; there is no higher-order or per-layer asymmetry to detect.
   The `+0.3411 / +0.2794` raw ER-directed LOSO gains were already shown to be optimizer
   exposure, not semantics (F8, DG-0001), and the step-controlled residuals were `+0.0053`
   (CI spans zero) and `−0.0589`.
2. **The survey classifies the directed object as a symmetric precision in disguise.**
   Verbatim from v3 §2.4: *“Though A is asymmetric, from the perspective of the
   regularizer, the task relations here are symmetric and act as the task precision matrix
   with a restrictive form.”* Algebraically, `‖W − WA‖²_F = tr(W(I−A)(I−A)ᵀWᵀ)`, i.e.
   Eq. (25) is Eq. (21) with `Ω⁻¹ = (I−A)(I−A)ᵀ`. The regularizer sees only the symmetric
   product `(I−A)(I−A)ᵀ`; the directionality enters through the **factorisation** and the
   loss-scaled rows, not through the coupling matrix. This is why DEC-0014 had to label
   the directed arm project-original and why a directed *published* pass was not available
   (LT-0002 §3–4): at the regularizer level the benchmark cannot distinguish “directed
   relation” from “a restricted-form symmetric precision”.
   *Two readings are recorded, neither promoted* (matching the taxonomy's handling): the
   survey's symmetrised reading above is `OBSERVED` at survey level, while the primary
   paper keeps a directed non-negative `B` (`B ⪰ 0`, `B_tt = 0`, rows ℓ₁-scaled by the
   task's own loss) and its Theorem 1 is a directed statement — see the AMTL card
   [`lee-2016-asymmetric-mtl.md`](../../improvements/taskrelation/research/literature/lee-2016-asymmetric-mtl.md)
   and [`studies/LT-0002/analysis.md`](../../improvements/taskrelation/research/studies/LT-0002/analysis.md) §3.
   The `m = 3` consequence used here (6 directed entries, 3 asymmetry differences) holds
   under either reading; the treatment of the directed family is developed further in the
   sibling doc `ASYMMETRIC_RELATIONS.md`.

---

## 3. Statistical resolution and uncertainty at `m = 3`

### 3.1 What the effective sample size for the relation object is  `INFERRED` / gap flagged

p-MSSL assumes the rows `w^j ∈ ℝ^m` are i.i.d. `N(0, Σ)`, so the nominal sample size for
the `3 × 3` precision is the number of rows, `d` (`Gonçalves §3.3`). In the repository's
adapter the summary matrix is `[m, d] = [3, 2001]` (`04-mssl/README.md`,
`studies/TR-0007/PLAN.md`), so `d = 2001` nominally supplies the precision with 2001
observations. But the `d` index runs over the **coordinates of one averaged summary
vector per task** (2000 mean head-weight coordinates plus the mean bias), not over
independently sampled task vectors, and come from a single training trajectory.
The feature-wise independence assumption is unverified; an effective number of independent observations cannot
be inferred from `d` alone, and the paper's high-probability recovery
results do not transfer unconditionally. **This is an open source-verification /
modelling gap, not a resolved fact**, and it is why §3.2 treats seeds/folds — not `d` —
as the independent observations for a `3 × 3` relation claim.

### 3.2 The project's measured `Ω` uncertainty  `OBSERVED`

From F6 / `MTRL_DIAGNOSTIC_SYNTHESIS.md` (classical MTRL covariance entries; MSSL's
precision counterpart is **unmeasured**):

| Setting / axis | KS↔SI | KS↔ER | SI↔ER |
|---|---|---|---|
| `smp` 25 L, seeds 0–4 | `+0.33308 ± 0.00017` (5/5 +) | `+0.200 ± 0.266` (4/5 +) | `+0.200 ± 0.266` (4/5 +) |
| `smp` 25 L, 10 LOSO folds | `+0.286 ± 0.012` (10/10 +) | `+0.098 ± 0.072` (9/10 +) | `+0.033 ± 0.103` (7/10 +) |
| `smp` 16 L, seeds 0–4 | `+0.110 ± 0.236` (4/5 +) | `+0.035 ± 0.213` (4/5 +) | `−0.256 ± 0.076` (5/5 −) |

Reading: an edge whose magnitude is below roughly `0.1–0.25` is **not** distinguishable
from noise at `n = 5` seeds, and the ten LOSO folds are not independent observations (they
vary only the ER speaker). The one extremely stable entry (`0.33308 ± 0.00017`) is the
*saturated* one, so stability there carries no pair-specific information (F6, F7). Three
edges is a small multiple-comparison burden (a max-statistic over 3 entries is tractable),
but each entry's interval must come from seed/fold replicates, not from the nominal `d`.

### 3.3 The paper's own uncertainty control  `OBSERVED`

MSSL selects `λ₁, λ₂` by cross-validation (Algorithm 1: “penalty parameters chosen by
cross-validation”) and §4.1 uses a stability-selection procedure (Meinshausen &
Bühlmann 2010, as cited by the paper) — a sub-sampling method “that provides a way to find
stable structures and hence a principle to choose a proper amount of regularization”. For
a solver whose zeros are exact, the right uncertainty object is therefore the **stability
of the support** across replicates, not the numerical magnitude of a shrunk entry. The
project's screen uses a researcher-fixed `λ₂ = 0.01` and no stability selection
(DEC-0016), so it cannot report a stability-selected graph; the support is a single
discrete observation. This limitation must travel with any `m = 3` sparsity claim.

### 3.4 Resolution limits that follow from `m = 3`

1. **Support resolution**: at most 3 edges; 8 possible graphs; a one-edge difference is
   12.5 % of the hypothesis class.
2. **Conditioning depth**: the conditional set is always exactly one task, so
   “conditional independence” and “explained away by the third task” are the same
   statement. Higher-order relations offer no escape: the survey's high-order object of
   Eq. (22) is `(WᵀW)^t` trace-normalised, a power of the same `3 × 3` second-moment
   matrix, and integer powers of a `3 × 3` matrix add **no new parameters** — so Eq. (22)
   is a reparameterisation of the same 3-entry object at `m = 3`.
3. **Asymmetry resolution**: 3 differences (§2.3).
4. **Grouping resolution**: 5 partitions, 2 trivial (§2.1).
5. **Rank resolution**: 3 possible ranks, and rank 1 is the observed failure mode (§2.2).
6. **Exchangeability trap**: if the summary geometry is near-equicorrelated, both the
   covariance and the (reparameterised) precision are near-uniform, and no pair-selectivity
   can be recovered from either (`MSSL_SPARSITY_ANALYSIS.md` §3.2, prediction P3).
7. **Evaluation conditions**: the sample-weighted aggregate is SI-dominated, so it can
   mask ER movement (protocol §4); ER's ordinary split inflates by ≈ 15 pp and any ER
   claim needs LOSO (F3); sub-1 pp effects need ≥ 5 seeds (F1).

---

## 4. What `m = 3` can still decide  `INFERRED`

The limitation is resolution, not impossibility. At `m = 3` the following are genuinely
testable, and each is a discrete, pre-registerable outcome:

* **presence vs absence of a single strong edge**, by an exact zero in the precision
  (an identifiable event, since the solver's support is exact);
* **sign flips** of an edge across seeds/folds (the measured `±0.266` spread shows a flip
  is detectable but not always resolved);
* **saturated vs interior** (`rank 1` vs `Ω ≻ 0`), which separates the MTRL failure mode
  from a precision parameterisation;
* **uniform vs selective** normalised couplings, which tests whether any pair structure
  exists in `R̃` at all;
* **directed vs symmetric on the 3 differences**, subject to the Eq. (25) caveat that the
  regularizer sees only `(I−A)(I−A)ᵀ` (§2.3).

None of these should be reported without a matched seed/fold count and the conditions the
signals were measured under (F5, F6, F10) — the project's own rule is that a signal is
interpretable only with its conditions (FRAMEWORK.md §1).

---

## 5. Crosslinks

* [`MSSL_SPARSITY_ANALYSIS.md`](./MSSL_SPARSITY_ANALYSIS.md) — Schur complement,
  the exact `3 × 3` conditional-independence identity, the equicorrelation and rank-1
  saturation results, and the `λ₂` scale analysis that `m = 3` sparsity claims depend on.
* [`TASK_RELATION_TAXONOMY.md`](./TASK_RELATION_TAXONOMY.md) — category boundaries
  (`#category-test-zhang-and-yang-24`), relation-object dimensions
  (`#relation-object-dimensions`), D8 estimability at `m = 3`, the
  “m = 3 geometry constraints” arithmetic this file treats in full, and
  conditional-vs-direct dependence (`#conditional-vs-direct-dependence`); §2.1–2.3 here
  are the `m = 3` instances of those categories.
* [`SURVEY_MAP.md`](./SURVEY_MAP.md) and [`PRIMARY_LITERATURE.md`](./PRIMARY_LITERATURE.md)
  — survey §2.4 map and primary-source verdicts, including the Eq. (25) → Eq. (21)
  reduction quoted in §2.3.
* `ASYMMETRIC_RELATIONS.md` (sibling doc, in preparation) — the directed-family treatment
  that §2.3 motivates.
* Repository: [`VARIANT_BENCHMARK_PROTOCOL.md`](../../improvements/taskrelation/research/VARIANT_BENCHMARK_PROTOCOL.md)
  (frozen `m = 3`, ≥ 5 seeds, LOSO, SI-dominated aggregate),
  [`FINDINGS.md`](../../improvements/taskrelation/research/FINDINGS.md) F1–F10,
  [`studies/TR-0007/PLAN.md`](../../improvements/taskrelation/research/studies/TR-0007/PLAN.md),
  [`studies/DG-0001/result.json`](../../improvements/taskrelation/research/studies/DG-0001/result.json),
  [`DECISIONS.md`](../../improvements/taskrelation/research/DECISIONS.md) DEC-0013/0014/0016.
