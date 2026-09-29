# RELATION_SEMANTICS — what a 3-task "relation" is, and what a zero in it does *not* mean

> **Status note (2026-09-29, integration).** This document was written against checkout
> `664c572` / `9955166`, before the TR-0007 screen closed and before DG-0007 was registered. Its
> reasoning, derivations and predictions are unchanged and none of its claims is withdrawn; only
> sentences that report a *corrected seed-42 TR-0007 result*, a *DG-0007 record*, or an *edge-support
> reading* as absent **in this checkout** are stale, and they are preserved as that checkout's
> history. Since then: the screen closed `REJECTED` at the researcher-fixed `λ₂ = 0.01` (dense
> support, all three partial correlations `+0.4911`), DG-0007 is registered and pre-registered with
> **no result**, and the post-TR-0007 reconciliation — including the exact-solve reading that makes
> the published `λ₂` axis empty of a scale fix — is
> [`POST_TR0007_SYNTHESIS.md`](./POST_TR0007_SYNTHESIS.md) (§5 lists every stale sentence by file and
> line). Source: `research/taskrelation-literature` `9955166`, `a53f29d`, integrated here.


Status: **literature/theory only.** No code, no configs, no registered study, no run, no
ranking, no winner. This file exists so that the words *covariance*, *precision*,
*partial correlation*, *directed relation* and *transfer* are not used interchangeably in
any future wavCSE task-relation write-up.

Companion documents in this directory: [`IDENTIFIABILITY.md`](./IDENTIFIABILITY.md)
(the statistical companion — what an `m = 3` study can and cannot identify),
[`MSSL_SPARSITY_ANALYSIS.md`](./MSSL_SPARSITY_ANALYSIS.md) (where the `3 × 3` identity used
here is derived), [`THREE_TASK_ANALYSIS.md`](./THREE_TASK_ANALYSIS.md) (`m = 3` geometry and
resolution), [`TASK_RELATION_TAXONOMY.md`](./TASK_RELATION_TAXONOMY.md) (object dimensions
D1–D8), [`ASYMMETRIC_RELATIONS.md`](./ASYMMETRIC_RELATIONS.md) (the directed family),
[`PRIMARY_LITERATURE.md`](./PRIMARY_LITERATURE.md), [`SURVEY_MAP.md`](./SURVEY_MAP.md),
[`CANDIDATES.md`](./CANDIDATES.md), [`SUCCESSOR_STUDIES.md`](./SUCCESSOR_STUDIES.md),
[`OPEN_QUESTIONS.md`](./OPEN_QUESTIONS.md).

**Epistemic key.** `OBSERVED` = read in a primary source **or** in a repository artefact
(the artefact is named); `INFERRED` (= the assignment's *DERIVED*) = derived or computed here,
algebra/arithmetic shown; `HYPOTHESIZED` = falsifiable prediction, not measured;
`PRIMARY_SOURCE_NOT_VERIFIED` = a claim that a prior pass explicitly declined to verify — it
is never promoted here.

**Source provenance for this note.** No primary PDF was reopened for this file. Every
source-attributed claim below is taken from the repository's existing verified ledgers and is
labelled with the ledger it comes from: [`PRIMARY_LITERATURE.md`](./PRIMARY_LITERATURE.md)
(papers read directly in the previous literature pass), [`SURVEY_MAP.md`](./SURVEY_MAP.md) §10
and [`ASYMMETRIC_RELATIONS.md`](./ASYMMETRIC_RELATIONS.md) §10 (their own verification
ledgers). Where a prior pass recorded a verification gap, the gap is carried forward
verbatim rather than closed by assertion.

---

## 1. The worked example: one `3 × 3` matrix, two different "relations"

The protocol fixes exactly three tasks (`ks_si_er`; [protocol §1](../../../../improvements/taskrelation/research/VARIANT_BENCHMARK_PROTOCOL.md))
with head widths 12 / 1251 / 4 and `smp` 0.5 over all 25 layers. Write the task-parameter
covariance the relation objects are built from as

```
Sigma = [[ 1.0 , 0.2 , 0.4 ],
         [ 0.2 , 1.0 , 0.5 ],
         [ 0.4 , 0.5 , 1.0 ]]        Sigma_12 = .2,  Sigma_13 = .4,  Sigma_23 = .5,  diag = 1
```

### 1.1 It is a legitimate positive-definite covariance  `INFERRED` (verified exactly)

With integer/fraction arithmetic:

* `det Sigma = 1 - (.2^2 + .4^2 + .5^2) + 2(.2)(.4)(.5) = 1 - 0.45 + 0.08 = 0.63`;
* the three `2 × 2` principal minors are `1 - .04 = 24/25`,
  `1 - .16 = 21/25`, `1 - .25 = 3/4`, all positive;
* every principal minor positive ⇒ `Sigma ≻ 0` (equivalently, Sylvester's criterion on the
  leading minors `1, 24/25, 0.63`).
* eigenvalues `{0.447562, 0.805996, 1.746442}`, condition number `3.902` — an **interior,
  well-conditioned** member of the cone, not a degenerate or near-singular matrix.

### 1.2 Its exact inverse, with an *exactly zero* off-diagonal  `INFERRED` (verified exactly)

The adjugate/cofactor form of the `3 × 3` inverse (the identity `(C)` of
[`MSSL_SPARSITY_ANALYSIS.md`](./MSSL_SPARSITY_ANALYSIS.md) §2.3,
`P_ij = (Sigma_ik Sigma_jk - Sigma_ij Sigma_kk) / det Sigma`, `k` the third task) gives

```
P = Sigma^{-1} = [[ 25/21 ,   0     , -10/21 ],
                  [  0    ,  4/3    ,  -2/3  ],
                  [ -10/21,  -2/3   ,  32/21 ]]      (exact fractions; Sigma P = P Sigma = I)
```

so

> **`P_12 = 0` exactly, while `Sigma_12 = 0.2 ≠ 0`.**

The zero is not a rounding artifact of a badly conditioned matrix: it holds at
`det Sigma = 0.63` and condition number `3.9`.

### 1.3 Why: the third task explains the pair away  `INFERRED` (exact)

The identity `(D)` of [`MSSL_SPARSITY_ANALYSIS.md`](./MSSL_SPARSITY_ANALYSIS.md) §2.3 says
`P_ij = 0` ⟺ `Sigma_ij = Sigma_ik Sigma_jk / Sigma_kk`. Here

```
Sigma_13 * Sigma_23 / Sigma_33 = (0.4)(0.5)/1.0 = 0.20 = Sigma_12 .
```

So the *marginal* association `Sigma_12 = 0.2` is accounted for by
conditioning on task 3. In the Gaussian model, tasks 1 and 2 are
conditionally independent given task 3, despite their nonzero marginal
association. The undirected dependence path is `1--3--2`; **no causal
arrows** follow from the covariance or precision matrix.
The distinction is formalized in [`TASK_RELATION_TAXONOMY.md`](TASK_RELATION_TAXONOMY.md) D6
and [`MSSL_SPARSITY_ANALYSIS.md`](MSSL_SPARSITY_ANALYSIS.md) §2.

### 1.4 The same content in two more forms  `INFERRED` (verified exactly)

* **Schur complement** (conditional law of tasks 1–2 given task 3):
  `Sigma_{12|3} = [[0.84, 0.0], [0.0, 0.75]]` — the conditional covariance is *diagonal*.
  Marginal correlation `0.2`; conditional correlation `0.0`.
* **Partial correlations** `rho_ij.k = -P_ij / sqrt(P_ii P_jj)`:

  | pair | marginal correlation `Sigma_ij/sqrt(Sigma_ii Sigma_jj)` | partial correlation | raw precision `P_ij` |
  |---|---:|---:|---:|
  | 1–2 | `+0.200` | `0.0` | `0` |
  | 1–3 | `+0.400` | `+0.353553` `= 1/(2*sqrt2)` | `-10/21` |
  | 2–3 | `+0.500` | `+0.467707` | `-2/3` |

Read together: the *same* three numbers that describe a dense, all-positive covariance with
three nonzero marginals describe a precision whose `(1,2)` entry is exactly zero and whose
remaining two entries are conditionally positive. Both descriptions are correct. They are
descriptions of **different objects**.

### 1.5 What the example is *not*

It is not a claim about KS/SI/ER (§5), not evidence for or against any estimator, and not an
argument that any edge *should* be zero. It is a minimal witness that the sentence
"tasks *i* and *j* have no relation" is **under-specified**: it is true or false depending on
which object is meant. `INFERRED`.

---

## 2. Five distinct objects that get called "the relation"

`TASK_RELATION_TAXONOMY.md` (`D1` object type, `D2` symmetry, `D6` direct vs conditional)
separates these by construction; this section separates them by **what a zero in each one
means** and **what evidence could identify it**.

| # | Object | How obtained | Symmetry | What an exact zero means | What identifies it | Where it lives in this project |
|---|---|---|---|---|---|---|
| **(a)** | covariance `Sigma` (`Ω` in MTRL's trace-1 normalisation) | closed form / moment estimator; MTRL: `Omega = sqrt(W^T W) / tr(sqrt(W^T W))` (Zhang & Yeung UAI 2010, [primary ledger](PRIMARY_LITERATURE.md)) | symmetric | marginal zero *within the assumed coefficient model*, not an absence of direct transfer | coefficient distribution assumptions and a stable estimator | classical MTRL; Ω histories (F5/F6) |
| **(b)** | precision `P = Sigma^{-1}` and partial correlation `−P_ij/√(P_ii P_jj)` | p-MSSL estimates `P` directly by penalized likelihood ([Gonçalves et al. 2016](https://jmlr.org/papers/volume17/15-215/15-215.pdf), §§3.1,3.3) | symmetric | a zero encodes conditional independence under the assumed Gaussian coefficient model | a stable fitted support and a defensible coefficient sampling model | registered TR-0007; corrected screen result absent in this checkout |
| **(c)** | directed reconstruction `B`, `W ≈ W B`, `B_tt = 0` | alternating weighted (non-negative) lasso on aligned task columns — AMTL (Lee, Yang & Hwang, ICML 2016, verified via [`PRIMARY_LITERATURE.md`](./PRIMARY_LITERATURE.md)); GAMTL; AutoTR | asymmetric **object** | no source-reconstruction edge; with `B ≥ 0` a harmful effect must appear as an *absent* edge, not a negative one | an aligned-column fit **plus** a rank/conditioning argument (near-collinear columns make `B` non-unique) | family E; `ASYMMETRIC_RELATIONS.md`; project-original TR-0008 |
| **(d)** | symmetric *induced* penalty `(I - B)(I - B)^T` | algebra from (c): the squared Frobenius norm of `W - W B` equals `tr(W (I-B)(I-B)^T W^T)`. Survey §2.4: *"Though A is asymmetric, from the perspective of the regularizer, the task relations here are symmetric and act as the task precision matrix with a restrictive form."* ([`SURVEY_MAP.md`](./SURVEY_MAP.md) §3.2, `OBSERVED` at survey level) | **symmetric by construction**, whatever `B` is | a *factorisation* constraint on a precision, not a direction | the same fit as (c); the operator is always PSD and symmetric | `ASYMMETRIC_RELATIONS.md` §1.1; `THREE_TASK_ANALYSIS.md` §2.3 |
| **(e)** | **controlled performance transfer** `T(A ← B) = acc(A with B) − acc(A alone)` | trained interventions, exposure-matched | n/a (two numbers per pair) | no measured benefit of one task's data on another's accuracy | matched optimizer steps / per-task batch size / loss scaling / epoch budget / checkpoint tag, plus ER LOSO | DG-0001 stages A–D; F8; `ASYMMETRIC_RELATIONS.md` §8 |

**Why they may not be substituted for one another.** `INFERRED`.

1. **(a) → (b) is not an edge-preserving relabelling.** The map
   `Sigma ↦ Sigma⁻¹` is a bijection on positive-definite matrices, but
   the **zero sets differ**: §1 has `P_12=0` and `Sigma_12≠0`.
   Reading a covariance entry as if it were the corresponding precision
   entry is a mathematical error ([MSSL analysis](MSSL_SPARSITY_ANALYSIS.md) §1.2).
2. **(b) → (c)/(d) is not an implication.** A zero *conditional* edge does not make `B_st`
   zero, and a directed `B` can be nonzero with an entirely dense precision. The directed
   family buys a *factorised, sparse, interpretable* precision and a hypothesis about
   direction — not a different kind of penalty (`ASYMMETRIC_RELATIONS.md` §1.1, §3).
3. **(c) → (e) is not an implication.** `B_st > 0` is a coefficient of one task's parameter
   vector in *another's reconstruction*; directionality of *performance* requires the
   intervention contrast `T(A ← B) − T(B ← A)`. The project's own raw asymmetry
   (`+0.3411` / `+0.2794` LOSO) was reproduced by **optimizer-exposure-only** controls
   (+0.3358 / +0.3383), leaving `+0.0053` `[-0.0289, +0.0394]` and `-0.0589`
   `[-0.0883, -0.0295]` — i.e. one unresolved and one *negative*
   (`FINDINGS.md` F8; `studies/DG-0001/analysis.md`; §8.2 of `ASYMMETRIC_RELATIONS.md`).
4. **The AMTL-like reconstruction penalty uses a symmetric operator.**
   For the stated squared penalty it depends on `(I−B)(I−B)^T`;
   graph interpretation and other loss-weighting terms may still depend
   on `B`. Coefficient direction is not identified as causal transfer
   by rewriting the quadratic form.

**Counting consequence at `m = 3`.** `OBSERVED`/`INFERRED` (counts re-verified in
[`THREE_TASK_ANALYSIS.md`](./THREE_TASK_ANALYSIS.md) §2): (a)/(b)/(d) each hold **3**
free off-diagonal numbers; (c) holds **6**; (e) is a **6-cell** matrix of measured deltas.
The hypothesis classes are 8 (undirected supports), 64 (simple digraphs), 25 (DAGs), 5
(partitions). One edge flip is 12.5 % of the undirected support class.

---

## 3. What the three objects cannot imply

These implications fail for distinct reasons; none supplies an empirical
direction of useful transfer. `DERIVED`.

### 3.1 Conditioning is not marginal independence

`X_1 ⊥ X_2 | X_3` holds in §1 (`Sigma_{12|3}` diagonal), while
`X_1` and `X_2` are **not** independent marginally (`Sigma_12=.2`).
The covariance zero condition is nonlinear in the entries of `Sigma`;
p-MSSL instead puts ℓ₁ directly on precision `P`, making *its* fitted
entry exactly zero under thresholding. Other estimators could penalize
conditional covariance differently; the published p-MSSL choice is
not a mathematical impossibility for them.

### 3.2 Basis: zeros are coordinate-dependent, so they do not "follow" a change of task basis

Independence of coordinate axes is not a basis-free property. Rotate the `(1,2)` task plane
by `30°` (`R = rotation`, `Sigma' = R Sigma R^T`):

```
Sigma'_12 :  0.200  ->  0.100000          (the marginal number moves)
P'_12     :  0.000  -> -0.061859          (the exact zero disappears)
```

`DERIVED`. Conditional independence is a statement about the **named task
coordinates**, not invariant under arbitrary mixing of tasks. This illustrative
rotation does **not** model a permissible redefinition of the KS/SID/ER labels.
Pooling or summary changes can also change a learned relation (F5), but that
is a separate empirical representation effect, not a consequence of this
task-axis rotation. An orthogonal change of *feature* coordinates preserves
the task Gram and does not establish such an effect.

### 3.3 Transfer: no algebraic path from any parameter-object zero to a performance fact

Even granting (a)–(d) exactly, none of them *follows* into (e):
`T(A ← B)` is defined by a retrained model under matched exposure. The repository's own
record is unambiguous that the raw cell values of the transfer matrix are dominated by
optimizer exposure (F8), and that the measured `Omega` sign agreement with controlled
transfer is 6/10 (KS↔ER) and 3/10 (SI↔ER) (`studies/DG-0001/analysis.md`;
`FINDINGS.md` F6). "No relation" in the parameter object does not mean "no transfer" in the
network: the shared trunk moves under every task's loss even if one head-relation edge is zero
([`SUCCESSOR_STUDIES.md`](./SUCCESSOR_STUDIES.md) CANDIDATE-A §5; `TASK_RELATION_TAXONOMY.md`
D6).

---

## 4. Relation to the previous numerical smoke

The preceding literature pass's throwaway numerical smoke used exactly
`Sigma_12=.2`, `Sigma_13=.4`, `Sigma_23=.5`, diagonal 1: determinant `.63`,
nonzero marginal `.2` and zero precision `P_12`. §1 reproduces that
calculation with exact fractions and its full inverse. Other prior
illustrations in [`MSSL_SPARSITY_ANALYSIS.md`](MSSL_SPARSITY_ANALYSIS.md)
exercise different Gram matrices; none is a measured KS/SID/ER covariance.

| Previous-pass numeric check | Value it produced | How §1 relates |
|---|---|---|
| [`MSSL_SPARSITY_ANALYSIS.md`](./MSSL_SPARSITY_ANALYSIS.md) §2.3 — identity `(C)` verified on `Sigma = [[2,.7,.3],[.7,1.5,.5],[.3,.5,1.2]]` | `rho_{12.3} = 0.36465` (nonzero) | another point of the same conditional-covariance identity, contrasting with the exact zero in §1; neither is a project measurement. |
| §3.2 — equicorrelation family `R = (1-r)I + r 11^T` with partial correlation `rho = r/(1+r)`, bounded by `1/2` | `r = .5 -> rho = .3333`; `r = .9 -> .4737`; `r = .999 -> .4997` | §1 is **not** equicorrelated: its three marginal correlations `(.2, .4, .5)` and three partial correlations `(0, .353553, .467707)` differ across pairs. It is a concrete instance of the "unequal fitted diagonals or Gram entries can break the analogy" branch of prediction **P3** in that document. |
| §3.3 — trace-1 saturation limit `Omega* = (1/3) 11^T`, `rank = 1`, `tr = 1`; reproduces the observed `0.33308 ± 0.00017` (F6) as the PSD-cone boundary | `r = .999 -> Omega_off = 0.3157`; exact limit `1/3` | §1 is an **interior** point of the cone (`det = .63`, `cond = 3.902`, rank 3). It therefore isolates the *algebraic* zero from the *degeneracy* story: the zero is not produced by near-singularity. |
| §5.2 — illustrative graphical lasso on unit-diagonal `R̃` with off-diagonals `(.3, .2, .4)`: exact zeros `0/3` at `λ₂ = 0.01`, transition at `λ₂ ≈ 0.2–0.5` | `0/3` zeros at the frozen screen value; `-0.290/-0.106/-0.449 -> -0.280/-0.100/-0.435` shrinkage | §1's `P` is the **`λ₂ = 0` unpenalised optimum** (`P = Sigma^{-1}`). It shows what a returned *exact* zero would have to be compared against: at `λ₂ = 0` the zero is data-determined; at `λ₂ > 0` a zero is a soft-thresholding event. The screen's `λ₂ = 0.01` is researcher-fixed (DEC-0016), not selected. |
| `improvements/taskrelation/research/tests/test_mssl_omega_solver.py` — solver smoke: `λ₂ = 0` recovers `S^{-1}` (`test_zero_penalty_recovers_the_inverse_covariance`), `λ₂ > 0` soft-thresholds off-diagonals to exact zeros, primal–dual certificate | `Ω = S^{-1}` at `λ₂ = 0`; exact zeros at large `λ₂` | §1 *is* such an `S^{-1}` for a unit-diagonal `S`. The smoke test and this example therefore agree on the same object; the example adds the semantic point the test cannot make (a zero here is conditional independence, and it coexists with a nonzero covariance). |

**Reading of the comparison (`DERIVED`):** one well-conditioned covariance
can have a nonzero marginal entry and a zero conditional precision entry;
the ℓ₁-regularized screen may additionally *select* zeros even when an
unpenalized inverse is dense. Neither phenomenon demonstrates useful transfer.

---

## 5. Relation to the actual measured KS / SI / ER relation entries

### 5.1 What the project actually measures  `OBSERVED`

The measured relation object in this repository is classical MTRL's **trace-1 covariance**
`Omega = sqrt(W^T W)/tr(sqrt(W^T W))` over the class-mean head summary (12/1251/4 classes →
one mean weight row + mean bias each, `normalize_w: true`, `d = 2001`)
(`task_relations/MTRL_DIAGNOSTIC_SYNTHESIS.md`; `04-mssl/README.md`). Its off-diagonals are
**`Sigma`-type numbers, not precision entries**. Reported values (`FINDINGS.md` F6;
`MTRL_DIAGNOSTIC_SYNTHESIS.md` "Relation stability", both `OBSERVED`):

| Setting / axis | KS↔SI | KS↔ER | SI↔ER |
|---|---:|---:|---:|
| `smp` 25 L, seeds 0–4 | `+0.33308 ± 0.00017` (5/5 +) | `+0.200 ± 0.266` (4/5 +) | `+0.200 ± 0.266` (4/5 +) |
| `smp` 25 L, 10 LOSO folds | `+0.286 ± 0.012` (10/10 +) | `+0.098 ± 0.072` (9/10 +) | `+0.033 ± 0.103` (7/10 +) |
| `smp` 16 L, seeds 0–4 | `+0.110 ± 0.236` (4/5 +) | `+0.035 ± 0.213` (4/5 +) | `-0.256 ± 0.076` (5/5 −) |

### 5.2 The error §1 exposes — and it is the tempting one here

Reading these three numbers as "the conditional relation between the tasks" is the §1
mistake. They are covariance entries. Converting them to partial correlations requires
`P = Omega^{-1}`, i.e. the **full** matrix (diagonals included), and the conversion is *not*
a rescaling of the off-diagonals. Two facts make this concrete:

* the third-task coincidence `(D)` says the conditional edge `(i,j)` is the *residual*
  `Sigma_ij - Sigma_ik Sigma_jk / Sigma_kk`; the reported table contains no such residuals
  and no diagonals, so the conditional edge is **not derivable from the published table**;
* under the *assumption* of an equicorrelated trace-1 matrix (off-diagonals equal, diagonal
  `1/3`), a covariance entry `0.2` corresponds to a correlation of `0.6` and a partial
  correlation of `0.375`, and the saturated entry `0.33308` corresponds to correlation
  `≈0.9992`, partial correlation `≈0.4998`, `det R ≈ 0`. `INFERRED` — and explicitly
  **conditional on an assumption the data do not establish** (F6 shows the entries are *not*
  equal except at saturation). It is given only to show the size of the gap, not as a
  measurement.

### 5.3 Generic unpenalized precision is dense; selected support needs evidence  `DERIVED`

Before regularization, `P_ij=0` imposes one equation
`rho_ij=rho_ik rho_jk` within a three-dimensional correlation domain;
generic continuous covariance draws have nonzero precision entries.
After **ℓ₁** penalization, exact fitted zeros are selected on whole
regions of Gram space, and their frequency depends on the actual data
and λ₂. The representative numerical solve in the previous pass is
*not* the actual TR-0007 Gram; a dense/sparse screen result remains open.

### 5.4 Two different failure modes hide behind "unstable / saturated"  `INFERRED`

| Observation | Which object is limited | What it does *not* license |
|---|---|---|
| KS↔ER and SI↔ER `±0.266` at 5 seeds ⇒ derived `SE ≈ 0.119` (`SD/sqrt(n)`, `INFERRED`); a measured `0.200` is only `≈1.7 SE` from zero, so a two-sided 95 % interval (`±t₄·SE = ±0.330`) covers zero | the **estimator's precision** (noise-limited) | reading `±0.2` as a resolved edge; also not reading it as "zero" — statistical indistinguishability is not an algebraic zero (§3.1, and `IDENTIFIABILITY.md` §1) |
| KS↔SI `0.33308 ± 0.00017`, 5/5, LOSO `0.286 ± 0.012` | the **geometry** (degeneracy-limited: the trace-1 cone boundary `Omega* = (1/3) 11^T`, F5/F7) | reading the stability as pair-specific information: at the boundary all three entries are equal by construction (`MSSL_SPARSITY_ANALYSIS.md` §3.3) |
| 16 L SI↔ER `-0.256 ± 0.076`, 5/5 negative, while 25 L is `+0.200 ± 0.266` | the **representation/axis** (§3.2) | any task-intrinsic sign claim (F5/F6: "relation strength and confidence are different quantities") |

So: a *stable* edge can be uninformative (saturation), an *unstable* edge can be
unresolvable (noise), and neither is a statement about KS/SI/ER as tasks. This is F6/F7 and
`FRAMEWORK.md` §1 rows 2–3 and row 8, restated in the terms of §1–§3 above.

### 5.5 What the numbers do not say  `OBSERVED`

No measured entry is exactly zero, no stability-selection procedure was run for the reported
screen (DEC-0016 fixes `λ₂ = 0.01` with no selection grid), and the ER-involving entries are
fold-sensitive. Consequently the repository holds **no** measured example of the §1
phenomenon (a nonzero covariance with a zero conditional edge) and **no** measured example
of its opposite (a zero covariance with a nonzero conditional edge). Both remain
*algebraically* available and *empirically* unobserved.

---

## 6. Crosslinks

* [`IDENTIFIABILITY.md`](./IDENTIFIABILITY.md) — the statistical companion: what an `m = 3`
  study can identify, the `m = 3` limits, and the descriptive / inferential-weak / invalid
  classification of relation claims.
* [`MSSL_SPARSITY_ANALYSIS.md`](./MSSL_SPARSITY_ANALYSIS.md) — derivation of `(C)`/`(D)`, the
  Schur complement `(S)`, the equicorrelation and saturation results, the graphical-lasso
  threshold table, and predictions P1–P5 used above.
* [`THREE_TASK_ANALYSIS.md`](./THREE_TASK_ANALYSIS.md) — `3 vs 6` parameter counts, support
  counts (8/64/25/5), rank geometry, `d` vs effective sample size.
* [`TASK_RELATION_TAXONOMY.md`](./TASK_RELATION_TAXONOMY.md) — dimensions D1/D2/D3/D6/D8 and
  the six-encoding comparison table this section's §2 mirrors.
* [`ASYMMETRIC_RELATIONS.md`](./ASYMMETRIC_RELATIONS.md) — the directed family (objects (c)
  and (d)), the survey's symmetrisation caveat, and the §8 design that a transfer claim
  `(e)` would require.
* [`OPEN_QUESTIONS.md`](./OPEN_QUESTIONS.md) — question 1 (independent samples), 3 (shared
  trunk), 4 (unique causal meaning of a directed coefficient), 8 (signed precision vs
  negative transfer).
* Repository: [`VARIANT_BENCHMARK_PROTOCOL.md`](../../../../improvements/taskrelation/research/VARIANT_BENCHMARK_PROTOCOL.md),
  [`FINDINGS.md`](../../../../improvements/taskrelation/research/FINDINGS.md) F5–F10,
  [`FRAMEWORK.md`](../../../../improvements/taskrelation/research/FRAMEWORK.md) §1–§3,
  [`studies/TR-0007/PLAN.md`](../../../../improvements/taskrelation/research/studies/TR-0007/PLAN.md),
  [`studies/DG-0001/analysis.md`](../../../../improvements/taskrelation/research/studies/DG-0001/analysis.md),
  `task_relations/MTRL_DIAGNOSTIC_SYNTHESIS.md`,
  [`DECISIONS.md`](../../../../improvements/taskrelation/research/DECISIONS.md) DEC-0013/0014/0015/0016.

## 7. Non-claims

* No winner, no ranking, no candidate endorsed, no study registered or authorised.
* No primary PDF was reopened for this note; all source claims are labelled with the existing
  ledger they come from (see the provenance paragraph at the top of this file).
* The §1 matrix is an illustrative covariance, not a measured one. The `30°`-rotation and
  `SD/sqrt(n)` numbers are `INFERRED` from it and from the recorded dispersions.
* Nothing here re-opens or re-specifies the protocols of `TR-0007`, `DG-0007` or `TR-0008`.
