# MSSL sparsity analysis — precision vs covariance, the Schur complement, and the 3 × 3 case

**Scope.** Theory only. This document analyses the relation object of the published
p-MSSL arm (Gonçalves, Von Zuben & Banerjee 2016) against classical symmetric MTRL,
with the 3-task (`m = 3`) case worked out exactly. It is a source for a future
`TR-0007`-family write-up and for the shared benchmark
[`VARIANT_BENCHMARK_PROTOCOL.md`](../../../../improvements/taskrelation/research/VARIANT_BENCHMARK_PROTOCOL.md);
it does not decide a winner and does not authorise a study.

**Sources.**
*Primary, opened and checked:* Gonçalves, Von Zuben & Banerjee, “Multi-task Sparse
Structure Learning with Gaussian Copula Models”, *JMLR* 17(33):1–30, 2016 —
<https://jmlr.org/papers/volume17/15-215/15-215.pdf>, §§3.1–3.4, 3.6–3.7, 4.1 and
Eqs. (1)–(4b), (8)–(11). Zhang & Yang, “A Survey on Multi-Task Learning”,
arXiv:1707.08114**v3** (2021) — §2.4 (Task Relation Learning Approach) and §2.8
(Another Taxonomy for Regularized MTL Methods), Eqs. (20)–(25), (32), (33).

*Repository records:* cards
[`goncalves-2016-mssl.md`](../../../../improvements/taskrelation/research/literature/goncalves-2016-mssl.md)
and [`zhang-yang-2021-mtl-survey.md`](../../../../improvements/taskrelation/research/literature/zhang-yang-2021-mtl-survey.md);
arm [`04-mssl/README.md`](../../../../improvements/taskrelation/04-mssl/README.md);
pre-registration [`studies/TR-0007/PLAN.md`](../../../../improvements/taskrelation/research/studies/TR-0007/PLAN.md);
findings [`FINDINGS.md`](../../../../improvements/taskrelation/research/FINDINGS.md) F4–F10;
decisions [`DECISIONS.md`](../../../../improvements/taskrelation/research/DECISIONS.md)
DEC-0013/0014/0015/0016; the MTRL diagnostic
[`MTRL_DIAGNOSTIC_SYNTHESIS.md`](../../../../improvements/taskrelation/research/task_relations/MTRL_DIAGNOSTIC_SYNTHESIS.md).

**Epistemic key.** `OBSERVED` = read from a primary source or a repository artefact;
`INFERRED` = derived here, algebra shown; `HYPOTHESIZED` = predicted, falsifiable, not
yet measured. Source-verification gaps are flagged where they occur.

---

## 1. What each method actually couples — and what it does not

### 1.1 The survey's two unified regularizers  `OBSERVED`

Zhang & Yang v3 §2.8 give two unified objectives that differ only in which matrix
multiplies the task-parameter second moment:

* feature-covariance family (feature-based MTL), Eq. (32):
  `min L(W,b) + tr(Wᵀ Λ⁻¹ W) + f(Λ)`, with `W ~ MN(0, Λ⁻¹ ⊗ I)`;
* **task-relation family** (parameter-based MTL), Eq. (33):
  `min L(W,b) + λ₂ tr(W Ω⁻¹ Wᵀ) + g(Ω)`, with `W ~ MN(0, I ⊗ Ω)`,

where `W ∈ ℝ^{d×m}`, `Ω ∈ ℝ^{m×m}` is the **column covariance** (one column per task),
so “Ω is to model the task relations since Ω is the column covariance with each column
in W corresponding to a task” (§2.8, verbatim). §2.4 then lists the instantiations:
Eq. (20)–(21) = Zhang & Yeung MTRL (`Ω` covariance, ℓ₁ on `Ω` in the sparse variant);
Eq. (24) = asymmetric task similarity (`m × m`, `Ω_ii` largest, near-symmetric);
Eq. (25) = Lee et al. 2016 AMTL (`W ≈ WA`, `a_ij ≥ 0`).

### 1.2 Both arms couple a *precision-shaped* matrix to the same second moment  `INFERRED`

Let `w_k ∈ ℝ^d` be task `k`'s parameter vector and
`R̃ := WᵀW ∈ ℝ^{m×m}`, `R̃_{kl} = ⟨w_k, w_l⟩`, the task-parameter Gram matrix.
(In the repository's `[m, d]` layout the same object is `WWᵀ`.)

* Classical MTRL (survey Eq. 21) minimises `λ₂ tr(W Ω⁻¹ Wᵀ) = λ₂ ⟨Ω⁻¹, R̃⟩`:
  the **coupling matrix is the inverse of the learned covariance** `Ω`.
  In this repository `Ω` is the closed-form, trace-normalised moment estimator
  `Ω = (WᵀW)^{1/2} / tr((WᵀW)^{1/2})` with `tr Ω = 1`
  (`MTRL_DIAGNOSTIC_SYNTHESIS.md`, “What classical MTRL assumes”).
* p-MSSL (JMLR Eq. 3) assumes the rows `w^j ∈ ℝ^m` of `W` are `N(0, Σ)` and takes the
  **precision `Ω = Σ⁻¹` as the parameter**; its coupling is
  `λ₀ tr(W Ω Wᵀ) = λ₀ ⟨Ω, R̃⟩` (JMLR §3.3, p. 7; Eq. 4a).

So the honest contrast is **not** “covariance vs precision as the coupling direction” —
both terminate in a precision-like coupling matrix. The differences that matter are:

| | classical MTRL | p-MSSL |
|---|---|---|
| learned object | covariance `Ω`, `tr Ω = 1` | precision `Ω = Σ⁻¹` |
| coupling matrix applied to `R̃` | `Ω⁻¹` | `Ω` |
| estimator | closed-form moment `(WᵀW)^{1/2}` normalised — no likelihood | penalised MLE: graphical lasso |
| constraint shape | `Ω ⪰ 0` (PSD) | `Ω ≻ 0` (PD) with log-det barrier |
| edge selection | none (dense) | exact zeros possible via off-diagonal ℓ₁ |
| fixed scale knob | `Ω` fixed by trace normalisation | `λ₀`, `λ₂` on an objective that is not scale-free |

**Consequence.** A “precision instead of covariance” framing overstates the change. The
reparameterisation `Ω = Σ⁻¹` is a bijection on the PD cone and adds *no information*;
what the arm actually changes is the **estimator and the penalty geometry**, plus the
numerical conditioning of the estimate (the paper's own stated motivation: learning the
inverse covariance directly “tends to be more stable than computing covariance and then
inverting it”, JMLR §2.3, p. 4). §5 develops why this distinction decides what the arm
can and cannot show.

---

## 2. Conditional independence, exactly: the Schur complement

### 2.1 The identity  `OBSERVED`

MSSL §3.1 (p. 6) states the Gaussian graphical-model fact it builds on: for
`V ~ N(0, Σ)` with `Ω = Σ⁻¹`, `Ω_ij = 0` **iff** `V_i ⊥ V_j | V_{[m]\{i,j}}`. §3.3
repeats it: “`Ω_ij = 0` if and only if `w_i ⫫ w_j | W_{\{i,j\}}` … enforcing sparsity on
`Ω` will highlight the conditional independence among task parameters”. The partial
correlation is `ρ_{ij·r} = −Ω_ij / √(Ω_ii Ω_jj)` (MSSL §3.3; also the repository's arm
README, which reports this normalised form).

### 2.2 Derivation via the Schur complement  `INFERRED`

Partition `V = (V_i, V_j, V_r)` with `r = [m] \ {i,j}`. The conditional covariance of
`(V_i, V_j)` given `V_r` is the Schur complement of the `rr` block:

```
Σ_{ij|r} = Σ_{ij} − Σ_{i,r} Σ_{rr}⁻¹ Σ_{r,j}.                          (S)
```

Conditional independence `V_i ⊥ V_j | V_r` is exactly `(Σ_{ij|r})_{ij} = 0`. Expanding
the block inverse of `Σ` (or applying the standard result) gives the equivalent
statement on the precision, `Ω_ij = 0`, with the sign carried by the normalisation
`ρ_{ij·r} = −Ω_ij/√(Ω_ii Ω_jj)`. So (S) and the zero-pattern of `Ω` are two views of
one object; the second is what ℓ₁ can act on, because it is a **zero of a parameter**,
whereas (S) is a zero of a *nonlinear function* of the covariance.

### 2.3 The exact 3 × 3 closed form  `INFERRED` (numerically verified)

For `m = 3` write `Σ = [[Σ₁₁, Σ₁₂, Σ₁₃],[Σ₁₂, Σ₂₂, Σ₂₃],[Σ₁₃, Σ₂₃, Σ₃₃]]` and let
`{i, j, k} = {1, 2, 3}`. The adjugate (cofactor) form of the inverse gives

```
Ω_ij = ( Σ_ik Σ_jk − Σ_ij Σ_kk ) / det(Σ).                              (C)
```

Hence, at `m = 3`,

```
Ω_ij = 0   ⟺   Σ_ij = Σ_ik Σ_jk / Σ_kk,   k the third task.            (D)
```

Equations (C)–(D) are the whole conditional-independence content of the 3-task case.
Verified numerically on `Σ = [[2, .7, .3],[.7, 1.5, .5],[.3, .5, 1.2]]`:
(C) reproduces all three off-diagonals of `Σ⁻¹` to machine precision, and the Schur
route (S), `ρ_{12·3} = (Σ₁₂ − Σ₁₃Σ₂₃/Σ₃₃)/√(σ_{11|3} σ_{22|3}) = 0.36465`, equals
`−Ω₁₂/√(Ω₁₁Ω₂₂) = 0.36465`.

**Reading (D):** the third task “explains away” the marginal association. A large
marginal `Σ_ij` and a zero `Ω_ij` are entirely compatible — this is the confusion the
next section corrects.

### 2.4 Two directions the project must keep apart  `INFERRED`

* **marginal association** `Σ_ij ≠ 0` ⇏ dependence given the rest, and
  **conditional independence** `Ω_ij = 0` ⇏ `Σ_ij = 0`. In (D), `Ω_ij = 0` is a
  *coincidence* of the three marginal entries, not a small marginal correlation.
* In the repository's measured regime the learned-off-diagonal magnitudes saturate
  (§3.3), so the two objects can carry **opposite** qualitative readings: a saturating
  covariance can coexist with *bounded* partial correlations (§4).

---

## 3. The 3 × 3 feasibility geometry

### 3.1 Partial correlations live in a curved region  `INFERRED`

Normalise `Σ` to a correlation matrix `R` (unit diagonal), with
`ρ₁₂, ρ₁₃, ρ₂₃` the marginal correlations. Positive definiteness is

```
det R = 1 − ρ₁₂² − ρ₁₃² − ρ₂₃² + 2 ρ₁₂ ρ₁₃ ρ₂₃ > 0.                   (G)
```

This is the only coupling constraint among the three 3-task relation entries; the
feasible set is the interior of a curved region (the boundary `det R = 0` is rank-1
degeneracy). Everything an `m = 3` relation estimator can express lies in a
**3-dimensional** space (or 6 for an unnormalised covariance).

### 3.2 Equicorrelation: both objects have a closed form  `INFERRED` (verified)

Take the equicorrelated case `R = (1 − r) I + r **11**ᵀ` (all tasks exchangeable). Then

* covariance entries are `r` off-diagonal, `1` diagonal (this is the *input*);
* `Ω = Σ⁻¹` normalises to partial correlation
  `ρ = r / (1 + r)` — **bounded by 1/2** as `r → 1`, with `r > −1/2` from (G)
  (equicorrelation `det R = (1 − r)²(1 + 2r)`).

Numerically: `r = 0 → ρ = 0`, `r = 0.5 → ρ = 0.3333`, `r = 0.9 → ρ = 0.4737`,
`r = 0.999 → ρ = 0.4997`. So **an equicorrelated, near-degenerate task geometry maps to
an equicorrelated precision** — the same lack of pair-selectivity, only with a bounded
normalised value instead of a saturating raw one.

### 3.3 The MTRL saturation point, read geometrically  `INFERRED` (verified)

The repository's classical MTRL `Ω` is the trace-normalised `(WᵀW)^{1/2}`. For
`WᵀW = (1 − r)I + r**11**ᵀ` (unit-norm summary rows, `normalize_w: true`), the matrix
square root is `√(1−r) I + ((√(1+2r) − √(1−r))/3) **11**ᵀ`, whose trace normalisation
gives `Ω_ii = 1/3` for every `r` and `Ω_ij → 1/3` as `r → 1`. The limit is

```
Ω* = (1/3) **11**ᵀ,   tr Ω* = 1,   rank Ω* = 1.                        (S1)
```

Numerically: `r = 0.9 → Ω_off = 0.1962`; `r = 0.999 → 0.3157`;
`r = 1 − 10⁻¹² → 0.333333 = 1/3`. This reproduces the project's observed
`smp` 25 L value `0.33308 ± 0.00017` (F6) as **the rank-1 boundary of the trace-1 PSD
cone** — all three off-diagonals equal, no pair information (F5, F7, F9). `OBSERVED`
that the value occurs; `INFERRED` that it is the boundary (S1).

---

## 4. Why the precision does not saturate the way the covariance does

This is the sharpest correction the project needs before reading any MSSL result.

### 4.1 The log-det barrier forbids the degenerate endpoint  `INFERRED`

MSSL's `Ω`-step (JMLR Eq. 4b / Eq. 8, p. 9) carries `−log|Ω|`. Since `|Ω| = 1/|Σ|`,
this term `→ +∞` as `|Σ| → 0`. The MTRL endpoint (S1) is *exactly* `|Σ| = 0`
(`Ω*` has one zero eigenvalue, so `Σ* = Ω*⁻¹` does not exist and any near-boundary
covariance has a vanishing determinant). Therefore:

* the MTRL optimum is permitted to sit **on** the PSD boundary (its constraint is
  `Ω ⪰ 0`, survey Eq. 21), and in the measured regime it does (S1);
* the MSSL objective **diverges** there, so its optimum is **interior**;
  `Ω ≻ 0` forces every partial correlation strictly inside `(−1, 1)`.

**This is the correct and only precise sense of “does not saturate by construction”**
(arm README, “The relation object”). It is a statement about the *barrier*, not about
information.

### 4.2 The barrier does not manufacture pair-selectivity  `INFERRED`

A reparameterisation is information-preserving (§1.2). If the data second moment `R̃`
is genuinely near-equicorrelated, then by §3.2 the precision is near-equicorrelated
too; `−log|Ω|` prevents a *singular* estimate but cannot split three equal couplings
into informative ones. **Prediction P3 below is exactly this falsifiable claim.**
What the barrier buys is numerical: an interior, well-conditioned `Ω` instead of
inverting a near-singular `Ω_cov` (the paper's stability claim, JMLR §2.3).

### 4.3 What ℓ₁ on `Ω` actually does  `OBSERVED` + `INFERRED`

The paper writes `‖Ω‖₁` in Eqs. (3),(8) and gives element-wise soft thresholding
in Eq. (11), without explicitly excluding diagonal entries in those equations.
The repository's DEC-0015 specifies **off-diagonal-only** penalization by reference
to the graphical-lasso convention. Thus the exact selection thresholds below
describe that project interpretation; penalizing diagonal entries would change them.

Two consequences follow under the repository's off-diagonal-only interpretation:

1. zeroing is a **hard selection** of a conditional-independence edge, not a shrinkage
   of a correlation; the returned `Z` has exact zeros (arm README);
2. the ℓ₁ penalty acts on `Ω` in its **raw scale**, while the interpretable quantity is
   the normalised `ρ_{ij·k} = −Ω_ij/√(Ω_ii Ω_jj)`. The same `λ₂` therefore penalises
   edges unevenly when the diagonals differ, and the effective edge threshold is *not*
   a partial-correlation threshold. Any “sparsity” statement must say which scale.

At `m = 3` the selection space is therefore `2³ = 8` possible support patterns (§2 of
[`THREE_TASK_ANALYSIS.md`](./THREE_TASK_ANALYSIS.md)): exactly one is the full triangle
(dense), three are two-edge chains, three are single edges, and one is the empty graph.
An empty conditional graph leaves a diagonal penalty
`λ₀ Σ_k Ω_kk ‖w_k‖²`, still a per-task parameter regularizer. It does
**not** restore the matched baseline: the model still shares its trunk and
the diagonal penalty can change optimization. A two-edge chain is the support
pattern where the endpoints are conditionally independent *given* the center;
a one-edge graph instead isolates a third task.

---

## 5. λ policy for this representation

### 5.1 Eq. (8) is not scale-free, and the project's adapter fixes the scale  `INFERRED`

The repository feeds MSSL the **same mean-head-summary matrix** the classical control
uses: `W ∈ ℝ^{3×2001}`, each row `w_k = [mean(classifier weight rows); mean(bias)]`
with `normalize_w: true`, so `‖w_k‖ = 1` and `R̃_kk = 1` (`04-mssl/README.md`;
`studies/TR-0007/PLAN.md`, deviations; `d = 2001`). Then

```
S = (1/d) W Wᵀ = R̃ / d,      R̃ unit-diagonal.                          (A)
```

Substitute `Ω = d M` into the implemented `Ω`-step (JMLR Eq. 8):
`λ₀ tr(SΩ) − log|Ω| + (λ₂/d)‖Ω‖₁`. Since `tr(SΩ) = tr(R̃ M)`,
`log|dM| = 3 log d + log|M|` and `(λ₂/d)‖dM‖₁ = λ₂‖M‖₁`, the objective equals

```
λ₀ tr(R̃ M) − log|M| + λ₂ ‖M‖₁ + const(d),      M = Ω/d.               (B)
```

**(B) is the paper's graphical lasso on the unit-diagonal correlation-like matrix
`R̃`, with penalty `λ₂` exactly in the paper's standardised scale.** The `1/d` of
Eq. (8) and the natural precision scale `≈ d` cancel. Verified numerically: on `d =
2001`, `R̃ = [[1,.3,.2],[.3,1,.4],[.2,.4,1]]`, the Eq. (8) objective and (B) agree to
`3.6 × 10⁻¹⁵`, and the unpenalised optimum `M = R̃⁻¹`.

**Consequences for the λ policy.**

* `λ₂` is *directly* comparable to the paper's published classification grid
  `{0.01, 0.1, 1, 10, 100}` (JMLR §4.1 / Algorithm 1 text: parameters “chosen by
  cross-validation”) — **but only because `normalize_w` is on**. If the summary rows
  have squared norm `c²` instead of 1, the same change of variables gives
  `λ₀ tr((R̃/c²) M) − log|M| + (λ₂/c²)‖M‖₁`, i.e. an effective penalty `λ₂/c²`: the
  numeric `λ₂` is then no longer in the paper's standardised scale. This is a units
  trap of the same kind the arm module documents for the `1/d` factor, and it is why the
  adapter setting is part of the frozen protocol rather than an arm-local choice.
* `λ₀ = 1` is the paper's setting (“set to one in all experiments”, JMLR §4.1) and is
  held fixed in the arm. Because (B) is not scale-free and `λ₀`, `λ₂` trade off, the
  paper's own procedure (λ₀ fixed at 1; `λ₁`, `λ₂` selected per data set by CV /
  stability selection) is strictly more general than the frozen
  `(λ₀, λ₂) = (1, 0.01)` the screen uses.

### 5.2 What `λ₂ = 0.01` predicts  `HYPOTHESIZED`

`λ₂ = 0.01` is the smallest value of the paper's reported classification
grid, but it is researcher-fixed here rather than selected on validation
(DEC-0016). Whether this value zeros an edge depends on the **actual**
head-summary Gram and solver normalization. The classical MTRL `Ω` entries
near `1/3` (F6) are *not* measured p-MSSL Gram correlations, and cannot
by themselves set a graphical-lasso edge threshold. The illustrative
matrices below predict dense precision *for those matrices*, not for
the unseen actual TR-0007 screen. A dense returned precision would test
an estimator/penalty contrast without a sparse-support contrast;
it would not demonstrate that sparse conditional relations are harmful.

Solving the `m = 3` graphical lasso on the unit-diagonal `R̃` (representative, not the
project's unmeasured `R̃`; the `Ω`-step of (B), ADMM as in Eq. 9–11) gives the threshold
explicitly:

| `R̃` off-diagonals | `λ₂ = 0.001` | `0.01` | `0.05` | `0.1` | `0.2` | `0.25` | `0.33` | `0.5` |
|---|---|---|---|---|---|---|---|---|
| `(.3, .2, .4)` — exact zeros | 0/3 | **0/3** | 0/3 | 0/3 | 1/3 | 1/3 | 2/3 | 3/3 |
| `(.33, .25, .30)` — exact zeros | 0/3 | **0/3** | 0/3 | 0/3 | 0/3 | 1/3 | 3/3 | 3/3 |

(off-diagonal-only ℓ₁, as Friedman–Hastie–Tibshirani; ADMM as in JMLR Eqs. 9–11; sign
convention: `Ω` off-diagonals are negative where marginal association is explained away,
so `ρ_{ij·k} = −Ω_ij/√(Ω_iiΩ_jj) > 0`.) At `λ₂ = 0.01` the off-diagonals shrink only
marginally relative to the unpenalised MLE: for the first `R̃`,
`−0.290/−0.106/−0.449 → −0.280/−0.100/−0.435`. The transition to exact zeros happens at
`λ₂ ≈ 0.2–0.5` — the paper's next grid values.

*This is a prediction, not an observation.* It is grounded in the scale identity (B)
and in the illustrative `m = 3` solves above, not in a solved `Ω` from the arm (the
screen has not been read here). It is falsified if the returned `Ω` contains an exact
off-diagonal zero at `λ₂ = 0.01` across seeds.

*Context for the dense-vs-sparse choice (crosslink, not this file's claim).* The
taxonomy records a primary-source warning that sparse task relations are a many-task
strategy and can lose to dense ones at small `m` (SPATS, discussed in the
“m = 3 geometry constraints” section of
[`TASK_RELATION_TAXONOMY.md`](./TASK_RELATION_TAXONOMY.md)).
If that applies to KS/SI/ER, the pre-registered `λ₂ = 0.01` (dense `Ω`) is the *more*
reasonable regime to screen at `m = 3` — which sharpens, rather than weakens, the point
that the screen tests the dense precision reparameterisation and not the sparse graph.

---

## 6. Predictions and controls

### 6.1 Falsifiable predictions

| # | Prediction | Falsified if |
|---|---|---|
| **P1** | At `λ₂ = 0.01` (unit-normalised summaries, `d = 2001`) p-MSSL *may* retain all three conditional edges, as in the representative Gram calculations; the actual support must be read from the study. | Any exact zero disproves a dense-support prediction for that seed, not the conditional-independence model. |
| **P2** | Dense p-MSSL may still differ from classical MTRL, because log-det/trace constraints, diagonal precision scales and estimator gradients differ even without support zeros. A paired outcome and raw matrix comparison can distinguish these. | A claim that a dense graph alone proves equivalence is invalid regardless of outcome. |
| **P3** | If the summary Gram is genuinely near-equicorrelated and diagonals balanced, a symmetric graphical-lasso fit should have near-equal normalised partial correlations; unequal fitted diagonals or Gram entries can break the analogy. | Strong pair-differentiation under an actually equicorrelated Gram with a symmetric solver. |
| **P4** | Increasing `λ₂` can zero off-diagonal precision entries, but the threshold depends on the *actual* Gram and normalization, and empty support still leaves diagonal regularization plus shared-trunk training. | No zeros under a sufficiently large prospectively justified λ, or edge support changes without corresponding performance evidence. |
| **P5** | The log-det barrier enforces a positive-definite optimum for fixed finite input and penalty; it cannot *exactly* equal the rank-one MTRL limiting covariance. A poorly conditioned interior fit is still possible. | An exactly singular optimum with finite objective (would contradict the stated optimization problem). |

These are mathematical conditional predictions, not instructions for a λ sweep.
Relation diagnostics plus paired validation/LOSO and controls are required to
attribute any effect; no new architecture follows from them.

### 6.2 Controls, and the diagnostics this analysis requires

**Protocol controls (mandatory, §2 of the protocol):** classical symmetric MTRL
(`01-mtrl/mtrl_poolingwinner_25L_config.yml`) and the matched wavCSE baseline. A result
against MSSL alone is uninterpretable (F2, F4).

**Diagnostic controls this analysis adds** (all cheap, all from the same run):

1. **Report the raw Gram/correlation matrix `R̃` next to `Ω`.** Since `Ω = Σ⁻¹` is a
   reparameterisation (§1.2), the reader must see whether pair structure exists in
   `R̃` at all — without it P3 is untestable and a “MSSL found structure” claim is
   unfalsifiable.
2. **Report both scales of the relation object:** raw `Ω` *and* the normalised partial
   correlations `−Ω_ij/√(Ω_iiΩ_jj)`. The ℓ₁ acts on the raw scale (§4.3).
3. **Report the exact support** (which of the 3 edges are exactly zero) per epoch, not
   just the magnitudes, plus the solver’s optimality certificate (arm README: the
   solver is self-certifying; the returned `Z` has exact zeros).
4. **Report the saturation state** per protocol §4 and F7 — a saturated relation object
   carries no pair-specific information. Note for MSSL this means the *scale* of `Ω`
   and the equicorrelated-ness of the partial-correlation triple (§3.2), not the MTRL
   `±1/3` signature.
5. **Matched exposure and composition** (F8, F10): all arms share the sampler and the
   ≈ 2,820 steps; MSSL adds no epochs. This is a protocol condition, not optional.
6. **λ₂ labelling:** every table must carry `lambda_2_selection: researcher-fixed`
   (DEC-0016). A `REJECTED` screen reads “MSSL did not help at this fixed `λ₂`”, not
   “MSSL cannot help”.

**Not evidence** (each is a recorded project failure mode): a single-seed accuracy
delta (F1); an ordinary-split ER delta (~15 pp speaker leak, F3); any comparison with
unmatched pooling/layers (F2 — pooling alone moved ER by 3.25 pp, larger than any
relation effect measured); a saturated-object reading as relation structure (F7); a
gradient-scale rationale for a relation mechanism (F9 → F10, DEC-0010).

**Known source-verification gap (stated, not hidden).** This document has opened the
JMLR MSSL PDF and the arXiv v3 survey; it has *not* opened Friedman et al. (2008),
Banerjee et al. (2008) or Meinshausen & Bühlmann (2010), which are cited only as
MSSL's own citations for the Ω-step, the ADMM derivation and the stability-selection
procedure. The threshold behaviour used in P1/P4 is therefore attributed to the
primary source's citation, not independently verified against those papers.

---

## 7. Crosslinks

* [`THREE_TASK_ANALYSIS.md`](./THREE_TASK_ANALYSIS.md) — the `m = 3` parameter counts,
  rank and grouping geometry, and the statistical resolution of a 3-edge relation; the
  Schur complement (C)–(D) here is its conditional-independence primitive.
* [`TASK_RELATION_TAXONOMY.md`](./TASK_RELATION_TAXONOMY.md) — category boundary
  (`#category-test-zhang-and-yang-24`), relation-object dimensions
  (`#relation-object-dimensions`), and the conditional-vs-direct dependence distinction
  (`#conditional-vs-direct-dependence`) that §2 of this file instantiates.
* [`SURVEY_MAP.md`](./SURVEY_MAP.md) — survey §2.4 / §2.8 map and the MTRL-vs-MSSL
  lineage.
* Repository: [`VARIANT_BENCHMARK_PROTOCOL.md`](../../../../improvements/taskrelation/research/VARIANT_BENCHMARK_PROTOCOL.md)
  (fixed conditions and mandatory controls), [`studies/TR-0007/PLAN.md`](../../../../improvements/taskrelation/research/studies/TR-0007/PLAN.md)
  (pre-registration, λ₂ decision), [`04-mssl/README.md`](../../../../improvements/taskrelation/04-mssl/README.md)
  (implemented equations and declared deviations),
  [`FINDINGS.md`](../../../../improvements/taskrelation/research/FINDINGS.md) F4–F10,
  [`DECISIONS.md`](../../../../improvements/taskrelation/research/DECISIONS.md)
  DEC-0013/0014/0015/0016.
