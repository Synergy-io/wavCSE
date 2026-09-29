# TR-0007 Run Note

Status: PRE-REGISTRATION BLOCKED — no code committed, no run, no MLflow run
Type: mechanism (published variant; family B of DEC-0013/LT-0002)
Created: 2026-09-29
Research family: Task Relation Learning
Authorization: DEC-0013 (published-variant benchmark), DEC-0014 §5 (this arm is the published one)
Protocol: `../../VARIANT_BENCHMARK_PROTOCOL.md`
Compute: none. No authorization envelope exists for this scope.

## Hypothesis

Under the shared protocol, p-MSSL's sparse task **precision** Ω (graphical lasso on the
task-parameter summary, `−d log|Ω| + λ₂‖Ω‖₁`) changes the learned relation object enough to
separate the candidate from both controls: it must beat classical symmetric MTRL (the
in-category control) **and** the matched wavCSE baseline, per-task and aggregate, without
material regression on any task.

Competing explanation stated before any run: the same comparison can be positive for the
*summary adapter* rather than for the estimator (both arms consume the same mean-head
summary), so a win is only attributable to the Ω estimator when both controls are matched
and the Δ is larger than the seed variability F1 already documented.

## Current state of the arm

Implementation exists as an unregistered draft at `improvements/taskrelation/04-mssl/`
(`mssl_model.py`, `mssl_trainer.py`, commit `05fa10c`). It is **not runnable as a recorded
job**: there is no `mssl_config.yml`, and `improvements/run_improvements.py` has no `mssl`
model/trainer branch or `CONFIG_PATH_OVERRIDES` entry, so `--model mssl` cannot load. That
wiring is mechanical and is not what this gate is about.

The solver itself is sound: an independent check in this cycle's analysis
(`graphical_lasso_admm` at λ₂=5 on a 4×4 covariance) drives every off-diagonal below
6.5e-08 with a positive-definite result, and λ₂=0 reproduces S⁻¹. The three failures of
`research/tests/test_mssl_omega_solver.py` are test-side: an analytic-gradient formula
written transposed (`summary @ omega` instead of `omega @ summary`), and two assertions
demanding bit-exact float32 zeros (one of which also counts the diagonal it subtracted).
None of them reports a solver defect.

## What is unresolved — and why it needs the researcher

The draft solves a **different Ω problem from the published, verified one**:

| Source | Ω step |
|---|---|
| `literature/goncalves-2016-mssl.md` (JMLR 17(33) Eq. 8, verified against the PDF; LT-0002 §2) | `min_{Ω≻0} tr(SΩ) − d log|Ω| + λ₂‖Ω‖₁`, `S = (1/d)WᵀW` |
| `04-mssl/mssl_model.py::graphical_lasso_admm` (+ its docstring) | `min_{Ω>0} λ₀·tr(SΩ) − log|Ω| + λ₂‖Ω‖₁`, `S = WWᵀ/d`, ℓ₁ on off-diagonals only |

The two differ by a factor `d` on the data term **relative to the barrier**, with
`d = 2001` in the summary adapter, so they have different minimisers at the same λ₂ and
different sparsity at the same λ₂. λ₂ is therefore not comparable between them, and the
paper's headline property (a bounded, regularised precision that does not saturate, F7)
acquires a normalisation choice the source does not leave open. The draft's own text
contradicts itself: the module docstring states the objective includes `−d log|Ω|` and that
the paper's `1/n_k` scaling is satisfied by construction, while the solver's comment says
"the 1/d factor the paper folds into lambda_2 (Eq. 8) is applied by the caller". The
currently passing `test_zero_penalty_recovers_the_inverse_covariance` pins Ω = S⁻¹, i.e. the
code's barrier, so the tests encode the deviation rather than detecting it.

This is a faithfulness/deviation question, not a repair: DEC-0013 §3 requires faithfulness
to the source to be checked before screening, DEC-0014 §4 keeps MSSL's
"faithful-implementation" status, and both admissible fixes are scientifically different
(they are different estimators at the same λ₂). `.agents/policies/autonomy.md` classifies
"decide a faithfulness/deviation question" and "choose between scientifically different
fixes for the same failure" as HUMAN_DECISION. The tests cannot be "made green" first: the
λ₂ magnitudes they assert pin the very convention under decision.

---

## Human gate

    HUMAN_DECISION_REQUIRED
    Scope / Study / class: TR-0007 (MSSL, published arm) / faithfulness-deviation decision —
      autonomy policy HUMAN_DECISION ("decide a faithfulness/deviation question"; "choose
      between scientifically different fixes for the same failure").
    Blocking fact: the committed draft's Ω step solves a rescaled graphical lasso
      (`− log|Ω|`, off-diagonal ℓ₁, S = WWᵀ/d) instead of the verified published Eq. 8
      (`− d log|Ω|`, λ₂‖Ω‖₁, S = (1/d)WᵀW). With d = 2001 the two have different minimisers
      and different sparsity at the same λ₂, so the arm's λ₂ (and thus the mechanism's
      strength) cannot be pre-registered without choosing a convention — nor can the
      solver tests be corrected, since they pin the convention.
    Why autonomous continuation is forbidden: DEC-0013 §3 / DEC-0014 §4 require a
      faithful implementation for this arm; choosing between the published normalisation
      and a declared deviation changes what a result means (external validity), and the
      autonomy policy reserves that class for the researcher. No run may be pre-registered
      on an undecided mechanism definition.
    Option A — faithful as published: implement Eq. 8 exactly (barrier `d log|Ω|`,
      `S = (1/d)WᵀW`, ℓ₁ as the paper's cited graphical lasso defines it), re-derive λ₀/λ₂
      in that normalisation from the paper's defaults (stability selection range) or the
      protocol's pre-declared tiny validation grid, and rewrite the solver tests to that
      convention. Scientific consequence: the arm keeps faithful-implementation status and
      a result speaks to p-MSSL as published; the summary adapter remains the one declared
      deviation, shared with the control. Compute consequence: unchanged (~1.8 GPU-h for
      the three-run screen; same envelope as Option B).
    Option B — declared normalisation deviation: keep the draft's `− log|Ω|` /
      off-diagonal-ℓ₁ parameterisation, declare in `PLAN.md` that it is Eq. 8 rescaled by
      1/d (so λ₂ is *not* the paper's λ₂), fix the test-side defects only, and pre-register
      λ₀/λ₂ in that parameterisation. Scientific consequence: the arm becomes "published
      mechanism + declared scaling deviation" and is labelled as such in every table
      (beside the summary adapter); a null result cannot separate the mechanism from the
      deviation, and no statement about p-MSSL as published can be made. Compute
      consequence: unchanged (~1.8 GPU-h).
    Option C — defer this arm: pre-register `TR-0008` (directed relation, project-original,
      activated by DEC-0014 §5) and return to TR-0007 afterwards. Its ingredients are named
      by DEC-0014 §3 (loss-scaled sparsity, non-negativity, directed rows attributable to
      Lee et al. 2016) but the exact rule is design work the policy classes as introducing a
      project-original mechanism, so it needs either an explicit authorization to design it
      autonomously or a design review. Compute consequence: a further compute envelope for
      TR-0008 (same order as TR-0007).
    Current paid resources and estimated spend / remaining authorization: one pre-existing
      RunPod network volume `wavcse-vol-cache-85bcc73be71f` (200 GB STANDARD, EU-RO-1,
      AVAILABLE, provider id `9y5t57z98h`) which bills storage only. Zero workers, zero
      jobs in flight, no leases, no authorization for any scope: `authorizations/TR-0007.yaml`
      does not exist, so paid work stays blocked in every option.

## Next step

On a decision: write `PLAN.md`, the arm config and `compute/plan.json` (screen: one seed,
three arms — MSSL, MTRL control, wavCSE baseline — under the shared protocol), fix the
solver tests to the chosen convention, commit, then publish the exact commit (it is
currently local-only) before requesting the compute envelope.
---

# 2026-09-29 — Human gate answered: OPTION A (faithful published formulation)

Appended, not substituted: the pre-registration gate above stays on record as the state the
Study was in when the decision was taken.

## Decision (human, DEC-0015)

**Option A — implement the published MSSL formulation faithfully.** The arm stays a
faithful published-method implementation (DEC-0013 §3, DEC-0014 §4); it is not a declared
deviation and not a project-original mechanism. Full contract and pre-registration:
`PLAN.md`.

## Correction to the equation the gate was stated against (primary source)

The gate table above quoted the published Ω step as
`min tr(SΩ) − d log|Ω| + λ₂‖Ω‖₁`. Re-read directly from the JMLR PDF
(https://jmlr.org/papers/volume17/15-215/15-215.pdf) on 2026-09-29, the paper writes:

* Eq. (4b): `f_W(Ω; X, Y, λ₀, λ₂) = λ₀ tr(W Ω Wᵀ) − d log|Ω| + λ₂ ‖Ω‖₁`
* Eq. (8): `min_{Ω≻0} λ₀ tr(S Ω) − log|Ω| + (λ₂/d) ‖Ω‖₁`, `S = (1/d) WᵀW`, followed by
  "As λ₂ is a user defined parameter, the factor 1/d can be incorporated into λ₂."
* Algorithm 1: "Input: λ₀, λ₁, λ₂ > 0. // penalty parameters chosen by cross-validation";
  §4.1: "The parameter λ₀ was set to one in all experiments."

So the gate's rendering was a mis-transcription: it mixed Eq. (4b)'s `−d log|Ω|` barrier with
Eq. (8)'s data term `tr(SΩ)`, which is a *third* convention (Eq. (8) with λ₀ = 1/d). The two
faithful renderings — Eq. (4b) (`−d log|Ω|`, data term `tr(WΩWᵀ)`) and Eq. (8)
(`tr(SΩ) − log|Ω| + (λ₂/d)‖Ω‖₁`, `S = (1/d)WᵀW`) — are the same estimator and are exactly the
ingredients Option A names; the correct transcription is now in
`../../literature/goncalves-2016-mssl.md`. The literature card's "Eq. 8" line carried the
same error and is corrected there; the historical text in this NOTE is left intact.

## What the gate's substantive point was, and how it is now settled

The draft's solver objective (`λ₀ tr(SΩ) − log|Ω| + λ₂‖Ω‖₁`, λ₀ = 1) *is* Eq. (8) with the
`1/d` absorbed into λ₂ — the paper's own stated freedom — so it was never a different
estimator family, only a different convention for the numeric λ₂ (and therefore for its
scale, and thus for the strength of the mechanism). Option A fixes the convention: `λ₂` is
the paper's Eq. (3) penalty, the `1/d` of Eq. (8) is applied by the solver, and `d =
W.shape[1]` is an explicit argument. The implementation now also:

* returns the ADMM's split variable `Z`, which carries the ℓ₁ support exactly;
* certifies each solve with the problem's own primal–dual optimality certificate;
* rescales nothing, and adapts ρ by Boyd et al. (2011) §3.4.1 — the reference the paper
  cites for the ADMM derivation — with float64 arithmetic, because the mean-head summary
  enters Eq. (8) at scale ~1e-4 and a fixed ρ silently fails to converge there
  (measured: the old solver returned Ω ≈ 63·I where the optimum is Ω ≈ 7e4·I).

## λ₂ — the one input the paper does not fix (still open, still a human decision)

The paper publishes λ₀ = 1 and selects λ₁, λ₂ on data (stability selection for its regression
experiments; cross-validation over `{0.01, 0.1, 1, 10, 100}` for its classification
experiments). No default λ₂ exists in the paper or in this repository's record, and its
numeric scale is not transferable between representations. `PLAN.md` pre-registers the
recommended resolution (select λ₂ on the validation split from the paper's grid restricted to
`{0.01, 0.1}`, budget-matched to the control's two-value λ selection) and states the
fixed-value alternative; the choice changes the screen's run count from three to five, so it
is put to the researcher rather than assumed.

## Evidence in the tracking record that this branch did not know about

On 2026-09-29 the DagsHub MLflow repository was inspected read-only (zero-cost
authentication check, no run created). It already contains TR-0007 evidence from an earlier
code line that is not in this repository's history (commits `10aaaea3…`, `3df542d…`; all runs
in experiment `taskrelation-variant-benchmark`):

| Run | Seed | Result (`test_epoch_acc_all`) |
|---|---|---|
| `TR-0007__screen__p-mssl__…` | 0 | 0.9607 (KS 0.9867 / SI 0.9519 / ER 0.7703) |
| `TR-0007__screen__classical-mtrl__…` | 0 | 0.9744 (KS 0.9843 / SI 0.9789 / ER 0.7848) |
| `TR-0007__screen__wavcse-baseline__…` | 0 | 0.9737 (KS 0.9845 / SI 0.9771 / ER 0.7884) |
| `TR-0007__scale-corrected__p-mssl-correlation__…` | 0 | 0.9644 (KS 0.9855 / SI 0.9599 / ER 0.7703) |

All four carry `status: rejected`. They are retained as evidence (never deleted, per the
record policy), they are *not* this Study's runs, and they used a different implementation
(the params show a `covariance_normalization` knob this repository's arm does not have) at
seed 0 rather than the registered screen seed 42. Recorded here so the earlier negative
screen cannot be rediscovered as new, and so the human decides whether the corrected
implementation still warrants its own screen.

---

# 2026-09-29 — Screen authorized with a researcher-fixed lambda_2 (DEC-0016)

Appended. The pre-registration record above stays as the state the Study was in before paid
execution.

**Human decision (DEC-0016).** The screen runs `lambda_2 = 0.01`, `lambda_0 = 1.0`, with **no**
validation-selection grid, to keep the initial diagnostic screen small and inexpensive. The
value is a **researcher-fixed screening value**: the source paper selects `lambda_1`/`lambda_2`
by cross-validation (Algorithm 1) and publishes neither a value nor a transferable scale, so
`0.01` must never be described as prescribed by the paper, and a `REJECTED` screen reads "p-MSSL
did not help at this fixed lambda_2", not "p-MSSL cannot help". The label travels with the run:
`mssl.lambda_2_selection = researcher-fixed` is logged as an MLflow parameter (the config is
flattened into params), and the plan's arm labels carry it too.

**Scope of the authorization (screen only).** Three arms at seed 42 under the frozen protocol
(p-MSSL, classical MTRL, matched wavCSE baseline; all 25 layers; `smp` 0.5; 30 epochs; batch
2048; AdamW lr 0.0025). Bounds: 0.80 USD/GPU-hour, 5.00 USD total, 6 paid wall-clock hours, one
worker, replacements allowed, the existing network volume allowed, no new persistent resources,
60 GB container disk, destroy on completion. Confirmation, any lambda grid, other experiments,
`TR-0008`, `DG-0007` and new persistent resources are **not** authorized. Nothing is tuned
during the screen.

**Compute choice.** Provider inventory and prices were refreshed at execution time. The
cheapest compatible offer inside the network volume's datacenter was selected on expected total
cost, not on the highest hourly price or on a previously used GPU model; the run reuses one
worker for all three arms.

**Historical runs remain historical.** The `TR-0007__screen__*` runs already in DagsHub come
from code lines absent from this repository's history and are retained, unrelabelled and
unmodified, as historical evidence only. They are not substituted for any arm of this screen,
and their provenance limitation (unknown commit line, seed 0, a different implementation with
its own `covariance_normalization` knob, no `lambda_2_selection` tag) is stated wherever they
are cited. This screen's runs are identifiable by their exact git SHA, `study_id`, `stage`,
`seed`, `lambda_2`, `lambda_2_selection`, layer policy and protocol version.
