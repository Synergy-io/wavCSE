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
