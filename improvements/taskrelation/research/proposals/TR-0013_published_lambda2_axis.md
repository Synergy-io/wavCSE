# TR-0013 — pre-registration DRAFT: the published `λ₂` axis, before any deviation

**Status:** `DRAFT — NOT REGISTERED`. Allocated identifier `TR-0013` for drafting only: no
`STUDIES.jsonl` entry, no `studies/TR-0013/` folder, no authorization envelope, no config, no run.
Registration needs a compute envelope for the screen (the arm's own documentation already
pre-registers the question, and the arm stays faithful, so no faithfulness decision is required for
this one).

**Written:** 2026-09-29, against canonical `90b73b2`.
**Predecessor:** `TR-0007` (`REJECTED`, `FL-0005`), its post-hoc scale reading
(`studies/TR-0007/analysis.md` §2026-09-29), `literature_survey/POST_TR0007_SYNTHESIS.md` §8 S2.
**Question source:** the arm's own `04-mssl/README.md` deviation 3 and `studies/TR-0007/PLAN.md`,
both of which pre-register "validation-selected `λ₂` over the paper's two smallest grid values" as
an open human item, and `DEC-0015` §3, which records that λ₂ has no published default.

---

## 1. Observation motivating the study

`OBSERVED`: the screen ran one researcher-fixed value, `λ₂ = 0.01` (`DEC-0016`), and its
`REJECTED` classification is scoped to that value by its own caveats. The source paper selects
`λ₂` on data (Algorithm 1, "chosen by cross-validation"; grid `{0.01, 0.1, 1, 10, 100}`,
JMLR §4.1) and publishes no transferable default, so the screen tested neither the paper's
procedure nor the hypothesis that the penalty *value* was the defect.

`DERIVED` (`studies/TR-0007/coupling_scale_result.json`; exact Eq. (8) solves at the three measured
Grams): raising `λ₂` does not move the coupling's magnitude. `λ₂ = 0.01` → 0 exact off-diagonal
zeros and gradient `1.6e5`; `0.1` → 0 zeros, `5.6e4`; `≥ 1` → 3 zeros, `6.9e3`. On the uncoupled
baseline's Gram the same values give `7 240`, `7 029`, `6 942` with 0/2/3 zeros. So every published
value leaves the coupling term at `≥ 7e3` against recorded task gradients of `≈ 1e-2`, and the value
that removes all conditional edges still carries `3d` on its diagonal.

## 2. Falsifiable hypothesis, competing explanations, third outcome

**H-λ (the faithful explanation, tested first).** A validation-selected `λ₂` from the paper's grid
restores a resolvable run: either it removes the harmful edge, or its shrinkage reduces the
precision enough that the coupling stops dominating the task loss, and the run's outcome becomes
attributable to the relation object rather than to the regularizer.

**Competing explanation H-scale (primary, this synthesis).** `λ₂` cannot: the coupling's magnitude
is `λ₂`-invariant for every value that retains an edge, and the values that shrink the precision to
its diagonal still impose `3d`. Under H-scale the arm closes as a *faithful* negative result, and
that is what justifies a scale-convention deviation (`TR-0012`) instead of more `λ`.

**Pre-declared third outcome.** Both values `{0.01, 0.1}` produce the same collapse *and* the same
support (0 zeros), i.e. the validation signal cannot discriminate them at all — in which case the
`λ₂` "selection" is vacuous at this scale, and that fact is the study's result.

## 3. Independent variable and controls

**Changed variable:** the `λ₂` *policy* — researcher-fixed `0.01` (as screened) → validation-selected
over the paper's two smallest grid values `{0.01, 0.1}`, budget-matched to the in-category control's
own two-value selection (`mtrl_lambda` `0.01` vs `0.05`). Selection is on **validation** only;
test is read once, after the freeze.

**Held constant:** the arm stays a faithful published implementation — `λ₀ = 1` (both half-steps),
the paper's ADMM with residual balancing, float64, off-diagonal ℓ₁, `d = W.shape[1]` applied inside
the solver, the sum-of-splits variable `Z` with exact zeros, the optimality certificate, `λ₁ = 0`,
the shared class-mean summary adapter, and everything in `POST_TR0007_SYNTHESIS.md` §7 (frozen
WavLM Large, `smp(0.5)`, all 25 layers, `ks_si_er`, shared `1024→512→2000` trunk, 30 epochs, batch
2048, AdamW `2.5e-3`, `≈ 2820` steps, epoch checkpoint, sampler and composition, splits, metrics).

**Controls:** the historical classical MTRL arm (retained by `DEC-0017`) and the matched wavCSE
baseline, at the same commit and protocol.

## 4. Cheapest adequate experiment

* **Stage `screen`, seed 42:** candidate at `λ₂ = 0.01` and candidate at `λ₂ = 0.1`, plus the two
  controls = **5 runs**, one worker, `≈ 1.5-1.8` GPU-h. Both candidate runs are recorded even though
  only the validation-selected one enters the classification (that is what "record every attempted
  configuration" means).
* **Instrumentation (in-run, recorded per epoch):** the same bundle S1 defines — summary cosines and
  Gram spectrum, `Ω` trace/support/partial correlations, the ADMM certificate, the coupling value
  beside the task losses, and the coupling-to-task gradient ratio.
* **Decision rule:** the family's pre-registered one, unchanged — `PROMISING` iff the selected
  candidate exceeds both controls on `test_epoch_acc_all` with no per-task regression beyond
  `0.20pp`; else `REJECTED`; `INCONCLUSIVE` if an arm cannot run under the frozen §1 settings or an
  exposure gate fails. A screen cannot promote (F1).
* **Not in this study:** `λ₂ ∈ {1, 10, 100}` (predicted to remove all three conditional edges —
  a different hypothesis, H-inert, which `DERIVED` already states); any other hyperparameter; a
  scale-convention change (that is `TR-0012`); confirmation seeds or LOSO (separate, later, on the
  screen's own outcome).

## 5. Prediction and falsification

**Predicted (H-scale).** At `λ₂ = 0.1`: the summary cosines still exceed `0.9` within two epochs of
the coupling engaging; the coupling value stays `≈ 2·10³` per task; training still plateaus; the
support shrinks relative to `0.01` (predicted 2 of 3 edges gone on a healthy Gram, 0 of 3 on the
collapsed one); the outcome stays `REJECTED` or `INCONCLUSIVE` with the same SI regression.

**Falsification of H-scale (and of this synthesis's §2.3).** Any of: a `λ₂ ∈ {0.01, 0.1}` run whose
coupling-to-task gradient ratio stays below `10` through training; summary cosines staying below
`0.5`; a validation curve without the epoch-`≈ 10` plateau; or a support change accompanied by an
outcome that separates from both controls. Any of these reopens the faithful route and would make
`TR-0012`'s deviation unnecessary for the moment.

**Falsification of H-λ.** Both values reproduce the collapse and the plateau with the same support.

**Classification honesty.** Even a `PROMISING` screen here would be *screening-tier*; the promotion
bar (beats both controls across seeds `0-4` with per-task paired intervals, LOSO for any ER claim) is
a separate, explicitly authorized stage. ER on the ordinary split is not a result (F3).

## 6. Expected information gain, risks, cost

* **Information gain:** it closes the last *faithful* explanation of TR-0007's failure at the
  cheapest possible cost, and it is the pre-registered resolution the arm's own documentation has
  carried since before the screen ran. Its most likely outcome (both values fail identically) is
  exactly the evidence that makes a deviation-class intervention (`TR-0012`) scientifically
  necessary rather than convenient. Its least likely outcome (the axis contains a working value)
  would rescue the published arm without any deviation.
* **Risks.** At `λ₂ = 0.1` two things change at once (scale and support), so a null cannot separate
  "scale fix failed" from "the removed edge did not matter"; the instrumentation is what allows the
  separation and must be recorded. The two-value space cannot find an optimum even if one exists
  between the grid points, and extending it would be the broad `λ` sweep `SUCCESSOR_STUDIES.md` §7
  forbids.
* **Cost:** 5 runs, one worker, `≈ 1.5-1.8` GPU-h; a request of `0.80 USD/GPU-hour`, `3.00 USD` and
  a `4`-hour window covers it with margin. Compatible part `sm_89`/`sm_86`; the raw-dataset farm
  `/root/voice_dataset -> /workspace/cache/raw` must be built in setup, as TR-0007 and DG-0007 both
  had to do.
* **Provenance:** commit config + this pre-registration before launch; record the executed commit;
  runs named `TR-0013__screen__<method>__ks_si_er__smp25__s42` with `lambda_2` and
  `lambda_2_selection` tags; DagsHub run note; every attempted value recorded.

## 7. Category-boundary check

Unchanged from TR-0007: the arm is the published p-MSSL precision arm, a formal Task Relation
Learning mechanism (Zhang & Yang 2021 §2.4, family B). A `λ₂` policy change is a protocol correction
inside the published method, not a new mechanism, not loss weighting, gradient surgery, low-rank,
clustering or decomposition (`FRAMEWORK.md` R7, `DEC-0005` §3).

## 8. What this draft is not

Not a registration, not an authorization and not a λ sweep. It is one faithful, pre-registered,
cheap test of the last non-deviating explanation, written so that its falsification is as
informative as its success.
