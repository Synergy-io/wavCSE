# TR-0012 — pre-registration DRAFT: scale-commensurate relation coupling

**Status:** `DRAFT — NOT REGISTERED`. Allocated identifier `TR-0012` for drafting only: there is no
`STUDIES.jsonl` entry, no `studies/TR-0012/` folder, no authorization envelope, no config and no
run. Nothing here may be launched. Two gates precede registration:

1. a **human faithfulness/deviation decision** (the intervention is a declared deviation on a
   published arm; `.agents/policies/autonomy.md` reserves that class to the researcher, and
   `DEC-0015` closed the "do not invent a new normalisation, do not tune the formulation" question
   for the TR-0007 phase);
2. a **compute authorization** for the screen envelope, in the form `DEC-0018` used for DG-0007.

**Written:** 2026-09-29, against canonical `90b73b2`.
**Predecessor:** `TR-0007` (`REJECTED`, `FL-0005`), its post-hoc scale reading
(`studies/TR-0007/analysis.md` §2026-09-29) and `literature_survey/POST_TR0007_SYNTHESIS.md` §8 S1.
**Family:** Task Relation Learning, relation-estimator-imposition strength (published p-MSSL arm,
Gonçalves et al., JMLR 17(33), 2016).

---

## 1. Observation motivating the study

`OBSERVED` (`studies/TR-0007/{result.json,analysis.md,coupling_scale_result.json}`): the registered
p-MSSL screen ended `REJECTED` (p-MSSL `0.9662` vs classical MTRL `0.9752` and matched baseline
`0.9748`; SI `−1.79pp/−1.67pp`), and the mechanism was recorded rather than inferred — the summary
cosines jump to `≈ 0.981` at the epoch the coupling engages, `S` becomes near-singular, `Ω` jumps to
a near-uniform precision of trace `1.13e5`, the coupling term (`≈ 2015`) dominates the task losses
(`≈ 0.5`), and training plateaus from epoch `≈ 10`.

`DERIVED` (`coupling_scale_result.json`): the coupling's scale is not a `λ` problem. At the Ω step's
own optimum `tr(ΩR) = 3d`, so at the published `λ₀ = 1` the coupling carries `≈ 3d = 6003` in value
and `≈ 6.9e3` in gradient even for `Ω = d·I` (against a batch-mean loss of `≈ 0.5` and recorded task
gradients of `≈ 1e-2`), and this survives every published `λ₂` (`0.01` → gradient `1.6e5` with 0
exact zeros; `0.1` → `5.6e4` with 0 zeros; `≥ 1` → 3 zeros and still `6.9e3`) and every `λ₀` (at
`λ₀ = 1e-4`, gradient `21.7` — still `≈ 2e3`× the task scale — and all three edges already zero).

`OBSERVED` (same script): the collapse is a consequence, not a cause. The **uncoupled** matched
baseline ends the identical protocol with near-orthogonal summaries (`cos +0.211, +0.031, +0.023`,
Gram condition `1.54`), and the candidate's own pre-activation geometry (`cos +0.261, +0.035,
+0.009` at epoch 3) was in that same healthy band.

`RECORD`: the arm is a faithful published implementation whose declared deviations are the shared
mean-head summary adapter and `λ₁ = 0`. The scale of the penalty relative to the adapted loss is
**not** among its declared deviations, although the paper's `λ₀ = 1` presupposes its own
sum-normalised likelihood convention.

## 2. Falsifiable hypothesis, competing explanations, third outcome

**H1 (primary).** Imposing the *same* published relation object at a magnitude the optimizer can
weigh against the task loss removes the domination: the summary geometry stays in its uncoupled
band, the coupling's gradient contribution is comparable to the task loss's, and the run becomes a
readable test of the relation set instead of a training of the regularizer.

**Competing explanation H2 (stated before any run).** The relation object carries no useful
information for these three tasks at all: at *any* imposition strength the arm's outcome stays at
the controls, so a healthy-looking trajectory would still end in a null. (This is not the same claim
as H1's failure: H1 is about the mechanism, H2 about the usefulness.)

**Pre-declared third outcome H3.** The calibration is dominated by the *conditioning* rather than by
the summary length: with a scale-matched coupling the geometry stays healthy but the objective's
alternation becomes unstable (oscillating `Ω` support or diverging validation), i.e. the estimator
and the coupling interact badly even at matched scale. Pre-declared as a possible outcome, not as a
success.

## 3. Independent variable, and the deviation it carries

**The coupling term's magnitude policy — one scalar, measured, not tuned.** In the first epoch after
`warmup_epochs`, on the candidate arm's own first post-warmup step, measure
`G_task = ‖∂L_task/∂θ‖` and `G_coupling(1) = ‖∂(λ₀·tr(WΩWᵀ))/∂θ‖` over the shared parameters and the
classifier heads (`θ` unchanged from the protocol's parameter set), then set

```
lambda_0_coupling = rho* · G_task / G_coupling(1),      rho* = 1
```

and freeze it for the whole run. `rho*` is a single pre-registered value; the measured `G_task`,
`G_coupling(1)` and the resulting coefficient are logged in the run and in the run note. This is a
calibration of a *convention*, not a search: no second value is tried, and no outcome is consulted.

**The Ω step is untouched**: `lambda_0 = 1`, `lambda_2 = 0.01`, the paper's ADMM with residual
balancing, float64, off-diagonal ℓ₁, the optimality certificate, and the split variable `Z` with
exact zeros. The estimator and the *learned relation object* are therefore exactly the published
ones; only the strength with which the estimated relation is imposed on the parameters changes.

**Deviation declaration (mandatory, to be stated in the arm README before any run).** The W step and
the Ω step then use different `λ₀`; the published objective's alternation assumes one. The arm is
therefore **not** "p-MSSL as published" and must be labelled *published relation object, declared
scale-convention deviation* in every table, beside the adapter deviation. A null result cannot be
attributed to p-MSSL as published.

Alternative realisation the researcher may prefer (recorded here because the choice is theirs, and
their predictions differ): the **natural `1/d` normalization**, `lambda_0_coupling = 1/d = 1/2001`,
which makes the coupling's *value* `≈ m = 3` instead of `≈ 3d`. Its predicted coupling gradient is
`≈ 3.5` at a healthy geometry and `≈ 80` at the collapsed one — i.e. 350-8000× the task scale, so it
is predicted to be *insufficient* where `rho* = 1` is predicted to be commensurate. Choosing `1/d`
would test a weaker claim and would still need the deviation declaration.

## 4. Matched controls and held-constant factors

Held constant, per `VARIANT_BENCHMARK_PROTOCOL.md` §§1-5 and `POST_TR0007_SYNTHESIS.md` §7: frozen
WavLM Large embedding, mean frame pool, `smp(0.5)` over all 25 layers, `ks_si_er`, shared
`1024→512→2000` trunk, 30 epochs, batch 2048, AdamW `2.5e-3`, `≈ 2820` steps, epoch checkpoint,
sampling composition, the shared class-mean summary adapter (both MTRL arms), `λ₁ = 0`, `λ₂ = 0.01`
and `λ₀ = 1` **inside the Ω step**, the Ω cadence and warmup, the datasets and splits, the metrics
and thresholds, the seed treatment.

Controls: the **historical classical MTRL** arm (`mtrl_poolingwinner_25L_config.yml`, `normalize_w:
true`, retained as the in-category control by `DEC-0017`) and the **matched wavCSE baseline**. A
result against p-MSSL alone would be uninterpretable (F2, F4). No DG-0007 arm is used: it has no
result.

## 5. Cheapest adequate experiment

* **Stage `screen`, seed 42, three arms** (candidate, classical MTRL, matched baseline), one run per
  arm, `≈ 2820` steps, one worker reused for all three, the pre-existing network volume.
* **Instrumentation (in-run, zero extra cost, recorded per epoch and fed to the analysis):** the
  three summary cosines and the Gram's eigenvalues/condition; `Ω`'s trace, diagonal, exact support
  and partial correlations; the ADMM's certificate and iteration count; the coupling term's value
  next to each task's loss; and `G_task`/`G_coupling` at the calibration step and every 10 epochs
  afterwards.
* **Decision rule (pre-registered, unchanged from the family's convention):** `PROMISING` iff the
  candidate exceeds **both** controls on `test_epoch_acc_all` **and** every per-task accuracy is
  within `0.20pp` of both; else `REJECTED`; `INCONCLUSIVE` if the exposure gate or the calibration
  measurement fails (see §6). A screen cannot promote (F1).
* **Not in this study:** confirmation seeds, LOSO, any additional arm (including a coupling-absent
  control), any pooling/layer/epoch change, any `λ` grid. Confirmation seeds `0-4` and, for any ER
  claim, the ten speaker-independent LOSO folds are a *separate, later* authorization decided by the
  screen's own outcome and the protocol's promotion bar.

## 6. Success, rejection, falsification — the mechanism gates

The screen's classification comes from the accuracy rule above. The *hypothesis* is decided by three
pre-registered mechanism measurements, which are why this study is worth its compute either way:

| Gate | Pre-registered criterion | Verdict if it fails |
|---|---|---|
| **G1 calibration** | at the calibration step, `G_coupling/G_task ∈ [0.5, 2.0]` after the coefficient is applied; the measurement must be finite and non-degenerate | the study's own independent variable could not be set; `INCONCLUSIVE`, re-pre-register, no further arms |
| **G2 no collapse** | summary cosines `≤ 0.5` at every epoch after the coupling engages, and the summary Gram's smallest eigenvalue `≥ 0.1 ·` its largest | **H1 falsified**: the collapse is not (only) a scale effect |
| **G3 no domination** | the coupling's value stays within `10×` the total task loss, and `G_coupling/G_task ≤ 10` for at least `90%` of post-warmup epochs | **H1 falsified** in the sense that a scale-matched coupling still dominates |
| **G4 readable outcome** | the outcome is either `PROMISING` with no task regression, or a `REJECTED` in which G2/G3 held and the per-task picture is attributable | G2/G3 held but the outcome is uninterpretable → design error, recorded as such |

Falsification of **H1** (any of G2, G3 failing) sends the explanation back to the estimator or the
representation, and is the outcome that would most change the programme's direction. Confirmation of
G2/G3 with a `REJECTED` outcome is the first *interpretable* negative result for a learned relation
in this repository — evidence for H2, and the precondition for asking whether a different relation
object (directed, covariance-shaped, or a different summary) is worth testing at all.

**ER:** the ordinary split leaks speakers (F3); every ER statement from a screen is context only. No
ER claim without the ten-fold LOSO design in the operator skill.

## 7. Expected information gain, and why it beats the alternatives

It is the first design in which a learned relation is imposed at a strength that can be compared
with the task loss, so both of its outcomes are informative: a promotion (no relation arm in this
programme has ever produced one) or a clean, attributable null that licenses or closes the family.
Cheaper alternatives are already answered by arithmetic (a `λ` grid: `POST_TR0007_SYNTHESIS.md` §2.3,
and `TR-0013` as the training-side confirmation); more expensive ones (a representation arm, a gate,
a directed arm) are predicted ineffective or are confounded by the same scale defect
(`POST_TR0007_SYNTHESIS.md` §6).

## 8. Category-boundary check

The change is *how strongly a learned task-relation object is imposed on the task parameters* — the
Task Relation Learning family of Zhang & Yang (2021) §2.4, instantiated by the published p-MSSL
precision. It is **not** loss weighting, gradient surgery, low-rank/basis sharing, clustering or
decomposition: no per-task scalar weight is learned, no gradient is edited, no task partition or
factorisation is introduced. `FRAMEWORK.md` R7 and `DEC-0005` §3 hold.

## 9. Compute, risks, provenance

* **Stage-1 screen (the authorization request):** one worker, `≈ 1.8` GPU-h for three arms, plus
  provisioning; a request of `0.80 USD/GPU-hour`, `3.00 USD` total and a `4`-hour wall-clock window
  would cover it with the same margin `DEC-0018` granted. Compatible part required: `sm_89`/`sm_86`
  (the Blackwell `sm_120` failure of DG-0007's second attempt must not be repeated). A fresh worker
  needs the raw-dataset farm `/root/voice_dataset -> /workspace/cache/raw` built in setup, as
  TR-0007 and DG-0007 both had to do.
* **Risks.** (i) The deviation declaration makes the arm non-faithful, so the result speaks to the
  *relation-object-imposition* question, not to p-MSSL as published; (ii) the calibration is a single
  measurement on one seed — a noisy calibration would be caught by G1 rather than silently absorbed;
  (iii) if the collapse is partly a *feature* of the tasks, G2 could fail for reasons unrelated to
  scale (the falsification is designed to be able to say so).
* **Provenance.** Commit the arm config, the calibration code and this pre-registration **before**
  launching; record the executed commit in `STUDIES.jsonl`; one MLflow experiment
  (`taskrelation-mssl` or a new `taskrelation-tr0012` per the naming rule), run name
  `TR-0012__screen__<method>__ks_si_er__smp25__s42`, the standard tags plus
  `lambda_0_coupling`/`calibration_*`, and a DagsHub run note. Every attempted configuration is
  recorded, including failures.

## 10. What this draft is not

Not a registration, not an authorization, not a config, not a claim that the intervention works. It
is the proposal the human decision needs: the observation, the one variable, the deviation it
carries, the gates that can falsify it, and the cost.
