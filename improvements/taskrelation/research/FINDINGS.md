# Findings — Task Relation Learning

Canonical registry of established findings for this research branch. `STATE.md`
carries a one-line summary of each; **this file is authoritative** for the text,
the numbers and the provenance.

Read this before proposing anything. A finding here must not be rediscovered or
re-litigated as new. If you believe one is wrong, record the contradicting
evidence, open a diagnostic study (see `BACKLOG.md`), and update the entry with
its provenance — do not delete it.

## Status vocabulary

* **ESTABLISHED** — survived the strongest protocol the project has for that
  claim (multi-seed, speaker-independent LOSO, or structural evidence that
  replicates across folds/settings).
* **SCREENING** — single-seed / single-split evidence only. Cannot carry a
  research claim (see F1).
* **RECORD** — an observation about the project's experimental record itself
  (what has and has not been run), not a scientific claim about the tasks.

Provenance format: source doc / MLflow experiment / artifact holding the raw
numbers.

---

## F1 — Single-seed improvements are not trustworthy  (ESTABLISHED)

An apparent MTRL improvement (0.9728 vs 0.9724 at `smp`+16L) did not survive a
matched multi-seed check (5 seeds each, seeds 0–4, identical configs):

| | mean `test_opt_acc_all` | std | range |
| --- | --- | --- | --- |
| baseline | 0.97134 | 0.00130 | 0.96995 – 0.97372 |
| MTRL | 0.97124 | 0.00091 | 0.96969 – 0.97218 |

The mean difference (0.0001, baseline ahead) is an order of magnitude below
either side's seed-to-seed standard deviation. Before `improvements/seed_utils.py`
existed, identical configs differed by up to 2.35pp on `er`; that variance is
what produced and then retracted the win.

Consequences:

* one seed = screening evidence only;
* a single high score must never be described as a confirmed improvement;
* any comparison of a <1pp effect needs ≥5 seeds;
* promising candidates require multi-seed confirmation before promotion.

Provenance: `01-mtrl/README.md` (top-of-file retraction, 2026-09-01);
`CONTINUATION.md` §7 "Reproducibility" and §9.2(b); MLflow experiment
`taskrelation-mtrl`.

---

## F2 — Pooling is a major confound  (ESTABLISHED)

Pooling/representation choice produces effects comparable to or larger than
Task Relation Learning architecture changes. Merely switching the *baseline*
from `mix` to `lnp` moved it from all/er `0.9710 / 0.7559` to
`0.9730 / 0.7884` (+0.2pp overall, **+3.25pp on `er`**) — larger than any
MTRL-vs-baseline delta measured in the entire `01-mtrl` campaign.

Consequences:

* every architecture comparison must pin pooling, layer selection, data,
  epochs, optimizer settings, evaluation protocol and seed treatment;
* a comparison with unmatched pooling does not establish an architectural
  effect (this is why `02-lnp/` exists as a control);
* "N layers beats M layers" is not a general claim — it is pooling-specific
  (`mix`/`weighted`/`lnp` cannot exploit the extra 9 layers, `smp` can).

The durable positive result of this line of work is a *pooling* recommendation,
not an architecture win: `smp` (λ=0.5) beat every other pooling type at both
16L and 25L, on both architectures, without exception (grid: 18 param/type
combos per layer count, 10-epoch screening → top-3 confirmed at 30 epochs).
Two failure modes found: `lse` produces NaN losses (`exp(r·x)` overflow), and
`lnp` collapses at 25L regardless of power parameter.

Provenance: `02-lnp/README.md`; `../base/POOLING_GRID_SEARCH.md`;
`CONTINUATION.md` §9.3; MLflow experiments `wavcse-base-lnp`,
`wavcse-base-poolingsweep` (60 runs).

---

## F3 — ER's historical single split is speaker-leaky  (ESTABLISHED)

`downstream/dataset/load_embedding.py::_load_iemocap()` pools all 10 IEMOCAP
speakers and slices train/val/test by a fixed stride, so train and test share
speakers. At the best config (`smp`+25L) the leaky single-split `er` number is
0.7902; the leave-one-speaker-out mean is **0.6391 ± 0.0506** — roughly **15pp
inflation**, i.e. speaker memorization rather than emotion generalization.

Consequences:

* single-split `er` numbers may be used for cheap screening only, and must be
  labelled as screening evidence;
* the authoritative protocol for any ER claim is speaker-independent LOSO
  (`run_base_er_kfold.py`, `01-mtrl/mtrl_er_kfold.py`: 10 folds, 5 epochs/fold,
  test = speakers[i], val = speakers[(i+1) % 10], train = other 8);
* per-fold `er` std is ~5pp (range 0.556–0.718), so single-fold results are
  meaningless;
* the honest `er` ceiling for this architecture is ~64%, not ~79%.

Provenance: `CONTINUATION.md` §9.2(a); `../base/README.md` ("ER 10-fold
cross-validation"); MLflow experiments `wavcse-baseline-er-kfold` (27 runs),
`taskrelation-mtrl-er-kfold` (11 runs).

---

## F4 — Classical MTRL shows no reproducible advantage over matched wavCSE  (ESTABLISHED)

**Observation.** Classical MTRL has no meaningful reproducible aggregate
advantage at `smp` 16L or 25L and no resolved speaker-independent ER effect.
Task-wise five-seed deltas are small except for a 25L ER regression confined to
the speaker-leaky split.

**Evidence.**

| Matched protocol | KS Δ | SI Δ | ER Δ | Aggregate Δ |
| --- | ---: | ---: | ---: | ---: |
| `smp` 16L, seeds 0–4 | +0.00059 | −0.00075 | +0.00108 | −0.00010 |
| `smp` 25L, seeds 0–4 | −0.00079 | +0.00029 | −0.01049 | −0.00056 |

Every 16L paired 95% interval includes zero. At 25L, all intervals include zero
except ordinary-split ER [−0.01910, −0.00188]; that split is speaker-leaky and
the stronger LOSO protocol contradicts a general ER effect:

| ER LOSO tag, 10 matched folds | baseline | MTRL | Δ | Paired 95% CI |
| --- | ---: | ---: | ---: | --- |
| `opt` | 0.6391 | 0.6380 | −0.0011 | [−0.0307, +0.0285] |
| `best` | 0.6307 | 0.6423 | +0.0116 | [−0.0125, +0.0358] |
| `epoch` | 0.6278 | 0.6276 | −0.0002 | [−0.0303, +0.0299] |

**Interpretation.** MTRL is outcome-neutral within the resolution of the
completed protocols. The sample-weighted aggregate does not conceal a stable KS
or SI gain, and LOSO does not support either ER benefit or harm.

**Alternative explanations.** Five seeds may not resolve effects below about
0.1 percentage points, and LOSO uses one training realization per fold with a
five-epoch budget. Those limitations cannot support an improvement claim; they
only bound the size currently detectable.

**Confidence.** Strong for “no meaningful demonstrated advantage”; not evidence
of exact equivalence.

**Implications.** MTRL remains the formal baseline method (DEC-0001), not the
champion. Generic retuning is forbidden: λ=0.05 was worse than 0.01 on every
task. Diagnose the modelling assumption before selecting a replacement.

**Required follow-up.** DG-0001 must measure directed empirical transfer;
DG-0003 or conditional DG-0002 then separates relation representation,
estimation and optimization failure.

Provenance: `01-mtrl/README.md`; `CONTINUATION.md` §9.1–9.2; MLflow
experiments `taskrelation-mtrl`, `wavcse-baseline`,
`taskrelation-mtrl-er-kfold`, `wavcse-baseline-er-kfold`; paired analysis in
`research/task_relations/MTRL_DIAGNOSTIC_SYNTHESIS.md`.

---

## F5 — Learned task relations are representation-dependent  (ESTABLISHED)

The learned Ω changes qualitatively with pooling — it is not an intrinsic,
representation-independent property of KS/SI/ER:

| Setting | ks↔si | ks↔er | si↔er | Structure |
| --- | --- | --- | --- | --- |
| `weighted` 16L | −0.333 | +0.333 | −0.333 | sign-structured |
| `smp` 16L | +0.185 | +0.229 | −0.054 | differentiated |
| `smp` 25L | +0.333 | +0.333 | +0.333 | saturated, uniform |
| `lnp` 16L | +0.333 | +0.333 | +0.333 | saturated, uniform |

Interpretation to carry into the thesis: Ω describes relationships between
tasks *as expressed through the current representation, architecture, sample
and optimization process*, not a pooling- or split-independent truth. This is a
candidate final-framework finding in its own right.

Provenance: `02-lnp/README.md` (Ω under `lnp`); `CONTINUATION.md` §9.4
(per-setting table); MTRL `omega_history.json` artifacts in `results_*/`.

---

## F6 — Relation stability is conditional on evaluation axis  (ESTABLISHED, NARROWED 2026-09-21)

**Observation.** Within the 25L `smp` LOSO condition, KS↔SI is stable across
held-out speakers while ER-involving Ω entries vary and can change sign. That
ordering is not a task-intrinsic rule: matched five-seed evidence at 16L makes
KS↔SI seed-unstable and SI↔ER the only edge with a fully consistent sign.

**Evidence.**

| Axis / setting | ks↔si | ks↔er | si↔er |
| --- | --- | --- | --- |
| 25L `smp`, 10 LOSO folds | 0.286 ± 0.012, 10/10 positive | 0.098 ± 0.072, range −0.079–0.201 | 0.033 ± 0.103, range −0.202–0.205 |
| 16L `smp`, seeds 0–4 | 0.110 ± 0.236, 4/5 positive | 0.035 ± 0.213, 4/5 positive | −0.256 ± 0.076, 5/5 negative |
| 25L `smp`, seeds 0–4 | 0.33308 ± 0.00017, 5/5 positive | 0.200 ± 0.266, 4/5 positive | 0.200 ± 0.266, 4/5 positive |

**Interpretation.** Relation strength and confidence are different quantities.
Stability depends on the representation and on whether the perturbation is
speaker fold, seed or layer selection. A saturated stable edge is not
necessarily useful: 25L KS↔SI stays near +1/3 without a material KS or SI gain.

**Alternative explanations.** The 16L sign changes may reflect rank-near-one
saturation of normalized mean-head summaries rather than genuine task-relation
changes. LOSO varies the ER sample while the seed comparison varies
initialization and minibatch order, so their variances are not interchangeable.

**Confidence.** Strong for each stated configuration (10 folds or five matched
seeds); moderate for the broader conclusion that no pair has a universal
stability ranking.

**Implications.** Do not call KS↔SI intrinsically stable or use final Ω
magnitude as a transfer proxy. Every relation claim must name pooling, layers,
seed/fold axis and saturation state.

**Required follow-up.** DG-0001 found that raw directed-transfer estimates are
dominated by optimizer exposure (F8), so Ω correspondence remains unresolved.
DG-0002 must measure gradient interaction under an exposure-controlled sampler.

Provenance: `01-mtrl/results_mtrl_kfold/*/omega_history.json`; MLflow
experiment `taskrelation-mtrl`, five-seed `smp` 16L/25L runs at commit
`bfb1ad44`; `research/task_relations/MTRL_DIAGNOSTIC_SYNTHESIS.md`.

---

## F7 — Ω saturation destroys the regularizer's discrimination  (ESTABLISHED)

When Ω saturates toward uniform (+1/3 everywhere; observed at `smp`+25L and
`lnp`+16L), the relation regularizer has no pair-specific information left to
act on. This is a plausible explanation for MTRL performing *at or below*
baseline at `smp`+25L despite the extra layers — the mechanism degenerates into
a uniform coupling penalty.

Not yet established: whether early saturation causes the accuracy loss, is
caused by it, or is merely correlated. This is a diagnostic question, not a
finding.

Provenance: `CONTINUATION.md` §9.4 ("Saturation is a failure mode for
interpretability"); per-epoch Ω summaries in the `01-mtrl` iteration log and
`omega_history.json`.

---

## F8 — Same-epoch pairwise transfer is confounded by optimizer exposure  (ESTABLISHED)

**Observation.** Under the current concatenated-dataset trainer, adding a large
auxiliary task changes the number of optimizer updates and the effective
minibatch size of the target task. Raw pair-minus-single accuracy therefore
does not isolate task-semantic transfer.

**Evidence.** DG-0001 first observed large speaker-independent LOSO gains over
an ER-only batch-2048 control: ER<-KS +0.3411 and ER<-SI +0.2794, each positive
in 10/10 folds. ER-only controls with approximately matched update counts
reproduced those gains without auxiliary data:

| Comparison, fixed epoch | Mean Δ | Paired 95% CI | Fold signs |
| --- | ---: | --- | --- |
| KS+ER − ER-only batch 160 | +0.0053 | [−0.0289, +0.0394] | 5 positive / 5 negative |
| SI+ER − ER-only batch 64 | −0.0589 | [−0.0883, −0.0295] | 0 positive / 10 negative |

The exposure controls explain +0.3358 of the raw +0.3411 KS-associated gain and
+0.3383 of the raw +0.2794 SI-associated gain. Five-epoch reverse sensitivities
were KS<-ER −0.0038 and SI<-ER −0.0231.

**Interpretation.** DG-0001 does not establish beneficial asymmetric transfer.
KS adds no resolved ER benefit after approximate step matching; SI and ER
interfere in both directions. The raw directed matrix measured optimization
opportunity more than task relationship.

**Alternative explanations.** Batch 160/64 only approximate the pair arms'
update counts and also change gradient noise. Reverse directions use one
single-task seed-42 reference rather than ten refits. These limitations prevent
precise semantic-effect estimation, but they cannot support the rejected raw
28–34 point transfer interpretation.

**Confidence.** Strong that the original raw ER gains are dominated by
optimizer exposure; moderate that SI/ER transfer is genuinely negative;
preliminary for KS/SI and exact reverse magnitudes.

**Implications.** Any empirical transfer study must match or explicitly model
optimizer steps, effective per-task batch size, loss scaling, epoch budget and
checkpoint policy. DG-0001 cannot justify an asymmetric replacement method.
Ω/transfer correspondence must use a controlled transfer target.

**Required follow-up.** DG-0002's seed-42 screen found no persistent pairwise
conflict but did find strong ER gradient-norm dominance (F9). Confirm that
signal across matched seeds before any next-method literature search.

Provenance: `research/studies/DG-0001/{STUDY.md,analysis.md,result.json}`;
`research/task_relations/{empirical_transfer.json,loso_transfer.json,optimization_control.json}`;
MLflow experiment `taskrelation-diagnostics`.

---

## F9 — ER shared-gradient norm dominance is reproducible and MTRL does not mitigate it  (ESTABLISHED, 2026-09-22)

**Observation.** Under matched three-task `smp` 25-layer training, ER's shared-parameter gradient norm dominates KS/SI in the middle and late thirds across seeds `0–4`. Classical MTRL does not consistently reduce that scale imbalance. Persistent pairwise gradient conflict is not supported.

*Refined 2026-09-22 by DG-0005 (F10): the dominance reproduces only under the standard training mixture. Raising ER's per-batch share removes it in 5/5 seeds, so it is a sampling-regime property rather than a task-intrinsic relation property.*

**Evidence.** Each baseline/MTRL seed pair sampled the same 142 of 2,820 optimizer steps with identical per-step valid-example counts and the same 1,550,800 shared parameters. Seed-level phase summaries are the independent observations:

| Method | Phase | Mean max/min task-norm ratio | Seed SD | 95% t interval | Seeds ≥ 3 |
| --- | --- | ---: | ---: | --- | ---: |
| baseline | middle | 7.243 | 0.272 | [6.905, 7.581] | 5/5 |
| baseline | late | 8.962 | 0.456 | [8.396, 9.528] | 5/5 |
| MTRL | middle | 7.129 | 0.421 | [6.606, 7.652] | 5/5 |
| MTRL | late | 9.004 | 1.059 | [7.689, 10.319] | 5/5 |

Paired MTRL-minus-baseline ratio differences were −0.114 (95% CI [−0.717, +0.490]) middle and +0.042 ([−1.473, +1.556]) late. No pair met the pre-registered persistent-conflict threshold in any seed; baseline late mean cosines were KS↔SI `+0.002`, KS↔ER `+0.001`, and SI↔ER `+0.022`.

**Relation to Ω.** Final off-diagonal Ω magnitudes saturated near `1/3` in all five seeds. Seeds 0–3 were uniform positive; seed 4 had both ER edges near `−1/3`, reproducing F6's joint sign flip. Magnitude saturation is reproducible, but uniform positive coupling is not. Ω's sign varies while the norm-dominance signal persists.

**Outcome context.** Fixed-epoch MTRL-minus-baseline deltas were aggregate −0.00031, KS −0.00018, SI −0.00036, and ordinary-split ER −0.00109; every paired 95% interval included zero. This confirms no architecture improvement. The ordinary-split ER outcome remains speaker-leaky and supports no ER performance claim.

**Interpretation.** Classical MTRL's head-parameter covariance neither represents nor regulates the dominant shared-optimization scale behavior measured here. This is a concrete limitation to drive targeted literature research. It is not causal evidence that norm imbalance explains all of MTRL's outcome null.

**Alternative explanations.** ER contributes about 47 valid examples per sampled batch versus 539 KS and 1,462 SI. Data scarcity, task difficulty, label noise and gradient-estimate variance may cause the larger norms. **DG-0005 resolved the strongest part of this question (2026-09-22): the dominance is training-mixture dependent, not task-intrinsic.**

**DG-0005 data-regime control (CONFIRMED).** Five matched seed pairs at commit `8032a937`, same `smp` 25-layer protocol, changing only the training split's per-task composition (ER weight `11.5`, `num_samples` unchanged, validation/test untouched, 2,820 steps and 142 diagnostic samples in every arm):

| Phase | A0 standard composition | A1 ER-weighted | Paired A1−A0 | 95% t interval |
| --- | ---: | ---: | ---: | --- |
| middle | `7.243 ± 0.272` | `2.818 ± 0.272` | `−4.425` | `[−4.980, −3.869]` |
| late | `8.962 ± 0.456` | `2.352 ± 0.270` | `−6.610` | `[−7.353, −5.867]` |

A1 realized ER÷KS sampled-batch counts of `0.990–1.008` against A0's `48.1 ÷ 540.7 = 0.089`, every exposure check passed, and A1's late ratio was below the `3.0` dominance threshold in `5/5` seeds. Late mean norms moved ER `6.972 → 1.608`, KS `0.828 → 0.894`, SI `0.804 → 0.687`, so the collapse is ER-localized. Late cosines stayed near zero in both arms, so no conflict mechanism is involved. A0 exactly reproduces this finding's own baseline numbers (`7.243 ± 0.272`, `8.962 ± 0.456`), confirming the sampling knob is default-off.

**Interpretation.** The reproducible dominance is a property of the sampling regime, not of ER's task semantics at this representation: ER's gradient *estimate* is inflated when it contributes ~47 of 2,048 examples and collapses toward the pool-mean gradient at ~435. A mechanism motivated by "ER is intrinsically a large-gradient task" now rests on a refuted premise; a mechanism motivated by estimator scale must be justified from the training mixture instead.

**Remaining confound.** The same knob also multiplies ER's optimizer updates per epoch (the ~43k ER examples are drawn ~9× more often), and A1's ER head saturates harder (train ≈0.99 vs validation ≈0.82; final train−val gap `0.134 → 0.174` at the screening seed). Part of the norm reduction may be convergence/overfitting rather than estimator variance. The one-knob design cannot separate these, and the pre-registered A2 reverse arm would not either.

**Outcome status.** DG-0005 makes no ER performance claim. Its ordinary-split deltas (aggregate `+0.00106 [+0.00026, +0.00186]`, ER `+0.01049 [+0.00249, +0.01849]`) are speaker-leaky context (F3) whose ER direction is consistent with memorization, not generalization.

Provenance: `research/studies/DG-0002/{PLAN.md,analysis.md,confirmation_result.json}`; MLflow experiment `taskrelation-diagnostics`, stage `confirm`, ten runs at commit `7f6d5248`; `research/studies/DG-0005/{PLAN.md,analysis.md,NOTE.md,confirmation_result.json}`, stage `confirm`, ten runs at commit `8032a937050d8bbd3114b172cb813a8fc7370b37`.

---

## F10 — Gradient scale is a training-mixture property, not a task-intrinsic relation property  (ESTABLISHED, 2026-09-22)

**Observation.** Under the matched `smp` 25-layer `ks_si_er` protocol at a fixed global batch of 2,048 and fixed optimizer exposure, the per-task shared-gradient norm ratio is governed by how many examples of each task the batch contains. It is not an intrinsic property of ER at this representation.

**Evidence.** DG-0005's five matched seed pairs changed only ER's sampling weight in the training split. ER's sampled share rose from ~`47/2048` to ~`435/2048` while validation and test data were untouched and every arm kept 2,820 optimizer steps and 142 diagnostic samples. Late max/min task-norm ratio fell `8.962 ± 0.456 → 2.352 ± 0.270` (paired `−6.610`, `[−7.353, −5.867]`), middle fell `7.243 ± 0.272 → 2.818 ± 0.272`, A1's late ratio cleared the pre-registered `< 3.0` threshold in `5/5` seeds, and the change is ER-localized (late ER norm `6.972 → 1.608`; KS `+0.066`, SI `−0.117`). A0 reproduces the baseline numbers exactly, so the knob is inert when unused.

**Interpretation.** `E‖g‖ ≥ ‖E g‖`: a batch gradient estimated from few examples has an inflated norm relative to the pool-mean gradient it approximates. ER at ~47 examples per batch is the smallest task, so it carries the largest estimate inflation. This is scale, not relation: the learned Ω had already failed to track the scale signal (F9), and the pairwise cosines stayed near zero here.

**Framework consequence.** Gradient scale joins relation magnitude and relation confidence as a *conditioned* quantity rather than an intrinsic task property. Reported "task gradient scale" must therefore name the sampling regime, exactly as relation claims must name pooling, layers and seed/fold axis (F5, F6).

**Alternative explanations — bounded post-hoc (2026-09-22).** The same knob multiplies ER's optimizer updates per epoch and its head saturates harder (train ≈0.99 vs validation ≈0.82; final train−val gap `0.134 → 0.174` at the screening seed). A pre-registered post-hoc analysis (`studies/DG-0005/analyze_noise_shape.py`, committed at `2ac7f3d` before any statistic was computed) bounded how much of the drop estimator size can carry. With `σ ∝ 1/√n` a noise-dominated norm must satisfy `E‖g‖ ∝ 1/√n`, so the mean-norm ratio between arms is at most `√(n_A1/n_A0) ≈ 3.0`; the observed late ratio was `3.99–4.69`, giving an estimator share bound of `0.758` `[0.715, 0.801]` across seeds — i.e. **at least ~20% (mean 24%) of the late drop requires a smaller mean gradient or noise growing faster than `1/√n`**. In the middle phase the estimator account is sufficient on its own (share `1.028` `[0.975, 1.082]`), so the unexplained component is specific to late training. Two further results: the observed within-phase dispersion (`CV ≈ 0.32`) exceeds the isotropic-noise ceiling for `d = 1,550,800` (`≈ 0.00057`) by ~`560×`, so dispersion cannot proxy estimator variance at this instrumentation and no shape statistic may be read as evidence about it; and DG-0002's matched baseline reproduces the A0 statistics value for value. This is a bound, not an identification: the estimator share lies in `[0, bound]`, and "ER head saturates faster" is not separated from "steeper-than-`1/√n` noise growth". It is reported inside F10 rather than as a new finding because it constrains an existing claim rather than establishing a new one.

**Confidence.** Strong for mixture dependence across five matched seeds with a passing exposure gate. Moderate for the pure estimator-variance mechanism — reduced, in the late phase, by the bound above.

Provenance: `research/studies/DG-0005/{PLAN.md,analysis.md,NOTE.md,confirmation_result.json,noise_shape_result.json,analyze_noise_shape.py}`; MLflow experiment `taskrelation-diagnostics`, stages `screen` and `confirm`; screen commit `0162224`, confirmation commit `8032a937050d8bbd3114b172cb813a8fc7370b37`; post-hoc bound script commit `2ac7f3d`.

---

## Record-level observations

### R1 — DG-0001 created the first single-task and pairwise runs  (RECORD, UPDATED 2026-09-21)

Before DG-0001, all 300+ project runs used `ks_si_er` or `ks_si_er_ic`.
DG-0001 then exercised `ks`, `si`, `er`, `ks_si`, `ks_er` and `si_er`, including
matched LOSO and optimization-exposure controls.

Consequences:

* the dynamic task path is now behaviorally exercised for one-, two- and
  three-task settings;
* single-task baselines now exist, but only DG-0001's seed/budget/protocol may
  be compared directly;
* the first raw transfer matrix is not a semantic relation target because F8
  shows task-count-dependent optimizer exposure dominates it.

### R2 — Historical runs predate Study-ID tagging  (RECORD, 2026-09-21)

The runs behind F1–F7 (`taskrelation-mtrl*`, `taskrelation-gbc`,
`wavcse-base-lnp`, `wavcse-base-poolingsweep`, the historical ER k-folds) were
logged before the Study ID convention and remain legacy evidence. DG-0001 is
the first registered Study in `research/STUDIES.jsonl`; all of its runs carry
Study ID, stage, seed, task set, pooling/layers, git SHA and DagsHub run notes.

### R3 — The F9 literature gate found no eligible published mechanism  (RECORD, 2026-09-22)

`LT-0001` screened eight primary sources against pre-registered gates for a
published Task Relation Learning method that directly addresses F9's
shared-gradient scale imbalance in heterogeneous deep classification without
category drift.

Outcome: explicit relation methods either lacked a direct optimization-scale or
reliability mechanism or required aligned Gaussian/mean-estimation assumptions;
the methods that directly address scale (uncertainty weighting, GradNorm) learn
per-task scalars with no relation object and belong to optimization/loss
weighting; the 2024 informative-relation method depends on parameter
decomposition.

Consequences:

* no mechanism Study may cite F9 for authorization; the programme is
  `NEEDS-HUMAN-REVIEW` (DEC-0008);
* relation magnitude, relation confidence/noise and optimization weight are
  distinct quantities and must not be conflated in the framework;
* this gate is complete for F9 — do not repeat the same search without new
  evidence or an explicit scope change.

Provenance: `research/studies/LT-0001/{PLAN.md,analysis.md,result.json}`;
`research/literature/INDEX.md` (eight paper cards).
