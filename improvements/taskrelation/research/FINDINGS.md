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

## F9 — ER shared-gradient norm dominance is a confirmation candidate  (SCREENING, 2026-09-22)

**Observation.** In DG-0002's exposure-matched seed-42 baseline, ER's mean
shared-parameter gradient norm was 7.36× and 7.66× the smallest task norm in the
middle and late training thirds. Classical MTRL did not remove the imbalance:
its ratios were 6.75× and 9.28×. No pair met the pre-registered persistent
gradient-conflict threshold.

**Evidence.** Each arm sampled the same 142 of 2,820 optimizer steps. Per-step
valid-example counts matched exactly across methods. Baseline phase means
(KS/SI/ER) were 1.117/1.389/8.218 in the middle third and
0.864/0.831/6.363 late. MTRL means were 1.148/1.123/7.576 and
0.862/0.651/6.038. Baseline late pairwise mean cosines were near zero
(KS↔SI +0.005, KS↔ER +0.003, SI↔ER +0.025), not persistently negative.

**Relation to Ω.** MTRL Ω saturated by late training to uniform positive
coupling: final off-diagonals were 0.33304/0.33311/0.33307, range `6.95e-5`.
Thus the current mechanism lost pair discrimination while the shared
optimization remained strongly scale-imbalanced. This is correspondence, not
causal evidence that saturation created the imbalance.

**Alternative explanations.** ER contributed only 47.2 examples per sampled
mixed batch on average, versus 539.1 KS and 1461.7 SI. Its norm may reflect
sample scarcity, gradient-estimate noise, task difficulty or label noise rather
than semantic task relations. One seed cannot establish a general task
property. The within-run samples are correlated and are not independent
replicates.

**Implication.** The pairwise-conflict explanation is weakened for this seed;
optimization-scale imbalance is the confirmation candidate. Do not select a
mechanism or enter literature mode yet.

**Required follow-up.** Continue DG-0002 with matched baseline/MTRL seeds 0–4
and use seed-level phase summaries as the independent observations.

Provenance: `research/studies/DG-0002/{PLAN.md,analysis.md,result.json}`;
MLflow runs `71b89be472284af9a855f46871eb9f38` and
`12a92107a6a34f26a705061f0e373239`.

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
