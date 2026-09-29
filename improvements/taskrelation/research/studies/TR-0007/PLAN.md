# TR-0007 — PLAN (pre-registration)

Status: **PRE-REGISTERED — screen not yet submitted.** No compute has been authorized for
this scope (`authorizations/TR-0007.yaml` does not exist). No run was launched by this plan.
Created: 2026-09-29. Study registry: `../../STUDIES.jsonl`. Gate decision: `DEC-0015` (human,
2026-09-29: Option A — faithful published MSSL formulation).
Arm identity: **faithful published-method implementation** (DEC-0013 §3, DEC-0014 §4).

## Observation motivating the study

Classical symmetric MTRL (`01-mtrl`) has no reproducible advantage over the matched wavCSE
baseline (F4), and its Ω is built by a closed-form `(WᵀW)^{1/2}` trace normalisation that
saturates near uniform (`+1/3`) under `smp` + 25 layers, leaving no pair-specific
information (F5, F7, F9). Family B of `LT-0002` asked for a *published* relation estimator
that does not saturate by construction; the literature pass returned one clean pass:
p-MSSL (Gonçalves, Von Zuben & Banerjee, JMLR 17(33), 2016), whose Ω is a sparse task
**precision** estimated by a penalised likelihood rather than a normalised covariance.

## Hypothesis (one primary hypothesis)

**H1.** Under the shared protocol, replacing classical MTRL's trace-normalised covariance
with p-MSSL's sparse graphical-lasso precision — same protocol, same summary adapter, same
schedule — changes the learned relation object enough to beat **both** classical symmetric
MTRL (in-category control) and the matched wavCSE baseline, per-task and aggregate, without
material regression on any task.

**Competing explanation (stated before any run).** Any movement in the comparison can come
from the shared *mean-head summary adapter* rather than from the Ω estimator: both relation
arms consume the same summary matrix, and the representation — not the estimator — already
produced larger effects than any relation mechanism measured in this branch (F2: pooling
alone moved ER by 3.25pp). A win is attributable to the estimator only if both controls are
matched **and** the Δ exceeds the seed variability F1 already documented.

**Third, non-exclusive outcome.** The estimator may be mechanism-live but behaviourally
inert: if the λ₂ selected on validation drives Ω to the diagonal, the coupling term
degenerates to per-task weighting, the arm is a no-relation control, and the screen can only
produce `REJECTED` — which is recorded as a null result, not as evidence against the method.

## Falsification / decision rule (declared before the run)

Screen stage (one seed), read at the protocol checkpoint (`epoch`, the protocol's primary
checkpoint; `best`/`opt` are reported but do not decide):

* `PROMISING` iff `test_epoch_acc_all` > max(both controls' `test_epoch_acc_all`) **and**
  every per-task `test_epoch_*_acc` is within `0.20pp` of both controls.
* `REJECTED` otherwise.
* `INCONCLUSIVE` if an arm cannot run under the shared conditions (protocol §5) or if a
  pre-declared exposure check fails.

A screen can never promote the arm (F1). Confirmation (seeds `0–4`, paired per-seed
differences) is a separate, later stage and is **not** part of this plan's submission.

## Independent variable

Exactly one: **the relation mechanism** — the relation object, its estimator and its update
rule (protocol §5) — i.e. MSSL's sparse precision Ω solved by the published ADMM
graphical lasso (Eqs. (8)–(11)) versus MTRL's closed-form covariance.

## The three registered arms (seed 42)

| Arm | Config used by the plan | Copies (unchanged apart from the study's provenance block and output roots) | Experiment |
|---|---|---|---|
| 1. p-MSSL (candidate) | `04-mssl/mssl_config.yml` | — (its architecture folder owns it) | `taskrelation-mssl` |
| 2. classical MTRL (in-category control) | `studies/TR-0007/configs/classical-mtrl.yml` | `01-mtrl/mtrl_poolingwinner_25L_config.yml` | `taskrelation-mtrl` |
| 3. matched wavCSE baseline | `studies/TR-0007/configs/wavcse-baseline.yml` | `base/configs/base_poolingwinner_25L_config.yml` | `wavcse-baseline` |

Every arm's config carries the `research:` block (`study_id: TR-0007`, `stage: screen`,
`method: <arm>`), so the three runs of this Study are reconstructible by tag across the three
experiments the naming convention assigns them to (`improvements/README.md`: one experiment per
category/architecture, runs grouped across experiments by `study_id`/`stage`/`method` tags, not by
co-location). The control configs are byte-identical to their source configs in every shared block
(`upstream`, `dataset`, `pooling`, `training`, `evaluation`, `model`), asserted by
`test_tr0007_protocol.py::test_controls_stay_matched_and_untuned`.

The controls are the ones the protocol fixes (§2); the candidate is the only arm this study
implements.

## Published formulation implemented (Option A, DEC-0015)

Paper objective Eq. (3) / Eq. (4b), with λ₀ = 1 (the paper's value in *all* experiments):

```
min_{W,Omega>0}  L(W) + lambda_0*tr(W Omega W^T) - d*log|Omega| + lambda_1*|W|_1 + lambda_2*|Omega|_1
```

Ω step, Eq. (8) and the ADMM of Eqs. (9), (10a–c), (11):

```
min_{Omega>0}  lambda_0*tr(S Omega) - log|Omega| + (lambda_2/d)*||Omega||_1,   S = (1/d) W^T W
```

* `lambda_2` is the paper's Eq. (3) penalty; the `1/d` that Eq. (8) displays is applied inside
  the solver, which takes `d = W.shape[1]` explicitly. The paper states the equivalence in
  words ("the factor 1/d can be incorporated into lambda_2"); with `d = 2001` the two
  conventions are different estimators at the same numeric value, so the convention is pinned
  by a regression test.
* ℓ₁ applies to the **off-diagonal** entries, following the graphical lasso of Friedman,
  Hastie & Tibshirani (2008), which the paper cites for this step.
* `lambda_0 = 1.0` (paper setting); `lambda_1 = 0.0` (the paper's own named
  "exclusive Gaussian prior" case; our W is a mean-of-head summary for which an ℓ₁ penalty has
  no feature-selection analogue). Both remain configurable knobs.

### Declared deviations (mandatory labelling in every table)

1. **Mean-head summary adapter (shared with the control).** The paper's per-task parameter
   vector is a whole linear model; our heads have 12 / 1251 / 4 outputs and no aligned columns,
   so each task contributes `[mean(classifier weight rows); mean(bias)]`, with the control's
   row normalisation (`normalize_w: true`), giving S a common per-task dimension by
   construction. This is a reduction, not a faithful reproduction of the paper's setting.
   Both relation arms consume the same matrix, so the comparison isolates the estimator.
2. **λ₂ is a data-selected quantity in the source paper** (Algorithm 1: "penalty parameters
   chosen by cross-validation"; the regression experiments use stability selection and the
   classification experiments cross-validate over `{0.01, 0.1, 1, 10, 100}`). The paper
   publishes no default and λ₂'s numeric scale is not transferable between representations
   (the paper's own setting standardises its data). See *Open item* below.

## Frozen protocol conditions (protocol §1; identical in all three arms)

| Component | Setting |
|---|---|
| Upstream | `wavlm_large`, frozen; frame pooling `mean`; layer pooling `smp` 0.5 over **all 25 layers** |
| Task set | `ks_si_er` (KS 12-class, SI 1251-class, ER 4-class) |
| Shared trunk | 1024→512 projector, 512→2000 hidden, dropout 0.4 / 0.6 |
| Epochs | 30 |
| Global batch | 2048, `drop_last_train: true` |
| Optimizer | AdamW, lr 0.0025, weight decay 5e-8 |
| Regularization | l1 1e-7, l2 1e-5 |
| Scheduler | ReduceLROnPlateau factor 0.5 (config key `patience: 5` is inert everywhere) |
| Checkpoint policy | best / opt / epoch; protocol checkpoint = `epoch` |
| Data splits | unchanged; `subset_percentage: 100` |
| Optimizer exposure | same ≈2,820 steps per run; all three arms share the concatenated sampler |
| Gradient diagnostics | same schedule as `improvements/gradient_diagnostics.py` |
| Seed | 42 (screen) |
| Stage | `screen`, one run per arm |

**All 25 layers.** `upstream.selected_transformer_layers: all` resolves through
`downstream/utils/parse_transformer_layers.py` to `list(range(25))` for `wavlm_large`, and the
loader slices the full layer axis (25 × 1024 per utterance) before `Pooling` reduces it. This
is enforced executably, not only in prose: see *Executable validation* below.

## Arm-specific hyperparameters (only the mechanism's own paper parameters)

| Key | Value | Source |
|---|---|---|
| `mssl_lambda_0` | 1.0 | paper: "set to one in all experiments" |
| `mssl_lambda_1` | 0.0 | paper's exclusive-Gaussian-prior case; no analogue for a summary W |
| `mssl_lambda_2` | see *Open item* | paper: cross-validated per data set |
| `mssl_admm_rho` | `null` (solver auto-initialises and rebalances, Boyd et al. §3.4.1) | numerical, not scientific |
| `mssl_admm_iterations` | 2000 | numerical |
| `normalize_w` | `true` | the control's adapter setting (shared representation) |
| `mssl.warmup_epochs` | 3 | the control's schedule, so the mechanism engages at the same epoch |
| `mssl.omega_update_frequency` | 1 | the control's schedule |

No other hyperparameter may change. In particular, the protocol values above are frozen and
were not re-tuned.

## Open item — λ₂, the one pre-registration input the paper does not supply (HUMAN_DECISION)

The source paper fixes λ₀ = 1 and **selects λ₂ on data**; the repository's literature record
says the same ("the paper uses stability selection to choose them"). Two coherent
pre-registrations exist, and they differ in run count and in what a `REJECTED` outcome means:

* **(i) Validation-selected λ₂ (recommended).** The paper's own procedure, restricted by
  protocol §6: select λ₂ on the **validation** split at the screen seed from the paper's
  published classification grid restricted to its two smallest values `{0.01, 0.1}` (2 values,
  matching the control's own two-value λ selection budget), with the selection metric
  pre-declared (aggregate validation accuracy at the protocol checkpoint; ties → the sparser
  Ω), every attempt recorded, the value frozen before the three-arm comparison. Cost: two
  extra MSSL runs at seed 42 (+1.2 GPU-h; the screen becomes 5 runs ≈ 3.0 GPU-h). A
  controller-side scan (2026-09-29, CPU, no compute) shows this is the live region for this
  summary: at `normalize_w: true` the off-diagonal support goes from 4/6 non-zero at
  λ₂ = 0.01 to 0/6 at λ₂ = 0.1, i.e. the grid spans "mechanism live" → "Ω diagonal".
* **(ii) Fixed λ₂ named by the researcher.** Keeps the registered screen at exactly three
  runs (≈1.8 GPU-h) but requires a value the paper does not publish; a `REJECTED` outcome then
  reads "MSSL at that λ₂ did not help", not "MSSL did not help".

Until the human answers, this plan pre-registers **(i)** and `mssl_config.yml` carries
`mssl_lambda_2: 0.01` — the smallest value of the paper's own grid, i.e. the grid point whose
mechanism is verifiably live — as the value that variant (ii) would freeze.

## Cheapest adequate experiment

The registered **screen**: three runs (one per arm) at seed 42, ~0.6 GPU-h each on one
1×GPU worker, each ≤ 6 h wall clock, with the two controls run under the identical protocol
(the control configs are unchanged, and no control is re-tuned). For variant (i) of the open
item, two additional MSSL runs at the grid values precede the comparison.

Expected information gain: a first, honest read on whether a *published* sparse-precision
relation estimator moves KS/SI/ER at all under the frozen protocol — either a `PROMISING`
candidate for multi-seed confirmation, or a bounded negative result that closes the
sparse-precision family for this representation and feeds the framework's selection rules.

## Expected compute and GPU allocation

* Screen: 3 runs (5 with variant (i)) ≈ 1.8 (3.0) GPU-h on a single 1×GPU worker.
* Confirmation (later stage, not submitted now): 9.0 GPU-h.
* At most one worker at a time; no parallel worker. `df -h` and `nvidia-smi` are checked by
  the backend before any worker request; container disk ≤ 60 GB.

## Proposed authorization envelope (reproduced from Phase 4 for human review; NOT created yet)

```
max_gpu_hourly_usd: 0.80
max_total_gpu_usd: 5.00
max_wall_clock_hours: 6
max_simultaneous_workers: 1
replacement_workers_allowed: true
existing_network_volume_allowed: true
new_persistent_resources: false
container_disk_gb_max: 60
destroy_on_completion: true
retain_for_reuse_hours: 0
```

## Provenance

* MLflow/DagsHub: tracking URI `https://dagshub.com/Ke-vin-S/wavCSE.mlflow`; experiments as in
  the arm table above (the documented `taskrelation-<model>` / `wavcse-*` convention). Credentials reach
  the worker only as `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` through the
  backend's `environment_secrets` allow-list; they are never written into this plan, a
  config, a jobspec body or a research record.
* Every run carries `study_id`, `stage`, `arm`/`method`, `seed`, `layers`, pooling, the exact
  git commit and a DagsHub run note (`NOTE.md`), plus the arm's mechanism parameters.
* Run name: `TR-0007__screen__p-mssl__ks_si_er__smp25__s42` (research-block naming).
* The Study's result, analysis and any negative evidence are recorded in `result.json`,
  `analysis.md`, this folder, `STUDIES.jsonl`, `FINDINGS.md`/`FAILURES.md` as applicable.

## Executable validation (run before submission)

1. `improvements/taskrelation/research/tests/test_tr0007_protocol.py` — resolves the arm's
   config and proves it uses all 25 layers, the frozen protocol values, the published λ
   convention, the three registered arms at seed 42 and no secret material.
2. `improvements/taskrelation/research/tests/test_mssl_omega_solver.py` — the Ω step solves
   Eq. (8) (closed forms, optimality certificate, λ₂ units) and the coupling term's analytic
   gradient.
3. `make check`, `make agents-check`, `make research-check-all`, `git diff --check`.
4. The compute backend's own `plan`/`preflight` verbs (dry run) before any worker request.

## Category-boundary check (protocol / DEC-0005 §3)

The arm is Task Relation Learning in the Zhang & Yang §2.4 sense: an explicit relation object
(a sparse task precision with a partial-correlation interpretation) estimated from task
parameters. It is not loss weighting, gradient surgery, low-rank, clustering or
decomposition. It stays inside `improvements/taskrelation/`.

## Risks and known gotchas

* **ER is speaker-leaky under the ordinary split** (F3, ~15pp inflation): screen ER numbers
  are context only; no ER claim is made here and LOSO remains required for any later ER claim.
* **Task-size masking**: the sample-weighted aggregate is SI-dominated; per-task numbers are
  reported and read with the aggregate.
* **Exposure**: all three arms share the sampler and the step count, so no matching control
  is needed for exposure; any deviation invalidates the comparison (protocol §10).
* **Relation saturation**: a diagonal Ω means the mechanism is inert; the per-epoch relation
  object and its optimality certificate are logged so this is visible rather than inferred.

## Prior evidence on DagsHub (record conflict, recorded not smoothed over)

The tracking repository already contains TR-0007 runs from an **earlier code line that does
not exist in this repository's history** (commits `10aaaea3…` and `3df542d…`, experiment
`taskrelation-variant-benchmark`): a `screen` stage at **seed 0** with `p-mssl`,
`classical-mtrl` and `wavcse-baseline` (all tagged `status: rejected`), plus a
`scale-corrected` `p-mssl-correlation` run. This branch's registry did not know about them.
They are retained as evidence; this plan's runs are a *protocol-corrected* implementation at
the registered screen seed 42 and are identified by their commit, not by the method name
alone. See `DEC-0015` and `NOTE.md`.
