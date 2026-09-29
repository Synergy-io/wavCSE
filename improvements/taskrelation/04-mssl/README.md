# wavCSE-MSSL — Multi-task Sparse Structure Learning

**Status: the faithful published arm of the Task Relation Learning variant
benchmark.** Registered as Study **TR-0007** in
`../research/studies/TR-0007/`; authorized by `../research/DECISIONS.md`
DEC-0013 (benchmark of published variants) and DEC-0014 §5 (MSSL keeps its
faithful-implementation status because it passed the LT-0002 family-B
source gate and its input is literally the task-parameter summary matrix).
DEC-0015 (human, 2026-09-29) fixed this arm's formulation at the published
Eq. (3)/(8) objective before any run.

This folder is a self-contained `0N-<name>/` architecture attempt — own model
(`mssl_model.py`), trainer (`mssl_trainer.py`), config (`mssl_config.yml`) and
README — following the numbering convention `01-mtrl/` started.

The published method is Goncalves, Von Zuben & Banerjee (2016), "Multi-task
Sparse Structure Learning", JMLR 17(33):1-30 — the **p-MSSL** instantiation.
Its relation object is a **sparse task precision** Ω (a Gaussian graphical
model's inverse covariance), estimated per Ω-step by the paper's ADMM
graphical lasso, not by gradient descent.

This is *not* `models/pmr_model.py`. PMR is quarantined (DEC-0004): it learned
a precision matrix by gradient descent inside one combined loss and never
trained. MSSL's Ω is solved by the published alternating scheme and is never
touched by the optimizer.

## What it does

With `W` the `[num_tasks, d]` task-parameter matrix and `Omega` the
`[num_tasks, num_tasks]` precision, the paper's joint objective (Eq. 3) is

```
L(W) + lambda_0 * tr(W Omega W^T) - d * log|Omega| + lambda_1 * |W|_1 + lambda_2 * |Omega|_1
```

minimised by alternating minimization (Algorithm 1):

* **Eq. (4a) — the W step.** Given Ω, descend on `L(W) + lambda_0 * tr(W Ω Wᵀ)`
  (plus the optional `lambda_1 |W|_1`). This is the ordinary training step with
  the coupling term added to the batch loss; because it is built from the live
  classifier parameters, it back-props into the task heads. The trainer adds it
  only after `mssl.warmup_epochs`, mirroring the control's schedule.
* **Eq. (4b)/(8) — the Ω step.** Given W, solve

  ```
  min_{Omega > 0}  lambda_0 * tr(S Omega) - log|Omega| + (lambda_2 / d) * ||Omega||_1,
  S = (1/d) W^T W
  ```

  which is Eq. (4b) divided by `d`. `mssl_model.graphical_lasso_admm` solves it
  by the paper's ADMM — Eqs. (9), (10a)-(10c), (11): an eigendecomposition
  update of Ω, element-wise soft-thresholding of the split variable `Z`, and the
  scaled dual update. The l1 penalty is on the off-diagonal entries, as in the
  graphical lasso of Friedman, Hastie & Tibshirani (2008) that the paper cites
  for this step.

`lambda_0 = 1` is the paper's value in all experiments ("The parameter lambda_0
was set to one in all experiments"); it is kept explicit and overridable.
`lambda_1 = 0` is the paper's own named exclusive-Gaussian-prior case, and is
the setting used here (see deviations below).

Implementation notes:

* **Ω is a `register_buffer`, never an `nn.Parameter`.** It is refreshed by the
  trainer every `mssl.omega_update_frequency` epochs after
  `mssl.warmup_epochs`, so the optimizer never sees it.
* **The solver is deterministic and self-certifying.** It runs in float64,
  starts `rho` at the scale-consistent `1/mean(diag(S))`, rebalances it by the
  residual rule of Boyd et al. (2011) §3.4.1 — the reference the paper cites
  for the ADMM derivation — and stops on the primal-dual optimality
  conditions of Eq. (8), not on a comparison against a reference solver. The
  estimate returned is the split variable `Z`, whose zeros are exact, so the
  reported sparsity is exact rather than approached.
* **Units.** `lambda_2` everywhere in the module is the paper's Eq. (3)
  penalty; the `1/d` coefficient that Eq. (8) displays is applied inside the
  solver, which takes `d = W.shape[1]` explicitly, so the two conventions
  cannot be silently conflated.

## The relation object

Ω is a *precision*: `Omega_ij = 0` means tasks `i` and `j` are conditionally
independent given the others (paper §3.1, 3.3). Per the shared protocol it is
reported as the **partial correlations** `-Omega_ij / sqrt(Omega_ii Omega_jj)`
(with a zero diagonal), which is the interpretable form of the same object.

The claim this arm tests is that a proper penalised likelihood — the
`-log|Omega|` barrier plus l1 shrinkage — is a bounded, regularised
parameterisation that **does not saturate by construction**, unlike the
in-category control's closed-form `(W^T W)^{1/2}` trace-normalisation, which
collapses toward ±1/3 and loses pair-specific information (findings F5/F7/F9;
the paper states the same contrast: learning the inverse covariance directly
"tends to be more stable than computing covariance and then inverting it").
Saturation state is reported per protocol §4 — a saturated relation object
carries no pair-specific information.

## Declared deviations from the paper

Recorded in the Study before any run (see `mssl_model.py`'s module docstring
and `../research/studies/TR-0007/`):

1. **Summary adapter.** The paper's per-task parameter vector is a whole linear
   model; with a shared trunk and 12 / 1251 / 4-class heads there is no such
   object. Each task contributes the same fixed-length summary the in-category
   MTRL control uses — mean of its classifier weight rows concatenated with its
   mean bias — so the estimator is the only difference between the two arms.
   This is a reduction, and results are reported as such.
2. **`lambda_1 = 0`.** Disabled; in the paper's linear model it selects
   features, for which a mean-of-head-rows summary has no analogue. This is
   also the paper's own named special case ("Setting lambda_0 to one and
   lambda_1 to zero, we return to the exclusive Gaussian prior").
3. **`lambda_2` is selected on data by the source paper.** Algorithm 1 declares
   the penalty parameters "chosen by cross-validation", and the classification
   experiments use the grid `{0.01, 0.1, 1, 10, 100}`. The Study
   **pre-registers** `0.01` — that grid's smallest value — before any run; it
   is not the outcome of a search in this Study. Any result therefore speaks to
   p-MSSL at that pre-registered penalty, not to a per-dataset CV-selected
   value. The Study's `PLAN.md` records one open pre-registration item on this
   key — a validation-selected `lambda_2` over the paper's two smallest grid
   values, awaiting the human decision — under which `0.01` is the value this
   config freezes.

## Running

```bash
cd wavCSE   # repo root, NOT this folder
uv run --locked python -m improvements.run_improvements \
  --model mssl --task_type ks_si_er \
  --config improvements/taskrelation/04-mssl/mssl_config.yml
```

Must be invoked as a module (`python -m improvements.run_improvements`), not as
a file path — its `from improvements....` imports only resolve with the repo
root on `sys.path`. The `mssl` arm is dispatched by
`improvements/run_improvements.py` (its config path is registered in
`CONFIG_PATH_OVERRIDES`, so `--config` may also be omitted). It is
deliberately **not** part of `--model all`: it is a pre-registered Study arm,
not a generic architecture sweep.

Run name and tags come from the config's `research:` block (Study ID, stage,
method, task set, representation, seed) and land in the MLflow experiment
`taskrelation-mssl` on the shared DagsHub server.

## Protocol

The arm runs under `../research/VARIANT_BENCHMARK_PROTOCOL.md`, fixed before
any candidate was implemented: `wavlm_large` with all 25 transformer layers,
frame pooling `mean`, layer pooling `smp` 0.5, task set `ks_si_er`, the shared
1024→512→2000 trunk with dropout 0.4/0.6, 30 epochs, batch 2048 with
`drop_last_train: true`, AdamW at lr 0.0025 / weight decay 5e-8, l1 1e-7 /
l2 1e-5, ReduceLROnPlateau factor 0.5 (the config's `patience: 5` key is inert
everywhere — effective patience is 1; it is copied verbatim, not fixed here),
best/opt/epoch checkpoints, `subset_percentage: 100`, 4 workers, seed 42.

Its two mandatory controls are the in-category classical MTRL arm
(`01-mtrl/mtrl_poolingwinner_25L_config.yml`) and the matched wavCSE baseline
(`improvements/base`). Per the protocol the arm has to beat **both** — and
any ER claim additionally requires speaker-independent LOSO (F3), because the
ordinary ER split leaks speakers.

## Evidence

`../research/studies/TR-0007/` is where this arm's evidence lives: the
pre-registration (`PLAN.md`) and run note (`NOTE.md`) and, per protocol §7,
every result and every attempted configuration. MLflow/DagsHub keeps the
execution record; the repository keeps the interpretation.
