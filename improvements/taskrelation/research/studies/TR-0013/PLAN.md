# TR-0013 — PLAN (pre-registration)

Status: **REGISTERED — screen approved by the researcher, executed under
`authorizations/TR-0013.yaml`.** Written 2026-09-29. Study registry: `../../STUDIES.jsonl`.
Study ID: `TR-0013`. Predecessor: `TR-0007` (`REJECTED`, `FL-0005`).
Arm identity: **faithful published-method implementation** (DEC-0013 §3, DEC-0014 §4, DEC-0015
Option A). No deviation is introduced by this study.

Authoritative design source: `../../proposals/TR-0013_published_lambda2_axis.md` (the draft the
researcher approved) and `../../literature_survey/POST_TR0007_SYNTHESIS.md` §8 S2. This plan
implements that design and does not extend it.

## 1. Observation motivating the study

`OBSERVED`: TR-0007 screened the faithful p-MSSL arm at a **researcher-fixed**
`lambda_2 = 0.01` (DEC-0016) and classified it `REJECTED` (0.9662 vs classical MTRL 0.9752 and
matched wavCSE baseline 0.9748 at the protocol checkpoint; SI −1.79pp / −1.67pp), with the
mechanism recorded as coupling-scale domination: the summary rows align to cos ≈ 0.981, `S`
becomes near-singular, Ω jumps to a near-uniform precision of trace 1.13e5 whose gradient
(`~1e5`) swamps the task gradients (`~1e-2`), and training plateaus from epoch ≈ 10.

`RECORD`: the θ2 value was never selected by the paper's own procedure. Algorithm 1 of the source
paper declares the penalty parameters "chosen by cross-validation" and its classification
experiments cross-validate over `{0.01, 0.1, 1, 10, 100}` (JMLR 17(33) §4.1); the paper publishes
no default and its numeric scale is not transferable between representations. The arm's own README
(deviation 3), `TR-0007/PLAN.md` and `DEC-0015` §3 all carry validation-selected `lambda_2` as an
open item, so TR-0007's `REJECTED` is scoped by its own caveats to a value the paper would have
chosen on data.

`DERIVED` (post-hoc, `studies/TR-0007/coupling_scale_result.json`): at the Grams the screen
actually reached, drawing `lambda_2` upward from `0.01` to `0.1` leaves the coupling's magnitude of
order `d` (gradient `1.6e5 → 5.6e4` at the collapsed geometry; `7 240 → 7 029` at the uncoupled
baseline's geometry), while `lambda_2 ≥ 1` removes all three conditional edges and still carries
`3d` on the diagonal. The axis is therefore predicted to move the **support**, not the scale.

## 2. Hypothesis (one primary hypothesis), competing explanation, third outcome

**H-λ (the faithful explanation, tested first).** A validation-selected `lambda_2` from the paper's
two smallest grid values restores a resolvable run: either the selected value removes the harmful
edge, or its shrinkage lowers the precision enough that the coupling stops dominating the task
loss, so the run's outcome becomes attributable to the relation object rather than to its own
regularizer.

**Competing explanation H-scale (the synthesis's primary prediction).** `lambda_2` cannot do that:
the coupling's magnitude is `lambda_2`-invariant for every value that retains an edge, and the
value that removes all edges still imposes `3d` on the diagonal, so the arm closes as a *faithful*
negative result — which is what would justify a scale-convention deviation (`TR-0012`) rather than
more `λ`.

**Pre-declared third outcome (non-exclusive).** Both values `{0.01, 0.1}` produce the same
collapse *and* the same support (0 exact off-diagonal zeros), i.e. the validation signal cannot
discriminate them at all. Then the `lambda_2` "selection" is vacuous at this scale, and that fact
is itself the study's result.

## 3. Independent variable and controls

**Changed variable (exactly one): the `lambda_2` *policy*.**
TR-0007 ran a researcher-fixed `0.01` (DEC-0016). TR-0013 runs the arm's own pre-registered
resolution: `lambda_2` is **validation-selected over the paper's two smallest published grid values
`{0.01, 0.1}`**, budget-matched to the in-category control's own two-value `mtrl_lambda` selection
(`0.01` vs `0.05`, same 16-layer campaign). Selection is made on the **validation** split only;
the test split is read once, after the freeze.

**Selection rule (pre-registered, before any run).**

1. Both values are executed as their own full 30-epoch runs at seed 42; every attempted value is
   recorded (`DEC-0006`, and the repository's "record every attempted configuration" convention).
2. The selection metric is the candidate's **`val_acc_all` at the protocol checkpoint (epoch 30)**
   — the same epoch the classification reads on test, on the arm's own validation loaders.
3. The selected value is the one with the higher `val_acc_all`.
4. **Tie rule**: if the two are within `0.20pp` of each other (the protocol's own per-task
   regression bar), the selection is declared **indiscriminate** and the smaller value
   (`lambda_2 = 0.01`) is used as the registered representative, because it is the arm TR-0007
   actually screened and the smaller penalty. An indiscriminate selection is reported as such.
5. The selection is recorded on the runs (`lambda_2_selection` tag/param), and the classification
   below names which run it used.

**Held constant (the protocol, `VARIANT_BENCHMARK_PROTOCOL.md` §1, unchanged from TR-0007):**

| Component | Setting |
|---|---|
| Upstream | `wavlm_large`, frozen; frame pooling `mean` |
| Layer pooling | `smp` 0.5 over **all 25 layers** (executably validated) |
| Task set | `ks_si_er` (KS 12-class, SI 1251-class, ER 4-class) |
| Shared trunk | 1024→512 → 512→2000, dropout 0.4 / 0.6 |
| Epochs | 30 |
| Global batch | 2048, `drop_last_train: true`, `subset_percentage: 100` |
| Optimizer | AdamW, lr 0.0025, weight decay 5e-8 |
| Regularization | l1 1e-7, l2 1e-5 |
| Scheduler | ReduceLROnPlateau factor 0.5 (the config's `patience: 5` is inert) |
| Checkpoints | best / opt / epoch; protocol checkpoint = `epoch` |
| Sampling / exposure | the same concatenated sampler; ≈ 2,820 steps per run in every arm |
| Summary adapter | the shared mean-head adapter (`normalize_w: true`) |
| MSSL internals | `lambda_0 = 1.0` (both half-steps), `lambda_1 = 0.0`, ADMM with residual balancing, float64, `d = W.shape[1]` inside the solver, split variable `Z` carrying the exact ℓ₁ support, per-solve optimality certificate, warmup 3, Ω update every epoch |
| Seed | 42 (screen) |

**Explicitly not changed** (the researcher's frozen list): `lambda_0`; the loss normalization; the
summary representation; phase gating; the architecture; the training budget; the evaluation
convention; the seed policy. TR-0012's scale correction is **not** introduced.

**Controls:** the historical classical MTRL arm (retained as the §2 in-category control by
`DEC-0017`) and the matched wavCSE baseline, at the same commit and protocol as the candidates.

## 4. Registered arms and runs (seed 42)

| Arm | Config | Experiment | `lambda_2` |
|---|---|---|---|
| `mssl-l2-0p01` (candidate) | `configs/mssl-l2-0p01.yml` | `taskrelation-mssl` | 0.01 |
| `mssl-l2-0p1` (candidate) | `configs/mssl-l2-0p1.yml` | `taskrelation-mssl` | 0.1 |
| `classical-mtrl` (control) | `configs/classical-mtrl.yml` | `taskrelation-mtrl` | n/a |
| `wavcse-baseline` (control) | `configs/wavcse-baseline.yml` | `wavcse-baseline` | n/a |

Four runs, not five: the approved draft's §4 wording ("candidate at `0.01` and candidate at `0.1`,
plus the two controls = 5 runs") counts one run twice — two candidate values plus two controls is
four. The fifth run the draft's cost line allowed for is retained as budget headroom, not as an
extra configuration; no λ value outside `{0.01, 0.1}` is run. This is recorded here so the count
cannot be read as a silent deviation.

Run names: `TR-0013__screen__<method>__ks_si_er__smp25__s42`, with `method` =
`p-mssl-l2-0p01` / `p-mssl-l2-0p1` / `classical-mtrl` / `wavcse-baseline`.

## 5. Mechanism measurements (recorded for every run)

In-run, per epoch, from the arm's own instrumentation (added to `04-mssl` by this study; it writes
no `.grad` buffer and is asserted not to change the optimizer step):

* `lambda_2`, `lambda_0`, `lambda_2_selection`;
* Ω: raw matrix, trace, eigenvalues, mean |off-diagonal|, exact off-diagonal zeros,
  **support/edge count**, partial correlations, ADMM `dual_violation` and `relative_duality_gap`;
* summary: Gram, pairwise cosines, Gram eigenvalues, row norms (normalised and raw);
* coupling value `lambda_0·tr(W Ω Wᵀ)` beside the task loss;
* **relation-gradient norm, task-gradient norm** (both over all trainable parameters and over the
  classifier heads) and their **ratio**, on the first training batch of every epoch;
* coupling onset epoch (derived: first epoch with a non-null relation value).

Post-hoc, from the stored checkpoints and metric curves of every arm (including the controls, which
have no in-run coupling instrumentation): summary Gram/row norms/cosines at each checkpoint, and
the plateau/onset shape of the validation curve.

## 6. Decision rule (pre-registered, unchanged from the family's rule)

Screen stage, one seed, read at the protocol checkpoint (`epoch`; `best`/`opt` are reported but do
not decide):

* `PROMISING` iff the selected candidate's `test_epoch_acc_all` > max(both controls'
  `test_epoch_acc_all`) **and** every per-task `test_epoch_*_acc` is within `0.20pp` of both
  controls;
* `REJECTED` otherwise;
* `INCONCLUSIVE` if an arm cannot run under the frozen conditions or a pre-declared exposure check
  fails.

A screen can never promote the arm (F1). Confirmation (seeds `0–4`), LOSO and any ablation are
separate, later stages requiring their own authorization, and are **not** part of this study.

## 7. Prediction and falsification (pre-registered before execution)

**Predicted (H-scale).** At `lambda_2 = 0.1`: summary cosines still exceed `0.9` within two epochs
of the coupling engaging; the coupling value stays `O(2e3)` per task; the relation/task gradient
ratio stays ≫ 10; training still plateaus; the support shrinks relative to `0.01` (predicted 2 of 3
edges gone on a healthy Gram, 0 of 3 on the collapsed one); the outcome stays `REJECTED` or
`INCONCLUSIVE`, with the same SI regression.

**Falsification of H-scale.** Any of: a run whose coupling-to-task gradient ratio stays below `10`
through training; summary cosines staying below `0.5`; a validation curve without the epoch-≈10
plateau; or a support change accompanied by an outcome that separates from both controls. Any of
these reopens the faithful route and would make `TR-0012`'s deviation unnecessary for the moment.

**Falsification of H-λ.** Both values reproduce the collapse and the plateau with the same support.

**Classification honesty.** Even `PROMISING` here would be *screening-tier*: the promotion bar
(beats both controls across seeds `0–4` with per-task paired intervals; LOSO for any ER claim) is a
separate, explicitly authorized stage. ER on the ordinary split is speaker-leaky context only and
is not an ER result (F3).

## 8. Cost, worker choice, authorization

* Five-run allowance, four runs executed; expected ≈ 1.5–1.8 GPU-h of training plus preparation,
  one worker, inside a 4 h envelope.
* Worker: the cheapest compatible part in the existing network volume's data center (EU-RO-1),
  discovered at execution time. At provisioning that was `NVIDIA L4` at `$0.49/h` (SECURE,
  direct SSH, `sm_89`), so it is the plan's pinned part; the next-cheapest compatible offer was
  `NVIDIA GeForce RTX 4090` at `$0.74/h`. Compatibility is a hard constraint: the pinned PyTorch
  (`runpod/pytorch:2.4.0-py3.11-cuda12.4.1`) tops out at `sm_90`, so Blackwell-class offers
  (`NVIDIA RTX PRO 4000/4500/6000 Blackwell`, `RTX 5090`) are excluded regardless of price —
  DG-0007 recorded that failure. One worker is reused for all four arms, sequentially.
* Container disk 100 GB (`container_disk_gb_max: 120` in the envelope): TR-0007 lost two
  job-preparation attempts to 60 GB container-disk exhaustion during checkout and materialization.
* Envelope: `authorizations/TR-0013.yaml` — 0.80 USD/GPU-h, 3.00 USD total, 4 paid wall-clock
  hours, one worker, replacements allowed, the existing network volume allowed, no new persistent
  resources, destroy on completion. It authorizes this screen only: not confirmation, not LOSO, not
  `TR-0012`, not `TR-0008`, not `DG-0007`, not any other λ value.

## 9. Executable validation (run before submission)

1. `research/tests/test_tr0013_protocol.py` — the four registered arms at seed 42, all 25 layers,
   the frozen protocol values, `lambda_2 ∈ {0.01, 0.1}` and nothing else changed relative to
   TR-0007's candidate and control configs, the study identity tags on every config, and the plan's
   declared inputs/outputs.
2. `research/tests/test_mssl_mechanism_probe.py` — the instrumentation is read-only: a training
   step with the probe produces bit-identical parameters and optimizer `.grad` state to the same
   step without it, and the recorded bundle carries the required keys.
3. `make check`, `make agents-check`, `git diff --check`, the import smoke, and the backend's own
   `plan`/`preflight` dry runs.

## 10. Category-boundary check

Unchanged from TR-0007: the arm is the published p-MSSL precision arm, a formal Task Relation
Learning mechanism (Zhang & Yang 2021 §2.4, family B). A `lambda_2` *policy* change is a protocol
correction inside the published method — the paper's own Algorithm 1 procedure — not a new
mechanism, not loss weighting, gradient surgery, low-rank, clustering or decomposition
(`FRAMEWORK.md` R7, `DEC-0005` §3).

## 11. What this study is not

Not a λ sweep (`{1, 10, 100}` is a different hypothesis, H-inert, and is excluded); not a scale fix
(that is `TR-0012`, and it is a deviation-class question the researcher has not approved); not a
change to `lambda_0`, the loss normalization, the summary adapter, the schedule, the architecture,
the protocol or the seed; not a confirmation, and not an ER result.
