# DG-0007 — Normalization-corrected classical MTRL under the matched protocol

Status: READY_FOR_REVIEW — pre-registered, no run, no MLflow run, no compute authorized
Type: diagnostic (implementation-faithfulness control for the in-category MTRL arm)
Created: 2026-09-29T00:00:00+00:00
Research family: Task Relation Learning
Predecessor: the historical classical-MTRL campaign (`improvements/taskrelation/01-mtrl/`, Study IDs `LEGACY-PRE-ID` evidence within `DG-0001`/`DG-0002` and the folder README's iteration log, plus the F4 multi-seed and LOSO tables)
Reason: theory-to-implementation audit — `../audits/2026-09-29-mtrl-theory-to-implementation-audit.md`
Correction: `model.normalize_w: true → false` (the published `Ω` closed form and the published relation regularizer are applied to the task parameter matrix `W`, not to a row-unit-normalized copy of it)
Protocol: `../../VARIANT_BENCHMARK_PROTOCOL.md`
Config: `improvements/taskrelation/01-mtrl/mtrl_norm_corrected_25L_config.yml`
Checker: `check_runtime_faithfulness.py` (this folder) — applies the runtime gate below
Authorization: **UNRESOLVED** — no authorization envelope covers this scope yet. This audit creates none. Whether DEC-0013's published-variant benchmark envelope extends to a corrected implementation of the existing control is an orchestrator/human call.

## Label — what this arm is and is not

**This is "normalization-corrected MTRL". It is not "faithful Zhang & Yeung
MTRL", and it must not be labelled that way.**

`model.normalize_w: true → false` removes exactly one adaptation, the audit's
finding **D2**. It does not remove the others, all of which remain in both arms
and all of which are departures from the published method:

* **D1** — `W` is the project's declared mean-head-summary adapter, not the
  paper's task parameter matrix; the shared trunk is outside `W`.
* **D3** — the paper's `(λ₁/2) tr(W Wᵀ)` term on `W` is **not** implemented.
  The trainer instead adds an inherited generic `l2_lambda · Σ_θ θ²` over *all*
  4,086,067 model parameters (62 % heads, 38 % shared trunk and pooling), with
  no `1/2` and no configurable λ₁. The two are **not equivalent**: the
  repository's term covers ≈681× more elements, includes parameters the paper's
  `W` excludes, and its measured magnitude is `1261.35` against a
  paper-shaped `(1/2)‖W‖² = 0.0551`. `mtrl_trainer.py`'s docstring presents the
  generic L2 as the paper's λ₁ term; it is not.
* **D4** — the `warmup_epochs` burn-in has no counterpart in the paper.
* **D6** — the ε-regularized inverse is a numerical floor, not the paper's
  exact `Ω⁻¹`.
* **D7** — `Ω` is refreshed once per epoch rather than the paper's
  "repeated until convergence" of the dual `W`-subproblem.

So the experiment isolates D2. If it moves, the movement is attributable to the
normalization; if it does not, F4 extends to *this* configuration, not to the
published method as a whole.

## Observation

The audit found that every evidence-carrying historical MTRL run used
`model.normalize_w: true`, which row-unit-normalizes the task parameter matrix
`W` inside both `update_omega()` and `get_mtrl_regularizer_loss()`. Measured on
the protocol configuration with tiny deterministic tensors:

* the code's `Ω` sits **50.11 %** above the published minimum of
  `min_Ω tr(Ω⁻¹WᵀW)` for the task-parameter matrix the model actually trains;
* the relation penalty is **scale-invariant** — `penalty(3W)/penalty(W) = 1.0031`
  instead of the published `9.0`; the 1.0031 is the `omega_epsilon` floor and
  tends to exactly `1` as `ε → 0` (measured: 1.0000317 at `ε = 1e-6`,
  1.0000005 at `ε = 1e-8`), so the normalized penalty carries *no* information
  about parameter scale and is confined to `λ·[m, m²]`;
* the loss of task-scale information is the exact statement: under
  normalization the diagonal tracks the summary directions but never the tasks'
  parameter magnitude (`diag(Ω) = [0.334, 0.332, 0.334]` against a 17.58×
  spread in `‖w_t‖`, and a 40× rescale of one task moves `Ω` by only 2.8e-3,
  the ε residual, versus 0.97 without it). `diag(Ω) ≈ 1/m` is *not* a general
  invariant of the normalized arm — the 25L `smp` Phase-A run reports
  `[0.3027, 0.3076, 0.3897]`;
* the paper's `Ω` is a covariance and does respond to scale
  (`[0.351, 0.035, 0.614]` in the corrected setting), and the historical
  `normalize_w: false` runs showed exactly that behaviour (`diag = [0.0115,
  0.1512, 0.8373]` at 60 epochs).

The normalization-corrected configuration has **never** been run under the
binding protocol (`smp` 0.5 over all 25 layers, 30 epochs, seeds 0–4, LOSO).
The only `normalize_w: false` historical evidence is at the 16-layer `weighted`
config, single seed, no LOSO — and that attribution rests on the folder
README's own narrative, not on a committed config (see the audit's `U1`).

**Provenance caveat inherited from the audit (`U3`).** The complete executable
repository state of the historical five-seed and LOSO MTRL runs is
`INSUFFICIENT_EVIDENCE`: the commit the run metadata records (`bfb1ad44`) does
not contain the seeding machinery or the LOSO entry point that those runs used,
so the recorded code identity cannot be the executed one. What *is* proven is
narrower and sufficient for this study's premise: `mtrl_model.py` and
`mtrl_trainer.py` are byte-identical from their introduction commit `33603c3`
(an ancestor of `bfb1ad44`) to the audit HEAD, so whatever tree those runs
executed, the row normalization was in it, and every committed MTRL config —
including the one at `bfb1ad44` — sets `normalize_w: true`.

## Research question

Holding pooling, layer set, epochs, batch size, optimizer, scheduler, splits,
checkpoint policy, `mtrl_lambda`, `omega_epsilon`, warmup and update frequency
fixed, does replacing the row-normalized relation objective with the published
one change what the classical-MTRL arm measures — against both its own
historical adapted implementation and the matched wavCSE baseline?

## Hypothesis (H1 — the modification is material)

The normalization-corrected arm's per-task and aggregate results and its learned `Ω` differ
from the historical adapted arm beyond the documented single-seed variability,
so the historical null result (F4) **cannot be transferred** to Zhang & Yeung's
published method, and the `Ω`-based conclusions of F5/F6 must be re-conditioned
on the modification.

Pre-declared materiality threshold: any task's absolute per-seed mean difference
`> 0.20pp` (the project's `no_material_regression_pp`), or a `Ω` diagonal that
departs from `1/m` by more than a factor of 2 for at least one task in every
confirmation seed.

## Competing explanation (H2 — the modification is immaterial)

The normalization-corrected arm reproduces the historical adapted arm within noise on every
task and seed. Then F4 transfers to the published method as well, and the
normalization was a benign repackaging of the same mechanism. This is the
outcome that would *strengthen* the historical conclusion rather than weaken it.

## Pre-declared third outcome (non-exclusive — the adapter binds)

The normalization-corrected arm is numerically dominated by the largest-norm head, as the
historical iterations 1–4 documented (`diag(Ω)` up to `0.837` on `er`, coupling
`mean|off-diagonal| ≈ 0.01`, `Ω⁻¹` nearly decoupling `er`). If that recurs here,
the result is a **diagnostic**: the binding constraint is the declared
mean-head-summary adapter, not the normalization, and no configuration
of this adapter can express the published relation object usefully. This
outcome authorizes no mechanism; it records a boundary of the control.

## Falsification condition

* **Runtime faithfulness gate (invalidates the arm, not the hypothesis).** In
  every run, the logged final `Ω` must be the minimizer of
  `tr(Ω⁻¹WᵀW)` for the **un-normalized** task-parameter matrix to within
  `1e-4` relative, satisfy `tr(Ω) = 1 ± 1e-6`, and the realized penalty must be
  quadratic in `W`'s scale. A run failing this gate is deleted from the
  comparison and reported as an execution failure. **Applied by
  `check_runtime_faithfulness.py`** (this folder), which reads the run's
  checkpoint, its final logged `Ω`, and the run-identity record the training
  process emitted (`improvements/run_identity.py`, `ARC_RUN_IDENTITY`) — and
  nothing else.

  *The checkpoint to pass.* `TrainerCheckpoint` builds its paths with
  `create_file_path` (`downstream/trainer/trainer_utils.py`): the base name is
  `train_<task_type>.pth`, the per-epoch files are
  `train_<task_type>_epoch{N}.pth` for epoch `N`, and `_best`/`_opt` variants
  exist alongside. `saved_checkpoint_count: 1` keeps only the latest epoch file,
  so at this 30-epoch protocol the protocol checkpoint (`epoch`) is
  `train_ks_si_er_epoch30.pth`.

  *Invocation.* The identity fields are the arm's own `research:` block values
  (below), so the expectations are stated explicitly and cannot silently differ
  from the config that ran:

  ```
  python check_runtime_faithfulness.py \
      --checkpoint <checkpoints_root>/<run>/train_ks_si_er_epoch30.pth \
      --identity <ARC_RUN_IDENTITY file for that run> \
      --omega-history <results_root>/<run>/omega_history.json \
      --normalize-w false --omega-epsilon 1e-4 \
      --study-id DG-0007 --stage screen \
      --method mtrl_norm_corrected --model mtrl --seed 42 \
      --task-type ks_si_er --representation smp25 \
      [--expect-commit <the commit the study record pins>]
  ```

  For the historical control arm use `--method mtrl --normalize-w true`; for the
  matched baseline `--method wavcse-baseline --model original` (with
  `--checkpoint <...>/train_ks_si_er_epoch30.pth` as well — the baseline arm runs
  the same entry point and the same trainer). In `confirm` stages pass
  `--stage confirm` and the seed being confirmed.

  Exit `0` = evidence is usable and the gate passed; `1` = evidence is usable and
  the gate failed (a scientific result); `2` = the evidence is unusable —
  corrupt, truncated, malformed, structurally inconsistent, or not evidence for
  the requested run. A malformed artifact is never reported as a gate failure,
  and wrong-run evidence is refused rather than scored.

  *The evidence pair.* The identity record and the artifacts must belong to the
  same run. Every run writes into its own timestamped directory and the record
  carries those absolute paths, so the checkpoint must live in the record's
  `checkpoints_dir` and `omega_history.json` in its `results_dir`; identity from
  one run paired with a checkpoint from another is refused (exit `2`) even when
  the two share the arm, task shape, representation and seed. The gate is
  therefore run where the run wrote its artifacts, or against a staged copy
  whose identity record has been re-rooted. Identity completeness is presence
  and emptiness, never truthiness: **`seed: 0` is a valid identity value**, as
  are `1`–`4`, since confirmation runs seeds `0,1,2,3,4`, while an absent field,
  `None`, an empty string or a non-integer seed are refused.

  *Clause character (recorded, not silently chosen).* Clause 3's phrase "the
  realized penalty is quadratic in `W`'s scale" admits two readings: with `Ω`
  held fixed it is `c²` for *any* positive-definite `Ω`, so that reading cannot
  discriminate and is not a test. The checker applies the reading in which `Ω`
  is re-derived from the scaled parameters through the run's own `W`
  construction, which is what the audit and its review measured. Under that
  reading clause 3 is an algebraic identity for a `normalize_w=false` artifact
  and clause 2 is an analytic identity for any artifact whose `Ω` is the closed
  form, so both confirm the *declared policy*; **only clause 1 discriminates on
  the artifact** — it is evaluated against the checkpoint's own head parameters
  and genuinely fails historical artifacts (measured gap > 40 %). The
  pre-registered sentence above is unchanged.

  *Residual limitation (recorded, not fixed here).* The artifact set carries one
  `Ω` stream, so the checker cannot detect a run whose `Ω` step and penalty step
  disagreed *in-process*: the penalty used `(Ω + εI)⁻¹` derived from the same
  buffer, and that inverse is not logged. Clause 1 is the artifact-level proxy.
  The diagnostics section below already requires the missing per-epoch
  instrumentation (`tr(Ω)`, `cond(Ω)`, the realized penalty, the raw row norms);
  once a run produces it, this limitation can be closed without changing the
  gate.
* **Numerical-stability gate (pre-registered).** Record `cond(Ω)`, the realized
  penalty and the regularizer's gradient norm per epoch. If `cond(Ω) > 1e6`, or
  the penalty exceeds `20×` its first post-warmup value, the run is marked an
  *instability observation* and excluded from the accuracy comparison before any
  accuracy number is read.
* **Support H1:** the faithfulness and stability gates pass, and at least one
  task shows an absolute per-seed mean difference `> 0.20pp` from the historical
  adapted arm with a consistent sign across ≥4 of 5 confirmation seeds.
* **Reject H1 (support H2):** the gates pass and every task stays within
  `0.20pp` of the historical adapted arm on the per-seed means.

## Independent variable

Exactly one: `model.normalize_w` — `true` in the historical control arm,
`false` in the normalization-corrected arm. Both arms run the same code, at the same commit,
from configs that are value-identical apart from this key and the run's output
directories (the files also differ in their comment headers). The isolation is
asserted by `research/tests/test_mtrl_theory_faithfulness.py`, class
`F_ConfigurationRecord`.

## Matched controls

1. **Historical adapted MTRL** — `mtrl_poolingwinner_25L_config.yml`
   (`normalize_w: true`), the currently registered in-category control.
2. **Matched wavCSE baseline** — `improvements/base/configs/base_poolingwinner_25L_config.yml`.

Both are re-run at this study's commit for the confirmation seeds so that all
three arms share one commit and one protocol.

## Controlled variables (protocol §1, unchanged)

`wavlm_large` frozen; frame pooling `mean`; layer pooling `smp` 0.5 over **all
25** layers (`selected_transformer_layers: all`, mechanically `[0..24]`);
`ks_si_er` (12 / 1251 / 4); shared trunk 1024→512→2000 with dropout 0.4/0.6;
30 epochs; global batch 2048 `drop_last_train: true`; AdamW lr 0.0025,
wd 5e-8; l1 1e-7, l2 1e-5; ReduceLROnPlateau factor 0.5, effective patience 1
(the `training.patience: 5` key is inert, as protocol §8 records — it is left
untouched in every arm); identical splits, `subset_percentage: 100`; protocol
checkpoint `epoch`; `mtrl_lambda: 0.01`, `omega_epsilon: 1e-4`,
`warmup_epochs: 3`, `omega_update_frequency: 1` in both MTRL arms.

## Evaluation protocol

* Primary endpoint: per-task and aggregate **test** accuracy at the protocol
  checkpoint `epoch`, seed-level, reported as mean ± SD over seeds `0–4`, paired
  per-seed differences against each control with 95 % t intervals and sign
  counts (protocol §4).
* The sample-weighted aggregate is dominated by SI and can mask ER; per-task
  results are reported alongside it.
* ER: the ordinary split leaks speakers and is screening context only. Any ER
  claim requires the speaker-independent LOSO protocol, which this study does
  not run unless the single-split screen shows a per-task ER movement above the
  documented `~±1.3pp` single-split noise. That escalation is **not** budgeted
  below; no defensible LOSO cost measurement exists on the record.
* Screening is screening: seed 42 may produce `PROMISING` or `REJECTED`, never a
  promoted claim (F1).

## Diagnostics

* `omega_history.json` per run (the existing artifact), plus a new per-epoch
  record of `tr(Ω)`, `cond(Ω)` in float64, the realized penalty
  `λ·tr(WᵀΩ_ε⁻¹W)` on the raw `W`, and the raw row norms `‖w_t‖`.
* The audit's invariants are checked on the *logged* `Ω` and the checkpoint's
  head parameters after training by the committed
  `check_runtime_faithfulness.py`, not only on a synthetic model.
* Regularizer gradient norm and its share of the total head gradient, per task,
  at the first post-warmup step and at the final epoch — the graded consequence
  of the mean-head adapter (`1/C_t` per-element scaling).

## Screening protocol

Seed 42, three arms, concurrent on separate GPUs:
`A0` matched baseline, `A1` historical adapted MTRL (`normalize_w: true`),
`A2` normalization-corrected MTRL (`normalize_w: false`). ≈17–19 min per arm at this protocol.
Promote only when both gates pass.

## Confirmation protocol

Seeds `0,1,2,3,4`, three arms per seed = 15 runs at this study's commit,
distributed per project policy (GPU 0: seeds 0, 2, 4; GPU 1: seeds 1, 3).
Report per-seed results, mean/SD, paired arm-minus-control differences with 95 %
intervals, and sign consistency. Each seed is the independent unit.

## Compute estimate

**Single-split screen and confirmation (the only budget this plan requests):**

* Screening: 3 runs (seed 42) ≈ 1.1 GPU-hours.
* Confirmation: 15 runs (3 arms × seeds 0–4) ≈ 5.3 GPU-hours.
* Total bounded by ≈ 6.5 GPU-hours, at most 2 concurrent jobs.

**LOSO escalation (separate, not requested, and deliberately uncosted):**

* Reached only if the single-split screen shows a per-task ER movement above the
  documented `~±1.3pp` noise, per the evaluation protocol above.
* Structure if it is ever reached: the ER split folds over 10 speakers, only
  `er`/IEMOCAP is folded and training stays joint `ks_si_er`, so an escalation
  is 10 folds per arm at the 5-epoch fold budget (`mtrl_kfold_config.yml`,
  `base_kfold_config.yml`) — 30 runs for the three arms, not 3.
* **Cost: UNKNOWN.** No wall-clock or GPU-hour measurement for the k-fold
  protocol exists anywhere on the record, and the 5-epoch fold budget is a
  different shape from the 30-epoch single-split runs the ≈0.35 GPU-hours/run
  figure above is derived from. No number is invented here. A measured estimate
  must be produced (e.g. from one timed fold per arm) and recorded before any
  LOSO escalation is authorized.
* The 6.5 GPU-hours above therefore covers the screen and confirmation only.

No compute is requested, reserved or authorized by this plan.


## GPU allocation

Not allocated. Screening would use GPU 0 = A0, GPU 1 = A1 first, then A2; the
protocol's concurrency limit of two simultaneous jobs applies, and `df -h` and
`nvidia-smi` must be checked immediately before launch per `AGENTS.md`.

## Verified codebase facts (read 2026-09-29 at `63b639f`)

1. `mtrl_model.py` and `mtrl_trainer.py` are byte-identical to their
   introduction commit `33603c3` (`git diff --stat 33603c3 HEAD` is empty), and
   `33603c3` is an ancestor of `bfb1ad44`, the commit the historical run
   metadata records. **What this proves and what it does not** (audit `U3`):
   the *files* implementing the row normalization are unchanged across the
   whole relevant history, so every tree that ran MTRL contained them. It does
   **not** prove the complete executable repository state of the historical
   five-seed and LOSO runs: `bfb1ad44` contains neither the seeding machinery
   (`improvements/seed_utils.py`, `--seed`, `cfg["seed"]` all arrive later, in
   `03c28fe`) nor the LOSO entry point and config (`mtrl_er_kfold.py`,
   `mtrl_kfold_config.yml`, arriving in `03c28fe`/`c604cc9`), so those runs
   cannot have executed that commit as committed. The provenance of the
   executed tree is `INSUFFICIENT_EVIDENCE`; a dirty checkout or a wrong
   recorded commit are both possible and neither is established. The audit does
   **not** infer that the runs did not happen, and it invents no replacement
   commit. The correction is still a config value, not a new code path.
2. `build_model` (`run_improvements.py:137-146`) forwards
   `normalize_w=model_cfg.get("normalize_w", False)`, and
   `mtrl_er_kfold.py:132` does the same, so the flag reaches the model in both
   the single-split and LOSO paths. No wiring defect exists.
3. `parse_transformer_layers("all", "wavlm_large")` returns `list(range(25))`;
   `smp` pooling softmaxes over the layer axis, so all 25 slots participate.
4. `_process_batch` adds the MTRL term whenever
   `current_epoch >= mtrl_warmup_epochs`, **including on the validation path**;
   `ReduceLROnPlateau` therefore steps on a loss containing the regularizer.
   This is shared in kind with the base trainer's L1/L2 handling and is left
   unchanged in every arm so as not to introduce a second variable.
5. `update_omega()` writes `self.omega` (exact, `tr = 1`) and
   `self.omega_inv = (Ω + 1e-4 I)⁻¹`; the logged `Ω` is the exact matrix.

## Category-boundary check

Task Relation Learning, and specifically a faithfulness control rather than a
mechanism. It changes no relation object, no estimator and no update rule; it
removes a project-introduced transformation of the published objective. It is
not loss weighting, gradient surgery, low-rank, clustering or decomposition,
and it authorizes no mechanism (`BACKLOG.md` taxonomy gate; `DEC-0013`). It does
not increment the plateau counter.

## Expected information gain

High per GPU-hour. It decides what the programme's first negative result is
about:

* **H1 →** the historical MTRL evidence must be re-attributed: it is evidence
  about a project modification of MTRL, the `Ω`-shaped findings of F5/F6 are
  re-conditioned on the modification, and the in-category control named by
  `VARIANT_BENCHMARK_PROTOCOL.md` §2 needs a `DEC`-level correction.
* **H2 →** F4 transfers to the published method, and the audit closes as a
  documentation correction only.
* **Adapter-bound outcome →** the mean-head summary, already a declared
  deviation, is identified as the binding constraint, which is a stronger and
  more useful statement for any future family-B arm than either accuracy
  outcome.

Either way the study produces the evidence a `DEC` decision on the protocol's
in-category control needs, without touching the historical record.

## Provenance

Run naming follows protocol §7:
`DG-0007__{screen|confirm}__{mtrl_norm_corrected|mtrl|wavcse-baseline}__ks_si_er__smp25__s{seed}`,
with the standard tags (`study_id`, `stage`, `method`, `category`, `model`,
`seed`, `pooling`, `layers`, `git_commit`, …) and a DagsHub run note. The
implementation and configs are committed before any launch, and the executed
SHA is recorded here and in `STUDIES.jsonl`.

**Execution configs.** The three arms are launched through the one study runner,
as DG-0002's convention does, against committed study configs that carry the
`research:` identity block:

```
# screen
python -m improvements.run_improvements --model mtrl     --task_type ks_si_er \
    --config improvements/taskrelation/research/studies/DG-0007/configs/norm_corrected_mtrl.yml \
    --device_index <n>
python -m improvements.run_improvements --model mtrl     --task_type ks_si_er \
    --config improvements/taskrelation/research/studies/DG-0007/configs/historical_mtrl.yml \
    --device_index <n>
python -m improvements.run_improvements --model original --task_type ks_si_er \
    --config improvements/taskrelation/research/studies/DG-0007/configs/baseline.yml \
    --device_index <n>
# confirmation: the same three with configs/confirm_*.yml and --seed {0..4}
```

Each config is its source config (the architecture folder's config for the two
MTRL arms, `improvements/base/configs/` for the baseline) plus the `research:`
block and per-arm output roots, and nothing else; both the identity and that
isolation are asserted by `research/tests/test_dg0007_run_identity.py`, which
runs the real `build_research_run_name` / `resolve_run_note` /
`set_standard_tags` / `run_identity.research_identity` helpers. Together with
`run_improvements.py`'s `emit_run_identity`, that makes every run carry
`study_id`, `stage`, `method` (the arm), `model`, `task_type`, `representation`,
`seed` and `git_commit` — in the MLflow run name and tags and in the
`ARC_RUN_IDENTITY` record the gate binds evidence to. The `method` field is what
separates the two MTRL arms, since both run with `--model mtrl`.

**Embedding materialization (pre-run).** The all-25-layer policy is verified
mechanically (parser, configs, pooling, and that a short store fails loudly
rather than being silently subset — see `test_dg0007_run_identity.py`), but the
canonical embedding store's own row count is not part of any metadata contract
the compute plan carries: `improvements/compute/embedding_layout.py` validates
each input by artifact name, `sha256`, `size_bytes` and file count only. The
store is upstream-frozen and absent from this environment, so no committed test
can assert its 25 rows and none is invented. This is therefore a
**pre-run materialization/runtime validation**: before any paid evidence is
accepted, the materialized store must be confirmed to have 25 layer slots per
utterance (a short store raises `IndexError` in the loader rather than silently
training on a subset, so the check is cheap and loud).

---

## Decision record (post-registration, 2026-09-29) — `DEC-0017`: RETAIN

Appended after the pre-registration above was written. **No element of the pre-registered
design changed** — not the hypothesis, the independent variable, the three arms, the seeds,
the gates, the decision rules, the cost, or the authorization status.

The open question this plan carried — whether the normalization-corrected configuration
should *replace* `VARIANT_BENCHMARK_PROTOCOL.md` §2's in-category control or remain an
additional arm — was put to the researcher and answered **`DEC-0017` (OPTION B, RETAIN)**:

* historical MTRL (`mtrl_poolingwinner_25L_config.yml`, `normalize_w: true`) remains §2.1's
  in-category control and the reproducibility anchor; it is **not** replaced or redefined;
* the normalization-corrected configuration (`mtrl_norm_corrected_25L_config.yml`,
  `normalize_w: false`) remains a **distinct `DG-0007` successor experimental arm**;
* whether the corrected configuration should later *become* the standing control is a
  separate decision for this Study's evidence, not this one.

Two forward-looking passages above are affected only in their *conditionality*, and neither
is rewritten:

* **"Matched controls"** — arm 1 stands exactly as written: the historical adapted MTRL
  config at `normalize_w: true`, the registered in-category control.
* **"Expected information gain" (H1 →)** — the `DEC`-level consequence of H1 is narrowed by
  `DEC-0017` to the *labelling and citation* of the control's `D2` modification. Adopting the
  corrected arm as the standing control would be a further decision taken on this Study's
  evidence, which this plan does not presuppose.

Authorization is unchanged and still **UNRESOLVED**: `DEC-0017` authorized no compute and
created no envelope, so this Study remains `BLOCKED` / `pre_registration` and nothing
described here may be launched.
