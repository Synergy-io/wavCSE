# Theory-to-implementation audit — classical MTRL (`improvements/taskrelation/01-mtrl/`)

*Audit date: 2026-09-29. Branch: `research/mtrl-theory-audit`. Base commit: `63b639f`
("Register TR-0007 and stop it at the faithfulness gate").
Read-only with respect to history: no historical metric, manifest, run identity,
config, checkpoint or record was modified. The corrected successor is a **new**
study and a **new** config.*

> **Corrected 2026-09-29 after independent review `5f72acd`.** Two claims in the
> first edition (`f05539c`, `2c0391e`) were withdrawn and replaced: the
> historical-execution provenance (§4.0, `U3`) and the regression-coverage
> claim (§9.3). One adjacent MSSL transcription error on this branch was
> repaired (§9.5). The central finding, D2, and its measured magnitudes are
> unchanged and were independently reproduced. Full log: **§9**.

---

## 1. What was audited

The classical MTRL arm of the Task Relation Learning programme, i.e.
`improvements/taskrelation/01-mtrl/` at HEAD:

| Artifact | Identity |
| --- | --- |
| Model | `improvements/taskrelation/01-mtrl/mtrl_model.py` (`DownstreamMultiTaskModelMTRL`) |
| Trainer | `improvements/taskrelation/01-mtrl/mtrl_trainer.py` (`MultiTasksModelTrainerMTRL`) |
| Entry point | `improvements/run_improvements.py --model mtrl` (`build_model`, `build_trainer`) |
| LOSO entry point | `improvements/taskrelation/01-mtrl/mtrl_er_kfold.py` |
| Task set | `ks_si_er` — KS 12 classes, SI 1251 classes, ER 4 classes |
| Method claim | "Classical symmetric MTRL" (`VARIANT_BENCHMARK_PROTOCOL.md` §2), attributed to Zhang & Yeung (UAI 2010 / TKDD 2014) |

## 2. The primary source and the exact method claimed

**Yu Zhang and Dit-Yan Yeung, "A Regularization Approach to Learning Task
Relationships in Multi-Task Learning", ACM TKDD 8(3), Article 12, pp. 1–31,
2014**, DOI 10.1145/2538028 — the journal version of the UAI 2010 paper
("A Convex Formulation for Learning Task Relationships in Multi-Task
Learning", UAI 2010; footnote 1 of the journal version).

Both versions were read directly for this audit (UAI 2010 author copy
`event.cwi.nl/uai2010/papers/UAI2010_0144.pdf`; journal author copy
`yuzhanghk.github.io/papers/Zhang_Yeung_TKDD14.pdf`). The repository's own
verified card is `literature/zhang-yeung-2014-mtrl-asymmetric.md`, which quotes
the same equations and was cross-checked against the PDFs here.

### 2.1 Published mathematics (implementation-oriented specification)

Notation of the paper: `m` tasks, `d` features, `W = (w_1, …, w_m) ∈ R^{d×m}`
— **columns are the task parameter vectors**, so `W` is the *task parameter
matrix* in a *shared* feature space; `Ω ∈ R^{m×m}` is the task covariance.

| # | Published requirement | Source |
| --- | --- | --- |
| P1 | The relation object is a symmetric PSD task **covariance** `Ω` (`m×m`), a global (not local/clustered) model of task relations | §1, §2.1 |
| P2 | Objective: `min_{W,b,Ω} Σ_i (1/n_i) Σ_j (y_j^i − w_iᵀx_j^i − b_i)² + (λ₁/2)·tr(W Wᵀ) + (λ₂/2)·tr(W Ω⁻¹ Wᵀ)` | Eq. (8) (UAI Eq. (7) states the same problem with `tr(Ω) = 1`). The TKDD text gives `λ₁ = 2ε²/ϵ²`, `λ₂ = 2ε²`; the UAI copy's coefficient line renders unreadably in both available extractions, so no UAI coefficient is claimed |
| P3 | Constraints: `Ω ⪰ 0` and `tr(Ω) ≤ 1`; `c` is "simply set to 1". The optimum satisfies `tr(Ω) = 1` (footnote 2, used to scale `Ω̃` in §2.3) | Eq. (7), (8); UAI writes `tr(Ω) = 1` |
| P4 | **Two** regularizers: an unweighted-complexity term on `W alone` (`(λ₁/2)tr(W Wᵀ)`, "penalizes the complexity of W") and the relation term (`(λ₂/2)tr(W Ω⁻¹ Wᵀ)`) | §2.1, Eq. (8) |
| P5 | Loss is task-balanced by `1/n_i`, i.e. each task contributes equally | Eq. (8) |
| P6 | `W`-subproblem (Ω fixed) is an unconstrained convex problem whose exact solution is a linear system / dual QP with multi-task kernel `k_MT(x₁,x₂) = e_{i₁}ᵀ Ω(λ₁Ω + λ₂I_m)⁻¹ e_{i₂} · x₁ᵀx₂`, solved by SMO | Eq. (9)–(13) |
| P7 | `Ω`-subproblem (W fixed): `min_Ω tr(Ω⁻¹ Wᵀ W)` s.t. `Ω ⪰ 0, tr(Ω) ≤ 1` | Eq. (14) |
| P8 | Closed form: **`Ω = (Wᵀ W)^{1/2} / tr((Wᵀ W)^{1/2})`**, attained minimum value `(tr((WᵀW)^{1/2}))²`; the algebra is the Cauchy–Schwarz argument in §2.2 | Eq. (14) |
| P9 | The `Ω`-subproblem is **independent of λ₁ and λ₂** — the closed form contains no regularization coefficient | Eq. (14) |
| P10 | Alternating optimization: solve for `W,b` with `Ω` fixed, then for `Ω` with `W,b` fixed, "repeated until convergence"; each subproblem convex | §2.2 |
| P11 | Initialization `Ω₀ = (1/m) I_m` ("all tasks are unrelated initially") | §2.2 |
| P12 | Substituting `Ω*` makes the third term "related to the **trace norm**" of `W` | §2.2 |
| P13 | `Ω` is a *function* of `W` under the alternating scheme; it is never updated by gradient descent | §2.2 |
| P14 | Experiments are shallow: linear/kernel regression and binary classification with aligned per-task weight vectors of equal length `d`; λ₁, λ₂ chosen by 5-fold CV on a predefined grid; convergence "no more than 15 iterations" | §4.1–§4.4 |

Repository-specific (declared, not in the paper): the **adapter** needed for
disjoint, heterogeneous heads. `W` is built as one row per task from
`[mean(head.weight, dim=0), mean(head.bias)]` — a `(hidden+1)`-length
*mean-head summary direction*. This is the project's own device, declared in
`literature/zhang-yeung-2014-mtrl-asymmetric.md` ("Gate 3 — heterogeneous
heads: PASS (as the control). Already solved in this project by mean-head
parameter summaries") and documented in `mtrl_model.py` and the folder
`README.md`. **This audit treats the adapter as given and does not test it.**

## 3. Paper → implementation mapping

Runtime call path traced end to end:
`run_improvements.main` → `build_model("mtrl", …)` (`run_improvements.py:137-146`)
→ `DownstreamMultiTaskModelMTRL` (`mtrl_model.py`)
→ `build_trainer("mtrl", …)` (`run_improvements.py:199-220`)
→ `MultiTasksModelTrainerMTRL.train()` (`mtrl_trainer.py`) → `_process_batch`
→ `get_mtrl_regularizer_loss` / `update_omega`, AdamW over all parameters.

| Published requirement | Repository implementation | Verdict |
| --- | --- | --- |
| P1 symmetric PSD `m×m` `Ω` | `register_buffer('omega')`, `[num_tasks, num_tasks]`, symmetric PSD by construction (`eigh`-based matrix square root) | **MATCH** |
| P2 relation term `(λ₂/2) tr(WΩ⁻¹Wᵀ)` | `mtrl_lambda * tr(Wᵀ Ω_ε⁻¹ W)` where `Ω_ε = Ω + εI` | **MATCH up to parameterization**: `mtrl_lambda ≡ λ₂/2` is a free-coefficient relabeling (λ is a selected hyperparameter); `Ω_ε` is a documented numerical floor, see D6 |
| P3 `Ω ⪰ 0`, `tr(Ω) = 1` | PSD by construction; `new_omega = sqrt_Ω / tr(sqrt_Ω)` ⇒ `tr(Ω) = 1` exactly; initial `tr = 1` | **MATCH** |
| P4 separate `(λ₁/2)tr(WWᵀ)` | not implemented as an MTRL term; a generic `l1_lambda·Σ|θ| + l2_lambda·Σθ²` over **all** model parameters (inherited verbatim from the frozen base trainer, identical in the baseline and MTRL arms) stands in | **DEVIATION (D3)** |
| P5 task-balanced loss | `loss_weight = 1/num_tasks` over per-task masked batch means (base-trainer convention, arm-matched) | **MATCH** |
| P6 `W`-subproblem solved to convergence by a dual QP | SGD/AdamW over mini-batches with `Ω` fixed | **DOCUMENTED APPROXIMATION (D7)** — the paper is shallow (linear model, ≤ 15 outer iterations); a deep model cannot run a dual SMO over the trunk. Alternating structure preserved |
| P7 `min_Ω tr(Ω⁻¹WᵀW)`, `Ω ⪰ 0`, `tr(Ω) ≤ 1` | `update_omega()`: `WWt = W @ W.T` (`[m,d]` row convention, so `W@Wᵀ ≡ W_paperᵀ W_paper`), `sqrt_Ω = V diag(√λ) Vᵀ`, `Ω = sqrt_Ω / tr(sqrt_Ω)` | **MATCH for the `W` it is given** — see §4 |
| P8 closed form `(WᵀW)^{1/2}/tr((WᵀW)^{1/2})` | bit-exact to the analytic expression (probe: max abs difference `0.0`); the attained penalty equals the published minimum `(tr A^{1/2})²` to float32 | **MATCH** |
| P9 `Ω`-step independent of λ | `update_omega()` reads no coefficient | **MATCH** |
| P10 alternating, `Ω` then `W` | `train()` updates `Ω` at the end of each epoch post-warmup (`omega_update_frequency`), `W` every batch | **MATCH (approximation D7)** |
| P11 `Ω₀ = I_m/m` | `torch.eye(num_tasks)/num_tasks` (and `omega_inv = (I/m + εI)⁻¹`) | **MATCH** |
| P12 trace-norm reading | verified numerically: with `Ω = Ω*` the term equals `λ·(tr(WᵀW)^{1/2})²` | **MATCH** |
| P13 `Ω` never gradient-updated | `register_buffer`, `@torch.no_grad()` on `update_omega`, absent from `optimizer.param_groups` | **MATCH** |
| P14 shallow/aligned model | deep trunk + heterogeneous heads with the declared mean-head adapter | **DECLARED ADAPTER (not tested here)** |

Tensor orientation was checked explicitly and is **correct**: the module
docstring reasons about the row convention, and the probe confirms
`tr(W_codeᵀ Ω⁻¹ W_code) = tr(W_paper Ω⁻¹ W_paperᵀ)` with `W_paper = W_codeᵀ`.
Gradients reach the head parameters (live, non-detached `W`) and never the
shared trunk — consistent with `W` being the task-specific parameters.

## 4. The finding: `normalize_w: true` is applied in every historical run

Every committed MTRL config that carries evidence sets
`model.normalize_w: true`:

| Config | Introduced | Pooling / layers | `normalize_w` |
| --- | --- | --- | --- |
| `mtrl_config.yml` | `33603c3` | `weighted`, 16 layers | `true` |
| `mtrl_alllayers_config.yml` | `6d35ea9` | `weighted`, all 25 | `true` |
| `mtrl_poolingwinner_16L_config.yml` | `73c055b` | `smp` 0.5, 16 layers | `true` |
| `mtrl_poolingwinner_25L_config.yml` | `73c055b` | `smp` 0.5, all 25 | `true` |
| `mtrl_kfold_config.yml` | `c604cc9` | `smp` 0.5, all 25, 5 epochs | `true` |

`build_model` forwards the key (`run_improvements.py:144-145`) and
`mtrl_er_kfold.py:132` does the same, so the flag genuinely reaches the model
in both the single-split and LOSO paths. There is no wiring defect.

With the flag set, `get_task_parameter_matrix()` row-normalizes `W` **before**
it is used, so *both* `update_omega()` and `get_mtrl_regularizer_loss()`
operate on `W~ = W / (‖w_t‖ + ε)`, not on `W`.

### 4.0 Historical execution identity (what the evidence actually ran)

| Item | Value | Source |
| --- | --- | --- |
| Historical study identity | `LEGACY-PRE-ID` (predates Study-ID tagging; `FINDINGS.md` R2, `FAILURES.md`) | `task_relations/MTRL_DIAGNOSTIC_SYNTHESIS.md` |
| Commit recorded by the five-seed and LOSO run metadata | `bfb1ad44ef987e6484183eda7d782f28cec5c667` — **recorded, but not established as the executed tree.** See `U3`: that commit cannot have executed those protocols as committed | `task_relations/MTRL_DIAGNOSTIC_SYNTHESIS.md` (quoting MLflow metadata) |
| Implementation commit for the `lnp` control comparison | `ae0b15ef` | same |
| **Proven (A)** | `mtrl_model.py`/`mtrl_trainer.py` are byte-identical to their introduction commit `33603c3` and to HEAD (`git diff --stat 33603c3 HEAD` empty for both files), and `33603c3` is an ancestor of `bfb1ad44`. So the two files implementing the row normalization were present in every tree across the relevant history | this audit |
| **Not proven (B)** | The complete executable repository state of the five-seed and LOSO runs. `bfb1ad44` contains neither the seeding machinery (`improvements/seed_utils.py`, `--seed`, `cfg["seed"]` — all arriving in `03c28fe`) nor `mtrl_er_kfold.py` / `mtrl_kfold_config.yml` (arriving in `03c28fe` / `c604cc9`), so the recorded protocols are not executable at the recorded commit. Two possibilities are consistent with the evidence and neither is established: the runs were launched from a **dirty checkout** whose HEAD was `bfb1ad44`, or the **recorded commit is wrong**. The runs are *not* inferred not to have happened, and no replacement commit is invented | this audit; review `5f72acd` §R1 |
| Executed hyperparameters | `mtrl_lambda = 0.01`, `omega_epsilon = 1e-4`, `normalize_w = true`, `warmup_epochs = 3` (30-epoch configs) / `1` (5-epoch LOSO config), `omega_update_frequency = 1` | all committed configs |
| Layers / pooling | `smp` 0.5, `selected_transformer_layers: all` → `[0..24]` | `parse_transformer_layers` + configs |
| Seeds | 42 (screen), `0,1,2,3,4` (confirmation); LOSO = 10 speaker folds, 5 epochs each | `STUDIES.jsonl`, synthesis |
| Historical results | 25L MTRL vs 25L baseline, paired over 5 seeds: KS `−0.00079` `[−0.00191, +0.00033]`, SI `+0.00029` `[−0.00235, +0.00293]`, ER `−0.01049` `[−0.01910, −0.00188]`, aggregate `−0.00056` `[−0.00243, +0.00130]`; LOSO ER `0.63797 ± 0.04837` vs `0.63907 ± 0.05065`, Δ `−0.00110` | synthesis evidence table |
| Not recoverable from git | the intermediate `normalize_w` state of iterations 1–4 of the 16-layer campaign (the implementation commit squash already records `true`); see U1 | this audit |

### 4.1 Measured consequence (float64 checks, `smp` 0.5 configs, tiny deterministic model)

| Quantity | Historical control (`normalize_w: true`) | Corrected (`normalize_w: false`) | Published requirement |
| --- | --- | --- | --- |
| `tr(Ω⁻¹ A_raw)` attained for the task parameter matrix actually differentiated | `0.32998297` | `0.21982526` | — |
| published minimum `(tr A_raw^{1/2})²` | `0.21982508` | `0.21982508` | P5/P8 |
| relative gap to the published optimum | **50.11 %** | **0.00 %** | gap must be 0 |
| `penalty(3W) / penalty(W)` | **1.0031** | **9.0000** | 9 (quadratic in `W`) |
| `penalty / λ` (m = 3) | `8.9542` | `0.2198` | structural bound `[m, m²] = [3, 9]` applies only to the normalized form |
| `diag(Ω)` | `[0.334, 0.332, 0.334]` | `[0.351, 0.035, 0.614]` | must be informative about task scale |
| raw row norms `‖w_t‖` | `[0.1648, 0.0164, 0.2877]` (spread 17.58×) | same | — |

`penalty(3W)/penalty(W)` is `1.0031` and **not** exactly 1 because the row
normalization carries the model's `omega_epsilon` floor. It is `ε`-limited and
tends to exactly 1 as `ε → 0` (measured: 1.0031278 at `ε = 1e-4`, 1.0000317 at
`1e-6`, 1.0000005 at `1e-8`); the separation from the published `9.0` is
unaffected. These magnitudes were independently reproduced in float64 by the
review of this audit (commit `5f72acd`), which reports the same values and,
over seeds 0–11, a gap of `+48.26 % … +52.07 %` and a scale ratio of
`1.0031–1.0032`.

Three mathematical facts follow, each verified:

1. **`Ω` is not the minimizer of the published relation subproblem for the
   parameters the model trains.** Row normalization is a composition that maps
   the `[m,d]` parameter matrix onto the unit sphere of each row, so the code
   solves `min_Ω tr(Ω⁻¹ W~ᵀW~)` while the interesting object — and the paper's
   `W` — is `W`. The gap is 50 % here, not a rounding-level difference.
2. **The relation penalty stops being the published penalty.** `normalize_w`
   makes `tr(W~ᵀΩ⁻¹W~)` invariant under any rescaling of `W`. With unit-norm
   rows the Gram `W~W~ᵀ` is a correlation-type matrix with unit diagonal, so
   `tr(Ω⁻¹ A)` is confined to `[(tr A^{1/2})²] ∈ [m, m²] · λ` independently of
   how the head parameters grow or shrink. The paper's term is a *degree-2
   homogeneous* penalty on `W` — and, after substituting `Ω*`, exactly the
   squared trace norm (P12) that supplies the "complexity of `W`" pressure the
   paper also assigns to `λ₁`. The normalized form supplies neither.
3. **The normalized `Ω` cannot express task *scale* — the exact statement.**
   `diag(Ω) ≈ 1/m` is **not** a general invariant of the normalized arm: the
   25L `smp` Phase-A run reports `diag(Ω) = [0.3027, 0.3076, 0.3897]`, and in
   general `diag(Ω) = diag(A^{1/2})/tr(A^{1/2})` for a unit-diagonal `A` can be
   anything feasible. What the normalization removes is the tasks' parameter
   *magnitude*. Measured: a 40× rescale of one task's parameters moves `Ω` by
   `2.8e-3` relative under `normalize_w` (the `ε` residual, which vanishes with
   `ε`) against `0.97` without it — where the diagonal reorganizes from
   `[0.351, 0.035, 0.614]` to `[0.149, 0.591, 0.260]`. The paper's `Ω` is a
   covariance and does respond to scale; the historical 60-epoch
   `normalize_w: false` run shows the same sensitivity
   (`diag(Ω) = [0.0115, 0.1512, 0.8373]`, reported in the folder `README.md`).

### 4.2 Why this matters for the *interpretation*, not just the number

* The structural attractor of the normalized penalty is **full ±collinearity**
  of the three mean-head directions: `min tr(W̃ᵀΩ⁻¹W̃)` is attained at
  `W~W~ᵀ = s sᵀ`, which forces `Ω = s sᵀ/m`, i.e. `mean|off-diagonal| = 1/m`
  with `diag(Ω) = 1/m` *at that attractor* (not in general — see §4.1 fact 3).
  The historical `16L weighted` run reports exactly
  `Ω` with off-diagonals `±0.3331` and diagonals `0.333x` — the *global
  optimum of the modified objective*, reached at saturation.
* The project's own diagnostic readings — "KS↔SI saturates positive in 5/5" in
  the 25L confirmation arm, and Ω magnitude "saturated in 5/5 seeds"
  (DG-0002, `FINDINGS.md` F6/F10) — are therefore **at least partly a property
  of the modification rather than an empirical discovery about KS/SI/ER**:
  saturation is where the modified penalty's own minimizer sits. The 16-layer
  `weighted` arm is the clearest case, because its final Ω is *numerically the
  exact attractor*: off-diagonals `±0.3331`, diagonals `0.333x`, sign pattern
  `s = (+1, −1, +1)` — i.e. `Ω = s sᵀ/3` to float32 precision, which for
  unit-norm rows is attainable *only* at `W~W~ᵀ = s sᵀ` (perfect ±collinearity
  of the three mean-head directions). Conversely, the LOSO arm of the same
  campaign (5 epochs per fold, so five Ω updates) reports the KS↔SI edge as
  *not* saturated (mean `0.286`, range `0.259–0.305`), so the degree of
  saturation is a function of optimization progress, not a stable estimate —
  consistent with an attractor rather than a measurement.
* The stability contrast the project reports — KS↔SI stable near `+1/3` while
  the ER edges flip sign across folds — is exactly the pattern the normalized
  objective predicts when two of three directions align and one is noisy, so it
  can no longer be read as an unmediated property of the tasks. This does
  **not** create a positive result; it weakens an interpretive one.
* Conversely, in the normalization-corrected configuration the penalty's minimum is
  `(tr(WᵀW)^{1/2})²`, whose value tracks the *magnitudes* of the learned task
  parameters, so `Ω` regains the accuracy/scale sensitivity the paper's
  covariance has, at the cost of being dominated by the largest-norm head —
  the pathology the folder `README.md` documents for iterations 1–4
  (`diag(Ω)` up to `0.837` on `er`).

## 5. Classified findings

| ID | Class | Finding |
| --- | --- | --- |
| M1 | **MATCH** | `Ω = (WᵀW)^{1/2}/tr((WᵀW)^{1/2})` bit-exact, `Ω ⪰ 0`, `tr(Ω) = 1`, attained penalty equals the published minimum |
| M2 | **MATCH** | Tensor orientation (row-stored `W` vs the paper's column convention) is handled correctly and documented |
| M3 | **MATCH** | `Ω₀ = I_m/m`; `Ω` never touched by the optimizer; `update_omega` under `no_grad` |
| M4 | **MATCH** | Regularizer = `λ·tr(W_paper Ω⁻¹ W_paperᵀ)`; gradient flows through live head weights; no gradient to the shared trunk |
| M5 | **MATCH** | Alternating structure: `W` step every batch with `Ω` fixed, `Ω` step closed-form on a schedule |
| M6 | **MATCH** | All-25-layer policy: `selected_transformer_layers: all` resolves to exactly `[0..24]` for `wavlm_large` (`parse_transformer_layers`), `smp` 0.5, no layer dropped or selected |
| M7 | **MATCH** | The historical control (`mtrl_poolingwinner_25L_config.yml`) and the matched baseline (`base_poolingwinner_25L_config.yml`) agree on every protocol factor: pooling, layers, 30 epochs, batch 2048, lr 2.5e-3, wd 5e-8, l1 1e-7, l2 1e-5, splits, checkpoint policy |
| B1 | **IMPLEMENTATION_BUG** | none found |
| D1 | **DOCUMENTED_ADAPTATION** (declared in the literature card and folder README) | `W` is a mean-pooled head summary `[mean(weight,0), mean(bias)]`, not the task parameter vector; the shared trunk is outside `W`. Consequence, measured: the regularizer's gradient is *identical across a head's class rows* (spread `0.0`) and scales as `1/C_t` per element, so its per-parameter reach on SI (1251 classes) is ~100× weaker than on KS (12). The adapter is a project-level necessity, not a defect — but it is the reason the method is an *adaptation of* MTRL rather than MTRL |
| D2 | **THEORETICAL_MISMATCH**, documented only as an implementation *assumption*, never as a deviation from the published method | `normalize_w: true` (**every** evidence-carrying historical run). The fact is on the record — `task_relations/MTRL_DIAGNOSTIC_SYNTHESIS.md` §"What classical MTRL assumes" item 4 states "With `normalize_w=true`, these summaries are unit-normalized", and item 6 says the closed form "is exact conditional on the current summary matrix" (correct: it is exact for `W~`). What is *not* recorded anywhere is the consequence: the summary matrix is a rescaled copy of the trained parameters, so `Ω` is not the minimizer of the published subproblem (50.11 % gap, §4.1), the penalty loses degree-2 homogeneity (a scale ratio of `1.0031`, `ε`-limited and tending to exactly 1 as `ε → 0`, against the published `9.0`), and the normalized `Ω` cannot represent the tasks' parameter *magnitude* (a 40× per-task rescale moves it by `2.8e-3` versus `0.97`; `diag(Ω) ≈ 1/m` at the attractor, but not in general — the 25L Phase-A run reports `[0.3027, 0.3076, 0.3897]`). No record states that the executed objective is therefore not Eq. (8) — `VARIANT_BENCHMARK_PROTOCOL.md` §2, `literature/zhang-yeung-2014-mtrl-asymmetric.md` and `DECISIONS.md` all describe `01-mtrl` as "classical symmetric MTRL" |
| D3 | **UNDOCUMENTED_ADAPTATION — not equivalent to the paper's term** | The paper's `(λ₁/2)tr(W Wᵀ)` term on `W` has no counterpart, and it is *not* interchangeable with what is there. `_process_batch` adds `l1_lambda·Σ_θ\|θ\| + l2_lambda·Σ_θ θ²` over `self.model.parameters()` — **all 4,086,067** parameters (62 % classifier heads, 38 % shared trunk and pooling), with no `1/2` and no configurable `λ₁` (`build_model`/`build_trainer` never pass one). Measured: the repository's term covers **≈681×** more elements than `W` (6,003) and includes parameters the paper's `W` explicitly excludes. At initialization the raw sum `Σ_θ θ²` is `1261.35`, realized as `l2_lambda·Σ_θ θ² ≈ 0.0126`; the paper-shaped comparator `(1/2)‖W‖²_F = 0.0551` is at unit coefficient in `λ₁`, which the repository does not have. The non-equivalence is structural, not a scaling of the same term. `mtrl_trainer.py`'s docstring presents the generic L2 as the paper's λ₁ term; on the code and the mathematics it is not. Arm-matched with the baseline, so it does not bias the MTRL-vs-baseline comparison — but it does mean no arm in this repository implements the published objective, which is why `DG-0007` is *not* labelled fully faithful |
| D4 | **DOCUMENTED_ADAPTATION** | `mtrl.warmup_epochs: 3` (of 30; `1` of 5 in the LOSO config) disables the regularizer and freezes `Ω` for the first epochs. No counterpart in the paper. `Ω` is still `I/m` when the term first switches on, so the paper's initialization is respected |
| D5 | **NUMERICAL_IMPLEMENTATION_DETAIL** | `mtrl_lambda` multiplies the term where the paper has `λ₂/2` ⇒ `λ₂ = 2·mtrl_lambda`. Immaterial: λ is a selected coefficient and the closed form (P9) is λ-free |
| D6 | **NUMERICAL_IMPLEMENTATION_DETAIL** | `Ω_ε = Ω + εI`, `ε = 1e-4`, used for the inverse only; the logged `Ω` is exact. Perturbation measured at **≈3e-4** (3.011e-4 on the max-entry relative norm) on `Ω⁻¹` at the model's own conditioning. It is also the floor that makes the normalized scale ratio 1.0031 instead of exactly 1 (see §4.1). Documented |
| D7 | **DOCUMENTED_ADAPTATION** | `Ω` cadence is once per epoch (`omega_update_frequency: 1`) rather than "repeated until convergence" of the dual `W`-subproblem. Necessary for a deep model; documented in the trainer docstring |
| C1 | **EXPERIMENTAL_CONFOUND** (minor) | `_process_batch` adds the MTRL term on the **validation** path too (the gate is only `current_epoch >= mtrl_warmup_epochs`, not `train_mode`). The historical arm's reported/validation loss therefore differs from the baseline's by `R(W) ∈ [mλ, m²λ]`-derived magnitudes, and `ReduceLROnPlateau` steps on that loss. `checkpoint` selection uses validation *accuracy*, so the protocol endpoint is unaffected. Not corrected here: it is inherited from the frozen base trainer's L1/L2 handling and is protocol-shared in kind, and changing it would be a second variable |
| U1 | **INSUFFICIENT_EVIDENCE** | Whether iterations 1–4 of the 16-layer campaign really ran `normalize_w: false` (the folder README says so, and the model default is `False`) cannot be confirmed from git: the implementation commit `33603c3` already records `normalize_w: true` in `mtrl_config.yml`, so the campaign's intermediate config states are not in history. Irrelevant to the evidence-carrying runs, which all use the committed `true` configs |
| U2 | **INSUFFICIENT_EVIDENCE** | The paper's exact `λ₁`, `λ₂` for a deep heterogeneous model are undefined by the paper (its §4 values are for linear/kernel CV on toy and benchmark data). No paper value was expected at this scale and none is claimed |
| U3 | **INSUFFICIENT_EVIDENCE** | The complete executable repository state of the historical five-seed and LOSO MTRL runs. The run metadata records `bfb1ad44`, but that commit contains neither the seeding machinery (`improvements/seed_utils.py`, `--seed`, `cfg["seed"]`: all arriving in `03c28fe`, 2026-09-02) nor `mtrl_er_kfold.py` / `mtrl_kfold_config.yml` (arriving in `03c28fe` / `c604cc9`, 2026-09-02), while `bfb1ad44` is dated 2026-09-01 and is their ancestor. `resolve_git_commit()` reads `git rev-parse HEAD` at run time, so a dirty checkout at that HEAD, or a wrong recorded commit, both fit; neither is established. The runs are not inferred not to have happened and no replacement commit is invented. **This does not weaken D2**: `normalize_w` was introduced in `33603c3` (an ancestor of `bfb1ad44`), the model file implementing it is byte-identical from there to HEAD, and every committed MTRL config — including the one at `bfb1ad44` — sets it `true`. Whatever tree executed, it contained the row normalization |

## 6. Effect on the interpretation of the historical MTRL results

**Unchanged — and still valid as stated:**

* **F4 ("classical MTRL does not currently beat wavCSE meaningfully")** and the
  multi-seed 16L/25L tables and the LOSO table remain valid *as observations
  about the arm that ran*: within the shared protocol the only difference
  between the two arms is the presence of the relation term, so "this
  regularizer, as configured, did not produce a reproducible advantage" stands.
* The retraction history (the 16L "win" was seed noise; the LOSO verdict is
  null) is untouched.
* No historical number is invalidated as a *measurement*; nothing here is a
  leak, a split error or a metric error.

**Changed — the attribution and the mechanism reading:**

1. **The historical arm cannot be cited as evidence about Zhang & Yeung MTRL.**
   It is evidence about *MTRL with a row-normalized task-parameter matrix*, a
   project-specific modification of the regularizer. `VARIANT_BENCHMARK_PROTOCOL.md`
   §2's phrase "Classical symmetric MTRL — the in-category control" is, on this
   evidence, an inaccurate label for `mtrl_poolingwinner_25L_config.yml`. This
   is a protocol-ownership question (DEC-0013 is the human's), so this audit
   does **not** edit the protocol.
2. **`Ω`-based conclusions are conditioned on the modification.** F5/F6 and the
   "saturated in 5/5 seeds" reading of Ω are, at least in part, the
   *structural* behaviour of the normalized penalty (its minimizer sits at
   `Ω = s sᵀ/m`). The stability contrast the project reports — KS↔SI stable
   near `+1/3` while the ER edges flip sign — is exactly the pattern the
   normalized objective predicts when two of three directions align and one is
   noisy, so it can no longer be read as an unmediated property of the tasks.
   This does **not** create a positive result; it weakens an interpretive one.
3. **The programme's "no mechanism has displaced the baseline" state is
   unaffected** — the normalization-corrected configuration has never been run under the
   current `smp` 25-layer protocol (the `normalize_w: false` historical runs
   used the 16-layer `weighted` config, single seed, no LOSO), so the null
   result cannot be transferred to it.

## 7. Is a successor experiment required?

Yes. There is a demonstrable, quantified difference between the published
objective and the objective that ran, and the *corrected* configuration has
never been evaluated under the binding protocol. The correction is
unambiguous — the paper contains no normalization of `W`, its `W` is the task
parameter matrix, and the adapter through which this project supplies that
matrix is already declared — so it is a configuration change, not a new
scientific decision:

> **`model.normalize_w: true → false`**, everything else value-identical.

Historical implementation and corrected implementation therefore share one
code path and differ in exactly one boolean, which isolates the consequence of
the mismatch with no implementation risk.

**What the correction does and does not remove.** It removes D2. It does not
make the arm a faithful implementation of Eq. (8): D1 (the mean-head summary
adapter), D3 (the absent `(λ₁/2)tr(WWᵀ)` term — shown not equivalent, above),
D4, D6 and D7 all remain, in both arms. The successor is therefore named
**normalization-corrected MTRL**, not "faithful MTRL", and its claims are
limited to the normalization. Calling it fully faithful would silently resolve
D3 by naming, which the evidence does not support.

**Successor study: `DG-0007`** —
`research/studies/DG-0007/` (`PLAN.md`, `NOTE.md`), config
`improvements/taskrelation/01-mtrl/mtrl_norm_corrected_25L_config.yml`, registered in
`STUDIES.jsonl`.

## 8. What is *not* claimed or changed

* History is untouched: no historical metric, manifest, run identity, config,
  checkpoint, study folder, `FINDINGS.md`/`FAILURES.md`/`DECISIONS.md`/
  `STATE.md` entry or `STUDIES.jsonl` record was edited or deleted. The audit
  is additive.
* No paid compute, worker, job, authorization, S3 or Network-Volume operation
  was performed. All work is CPU-only, in this worktree.
* This audit does **not** alter the protocol, does not re-label the historical
  control, and does not promote or demote a finding. Adopting the corrected
  configuration *as* the in-category control is a `DEC`-level human decision,
  for which `DG-0007` supplies the evidence.


## 9. Correction log (2026-09-29) — independent review `5f72acd`

This audit was independently reviewed on `review/mtrl-theory-audit` at commit
`5f72acd` ("Review the MTRL theory audit independently, and pin what it
measured"), which returned `AUDIT_REQUIRES_CORRECTION`. The review's own
artifacts are
`audits/2026-09-29-mtrl-theory-audit-independent-review.md` and
`tests/test_mtrl_theory_faithfulness_review.py` on that branch. Corrections
below are **additive commits**: `f05539c` and `2c0391e` are not amended,
rebased or rewritten.

### 9.1 What the review confirmed (unchanged)

The central finding (D2), the Section 4.1 magnitudes, the primary-source
equations and closed form, the orientation, the gradient path, `M1`–`M7`,
`D1`–`D7`, `C1`, `U1`–`U2`, the config isolation, the all-25-layer policy, the
LOSO result, and `DG-0007`'s one-independent-variable structure. The review
re-derived the measurements from scratch in float64 without importing this
audit's helpers and reports the same values, adding a seed sweep
(`+48.26 % … +52.07 %` gap; ratio `1.0031–1.0032`) and a random-search check of
the constrained minimiser (0 violations in 20,000 unit-trace PSD candidates,
`max‖Ω_code − Ω*‖ = 5e-8`).

### 9.2 R1 — historical execution provenance (corrected)

**Was:** §4.0 asserted that the five-seed and LOSO comparisons "ran at
`bfb1ad44`", and the report, the `01-mtrl` README, `DG-0007`'s plan and the
`f05539c` message generalised to "the audited code is the code every historical
run executed".

**Now:** the provenance of the *executed tree* is classified `U3 —
INSUFFICIENT_EVIDENCE`, and the report separates what is proven (A: the two
MTRL files are byte-identical from their introduction commit `33603c3`, an
ancestor of `bfb1ad44`, to HEAD, so the row normalization was present in every
tree across that history) from what is not (B: the complete executable state).
The contradiction is recorded rather than smoothed: `bfb1ad44` contains neither
the seeding machinery nor the LOSO entry point and config those runs used. A
dirty checkout and a wrong recorded commit are listed as possibilities only;
the runs are not inferred not to have happened, and no replacement commit is
invented. **D2 is unaffected**, for the reason recorded in `U3`.

### 9.3 R2 — regression coverage (corrected)

**Defect.** The suite's `assert_published_invariants` began with
`assertFalse(model.normalize_w)`, so for the historical configuration it failed
on a *configuration* assertion and the mathematical assertions below it were
unreachable. The claim in the first edition of this report — "the published
invariants pass for the faithful configuration and fail for the historical one"
(its wording at the time; "faithful" is relabelled in §9.6 below) — was
therefore true only in the trivial sense that a YAML key differed. The measured
magnitudes were exercised by the report's probes and by the review, but not by
the committed suite.

**Fix.** `tests/test_mtrl_theory_faithfulness.py` was restructured so that no
mathematical assertion sits behind a configuration assertion:
`assert_published_W_invariants` returns measured values and asserts nothing
about configuration; `B_HistoricalNormalizedMathematics` builds the model and
asserts the historical arm's Ω gap (`> 0.40`), its non-quadratic scale ratio,
the `λ·[m, m²]` confinement with both endpoints, the `ε`-limiting of the ratio,
and the loss of scale information; `C_NormalizationCorrectedMathematics`
asserts the same criterion passes (`gap < 1e-3`, ratio `9.0`); and
`D_Discriminator` applies **one** criterion to both policies and asserts the
verdict flips, plus a guard that the criterion is not trivially satisfiable.
Configuration facts moved to `F_ConfigurationRecord`, asserted separately.
23 tests, all passing.

**Reviewer artifacts.** The review's test module is not copied in; its coverage
is reconciled into this suite and a second near-duplicate module would obscure
which test proves which invariant. Concretely, what the committed suite now
asserts of the review's checks is:

* the `λ[m, m²]` confinement **with both endpoints exactly** (`A = I → m²`,
  `A = s sᵀ → m`, plus 200 random unit-row cases inside the interval) —
  `B_HistoricalNormalizedMathematics.test_normalized_penalty_is_confined_...`;
* the separation between the two policies by a **single criterion whose verdict
  flips with only the flag** — `D_Discriminator`, which includes a guard that
  the criterion is not trivially satisfiable (it checks one alternative
  candidate, `Ω = I/3`, not an exhaustive search);
* what is **not** reproduced here is the review's *search-based* argument for the
  constrained minimiser (its 20,000-candidate random-PSD sweep) and its
  exhaustive-candidate guard. Those remain the review's independent evidence,
  cited as such; the committed suite instead asserts the stronger analytic
  identity — that the code's `Ω` attains the published minimum `(tr A^(1/2))²`
  to `1e-4` relative — in `A_PublishedOmegaSolution` and
  `C_NormalizationCorrectedMathematics`.

The review's float64 seed sweep is quoted in §4.1 and §9.1 as independent
reproduction. `DG-0007`'s pre-registered runtime gate, which the review noted
had no implementation, now has one (§9.4).

### 9.4 DG-0007 planning gaps (closed)

1. **Runtime-faithfulness checker.** `studies/DG-0007/check_runtime_faithfulness.py`
   applies the pre-registered gate unchanged to a finished run's checkpoint,
   logged `Ω` and run-identity record (exit 0/1/2). One ambiguity in the gate's
   third clause is **reported rather than resolved silently**: read with `Ω`
   fixed, "the penalty is quadratic in `W`'s scale" is `c²` for any
   positive-definite `Ω` and cannot discriminate; the checker applies the
   read-with-`Ω`-re-derived interpretation, which is what this audit and its
   review measured, and the PLAN records that choice.
   *(Superseded in part by §10.2–§10.3: the exit contract and run-identity
   binding were extended after rereview `5610382`.)*
2. **Budget separation.** `PLAN.md` now separates the requested single-split
   budget (3-arm screen + 15-run confirmation, ≈6.5 GPU-hours) from the LOSO
   escalation, which is reached only on a pre-declared ER trigger. The LOSO
   cost is recorded as **UNKNOWN**: no wall-clock or GPU-hour measurement for
   the k-fold protocol exists anywhere on the record, so no number is invented.
   The structure (10 folds × 3 arms × the 5-epoch fold budget) is stated, with
   a measured estimate required before authorization.

### 9.5 R3 — adjacent MSSL transcription error (repaired, already resolved canonically)

`TR-0007`'s gate on this branch rendered the published MSSL Ω step as
`−d log|Ω| + λ₂‖Ω‖₁`, `S = (1/d)WᵀW`, mixing Eq. (4b)'s barrier with Eq. (8)'s
data term, and called the draft and the paper "different estimators at the same
`λ₂`". Re-read from the primary source (JMLR 17(33),
`jmlr.org/papers/volume17/15-215/15-215.pdf`):

* Eq. (4b) `λ₀ tr(WΩWᵀ) − d log|Ω| + λ₂‖Ω‖₁`
* Eq. (8) `min_{Ω≻0} λ₀ tr(SΩ) − log|Ω| + (λ₂/d)‖Ω‖₁`, `S = (1/d)WᵀW`, followed by
  *"As λ₂ is a user defined parameter, the factor 1/d can be incorporated into
  λ₂."*
* `λ₀ = 1` in all experiments; `λ₀, λ₁, λ₂` selected by cross-validation.

Scope of this branch's repair, kept deliberately minimal because the canonical
branch already owns the resolution:

* `literature/goncalves-2016-mssl.md` — replaced with the canonical branch's
  corrected version (byte-identical to
  `feature/mssl-task-relation-study`, so integration is a no-op there).
* `STATE.md` — the live declarative sentence corrected in place, with a note
  that the gate is resolved canonically as `DEC-0015` and that integration must
  prefer the canonical branch's `STATE.md`.
* `studies/TR-0007/NOTE.md` — the historical gate text **left intact** (which
  is what the canonical branch itself does) with an append-only correction
  block pointing at `DEC-0015`.
* `STUDIES.jsonl` — **not touched**: the canonical branch deliberately preserves
  the historical `escalated_to_human` text, so changing it here would diverge
  from the newer canonical record.

This is an adjacent stale-branch repair, **not** a result of the MTRL audit, and
it creates no competing MSSL decision. Integration note: `STATE.md`,
`studies/TR-0007/NOTE.md` and `STUDIES.jsonl` will conflict with the canonical
branch in these regions; the canonical branch's newer text wins in every case.

### 9.6 D3 and the arm label (determined from code and mathematics)

The review required the label question to be settled by evidence rather than by
naming. It is: the repository's `λ₁`-equivalent term is **not** equivalent to
the paper's, on three independent grounds (parameter set: all 4,086,067
parameters including the 38 % shared trunk the paper's `W` excludes, versus
`W`'s 6,003 elements, ≈681×; no `1/2` factor; no configurable `λ₁` at all).
Quantitatively, and keeping the two quantities distinct: at initialization the
**raw sum** `Σ_θ θ²` over all parameters is `1261.35`, which the trainer scales
by `l2_lambda = 1e-5` into a **realized regularization contribution** of
≈`0.0126`; the paper-shaped comparator on `W` alone is
`(1/2)‖W‖²_F = 0.0551` **at unit coefficient** in `λ₁`, not a realized
contribution, since the repository has no `λ₁` to apply. The conclusion rests on
the structural differences above, not on this comparison. `DG-0007` is therefore labelled
**normalization-corrected MTRL** — `method: mtrl_norm_corrected`, config
`mtrl_norm_corrected_25L_config.yml` — and explicitly is **not** labelled fully
faithful Zhang & Yeung MTRL. This is a factual determination from the code, not
a scientific decision, so it does not require the human. The experiment's
independent variable is unchanged: `model.normalize_w` only.

### 9.7 Precision nits (corrected)

* `Ω⁻¹` perturbation `"< 3e-4"` → `"≈3e-4 (3.011e-4)"`, with the note that the
  same floor is what makes the normalized scale ratio 1.0031 rather than 1.
* `diag(Ω) ≈ 1/m` is no longer stated as a general invariant of the normalized
  arm; the stronger and correct statement (loss of task-*scale* information,
  with the 25L Phase-A counterexample `[0.3027, 0.3076, 0.3897]`) replaces it,
  in §4.1, §4.2 and the README.
* The `1.0031` scale ratio is described as `ω_epsilon`-limited and tending to
  exactly 1 as `ε → 0` (with the three measured points).
* Config isolation is described as **value**-identical rather than
  byte-identical for the two MTRL configs (their comment headers differ; the
  two *code* files are byte-identical, which is a separate and correct use).
* The unverifiable UAI-version parenthetical for `λ₁`/`λ₂` is dropped; only the
  TKDD coefficient text (`λ₁ = 2ε²/ϵ²`, `λ₂ = 2ε²`) is quoted.

### 9.8 Deliberately unchanged

The independent variable, the study's design, the protocol, the historical
record, the historical control's label (`VARIANT_BENCHMARK_PROTOCOL.md` §2 —
still a `DEC`-level human question), and the four-classification of D2's
*evidence* (the fact of normalization is on the record; its consequence never
was). No new scientific change was added to `DG-0007`.

## 10. Remediation log (2026-09-30) — targeted rereview `5610382`

The rereview at `review/mtrl-theory-audit` commit `5610382` ("Rereview the
corrected MTRL theory audit: R1/R2/R3 resolved, study registration still open")
returned `AUDIT_REREVIEW_REQUIRES_CORRECTION` while confirming the science:
R1, R2 and R3 resolved; D2 independently reproduced; D3 established; the arm's
terminology correctly narrowed; `ONE_VARIABLE_CONFIRMED`;
`25_LAYER_POLICY_CONFIRMED`; the clause-3 criterion not silently changed; the
budget/LOSO separation confirmed; historical immutability confirmed. The
blockers were research-run identity wiring (H1) and mechanical
robustness/documentation defects (M1–M3, four L, two N). Remediation is additive
commits; `f05539c`, `2c0391e`, `ae23842` and `ef1bf7f` are not amended.

**No scientific change was made.** The frozen set — the study, the
`normalization-corrected MTRL` interpretation, `model.normalize_w`
`true → false`, screen seed 42 and confirmation seeds 0–4, the three arms, all
25 layers, `smp` 0.5, and the conditional separately-budgeted LOSO escalation —
is untouched. The remediation shows up in this report's appendices and in
`DG-0007`'s registration and plan, not in the finding.

### 10.1 H1 — DG-0007 had no study-identity configs at runtime

**Root cause.** Nothing in the audit's registration gave the three arms a
`research:` block. The runner's own helpers are what bind a run's identity
(`run_improvements.main` → `mlflow_utils.build_research_run_name` →
`resolve_run_note` → `set_standard_tags`, plus
`run_identity.emit_run_identity`), and all of them read `cfg["research"]`.
Without it `build_research_run_name` returns `None` and the runner falls back to
the legacy `{category}_{model}_{task_type}_{timestamp}` name, `resolve_run_note`
returns `None`, and `study_id`, `stage`, `parent_study` and `representation` are
published as `None` — contradicting `/AGENTS.md` invariant 12 and
`VARIANT_BENCHMARK_PROTOCOL.md` §7. The compute path's staged-evidence manifest
does carry `study_id`/`arm`/`seed`, but it identifies the *staged evidence*, not
the MLflow run, and DG-0007 requests no compute.

**Fix.** Six metadata-carrying execution configs under
`studies/DG-0007/configs/` — `norm_corrected_mtrl.yml`,
`historical_mtrl.yml`, `baseline.yml` and their `confirm_*` counterparts —
following DG-0002's and DG-0005's convention exactly (full standalone configs
with the canonical `research:` field set: `study_id`, `stage`, `family`,
`method`, `hypothesis_slug`, `task_set`, `representation`, `parent_study`,
`baseline_study`, `agent_generated`, `status`, `run_note_file`). Each is its
source config (the architecture folder's for the two MTRL arms,
`improvements/base/configs/` for the baseline) plus that block and per-arm
output roots, asserted in `tests/test_dg0007_run_identity.py`. All three arms
run through the one study runner, as DG-0002 does: `--model mtrl` for both MTRL
arms, `--model original` for the matched baseline.

One gap that fix exposed: `model` is `"mtrl"` for **both** MTRL arms, so a
record with only `study_id`/`stage` cannot say *which* arm produced an artifact.
`run_identity.research_identity` therefore carries `method`, `representation` and
`git_commit` alongside `study_id`/`stage` — all canonical research-block names,
already published as MLflow tags — and both entry points
(`run_improvements.py`, `base/run_base.py`) emit them.

### 10.2 M1 — the checker's exit contract

The first version caught only `OSError`/`ValueError`/`KeyError`, so a
wrong-shaped `Ω`, inconsistent classifier shapes or a truncated checkpoint
raised out of `check()` and surfaced as exit `1` — indistinguishable from a
scientific gate failure, and the most likely real-world bad input (AGENTS.md
documents mid-write checkpoint corruption when the disk fills). `check()` now
validates every input before evaluating any clause and raises `BadEvidence`;
`main()` maps that, and any other unexpected failure, to exit `2` with a bounded
diagnostic (message truncated at 240 characters, structured detail, never a
tensor dump). Valid evidence with historical mathematics still exits `1`.

### 10.3 M2 — the checker accepted artifacts with no run identity

A two-task checkpoint with any consistent `Ω` passed. The checker now requires
the run's `ARC_RUN_IDENTITY` file, read through the compute worker's own
`improvements/run_identity.py::read_identity_file`, and asserts the record names
the requested `study_id`, `stage`, `method` (the arm), `model`, `task_type`,
`representation` and `seed`, with an optional `--expect-commit` for the commit
the study record pins. It additionally validates the checkpoint structurally
against the identity's `task_type` using the repository's own
`utils.constant_mapping` (task count and per-task class counts), so a
well-formed checkpoint from another task set is refused rather than scored.
Mismatched evidence returns `2` because it is not evidence for the requested run.

### 10.4 M3 and the LOW items

* **M3** — `DG-0007` now appears in `STATE.md`'s *Current Pending Work* and in
  the `DG-xxxx` list: registered, pre-registered, `BLOCKED`, screen not
  executed, no compute authorized, with its audit lineage, its label, its
  independent variable, the arm/layer/pooling protocol and the authorization
  question. The newer MSSL state is preserved.
* **L1** — the registry record's `competing_explanation` and
  `independent_variable` still said "faithful arm"; both now say
  normalization-corrected. The same overclaim in `hypothesis` ("cannot be
  transferred to Zhang & Yeung's published method") was narrowed to the
  configuration, because §9.6's D3 finding means the corrected arm is still not
  the published method. No experimental meaning changed.
* **L2** — the plan's checker example named `train_ks_si_er_epoch.pth`, which no
  run writes. It now states the rule (`create_file_path` → `train_<task_type>.pth`
  base, `_epoch{N}`, `_best`, `_opt`; `saved_checkpoint_count: 1` keeps only the
  latest) and uses the real protocol artifact `train_ks_si_er_epoch30.pth`.
  Runtime naming was not changed.
* **L3** — §9.3 claimed the review's search-based minimiser argument and
  exhaustive-candidate guard were "now covered here". They are not; §9.3 now
  lists precisely what the committed suite asserts (the `λ[m, m²]` endpoints and
  the single-criterion flip guard), states that the search-based argument remains
  the review's independent evidence cited as such, and points at the analytic
  attainment identity the suite asserts instead.
* **L4** — D3's `1261.35` is the raw `Σ_θ θ²` over all parameters; the realized
  contribution at `l2_lambda = 1e-5` is ≈`0.0126`, and the
  `(1/2)‖W‖²_F = 0.0551` comparator is at unit coefficient in a `λ₁` the
  repository does not have. Both the findings table and §9.6 now separate the
  three quantities. The non-equivalence conclusion is structural and unchanged.
* **N1** — recorded in both the checker docstring and the plan: clauses 2 and 3
  confirm the *declared policy*; **clause 1 alone discriminates on the
  artifact**, which is why it is checked first.
* **N2** — recorded as a residual limitation: the artifact set carries one `Ω`
  stream, so an in-process divergence between the `Ω` step and the penalty step
  is not detectable from evidence. The plan's diagnostics section already
  requires the per-epoch instrumentation that would close it; no protocol change
  and no new instrumentation was added for it here.

### 10.5 All-25-layer stored-artifact question

`improvements/compute/embedding_layout.py` validates each materialized input by
artifact name, `sha256`, `size_bytes` and file count — the stored layer count is
**not** part of any metadata contract the compute plan carries, and the store is
upstream-frozen and absent from this environment. No data was invented and
nothing was downloaded. What is asserted deterministically is the load-bearing
behaviour: the loader indexes the store with the layer array
(`embedding[transformer_layer_array, :]`), so a 25-row store is used in full and
a short store raises rather than silently training on a subset. The store's own
row count is recorded in `PLAN.md` as a **pre-run materialization/runtime
validation** that must be confirmed before paid evidence is accepted.

## 11. Final remediation log (2026-09-30) — final rereview `3b25152`

The final targeted rereview at `review/mtrl-theory-audit` commit `3b25152`
returned one HIGH and one MEDIUM. Everything else was confirmed closed: H1, M1,
M2 (for every listed mismatch), M3, `ONE_VARIABLE_STILL_CONFIRMED`, no baseline
regression, the 25-layer enforcement, L1–L4, N1/N2 recorded, science unchanged,
historical immutability. Additive commits only; `f05539c`, `2c0391e`, `ae23842`,
`ef1bf7f`, `be2d04a` and `1deac79` are not amended. **No scientific change.**

### 11.1 H-NEW-1 — `seed = 0` was refused as unusable evidence

**Root cause.** `load_identity` tested identity fields for *truthiness*:

```python
missing = [field for field in IDENTITY_FIELDS if not record.get(field)]
```

`seed` is an integer and `0` is falsey, so a legitimate seed-0 record was
reported as *incomplete* and the run was refused with exit `2` — described by the
plan as "deleted from the comparison and reported as an execution failure". The
confirmation stage runs seeds `0,1,2,3,4`, so three of the fifteen pre-registered
confirmation runs (seed 0 for all three arms) would have been discarded after
paid compute. The suite never exercised a zero-valued field
(`EXPECTED_IDENTITY["seed"] = 42`, the mismatch test used `7`, the
incompleteness test set fields to `None`), which is why it escaped.

**Fix — exact validation semantics.** Replacement is schema-driven, not
truthiness-driven:

```python
def identity_field_problem(field, value):
    if value is None:
        return "missing"
    if field in _INTEGER_IDENTITY_FIELDS:          # {"seed"}
        if isinstance(value, bool) or not isinstance(value, int):
            return "not an integer seed"
        return None
    if not isinstance(value, str) or not value.strip():
        return "empty"
    return None
```

So: an absent key and an explicit `None` are **missing**; `0` is a **valid
integer seed**; `""`/whitespace and a non-string are **invalid** for the
text fields; a numeric string (`"0"`), a boolean (`True`, refused because
`True == 1` would silently satisfy a seed comparison) and a missing /
`None` / empty `git_commit` are refused. `git_commit` keeps its prior
behaviour: `resolve_git_commit` returns `None` outside a git tree and `None` was
already refused, so nothing is loosened. No special case for DG-0007 or for the
value zero — the rule is the schema's.

**Reproduced against the pre-fix tree.** With `1deac79`'s checker extracted to
`/tmp`, seed-0 confirmation evidence (its own record, its own artifact):

| | pre-fix `1deac79` | post-fix |
| --- | --- | --- |
| seed 0, `--seed 0` | **exit 2** | **exit 0** |

### 11.2 M-REM-1 — identity from run A could be paired with run B's artifact

**Investigation.** The identity record already contained the authoritative
binding and the checker was ignoring it: `identity_record` records
`results_dir` and `checkpoints_dir` as absolute paths, and the trainer writes
each run into its own timestamped directory
(`downstream/trainer/trainer_model.py`: `results_dir = <results_root>/<run_id>`,
`ckpt_dir = <checkpoints_root>/<run_id>`), so two runs differ in directory and
nothing else. No new provenance system was needed and none was added. (The
compute path binds its *staged* copies by `sha256`/`size_bytes` in
`MANIFEST.json`, and records the identity's `run_id`/`checkpoint_run_id`; DG-0007
requests no compute, so its acceptance path is the per-run invocation where the
record's own paths are directly authoritative.)

**Status: `CLOSED`.** `require_within` now requires the checkpoint to live in
the record's `checkpoints_dir` and `omega_history.json` in its `results_dir`,
comparing `os.path.realpath` so a symlink cannot smuggle a file across. The
identity record's `results_dir`/`checkpoints_dir` are also validated as
non-empty strings. Mismatches return exit `2` ("this checkpoint is not from the
run the identity describes"), never a scientific exit `1`.

**The reviewer's exact attack**, with `1deac79`'s checker for comparison — two
runs of the same arm, task shape, representation *and seed*, differing only in
their per-run directories:

| | pre-fix `1deac79` | post-fix |
| --- | --- | --- |
| identity A + checkpoint A | exit 0 | exit 0 |
| identity A + checkpoint B | **exit 0** | **exit 2** |

**Residual, recorded not hidden.** Copying a foreign checkpoint *into* the run's
own directory is not detectable from the identity record; it requires write
access to the evidence directory, and the compute path covers it with staged
digests. Stated in the checker docstring and in the plan's evidence-pair
paragraph.

### 11.3 What the final remediation changed and what it did not

Changed: `check_runtime_faithfulness.py` (identity completeness semantics and
artifact binding), its test module (24 tests, up from 17: the seed table, the
schema-rejection table, the two-run substitution attack, the foreign
`omega_history` attack and the outside-the-directory case), and the plan's
gate/evidence-pair wording. Not changed: the study, its interpretation
(`normalization-corrected MTRL`, still not fully faithful Zhang & Yeung MTRL),
`model.normalize_w true → false`, screen seed 42, confirmation seeds `0,1,2,3,4`,
the three arms, all 25 layers, `smp` 0.5, the LOSO escalation, and every
pre-registered clause and threshold.
