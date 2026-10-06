# LAYERSWEEP-01 — Layer-subset sensitivity of the normalization-corrected MTRL arm

Status: PRE-REGISTRATION — declared and config-generated; no run, no MLflow run,
no paid compute yet.
Type: representation sweep (diagnostic). It is **not** a mechanism study: no new
architecture is introduced, and nothing here is a published-method claim.
Created: 2026-10-06.
Predecessor: `DG-0007` (normalization-corrected classical MTRL, the best-scoring
task-relation arm so far). Parent study: `DG-0007`. Reference control:
`DG-0007-baseline` / the all-25-layer selection.
Independent variable: `upstream.selected_transformer_layers`, one of 24
pre-registered subsets.
Machine-readable spec: `sweep.json`. Tooling: `improvements/sweep`.

## 1. Why this study exists

`DG-0007`'s arm is defined at `selected_transformer_layers: all` (25 WavLM-large
layers, mean frame pooling, `smp` 0.5 layer pooling). Which layers those are was
never varied inside the arm, so the arm's own result is conditioned on one
representation choice. `F2` already establishes that representation choice can
produce effects larger than architecture choice, so an arm whose headline number
rests on an unexamined 25-layer selection has an unmeasured confound inside it.

This study varies that selection with everything else held fixed, and asks
whether the *drop scheme* matters — not merely how many layers are kept.

## 2. Hypothesis and competing explanation

**H1 (registered).** At the fixed protocol below, the choice of layer subset
changes validation accuracy beyond seed noise: dropping layers that are
redundant or uninformative does not degrade the arm, and at least one
pre-registered subset exceeds the all-25-layer selection on **validation**.

**H2 (competing explanation, must be reported if it holds).** The effect is
driven by layer *count* (and the pooled magnitude that comes with it), not by
which layers are dropped. Under H2, two subsets with equal `k` perform within
seed noise of each other regardless of their group, and the ordering of the 24
subsets is explained by `k` alone.

**Third pre-declared outcome.** Every subset performs within noise of the
all-25 selection. The arm is then insensitive to this axis at this protocol
resolution, which is a usable negative result and bounds any later
representation claim.

H1 is falsified by a flat ordering across groups at equal `k`, or by no subset
separating from all-25 on validation.

## 3. The 24 pre-registered subsets

Six drop schemes × `k ∈ {2,4,6,8}`, where `k` is the number of layers dropped
(25 − `k` kept). Exact layer lists are in `sweep.json` (fields `id`, `group`,
`k`, `layers`) and were confirmed by the researcher before generation:

| group | intent |
| --- | --- |
| `top layer dropping` | remove the highest-index layers (the task-head-adjacent end) |
| `even alternate dropping` | remove every other layer from the top, starting at layer 24 |
| `odd alternate dropping` | remove every other layer from the top, starting at layer 23 |
| `symmetric dropping` | remove a contiguous block centred in the stack |
| `bottom layer dropping` | remove the lowest-index layers (the acoustic end) |
| `contribution based dropping` | remove the layers judged least contributing |

The 25-layer reference (`all`) is not a member of `combos`; it is the control and
is taken from `DG-0007`'s recorded all-25 runs at the same commit lineage, or
re-run at the same commit if the sweep's commit moves the protocol.

## 4. Protocol (nothing below varies across the 24 runs)

Frame pooling `mean`; layer pooling `smp` λ = 0.5; 30 epochs; batch 2048; AdamW
lr 2.5e-3, wd 5e-8; l1 1e-7, l2 1e-5; `mtrl_lambda` 0.01, `normalize_w: false`,
`omega_epsilon` 1e-4, warmup 3 epochs; full data (100%); identical splits; test
metrics never used for selection. Only `upstream.selected_transformer_layers`
differs, which `test_layer_sweep_config_gen.py` asserts by diffing two rendered
configs and refusing any other change.

`tests`/`val` selection checkpoint: the protocol's primary checkpoint, as
recorded by the template.

## 5. Stages and seeds

| stage | combos | seeds | purpose |
| --- | --- | --- | --- |
| `screen` | all 24 | 42 | one run each, ranked on validation |
| `confirm` | top 2 by validation (`top-k2`, `sym-k2` as placeholders in the spec, to be set from the screen) | 0–4 | five matched seeds vs the all-25 reference |

The `confirm` selection in `sweep.json` **must be updated from the screen's
validation ranking before the confirm stage starts**; the two ids currently
listed are placeholders and the screen is the thing that chooses them.

## 6. Decision rule (pre-registered)

* Screens are screening evidence only: one seed and one split cannot establish
  an improvement (`F1`). Validation ranks; test is reported.
* Promote a subset only if its five-seed mean validation accuracy exceeds the
  matched all-25 reference by more than the paired seed spread, with no per-task
  test regression beyond 0.20 pp.
* Compare spread *within* a `k` across groups against spread *within* a group
  across `k`. If the former is comparable to the latter, H2 is supported and H1
  is not.
* No ER claim without speaker-independent LOSO: the ordinary split is
  speaker-leaky (`F3`), so the `er` column here is screening context only.

## 7. Execution model and its recorded deviation

`improvements/sweep` is a direct runner, not `improvements.compute`: no
envelope, no deterministic job spec, no evidence validator, no `reap` timer.
The trade is recorded, not hidden — spend is bounded only by
`policy.max_wall_seconds`, and the supervisor enforces that plus a DRAIN flag.

**Deviation from `AGENTS.md` invariant 16** (at most two simultaneous GPU
training jobs): the researcher replaced the fixed cap with an automatic,
resource-derived cap for this sweep (2026-10-06, this session). It is recorded
in `sweep.json` → `policy.concurrency_basis` rather than left implicit, so a
reader can see the invariant was set aside deliberately.

Concurrency is derived per admission pass from free VRAM, available RAM, cores
and disk, and each run's checkpoint/metric files are hashed into the sweep
ledger at completion (the validator the backend would otherwise have provided).

## 8. Grouping and where the results live

One folder (this study), one MLflow experiment
(`taskrelation-mtrl-layersweep`), one tag vocabulary:

| field | where |
| --- | --- |
| `research.group`, `research.k`, `research.layers` | config → MLflow **params** |
| `sweep_group`, `sweep_k`, `sweep_combo`, `sweep_layers` | MLflow **tags**, applied by the supervisor at completion |
| group / k / combo / kept / dropped layer list | the **run note** (`mlflow.note.content`), i.e. the DagsHub details field |
| run name | `LAYERSWEEP-01__<stage>__<combo>__ks_si_er__smp<L>L__s<seed>` |

## 9. Known risks

1. **Layer count vs layer identity.** `k` also changes the pooled magnitude
   under `smp`; H2 exists precisely because "fewer layers" is not the same claim
   as "these layers are redundant".
2. **Single-seed screens.** 24 screening runs are not 24 results. Only the
   confirm stage can support a comparison.
3. **Concurrency and reproducibility.** Runs sharing one GPU change reduction
   order; a subset that wins on validation must be re-confirmed in a matched
   configuration before any claim, and the executed commit must be recorded.
4. **No upstream re-extraction is involved** — layer selection is applied at
   load time in the (frozen) downstream loader, so these runs share one
   embedding set by construction.
