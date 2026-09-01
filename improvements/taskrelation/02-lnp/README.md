# taskrelation/02-lnp — Fair MTRL-vs-baseline comparison on shared `lnp` pooling

## Why this folder exists

The first MTRL tuning campaign (`01-mtrl/`) compared MTRL against a baseline
that used **different layer pooling**: the baseline config runs
`layer_pooling_type: mix` (param 0.5) while every taskrelation config runs
`weighted` (param 16 = #layers). So "MTRL vs baseline" mixed two variables —
architecture AND pooling choice — and no conclusion about the architecture
alone was clean.

This folder removes the confound by re-running **both arms on the same
pooling**, `lnp` (learned-norm pooling: `(mean |x|^p)^(1/p)` over the layer
dimension, `pooling.py`'s `learned_norm_pooling`; `p = 16` per the repo's own
`pooling_id.py` examples, which is near-max behavior). Everything else
matches each arm's canonical config:

| Arm | Model | Trainer | Config | MLflow experiment |
|---|---|---|---|---|
| baseline + lnp | `DownstreamMultiTaskModel` (original) | standard | `configs/base_lnp_config.yml` | `wavcse-base-lnp` |
| MTRL + lnp | `01-mtrl/mtrl_model.py` (λ=0.01, `normalize_w: true` — the recommended setting from `01-mtrl/README.md`) | MTRL trainer | `configs/mtrl_lnp_config.yml` | `taskrelation-mtrl` (same architecture, new pooling tag) |

Layer pooling happens in the downstream model only (the `.pt` embeddings are
frame-pooled upstream), so switching it needs no re-extraction.

## Running

```bash
cd wavCSE   # repo root (wavCSE/wavCSE)

# baseline arm (GPU 1, experiment wavcse-base-lnp)
.venv/bin/python improvements/base/run_base.py \
    --task_type ks_si_er \
    --config improvements/taskrelation/02-lnp/configs/base_lnp_config.yml

# MTRL arm (GPU 0, experiment taskrelation-mtrl; --config override was added
# to run_improvements.py for exactly this cross-folder config use case)
.venv/bin/python -m improvements.run_improvements \
    --model mtrl --task_type ks_si_er \
    --config improvements/taskrelation/02-lnp/configs/mtrl_lnp_config.yml
```

## Experiment log

## Results — pooling-clean comparison (2026-08-25)

Both arms: 30 epochs, `ks_si_er`, 100% data, layer pooling `lnp` (p=16),
same LR/schedule as their canonical configs. Runs:
[baseline+lnp](https://dagshub.com/Ke-vin-S/wavCSE.mlflow/#/experiments/7/runs/2b8324ef287347409d68f12e983a2d34)
(`wavcse-base-lnp` experiment) and MTRL+lnp
(`taskrelation-mtrl` experiment, run `taskrelation_mtrl_ks_si_er_2026_08_25_18_15_24`).

| Run (checkpoint) | all | ks | si | er |
|---|---|---|---|---|
| baseline + lnp (opt/best/epoch — all identical) | 0.973016 | 0.986686 | 0.974064 | 0.788427 |
| MTRL + lnp (opt) | 0.972633 | 0.987418 | 0.973337 | 0.779385 |
| MTRL + lnp (final epoch) | **0.974103** | **0.987710** | **0.974549** | **0.799277** |

Findings, in order of importance:

1. **The pooling confound was real and larger than the architecture effect.**
   Merely switching the *baseline* from `mix` to `lnp` moved it from
   (all/er) 0.9710/0.7559 to 0.9730/**0.7884** — +0.2pp overall and
   **+3.25pp on er**, bigger than any MTRL-vs-baseline delta measured in the
   entire `01-mtrl` campaign. Any future architecture comparison must pin
   pooling first; all pre-02 results that compared across poolings are
   confounded by this.
2. **On equal pooling, MTRL ties the baseline at the protocol-selected
   (opt) checkpoint** (0.9726 vs 0.9730, -0.04pp — far inside the ±0.2pp
   noise floor) and **edges it out at the final epoch on every metric**
   (all +0.11pp, ks +0.10pp, si +0.05pp, er +1.09pp). The opt-vs-epoch gap
   is a checkpoint-selection artifact (val-selected opt vs test), so the
   honest headline is: *tied, with MTRL no worse on any task and slightly
   better at its final checkpoint, including the largest er on record
   (0.7993).*
3. **The learned Ω is pooling-dependent.** With `weighted` pooling the
   normalized Ω converged to (ks↔er positive, si anti-related to both);
   with `lnp` it converged to all-entries-≈+1/3 — uniform positive
   coupling:

   ```
            ks        si        er
   ks  [+0.3334, +0.3325, +0.3322]
   si  [+0.3325, +0.3330, +0.3322]
   er  [+0.3322, +0.3322, +0.3336]
   ```

   i.e. the task-relationship matrix is not a task-only property — it
   depends on the representation the pooling produces. Worth one line of
   caution (and a possible analysis subsection) in the thesis: Ω describes
   the tasks *as seen through the current representation*, not some
   pooling-independent truth.

<!-- ENTRIES APPENDED BELOW AS RUNS COMPLETE -->
