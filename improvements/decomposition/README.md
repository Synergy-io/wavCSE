# Decomposition-Based Sharing

Decomposition (Zhang & Yang §2.5), Factorized Tensor Network-inspired

## Architecture

This folder implements decomposition-based multi-task learning with
FTN-inspired low-rank task updates. It is an application of the decomposition
concept, not a literal reproduction of the original FTN network.

For each task `t`:

\[
W_t = W_shared + Delta W_t
\]

\[
Delta W_t = U_tV_t
\]

The block is inserted after the baseline FC2 representation and its dropout,
and before the existing task classifiers. There is one trainable shared
adapter and one bias-free low-rank update for each task. The shared adapter is
identity-initialized, while each task update starts at zero through a
zero-initialized up-projection.

No residual connection, activation, normalization, decomposition-specific
loss, or dense trainable delta matrix is used.

## Scope

This first version supports only the exact ordered task string `ks_si_er`:

- `ks` — Keyword Spotting
- `si` — Speaker Identification
- `er` — Emotion Recognition

Intent Classification (`ic`) is intentionally excluded from this version.

## Files

- Model: `models/ftn_model.py`
- Config: `configs/ftn_config.yml`
- Canonical runner: `run_ftn.py`
- Trainer: `trainers/ftn_trainer.py`, a thin gradient-clipping/scheduler
  extension of the standard downstream trainer
- Evaluator: existing standard implementation under `downstream/`

## Run

From this directory:

```bash
python run_ftn.py \
  --task_type ks_si_er \
  --config configs/ftn_config.yml \
  --device_index 0
```

For a long unattended run that survives a disconnected terminal:

```bash
nohup python run_ftn.py \
  --task_type ks_si_er \
  --config configs/ftn_config.yml \
  --device_index 1 \
  > /tmp/run_ftn_ks_si_er.log 2>&1 < /dev/null &
disown
```

Monitor its output with:

```bash
tail -f /tmp/run_ftn_ks_si_er.log
```

The runner can also be invoked from another working directory because its
default paths are resolved relative to `run_ftn.py`.

The model is also available through the shared improvements runner:

```bash
python improvements/run_improvements.py --model ftn --task_type ks_si_er --device_index 0
```

## Outputs and tracking

- Local results: `results_decomposition_ftn/`
- Local checkpoints: `checkpoints_decomposition_ftn/`
- MLflow experiment: `wavcse-decomposition-ftn`

Relative output paths used by the canonical runner are owned by this folder.
The rank is configured as `model.ftn_rank`. The stabilized full-data config
uses rank 16, learning rate `0.001`, gradient clipping at `1.0`, and an
immediate plateau scheduler with a minimum learning rate of `0.00001`.

In MLflow, the runner additionally records balanced task accuracy, the
pre-clipping gradient norm, each low-rank update norm, and the shared adapter's
distance from its identity initialization. These diagnostics make late shared
parameter drift and task interference visible in subsequent runs.
