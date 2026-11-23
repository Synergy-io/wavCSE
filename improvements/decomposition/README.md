# FTN decomposition experiments

This implementation uses the existing downstream model, loader, trainer and evaluator. For every active task `t`, FC2 uses `W_shared + U_t V_t` and the shared FC2 bias. Only active tasks have a low-rank update and classifier. A single-task experiment retains the same base-plus-residual block; its base is trained for that task alone, so it is not cross-task sharing.

The upstream checkpoint setting remains `wavlm_large`, with frozen, precomputed embeddings. The loader selects hidden states 0–24 (embedding output and 24 transformer outputs) and the existing `smp` layer pooling aggregates all 25. The runner checks the configured indices and a sample from each split; the model checks every forward input. A cache with fewer than 25 states fails instead of silently selecting a subset. Dataset preprocessing and train/validation/test splits are unchanged.

## Run one combination

From the repository root, use any of:

```bash
.venv/bin/python improvements/decomposition/run_ftn.py --tasks KS
.venv/bin/python improvements/decomposition/run_ftn.py --tasks SID
.venv/bin/python improvements/decomposition/run_ftn.py --tasks ER
.venv/bin/python improvements/decomposition/run_ftn.py --tasks KS SID
.venv/bin/python improvements/decomposition/run_ftn.py --tasks KS ER
.venv/bin/python improvements/decomposition/run_ftn.py --tasks SID ER
.venv/bin/python improvements/decomposition/run_ftn.py --tasks KS SID ER
```

`--task_type ks_si_er` remains available. `--config` and `--device_index` are supported. The configured seed, optimizer, scheduler, loss averaging, regularization, batch policy and epochs apply to every run. The command starts full training for one combination. To run all sequentially, use `bash improvements/decomposition/run_all.sh`.

Results go under `results_decomposition_ftn/<COMBINATION>/<run timestamp>/`; checkpoints go under `checkpoints_decomposition_ftn/<COMBINATION>/<run timestamp>/`. Each run records its configuration, per-task training/validation text logs, predictions, and `eval_metrics_{opt,best,epoch}.json` plus the existing text metrics. 

## Verification

```bash
PYTHONPATH=.:downstream .venv/bin/python -m unittest improvements.decomposition.test_combinations -v
```

This uses synthetic 25-layer embeddings and one tiny CPU epoch for each combination. It checks output shapes, input gradients through all 25 positions, shared and adapter gradients, checkpoint creation, and evaluation. It does not measure research accuracy. Cached embedding samples for KS, SID, and ER were each checked to have shape `(25, 1024)`.
