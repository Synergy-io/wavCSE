# RAMUSA Layer-Reduction Experiment

The 24 configs under `configs/layer_reduction/` use the GPU-0 RAMUSA baseline
settings, seed 42, and the exact preserved-layer sets in the experiment spec.
The all-layer baseline is not included. GPU 0 runs top-layer, even-alternate,
and odd-alternate dropping; GPU 1 runs symmetric, bottom-layer, and
contribution-based dropping. Each launcher executes its 12 runs sequentially.

## Preflight and launch

From the repository root:

```bash
./.venv/bin/python improvements/lowrank/layer_reduction.py validate
bash improvements/lowrank/run_layer_reduction_gpu0.sh
bash improvements/lowrank/run_layer_reduction_gpu1.sh
```

The launchers also run preflight before training and activate the repository
`.venv` when it exists. To skip configs with completion markers on a later
invocation, set `SKIP_COMPLETED=1`. By default, a completed config causes the
launcher to stop rather than overwrite it. Failed or interrupted runs do not
get a completion marker and do not produce a completed summary row.

Each config has unique results and checkpoint roots under
`results_ramusa[/_gpu1]/layer_reduction/<strategy>/k<k>_seed42` and
`checkpoints_ramusa[/_gpu1]/layer_reduction/<strategy>/k<k>_seed42`. Timestamped
trainer run directories live beneath those roots. Per-attempt logs are written
to `logs/layer_reduction/gpu<index>/<strategy>/`; the matching `.completed`
marker is created only after training, both checkpoint evaluations, and
summary aggregation succeed.

## Evaluation and cache

The existing `run.py` path performs training and evaluates both `opt` and
`best` checkpoints. The strategy tables use the `best` checkpoint's structured
`layer_reduction_metrics.json`: task accuracy for KS/SID/ER, evaluator `acc_all`
and `loss_all`, plus macro accuracy computed as the unweighted mean of the
three task accuracies. `layer_reduction.py record` updates the strategy
`results_summary.log` idempotently and the shared
`logs/layer_reduction/master_results.csv` under a file lock.

Precomputed embeddings contain the full 25-layer WavLM Large representation.
The cache key uses only upstream model and frame-pooling identity, and the
downstream dataset selects requested layer indices after loading; no
layer-specific cache is needed. `run.py` sizes weighted pooling to the selected
layer count, and weighted/gated pooling now checks that the incoming layer axis
matches its learned position-weight vector.

Python, NumPy, PyTorch CPU, and available PyTorch CUDA random generators are
seeded from the config before data loaders and the model are created. No
deterministic/CuDNN settings are changed.