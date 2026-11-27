# Network Clustering for Multi-Task Learning (NCMTL)

Owner: Induwara
MTL branch: Task Clustering

This folder implements NCMTL for keyword spotting (`ks`), speaker
identification (`si`), and emotion recognition (`er`). Use any nonempty
combination of these tasks in `--task_type`, without duplicates. Intent
classification is not supported.

## Architecture

The baseline FC1, layer pooling, and FC2 path remains shared. After the
2000-dimensional FC2 representation, every task has a cluster candidate layer
with shape `2000 -> 2000` and no bias. Its dimension is derived automatically
from the actual FC2 output dimension, so NCMTL introduces clustering without an
additional representation bottleneck. K-Means++ groups only the three candidate
weight matrices. Tasks in a cluster are projected to their exact PyTorch mean
after optimizer updates, while the final classifiers remain task-specific.

`get_all_embeddings()` continues to return the common FC2 representation before
the candidate layers.

## Layout

```text
clustering/
├── run.py
├── configs/ncmtl_config.yml
├── models/ncmtl_model.py
├── trainers/ncmtl_trainer.py
├── utils/ncmtl_clustering.py
└── tests/test_ncmtl.py
```

## Key configuration

- `num_clusters`: K for K-Means++; initially 2 and valid from 1 through 3.
- `alpha`: cluster-loss coefficient; initially 0.001.
- `cluster_every_n_batches`: training clustering interval; initially every batch.
- `warmup_epochs`: number of complete epochs of independent candidate training
  before clustering starts.
- `identical_candidate_initialization`: initializes every candidate from the same
  normally initialized weight matrix while retaining separate parameters.
- `kmeans_n_init`: number of K-Means initializations tried at each clustering step.
- `stability_patience`: unchanged partitions needed before freezing.
- `max_recluster_epochs`: last epoch during which K-Means++ may run.
- `training.label_smoothing`: smooths training targets to reduce overconfidence;
  validation and test loss remain standard unsmoothed cross-entropy.

These values are initial experiment settings, not claimed optimal values.

The 30-epoch, full-layer task-combination run is configured in
`configs/task_combinations/soft_margin_all_layers_seed42_30ep.yml`. Run all six
single and pairwise selections sequentially with
`bash configs/task_combinations/run_task_combinations_30ep.sh 0` from this
directory. The script starts with `ks_si_er`, then runs the three pairs and
the three single tasks. The GPU index is the script's optional argument. Console logs and
training results are under `logs/epoch30/task_combinations/`.

With two tasks, the one candidate pair uses normalized row distance to gate
sharing after adaptive warm-up: close rows share, distant rows stay independent,
and intermediate rows share partially. With one task, the candidate layer is
retained but there is no pair to cluster or share. The original three-task
soft-margin behavior is unchanged.

## Run

From any working directory:

```bash
cd improvements/clustering
conda activate tc
python run.py \
  --task_type ks_si_er \
  --config configs/ncmtl_config.yml \
  --device_index 0
```

The runner is independent of `improvements/run_improvements.py`. Training and
evaluation use `results_ncmtl/` and `checkpoints_ncmtl/`. Each
result run contains `cluster_history.csv` and `cluster_summary.json`, alongside
the inherited metrics, plots, and prediction files. When configured, the run is
also logged to MLflow/DagsHub; unavailable remote credentials do not prevent
local execution.

K-Means++ runs only on training batches. Validation and evaluation never
recluster. Once the assignment freezes, K-Means++ stops, but mean projection
continues after every optimizer update so cluster members remain shared.

- To run:

```bash
  nohup python run.py \
    --task_type ks_si_er \
    --config configs/ncmtl_config.yml \
    --device_index 0 \
    > /tmp/run_ncmtl_ks_si_er.log 2>&1 < /dev/null &

  disown
```
