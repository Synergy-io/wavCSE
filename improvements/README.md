# improvements/

## Layout

```
improvements/
├── base/            # plain wavCSE downstream model, MLflow-tracked (no new architecture)
├── taskrelation/     # Kevin -- GBC, TSM, PMR
├── lowrank/          # Chehan -- not yet implemented
├── clustering/        # Induwara -- NCMTL three-task candidate-network clustering
├── decomposition/    # Pathumi -- not yet implemented
├── mlflow_utils.py    # shared MLflow helpers, used by base/ and importable by any owner folder
└── run_improvements.py  # runs taskrelation variants
```

## Prerequisites

- Upstream embeddings already extracted to disk (see `upstream/` -- this folder only runs the downstream stage).
- Conda env `opencv` with `mlflow` and `python-dotenv` installed (`environment.yml`).
- `.env` at the repo root (`wavCSE/wavCSE/.env`, copy from `.env.example`) with:
  ```
  MLFLOW_TRACKING_URI=https://dagshub.com/Ke-vin-S/wavCSE.mlflow
  MLFLOW_TRACKING_USERNAME=<your dagshub username>
  MLFLOW_TRACKING_PASSWORD=<your dagshub access token>
  ```

## TensorBoard (enabled by default)

Every trainer launched from `improvements/` writes TensorBoard events to
`<results_root>/<run_id>/tensorboard/`. The dashboard includes train/validation
loss and accuracy, per-task metrics, learning rate, parameter and gradient
norms, relative parameter changes, and periodic parameter histograms.

```bash
tensorboard --logdir results_ncmtl
```

To tune or disable it, add this optional block to any improvement config:

```yaml
tensorboard:
  enabled: true
  gradient_log_interval: 100
  histogram_epoch_interval: 5
  # log_dir: ~/custom_tensorboard_directory
```

## Where data lives (this server)

Raw datasets (audio + label/split metadata) -- required even though embeddings are precomputed, because `torchaudio`'s dataset classes need the file listing:

| `root_data_path` subfolder | Points at |
|---|---|
| `~/voice_dataset/speechcommand` | `~/voice_dataset_local/speechcommand` (downloaded fresh -- the shared copy at `/data/braveenan/voice_dataset/SpeechCommand` has a `750 root:root` directory blocking read access) |
| `~/voice_dataset/voxceleb` | `~/voice_dataset_local/voxceleb` (`wav/` symlinked from the shared copy, `iden_split.txt` downloaded fresh -- the shared copy's `iden_split.txt` is `600 root:root`) |
| `~/voice_dataset/iemocap` | symlink -> `/data/braveenan/voice_dataset/IEMOCAP` (shared copy, fully readable) |
| `~/voice_dataset/fluentspeechcommand` | symlink -> `/data/braveenan/voice_dataset/FluentSpeechCommand` (shared copy, fully readable) |

Precomputed upstream embeddings (`.pt` files + CSVs), one folder per dataset:
```
~/dataset/embedding/wavlm_large/mean/{speechcommand,voxceleb,iemocap,fluentspeechcommand}/
```
`base_config.yml`'s `paths.root_emb_path` points here.

## Run the base model

```bash
cd improvements/base
python run_base.py --task_type ks_si_er --config configs/base_config.yml [--device_index 0]
```

- `--task_type`: underscore-joined task tokens -- `ks` (keyword spotting), `si` (speaker ID), `er` (emotion), `ic` (intent). Any subset, e.g. `ks_si_er_ic` for all four.
- Runnable from any working directory (paths are resolved relative to `run_base.py`, not cwd).
- Trains + evaluates (opt/best/epoch checkpoints), writes local results to `results_base/<run_id>/` and `checkpoints_base/<run_id>/`, and logs params/live per-epoch metrics/test metrics/artifacts to MLflow under the `wavcse-baseline` experiment.
- View runs: https://dagshub.com/Ke-vin-S/wavCSE.mlflow

For a long unattended run, detach it from the terminal so it survives a dropped connection:
```bash
# 3 original wavCSE tasks (ks_si_er)
nohup python run_base.py --task_type ks_si_er --config configs/base_config.yml > /tmp/run_base_ks_si_er.log 2>&1 < /dev/null &
disown

# all 4 tasks, including intent classification (ks_si_er_ic)
nohup python run_base.py --task_type ks_si_er_ic --config configs/base_config.yml > /tmp/run_base_ks_si_er_ic.log 2>&1 < /dev/null &
disown
```
Each backgrounds a separate process -- run both if you want both baselines, or just one. Progress: `tail -f /tmp/run_base_<task_type>.log`, or watch the run live on the MLflow/DagsHub page above.

## Run a taskrelation variant (existing)

```bash
python run_improvements.py --model [gbc|tsm|pmr|original|all] --task_type ks_si_er
```
Tracked in MLflow the same way as `base/` (params/live epoch metrics/test
metrics/artifacts). Config is hardcoded to `taskrelation/configs/`.

## Experiment naming & tagging convention

Every owner folder shares one MLflow tracking server (DagsHub, see above) but
gets its **own experiment per (category, architecture)** pair, so runs stay
comparable within an architecture while still being groupable across all of
them. The `wavcse-` prefix is reserved for base and base-derived runs only --
it marks "a variant of the baseline." Every other category is its own
top-level architecture, not a baseline variant, so it's named without the
prefix:

| Category | Experiment name pattern | Example |
|---|---|---|
| base | `wavcse-baseline` | -- |
| base + kfold | `wavcse-baseline-er-kfold` | -- |
| base + small improvement (not a full new architecture) | `wavcse-base-<improvement-slug>` | `wavcse-base-poolingsweep` |
| taskrelation (Kevin -- GBC/TSM/PMR) | `taskrelation-<model>` | `taskrelation-gbc`, `-tsm`, `-pmr` |
| lowrank (Chehan) | `lowrank-<variant>` | once a variant is named |
| clustering (Induwara) | `clustering-<variant>` | once a variant is named |
| decomposition (Pathumi) | `decomposition-<variant>` | once a variant is named |

Run names follow `{category}_{model}_{task_type}_{timestamp}`, e.g.
`taskrelation_gbc_ks_si_er_2026_08_18_10_00_00` -- built by
`mlflow_utils.build_run_name(category, model, task_type)`.

Every run also carries standard tags (`mlflow_utils.set_standard_tags`), so
runs are filterable/groupable *across* experiments via
`MlflowClient.search_runs`, not just within one: `category`, `model`,
`pooling_frame` (`cfg.pooling.frame_pooling_type`), `pooling_layer`
(`cfg.pooling.layer_pooling_type`). All other hyperparameters are already
logged as MLflow params by `mlflow_utils.log_config_params` (the whole config,
flattened) -- tags are only for the coarse axes you want to slice on.

When implementing `lowrank/`, `clustering/`, or `decomposition/`: copy
`taskrelation/`'s config layout (add an `mlflow:` block with
`experiment_name` following the table above), add your model_type(s) to
`run_improvements.py`'s `MODEL_CATEGORY` dict (or your own run script if you
don't reuse `run_improvements.py`), and the tracking wiring in
`build_model`/`build_trainer`/`run_single_model` applies automatically.

## Cross-category leaderboard

```bash
python mlflow_report.py [--metric metrics.test_opt_acc_all] [--top 20] [--csv report.csv]
```
Pulls every run across every experiment matching the table above via
`mlflow.search_runs()` and prints one sorted comparison table -- this is what
answers "compare, group, see progress" across categories, since MLflow's own
UI only compares runs within a single experiment at a time.
