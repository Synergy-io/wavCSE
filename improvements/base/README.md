# Base (plain wavCSE, MLflow-tracked)

Not an owner's MTL architecture -- this runs the original, unmodified
`DownstreamMultiTaskModel` (same model `run_improvements.py --model original`
would use) so its results can serve as the baseline every architecture in
`taskrelation/`, `lowrank/`, `clustering/`, `decomposition/` is compared
against. It exists to add MLflow experiment tracking (backed by a
DagsHub-hosted tracking server) without touching `downstream/`.

- `run_base.py` -- orchestration script: loads embeddings, builds the plain
  `DownstreamMultiTaskModel`, trains and evaluates it via the unmodified
  `downstream/trainer` and `downstream/evaluator`, and logs params/metrics/
  artifacts to MLflow using `improvements/mlflow_utils.py`.
- `configs/base_config.yml` -- full copy of `downstream/configs/build_model.yml`'s
  schema plus an `mlflow:` block (`tracking_uri`, `experiment_name`).
- `run_base_er_kfold.py` / `kfold_iemocap.py` / `configs/base_kfold_config.yml`
  -- leave-one-speaker-out cross-validation for the `er` task, on top of the
  same joint model. See "ER 10-fold cross-validation" below.

## Running

```bash
cd wavCSE/wavCSE/improvements/base
python run_base.py --task_type ks_si_er --config configs/base_config.yml --device_index 0
```

Can be run from any working directory -- all paths (downstream/, improvements/,
this folder's own config) are resolved relative to `run_base.py`'s own location,
not the working directory.

## MLflow / DagsHub setup

1. Copy `.env.example` (at the `wavCSE/wavCSE/` repo root) to `.env` and fill in
   your DagsHub username + access token. `.env` is gitignored -- never commit it.
2. `base_config.yml`'s `mlflow.tracking_uri` points at
   `https://dagshub.com/Ke-vin-S/wavCSE.mlflow`. Change it if you're tracking
   against a different DagsHub repo.
3. Each run appears under the `wavcse-baseline` experiment on DagsHub's
   Experiments tab, with logged hyperparameters, per-epoch train/val loss and
   accuracy curves (overall + per task), final test metrics for the `opt`/
   `best`/`epoch` checkpoints, and artifacts (plots, checkpoints, prediction
   CSVs) pulled from the same `results_base/`/`checkpoints_base/` directories
   this script writes locally regardless of MLflow.

## ER 10-fold cross-validation (leave-one-speaker-out)

`downstream/dataset/load_embedding.py`'s `_load_iemocap()` pools all 5
IEMOCAP sessions (10 speakers) together and takes a fixed positional stride
(`indices[4::10]`/`indices[9::10]`) for val/test. That's speaker-leaky --
train and val/test almost certainly share speakers, just different
utterances from them, so the model can partly succeed by memorizing
speaker-specific acoustic patterns rather than learning emotion cues that
generalize. This inflates `er` accuracy on the single-split baseline and
makes it an unreliable number to tune against.

**Fix: leave-one-speaker-out (LOSO) cross-validation**, the standard SER
evaluation protocol on IEMOCAP -- it has exactly 5 sessions x 2 speakers =
10 speakers, which maps onto 10 folds. `run_base_er_kfold.py` implements
this:

- **Only `er`/IEMOCAP is folded.** `ks` (SpeechCommand) and `si` (VoxCeleb)
  keep their normal official splits every fold -- they're loaded
  independently of IEMOCAP (confirmed in `LoadEmbedding.load_embedding()`),
  so folding one doesn't require folding the others.
- **Training stays joint** (`ks_si_er`) each fold, matching the base
  model's actual architecture (shared backbone + task heads trained
  together) -- this measures whether the *joint* model generalizes across
  ER speakers, not an isolated single-task ER classifier.
- **Fold assignment**: speakers sorted deterministically
  (`Ses01F, Ses01M, Ses02F, ...`). For fold `i`: `test = speakers[i]`,
  `val = speakers[(i+1) % 10]`, `train = the remaining 8 speakers`. Every
  speaker is used as test exactly once and as val exactly once across the
  10 folds -- using the test speaker itself for validation would leak
  information about the fold being scored.
- **Fresh model per fold** -- no cross-fold weight sharing, standard
  k-fold practice.
- **5 epochs per fold**, not the 30 used for the tuned single-split
  baseline -- a deliberate compute-budget tradeoff, since this is 10 full
  joint trainings instead of 1. Expect per-fold numbers to be noisier than
  the tuned baseline; this run answers "does `er` generalize across
  speakers," not "best possible tuned number." All other hyperparameters
  (`patience`, `factor`, `learning_rate`, ...) are unchanged from
  `base_config.yml` -- not in scope for this experiment.
- `downstream/` is still never touched: `_LOSOLoadEmbedding` in
  `run_base_er_kfold.py` reuses `LoadEmbedding._load_iemocap()`'s
  `IEMOCAPEmbedding` construction unmodified and only re-slices its output
  into LOSO indices.

```bash
cd wavCSE/wavCSE/improvements/base
python run_base_er_kfold.py --task_type ks_si_er --config configs/base_kfold_config.yml --num_folds 10 --device_index 0

# detached (survives a dropped connection)
nohup python run_base_er_kfold.py --task_type ks_si_er --config configs/base_kfold_config.yml > /tmp/run_base_er_kfold.log 2>&1 < /dev/null &
disown
```

Each fold logs to its own nested MLflow run (`fold_0` ... `fold_9`) under one
parent run (`base_ks_si_er_kfold_<timestamp>`) in the
`wavcse-baseline-er-kfold` experiment on DagsHub, tagged with its
`held_out_test_speaker`/`held_out_val_speaker`. The parent run aggregates
`er` accuracy/loss mean and std across folds per checkpoint tag
(`kfold_er_{opt,best,epoch}_{acc,loss}_{mean,std}`) and attaches
`results_base_kfold/kfold_summary.json` (per-fold + aggregate numbers) as an
artifact. Per-fold local outputs land in
`results_base_kfold/fold_<i>/`/`checkpoints_base_kfold/fold_<i>/`.
