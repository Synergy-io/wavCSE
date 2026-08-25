---
paths:
  - "improvements/**"
---

# Working in improvements/

## MLflow / DagsHub tracking

`.env` (gitignored, copy from `.env.example`) holds
`MLFLOW_TRACKING_USERNAME`/`MLFLOW_TRACKING_PASSWORD` (a DagsHub access
token) — loaded via `python-dotenv`'s `load_dotenv()`. Never print or expose
these values. `mlflow.tracking_uri` in each config points at
`https://dagshub.com/Ke-vin-S/wavCSE.mlflow`. One MLflow experiment per
(category, architecture) pair -- `wavcse-baseline` (single-split base runs),
`wavcse-baseline-er-kfold` (ER leave-one-speaker-out CV runs),
`taskrelation-{gbc,tsm,pmr,mtrl}`, and `lowrank-*`/`clustering-*`/
`decomposition-*` once those owners implement their architectures. Only
base/base-derived experiments carry the `wavcse-` prefix. Full naming/tagging
convention and the cross-category `mlflow_report.py` leaderboard script are
documented in `improvements/README.md`.

## Running `improvements/base/`

```bash
cd improvements/base
python run_base.py --task_type ks_si_er --config configs/base_config.yml --device_index 0

# detached (survives a dropped connection)
nohup python run_base.py --task_type ks_si_er --config configs/base_config.yml > /tmp/run_base.log 2>&1 < /dev/null &
disown
```

Full details, including the `ks_si_er_ic` (4-task) variant, in
`improvements/base/README.md`.

## Running `improvements/run_improvements.py`

Must be invoked as a module from the repo root (`wavCSE/wavCSE/`), not run
directly:

```bash
python -m improvements.run_improvements --model [gbc|tsm|pmr|mtrl|original|all] --task_type ks_si_er
```

Running `python improvements/run_improvements.py ...` fails with
`ModuleNotFoundError: No module named 'improvements'` -- the script's own
directory (not the repo root) ends up on `sys.path[0]`, so its
`from improvements.... ` imports can't resolve unless the repo root is on
`sys.path`, which `python -m` guarantees and a direct file invocation
doesn't.

## Numbered architecture folders (`0N-<name>/`)

`taskrelation/01-mtrl/` is the first of a convention for self-contained
architecture folders (own `<name>_model.py`/`<name>_trainer.py`/
`<name>_config.yml`/`README.md`) -- for a "real" architecture attempt, as
opposed to the flat `models/`/`trainers/`/`configs/` layout GBC/TSM/PMR use.
A leading digit and a hyphen make `01-mtrl` an invalid Python package path,
so `run_improvements.py`'s `build_model()`/`build_trainer()` load these
folders' modules via `_load_module_from_path()` (an `importlib.util`
file-path loader defined near the top of the file) instead of a dotted
`from improvements.taskrelation.01-mtrl...` import, which is a
`SyntaxError`. Its config is found via `CONFIG_PATH_OVERRIDES` rather than
the flat `taskrelation/configs/<model>_config.yml` lookup every other
variant uses. Full architecture writeup + experiment log:
`taskrelation/01-mtrl/README.md`.

## ER 10-fold cross-validation (`run_base_er_kfold.py`)

Fixes speaker leakage in `downstream/dataset/load_embedding.py`'s
`_load_iemocap()`, which pools all 10 IEMOCAP speakers and slices train/val/
test by a fixed stride — train and test end up sharing speakers, inflating
`er` accuracy via speaker memorization rather than genuine emotion
generalization.

- **Protocol**: leave-one-speaker-out. IEMOCAP has 5 sessions × 2 speakers =
  10 speakers → 10 folds. Speakers sorted deterministically; for fold `i`:
  `test = speakers[i]`, `val = speakers[(i+1) % 10]`, `train` = the other 8.
  Every speaker is test exactly once, val exactly once, across all folds.
- **Only `er`/IEMOCAP is folded** — `ks`/`si` keep their normal official
  splits every fold, since they're loaded independently
  (`_load_speechcommand`/`_load_voxceleb`). Training stays joint
  (`ks_si_er`), matching the real architecture.
- **5 epochs per fold** (not the 30 used for the tuned single-split
  baseline) — a deliberate compute-budget tradeoff since this is 10 full
  joint trainings. Expect noisier per-fold numbers; this measures
  cross-speaker generalization, not a best-tuned number.
- Implementation reuses `LoadEmbedding._load_iemocap()`'s
  `IEMOCAPEmbedding` construction unmodified (via `_LOSOLoadEmbedding`
  subclassing in `run_base_er_kfold.py`) and only re-slices its output —
  `downstream/` is never touched. Full design rationale in
  `improvements/base/README.md`'s "ER 10-fold cross-validation" section.
- Reported result: aggregated `er` accuracy mean ± std across all 10 folds
  (in `results_base_kfold/kfold_summary.json` and the parent MLflow run's
  `kfold_er_{opt,best,epoch}_acc_mean/std` metrics) — not any single fold.

## Data-loading: always load to CPU

Every `LoadEmbedding(...)` call in `improvements/` must pass
`device=loading_utils.get_loader_device()` (always `torch.device("cpu")`),
**never** the training GPU device. Reason: `MultiTasksModelTrainer`/
`MultiTasksModelEvaluator` already move each batch to the training device
explicitly right before use, so the embedding load itself doesn't need
CUDA — and CUDA contexts can't be shared across forked `DataLoader` worker
processes. Loading to CPU is what makes `num_workers > 0` safe. All configs
under `improvements/` (`base/configs/*.yml`, `taskrelation/configs/*.yml`)
therefore use `num_workers: 4` / `pin_memory: true`, not the `0`/`false`
`downstream/configs/build_model.yml` ships with. Without this, training is
I/O-bound (GPU idles at 0% between batches, serial single-process
`torch.load()` per sample) — with it, throughput improved roughly 40x in
testing.

## Dataset paths

`~/voice_dataset` is a set of symlinks (case-normalized to lowercase) into
the shared `/data/braveenan/voice_dataset/`. Two paths there
(`SpeechCommand/SpeechCommands/speech_commands_v0.01/`,
`VoxCeleb/iden_split.txt`) were previously blocked by restrictive
permissions for non-root users; both are now readable via POSIX ACL grants
added externally. Local fallback copies still exist at
`~/voice_dataset_local/{speechcommand,voxceleb}/` if the shared path ever
breaks again — no need to re-download from scratch.

`~/embedding` (the `root_emb_path` every `taskrelation/configs/*.yml` uses)
is **not** created by default -- symlink it once with
`ln -s /data/braveenan/embedding ~/embedding` (same pattern as
`~/voice_dataset` above; `/data/braveenan/embedding` already holds every
extracted `wavlm_large/mean/{speechcommand,voxceleb,iemocap,
fluentspeechcommand}` embedding set, ~127GB, world-readable). Do **not**
copy/extract it -- the root disk usually has only a few GB free (see "Disk
space" below).

## Disk space

The root disk (`/`) has repeatedly run at ~100% full (single-digit GB
free). Training runs write timestamped `results_*/checkpoints_*`
directories that are never auto-cleaned; a checkpoint save can fail
mid-write (`PytorchStreamWriter failed writing file ...`,
`RuntimeError: ... file write failed`) if the disk fills up during a run.
Check `df -h` before starting a new run, especially a long one, and clear
out old timestamped run directories under `results_*/checkpoints_*` if
space is low.
