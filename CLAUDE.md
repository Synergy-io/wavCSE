# CLAUDE.md

Guidance for Claude Code when working in this repo directly (`cd wavCSE/wavCSE/`).

## What this is

wavCSE: a two-stage speech multi-task-learning pipeline (frozen WavLM Large
embedding extraction → downstream multi-task training). Baseline for the FYP
"Analyzing the Impact of Multi-task Learning Strategies on Speech Task
Performance." This repo is a git submodule of
`github.com/Ke-vin-S/fyp-multi-task-learning`, managed from the wrapper
folder one level up. Full docs, paper summary, and architecture deep-dive
live there: `../docs/`, `../WAVCSE_ANALYSIS.md`.

## Environment

```bash
uv sync   # creates .venv/ — Python 3.9, PyTorch 2.7.1+cu126, deps pinned in uv.lock
uv run python ...   # or: source .venv/bin/activate
```

Dependencies are managed with uv (`pyproject.toml` + committed `uv.lock`); the
old conda setup was removed. See UV.md for the full workflow.

## Running the baseline pipeline

```bash
# Stage 1 — upstream embedding extraction (run from upstream/)
cd upstream && python main.py --config configs/extract_embedding.yml

# Stage 2 — downstream multi-task training + eval (run from downstream/)
cd downstream && python main.py --config configs/build_model.yml
```

Decoupled by `.pt` embedding files on disk — if pooling/upstream config
changes, re-extract before downstream sees it.

## `improvements/` at a glance

Our MTL work, structured **by owner** (`taskrelation/`, `lowrank/`,
`clustering/`, `decomposition/`) so each person's branch merges cleanly.
`taskrelation/` (Kevin) has four variants: three flat exploratory ones
(GBC/TSM/PMR, in `models/`/`trainers/`/`configs/`) plus `01-mtrl/`, the first
of a numbered convention (`0N-<name>/`) for self-contained architecture
folders; the rest of the owner folders are scaffolded. Entry point:
`python -m improvements.run_improvements --model [gbc|tsm|pmr|mtrl|original|all]`
(must be run as a module from the repo root — see Gotchas).
`improvements/base/` runs the plain baseline with MLflow/DagsHub tracking.
Shared helpers: `mlflow_utils.py`, `loading_utils.py`. See
`.claude/rules/improvements.md` for the full experiment-running workflow —
kept out of this file since it only matters when actually working there.

## Gotchas

- **Python 3.9** — no `match` statements, no `X | Y` type unions.
- **`downstream/` is never edited.** It mirrors baseline code co-authored by
  the co-supervisor and is synced as a submodule into the team repo. New
  logic goes in `improvements/` as new files (see precedent:
  `taskrelation/trainers/*.py` subclass rather than edit `trainer_model.py`).
- **`LoadEmbedding`'s `device=` must be CPU**, via
  `improvements/loading_utils.get_loader_device()` — not the training GPU.
  `MultiTasksModelTrainer`/`MultiTasksModelEvaluator` already move each batch
  to the training device explicitly, so the embedding load doesn't need to
  target CUDA — and CUDA contexts can't be safely shared across forked
  `DataLoader` worker processes, which is why `num_workers > 0` is only safe
  once loading targets CPU.
- **`run_improvements.py` must be invoked as a module**:
  `python -m improvements.run_improvements ...` from the repo root — running
  the file directly (`python improvements/run_improvements.py`) fails with
  `ModuleNotFoundError: No module named 'improvements'`, since only
  `python -m` puts the repo root on `sys.path` (the script's own directory
  isn't enough for its `from improvements.taskrelation....` imports to
  resolve).
- **`embedding.tar.gz` is DVC-tracked** (DagsHub remote via `.dvc/`), not
  committed to git directly.
- WavLM Large is ~1.2 GB on disk; extraction needs a GPU with enough VRAM.
- **Root disk fills up fast** — `results_*`/`checkpoints_*` accumulate a
  timestamped directory per run, never auto-cleaned, and the disk has
  repeatedly run at ~100% full (single-digit GB free). A training run's
  checkpoint save can fail mid-write (`PytorchStreamWriter failed writing
  file ...`) if it does. Check `df -h` before a new run, especially a long
  one, and clear old timestamped run dirs if space is low.

## Git

Submodule of `github.com/Ke-vin-S/fyp-multi-task-learning` — run git
commands from here, not the wrapper. Stage selectively, never `git add -A`.
Imperative-mood commit messages explaining why, one logical change per
commit.
