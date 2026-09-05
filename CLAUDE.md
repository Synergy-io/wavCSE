# CLAUDE.md

Guidance for Claude Code working in this repo. This file is shared with the
whole team — keep it generic and project-wide; put personal notes in
`CLAUDE.local.md` instead (gitignored, not shared).

## Project

**Analyzing the Impact of Multi-task Learning Strategies on Speech Task
Performance** (target venue: ACL 2027). wavCSE is the feature-based MTL
baseline for the project; this repo extends it with **parameter-based** MTL
architectures adapted from computer vision, one per team member, each
required to clear two benchmarks: beat wavCSE's feature-based baseline, and
beat single-task models trained separately.

| Branch (Zhang & Yang 2021 taxonomy) | Owner | Folder |
|---|---|---|
| Task Relation Learning (§2.4) | Kevin | `improvements/taskrelation/` |
| Low-Rank Approach (§2.2) | Chehan | `improvements/lowrank/` |
| Task Clustering (§2.3) | Induwara | `improvements/clustering/` |
| Decomposition (§2.5) | Pathumi | `improvements/decomposition/` |

## Architecture: two-stage pipeline

```
Raw Audio → WavLM Large (frozen) → Frame Pooling → Layer Pooling → .pt Embeddings
                                                                        │
                                    DownstreamMultiTaskModel  ←─────────┘
                                         │
                                    Per-Task Heads → Metrics (KS, SID, ER)
```

**`upstream/`** (stage 1): frozen WavLM Large extracts all 25 transformer
layer outputs; frame pooling reduces the time dimension; layer
concatenation + FC projects to a fixed-size embedding; saved as `.pt` files.

**`downstream/`** (stage 2): loads `.pt` embeddings → shared backbone (2 FC
layers) → task-specific classification heads, trained jointly. Baseline
tasks: keyword spotting (KS), speaker identification (SID), emotion
recognition (ER).

The two stages are decoupled by `.pt` files on disk — changing
pooling/upstream config requires re-extracting embeddings before downstream
sees the change.

## Environment

```bash
uv sync              # creates .venv/ — Python 3.9, PyTorch 2.7.1+cu126, deps pinned in uv.lock
uv run python ...    # or: source .venv/bin/activate
```

See `UV.md` for the full workflow.

## Running

```bash
# Stage 1 — upstream embedding extraction
cd upstream && python main.py --config configs/extract_embedding.yml

# Stage 2 — downstream multi-task training + eval
cd downstream && python main.py --config configs/build_model.yml
```

## `improvements/` at a glance

Each owner's MTL architecture lives in its own folder
(`taskrelation/`/`lowrank/`/`clustering/`/`decomposition/`) so branches merge
without colliding on shared `models/`/`configs/`/`trainers/` files. Common
entry point:

```bash
python -m improvements.run_improvements --model <name> --task_type ks_si_er
```

`improvements/base/` runs the plain baseline with MLflow/DagsHub tracking.
Shared helpers: `mlflow_utils.py`, `loading_utils.py`. See
`.claude/rules/improvements.md` for the full experiment-running workflow
(MLflow conventions, disk-space handling) — kept out of this file since it
only matters when actively running experiments.

## Gotchas

- **Python 3.9** — no `match` statements, no `X | Y` type unions.
- **`downstream/` is never edited — except the sanctioned `mtlkit` migration.**
  It mirrors baseline code co-authored by the project's co-supervisor. New
  logic normally goes in `improvements/` as new files that subclass rather
  than modify (precedent: `taskrelation/trainers/*.py` subclasses
  `trainer/trainer_model.py`). **Exception (Eng Review decision D1,
  2026-09-06, see `docs/designs/speech-mtl-framework.md`):** this pass
  deliberately rewrites `downstream/`'s modules into thin wrappers over the
  new `mtlkit/` package, gated by facade-parity and numeric-parity tests
  (`mtlkit/tests/test_facade_parity.py`) rather than left to trust. Every
  symbol other consumers import keeps its exact signature and behavior. Any
  FUTURE change to `downstream/` outside this migration still needs the
  same bar: a stated, reviewed exception plus a parity test proving nothing
  broke — not a silent one-off edit.
- **`LoadEmbedding`'s `device=` must be CPU**, via
  `improvements/loading_utils.get_loader_device()` — not the training GPU.
  Trainers/evaluators move each batch to the training device explicitly, and
  CUDA contexts can't be safely shared across forked `DataLoader` worker
  processes, so `num_workers > 0` is only safe once loading targets CPU.
- **`run_improvements.py` must be invoked as a module**
  (`python -m improvements.run_improvements ...`) from the repo root, not
  run as a file directly — its `from improvements....` imports only resolve
  once the repo root is on `sys.path`, which only `python -m` guarantees.
- **`embedding.tar.gz` is DVC-tracked** (DagsHub remote via `.dvc/`), not
  committed to git directly.
- **This is a shared training machine — disk fills up fast.** Training runs
  write timestamped `results_*`/`checkpoints_*` directories that are never
  auto-cleaned, and the root disk has repeatedly hit single-digit GB free. A
  checkpoint save fails mid-write (`PytorchStreamWriter failed writing file
  ...`) if it does. Check `df -h` before a long run and clear old timestamped
  dirs if space is low.
- WavLM Large is ~1.2 GB on disk; extraction needs a GPU with enough VRAM.

## Git

Stage selectively — never `git add -A` or `git add .`. Commit messages:
imperative mood, explain why not what, one logical change per commit.
