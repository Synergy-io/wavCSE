# CONTINUATION — Task Relation Learning (Kevin)

Everything needed to pick this work up cold: the project, the conventions,
where things live, how to run them, what has already been tried, what the
results actually say (including corrections/retractions), what is still
open, and the traps that will waste your time if you don't know them.

**Status of this document: 2026-09-21.** Facts below were verified against
the working tree on that date. The three *runnable* sources of truth are the
per-folder READMEs, the code itself, and MLflow/DagsHub — when this file and
those disagree, they win.

---

## 1. Quick orientation

| Question | Answer |
|---|---|
| What is this? | FYP: *Analyzing the Impact of Multi-task Learning Strategies on Speech Task Performance* (target ACL 2027) |
| My part | **Task Relation Learning** (Zhang & Yang 2021 §2.4) — `improvements/taskrelation/` |
| Active architecture | **MTRL** (`01-mtrl/`) — learns a task covariance matrix Ω via `tr(W Ω⁻¹ Wᵀ)` |
| Has it worked? | **No.** 16L, 25L, LOSO, and 5-seed checks all agree: no reproducible MTRL-vs-baseline advantage. See §9 |
| Best config found | `smp` pooling (λ=0.5) — a real, reusable win (see §9.3) |
| Where do I read results? | `01-mtrl/README.md`, `base/POOLING_GRID_SEARCH.md`, `base/README.md`; live on DagsHub |
| What's next? | §10 — the MTRL decision is the blocking one |
| Biggest landmine | `er` numbers from the standard split are **speaker-leaky**, ~15pp inflated. Use LOSO (§9.2) |

---

## 2. The project

**Title:** Analyzing the Impact of Multi-task Learning Strategies on Speech
Task Performance. **Target venue:** ACL 2027.

**Research problem.** There are few MTL models in speech processing and none
that combine all four task types (content, speaker, emotion, intent). wavCSE
(Sritharan & Thayasivam, IJCNLP 2025) covers only three, and does so with a
**feature-based** approach — one shared encoder, equal-weighted loss. This
project builds the **first parameter-based MTL architectures for speech**,
adapting methods developed mostly for computer vision.

**The two benchmarks every architecture must clear:**

1. **Beat wavCSE** — parameter-based must beat the feature-based baseline.
2. **Beat single-task models** — prove training tasks together beats training
   them separately.

**Tasks.** Content/keyword-spotting (`ks`, 12 classes), Speaker ID (`si`,
1251 classes), Emotion (`er`, 4 classes), plus Intent Classification (`ic`,
31 classes) as an expansion target. `ks_si_er` is the standard 3-task setup;
`ks_si_er_ic` the 4-task one.

**Deliverables.** A Python MTL library with innovative task-imbalance
handling, plus a systematic experimental analysis (the paper).

**People.**

| Role | Who |
|---|---|
| Supervisor | Dr. Uthayasanker Thayasivam ("Udaya sir") |
| Co-supervisor | Braveenan Sritharan (wavCSE author; co-authored `downstream/`) |
| Kevin | Task Relation Learning (§2.4) — `improvements/taskrelation/` |
| Chehan | Low-Rank (§2.2) — `improvements/lowrank/` (RAMUSA, per that folder's README) |
| Induwara | Task Clustering (§2.3) — `improvements/clustering/` (NCMTL, implemented) |
| Pathumi | Decomposition (§2.5) — `improvements/decomposition/` (FTN, implemented) |

**Rhythm.** Weekly supervisor meetings **Thursday 8:30 AM**. Weekly work goes
under `weekly/week-NN/` with shared `meeting-summary.md` and per-person `my/<name>/`.

Literature research happened in weeks 2–4 and is already done for all four
branches — see §4 for where that material lives; **you do not need to redo it.**

---

## 3. My subpart: Task Relation Learning

**The branch.** Zhang & Yang's §2.4: instead of forcing tasks into discrete
clusters or assuming a global low-rank structure, learn a **continuous,
pairwise similarity** between tasks — a matrix Ω — directly from data. It's
the most general of the four parameter-based branches (task clustering =
block-diagonal Ω; low-rank = rank-constrained Ω).

**What exists under `improvements/taskrelation/`:**

| Folder / file | What | Status |
|---|---|---|
| `01-mtrl/` | **MTRL** (Zhang & Yeung 2010/2014) — the central §2.4 method. Covariance Ω, closed-form update, `normalize_w` lever | **ACTIVE — formal starting method of the programme and its primary existing baseline method (DEC-0001).** Heavily evaluated; **negative result** (F4). Next contribution is diagnosis, not replacement |
| `02-lnp/` | Pooling-confound control: re-ran baseline vs MTRL on identical `lnp` pooling | **DIAGNOSTIC / CONTROL only — not an active method (DEC-0002).** Done; findings retained |
| `03-gbc/` | **GBC** — Global Bias Coupling (original design; see §11.4 on its citation) | **ARCHIVED / out of current formal scope (DEC-0003).** 2 runs, no results; must not motivate or rank literature-grounded methods |
| `models/tsm_model.py`, `trainers/tsm_trainer.py`, `configs/tsm_config.yml` | **TSM** (Ciliberto et al. 2015) — latent-basis structure matrix, alternating minimization | **QUARANTINED, unvalidated (DEC-0004)** — never trained, no MLflow run |
| `models/pmr_model.py`, `trainers/pmr_trainer.py`, `configs/pmr_config.yml` | **PMR** (Gonçalves et al. 2016) — sparse precision matrix | **QUARANTINED, unvalidated (DEC-0004)** — never trained; known shape bug (§11.5) |

`tsm`/`pmr` are the flat "exploratory" layout (predating the numbered-folder
convention); GBC was promoted to `03-gbc/` so it gets the same rigor MTRL got.
**TSM and PMR have no MLflow experiments at all** — they have never completed
a run. Neither is a backlog item: their validity audit (DEC-0004) is pending.
The scope classifications above were set by the 2026-09-21 research-scope reset —
see `research/DECISIONS.md`.

**Where my research lives** (all done, do not repeat):
- Survey deep-dive on §2.4: `weekly/week-02-mtl-architectures-survey/my/kevin/2.4_Task_Relation_Learning_Approach_explained.md`
- All four branches compared: `.../my/kevin/MTL_Architectures_Beyond_LibMTL.md`
- 30 classical §2.4 papers indexed: `weekly/week-03-more-mtl-architectures/my/kevin/papers/README.md`
- 45 modern (post-2020) MTL papers ranked: `weekly/week-04-recent-advances/my/kevin/RANKINGS.md`
- Classical vs modern head-to-head (MTRL vs Grad-TAG): `.../my/kevin/MTRL_vs_GradTAG.md`
- Architecture-integration diagrams per candidate method: `.../my/kevin/WAVCSE_ARCHITECTURE_INTEGRATION.md`
- Consolidated architecture-decision report: `weekly/week-05-project-proposal/my/kevin/mtl-evolution-and-trl-architecture-decision.md`
- Paper list tracked globally: `docs/papers.md`

---

## 4. Repo structure — three levels, three CLAUDE.md files

```
fyp-multi-task-learning/                  ← git repo #1 (docs, weekly, wrappers)
├── CLAUDE.md                             ← project-wide: people, conventions, gotchas
├── docs/{papers.md, project-description.md}
├── weekly/week-NN/{meeting-summary.md, assets/, my/<person>/}
├── LibMTL/LibMTL/
└── wavCSE/                               ← MANAGEMENT WRAPPER (not the code repo)
    ├── CLAUDE.md                         ← wavCSE-specific guidance
    ├── docs/01..07-*.md                  ← paper summary, architecture, FYP context, quick ref
    ├── WAVCSE_ANALYSIS.md                ← deep-dive on wavCSE internals
    ├── improvements/docs/                ← presentation assets ONLY
    └── wavCSE/                           ← git repo #2 — THE CODE
        ├── CLAUDE.md                     ← shared team guidance (generic only)
        ├── CLAUDE.local.md               ← personal notes, gitignored
        ├── .claude/rules/improvements.md ← experiment-running workflow (loaded for improvements/**)
        ├── upstream/                     ← stage 1: WavLM extraction
        ├── downstream/                   ← stage 2: the model. NEVER EDIT (see §11.2)
        ├── improvements/                 ← all our architectures
        ├── UV.md, pyproject.toml, uv.lock
        └── .env                          ← DagsHub creds, gitignored
```

**The trap that bites everyone:** there are **two** `improvements/` folders.
Code lives in `wavCSE/wavCSE/improvements/` (inner). `wavCSE/improvements/`
(outer) holds presentation assets only. Check which one you're in.

Git commands run from the **inner** repo (`wavCSE/wavCSE/`). Remotes:
`origin` = `github.com/Synergy-io/wavCSE` (ours), `upstream` = `github.com/aaivu/wavCSE`.

### Pipeline (two decoupled stages)

```
Raw audio → WavLM-Large (frozen, 25 layers) → frame pooling → layer pooling
          → .pt embeddings on disk
          → DownstreamMultiTaskModel → shared backbone (2 FC) → per-task heads
          → KS / SID / ER metrics
```

**The stages are decoupled by `.pt` files.** Changing upstream pooling config
means re-extracting embeddings. **But layer selection is downstream-only** —
every saved `.pt` already contains all 25 layers; `selected_transformer_layers`
is applied at load time. So switching between the 16-layer list and `all` is a
config change, **no re-extraction** (verified: 253,811 `.pt` files, all `[25, 1024]`).

### `improvements/` layout

```
improvements/
├── base/           # plain wavCSE model + MLflow tracking; the thing everyone is compared to
│   ├── run_base.py, run_base_er_kfold.py, kfold_iemocap.py, run_pooling_grid.py
│   ├── configs/{base,base_kfold,base_smoke,base_alllayers,base_poolingwinner_16L,base_poolingwinner_25L}_config.yml
│   └── POOLING_GRID_SEARCH.md, README.md, pooling_grid_results_{16,25}L.json
├── taskrelation/   # MINE — see §3
├── lowrank/        # Chehan
├── clustering/     # Induwara (NCMTL)
├── decomposition/  # Pathumi (FTN)
├── run_improvements.py     # entry point for taskrelation variants
├── mlflow_utils.py         # shared tracking helpers
├── loading_utils.py        # get_loader_device() → always CPU
├── seed_utils.py           # set_seed() — no run is reproducible without it
├── tensorboard_utils.py    # TensorBoardTrainerMixin
├── mlflow_report.py        # cross-experiment leaderboard
└── er_tag_report.py        # opt/best/epoch rescan for er
```

---

## 5. Environment and data

```bash
cd /home/induwara/kevin-mtrl/fyp-multi-task-learning/wavCSE/wavCSE
uv sync                                  # creates .venv (Py 3.9, torch 2.7.1+cu126)
.venv/bin/python -m improvements.run_improvements ...   # preferred
# or: uv run python ... / source .venv/bin/activate
```

**Conda is gone.** The old `opencv` conda env was removed mid-project when the
team migrated to uv. Any doc, script, or habit that says `conda activate
opencv` is stale — use `.venv`. `environment.yml` may still exist in older
references; `pyproject.toml` + `uv.lock` are authoritative.

**Data (this server).** Both are symlinks into the shared `/data/braveenan/`
— set them up once if missing:

```bash
ln -s /data/braveenan/embedding ~/embedding        # 127GB, all layers, world-readable
# ~/voice_dataset/* already symlinked (iemocap, fluentspeechcommand from the
# shared copy; speechcommand, voxceleb from ~/voice_dataset_local/ due to ACLs)
```

**Never copy or re-extract embeddings** — the root disk has only single-digit
to ~40GB free and 127GB will not fit. `~/dataset/embedding` is an alternative
symlink some configs point at; both resolve to the same data.

**`embedding.tar.gz` at the repo root is a 137-byte DVC pointer**, not data.

**GPUs.** 2× Tesla T4 (15GB each), on a **shared machine** — other users' jobs
appear and disappear on either GPU. Check before launching; expect contention.

---

## 6. How to run things

All commands assume `cd` to the repo root (`wavCSE/wavCSE/`) unless noted.

| What | Command |
|---|---|
| Baseline (plain wavCSE) | `python improvements/base/run_base.py --task_type ks_si_er --config improvements/base/configs/base_config.yml --device_index 0` |
| Baseline, 4-task | same, `--task_type ks_si_er_ic` |
| MTRL | `python -m improvements.run_improvements --model mtrl --task_type ks_si_er --device_index 0` |
| MTRL, cross-folder config | `python -m improvements.run_improvements --model mtrl --config improvements/taskrelation/02-lnp/configs/mtrl_lnp_config.yml` |
| GBC | `python -m improvements.run_improvements --model gbc --config improvements/taskrelation/03-gbc/gbc_poolingwinner_16L_config.yml --seed 42` |
| Any variant + explicit seed | add `--seed <int>` (overrides config; defaults to 42 if neither set) |
| ER LOSO k-fold (baseline) | `cd improvements/base && python run_base_er_kfold.py --task_type ks_si_er --config configs/base_kfold_config.yml --num_folds 10 --device_index 0` |
| ER LOSO k-fold (MTRL) | `python improvements/taskrelation/01-mtrl/mtrl_er_kfold.py --task_type ks_si_er --config improvements/taskrelation/01-mtrl/mtrl_kfold_config.yml --num_folds 10 --device_index 0` |
| Pooling grid (one layer count per process) | `cd improvements/base && python run_pooling_grid.py --num_layers 16 --device_index 0` |
| Cross-category leaderboard | `python improvements/mlflow_report.py [--metric metrics.test_opt_acc_all] [--top 20]` |
| TensorBoard | `tensorboard --logdir results_<whatever>` (every trainer writes `results_*/<run_id>/tensorboard/`) |

**Long runs:** detach them (`nohup ... > /tmp/x.log 2>&1 < /dev/null & disown`).
Two independent runs can go on different GPUs concurrently — that's how the
pooling grid was done (16L on GPU 0, 25L on GPU 1).

**Before any long run:** `df -h /` (see §11.1) and `nvidia-smi`.

---

## 7. Conventions

### Experiment naming & tagging (MLflow/DagsHub)

One experiment per (category, architecture) pair. Only base-derived runs carry
the `wavcse-` prefix.

| Category | Pattern | Live examples |
|---|---|---|
| base | `wavcse-baseline` | |
| base + kfold | `wavcse-baseline-er-kfold` | |
| base + small improvement | `wavcse-base-<slug>` | `wavcse-base-poolingsweep`, `wavcse-base-lnp` |
| taskrelation | `taskrelation-<model>` | `taskrelation-mtrl`, `-mtrl-er-kfold`, `-gbc` |
| other owners | `<category>-<variant>` | `wavcse-ncmtl*`, `wavcse-decomposition-ftn`, `wavcse-ramusa`, `lowrank-wavcse-paper` |

Run names: `{category}_{model}_{task_type}_{timestamp}` via
`mlflow_utils.build_run_name()`. Every run also carries standard **tags**
(`category`, `model`, `pooling_frame`, `pooling_layer`) via
`set_standard_tags()` so runs are groupable *across* experiments with
`MlflowClient.search_runs`. All hyperparameters are logged as **params** by
`log_config_params()` (the whole config, flattened) — tags are only for the
coarse axes you want to slice on. Full detail: `improvements/README.md`.

### Config schema

Every config is a YAML copy of `downstream/configs/build_model.yml` plus an
`mlflow:` block. Key fields:

```yaml
upstream: {model_type: wavlm_large, selected_transformer_layers: <"all" | comma-list>}
pooling:  {frame_pooling_type: mean, layer_pooling_type: <smp|mix|weighted|lnp|…>, layer_pooling_param: <float|int|null>}
model:    {embedding_dim_shared1: 512, embedding_dim_shared2: 2000, dropout_prob_shared1: 0.4, dropout_prob_shared2: 0.6}
training: {num_epochs: 30, batch_size: 2048, learning_rate: 0.0025, patience: 1|5, factor: 0.5, …}
mlflow:   {tracking_uri: https://dagshub.com/Ke-vin-S/wavCSE.mlflow, experiment_name: <per table above>}
```

Notes: `layer_pooling_param` for `weighted`/`gated` is auto-set to
`len(layers)` at runtime (leave `null`); for `smp` it's the softmax sharpness
λ=0.5; for `lnp` it's the norm power (16 used historically, int-cast); `mix`
is the max/mean blend ratio. `num_workers: 4` + `pin_memory: true` are
deliberate (§11.3).

### Numbered architecture folders (`0N-<name>/`)

Self-contained folders for "real" attempts: own model, optional trainer,
config, README. `01-mtrl/` (first), `03-gbc/` (second), `02-lnp/` (a
comparison experiment, not an architecture). A leading digit + hyphen makes
them invalid Python package names, so `run_improvements.py` loads them through
`_load_module_from_path()` (importlib file-path loader) and finds their configs
via `CONFIG_PATH_OVERRIDES` — see `.claude/rules/improvements.md`.

### Checkpoint tags

Three tags are evaluated per run: **`opt`** (checkpoint selected by *average
per-task* validation accuracy — the usual headline), **`best`** (by *overall*
validation accuracy), **`epoch`** (final epoch). They are all
validation-based selections, never test-based. Across 113 runs, `opt` vs
`best`/`epoch` on `er` is a wash (19 better / 32 worse / 62 tied) — pick a tag
per run on merit, but don't expect `opt` to be systematically wrong.

### Reproducibility

**Always pass a seed** (`--seed` or a `seed:` key) for anything you intend to
compare. Before `seed_utils.py` existed, no run was seeded and identical
configs differed by up to 2.35pp on `er` — that variance is what produced and
then retracted the "MTRL wins" claim. Any comparison of <1pp needs ≥5 seeds.

### Commits and docs

- Stage selectively — **never `git add -A`/`git add .`**; check
  `git diff --cached --name-only`. Imperative messages, explain *why*, one
  logical change per commit.
- PDFs are gitignored; track every paper in `docs/papers.md` (status:
  to-read/reading/done).
- `sessions/` folders and exported `.txt` conversation dumps are **archived
  conversations — do not read them** unless explicitly asked for a named file.
- Per-folder READMEs carry the experiment logs; keep appending to them rather
  than starting new files.

---

## 8. Where experiments are tracked

**DagsHub MLflow:** https://dagshub.com/Ke-vin-S/wavCSE.mlflow
(creds in `wavCSE/wavCSE/.env`, gitignored — never print them).

Live experiment inventory (2026-09-21, run counts approximate):

| Experiment | Runs | What |
|---|---|---|
| `wavcse-baseline` | 37 | base, all variants |
| `wavcse-baseline-er-kfold` | 27 | base ER LOSO |
| `wavcse-base-poolingsweep` | 60 | the pooling grid (screening + confirmation) |
| `wavcse-base-lnp` | 2 | the earlier lnp-pooling baseline pair |
| `taskrelation-mtrl` | 24 | MTRL (mine) |
| `taskrelation-mtrl-er-kfold` | 11 | MTRL ER LOSO |
| `taskrelation-gbc` | 2 | GBC — barely started |
| `wavcse-ncmtl` (+ 6 sub-experiments) | 36+ | Induwara's clustering |
| `wavcse-decomposition-ftn` | 35 | Pathumi's FTN |
| `wavcse-ramusa` | 65 | Chehan's low-rank |
| `lowrank-wavcse-paper` | 1 | — |

Query pattern used throughout (read-only, safe):

```python
from dotenv import load_dotenv; load_dotenv(".env")
import mlflow
mlflow.set_tracking_uri("https://dagshub.com/Ke-vin-S/wavCSE.mlflow")
c = mlflow.tracking.MlflowClient()
exp = c.get_experiment_by_name("taskrelation-mtrl")
for r in c.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=10):
    print(r.data.tags.get("mlflow.runName"), r.data.metrics.get("test_opt_acc_all"))
```

**Local artifacts** land in timestamped `results_<x>/<run_id>/` and
`checkpoints_<x>/<run_id>/` (gitignored). `results_*` holds the per-epoch text
log, plots, prediction CSVs, TensorBoard events, and — for MTRL —
`omega_history.json` (full per-update Ω matrices + final matrix). MLflow gets
the same plots/CSVs as artifacts, plus the best/opt checkpoints.

**Two report scripts** worth knowing: `mlflow_report.py` (cross-experiment
leaderboard table) and `er_tag_report.py` (per-run `er` accuracy for every
checkpoint tag, no retraining).

---

## 9. Results to date — and what they actually mean

### 9.1 Headline table

All numbers are `test_opt_acc_all` unless noted, 30 epochs, `ks_si_er`, 100%
data, single split.

| Config | all | ks | si | er |
|---|---|---|---|---|
| baseline, `mix` 0.5, 16L (original) | 0.9710 | 0.9854 | 0.9735 | 0.7559 |
| baseline, `mix` 0.5, 25L | 0.9676 | 0.9884 | 0.9628 | 0.7812 |
| baseline, `lnp` 16, 16L | 0.9730 | 0.9867 | 0.9741 | 0.7884 |
| baseline, **`smp` 0.5, 16L** | 0.9724 | 0.9842 | 0.9765 | 0.7667 |
| baseline, **`smp` 0.5, 25L** (best single run) | **0.9767** | 0.9864 | 0.9812 | 0.7902 |
| MTRL (`weighted`, λ=0.01, `normalize_w`) 16L | 0.9695 | 0.9842 | 0.9699 | 0.7812 |
| MTRL, `smp` 0.5, 16L | 0.9728 | 0.9849 | 0.9766 | 0.7649 |
| MTRL, `smp` 0.5, 25L (opt / best tag) | 0.9744 / 0.9754 | 0.9857 / 0.9854 | 0.9789 / 0.9799 | 0.7667 / 0.7848 |

### 9.2 Two corrections that invalidate most `er` discussion

**(a) Speaker leakage — ~15pp inflation.** `downstream/dataset/load_embedding.py`'s
`_load_iemocap()` pools all 10 IEMOCAP speakers and slices train/val/test by a
fixed stride, so train and test share speakers. Every `er` number above is
therefore "how well the model memorizes these 10 speakers," not generalization.
The fix (already implemented) is **leave-one-speaker-out**:
`improvements/base/run_base_er_kfold.py` and
`improvements/taskrelation/01-mtrl/mtrl_er_kfold.py` — 10 folds, `test =
speakers[i]`, `val = speakers[(i+1)%10]`, train = other 8, fresh model per fold,
5 epochs/fold, only IEMOCAP folded.

**Honest numbers at `smp`+25L:**

| tag | baseline | MTRL | Δ |
|---|---|---|---|
| opt | **0.6391** ± 0.0506 | 0.6380 ± 0.0484 | −0.0011 |
| best | 0.6307 ± 0.0448 | 0.6423 ± 0.0491 | +0.0116 |
| epoch | 0.6278 ± 0.0453 | 0.6276 ± 0.0462 | −0.0002 |

**The true ER ceiling for this architecture is ~64%, not ~79%.** Per-fold std
is ~5pp (range 0.556–0.718) — individual speakers differ a lot, which is itself
why any single split was never trustworthy.

**(b) Seed noise — MTRL's one apparent win was retracted.** The `smp`+16L
result (MTRL 0.9728 vs baseline 0.9724) did not survive 5 seeds each:

| | mean | std | range |
|---|---|---|---|
| baseline | 0.97134 | 0.00130 | 0.96995–0.97372 |
| MTRL | 0.97124 | 0.00091 | 0.96969–0.97218 |

Same at `smp`+25L (baseline 0.97476 ± 0.00108 vs MTRL 0.97419 ± 0.00071 — a
0.056pp edge, ~1 combined SEM: borderline, not significant).

**Combined verdict: MTRL shows no reproducible advantage over the plain
baseline anywhere — 16L, 25L, LOSO, or multi-seed.** That is a consistent
negative result from every angle checked.

### 9.3 The one durable win: `smp` pooling

A full grid over **all 10 pooling methods** (18 param/type combos per layer
count, two-stage: 10-epoch screening over everything → top-3 confirmed at the
full 30 epochs, both layer counts on both GPUs concurrently) found **`smp`
(softmax pooling, λ=0.5) beat every other pooling type at both layer counts,
on both architectures, without exception.** Full tables:
`improvements/base/POOLING_GRID_SEARCH.md`.

Run accounting, so the DagsHub count isn't confusing: that experiment holds 60
runs = 36 `screen` + 6 `confirm` (the grid itself) + 18 `smoketest` (a
throwaway 1-epoch/3%-data pass over all 18 combos, run first to catch bad
pooling-parameter bugs before committing to the grid). The two MTRL
confirmation runs at the winning pooling went to `taskrelation-mtrl`, not here.

| Layer count | best pooling | `test_opt_acc_all` |
|---|---|---|
| 16L | `smp` 0.5 | 0.9724 |
| 25L | `smp` 0.5 | **0.9767** |

Two genuine failure modes found and worth citing: **`lse` produces NaN losses**
(`r=5` at 16L; *every* `r` at 25L — `exp(r·x)` overflow), and **`lnp` collapses
at 25L** (0.80–0.82, driven by `si`) regardless of its power parameter, because
its `p` is fixed and cannot adapt to a changed layer count.

**Methodological finding (supersedes an earlier conclusion):** a previous
campaign found "16 layers beats 25 layers" and wrote it up as general. The grid
showed that was **pooling-specific** — true for `mix`/`weighted`/`lnp`
(none of which can use the extra 9 layers), false for `smp`. **Layer selection
and pooling method are not independent choices.** Treat any single-axis search
as provisional until cross-checked.

### 9.4 What Ω actually tells us

The learned task-relation matrix is **representation-dependent**, and its
stability is directly informative:

| Setting | ks↔si | ks↔er | si↔er | Saturated? |
|---|---|---|---|---|
| `weighted` 16L | −0.333 | +0.333 | −0.333 | yes, sign-structured |
| `smp` 16L | +0.185 | +0.229 | −0.054 | no, differentiated |
| `smp` 25L | +0.333 | +0.333 | +0.333 | yes, uniform |
| `lnp` 16L | +0.333 | +0.333 | +0.333 | yes, uniform |

Two things worth carrying into the thesis:

1. **Saturation is a failure mode for interpretability.** When Ω saturates to
   uniform (+1/3 everywhere), the regularizer has nothing to discriminate with —
   which is a plausible explanation for MTRL doing *worse* than baseline at
   `smp`+25L, despite more layers being available.
2. **Across the 10 LOSO folds, `ks`↔`si` is stable (0.286 ± 0.012) while
   `ks`↔`er` (0.098 ± 0.072) and `si`↔`er` (0.033 ± 0.103) change sign between
   folds** — i.e. the ER relationships Ω "learns" are artifacts of which small
   slice of ER data the model saw. The KS↔SI relationship looks real; the
   ER ones don't. This is *independent structural evidence* agreeing with the
   accuracy-based conclusion.

**Thesis framing:** Ω describes tasks *as seen through the current
representation and training sample* — not a pooling- or split-independent truth.

---

## 10. Open threads — what to do next

Ranked by what unblocks the most:

1. **MTRL's fate is settled: it stays, and it is diagnosed — not replaced.**
   MTRL is the programme's formal starting method and primary existing Task
   Relation Learning baseline (DEC-0001). The ranked shortlist in
   `MTRL_vs_GradTAG.md` / `RANKINGS.md` is *not* a to-do list: progression is
   literature-gated (DEC-0005), so the order is diagnose MTRL's failure
   (`research/BACKLOG.md`, DG-0001 first) → targeted literature search for a
   published method whose assumption matches the measured failure → implement.
   The Ω interpretability angle (F5/F6) is part of that diagnosis, not an
   alternative to it.
2. **GBC is archived — it is no longer an open thread (DEC-0003).** It was never
   literature-grounded (its citation was retracted), so it cannot carry this
   project's contribution claim and receives no autonomous cycles. Do not run
   the "cheapest remaining win" screening described in earlier revisions of this
   file.
3. **TSM and PMR are quarantined, and running them is not the next step
   (DEC-0004).** They are untrained and unvalidated; they may be relied on only
   after the validity audit (mathematical validity, literature source verified
   against the actual paper, implementation correctness — PMR additionally needs
   its shape bug fixed, §11.5 — and taxonomy placement).
4. **ER improvement** (if pursuing it): the honest baseline is ~0.639, so
   there is real headroom. The evidence says the problem is overfitting, not
   representation — `er` shows a **~10–15pp train/val gap in every run** while
   `ks` shows ~0.4pp — plus a tiny dataset. Candidate fixes: regularization or
   a smaller task-specific head for `er`, class-weighted loss (check IEMOCAP
   class balance first), or an angular-margin loss (AAM/ArcFace) on `er`'s head,
   which is the standard fix for large closed-set ID from a linear head.
   **Always evaluate with LOSO**, never the leaky split.
5. **Loss-aggregation baselines** (CAGrad / Nash-MTL / RLW) — not implemented.
   The paper needs them as the comparison table for *any* parameter-based
   method, and RLW (random weighting, ~10 lines) is the sanity floor.
6. **4-task `ks_si_er_ic`** has never been tried for MTRL (§2.4 methods are
   motivated by task imbalances that a 4th task would stress further).
7. **Task-imbalance handling** is a named project deliverable; currently the
   only imbalance evidence is the per-task accuracy spread, not a method.

---

## 11. Traps, gotchas, known bugs

1. **Disk.** Root `/` has repeatedly hit ~0 free. Training writes timestamped
   `results_*`/`checkpoints_*` that are never auto-cleaned; a full disk makes
   `torch.save` fail mid-write (`PytorchStreamWriter failed writing file …`). A
   pre-run guard now exists in `run_base.py`, `run_improvements.py`, and
   `run_pooling_grid.py` (aborts below 2GB). Check `df -h /` anyway, and delete
   superseded `checkpoints_*/` dirs — best/opt checkpoints are already on
   DagsHub as artifacts. Do **not** delete `results_*_kfold/` (aggregated LOSO
   summaries live there).
2. **`downstream/` is never edited.** It mirrors code co-authored by the
   co-supervisor. New behaviour goes in `improvements/` as files that subclass
   or wrap (precedent: `taskrelation/trainers/*.py` subclass
   `MultiTasksModelTrainer`; `_LOSOLoadEmbedding` subclasses `LoadEmbedding`).
3. **`LoadEmbedding(device=...)` must be CPU** — use
   `improvements/loading_utils.get_loader_device()`. Trainers/evaluators move
   batches to the GPU themselves, and CUDA contexts can't be shared across
   forked DataLoader workers. This is what makes `num_workers: 4` safe and
   ~40× faster than the shipped `num_workers: 0` default.
4. **`run_improvements.py` must be run as a module** (`python -m
   improvements.run_improvements …`) from the repo root; direct file invocation
   fails with `No module named 'improvements'`.
5. **PMR has a real, unfixed shape bug** (`models/pmr_model.py:get_task_parameter_matrix`):
   it flattens each head's weight+bias into rows of *different lengths*
   (`ks` 12×2000, `si` 1251×2000, `er` 4×2000) and `torch.stack`s them → would
   crash once the regularizer turns on after warmup. MTRL avoids this by
   mean-pooling each head into a fixed `(hidden+1)`-length row; PMR needs the
   same treatment before it can run.
6. **GBC has a harmless-but-confusing double bias**: `self.classifiers[t]` is a
   `nn.Linear` *with* its own bias, and the forward pass adds a separate
   `self.task_biases[t]` on top (`gbc_model.py:91-99,164-166`) — the docstring's
   "no bias from Linear" comment is wrong. Not broken; just don't be misled.
7. **Python 3.9** — no `match`, no `X | Y` unions.
8. **Two `improvements/` folders** (§4) and **two git repos** — always confirm
   which you're in.
9. **`CLAUDE.local.md` may contain stale warnings.** It previously flagged
   `.claude/rules/improvements.md` as reverted/stale; as of 2026-09-21 the
   working tree is clean (`git status` shows only an untracked
   `.claude/settings.json`) and the rules file is current. Trust `git status` /
   `git diff` over any note claiming otherwise.
10. **Shared machine.** Other users' jobs appear on either GPU mid-run
    (observed repeatedly). It has never corrupted a run here, but figure it in
    when timing things, and prefer the other GPU when one is occupied.
11. **`smp`/`lnp`/`mix` param meanings differ** — λ (sharpness), *p* (norm
    power, int-cast), blend ratio respectively. `weighted`/`gated` take
    `len(layers)` and are auto-computed.

---

## 12. Cheat sheet

```bash
cd /home/induwara/kevin-mtrl/fyp-multi-task-learning/wavCSE/wavCSE   # repo root

# health check before any run
df -h / && nvidia-smi

# baseline at the best-known config
python improvements/base/run_base.py --task_type ks_si_er \
  --config improvements/base/configs/base_poolingwinner_25L_config.yml --device_index 0

# MTRL at the best-known config
python -m improvements.run_improvements --model mtrl --task_type ks_si_er \
  --config improvements/taskrelation/01-mtrl/mtrl_poolingwinner_25L_config.yml --seed 42

# honest ER number (LOSO) — never quote the single-split one
cd improvements/base && python run_base_er_kfold.py --task_type ks_si_er \
  --config configs/base_kfold_config.yml --num_folds 10 --device_index 0

# what's already logged
python improvements/mlflow_report.py --metric metrics.test_opt_acc_all --top 20
python improvements/er_tag_report.py
```

**Read these, in this order, for results:** `01-mtrl/README.md` (MTRL's full
story, top-of-file callouts carry the retractions) →
`base/POOLING_GRID_SEARCH.md` (the durable win) → `base/README.md` (LOSO
protocol + honest ER numbers + checkpoint-tag rescan) → `03-gbc/README.md` and
`02-lnp/README.md` (the other two lines). Then the live DagsHub experiments.
