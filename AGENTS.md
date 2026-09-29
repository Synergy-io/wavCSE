# AGENTS.md — wavCSE

## What this repository is

`wavCSE` is the research repository for *Analyzing the Impact of Multi-task Learning Strategies on
Speech Task Performance*. It is a two-stage speech framework: `upstream/` extracts frozen WavLM Large
embeddings (frame pooling → layer pooling → `.pt` files), and `downstream/` trains a shared backbone
with one classification head per task — keyword spotting (KS), speaker identification (SI/SID) and
emotion recognition (ER) — over those embeddings. `downstream/` is the feature-based multi-task
baseline to beat; `improvements/` holds the parameter-based MTL architectures, one folder per research
branch. This repository's active programme is **Task Relation Learning** under
`improvements/taskrelation/`: characterize task properties → task relationships → transfer behaviour,
then justify parameter-based MTL mechanisms from those relationships — a framework for choosing
mechanisms, not merely higher accuracy.

## Boundary — what does not belong here

- **Infrastructure is a different repository.** Controller and worker lifecycle, GPU provisioning, job
  submission, storage transfer, artifact publication and caching belong to the separate `wavcse-infra`
  checkout and are reached through its `infra` CLI. Never reimplement provisioning, transfer, caching
  or publication in ad-hoc shell here, and never add infrastructure logic to this repository.
- **`downstream/` is frozen.** It mirrors baseline code co-authored by the project's co-supervisor:
  read it, never edit it. New behaviour goes in `improvements/` as new files that subclass existing
  classes rather than modify them.
- **Other branches are other scopes.** `improvements/lowrank/`, `clustering/` and `decomposition/` are
  separate MTL branches in the Zhang & Yang (2021) taxonomy; Task Relation Learning work stays in
  `improvements/taskrelation/` and stays distinct from them.
- **Procedure a skill already covers** is pointed at, never restated here (see *Skills*).
- The fixed upstream wavCSE/WavLM embedding changes only when the human changes project scope.

## Hard invariants — never violate

A skill mentioning one of these does not relax it.

**Evidence and claims**

1. Never modify test data, and never use test metrics for model selection — selection uses the
   validation set only.
2. A single-seed result is never a confirmed improvement. Never promote an architecture from one run:
   screen cheaply first, then confirm with multiple seeds.
3. Serious ER claims require speaker-independent leave-one-speaker-out (LOSO) evaluation; the
   historical ER single split has speaker leakage and can never be cited as an improvement.
4. Established findings must not be rediscovered as new. Search `STUDIES.jsonl`, `FINDINGS.md` and
   `FAILURES.md` before starting a study. Established, among others: single-seed wins are unreliable;
   MTRL has no reproducible significant improvement; pooling choice can create larger effects than
   architecture choice; learned Ω is considerably more stable for KS↔SI than for relations with ER.
5. Every architecture change begins with a falsifiable hypothesis, stated with a plausible competing
   explanation. One study tests one primary scientific hypothesis.
6. Measure task behaviour, not only aggregate accuracy. When a method fails, determine why before
   proposing another method. When local ideas are exhausted, search the literature from the observed
   failure mode, never generically for "better MTL".

**Comparability and protocol**

7. Never compare architectures across different pooling, embedding/layer selection, data split, epoch
   budget or evaluation protocol unless that difference is the explicit independent variable. Hold
   every other factor constant.
8. Never silently modify the baseline evaluation protocol.
9. Task Relation Learning must remain distinct from the low-rank, clustering and decomposition
   research branches.

**Records and provenance**

10. Every training run carries a Study ID. One study may contain many runs and seeds.
11. Commit the implementation and config before launching confirmation runs; the commit used by every
    run must be recorded.
12. Every MLflow run carries the standard research tags and a DagsHub run note.
13. Record failed experiments; never delete negative evidence. Negative evidence is reclassified,
    never erased.
14. Study records preserve ISO 8601 `created_at`, `started_at`, `completed_at` and status-transition
    dates, so weekly progress is reconstructed from evidence rather than conversation. Never rewrite
    historical timestamps.
15. Hyperparameter tuning is allowed only when scientifically justified — never a broad or
    opportunistic sweep to improve a reported number. Start from literature-recommended,
    theoretically natural or matched-baseline settings; change only hyperparameters relevant to the
    mechanism studied; keep the tuning budget comparable across competing methods; select on
    validation only, from a small predefined space, recording every attempt including failures;
    freeze the selected configuration before multi-seed confirmation and LOSO; never tune on test
    seeds or folds. Procedure: `.agents/skills/wavcse-experiment-operator/SKILL.md`.
16. At most two simultaneous GPU training jobs; check `df -h` and `nvidia-smi` before GPU training.

**Scope of the formal programme**

17. `improvements/taskrelation/01-mtrl/` is the active starting method and formal Task Relation
    Learning baseline; future mechanisms are motivated by experimentally observed MTRL limitations.
    `02-lnp/` is diagnostic-only — it showed pooling/representation choice can confound architecture
    comparisons and learned task relations; use that evidence, but LNP is not an active architecture.
    `03-gbc/` is archived: no defensible matching published method is established, so it is never
    presented as literature-grounded and never optimized unless the human explicitly reactivates it.
    Preserve all implementations and historical results.
18. Default progression: wavCSE baseline → MTRL → MTRL behavioural diagnostics → identify the
    modelling limitation → targeted literature search → literature-supported extension → screening →
    multi-seed confirmation → LOSO where required → ablation → framework synthesis. Never skip from
    MTRL to arbitrary new architectures.

## Validation — the gate for any change

```bash
uv sync                 # environment comes from uv.lock; commit uv.lock when pyproject.toml changes
make check              # agent assets + compute backend + research tests (no GPU, no network)
uv run python -m improvements.run_improvements --help    # entry-point import smoke
```

`make check` excludes exactly one module, `research/tests/test_mssl_omega_solver.py`,
which fails at HEAD (2 failures, 1 error) on the unregistered `04-mssl` draft; fixing
it means deciding MSSL's intended mathematics, so it is a scientific call, not a
mechanical repair. `make research-check-all` runs it, and `make research-check` prints
the exclusion rather than hiding it. Never weaken or delete a failing test to make
the gate pass.

- Run the improvements entry point **as a module** from the repository root
  (`python -m improvements.run_improvements --model <name> --task_type ks_si_er`); run as a file, its
  `from improvements....` imports do not resolve.
- Python 3.9: no `match` statements, no `X | Y` type unions.
- `LoadEmbedding(device=…)` must stay CPU via `improvements/loading_utils.get_loader_device()` — never
  the training GPU; trainers move each batch to the training device themselves, and `num_workers > 0`
  is safe only once loading targets CPU.
- Shared machine, disk fills fast: training writes never-auto-cleaned `results_*` / `checkpoints_*`
  dirs and a checkpoint save fails mid-write when full — check `df -h` before long runs.

## Engineering conventions

- **Paid compute goes through the backend, inside an envelope.** `improvements/compute`
  (`python -m improvements.compute …`) is the only route to paid compute: it drives the
  infrastructure CLI over its JSON contract, derives spend from provider facts, keys
  submissions by a deterministic job identity, and sweeps what it created. Worker
  deadlines are enforced by `reap` on a controller timer, not by the orchestrator staying
  alive; the exact commit must be provably available to workers before a worker is
  created; and a stored result is validated as this run's evidence before it is
  collected. The compute backend's README is the contract for all four. The human's
  written authority is `improvements/taskrelation/research/authorizations/<SCOPE>.yaml`;
  the backend consumes it and can never create, renew or widen it. What may be decided
  without the researcher is classified in `.agents/policies/autonomy.md`. Runtime facts
  (worker and job identifiers, prices, leases, spend) live under the controller's state
  directory, never in Git.
- **Numbered architecture folders.** A real architecture attempt is self-contained in
  `improvements/<category>/0N-<name>/` (`<name>_model.py`, optional `<name>_trainer.py`, `<name>_config.yml`,
  `README.md`), not the older flat `models/`/`trainers/`/`configs/` layout. The leading digit and
  hyphen make it an invalid Python package path, so `run_improvements.py` loads it by file path with
  `_load_module_from_path()` and resolves its config via `CONFIG_PATH_OVERRIDES`.
- **MLflow naming.** One experiment per (category, architecture) pair: `wavcse-baseline`,
  `wavcse-baseline-er-kfold`, `taskrelation-<model>`, `lowrank-*`/`clustering-*`/`decomposition-*`;
  the `wavcse-` prefix is reserved for base and base-derived runs, and credentials come from the
  gitignored `.env` (loaded with `load_dotenv()`) and are never printed. Details, tags,
  `improvements/mlflow_report.py`: `improvements/README.md`.
- **Data loading.** Improvement configs keep `LoadEmbedding(device=…)` on the CPU path (see
  *Validation*), so they set `num_workers: 4` / `pin_memory: true`, not downstream's `0`/`false`:
  CUDA contexts cannot be shared across forked `DataLoader` workers, so GPU loading leaves training
  I/O-bound. A `root_emb_path` root missing on a fresh machine is symlinked to the shared store,
  never copied or extracted (the root disk is small).
- **Git and repo hygiene.** Stage selectively — never `git add -A` or `git add .`; commit messages
  are imperative mood, explain why not what, one logical change per commit. `embedding.tar.gz` is
  DVC-tracked — pointer file at the repository root, remote under `.dvc/` — never committed to git directly.

## Skills — load the one that fits the work

- `.agents/skills/wavcse-research-runner/SKILL.md` — orchestrator and **default entry point for
  whole-system requests** (run the next research cycle, make embeddings ready, get the system
  experiment-ready). It reconciles state and delegates to the companions.
- `.agents/skills/wavcse-experiment-operator/SKILL.md` — designing, running, evaluating and recording
  one Task Relation Learning study: plan requirements, matched controls, evidence tiers (including the
  ER leave-one-speaker-out fold design), diagnostics vs mechanisms, tuning policy, MLflow/DagsHub
  provenance, closing a study.
- `.agents/skills/wavcse-embedding-generation/SKILL.md` — embedding readiness and extraction
  semantics: canonical artifact reuse, scientific configuration and invariants, source readiness,
  validation, dataset-specific (KS/SID/ER/FSC) pitfalls.
- Infra-side companions live in the `wavcse-infra` checkout: `wavcse-infra-operator`,
  `gpu-research-operator`, `wavcse-artifact-pipeline`. Load the relevant one before consequential work
  in its domain and follow it rather than paraphrasing it.
- Commands: `.agents/commands/{wav-cycle,wav-experiment,wav-embeddings,wav-analyze,wav-literature,wav-status,wav-weekly}.md`.

## Current state — reconcile, never assume

`improvements/taskrelation/research/` is the research memory:

- `STATE.md` — canonical restart point; `OBJECTIVE.md` — scope and gate;
- `STUDIES.jsonl` — study registry; `studies/` — one folder per Study ID;
- `FINDINGS.md`, `FAILURES.md`, `DECISIONS.md`, `BACKLOG.md`;
- `FRAMEWORK.md`, `VARIANT_BENCHMARK_PROTOCOL.md` — the comparison contract;
- `literature/` — verified sources; `weekly/` — supervisor-facing reports.

Before proposing an experiment, read the existing evidence: the architecture READMEs (`01-mtrl/`,
`02-lnp/`, `03-gbc/`) and `improvements/base/POOLING_GRID_SEARCH.md`.

Never trust conversation memory or an earlier session for dynamic state — what ran, what is queued,
which study is open, what a run produced, which commit was executed. The repository, not the
conversation, is the research memory: re-derive those facts from the records above, run identifiers and
commit SHAs included, and report a conflict between records instead of smoothing it over. Runtime
infrastructure state (workers, volumes, jobs) is owned by the `wavcse-infra` checkout — reconcile it
there with the `infra` CLI.

## Agent-facing assets

- `.agents/skills/<name>/SKILL.md` and `.agents/commands/<name>.md` are the canonical versioned agent
  assets. OMP discovers both natively; Codex discovers the skills.
- `.omp/AGENTS.md` is a relative symlink to this file, so this file is the single source of truth for
  project instructions — edit it here, never through the symlink.
- When an agent-facing asset is added, renamed or moved, keep the layout and the symlink consistent and
  run `make agents-check`.
