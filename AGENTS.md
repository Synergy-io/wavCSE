# wavCSE public model repository

This checkout owns reproducible upstream extraction, the frozen `downstream/` baseline, improvement model/trainer code, study runners and configs under `improvements/taskrelation/experiments/`, the exact-commit worker adapter under `improvements/execution/`, and public behavior tests.

- Keep `downstream/` frozen. New baseline-derived behavior belongs under `improvements/`.
- Run improvements as modules from the repository root: `uv run python -m improvements.run_improvements --help`.
- A public worker receives a SHA-verified private plan and input declaration; it must never import private research modules or discover an infrastructure checkout.
- Task Relation Learning records, literature, decisions, preflights and authorizations belong to private `Synergy-io/wavcse-research`. Never add those records, run ledgers, or generated results here. Do not use test data for model selection or claim a single-seed win; serious ER claims require speaker-independent LOSO.
- Provider mechanics, worker/job lifecycle, storage and credentials belong to standalone `Ke-vin-S/wavcse-infra`; this repo has no embedded infra package.
- Preserve existing public model behavior and tests. Run `make check` before committing. Do not stage untracked `docs/`, `mlruns/`, or generated experiment outputs.
