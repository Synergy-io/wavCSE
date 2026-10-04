# IN-0001 — infrastructure validation scope

`IN-0001` is an **execution scope of kind `infrastructure_validation`**. It is
**not** a research Study, it is **not** registered in `STUDIES.jsonl`, and it
confers **no** scientific authority — on itself, on `DG-0008`, or on any other
Study. No result it produces may ever be cited as evidence about the science.

It exists to exercise the real, integrated Colab execution path once:

```
improvements.compute
  -> external wavcse-infra (canonical checkout, explicit binding)
  -> provider=colab, one ephemeral T4
  -> bootstrap / readiness
  -> one trivial recorded exact-commit job
  -> one tiny deterministic artifact
  -> normal S3 persistence and read-back verification
  -> terminal release
```

## What this scope must never touch

No `DG-0008` data, no embeddings (none of the 15 digest-pinned objects), no
training of any kind, no `DG-0008` benchmark, no Speech Commands / Voxceleb /
IEMOCAP corpus, and no MLflow run. The job is
`improvements.compute.infra_probe`, which writes one small JSON record and
nothing else; it declares no runtime secret, so no research credential reaches
the worker.

## The authorization is the human's

`authorizations/IN-0001.yaml` is the only authority to spend here, and no agent
may create, renew, widen or lift it. `smoke_plan.json` is inert without it: the
backend fails closed with `AuthorizationError` for a scope that has no
committed envelope, and refuses a plan whose scope kind does not match the
envelope's.

## Validation performed (zero-cost, no provider call)

- `improvements.compute` builds the job spec for `smoke`, and the result is
  accepted by the canonical `wavcse-infra` `JobSpec` model at the bound commit:
  `setup: None`, `environment_secrets: []`, one declared output, metadata
  `scope=IN-0001` / `scope_kind=infrastructure_validation`.
- The same check on a Study plan (`TR-0007`) is unchanged: the wrapper command,
  `setup: ["uv","sync","--locked"]`, both MLflow secrets and the `MANIFEST.json`
  output are all still what they were.
- MLflow is **not** required for this smoke. The control plane resolves only the
  secrets a spec declares, and `improvements.compute.infra_probe` declares none;
  demanding MLflow here would put a research credential on a job that trains
  nothing and reaches no tracker.

## Running it (only after the envelope is committed)

The control plane must be bound **explicitly** — a merely discovered one is
refused for provider-mutating steps, because this repository's embedded
`infra/` subsystem otherwise wins resolution and cannot serve a job:

```bash
export WAVCSE_INFRA_CHECKOUT=~/projects/wavcse-infra
uv run --locked python -m improvements.compute resolve --json   # resolved_by: environment

PLAN=improvements/taskrelation/research/execution/IN-0001/smoke_plan.json
uv run --locked python -m improvements.compute status       --scope IN-0001 --json
uv run --locked python -m improvements.compute plan         --scope IN-0001 --plan "$PLAN" --stage smoke
uv run --locked python -m improvements.compute worker-ensure --scope IN-0001 --plan "$PLAN"
uv run --locked python -m improvements.compute advance      --scope IN-0001 --plan "$PLAN" --stage smoke
uv run --locked python -m improvements.compute advance      --scope IN-0001 --plan "$PLAN" --stage smoke   # until the job is terminal
uv run --locked python -m improvements.compute collect      --scope IN-0001 --plan "$PLAN" --stage smoke
uv run --locked python -m improvements.compute finish       --scope IN-0001 --plan "$PLAN"
```

`finish` releases the Colab session, which is terminal for Colab. If a step
fails, the lease deadline and the crash-independent reaper still stop the
session; cleanup is never blocked by a resolution-policy guard.
