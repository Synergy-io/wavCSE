# Compute backend

The seam between wavCSE's research orchestration and the infrastructure control
plane. It exists so that a research cycle can acquire authorized compute,
submit exact-commit jobs, verify their outputs and stop paying for them without
a human changing directory — and without wavCSE learning anything about
providers, SSH, S3 or job state beyond the control plane's CLI contract.

```
/wav-cycle  ->  improvements.compute  ->  (subprocess, --json)  ->  infra CLI
```

## What it does not do

- It never imports the infrastructure package as a library, and never calls a
  provider or an AWS API directly.
- It never implements worker lifecycle, storage transfer, caching, or job state;
  it drives the control plane's CLI and reads its JSON.
- It never moves artifact bytes. It declares required inputs (digest-verified)
  and required outputs; the control plane materializes and persists them.
- It never writes to Git. Research intent is committed by the cycle; runtime
  facts stay under the controller's state directory.

## Verbs

```
python -m improvements.compute status          --scope TR-0007 [--json]
python -m improvements.compute resolve         [--json]
python -m improvements.compute envelope-check  --scope TR-0007 --action submit-job
python -m improvements.compute preflight       --plan P [--commit SHA] [--json]
python -m improvements.compute sweep           --scope TR-0007 [--execute]
python -m improvements.compute reap            [--scope TR-0007] [--execute]
python -m improvements.compute reap-install    [--print|--install [--enable] [--user]]
python -m improvements.compute plan            --scope TR-0007 --plan P --stage screen
python -m improvements.compute worker-ensure   --scope TR-0007 --plan P
python -m improvements.compute advance         --scope TR-0007 --plan P --stage screen [--dry-run]
python -m improvements.compute collect         --scope TR-0007 --plan P --stage screen
python -m improvements.compute finish          --scope TR-0007 --plan P
python -m improvements.compute stop            --scope TR-0007 [--destroy]
```

Exit codes: `0` proceed, `2` usage or configuration failure, `3` refused by policy.
`status`, `resolve`, `envelope-check`, `preflight`, `plan`, `reap` without
`--execute`, and `--dry-run` are read-only and never write runtime state.

## Where the control plane is found

Resolution order, first validated candidate wins: an explicit subsystem
(`WAVCSE_INFRA_CHECKOUT`) or CLI (`WAVCSE_INFRA_CLI`), then the canonical
`infra/` directory in this repository, then an `infra` executable on `PATH`.
A legacy sibling checkout remains a temporary controller-cutover fallback only.
When nothing validates, compute steps stay blocked with an explicit reason —
never silently skipped, never substituted.

## Runtime state (never committed)

Everything under `~/.local/state/wavcse-research/` (override with
`WAVCSE_RESEARCH_STATE` for tests):

| Path | Contents |
|---|---|
| `envelopes/<SCOPE>.json` | the digest each run acted under, with history |
| `runs/<SCOPE>.json` | per-stage job entries: identity, attempts, state, verification |
| `leases.json` | creation intents and worker leases, including deadlines and the last observed billing facts |
| `specs/` | the exact job specs submitted, for auditing a run |
| `reaper/last-run.json` | what the last executing reap pass reconciled and stopped |
| `compute-events.jsonl` | append-only audit trail of every action and refusal |

Worker discovery uses a scope prefix plus a unique create-intent nonce. A
matching name alone is insufficient to start or destroy a worker: ARC also
requires its recorded creation lease.

## Accounting

Spend is derived, never self-reported: each scope worker's provider-reported
hourly price and observed lifetime are summed from `infra worker list --read-only --json`,
and a lease that this backend closed contributes its frozen cost. Every figure
is labelled an estimate. An unknown price or an unknown creation time makes the
total unbounded, and unbounded totals fail closed for new spend.

## Safety properties

- A create intent is persisted **before** the billable request; an ambiguous
  outcome is reconciled by the generated name, never re-requested.
- `--max-price` carries the envelope's hourly ceiling into the control plane's
  own guard, and the observed price is re-checked afterwards.
- A job is submitted only when nothing else for that scope is in flight, and is
  keyed by a deterministic identity (scope, name, commit, arm, seed) so a lost
  acknowledgement is adopted rather than duplicated.
- A recorded job is submitted only from a commit-clean tree.
- Retry budgets and attempt counts live in the run ledger, so a restart cannot
  reset them.
- The sweep is dry-run by default, never destroys, never touches a network
  volume, and never touches a worker it cannot attribute to a scope.
- A vanished worker's cost is finalized from its last durable observation as a provable
  upper bound, never guessed and never undercounted; when no bound is provable the scope
  stays fail-closed instead of continuing on an unknown total.
- Failure classes, retry bounds and the "OOM never changes the science" rule are
  implemented in `failures.py` and stated in `.agents/policies/autonomy.md`.

## Crash independence

A deadline in a lease is a record, not enforcement. `improvements/compute/reaper.py`
is the enforcement: an ordinary command, with no dependency on the session, the
research runner or the agent, that reads the same authoritative lease document and drives
the same stop path. It reconciles before it acts, stops but never destroys, never touches
a resource it cannot attribute to a scope lease, never plans a network volume, takes the
host-level controller guard and each scope lock (non-blocking, so it cannot interleave
with a live `/wav-cycle`), and records what it did. `reap` is a dry run unless
`--execute` is passed.

`reap-install` renders, and can install, the controller timer that runs it — a systemd
service plus timer with `Persistent=true`, so a controller that was off still reconciles
on the way back up. Rendering is pure and installing is idempotent; installing never
enables anything, and `--enable` is a separate explicit step. The default action prints
the units, so nothing in this path can start an unattended reaper by accident.

## One controller per host

ARC's locks are per state root, so nothing inside one root can notice a second root being
mutated at the same time — and two controllers over the same scopes is how a paid resource
is created twice. The invariant is therefore checked *outside* any root: every mutating
verb takes an advisory lock at a fixed host path
(`~/.local/state/wavcse-arc/controller.lock`, override `WAVCSE_ARC_CONTROLLER_LOCK`),
records the state root it is using there, and refuses a second mutating process with the
holder's root named. The lock lives in the kernel, so a killed controller releases it and
never leaves a permanent block; read-only verbs take neither lock, so a status check
always runs.

## Exact commit availability

A job spec pins a full commit and the worker checks it out of the anonymous remote the
spec names. A commit that exists only on this controller is a commit no worker can check
out, so `preflight` — and `worker-ensure`, before any billable request — asks the remote
itself: `git ls-remote` first, then a shallow blobless fetch of that exact object, then a
treeless fetch of its heads with a local object check, which is the same fallback the
worker runner uses. It never pushes, never mutates the remote, and never substitutes a
different commit; an unavailable commit is reported as an actionable publication
requirement, and an unreadable remote fails closed as infrastructure.

## Consuming the verified inputs

Infra verifies and materializes declared inputs, but places each one at a declared
*workspace file*; the loader wants an unpacked tree at `paths.root_emb_path`. A plan
therefore declares `embedding_layout`, the job wrapper turns those verified artifacts into
the loader's layout and names it explicitly, and the loader refuses any root that does not
carry the layout marker its preparer wrote.

## Semantic evidence validation

Stored bytes with a matching digest are necessary and not sufficient: a truncated
manifest, a neighbouring seed's metrics file, or a plausible `NaN` all survive a digest
check. Before an entry becomes COLLECTED, `improvements/compute/evidence.py` reads the
stored objects back through `storage read` (bound to the digest the control plane already
verified) and validates the manifest's identity against the job, its staged digests
against the stored objects, the required inputs' materialization, and the metrics against
the protocol's own task vocabulary and value domains. A structurally invalid result fails
as `EVIDENCE_INVALID` and is never retried; a scientifically negative but structurally
valid one is collected like any other.

## Study compute plan

Each study that will run on compute carries `studies/<ID>/compute/plan.json`:
the study and stage seeds, the arms with their argv, the required inputs file
and outputs, the worker request, the timeout, and — when the arm reads embeddings —
the `embedding_layout` that turns the declared inputs into the loader's tree. It is
committed with the study, so the job spec is derived from committed state and is
byte-identical when regenerated.

### Declared inputs reach the loader

`embedding_layout` names, for each dataset the protocol loads, the declared artifact that
carries it — `input` for a dataset that fits one stored object, `inputs` for a set the
artifact pipeline had to shard, because a single stored object cannot exceed the
provider's single-PUT ceiling:

```json
"embedding_layout": {
  "root": "embedding",
  "datasets": [
    {"dataset": "speechcommand",
     "inputs": ["embeddings/v1/speechcommand-wavlm-large-mean/training-000.tar",
                "embeddings/v1/speechcommand-wavlm-large-mean/training-001.tar",
                "embeddings/v1/speechcommand-wavlm-large-mean/training-002.tar",
                "embeddings/v1/speechcommand-wavlm-large-mean/validation-000.tar",
                "embeddings/v1/speechcommand-wavlm-large-mean/testing-000.tar"]},
    {"dataset": "voxceleb",      "input": "embeddings/v1/voxceleb-wavlm-large-mean/train-000.tar"},
    {"dataset": "iemocap",       "input": "embeddings/v1/iemocap-wavlm-large-mean.tar"}
  ]
}
```

`improvements/compute/embedding_layout.py`, run by `worker_stage` inside the checkout at
the job's exact commit, then does the following before training starts:

1. derives the loader-relative path from the **arm's own config** (`upstream.model_type`,
   `pooling.frame_pooling_type` / `frame_pooling_param`) and the dataset names from
   **`downstream`'s own** `TaskDatasetMapping`, so the prepared layout cannot drift from
   what the loader computes;
2. requires the layout to cover exactly the datasets the plan's task set loads — neither
   fewer nor extra;
3. re-hashes every materialized input against the plan's declared digest and refuses a
   mismatch, an absent file, or a size disagreement;
4. extracts each plain TAR under `<root>/<model_type>/<frame_pool_id>/`, accepting only
   regular members — never a traversal, a link, or a device. The canonical archives are
   TARs of the dataset's *contents*, named by the writer-relative path the loader
   resolves (`Session1/sentences/wav/…`, `speech_commands_v0.01/bed/…`), so they are
   extracted at `<root>/<model_type>/<frame_pool_id>/<dataset>/`; an archive that instead
   carries the dataset directory itself as every member's first path segment describes
   the same tree and is extracted one level up. A member naming another dataset the
   protocol loads, or an archive mixing the two shapes, is refused. A tree left by an
   earlier attempt is never merged, and two declared shards that would contribute the
   same member are refused, because which bytes won would then depend on extraction
   order rather than on the verified input;
5. writes `.arc_embedding_layout.json` under the root, naming every artifact and digest
   the root was built from, the extracted directories and the job identity;
6. exports `WAVCSE_ROOT_EMB_PATH=<job>/embedding`, which `improvements/embedding_root.py`
   honours **only** when the marker validates. There is no fallback: an unverifiable
   override is refused rather than quietly replaced by `~/embedding`, and with no override
   at all the configured value is returned unchanged, so controller behaviour does not
   move.

The wrapper records the resulting mapping in the job's `MANIFEST.json` as
`embedding_layout`, so the evidence states which verified artifacts the run's
loader-visible root was built from, and `evidence.py` checks that record before the result
is collected.
