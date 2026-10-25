# improvements/sweep

Declare a set of layer-combination trainings, then run as many at once as the
machine can actually hold — admitted and drained by live VRAM/RAM/CPU/disk
pressure rather than by a fixed `-P N`.

## Why this exists, and what it is not

The layer axis of the wavCSE representation is a **downstream load-time slice**
(`embedding[transformer_layer_array, :]` in
`downstream/dataset/preprocess_embedding.py`), so one layer-combination costs one
training process and **no upstream re-extraction**. A sweep is therefore cheap
to declare and expensive only in wall clock — exactly the case where running
several at once is worth having.

This local sweep runs on this machine or a declared, operator-owned
`static_ssh` host. Remote transport is exclusively the standalone
`infra worker` CLI; it does not accept a raw SSH alias, RunPod Pod or Colab
session. Paid GPU runs use the private `wavcse_research.execution` envelope and
recorded exact-commit jobs, never this detached-supervisor path.

Tradeoffs relative to the private execution backend:

| given up | consequence | mitigation in this package |
| --- | --- | --- |
| paid envelope / budget | not applicable to a declared non-billable static host; this path does not grant spend | `policy.max_wall_seconds` + a `DRAIN` flag bound the local supervisor |
| `jobspec` + evidence validator | no control-plane re-derivation of what ran | per-run identities recorded; checkpoints and metric files hashed into the ledger |
| `reap` timer | no controller-side deadline enforcement | the supervisor's own loop and drain flag |
| exact-commit job pinning | nothing pins the observed code | the sweep refuses a commit whose *own inputs* are uncommitted: a modified tracked file, or an untracked file inside the study directory (an uncommitted config). Untracked files elsewhere in the tree do not block a launch — a colleague's scratch file must not teach an operator to reach for `--allow-dirty` |

The remote host is declared under `[[static.workers]]` in standalone infra.
It is never provisioned, priced or destroyed by the sweep.

## Public executable layout

```text
improvements/taskrelation/experiments/<STUDY>/
  sweep.json                           # combos, stages, seeds and local policy
  configs/<stage>/<combo-id>.yml       # generated, committed before launch
  outputs/<combo-id>/{results,checkpoints}/  # ignored runtime data
  sweep_state/                         # ignored local ledger and logs
```

Study plans, decisions and analyses stay in private `wavcse-research`.

## Command flow

```bash
SPEC=improvements/taskrelation/experiments/<STUDY>/sweep.json

# 1. read-only: what would run, and the policy (works with the host unset)
python -m improvements.sweep plan    --spec $SPEC
python -m improvements.sweep plan    --spec $SPEC --json

# 2. write the per-combo configs, then COMMIT them (refuses a dirty tree)
python -m improvements.sweep configs --spec $SPEC

# 3. set WAVCSE_INFRA_CHECKOUT; use a declared static worker ID and a checkout
#    beneath that worker's configured root. This path cannot launch paid RunPod.
python -m improvements.sweep push  --spec $SPEC --host <static-worker-id> --checkout <worker-repo-root>
python -m improvements.sweep start --spec $SPEC --stage screen --host <static-worker-id> --checkout <worker-repo-root>

# 4. watch, inspect, stop, collect
python -m improvements.sweep status  --spec $SPEC [--json]
python -m improvements.sweep logs    --spec $SPEC --run <job-key>
python -m improvements.sweep stop    --spec $SPEC          # drain: nothing new starts
python -m improvements.sweep pull    --spec $SPEC
python -m improvements.sweep report  --spec $SPEC          # from MLflow, ranked on validation
```

Without a `remote.host`, `start` runs the supervisor **on this machine** — the
same code path, which is how the scheduling logic is exercised in tests.

### Remote runs: paths and overrides

Pass `--host <declared-static-worker-id>` and
`--checkout <checkout-beneath-the-worker-root>` to remote
`push`/`pull`/`start`/`status`/`logs`/`stop`. Bind
`WAVCSE_INFRA_CHECKOUT` to standalone `wavcse-infra`; an optional
`WAVCSE_INFRA_CLI` must resolve to that same checkout. Neither a sibling nor
an executable on PATH is discovered. The worker ID comes from its
`[[static.workers]]` declaration, not an SSH alias, and the infra CLI refuses
paths outside its declared root.

Remote paths are derived from that worker's checkout
(`improvements/sweep/manifest.py:remote_paths`), never from the local Study
directory. Paid RunPod/Colab worker IDs are refused before a sweep command.

`push` and `start` compare the static host's `git rev-parse HEAD` against
the committed public checkout and refuse a mismatch. `start` can source a
worker-local `--env-file <absolute-path>` before launching the supervisor;
infra constructs and executes that detached command without copying
credential values to this machine's command line.

## How "as many as fit" is decided

`supervisor` probes the host every `policy.probe_interval_s`, then asks
`scheduler.admit(snapshot, running, policy)`. The decision is a **pure function**
of the snapshot, so every rule below is unit-tested with fake snapshots and no
GPU. Admission is held when any of these fails:

* a GPU is required (`policy.allow_cpu: false`) and none is usable;
* `running` has reached the derived capacity: `min(free VRAM / vram_per_run,
  (free RAM − ram_reserve) / ram_per_run, cores / cpu_cores_per_run)` and a disk
  floor — the binding resource is named in the log line;
* `policy.max_concurrent` is an integer and `running` has reached it;
* GPU utilisation exceeds `max_gpu_util_pct` (no admitting into a saturated
  device) or 1-minute load exceeds `max_load_fraction` of the cores.

`policy.ceiling` bounds an automatic cap without turning it into a fixed number.
Nothing already running is ever killed: pressure stops admissions, it does not
interrupt a half-trained run.

The historical two-job concurrency limit was overridden for a specific
research sweep with a resource-derived cap. A new sweep must record its own
researcher decision in `policy.concurrency_basis`; the old override is not
standing authorization to launch more jobs.

## Restart safety and identity

* Identity is deterministic: `sha256(study | combo | seed | commit)[:12]`. The
  same sweep restarted cannot silently launch a second copy of a run.
* Every state change is appended to `sweep_state/runs.jsonl` before the next
  one happens; a `RUNNING` record whose recorded PID is gone is marked
  `ORPHANED` and requeued. A run that exits 0 without writing a run identity is
  recorded as **FAILED**, not as success.
* The ledger's `Succeeded` count, not the existence of an output directory, is
  what says a run finished.

## Where a run shows up in MLflow / DagsHub

One experiment per sweep (`experiment_name` in the spec), one run per combo and
seed, grouped by:

| what | where |
| --- | --- |
| `research.group`, `research.k`, `research.layers`, `research.combo` | params (the whole config is flattened into params) |
| `sweep_group`, `sweep_k`, `sweep_combo`, `sweep_layers` | tags, applied by the supervisor at completion (`--no-annotate` to skip) |
| group / k / combo, and the **full layer list** (kept and dropped) | the run note (`mlflow.note.content`) — the DagsHub details field |
| run name | `<study>__<stage>__<combo>__<task>__smp<L>L__s<seed>` |

The run name comes from `research.run_name`, a `str.format` template supported
additively by `improvements/mlflow_utils.build_research_run_name`. A config
without that key produces exactly the name it always did
(`test_research_run_name.py` pins both behaviours).

## Tests

```bash
uv run --locked python -m unittest discover \
  -s improvements/taskrelation/tests \
  -t . -p "test_layer_sweep_*.py"
uv run --locked python -m unittest discover \
  -s improvements/taskrelation/tests \
  -t . -p "test_research_run_name.py"
```

The layer grammar the spec accepts is asserted for **parity with the frozen
pipeline's parser** (`downstream/utils/parse_transformer_layers.py`) on a table
of inputs, and the generated configs are asserted to differ from the template in
the sweep's own keys and nothing else.
