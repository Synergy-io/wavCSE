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

This package deliberately does **not** go through `improvements.compute`. That
backend is a paid-compute contract: one authorized envelope, one deterministic
job per (arm, seed), one job in flight per scope, evidence staged and verified
by its own validator. Its `advance` transition submits nothing while a job of
the scope is in flight, so it cannot express "twenty trainings at once on one
hired device" — which is the whole point here. What that trade costs, stated
rather than hidden:

| given up | consequence | mitigation in this package |
| --- | --- | --- |
| envelope / budget | nothing bounds spend but the clock | `policy.max_wall_seconds` + a `DRAIN` flag; the supervisor reports elapsed time |
| `jobspec` + evidence validator | no control-plane re-derivation of what ran | per-run identities recorded; checkpoints and metric files hashed into the ledger |
| `reap` timer | no controller-side deadline enforcement | the supervisor's own loop and drain flag |
| exact-commit job pinning | nothing pins the observed code | the sweep refuses a commit whose *own inputs* are uncommitted: a modified tracked file, or an untracked file inside the study directory (an uncommitted config). Untracked files elsewhere in the tree do not block a launch — a colleague's scratch file must not teach an operator to reach for `--allow-dirty` |

The worker is assumed **already hired**: this drives an existing pod over ssh and
does not provision, price or destroy one.

## Layout it creates, per study

```
improvements/taskrelation/research/studies/<STUDY>/
  PLAN.md            # hypothesis, competing explanation, decision rule
  sweep.json         # combos (layers + group + k), stages, seeds, policy, remote
  configs/<stage>/<combo-id>.yml   # generated, COMMITTED before launch
  outputs/<combo-id>/{results,checkpoints}/   # gitignored
  sweep_state/                              # gitignored: runs.jsonl, logs/, identity/, SWEEP.json, DRAIN
```

## Command flow

```bash
SPEC=improvements/taskrelation/research/studies/<STUDY>/sweep.json

# 1. read-only: what would run, and the policy (works with the host unset)
python -m improvements.sweep plan    --spec $SPEC
python -m improvements.sweep plan    --spec $SPEC --json

# 2. write the per-combo configs, then COMMIT them (refuses a dirty tree)
python -m improvements.sweep configs --spec $SPEC

# 3. get them onto the pod and start the supervisor there
python -m improvements.sweep push    --spec $SPEC
python -m improvements.sweep start   --spec $SPEC --stage screen

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

Pass `--host <ssh-target>` (and `--checkout <pod repo root>` if the pod's
checkout differs from the spec's) to `push`/`pull`/`start`/`status`/`logs`/`stop`.
Both are command-line overrides on purpose: the ssh target and the pod's paths
are *environment*, not science, and committing them would dirty a tracked file —
which the launch check correctly refuses.

Every path a remote verb touches is derived from the **pod's** checkout
(`improvements/sweep/manifest.py:remote_paths`), never from this machine's study
directory: the controller's `sweep_state` path does not exist on a pod whose
checkout lives somewhere else, and a supervisor started with it would create a
stray tree. Use an `~/.ssh/config` alias rather than `host:port`, because the
target is passed to `ssh` as a single argument.

`push` and `start` also compare the pod's `git rev-parse HEAD` against this
checkout's and refuse to proceed when they differ — the pod's checkout is what
actually executes, so a stale HEAD there would attribute runs to a commit that
never ran.

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

**Deviation from `AGENTS.md` invariant 16** (at most two simultaneous GPU
training jobs) is a researcher decision for a sweep that wants an
automatic cap; it must be recorded in `policy.concurrency_basis`, not left
implicit.

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
  -s improvements/taskrelation/research/tests \
  -t improvements/taskrelation/research/tests -p "test_layer_sweep_*.py"
uv run --locked python -m unittest discover \
  -s improvements/taskrelation/research/tests \
  -t improvements/taskrelation/research/tests -p "test_research_run_name.py"
```

The layer grammar the spec accepts is asserted for **parity with the frozen
pipeline's parser** (`downstream/utils/parse_transformer_layers.py`) on a table
of inputs, and the generated configs are asserted to differ from the template in
the sweep's own keys and nothing else.
