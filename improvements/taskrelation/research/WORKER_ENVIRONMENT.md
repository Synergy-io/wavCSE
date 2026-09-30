# Worker environment for a wavCSE training run — what it needs, and how to rebuild it

**Status:** operational knowledge, kept because the cache volume it describes has been deprovisioned
(2026-09-30, after the TR-0013 screen). No canonical bytes were lost — S3 holds the only
authoritative copy of every embedding artifact — but the *raw-dataset farm* that lived on that
volume is gone with it, and this file is now the record of how to rebuild the environment from
nothing. Read it before the next paid run.

Everything here was verified on a real worker during TR-0007 (`NVIDIA RTX A4500`, EU-RO-1) and
TR-0013 (`NVIDIA L4`, EU-RO-1); the failures that taught each rule are named so the rules are not
re-derived the expensive way.

## 1. What the plan declares, and what it does not

The compute plan (`studies/<ID>/compute/{plan.json,inputs.json}`) declares:

* the 15 canonical **embedding** objects (SHA-256 each) and an `embedding_layout` that lets
  `improvements/compute/embedding_layout.py` extract them into the loader's root
  (`<job>/embedding/wavlm_large/mean/<dataset>/…`) and export `WAVCSE_ROOT_EMB_PATH`;

It does **not** declare, and cannot materialize:

* the **raw corpora** the loader's dataset classes read from `paths.root_data_path` (the configs'
  `~/voice_dataset`). Nothing in the plan provisions them, and no controller-side check notices they
  are missing — the job dies in `LoadEmbedding` instead (DG-0007 lost an attempt to exactly this,
  TR-0007's NOTE records the same gap).

So every fresh worker needs the raw tree built once, before the first job, and verified by counts.
This is environment preparation, not a scientific parameter: no config field changes.

## 2. The raw tree the loader actually needs

The dataset classes subclass torchaudio readers (`downstream/dataset/preprocess_embedding.py`), so
their *resolution rules* decide the paths:

| Task dataset | torchaudio base | What the root must contain |
|---|---|---|
| Speech Commands (`speechcommand`) | `SPEECHCOMMANDS(root, url="speech_commands_v0.01", folder_in_archive="SpeechCommands")` | `<root>/SpeechCommands/speech_commands_v0.01/` = the archive's *contents* (`bed/`, `cat/`, …, `validation_list.txt`, `testing_list.txt`, `_background_noise_`) |
| VoxCeleb1 (`voxceleb`) | `VoxCeleb1Identification(root, subset, meta_url)` | `<root>/wav/` (per-speaker directories) **and** `<root>/iden_split.txt` |
| IEMOCAP (`iemocap`) | `IEMOCAP(root, sessions)` | `<root>/IEMOCAP/Session1…Session5/…` — note the extra `IEMOCAP` level |

With `paths.root_data_path: ~/voice_dataset`, the verified working tree is:

```
~/voice_dataset/
├── speechcommand/SpeechCommands/speech_commands_v0.01 -> <store>/speechcommand
├── voxceleb/wav                                      -> <store>/voxceleb/wav
├── voxceleb/iden_split.txt                           -> <store>/voxceleb/iden_split.txt
└── iemocap                                           -> <store>/iemocap        # contains IEMOCAP/
```

**Verified counts (2026-09-29, the TR-0013 worker):** `64 727` wavs under
`speechcommand/SpeechCommands/speech_commands_v0.01`, `153 516` under `voxceleb/wav`, `10 039` under
`iemocap/Session1…5`. These match TR-0007's recorded counts exactly.

**Two traps, both of which cost attempts:**

1. The symlink must point at the level torchaudio resolves. TR-0013's first attempt linked
   `~/voice_dataset/iemocap` directly at a directory *named* `IEMOCAP`, so torchaudio looked for
   `…/IEMOCAP/IEMOCAP/Session1` and raised `RuntimeError: Dataset not found.` — the link target must
   be the **parent** of `IEMOCAP`.
2. Verify by constructing the loader, not by `ls`:

   ```bash
   cd <checkout> && .venv/bin/python - <<'PY'
   import sys, os; sys.path.insert(0, os.path.join(os.getcwd(), "downstream"))
   from dataset.load_embedding import LoadEmbedding
   from utils.parse_transformer_layers import parse_transformer_layers
   loader = LoadEmbedding(root_emb_path=os.environ["EMB"], root_data_path="/root/voice_dataset",
                          upstream_model_type="wavlm_large", frame_pooling_type="mean",
                          frame_pooling_param=None,
                          transformer_layer_array=parse_transformer_layers("all", "wavlm_large"),
                          device="cpu")
   train, val, test = loader.load_embedding("ks_si_er", subset_percentage=1)
   print("LOADER_OK", len(train), len(val), len(test), len(parse_transformer_layers("all","wavlm_large")))
   PY
   ```

   A passing run prints `LOADER_OK … 25`.

## 3. Where the raw corpora come from

Not recorded in the wavCSE repository: the tree described above was **pre-provisioned on the cache
volume** (likely by building it from the original corpus distributions) and TR-0007/DG-0007/TR-0013
all consumed it read-only. Rebuilding from scratch therefore needs the sources obtained again, which
the project's records do not name:

* Speech Commands v0.01 — public download (the archive whose *contents* become
  `speechcommand/SpeechCommands/speech_commands_v0.01/`).
* VoxCeleb1 — public (the tree whose `wav/` and `iden_split.txt` become `voxceleb/`).
* IEMOCAP — licence-gated (LDC); the sessions must be laid out as `iemocap/IEMOCAP/Session1…5/`.

Flagged rather than invented: if the raw farm is ever needed again, get the sources and lay them out
exactly as §2 describes, then re-verify with the count check and the loader construction above.

## 4. The cache volume (deprovisioned 2026-09-30)

`wavcse-vol-cache-85bcc73be71f` (provider id `9y5t57z98h`), 200 GB, STANDARD, EU-RO-1, created
2026-09-29T04:19Z, mounted at `/workspace/cache` on every worker. Observed contents at
2026-09-30T02:00Z:

```
/workspace/cache/
├── artifacts/        25 G   digest-addressed copies of the declared inputs (the embedding tars)
├── warm-staging/    1.4 G   partial/pre-warmed transfers
├── raw/              76 G   the raw corpora of §2 (speechcommand, voxceleb, iemocap)
├── work/             53 G   job scratch that had been left behind by earlier attempts
├── tools/           1.1 M   helper binaries fetched by bootstrap
├── inputs/  staging/   0    empty
└── cache.json               the cache index (512 B)
```

Rebuild it with the infra CLI, never by hand:

```bash
infra volume datacenters                      # confirm the data center can host the requested tier
infra volume create --data-center EU-RO-1 --size 200 --tier standard --name cache --yes
infra worker create --gpu '<exact-gpu-type-id>' --cloud SECURE --image '<reviewed-image>' \
  --container-disk <gb> --network-volume-id <volume-id> --start-ssh --require-direct-ssh \
  --max-price '<usd-per-hour>'
infra worker bootstrap <worker-id>            # proves the volume mount before any job
```

The volume is a rebuildable cache, not canonical storage: the `artifacts/` entries are re-created on
demand from S3 by the job framework, and `raw/` is the part that needs the corpora of §3. It must be
attached **at Pod creation** and the Pod must be in the volume's data center; the compute backend
does both (`worker.network_volume` in the plan, `resources.network_volume_selector` in the
authorization).

## 5. Container disk: the failure mode that costs the most

Each job's workspace (`/workspace/wavcse-jobs/<job-id>`, on the **container disk**, not the volume)
holds the checkout, the `.venv`, the 20 GB of declared input tars, their ~20 GB extracted copy under
`embedding/`, logs, state and checkpoints — **≈45–50 GB per job**, measured.

TR-0013 lost six attempts to this, in three distinct ways:

1. **Two workspaces at once** (a stale, unreferenced one from an earlier attempt plus the new one)
   filled a 100 GB container disk: the runner's own `os.makedirs`/`mkdir` failed with `ENOSPC`.
   *Rule: exactly one job workspace may exist.*
2. **`rm -rf` of a predecessor's ~50 GB workspace while the next job was starting** — the tree is
   ~200k small files; the two "process no longer running, no outcome" failures both coincided with
   such a deletion, with 500 GB of host RAM free and a clean disk otherwise.
   *Rule: sweep scratch only when nothing is live; never delete during a job.*
3. A **stale workspace the run record no longer references** is invisible to record-driven cleanup.
   *Rule: clean by directory, keeping only the in-flight job's id, not by record.*

With one workspace at a time, a 100 GB container disk is comfortable (`df` ≈50 GB free while a job
runs with the volume attached). The authorization's `container_disk_gb_max` should stay ≥100.

## 6. GPU compatibility and price discipline

The pinned image (`runpod/pytorch:2.4.0-py3.11-cuda12.4.1`) tops out at **`sm_90`**, so Blackwell
parts (`RTX PRO 4000/4500/6000 Blackwell`, `RTX 5090`) cannot run it — DG-0007 recorded that failure
and it is a hard filter before price. Discover at decision time (`infra worker gpu-types
--data-center <dc> --require-direct-ssh`), then choose the cheapest *compatible* offer inside the
data center the network volume lives in; on 2026-09-30 that was `NVIDIA L4` at `$0.49/h` in EU-RO-1
(next: `RTX 4090` `$0.74/h`; `RTX A4500` `$0.25/h` was historical only and not on offer).

## 7. Cost shape of a four-arm screen (measured)

One L4, one worker, four sequential jobs, warm cache: **≈19–22 min per job** including setup and
input materialization; the TR-0013 screen's *science* therefore costs ≈1.5 h ≈ `$0.75`, and the
paid worker existed 3.28 h (`$1.6074`) because of the failures in §5. Prepare the raw tree, keep one
workspace, and the expected cost of the next four-arm screen is the ≈`$0.75–0.90` figure — well
inside a `3.00 USD` envelope.

## 8. Sequence that worked (copy this)

```bash
# controller, after the plan and configs are committed and pushed
uv run python -m improvements.compute preflight   --plan <plan>                 # exact commit reachable
uv run python -m improvements.compute envelope-check --scope <SCOPE> --action create-worker
uv run python -m improvements.compute worker-ensure --scope <SCOPE> --plan <plan>
# on the worker, once: build and verify the raw tree of §2
# then, per arm, sequentially on that one worker:
uv run python -m improvements.compute advance --scope <SCOPE> --plan <plan> --stage screen
uv run python -m improvements.compute collect --scope <SCOPE> --plan <plan> --stage screen   # when terminal
# and when every entry is collected:
uv run python -m improvements.compute finish  --scope <SCOPE> --plan <plan>     # destroy, billing off
```

`finish` refuses to destroy while any entry is not collected; a failed arm therefore blocks it, and
the worker must be stopped explicitly (`infra worker stop <id> --yes`) so a failed run does not keep
billing.
