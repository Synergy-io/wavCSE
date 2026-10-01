# Stage 1 — upstream embedding extraction (`upstream/`)

Turns raw audio into the per-utterance `.pt` embedding files that `downstream/` and
`improvements/` train on. Nothing here trains: the encoder is frozen and the output is
plain files on disk, so a change to the encoder, pooling, or preprocessing requires
re-extraction before stage 2 can see it.

## Running it

`upstream/` must be the working directory — the modules import as `dataset.*`, `model.*`,
`utils.*`, and `model/wavlm/WavLM.py` imports `model.wavlm.modules`:

```bash
cd upstream
python main.py --dataset_name <dataset> --config configs/<config>.yml [--device_index N]
```

Supported `--dataset_name` values: `speechcommand`, `voxceleb`, `iemocap`,
`fluentspeechcommand`. Chinese remaining undocumented behaviour lives in
`dataset/build_embedding.py` (split construction) and `dataset/extract_embedding.py`
(per-dataset loading, trimming, label mapping).

## What it computes

1. `model/load_upstream.py` → `model/wavlm/wavlm_loader.py` loads a **fairseq-format**
   WavLM checkpoint (`torch.load` → `WavLMConfig(ckpt["cfg"])` → `WavLM` →
   `load_state_dict(ckpt["model"])`), moves it to the device and calls `.eval()`.
   WavLM code is vendored in this directory; only the checkpoint is external.
2. Each utterance is loaded at 16 kHz, normalised with
   `F.layer_norm(waveform, waveform.shape)` (matches the checkpoint's `cfg.normalize`),
   and optionally trimmed to a 60 s maximum-energy window when the file exceeds 1.7 MB.
3. `extract_features(..., output_layer=cfg.encoder_layers, ret_layer_results=True)`
   returns the encoder input plus every transformer layer — **25** tensors for
   WavLM Large (`cfg.encoder_layers == 24`).
4. They are concatenated on a new leading axis, `(25, T, 1024)`, and the configured
   frame pooling collapses the time axis only: mean pooling gives
   `x.mean(dim=1)` → **`(25, 1024)` float32** per utterance.
5. `utils/embedding_io.py` writes one `.pt` per utterance plus a CSV index.

Output layout (all three splits appended to one CSV):

```text
<root_emb_path>/<model_type>/<pooling_id>/<dataset_name>/
├── <dataset>_<model>_<pooling_id>.csv        # wavpath,label,file_size
└── <wavpath with .wav -> _<model>_<pooling_id>.pt>
```

`pooling_id` comes from `utils/pooling_id.make_pooling_id` (`mean`, `max`, `mix0d5`, …).
Labels come from `utils/constant_mapping.py`.

## Branch `feature/reproducible-embedding-generation`

This branch makes one run of stage 1 expressible as a recorded, reproducible job. It
changes **no** scientific behaviour: no preprocessing, resampling, layer selection,
checkpoint, pooling maths, precision, filtering, or split logic is touched.

| Change | Why |
| --- | --- |
| `main.py`: optional `--root_data_path`, `--root_emb_path`, `--checkpoint`, `--limit` | Let a run point at freshly materialised data/checkpoints on an ephemeral worker without editing a committed config file. CLI wins over config; defaults are unchanged when the flags are omitted. |
| `configs/extract_embedding_mean.yml` (new) | Pins the scientific configuration (`wavlm_large`, `frame_pooling_type: mean`, no parameter, whole dataset) as a reviewable file, instead of relying on whatever the default config happens to say. `extract_embedding.yml` is untouched. |
| `validate_embeddings.py` (new) | Read-only integrity check for a generated embedding directory + CSV: file count, loadability, shape/dtype consistency, NaN/Inf, empty/zero-byte files, duplicate rows, CSV file-size agreement, orphan/missing files, optional source↔embedding mapping. Exit code is non-zero on failure. |

Reproducible invocation:

```bash
cd upstream
python main.py --dataset_name fluentspeechcommand \
  --config configs/extract_embedding_mean.yml \
  --root_data_path  <raw datasets root> \
  --root_emb_path   <embeddings output root> \
  --checkpoint      <WavLM-Large.pt> \
  --limit           <N>      # omit for the whole dataset
python validate_embeddings.py --root <embeddings output root>/wavlm_large/mean/fluentspeechcommand \
  --expected-layers 25 --expected-dim 1024
```

## Verified artifact: Fluent Speech Commands, WavLM-Large + mean

Generated 2026-09-28 by a recorded job on a disposable RunPod RTX A4500 worker, from
unchanged code at commit `0ac8632aadf7ea87bfa9108d40f22d702eb7f0d4`, with the mean
configuration above:

| Property | Value |
| --- | --- |
| Utterances | 30,043 (train 23,132 / valid 3,118 / test 3,793) |
| Per-utterance tensor | `(25, 1024)` float32, 104,427 bytes on GPU |
| Total embeddings | 3,137,300,361 bytes (2,992 MiB) |
| Archive | plain tar, members `wavs/…` + `fluentspeechcommand_wavlm_large_mean.csv` |
| Archive size / SHA-256 | 3,186,790,400 bytes / `2092efb657f2bcf3c889ea3371a1e4aed2a30b882885a6358fa3fea77907eb3d` |
| S3 object | `embeddings/v1/fluentspeechcommand-wavlm-large-mean.tar` (+ `.manifest.json`) |
| Encoder checkpoint | WavLM Large, 1,261,965,425 bytes, SHA-256 `6fb4b3c3e6aa567f0a997b30855859cb81528ee8078802af439f7b2da0bf100f` |
| Dataset source | Fluent Speech Commands, 1,545,730,387 bytes, SHA-256 `eb7069f505da04e248eb214d53d8c799bcd69f74fe1e4e27d7f61b65e23cdb1f` |
| Environment | `uv.lock` (Python 3.9, torch 2.7.1, torchaudio 2.7.1) |

Evidence: every source archive member was CRC-verified and every CSV row matched 1:1 to a
wav before generation; a 6-utterance smoke run produced the expected `(25, 1024)` float32
tensors; the full output was validated twice (on the worker and again after extracting the
published archive on the controller) with all 30,043 files loadable, uniform shape/dtype,
no NaN/Inf, and the per-intent label distribution identical to an oracle computed
independently from the source corpus; the stored S3 ETag equals the archive's MD5.

Not claimed: the GPU model behind the original baseline embeddings (not recorded anywhere
in this repository), and numerical equivalence to any previously published embeddings —
no such comparison has been run.

## Known gaps in this directory (report, do not silently "fix")

- `configs/extract_embedding.yml` ships `dataset.limit: 1`, so running it as committed
  produces one sample per split — a smoke run, not a dataset. It cannot be the config that
  produced the published embedding sets.
- `readme.md` (repository root) states the WavLM checkpoint "is automatically downloaded";
  it is not. `load_upstream.py` raises `FileNotFoundError` for a missing checkpoint, so the
  checkpoint must be provisioned deliberately (and its identity recorded).
- `SPEECHCOMMANDS`' filesize lookup builds `<path>/<dataset>/<relative path>`, which never
  exists, so the 1.7 MB trim threshold is effectively inert for that dataset. Harmless
  there (clips are ~30 KB), but the branch does not change it.
- VoxCeleb/IEMOCAP/Fluent Speech Commands leave `label_index` unbound when a label is
  absent from the mapping (`UnboundLocalError`); Speech Commands falls back to index 10.
  Changing this would change label semantics, so it is left alone.
- IEMOCAP filtering assumes `mapping[stem]["path"].split("/")[3]`, i.e. a fixed relative
  path depth in the unpacked corpus.
- `process_and_write_data` calls `torch.cuda.empty_cache()` once per sample — pure
  overhead at 30 k samples, but it is execution behaviour, not semantics.

## Operator tooling

The deterministic packagers live in `tools/` and run as the packaging job's own input:

| Script | Dataset |
| --- | --- |
| `tools/package-iemocap-tar.py` | IEMOCAP (one archive, no sharding) |
| `tools/package-speechcommand-shards.py` | Speech Commands (per-split shards) |
| `tools/package-voxceleb-shards.py` | VoxCeleb1 (per-split shards) |

Each takes `--pooling <pooling>` (default `mean`) and derives the
`wavlm_large/<pooling>/` embedding directory, the `<dataset>_wavlm_large_<pooling>.csv`
index, the `.pt` filename suffix and the default archive name from it, so one packager
serves every pooling strategy. They are stdlib-only and depend on nothing in
`wavcse-infra`; a run pins the exact revision by checking out the job's commit.

The recorded-job specs, the S3 manifest builder, and the archive acceptance test remain on
the controller (`~/jobs/*.json`, `~/jobs/make-*.py`, `~/jobs/tools/*.sh`). They depend on
`wavcse-infra` (its storage namespace and manifest model), so they belong in that
repository rather than here.
