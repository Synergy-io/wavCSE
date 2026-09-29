---
name: wavcse-embedding-generation
description: Generate and validate wavCSE WavLM embeddings scientifically; use when configuration may force re-extraction, or when checking tensor, on-disk, split and label contracts per dataset.
---

# wavCSE Embedding Generation

Scope: the scientific semantics of upstream embedding extraction for wavCSE - the configuration
that defines a dataset, the tensor and on-disk contracts, per-dataset membership and label
semantics, and the validation that must pass before embeddings are reused or republished.

- Worker lifecycle, bootstrap, job submission, CLI flags: `wavcse-infra-operator`.
- GPU choice, cost, benchmarking, monitoring: `gpu-research-operator`.
- Canonical storage, S3 keys, manifests, transfer, cache, read-back verification:
  `wavcse-artifact-pipeline`.
- Study design, controls, provenance recording: `wavcse-experiment-operator`.

## Reuse before regenerate

Do not regenerate embeddings because a dataset is named. Establish, in order: what the current
research configuration requires; whether a canonical validated artifact already exists; whether
that artifact was produced by the required configuration. Reuse a compatible canonical artifact.

Layer selection and downstream layer pooling happen at load time. The stored tensor keeps the whole
WavLM layer axis, and consumers slice it - see `downstream/utils/parse_transformer_layers.py` and
the `embedding[transformer_layer_array, :]` step in `downstream/dataset/preprocess_embedding.py`.
So an experiment that changes the selected layers or its layer-pooling method reuses one
extraction. The repository relies on this: the all-layers baseline config records that the shared
embedding tree already stores every WavLM-Large layer row per utterance, so switching
`selected_transformer_layers` from a 16-layer subset to `all` needs no re-extraction.

Everything else that defines the tensor - frame pooling, checkpoint, preprocessing, filtering,
label mapping, dataset membership - is fixed at extraction time. Changing any of them requires a
new extraction into a new directory key; it is never patched in place.

## Configuration that defines a dataset

Derive this from the current repository, not from memory, and record it with the artifact:

- exact repository commit;
- dataset name and source identity, including the provenance of a repack or mirror;
- split semantics and filtering;
- loader: `upstream/dataset/build_embedding.py`, `upstream/dataset/extract_embedding.py`;
- `upstream.model_type` and the checkpoint path with its identity;
- frame pooling type and parameter (set in the extraction config);
- preprocessing, sample rate, normalization, precision;
- output tensor contract;
- label mapping: `upstream/utils/constant_mapping.py`.

None of these may be changed for speed, cost or convenience. An operational optimization must
preserve all of them; a change to any of them is a research change needing explicit experimental
treatment and a fresh extraction.

## Preprocessing chain

The current pipeline, in order, for every supported dataset: load the waveform; layer-normalize it
over the whole waveform; run WavLM `extract_features` over all encoder layers with
`ret_layer_results`; transpose and concatenate the layer results on the layer axis; pool over the
time axis with the configured frame pooling; save one tensor per utterance.

- Oversized-source trimming: when the source wav file exceeds 1.7 MB, the waveform is reduced to
  its highest-energy 60 s window (25 ms frames, 10 ms hop); a file at or below that size is used in
  full. The *source wav's* byte size and energy content are therefore scientifically meaningful.
- Fluent Speech Commands additionally forces mono (channel mean) and resamples to 16 kHz.
- The serialized `.pt` file size is incidental; tensor semantics are not.

## Tensor and on-disk contract

Per utterance the stored object is a float32 tensor whose first axis is the WavLM layer axis:
`(25, 1024)` for `wavlm_large`, `(13, 768)` for `wavlm_base`. Confirm the contract against the
current model variant and config rather than enforcing a remembered shape.

Layout under the configured embedding root:

```
<root_emb_path>/<upstream_model_type>/<frame_pool_id>/<dataset_name>/
    <wav relative path, with .wav replaced by _<upstream_model_type>_<frame_pool_id>.pt>
```

- `frame_pool_id` encodes pooling type and parameter (`mean`, `mix0d5`, ...), so a different frame
  pooling writes a different tree instead of overwriting an existing one.
- The sidecar CSV `<dataset_name>_<upstream_model_type>_<frame_pool_id>.csv` has the header
  `wavpath,label,file_size` (a human-readable size string); the train, validation and test writes
  share it, and the first write truncates it.
- That CSV is an informational index, not a membership oracle. The training loader never reads it;
  it re-derives each embedding path from the torchaudio sample's relative wav path. A missing or
  misnamed `.pt` therefore raises at load instead of being silently skipped.
- Config `dataset.limit` truncates each split's write. The shipped config sets it small, so a run
  can exit successfully having written only a few samples per split.

## Per-dataset semantics

### Speech Commands (`speechcommand`)

`SpeechCommands` archive v0.01 with the `SpeechCommands` folder in the archive; the
training/validation/testing subsets come from the archive's own `validation_list.txt` and
`testing_list.txt`. Labels use `LABEL2INDEX_SPEECHCOMMANDv1`: yes 0, no 1, up 2, down 3, left 4,
right 5, on 6, off 7, stop 8, go 9, `_unknown_` 10, `_silence_` 11. A label missing from the
mapping falls back to 10, so an unmapped keyword silently becomes `_unknown_`. Identity is the
relative wav path; keep the version-directory prefix consistent between the source metadata and
what the loader resolves.

### VoxCeleb1 (`voxceleb`)

`VoxCeleb1Identification` with `iden_split.txt` as the split oracle - its 1/2/3 prefix column
selects train/dev/test - over 1251 speakers. Labels use `LABEL2INDEX_VOXCELEB1`: the speaker id
string `i` maps to `i-1` for 1..1251, and a speaker outside that range maps to -1. Identity is
`idXXXXX/<video>/<utterance>.wav` relative to the dataset root; do not add a corpus-prefix level
the loader does not expect. Archive-level "dev"/"test" distribution boundaries are not the research
split: resolve membership from `iden_split.txt`, and the complete corpus may need the union of
several distribution archives. Never infer a split from an archive filename.

### IEMOCAP (`iemocap`)

`IEMOCAP` over sessions 1-5, tree `Session1..Session5` with `sentences/wav`. Two filters are applied
before extraction: utterances labelled `fru` are dropped, and only speakers of the requested
sessions are kept (matched on the speaker prefix in the path). Labels use `LABEL2INDEX_IEMOCAP`:
neu 0, hap 1, ang 2, sad 3, with `exc` folded onto `hap`; any other label maps to -1. Not every wav
in the corpus belongs to the experiment.

The train/validation/test split is positional, not random: over the filtered list, indices `[4::10]`
are validation, `[9::10]` are test, and the remainder is training
(`upstream/dataset/build_embedding.py`).
Because it is positional, any change to filtering, ordering or membership re-partitions the splits:
membership changes are never local.

### Fluent Speech Commands (`fluentspeechcommand`)

`FluentSpeechCommands` with its own train/valid/test subsets. The label is built from the normalized
(lowercased, stripped, spaces to underscores) action, object and location fields, joined with `_`
after dropping `none` parts, then mapped through `LABEL2INDEX_FLUENTSPEECHCOMMAND` (31 intents).
Do not treat historical sample counts as eternal; read the loader and config.

## Validation in three levels

1. Source contract, before any GPU spend: expected ids and counts, split membership, required
   metadata files, audio readability, sample rate and channel contract, labels, duplicate, missing
   and unexpected members. The generator must not be where a bad corpus is discovered after hours
   of paid compute.
2. Generated set, after extraction and over the complete expected output when computationally
   reasonable: exact count, identity set, no duplicates, no missing or unexpected ids, every tensor
   loadable, expected shape and dtype, finite values (no NaN, no Inf), non-zero files, correct label
   and split metadata, index-to-disk consistency.
3. Independent oracle: derive expected membership from the source - split lists, loader metadata,
   dataset annotations - never only from an index the generator itself wrote. Common-mode failure
   between producer and verifier is treated in `wavcse-artifact-pipeline`.

What the repository does and does not check: there is no standalone embedding validator script. The
only index the extractor produces about its own output is the CSV described above, which records a
path, a label and a human-readable size at write time. It cannot prove tensor contents, membership
or split correctness, and the training loader does not read it. Build the oracle from the dataset's
own metadata.

## Execution

Run extraction from an exact reproducible commit. Never hand-copy patched Python files to a worker,
never execute an uncommitted local scientific change, never run whatever is on the worker's `main`.
Checkout and job-submission mechanics: `wavcse-infra-operator`.

Treat the WavLM checkpoint as an immutable artifact and verify its identity before using a cached
copy. Do not substitute another WavLM size, another host's differing bytes, or a Hugging Face
conversion because the download is easier; a mirror is acceptable once identity/equivalence is
established.

Order: source contract, checkpoint identity, exact checkout, extraction, generated-set validation,
deterministic packaging, publication, independent verification, warm materialization. Extraction
process exit is not completion.

## Packaging and warm inputs

The generated membership defines the scientific content of a package: order members
deterministically and choose shard boundaries explicitly, never from filesystem traversal order, so
a re-run over the same inputs reproduces the same package. Byte-level rules - digests, size
ceilings, manifest, upload and read-back verification, cache and cleanup - are
`wavcse-artifact-pipeline`.

After publication, inspect how the current experiment loader consumes embeddings before deciding
what to materialize. It resolves one `.pt` per sample by path and slices the layer axis in memory,
so an unpacked verified tree on persistent storage is the natural warm form and already serves
layer-subset experiments without further extraction. Do not duplicate materialization the consumer
can read directly, and measure storage rather than trusting historical estimates.

## Consumption on a worker

A verified artifact that the run never reads proves nothing. The consumer contract is explicit,
and a compute plan must satisfy it rather than rely on a worker happening to hold a tree:

- the plan declares `embedding_layout`, naming for each dataset the archive artifact that carries
  it (see the compute backend README, "Declared inputs reach the loader");
- the job wrapper re-verifies each materialized archive against the plan's own digest, extracts it
  into `<root>/<upstream_model_type>/<frame_pool_id>/<dataset>/`, records the mapping in the job
  manifest, and names that root in `WAVCSE_ROOT_EMB_PATH`;
- `improvements/embedding_root.py` honours that override only when the root carries the marker the
  wrapper wrote, so a stale worker-local tree is refused rather than silently used, and a worker
  with no prepared root never falls back to the configured one.

The archive form is a plain TAR of the dataset directory, matching the canonical layout above, so
extraction is the only transformation and no filename or membership rule is reinvented here.

## Failure policy

Hard-stop the affected dataset on: source identity mismatch, incorrect membership, checkpoint
mismatch, scientific configuration mismatch, unexplained shape or dtype difference, NaN/Inf,
incomplete output, or failed verification of the published artifact.

Do not "fix" a dataset by deleting inconvenient samples. Do not silently reduce the workload after
an OOM. Retry, cost and stopping decisions belong to `gpu-research-operator` and
`wavcse-infra-operator`.

## Readiness

Report per dataset: source verified, exact scientific configuration recorded, embedding count,
tensor contract, canonical artifact state, warm representation, ready. The objective is not that
every historical dataset exists, but that every dataset the current experiment requires is
reproducibly ready. Readiness state names and canonical definitions: `wavcse-artifact-pipeline`.
