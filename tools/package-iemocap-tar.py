#!/usr/bin/env python3
"""Validate, then deterministically archive, a full IEMOCAP WavLM-Large run.

The sibling of ``package-speechcommand-shards.py`` / ``package-voxceleb-shards.py`` for the
ER task.  IEMOCAP is 5,531 utterances (~0.58 GB of tensors), so the set fits a single
uncompressed tar well below the 5 GB single-PUT ceiling; no sharding is needed.

``--pooling`` names the pooling identity of the run being packaged (default ``mean``).  It
selects the ``wavlm_large/<pooling>/`` embedding directory, the
``iemocap_wavlm_large_<pooling>.csv`` index, the ``.pt`` filename suffix and the default
archive name, so one packager serves every pooling strategy instead of a per-strategy copy.

Checks performed, in order, before anything is written:

  1. ``validate_embeddings.py`` over the whole embedding directory with the layer, dimension
     and dtype pinned -- loadability, shape/dtype uniformity, NaN/Inf, zero-byte files,
     duplicate rows, CSV size agreement, orphans, missing files, and source-waveform
     resolution.
  2. A *loader-derived oracle*: torchaudio 2.7.1's ``IEMOCAP`` listing (the class wavCSE's
     ``IEMOCAPEmbedding`` extends), the repository's own ``fru`` drop and ten-speaker
     filter, and the repository's own positional split (``val = indices[4::10]``,
     ``test = indices[9::10]``).  Because ``main.py`` writes train, then val, then test
     into one CSV, the oracle is compared **row by row, in order**, against the index, so a
     reordered, truncated, mislabelled or wrongly-split run cannot pass.

Archive layout: plain uncompressed tar holding the CSV index plus every ``.pt`` under its
writer-relative path (``Session1/sentences/wav/<SesXXY_dir>/<utt>_wavlm_large_<pooling>.pt``),
with normalized metadata so the bytes -- and therefore the digest -- are reproducible.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tarfile
import time

SPLITS = ("train", "val", "test")
SPLIT_ORDER = {name: index for index, name in enumerate(SPLITS)}
# Published in datasets/v1/iemocap/provenance.json and verified against the corpus there.
PUBLISHED_COUNTS = {"train": 4425, "val": 553, "test": 553}
PUBLISHED_HISTOGRAM = {0: 1708, 1: 1636, 2: 1103, 3: 1084}
CSV_HEADER = ("wavpath", "label", "file_size")


def human_size(size_bytes):
    """Reproduce upstream/utils/get_size.py, the writer of the index size column."""

    if size_bytes == 0:
        return "0 B"
    names = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    index = math.floor(math.log(size_bytes, 1024))
    return f"{round(size_bytes / math.pow(1024, index), 2)} {names[index]}"


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_validator(arguments):
    print("--- validation ---", flush=True)
    result = subprocess.run(arguments, check=False)
    if result.returncode != 0:
        raise SystemExit(f"validation failed with exit code {result.returncode}")


def add_deterministic(tar, path, arcname):
    """Add one regular file with normalized metadata, so the tar bytes are reproducible."""

    info = tar.gettarinfo(path, arcname=arcname)
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o644
    with open(path, "rb") as handle:
        tar.addfile(info, handle)


def loader_oracle(corpus_root, upstream_dir, expected_counts):
    """Return [(wavpath, label_index)] in the exact order wavCSE's ER pipeline emits them.

    Uses the real torchaudio dataset class and the repository's own filters and split
    strides, so the oracle is derived from the corpus rather than from the output under
    test.  Every deviation is a hard failure with the observed value next to the expected.
    """

    sys.path.insert(0, upstream_dir)
    from torchaudio.datasets import IEMOCAP
    from utils.constant_mapping import LabelKeywordMapping

    label_mapping = LabelKeywordMapping.get_label_mapping("iemocap")[0]
    base = IEMOCAP(str(corpus_root), ("1", "2", "3", "4", "5"), None)

    speakers = {f"Ses0{session}{gender}" for session in "12345" for gender in "FM"}
    stems = [stem for stem in base.data if base.mapping[stem]["label"] != "fru"]
    stems = [
        stem
        for stem in stems
        if base.mapping[stem]["path"].split("/")[3].split("_")[0] in speakers
    ]

    ordered = []
    for stem in stems:
        path = base.mapping[stem]["path"]
        label = base.mapping[stem]["label"]
        if label in label_mapping:
            index = label_mapping[label]
        elif label == "exc":
            index = label_mapping["hap"]
        else:
            raise SystemExit(f"oracle: label {label!r} has no index and is not 'exc'")
        ordered.append((path, index))

    total = len(ordered)
    val_indices = list(range(total))[4::10]
    test_indices = list(range(total))[9::10]
    val_set = set(val_indices)
    test_set = set(test_indices)
    train_indices = [index for index in range(total) if index not in val_set and index not in test_set]

    by_split = {}
    for name, indices in (("train", train_indices), ("val", val_indices), ("test", test_indices)):
        by_split[name] = [ordered[index] for index in indices]

    counts = {name: len(rows) for name, rows in by_split.items()}
    if counts != expected_counts:
        raise SystemExit(f"oracle split counts {counts}, expected {expected_counts}")

    histogram = {}
    for index in range(total):
        value = ordered[index][1]
        histogram[value] = histogram.get(value, 0) + 1

    expected_order = by_split["train"] + by_split["val"] + by_split["test"]
    return expected_order, counts, histogram


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-job", required=True, help="Job directory holding the extraction run")
    parser.add_argument("--upstream-dir", required=True, help="<source-job>/source/upstream")
    parser.add_argument("--corpus-root", required=True, help="Root passed to the loader: <root_data_path>/iemocap")
    parser.add_argument("--validator-script", required=True)
    parser.add_argument(
        "--expected-counts",
        default=",".join(str(PUBLISHED_COUNTS[name]) for name in SPLITS),
        help="Expected train,val,test counts (defaults to the published loader-projection counts)",
    )
    parser.add_argument(
        "--expected-histogram",
        default=",".join(f"{key}:{value}" for key, value in sorted(PUBLISHED_HISTOGRAM.items())),
        help="Expected overall class histogram as index:count pairs",
    )
    parser.add_argument("--expected-layers", type=int, default=25)
    parser.add_argument("--expected-dim", type=int, default=1024)
    parser.add_argument("--expected-dtype", default="float32")
    parser.add_argument(
        "--pooling",
        default="mean",
        help="Pooling identity of the source run (default: mean); selects the "
        "wavlm_large/<pooling>/ directory, the <dataset>_wavlm_large_<pooling>.csv index "
        "and the .pt suffix",
    )
    parser.add_argument(
        "--archive-name",
        default=None,
        help="Archive filename inside --output-prefix (default: iemocap-wavlm-large-<pooling>.tar)",
    )
    parser.add_argument("--output-prefix", default="outputs")
    args = parser.parse_args()

    if not args.pooling:
        raise SystemExit("--pooling must be a non-empty pooling identity")
    pooling_suffix = f"wavlm_large_{args.pooling}"
    archive_name = args.archive_name or f"iemocap-wavlm-large-{args.pooling}.tar"

    job = os.environ["WAVCSE_JOB_DIRECTORY"]
    embeddings = os.path.join(
        args.source_job, "outputs", "embeddings", "wavlm_large", args.pooling, "iemocap"
    )
    csv_path = os.path.join(embeddings, f"iemocap_{pooling_suffix}.csv")
    if not os.path.isfile(csv_path):
        raise SystemExit(f"no embedding index at {csv_path}")

    expected = [int(part) for part in args.expected_counts.split(",")]
    if len(expected) != len(SPLITS):
        raise SystemExit("--expected-counts must name train,val,test")
    expected_counts = dict(zip(SPLITS, expected))
    expected_histogram = {}
    for part in args.expected_histogram.split(","):
        key, _, value = part.partition(":")
        expected_histogram[int(key)] = int(value)

    # 1. Complete tensor/index integrity check over every embedding.
    run_validator(
        [
            sys.executable,
            args.validator_script,
            "--root",
            embeddings,
            "--source-root",
            os.path.join(args.corpus_root, "IEMOCAP"),
            "--expected-layers",
            str(args.expected_layers),
            "--expected-dim",
            str(args.expected_dim),
            "--expected-dtype",
            args.expected_dtype,
        ]
    )

    # 2. Loader-derived oracle compared row by row, in order, against the index.
    print("--- loader oracle ---", flush=True)
    oracle_rows, oracle_counts, oracle_histogram = loader_oracle(
        args.corpus_root, args.upstream_dir, expected_counts
    )
    print(f"oracle rows: {len(oracle_rows)}; splits {oracle_counts}; histogram {oracle_histogram}", flush=True)
    for key, value in sorted(expected_histogram.items()):
        if oracle_histogram.get(key, 0) != value:
            raise SystemExit(
                f"oracle class {key} holds {oracle_histogram.get(key, 0)} rows, expected {value}"
            )

    rows = []
    with open(csv_path, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if tuple(header) != CSV_HEADER:
            raise SystemExit(f"unexpected CSV header: {header}")
        for wavpath, label, file_size in reader:
            rows.append((wavpath, int(label), file_size))

    if len(rows) != len(oracle_rows):
        raise SystemExit(f"index holds {len(rows)} rows, the loader project holds {len(oracle_rows)}")
    for position, (row, oracle) in enumerate(zip(rows, oracle_rows)):
        if row[0] != oracle[0]:
            raise SystemExit(f"row {position}: index wavpath {row[0]!r} != loader order {oracle[0]!r}")
        if row[1] != oracle[1]:
            raise SystemExit(f"row {position} ({row[0]!r}): index label {row[1]} != loader label {oracle[1]}")
    print(f"index matches the loader projection exactly, row for row ({len(rows)} rows)", flush=True)

    split_of = []
    cursor = 0
    for name in SPLITS:
        count = oracle_counts[name]
        split_of.extend([name] * count)
        cursor += count
    if cursor != len(rows):
        raise SystemExit("internal: split partition does not cover the index")

    histogram = {}
    split_histogram = {}
    for (wavpath, label, _size), name in zip(rows, split_of):
        histogram[label] = histogram.get(label, 0) + 1
        bucket = split_histogram.setdefault(name, {})
        bucket[label] = bucket.get(label, 0) + 1
    for split in SPLITS:
        print(f"{split}: {oracle_counts[split]} rows, classes {dict(sorted(split_histogram[split].items()))}", flush=True)
    print(f"overall classes: {dict(sorted(histogram.items()))}", flush=True)

    # 3. Deterministic archive: every referenced .pt plus the index, sorted member order.
    members = []
    for wavpath, _label, recorded_size in rows:
        relative = wavpath.replace(".wav", f"_{pooling_suffix}.pt")
        source = os.path.join(embeddings, relative)
        if not os.path.isfile(source):
            raise SystemExit(f"missing embedding for {wavpath}: {source}")
        actual = human_size(os.path.getsize(source))
        if actual != recorded_size:
            raise SystemExit(f"{relative} is {actual} but the index recorded {recorded_size}")
        members.append((relative, source))

    out_dir = os.path.join(job, args.output_prefix)
    os.makedirs(out_dir, exist_ok=True)
    archive_path = os.path.join(out_dir, archive_name)
    arcnames = sorted([relative for relative, _ in members] + [os.path.basename(csv_path)])
    sources = dict(members)
    sources[os.path.basename(csv_path)] = csv_path
    with tarfile.open(archive_path, "w", format=tarfile.GNU_FORMAT) as tar:
        for arcname in arcnames:
            add_deterministic(tar, sources[arcname], arcname)
    print(f"archive {archive_path}: {len(arcnames)} members, {os.path.getsize(archive_path)} bytes", flush=True)

    report = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_job": args.source_job,
        "pooling": args.pooling,
        "embeddings_root": embeddings,
        "corpus_root": args.corpus_root,
        "total_rows": len(rows),
        "split_counts": oracle_counts,
        "split_class_histograms": {name: {str(k): v for k, v in sorted(split_histogram[name].items())} for name in SPLITS},
        "class_histogram": {str(k): v for k, v in sorted(histogram.items())},
        "oracle": {
            "kind": "torchaudio 2.7.1 IEMOCAP listing + the repository's fru/speaker filters + the 4::10,9::10 position strides",
            "rows": len(oracle_rows),
            "row_by_row_match": True,
        },
        "csv": {"bytes": os.path.getsize(csv_path), "sha256": sha256_of(csv_path)},
        "archive": {
            "name": archive_name,
            "bytes": os.path.getsize(archive_path),
            "sha256": sha256_of(archive_path),
            "members": len(arcnames),
            "layout": "plain uncompressed tar; the CSV index plus every .pt at its writer-relative path, normalized metadata",
        },
        "tensor": {"shape": [args.expected_layers, args.expected_dim], "dtype": args.expected_dtype},
    }
    report_path = os.path.join(out_dir, "iemocap-embeddings.json")
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print("PACKAGING OK", flush=True)


if __name__ == "__main__":
    main()
