#!/usr/bin/env python3
"""Validate, then deterministically shard, a full VoxCeleb1 WavLM-Large run.

Sibling of package-speechcommand-shards.py, adapted to VoxCeleb's conventions:

  * the index CSV is ``voxceleb_wavlm_large_<pooling>.csv``;
  * rows carry ``idXXXXX/<youtube>/<utt>.wav`` relative to ``<root>/voxceleb/wav``
    (torchaudio's ``_get_flist`` stores the iden_split path verbatim, so unlike
    Speech Commands there is no dataset-version prefix to re-add);
  * split membership comes from the dataset's own ``iden_split.txt``
    (1=train, 2=dev, 3=test), and is cross-checked three ways: per-split counts
    against the published values, row order against the extraction's write order,
    and set equality of the index against the split file.

Sharding rule: partition by split, then cut deterministic contiguous chunks of
``--shard-size`` rows (default 20000).  With 153,516 utterances that is 9 shards,
the largest ~2.1 GB -- comfortably under the 5 GB single-PUT ceiling.

Tar members are normalized (mtime 0, uid/gid 0, empty uname/gname, mode 0644) so
a shard's bytes, and therefore its digest, are reproducible.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import tarfile
import time

IDEN_TO_SPLIT = {1: "train", 2: "dev", 3: "test"}
SPLITS = ("train", "dev", "test")
SPLIT_ORDER = {name: index for index, name in enumerate(SPLITS)}
# Published iden_split.txt structure -- the same values the source tool verifies.
PUBLISHED_COUNTS = {"train": 138361, "dev": 6904, "test": 8251}


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


def read_iden_split(path):
    """Return {relative wavpath: split name} from iden_split.txt."""
    if not os.path.isfile(path):
        raise SystemExit(f"iden_split.txt is missing: {path}")
    membership = {}
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 2:
                raise SystemExit(f"{path}:{lineno}: expected 2 fields, got {len(fields)}")
            split = IDEN_TO_SPLIT.get(int(fields[0]))
            if split is None:
                raise SystemExit(f"{path}:{lineno}: unexpected split id {fields[0]}")
            membership[fields[1]] = split
    return membership


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-job", required=True, help="Job directory holding the run")
    parser.add_argument("--shard-size", type=int, default=20000)
    parser.add_argument(
        "--expected-counts",
        default=",".join(str(PUBLISHED_COUNTS[name]) for name in SPLITS),
        help="Expected train,dev,test counts (defaults to the published iden_split counts)",
    )
    parser.add_argument("--iden-split", required=True, help="iden_split.txt from the corpus used for the run")
    parser.add_argument("--validator-python", required=True)
    parser.add_argument("--validator-script", required=True)
    parser.add_argument("--source-root", required=True, help="<root_data_path>/voxceleb/wav")
    parser.add_argument(
        "--pooling",
        default="mean",
        help="Pooling identity of the source run (default: mean); selects the "
        "wavlm_large/<pooling>/ directory, the <dataset>_wavlm_large_<pooling>.csv index "
        "and the .pt suffix",
    )
    parser.add_argument("--output-prefix", default="outputs/shards")
    # Pinned rather than left to the validator's consistency-only default: a run that
    # selected the wrong layer set or the wrong encoder would otherwise be uniformly
    # wrong and still internally consistent.
    parser.add_argument("--expected-layers", type=int, default=25)
    parser.add_argument("--expected-dim", type=int, default=1024)
    parser.add_argument("--expected-dtype", default="float32")
    args = parser.parse_args()

    if not args.pooling:
        raise SystemExit("--pooling must be a non-empty pooling identity")
    pooling_suffix = f"wavlm_large_{args.pooling}"

    job = os.environ["WAVCSE_JOB_DIRECTORY"]
    embeddings = os.path.join(
        args.source_job, "outputs", "embeddings", "wavlm_large", args.pooling, "voxceleb"
    )
    csv_path = os.path.join(embeddings, f"voxceleb_{pooling_suffix}.csv")
    if not os.path.isfile(csv_path):
        raise SystemExit(f"no embedding index at {csv_path}")

    expected = [int(part) for part in args.expected_counts.split(",")]
    if len(expected) != len(SPLITS):
        raise SystemExit("--expected-counts must name train,dev,test")
    expected_by_split = dict(zip(SPLITS, expected, strict=True))

    # 1. Complete integrity validation over every embedding, before packaging anything.
    run_validator(
        [
            args.validator_python,
            args.validator_script,
            "--root",
            embeddings,
            "--source-root",
            args.source_root,
            "--expected-layers",
            str(args.expected_layers),
            "--expected-dim",
            str(args.expected_dim),
            "--expected-dtype",
            args.expected_dtype,
        ]
    )

    # 2. Split membership from the dataset's own split file.
    membership = read_iden_split(args.iden_split)
    published = {}
    for split in membership.values():
        published[split] = published.get(split, 0) + 1
    for split in SPLITS:
        if published.get(split, 0) != expected_by_split[split]:
            raise SystemExit(
                f"iden_split.txt holds {published.get(split, 0)} {split} rows, "
                f"expected {expected_by_split[split]}"
            )
    print(f"iden_split.txt: {published}", flush=True)

    rows = []
    with open(csv_path, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if tuple(header) != ("wavpath", "label", "file_size"):
            raise SystemExit(f"unexpected CSV header: {header}")
        for wavpath, label, file_size in reader:
            # The third column is the writer's human-readable size (e.g. "101.85 KB"),
            # not a byte count: it is re-derived per file and compared exactly.
            rows.append((wavpath, int(label), file_size))

    seen = set()
    split_of = []
    for wavpath, _label, _size in rows:
        split = membership.get(wavpath)
        if split is None:
            raise SystemExit(f"index row {wavpath!r} is not in iden_split.txt")
        if wavpath in seen:
            raise SystemExit(f"duplicate index row {wavpath!r}")
        seen.add(wavpath)
        split_of.append(split)
    if seen != set(membership):
        missing = len(set(membership) - seen)
        raise SystemExit(f"{missing} iden_split rows have no index entry; the run is incomplete")

    counts = {}
    for split in split_of:
        counts[split] = counts.get(split, 0) + 1
    for split in SPLITS:
        if counts.get(split, 0) != expected_by_split[split]:
            raise SystemExit(f"split {split} holds {counts.get(split, 0)} rows, expected {expected_by_split[split]}")
    order = [SPLIT_ORDER[name] for name in split_of]
    if order != sorted(order):
        raise SystemExit("CSV rows are not contiguous by split; refusing to guess shard membership")
    # The extraction writes sorted(f_list) per split. Requiring ascending order makes shard
    # boundaries a function of the split *sets* rather than of whatever order the writer
    # happened to emit, so a reordered index cannot silently reshuffle shard membership.
    last_split = last_path = None
    for row, split in zip(rows, split_of, strict=True):
        wavpath = row[0]
        if split == last_split and last_path is not None and wavpath <= last_path:
            raise SystemExit(
                f"index rows are not in ascending order within split {split!r}: {last_path!r} then {wavpath!r}"
            )
        last_split, last_path = split, wavpath

    os.makedirs(os.path.join(job, args.output_prefix), exist_ok=True)
    shards = []
    for name in SPLITS:
        members = [
            (wavpath, size)
            for (wavpath, _label, size), split in zip(rows, split_of, strict=True)
            if split == name
        ]
        for ordinal, start in enumerate(range(0, len(members), args.shard_size)):
            chunk = members[start : start + args.shard_size]
            shard_name = f"{name}-{ordinal:03d}.tar"
            out_path = os.path.join(job, args.output_prefix, shard_name)
            with tarfile.open(out_path, "w", format=tarfile.GNU_FORMAT) as tar:
                for wavpath, recorded_size in sorted(chunk):
                    relative = wavpath.replace(".wav", f"_{pooling_suffix}.pt")
                    source = os.path.join(embeddings, relative)
                    if not os.path.isfile(source):
                        raise SystemExit(f"missing embedding for {wavpath}: {source}")
                    actual = human_size(os.path.getsize(source))
                    if actual != recorded_size:
                        raise SystemExit(f"{relative} is {actual} but the index recorded {recorded_size}")
                    add_deterministic(tar, source, relative)
            shards.append(
                {
                    "shard": shard_name,
                    "split": name,
                    "count": len(chunk),
                    "bytes": os.path.getsize(out_path),
                    "sha256": sha256_of(out_path),
                    "first_wavpath": min(member[0] for member in chunk),
                    "last_wavpath": max(member[0] for member in chunk),
                }
            )
            print(f"shard {shard_name}: {len(chunk)} files, {shards[-1]['bytes']} bytes", flush=True)

    # 3. Publish the index alongside the shards so both describe the same run.
    csv_out = os.path.join(job, "outputs", f"voxceleb_{pooling_suffix}.csv")
    with open(csv_path, "rb") as src, open(csv_out, "wb") as dst:
        dst.write(src.read())

    report = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_job": args.source_job,
        "pooling": args.pooling,
        "embeddings_root": embeddings,
        "iden_split": args.iden_split,
        "total_rows": len(rows),
        "split_counts": counts,
        "shard_size": args.shard_size,
        "csv": {"bytes": os.path.getsize(csv_out), "sha256": sha256_of(csv_out)},
        "shards": shards,
    }
    with open(os.path.join(job, "outputs", "shards.json"), "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print("PACKAGING OK", flush=True)


if __name__ == "__main__":
    main()
