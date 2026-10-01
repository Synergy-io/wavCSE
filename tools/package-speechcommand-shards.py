#!/usr/bin/env python3
"""Validate, then deterministically package, a full Speech Commands embedding run.

Stages:

  1. Run the wavCSE repository's own read-only validator over the COMPLETE embedding
     set (every `.pt` loads, uniform shape/dtype, no NaN/Inf, no zero-byte file, no
     duplicate index row, the CSV size column re-derived from each file on disk, and
     every embedding mapped back to a real source waveform).
  2. Partition the index by the dataset's own `validation_list.txt` / `testing_list.txt`
     (everything else is training) and cut each split into contiguous shards, so a shard
     is a deterministic, reproducible subset rather than an arbitrary chunk.
  3. Write one uncompressed tar per shard whose members are exactly the `.pt` files,
     named relative to the embeddings root, with normalized tar metadata so a shard's
     SHA-256 is reproducible from the same inputs.
  4. Report per-shard count, bytes and SHA-256, plus the CSV's own SHA-256.

Nothing scientific is decided here: the tensors are copied byte-for-byte, the split
membership comes from the dataset's own lists, and the label mapping is the writer's.

The shard size exists because a worker upload is a single PUT capped at 5 GB: the whole
set is ~6.8 GB, and even the training split alone exceeds the cap.

Operator tooling: deliberately outside both the wavCSE and wavcse-infra repositories.
Published to S3 as an immutable, content-addressed job input.
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

SPLITS = ("training", "validation", "testing")
SPLIT_ORDER = {name: index for index, name in enumerate(SPLITS)}


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


def read_list(path):
    if not os.path.isfile(path):
        raise SystemExit(f"split list is missing: {path}")
    with open(path) as handle:
        return {line.strip() for line in handle if line.strip()}


def find_lists_root(source_root):
    """Return the dataset-version directory that holds the two split lists."""

    for name in sorted(os.listdir(source_root)):
        candidate = os.path.join(source_root, name)
        if os.path.isdir(candidate) and os.path.isfile(
            os.path.join(candidate, "validation_list.txt")
        ):
            return candidate
    raise SystemExit(f"no split lists found beneath {source_root}")


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
        required=True,
        help="Expected training,validation,testing counts, comma separated",
    )
    parser.add_argument("--validator-python", required=True)
    parser.add_argument("--validator-script", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument(
        "--pooling",
        default="mean",
        help="Pooling identity of the source run (default: mean); selects the "
        "wavlm_large/<pooling>/ directory, the <dataset>_wavlm_large_<pooling>.csv index "
        "and the .pt suffix",
    )
    parser.add_argument("--output-prefix", default="outputs/shards")
    # Pinned rather than left to the validator's consistency-only default: a run that
    # selected the wrong layer set or the wrong encoder would otherwise be uniformly wrong
    # and still internally consistent.
    parser.add_argument("--expected-layers", type=int, default=25)
    parser.add_argument("--expected-dim", type=int, default=1024)
    parser.add_argument("--expected-dtype", default="float32")
    args = parser.parse_args()

    if not args.pooling:
        raise SystemExit("--pooling must be a non-empty pooling identity")
    pooling_suffix = f"wavlm_large_{args.pooling}"

    job = os.environ["WAVCSE_JOB_DIRECTORY"]
    embeddings = os.path.join(
        args.source_job, "outputs", "embeddings", "wavlm_large", args.pooling, "speechcommand"
    )
    csv_path = os.path.join(embeddings, f"speechcommand_{pooling_suffix}.csv")
    if not os.path.isfile(csv_path):
        raise SystemExit(f"no embedding index at {csv_path}")

    expected = [int(part) for part in args.expected_counts.split(",")]
    if len(expected) != len(SPLITS):
        raise SystemExit("--expected-counts must name training,validation,testing")

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

    # 2. Partition by the dataset's own split lists and cut deterministic shards.
    #
    # The index's wavpaths are taken from torchaudio relative to the archive root, so they
    # carry the dataset version directory ("speech_commands_v0.01/bed/x.wav"), while the
    # split lists name members relative to that directory ("bed/x.wav"). Re-add the prefix
    # so the two agree; matching without it silently classifies every row as training.
    lists_root = find_lists_root(args.source_root)
    version_dir = os.path.basename(lists_root)
    validation = {
        f"{version_dir}/{line}"
        for line in read_list(os.path.join(lists_root, "validation_list.txt"))
    }
    testing = {
        f"{version_dir}/{line}" for line in read_list(os.path.join(lists_root, "testing_list.txt"))
    }

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

    counts = {}
    split_of = []
    for wavpath, _label, _size in rows:
        split = (
            "validation"
            if wavpath in validation
            else "testing"
            if wavpath in testing
            else "training"
        )
        split_of.append(split)
        counts[split] = counts.get(split, 0) + 1

    for index, name in enumerate(SPLITS):
        if counts.get(name, 0) != expected[index]:
            raise SystemExit(
                f"split {name} holds {counts.get(name, 0)} rows, expected {expected[index]}"
            )
    order = [SPLIT_ORDER[name] for name in split_of]
    if order != sorted(order):
        raise SystemExit("CSV rows are not contiguous by split; refusing to guess shard membership")

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
                        raise SystemExit(
                            f"{relative} is {actual} but the index recorded {recorded_size}"
                        )
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
    csv_out = os.path.join(job, "outputs", f"speechcommand_{pooling_suffix}.csv")
    with open(csv_path, "rb") as src, open(csv_out, "wb") as dst:
        dst.write(src.read())

    report = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_job": args.source_job,
        "pooling": args.pooling,
        "embeddings_root": embeddings,
        "total_rows": len(rows),
        "split_counts": counts,
        "shard_size": args.shard_size,
        "csv": {
            "bytes": os.path.getsize(csv_out),
            "sha256": sha256_of(csv_out),
        },
        "shards": shards,
    }
    with open(os.path.join(job, "outputs", "shards.json"), "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print("PACKAGING OK", flush=True)


if __name__ == "__main__":
    main()
