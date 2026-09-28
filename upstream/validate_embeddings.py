"""
Validate a generated upstream embedding set (`.pt` tensor files + CSV index).

Run this after `upstream/main.py` finishes a split or a whole dataset, and
before the embedding directory is archived/uploaded as a canonical artifact.
It is a read-only integrity check: it never modifies or deletes anything.

Reused from the writer:
  * utils.get_size.get_file_size -- the same human-readable size formatting
    that upstream/utils/embedding_io.py records in the CSV, so the recorded
    size column can be re-derived from the file on disk and compared exactly.

Errors (non-zero exit):
  * CSV index missing, unreadable, or with an unexpected header
  * duplicate `wavpath` rows
  * a row whose derived `.pt` path is missing, zero-byte, or unreadable
  * a `.pt` that does not load as a `torch.Tensor`
  * an empty (numel == 0) or non-2-D tensor
  * inconsistent tensor shapes or dtypes across the set
  * NaN or Inf values
  * a CSV `file_size` column that does not match the file on disk
  * a `label` that is not a non-negative integer

Warnings (reported, exit stays 0):
  * `.pt` files on disk that no CSV row references (e.g. a stale CSV)
  * all-zero tensors
  * a source waveform missing when `--source-root` is supplied

Usage (`upstream/` must be the working directory, as with main.py):

    cd upstream
    python validate_embeddings.py --root ~/embedding/wavlm_large/mean/speechcommand

    # smoke run: first 3 rows only
    python validate_embeddings.py --root <dir> --limit 3

    # also prove each embedding maps to a real source waveform
    python validate_embeddings.py --root <dir> --source-root ~/voice_dataset/speechcommand/SpeechCommands

Author: created for reproducible embedding generation (wavCSE).
"""

import argparse
import csv
import os
from collections import Counter
from typing import List, Optional, Tuple

import torch

from utils.get_size import get_file_size

CSV_HEADER = ("wavpath", "label", "file_size")


class Report:
    """Accumulates errors and warnings with bounded example output."""

    def __init__(self, max_reported: int = 10) -> None:
        self.max_reported = max_reported
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self._error_examples = Counter()
        self._warning_examples = Counter()

    def error(self, category: str, detail: str) -> None:
        self._error_examples[category] += 1
        if self._error_examples[category] <= self.max_reported:
            self.errors.append(f"{category}: {detail}")

    def warn(self, category: str, detail: str) -> None:
        self._warning_examples[category] += 1
        if self._warning_examples[category] <= self.max_reported:
            self.warnings.append(f"{category}: {detail}")

    def error_counts(self) -> Counter:
        return self._error_examples

    def warning_counts(self) -> Counter:
        return self._warning_examples

    @property
    def failed(self) -> bool:
        return bool(self._error_examples)


def derive_layout(
    root: str,
    model: Optional[str],
    pooling: Optional[str],
    dataset: Optional[str],
) -> Tuple[str, str, str]:
    """Derive (model, pooling, dataset) from <root_emb>/<model>/<pooling>/<dataset>."""
    if model and pooling and dataset:
        return model, pooling, dataset

    parts = os.path.normpath(os.path.abspath(root)).split(os.sep)
    if len(parts) < 3:
        raise SystemExit(
            f"Cannot derive model/pooling/dataset from root '{root}'. "
            "Pass --model, --pooling and --dataset explicitly."
        )
    derived = (parts[-3], parts[-2], parts[-1])

    if not model:
        model = derived[0]
    if not pooling:
        pooling = derived[1]
    if not dataset:
        dataset = derived[2]
    return model, pooling, dataset


def embedding_path(root: str, wavpath: str, model: str, pooling: str) -> str:
    """Reproduce upstream/utils/embedding_io.py's output path rule."""
    return os.path.join(root, wavpath.replace(".wav", f"_{model}_{pooling}.pt"))


def read_csv_rows(
    csv_path: str, report: Report
) -> List[Tuple[str, str, str]]:
    if not os.path.isfile(csv_path):
        report.error("csv-missing", csv_path)
        return []

    with open(csv_path, newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration:
            report.error("csv-empty", csv_path)
            return []
        if header != CSV_HEADER:
            report.error(
                "csv-header",
                f"{csv_path}: expected {list(CSV_HEADER)}, found {list(header)}",
            )
            return []
        rows = [tuple(row) for row in reader if row]

    expected_cols = len(CSV_HEADER)
    for row in rows:
        if len(row) != expected_cols:
            report.error("csv-row-width", f"{csv_path}: {row!r}")
    return [row for row in rows if len(row) == expected_cols]


def find_pt_files(root: str) -> List[str]:
    found = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".pt"):
                found.append(os.path.relpath(os.path.join(dirpath, name), root))
    return sorted(found)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate generated upstream embeddings (.pt + CSV)."
    )
    parser.add_argument(
        "--root", required=True,
        help="Embedding directory for ONE dataset: "
             "<root_emb_path>/<model>/<pooling>/<dataset>",
    )
    parser.add_argument("--csv", default=None, help="CSV index path (default: derived)")
    parser.add_argument("--model", default=None, help="Override model segment")
    parser.add_argument("--pooling", default=None, help="Override pooling segment")
    parser.add_argument("--dataset", default=None, help="Override dataset segment")
    parser.add_argument(
        "--expected-layers", type=int, default=None,
        help="Expected leading tensor dimension (e.g. 25 for WavLM Large)",
    )
    parser.add_argument(
        "--expected-dim", type=int, default=None,
        help="Expected trailing tensor dimension (e.g. 1024 for WavLM Large)",
    )
    parser.add_argument("--expected-dtype", default="float32")
    parser.add_argument(
        "--source-root", default=None,
        help="Directory the CSV wavpaths are relative to; when given, every "
             "embedding must map back to an existing source waveform",
    )
    parser.add_argument("--limit", type=int, default=None, help="Validate only first N rows")
    parser.add_argument("--max-reported", type=int, default=10)
    args = parser.parse_args()

    root = os.path.abspath(os.path.expanduser(args.root))
    report = Report(max_reported=args.max_reported)

    if not os.path.isdir(root):
        print(f"ERROR: embedding directory does not exist: {root}")
        return 1

    model, pooling, dataset = derive_layout(root, args.model, args.pooling, args.dataset)
    csv_path = args.csv or os.path.join(root, f"{dataset}_{model}_{pooling}.csv")
    source_root = os.path.abspath(os.path.expanduser(args.source_root)) if args.source_root else None
    expected_dtype = getattr(torch, args.expected_dtype)

    print(f"root       : {root}")
    print(f"model      : {model}")
    print(f"pooling    : {pooling}")
    print(f"dataset    : {dataset}")
    print(f"csv        : {csv_path}")

    rows = read_csv_rows(csv_path, report)
    total_rows = len(rows)
    if args.limit is not None:
        rows = rows[: args.limit]

    seen = set()
    shapes = Counter()
    dtypes = Counter()
    labels = Counter()
    referenced = set()
    total_bytes = 0
    global_min = None
    global_max = None
    global_abs_sum = 0.0
    global_numel = 0
    zero_tensors = 0

    for wavpath, label, recorded_size in rows:
        if wavpath in seen:
            report.error("duplicate-row", wavpath)
            continue
        seen.add(wavpath)

        try:
            label_value = int(label)
            if label_value < 0:
                raise ValueError
            labels[label_value] += 1
        except ValueError:
            report.error("bad-label", f"{wavpath}: {label!r}")

        if ".wav" not in wavpath:
            report.error("non-wav-path", wavpath)

        pt_rel = os.path.relpath(embedding_path(root, wavpath, model, pooling), root)
        pt_path = os.path.join(root, pt_rel)
        referenced.add(pt_rel)

        if not os.path.isfile(pt_path):
            report.error("pt-missing", wavpath)
            continue
        size = os.path.getsize(pt_path)
        if size == 0:
            report.error("pt-empty-file", wavpath)
            continue
        total_bytes += size

        actual_size = get_file_size(pt_path)
        if actual_size != recorded_size:
            report.error(
                "size-mismatch",
                f"{wavpath}: csv={recorded_size!r} actual={actual_size!r}",
            )

        try:
            tensor = torch.load(pt_path, map_location="cpu")
        except Exception as exc:
            report.error("pt-load", f"{wavpath}: {type(exc).__name__}: {exc}")
            continue

        if not isinstance(tensor, torch.Tensor):
            report.error("not-a-tensor", f"{wavpath}: {type(tensor).__name__}")
            continue
        if tensor.numel() == 0:
            report.error("empty-tensor", wavpath)
            continue
        if tensor.dim() != 2:
            report.error("tensor-ndim", f"{wavpath}: ndim={tensor.dim()}")
            continue

        shapes[tuple(tensor.shape)] += 1
        dtypes[str(tensor.dtype)] += 1

        if tensor.dtype != expected_dtype:
            report.error("dtype", f"{wavpath}: {tensor.dtype} != {expected_dtype}")

        as_float = tensor.to(torch.float64)
        if torch.isnan(as_float).any():
            report.error("nan", wavpath)
        if torch.isinf(as_float).any():
            report.error("inf", wavpath)
        if not torch.isfinite(as_float).all():
            report.error("non-finite", wavpath)
        if torch.count_nonzero(as_float) == 0:
            zero_tensors += 1
            report.warn("all-zero-tensor", wavpath)

        tensor_min = float(as_float.min())
        tensor_max = float(as_float.max())
        global_min = tensor_min if global_min is None else min(global_min, tensor_min)
        global_max = tensor_max if global_max is None else max(global_max, tensor_max)
        global_abs_sum += float(as_float.abs().sum())
        global_numel += as_float.numel()

        if source_root is not None:
            source_path = os.path.join(source_root, wavpath)
            if not os.path.isfile(source_path):
                report.warn("source-missing", source_path)

    on_disk = set(find_pt_files(root))
    for orphan in sorted(on_disk - referenced):
        report.warn("unreferenced-pt", orphan)
    for missing in sorted(referenced - on_disk):
        report.error("pt-missing-on-disk", missing)

    if args.expected_layers is not None and args.expected_dim is not None:
        expected_shape = (args.expected_layers, args.expected_dim)
        for shape in shapes:
            if shape != expected_shape:
                report.error("shape", f"observed {tuple(shape)}, expected {expected_shape}")

    print()
    print("--- summary ---")
    print(f"csv rows (validated / total) : {len(rows)} / {total_rows}")
    print(f"referenced .pt files         : {len(referenced)}")
    print(f".pt files on disk            : {len(on_disk)}")
    print(f"total .pt bytes              : {total_bytes} ({total_bytes / 1024 ** 2:.1f} MiB)")
    print(f"tensor shapes                : {dict(shapes)}")
    print(f"tensor dtypes                : {dict(dtypes)}")
    print(f"labels (count)               : {dict(sorted(labels.items()))}")
    print(f"all-zero tensors             : {zero_tensors}")
    if global_numel:
        print(f"value min / max              : {global_min:.6g} / {global_max:.6g}")
        print(f"value mean abs               : {global_abs_sum / global_numel:.6g}")

    if report.warnings:
        print()
        print("--- warnings ---")
        for line in report.warnings:
            print(f"WARN {line}")
        for category, count in sorted(report.warning_counts().items()):
            print(f"WARN {category}: {count} occurrence(s)")

    if report.failed:
        print()
        print("--- errors ---")
        for line in report.errors:
            print(f"ERROR {line}")
        for category, count in sorted(report.error_counts().items()):
            print(f"ERROR {category}: {count} occurrence(s)")
        print()
        print("RESULT: FAILED")
        return 1

    print()
    print("RESULT: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
