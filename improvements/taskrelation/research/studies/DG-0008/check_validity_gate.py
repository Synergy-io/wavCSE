"""DG-0008 Stage-1 validity gate driver: record every required-zero count.

Reads the stored run records and the frozen index, and writes ``gate.json``.
No held-out endpoint may be read until this reports ``stage1_valid: true``.

The coarse S sanity envelope and the declared embedding SHA-256 set are inputs
to this gate, never invented here: the caller must supply the measured envelope
and both digest maps, or the gate records the missing prerequisite and stays
invalid (fail closed).

Usage::

    python check_validity_gate.py --index artifacts/index.json --out outputs \
        --s-min <measured> --s-max <measured> \
        --expected-embeddings embeddings.json --observed-embeddings observed.json
"""

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
        )
    )
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from improvements.taskrelation.research.dg0008 import artifacts as frozen  # noqa: E402
from improvements.taskrelation.research.dg0008 import gate as gate_module  # noqa: E402
from improvements.taskrelation.research.dg0008.manifest import (  # noqa: E402
    ManifestError,
    canonical_json_bytes,
)

CELLS = ("ks_er", "si_er")
ARMS = ("pair", "control")


def _read_json(path):
    if path is None or not os.path.exists(path):
        return None
    with open(path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


def _run_record(out_root, cell, arm, fold, seed, repeat=False):
    bucket = "repeat" if repeat else "runs"
    path = os.path.join(
        out_root, bucket, cell, arm, "f{}".format(fold), "s{}".format(seed), "run.json"
    )
    return _read_json(path)


def collect_records(out_root):
    records = []
    for cell in CELLS:
        for arm in ARMS:
            for fold in range(10):
                for seed in range(5):
                    record = _run_record(out_root, cell, arm, fold, seed)
                    if record is not None:
                        records.append(record)
    return records


def build_gate(index, artifacts_directory, out_root, s_min, s_max, expected_embeddings,
               observed_embeddings):
    records = collect_records(out_root)
    expected_index = {"runs": index["runs"], "epochs": index["epochs"]}
    expected_members = {
        (cell, fold, seed, arm)
        for cell in CELLS for arm in ARMS for fold in range(10) for seed in range(5)
    }

    problems = []
    observed_identity_digest = index["identity_digest"]
    try:
        frozen.load_identity(index, artifacts_directory)
    except ManifestError as exc:
        observed_identity_digest = "<unverifiable: {}>".format(exc)
    if expected_embeddings is None or observed_embeddings is None:
        problems.append({"class": "MISSING_EMBEDDING_DIGESTS", "count": 1})
        expected_embeddings = expected_embeddings or {}
        observed_embeddings = observed_embeddings or {}

    read_count = 0
    ledger = os.path.join(out_root, "held_out_reads.jsonl")
    if os.path.exists(ledger):
        with open(ledger, "rb") as handle:
            read_count = sum(1 for line in handle if line.strip())

    gate = gate_module.stage1_gate(
        records,
        expected_index=expected_index,
        expected_identity_digest=index["identity_digest"],
        observed_identity_digest=observed_identity_digest,
        expected_embeddings=expected_embeddings,
        observed_embeddings=observed_embeddings,
        s_envelope=(s_min, s_max),
        repeat_first=_run_record(out_root, "si_er", "control", 0, 0),
        repeat_second=_run_record(out_root, "si_er", "control", 0, 0, repeat=True),
        held_out_reads=read_count,
        expected_matrix_members=expected_members,
    )
    gate["problems"] = problems + gate["problems"]
    if problems:
        gate["stage1_valid"] = False
    gate["matrix_digest"] = index["matrix_digest"]
    gate["identity_digest"] = index["identity_digest"]
    return gate


def main():
    parser = argparse.ArgumentParser(description="DG-0008 Stage-1 validity gate")
    parser.add_argument("--index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gate-out", default=None)
    parser.add_argument("--s-min", type=int, required=True)
    parser.add_argument("--s-max", type=int, required=True)
    parser.add_argument("--expected-embeddings", default=None)
    parser.add_argument("--observed-embeddings", default=None)
    args = parser.parse_args()

    index = frozen.load_index(os.path.abspath(os.path.expanduser(args.index)))
    out_root = os.path.abspath(os.path.expanduser(args.out))
    gate = build_gate(
        index,
        os.path.dirname(os.path.abspath(os.path.expanduser(args.index))),
        out_root,
        args.s_min,
        args.s_max,
        _read_json(args.expected_embeddings),
        _read_json(args.observed_embeddings),
    )
    destination = args.gate_out or os.path.join(out_root, "gate.json")
    with open(destination, "wb") as handle:
        handle.write(canonical_json_bytes(gate))
    print("DG0008_GATE", gate["stage1_valid"], json.dumps(gate["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
