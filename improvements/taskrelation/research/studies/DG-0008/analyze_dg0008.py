"""DG-0008 frozen seed-level analysis driver (classification preparation).

Refuses to run before the Stage-1 validity gate is valid.  Reads the 200
delayed held-out ER endpoints and writes the seed-level intervals and the single
H1/H2/H3/INCONCLUSIVE screening class.  Folds stay nested.

Usage::

    python analyze_dg0008.py --out outputs --index artifacts/index.json
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

from improvements.taskrelation.research.dg0008 import analysis as analysis_module  # noqa: E402
from improvements.taskrelation.research.dg0008 import artifacts as frozen  # noqa: E402
from improvements.taskrelation.research.dg0008 import gate as gate_module  # noqa: E402
from improvements.taskrelation.research.dg0008.manifest import (  # noqa: E402
    ManifestError,
    canonical_json_bytes,
)

CELL_TO_LABEL = {"si_er": "SI", "ks_er": "KS"}


def _endpoint(out_root, cell, arm, fold, seed):
    path = os.path.join(
        out_root, "endpoints", cell, arm, "f{}".format(fold), "s{}".format(seed), "endpoint.json"
    )
    if not os.path.exists(path):
        raise ManifestError("missing held-out endpoint {}".format(path))
    with open(path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


def collect(out_root, identity):
    speakers = {entry["fold"]: entry["test_speaker"] for entry in identity["iemocap_folds"]}
    per_seed = {"SI": [], "KS": []}
    nested_records = []
    for cell, label in CELL_TO_LABEL.items():
        for seed in range(5):
            fold_residuals = []
            for fold in range(10):
                pair = _endpoint(out_root, cell, "pair", fold, seed)
                control = _endpoint(out_root, cell, "control", fold, seed)
                if pair["run_manifest_digest"] != control["run_manifest_digest"]:
                    raise ManifestError("pair/control ran different manifests")
                residual = pair["er_accuracy"] - control["er_accuracy"]
                fold_residuals.append(residual)
                nested_records.append(
                    {
                        "cell": cell,
                        "seed": seed,
                        "fold": fold,
                        "pair_er_accuracy": pair["er_accuracy"],
                        "control_er_accuracy": control["er_accuracy"],
                        "test_speaker": speakers.get(fold),
                    }
                )
            per_seed[label].append(sum(fold_residuals) / len(fold_residuals))
    return per_seed, nested_records


def main():
    parser = argparse.ArgumentParser(description="DG-0008 frozen analysis")
    parser.add_argument("--out", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--delta", type=float, default=analysis_module.DELTA_DG)
    parser.add_argument("--analysis-out", default=None)
    args = parser.parse_args()

    out_root = os.path.abspath(os.path.expanduser(args.out))
    gate_path = os.path.join(out_root, "gate.json")
    with open(gate_path, "rb") as handle:
        gate_document = json.loads(handle.read().decode("utf-8"))
    gate_module.assert_held_out_admitted(gate_document)

    index = frozen.load_index(os.path.abspath(os.path.expanduser(args.index)))
    artifacts_directory = os.path.dirname(os.path.abspath(os.path.expanduser(args.index)))
    identity = frozen.load_identity(index, artifacts_directory)
    per_seed, nested_records = collect(out_root, identity)
    result = analysis_module.analyse(
        per_seed["SI"],
        per_seed["KS"],
        delta=args.delta,
        nested=analysis_module.nested_description(nested_records),
    )
    destination = args.analysis_out or os.path.join(out_root, "analysis.json")
    with open(destination, "wb") as handle:
        handle.write(canonical_json_bytes(result))
    print("DG0008_CLASS", result["classification"])


if __name__ == "__main__":
    main()
