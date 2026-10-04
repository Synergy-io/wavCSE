"""DG-0008 delayed held-out ER endpoint evaluation (the only test-split reader).

Refuses to run unless the Stage-1 validity gate is valid and no premature read
is recorded; appends every read to ``held_out_reads.jsonl`` before loading a
checkpoint, so a premature read can never go unnoticed.  Repeated invocation is
idempotent: an existing endpoint is returned unchanged.

Usage::

    python evaluate_dg0008.py --cell si_er --arm pair --fold 0 --seed 0 \
        --config configs/dg0008.yaml --manifest-index artifacts/index.json --out outputs
"""

import argparse
import json
import os
import sys

_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_existing_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
if _existing_cublas is not None and _existing_cublas != _CUBLAS_WORKSPACE_CONFIG:
    raise RuntimeError(
        "DG-0008 requires CUBLAS_WORKSPACE_CONFIG={!r} for bitwise reproducibility; "
        "refusing the conflicting value {!r}".format(
            _CUBLAS_WORKSPACE_CONFIG, _existing_cublas
        )
    )
os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG

_REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
        )
    )
)
for _path in (_REPO_ROOT, os.path.join(_REPO_ROOT, "downstream")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch  # noqa: E402
import yaml  # noqa: E402

from improvements.taskrelation.research.dg0008 import artifacts as frozen  # noqa: E402
from improvements.taskrelation.research.dg0008 import gate as gate_module  # noqa: E402
from improvements.taskrelation.research.dg0008.manifest import (  # noqa: E402
    ManifestError,
    canonical_json_bytes,
    records_for_split,
    sha256_hex,
)
from improvements.taskrelation.research.dg0008.sampler import (  # noqa: E402
    ManifestDataset,
    component_directories,
)
from improvements.taskrelation.research.dg0008.trainer import (  # noqa: E402
    apply_determinism,
    file_sha256,
)
from improvements.taskrelation.research.dg0008.trainer import (  # noqa: E402
    TARGET_COEFFICIENT,
)
from improvements import embedding_root  # noqa: E402
from model.downstream_model import DownstreamMultiTaskModel  # noqa: E402
from trainer.trainer_utils import masked_accuracy  # noqa: E402
from utils.parse_transformer_layers import parse_transformer_layers  # noqa: E402
from utils.pooling_id import make_pooling_id  # noqa: E402
from utils.setup_device import set_device  # noqa: E402


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _record_read(out_root, payload):
    path = os.path.join(out_root, "held_out_reads.jsonl")
    os.makedirs(out_root, exist_ok=True)
    with open(path, "ab") as handle:
        handle.write(canonical_json_bytes(payload) + b"\n")


def _held_out_read_count(out_root):
    path = os.path.join(out_root, "held_out_reads.jsonl")
    if not os.path.exists(path):
        return 0
    with open(path, "rb") as handle:
        return sum(1 for line in handle if line.strip())


def _load_gate(out_root):
    path = os.path.join(out_root, "gate.json")
    if not os.path.exists(path):
        raise ManifestError("held-out evaluation requires {}: run check_validity_gate.py".format(path))
    with open(path, "rb") as handle:
        document = json.loads(handle.read().decode("utf-8"))
    gate_module.assert_held_out_admitted(document)
    return document


def er_test_dataset(identity, cell, cfg, fold):
    keys, labels = records_for_split(identity, "iemocap", fold, "test")
    upstream_model_type = cfg["upstream"]["model_type"]
    frame_pool_id = make_pooling_id(
        cfg["pooling"]["frame_pooling_type"], cfg["pooling"]["frame_pooling_param"]
    )
    root_emb = embedding_root.resolve_root(cfg["paths"]["root_emb_path"])
    transformer_layers = parse_transformer_layers(
        transformer_layers=cfg["upstream"]["selected_transformer_layers"],
        upstream_model_type=upstream_model_type,
    )
    return ManifestDataset(
        [(1, key) for key in keys],
        labels,
        ["10", "01"],
        embedding_root=os.path.join(root_emb, upstream_model_type, frame_pool_id),
        component_dirs=component_directories(cell),
        upstream_model_type=upstream_model_type,
        frame_pool_id=frame_pool_id,
        transformer_layer_array=transformer_layers,
    )


def main():
    parser = argparse.ArgumentParser(description="DG-0008 delayed held-out ER evaluation")
    parser.add_argument("--cell", required=True, choices=("ks_er", "si_er"))
    parser.add_argument("--arm", required=True, choices=("pair", "control"))
    parser.add_argument("--fold", required=True, type=int, choices=range(10))
    parser.add_argument("--seed", required=True, type=int, choices=range(5))
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest-index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device_index", type=int, default=None)
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    apply_determinism()
    device_type = cfg["device"]["type"]
    device_index = args.device_index if args.device_index is not None else cfg["device"]["index"]
    device = set_device(device_type=device_type, device_index=device_index)

    index_path = os.path.abspath(os.path.expanduser(args.manifest_index))
    artifacts_directory = os.path.dirname(index_path)
    index = frozen.load_index(index_path)
    out_root = os.path.abspath(os.path.expanduser(args.out))

    gate_document = _load_gate(out_root)
    gate_digest = sha256_hex(canonical_json_bytes(gate_document))

    member_root = os.path.join(
        out_root, "runs", args.cell, args.arm, "f{}".format(args.fold), "s{}".format(args.seed)
    )
    run_path = os.path.join(member_root, "run.json")
    if not os.path.exists(run_path):
        raise ManifestError("no stored run record at {}".format(run_path))
    with open(run_path, "rb") as handle:
        run_record = json.loads(handle.read().decode("utf-8"))

    endpoint_directory = os.path.join(
        out_root, "endpoints", args.cell, args.arm, "f{}".format(args.fold), "s{}".format(args.seed)
    )
    endpoint_path = os.path.join(endpoint_directory, "endpoint.json")
    if os.path.exists(endpoint_path):
        print("DG0008_ENDPOINT_EXISTS", endpoint_path)
        return

    _record_read(
        out_root,
        {
            "cell": args.cell,
            "arm": args.arm,
            "fold": args.fold,
            "seed": args.seed,
            "gate_digest": gate_digest,
            "run_manifest_digest": run_record["run_manifest_digest"],
        },
    )

    identity = frozen.load_identity(index, artifacts_directory)
    if run_record["identity_digest"] != index["identity_digest"]:
        raise ManifestError("stored run identity does not match the frozen identity")

    dataset = er_test_dataset(identity, args.cell, cfg, args.fold)
    model = DownstreamMultiTaskModel(
        upstream_model_type=cfg["upstream"]["model_type"],
        task_type=args.cell,
        embedding_dim_shared1=cfg["model"]["embedding_dim_shared1"],
        embedding_dim_shared2=cfg["model"]["embedding_dim_shared2"],
        layer_pooling_type=cfg["pooling"]["layer_pooling_type"],
        layer_pooling_param=cfg["pooling"]["layer_pooling_param"],
        dropout_prob_shared1=cfg["model"]["dropout_prob_shared1"],
        dropout_prob_shared2=cfg["model"]["dropout_prob_shared2"],
    )
    checkpoint_path = os.path.join(out_root, run_record["fixed_final_checkpoint_path"])
    if file_sha256(checkpoint_path) != run_record["fixed_final_checkpoint_sha256"]:
        raise ManifestError("fixed-final checkpoint digest changed since training")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.to(device)
    model.eval()

    target_index = ["ks", "er"].index("er") if args.cell == "ks_er" else ["si", "er"].index("er")
    correct = 0
    total = 0
    order = list(range(len(dataset)))
    with torch.no_grad():
        for start in range(0, len(order), 2048):
            batch = [dataset[index] for index in order[start:start + 2048]]
            inputs = torch.stack([item[0] for item in batch]).to(device)
            labels = torch.stack([item[1] for item in batch]).long().to(device)
            outputs = model(input_seq=inputs)
            prediction = outputs.prediction[target_index]
            c, s = masked_accuracy(prediction, labels[:, target_index], ignore_index=-1)
            correct += c
            total += s
    accuracy = (correct / total) if total else 0.0

    os.makedirs(endpoint_directory, exist_ok=True)
    endpoint = {
        "schema": "dg0008.endpoint.v1",
        "cell": args.cell,
        "arm": args.arm,
        "fold": args.fold,
        "seed": args.seed,
        "target_coefficient": TARGET_COEFFICIENT,
        "er_accuracy": accuracy,
        "er_examples": total,
        "run_manifest_digest": run_record["run_manifest_digest"],
        "checkpoint_sha256": run_record["fixed_final_checkpoint_sha256"],
        "gate_digest": gate_digest,
    }
    with open(endpoint_path, "wb") as handle:
        handle.write(canonical_json_bytes(endpoint))
    print("DG0008_ENDPOINT_OK", args.cell, args.arm, args.fold, args.seed, accuracy)


if __name__ == "__main__":
    main()
