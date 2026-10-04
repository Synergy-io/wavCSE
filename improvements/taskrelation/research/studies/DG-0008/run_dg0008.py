"""DG-0008 fixed-LR five-epoch training entrypoint (one matrix member / repeat).

One invocation trains exactly one ``(cell, arm, fold, seed)`` member for five
epochs from the frozen opportunity manifests.  It never reads the held-out ER
test split; the endpoint is produced only by ``evaluate_dg0008.py`` after the
Stage-1 validity gate passes.

Every member opens a real MLflow run *before* training, named and tagged from
``configs/dg0008.yaml``'s ``research`` block (``AGENTS.md`` invariant 12).
Tracking failure aborts the member rather than letting it train without
provenance, and the finished run publishes the per-epoch history, the
fixed-final validation ER accuracy, the identity/manifest digests and the
per-step loss/LR/ER-count traces.

Usage::

    python run_dg0008.py --cell si_er --arm control --fold 0 --seed 0 \
        --config configs/dg0008.yaml --manifest-index artifacts/index.json --out outputs
"""

import argparse
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

import yaml  # noqa: E402

from improvements import embedding_root  # noqa: E402
from improvements import mlflow_utils  # noqa: E402
from improvements.device_utils import assert_training_device  # noqa: E402
from improvements.seed_utils import set_seed  # noqa: E402
from improvements.taskrelation.research.dg0008 import (
    CELL_COMPONENTS,
    trainer as training,  # noqa: E402
)
from improvements.taskrelation.research.dg0008 import artifacts as frozen  # noqa: E402
from improvements.taskrelation.research.dg0008.manifest import (  # noqa: E402
    ManifestError,
    canonical_json_bytes,
    cell_vectors,
)
from improvements.taskrelation.research.dg0008.sampler import (  # noqa: E402
    key_to_index,
    plain_dataset,
)
from model.downstream_model import DownstreamMultiTaskModel  # noqa: E402
from utils.parse_transformer_layers import parse_transformer_layers  # noqa: E402
from utils.pooling_id import make_pooling_id  # noqa: E402
from utils.setup_device import set_device  # noqa: E402


def _load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _assert_regime(cfg):
    training = cfg["training"]
    regime = cfg["dg0008"]
    expected = {
        "num_epochs": 5,
        "batch_size": 2048,
        "learning_rate": 0.0025,
        "weight_decay": 0.00000005,
    }
    for key, value in expected.items():
        if training[key] != value:
            raise ManifestError("training.{}={!r} violates the DG-0008 regime".format(key, training[key]))
    if training.get("scheduler_patience") is not None or training.get("task_sampling_weights") is not None:
        raise ManifestError("the DG-0008 regime has no scheduler or task weighting")
    if int(cfg["dataset"]["subset_percentage"]) != 100:
        raise ManifestError("subset_percentage must stay 100")
    if float(regime["target_coefficient"]) != 0.5 or float(regime["aux_coefficient"]) != 0.5:
        raise ManifestError("the cell coefficients are fixed at (0.5, 0.5) vs (0.5, 0)")
    if regime["target_task"] != "er":
        raise ManifestError("the sole scientific endpoint is the fixed-final-epoch ER accuracy")


# The trained architecture is the unmodified downstream baseline under the
# exact-opportunity regime, so category/model mirror DG-0007's baseline arm and
# the diagnostic groups with the wavcse-baseline experiment it is derived from.
MEMBER_CATEGORY = "base"
MEMBER_MODEL = "original"


def member_method(research_cfg, arm):
    """Resolve the committed per-arm research method from ``research.methods``."""

    methods = research_cfg.get("methods") or {}
    method = methods.get(arm) or research_cfg.get("method")
    if not method:
        raise ManifestError(
            "no research.method configured for arm {!r}".format(arm)
        )
    return str(method)


def open_member_run(member_config, method, cell):
    """Set up tracking and open this member's MLflow run; any failure is fatal.

    ``AGENTS.md`` invariant 12 makes provenance mandatory, so tracking is
    started with no fallback: if the tracking URI or experiment cannot be
    reached, the member aborts instead of training anonymously.
    """

    mlflow_utils.setup_mlflow(member_config)
    run_name = mlflow_utils.build_research_run_name(member_config, method, cell)
    if not run_name:
        raise ManifestError("DG-0008 requires a study-scoped MLflow run name")
    return mlflow_utils.mlflow.start_run(run_name=run_name)


def _write_member_records(*, member_root, out_root, trainer, cell, arm, fold, seed,
                          index, run_digests, run_manifest_digest, epoch_manifests,
                          method, implementation_commit):
    """Write the member's frozen records, provenance and per-step traces.

    ``provenance.json`` carries the identity/run/epoch digests and
    ``step_traces.json`` the per-step loss/LR/ER-count traces; the caller
    publishes both as MLflow artifacts.
    """

    record = {
        "schema": "dg0008.run-record.v1",
        "cell": cell,
        "arm": arm,
        "fold": fold,
        "seed": seed,
        "aux_loss_enabled": bool(arm == "pair"),
        "target_coefficient": training.TARGET_COEFFICIENT,
        "identity_digest": index["identity_digest"],
        "run_manifest_digest": run_manifest_digest,
        "epoch_digests": run_digests,
        "step_counts": [int(manifest["steps"]) for manifest in epoch_manifests],
        "consumed_n_er": trainer.consumed_n_er,
        "er_counts_match": bool(trainer.er_counts_match),
        "loss_trace": trainer.step_loss_trace,
        "lr_sequence": trainer.step_lr_trace,
        "initialization_sha256": trainer.initialization_digest,
        "rng_sequence_sha256": training.rng_sequence_digest(trainer.rng_trace),
        "rng_state_count": len(trainer.rng_trace),
        "final_validation_er_accuracy": trainer.final_validation_er_accuracy,
        "fixed_final_checkpoint_sha256": trainer.fixed_final_checkpoint_sha256,
        "fixed_final_checkpoint_path": os.path.relpath(
            trainer.fixed_final_checkpoint_path, out_root
        ),
        "checkpoint_readback_ok": True,
        "implementation_commit": implementation_commit,
        "held_out_read": False,
    }
    provenance = {
        "schema": "dg0008.run-provenance.v1",
        "study_id": "DG-0008",
        "cell": cell,
        "arm": arm,
        "fold": fold,
        "seed": seed,
        "method": method,
        "identity_digest": index["identity_digest"],
        "run_manifest_digest": run_manifest_digest,
        "epoch_digests": list(run_digests),
        "step_counts": record["step_counts"],
        "implementation_commit": implementation_commit,
    }
    traces = {
        "schema": "dg0008.step-traces.v1",
        "cell": cell,
        "arm": arm,
        "fold": fold,
        "seed": seed,
        "step_loss_trace": trainer.step_loss_trace,
        "step_lr_trace": trainer.step_lr_trace,
        "consumed_n_er": trainer.consumed_n_er,
    }
    os.makedirs(member_root, exist_ok=True)
    provenance_path = os.path.join(member_root, "provenance.json")
    traces_path = os.path.join(member_root, "step_traces.json")
    payloads = (
        (os.path.join(member_root, "run.json"), record),
        (
            os.path.join(member_root, "run_identity.json"),
            training.run_identity_record(
                cell=cell, arm=arm, fold=fold, seed=seed,
                run_manifest_digest=run_manifest_digest,
                epoch_digests=run_digests,
                initialization_digest=trainer.initialization_digest,
                implementation_commit=implementation_commit,
            ),
        ),
        (provenance_path, provenance),
        (traces_path, traces),
    )
    for path, payload in payloads:
        with open(path, "wb") as handle:
            handle.write(canonical_json_bytes(payload))
    return record, provenance_path, traces_path


def build_member(cfg, cell, arm, fold, seed, artifacts_directory, index, out_root, device,
                 repeat=False):
    identity = frozen.load_identity(index, artifacts_directory)
    run_digests, run_manifest_digest = frozen.verify_run(index, cell, fold, seed)

    train_vectors = cell_vectors(identity, cell, fold, "train")
    val_vectors = cell_vectors(identity, cell, fold, "validation")

    epoch_manifests = [
        frozen.load_epoch_manifest(index, artifacts_directory, cell, fold, seed, epoch)
        for epoch in range(5)
    ]
    expected_length = int(index["l"]["{}|{}".format(cell, fold)])
    expected_steps = int(index["s"]["{}|{}".format(cell, fold)])
    for epoch, manifest in enumerate(epoch_manifests):
        if (manifest.get("cell") != cell or int(manifest.get("fold")) != fold
                or int(manifest.get("seed")) != seed or int(manifest.get("epoch")) != epoch):
            raise ManifestError("frozen manifest identity mismatch at epoch {}".format(epoch))
        if int(manifest["length"]) != expected_length or int(manifest["steps"]) != expected_steps:
            raise ManifestError("frozen manifest length/steps mismatch at epoch {}".format(epoch))
        if len(manifest["step_keys"]) != expected_steps or len(manifest["n_er"]) != expected_steps:
            raise ManifestError("frozen manifest step/n_er length mismatch at epoch {}".format(epoch))

    upstream_model_type = cfg["upstream"]["model_type"]
    frame_pooling_type = cfg["pooling"]["frame_pooling_type"]
    frame_pooling_param = cfg["pooling"]["frame_pooling_param"]
    frame_pool_id = make_pooling_id(frame_pooling_type, frame_pooling_param)
    transformer_layers = parse_transformer_layers(
        transformer_layers=cfg["upstream"]["selected_transformer_layers"],
        upstream_model_type=upstream_model_type,
    )
    root_emb = embedding_root.resolve_root(cfg["paths"]["root_emb_path"])
    dataset_embedding_root = os.path.join(root_emb, upstream_model_type, frame_pool_id)

    train_dataset = plain_dataset(
        train_vectors, cell, embedding_root=dataset_embedding_root,
        upstream_model_type=upstream_model_type, frame_pool_id=frame_pool_id,
        transformer_layer_array=transformer_layers,
    )
    val_dataset = plain_dataset(
        val_vectors, cell, embedding_root=dataset_embedding_root,
        upstream_model_type=upstream_model_type, frame_pool_id=frame_pool_id,
        transformer_layer_array=transformer_layers,
    )

    model = DownstreamMultiTaskModel(
        upstream_model_type=upstream_model_type,
        task_type=cell,
        embedding_dim_shared1=cfg["model"]["embedding_dim_shared1"],
        embedding_dim_shared2=cfg["model"]["embedding_dim_shared2"],
        layer_pooling_type=cfg["pooling"]["layer_pooling_type"],
        layer_pooling_param=cfg["pooling"]["layer_pooling_param"],
        dropout_prob_shared1=cfg["model"]["dropout_prob_shared1"],
        dropout_prob_shared2=cfg["model"]["dropout_prob_shared2"],
    )
    model.to(device)

    bucket = "repeat" if repeat else "runs"
    member_root = os.path.join(out_root, bucket, cell, arm, "f{}".format(fold), "s{}".format(seed))
    results_dir = os.path.join(member_root, "results")
    checkpoints_dir = os.path.join(member_root, "checkpoints")
    training.refuse_resume(member_root)

    # The member's canonical identity: the per-arm method is resolved from the
    # committed config, and the CLI seed/cell are written back so the MLflow run
    # name and tags describe the member that is actually trained.
    member_config = dict(cfg)
    member_config["seed"] = seed
    member_config["research"] = dict(cfg["research"])
    method = member_method(member_config["research"], arm)
    member_config["research"]["method"] = method
    member_config["research"]["task_set"] = cell

    trainer = training.DG0008Trainer(
        model=model,
        device=device,
        task_type=cell,
        training_cfg=cfg["training"],
        results_root=results_dir,
        checkpoints_root=checkpoints_dir,
        training_data=train_dataset,
        validation_data=val_dataset,
        ignore_index=cfg["dataset"]["ignore_index"],
        seed=seed,
        aux_loss_enabled=(arm == "pair"),
        epoch_steps=[manifest["step_keys"] for manifest in epoch_manifests],
        target_task=cfg["dg0008"]["target_task"],
    )
    trainer.bind_key_index(
        key_to_index(train_vectors), [manifest["n_er"] for manifest in epoch_manifests]
    )

    # Provenance is mandatory (AGENTS.md invariant 12): open the real run with no
    # fallback, then train and publish every record inside it.
    with open_member_run(member_config, method, cell):
        mlflow_utils.set_standard_tags(
            MEMBER_CATEGORY, MEMBER_MODEL, member_config,
            extra_tags={"task_set": cell},
        )
        mlflow_utils.log_config_params(member_config)
        trainer.train()

        implementation_commit = mlflow_utils.resolve_git_commit()
        record, provenance_path, traces_path = _write_member_records(
            member_root=member_root, out_root=out_root, trainer=trainer,
            cell=cell, arm=arm, fold=fold, seed=seed, index=index,
            run_digests=run_digests, run_manifest_digest=run_manifest_digest,
            epoch_manifests=epoch_manifests, method=method,
            implementation_commit=implementation_commit,
        )
        mlflow_utils.log_trainer_history(trainer)
        mlflow_utils.mlflow.log_metric(
            "final_validation_er_accuracy", record["final_validation_er_accuracy"]
        )
        mlflow_utils.mlflow.log_artifact(provenance_path)
        mlflow_utils.mlflow.log_artifact(traces_path)

    print("DG0008_RUN_OK", cell, arm, fold, seed, run_manifest_digest)
    return record


def main():
    parser = argparse.ArgumentParser(description="DG-0008 training member")
    parser.add_argument("--cell", required=True, choices=sorted(CELL_COMPONENTS))
    parser.add_argument("--arm", required=True, choices=("pair", "control"))
    parser.add_argument("--fold", required=True, type=int, choices=range(10))
    parser.add_argument("--seed", required=True, type=int, choices=range(5))
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest-index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device_index", type=int, default=None)
    parser.add_argument("--repeat", action="store_true",
                        help="si_er control fold 0 seed 0 same-arm determinism repeat")
    args = parser.parse_args()
    if args.repeat and (args.cell, args.arm, args.fold, args.seed) != ("si_er", "control", 0, 0):
        parser.error("--repeat is reserved for si_er/control/fold=0/seed=0")

    cfg = _load_config(args.config)
    _assert_regime(cfg)
    set_seed(args.seed)
    training.apply_determinism()

    device_type = cfg["device"]["type"]
    device_index = args.device_index if args.device_index is not None else cfg["device"]["index"]
    device = set_device(device_type=device_type, device_index=device_index)
    assert_training_device(device_type, device_index, device, context="run_dg0008")

    index_path = os.path.abspath(os.path.expanduser(args.manifest_index))
    index = frozen.load_index(index_path)
    artifacts_directory = os.path.dirname(index_path)
    out_root = os.path.abspath(os.path.expanduser(args.out))
    build_member(cfg, args.cell, args.arm, args.fold, args.seed, artifacts_directory,
                 index, out_root, device, repeat=args.repeat)


if __name__ == "__main__":
    main()
