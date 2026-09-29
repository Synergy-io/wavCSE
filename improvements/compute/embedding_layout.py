"""Turn a job's declared embedding artifacts into the tree the loader actually reads.

The control plane verifies and materializes declared inputs, but it places each one at a
declared *workspace file*: it has no idea that the training loader wants an unpacked tree
at ``<root_emb_path>/<upstream_model_type>/<frame_pool_id>/<dataset>/``. "A verified
artifact exists" therefore says nothing about whether the run consumed it — and a run that
silently read the worker's leftover ``~/embedding`` would produce a plausible result from
the wrong bytes.

This module closes that gap with an explicit, deterministic mapping, executed by the job
wrapper inside the checkout at the exact commit the job names:

    declared artifact (digest-verified by the control plane)
        -> re-verified on the worker against the plan's own digest
        -> extracted to the loader-relative path the experiment derives
        -> an explicitly configured root (``WAVCSE_ROOT_EMB_PATH`` + layout marker)
        -> the loader, which refuses any root without that marker

Nothing here guesses. The loader-relative path is derived from the *arm's own config*
(model type, frame pooling) and from ``downstream``'s own task-to-dataset mapping, so the
prepared layout cannot drift from what the loader computes. An archive whose contents are
not the dataset it claims, an input whose bytes do not match the declared digest, a
missing dataset for the protocol's task set, or a pre-existing directory where the layout
belongs all fail the job before training starts.

The archive form is the one canonical storage already uses for embeddings: a plain TAR of
the dataset directory, so extracting it beneath the pooling directory yields exactly the
layout wavCSE expects.
"""

import hashlib
import json
import os
import shutil
import sys
import tarfile

from improvements import embedding_root
from improvements.compute import artifacts, jobspec
from improvements.compute.errors import ConfigurationError, EvidenceError, ImplementationBugError

_CHUNK = 1 << 20
_DOWNSTREAM_RELATIVE = "downstream"


class EmbeddingLayoutError(EvidenceError):
    """The job's embedding inputs cannot be turned into the loader's layout."""


def _downstream_import(name):
    """Import one module from the frozen ``downstream`` package by its own path root."""

    root = os.path.join(os.path.abspath(os.getcwd()), _DOWNSTREAM_RELATIVE)
    if not os.path.isdir(root):
        raise EmbeddingLayoutError(
            "the checkout at {} has no {} directory, so the loader's dataset contract "
            "cannot be derived".format(os.getcwd(), _DOWNSTREAM_RELATIVE)
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    module = __import__(name, fromlist=["*"])
    return module


def required_datasets(plan):
    """The dataset directories the protocol's task tokens resolve to.

    The mapping is imported from ``downstream`` rather than restated here: the prepared
    layout must be the same layout the loader asks for, and the only way to guarantee that
    is to ask the same code.
    """

    mapping = _downstream_import("utils.constant_mapping").TaskDatasetMapping
    tokens = [token for token in str(plan["task_type"]).split("_") if token.strip()]
    if not tokens:
        raise ConfigurationError("the plan has no task vocabulary to map to datasets")
    datasets = []
    for token in tokens:
        key = mapping.get_dataset_key(token)
        if not key:
            raise ConfigurationError(
                "task token {!r} maps to no dataset in the loader's own mapping".format(token)
            )
        if key not in datasets:
            datasets.append(key)
    return datasets


def loader_pooling(config_path):
    """The loader-relative path components, taken from the arm's own config."""

    import yaml

    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except OSError as exc:
        raise ConfigurationError(
            "the arm config {} could not be read: {}".format(config_path, exc)
        ) from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            "the arm config {} is not valid YAML: {}".format(config_path, exc)
        ) from exc
    if not isinstance(config, dict):
        raise ConfigurationError("the arm config {} is not a mapping".format(config_path))
    upstream = config.get("upstream") or {}
    pooling = config.get("pooling") or {}
    model_type = upstream.get("model_type")
    frame_type = pooling.get("frame_pooling_type")
    if not model_type or not frame_type:
        raise ConfigurationError(
            "the arm config {} does not name upstream.model_type and "
            "pooling.frame_pooling_type, which the loader uses to locate embeddings"
            .format(config_path)
        )
    make_pooling_id = _downstream_import("utils.pooling_id").make_pooling_id
    pool_id = make_pooling_id(frame_type, pooling.get("frame_pooling_param"))
    return str(model_type), str(pool_id)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_materialized(path, requirement, *, where):
    """Prove the file the control plane placed is the bytes the plan declared."""

    if not os.path.isfile(path):
        raise EmbeddingLayoutError(
            "{}: the declared input {} was not materialized at {}".format(
                where, requirement["artifact"], path
            )
        )
    actual = _sha256(path)
    if actual != str(requirement["sha256"]):
        raise EmbeddingLayoutError(
            "{}: the materialized input {} has sha256 {}, not the declared {}".format(
                where, requirement["artifact"], actual, requirement["sha256"]
            )
        )
    size = os.path.getsize(path)
    if requirement.get("size_bytes") is not None and size != int(requirement["size_bytes"]):
        raise EmbeddingLayoutError(
            "{}: the materialized input {} is {} bytes, not the declared {}".format(
                where, requirement["artifact"], size, requirement["size_bytes"]
            )
        )
    return size


def extract_dataset(archive, pooling_root, *, dataset):
    """Extract one dataset archive beneath its pooling directory.

    The archive is a plain TAR of the dataset directory (``tar -C <pooling_root>
    <dataset>``), so it is extracted *at* the pooling directory; only regular files and
    directories whose first path segment is the dataset itself are accepted. An archive
    that would write outside its own directory, carry a link, or contribute a differently
    named tree is refused rather than partially extracted.
    """

    destination = os.path.join(pooling_root, dataset)
    if os.path.exists(destination):
        # A tree left by an earlier attempt is never merged with a verified one.
        shutil.rmtree(destination)
    os.makedirs(pooling_root, exist_ok=True)
    extracted = 0
    try:
        with tarfile.open(archive, "r:*") as bundle:
            members = bundle.getmembers()
            if not members:
                raise EmbeddingLayoutError(
                    "the archive for {!r} is empty".format(dataset))
            for member in members:
                name = member.name.replace("\\", "/").lstrip("./")
                if not name:
                    continue
                parts = [part for part in name.split("/") if part]
                if not parts or name.startswith("/") or ".." in parts:
                    raise EmbeddingLayoutError(
                        "the archive for {!r} contains an unsafe member {!r}".format(
                            dataset, member.name))
                if parts[0] != dataset:
                    raise EmbeddingLayoutError(
                        "the archive for {!r} contains {!r}, which belongs to another "
                        "dataset; the layout would not be the one that was verified"
                        .format(dataset, member.name))
                if member.issym() or member.islnk() or member.isdev():
                    raise EmbeddingLayoutError(
                        "the archive for {!r} contains a non-regular member {!r}".format(
                            dataset, member.name))
                if member.isfile():
                    extracted += 1
            bundle.extractall(path=pooling_root)
    except tarfile.TarError as exc:
        raise EmbeddingLayoutError(
            "the archive for {!r} could not be read: {}".format(dataset, exc)) from exc
    if extracted == 0:
        raise EmbeddingLayoutError(
            "the archive for {!r} contains no files, so it cannot be the verified tree"
            .format(dataset))
    return extracted, destination


def prepare(plan, *, stage, arm, seed, checkout_root, job_directory, commit, job_id,
            log=None):
    """Prepare the loader-visible root for one job and return the layout marker.

    Raises before training if anything about the mapping cannot be established.
    """

    layout = plan.get("embedding_layout")
    if not layout:
        raise ConfigurationError(
            "the plan declares no embedding_layout, so a verify-and-consume mapping for "
            "the loader's root cannot be derived"
        )
    arm_spec = jobspec.arm_by_name(plan, arm)
    config_relative = arm_spec.get("config")
    if not config_relative:
        raise ConfigurationError(
            "arm {!r} declares no config, so the loader's embedding path cannot be "
            "derived from the run's own settings".format(arm))
    config_path = os.path.join(checkout_root, config_relative)
    model_type, pool_id = loader_pooling(config_path)

    required = required_datasets(plan)
    declared = {}
    for entry in layout.get("datasets") or ():
        if not isinstance(entry, dict) or not entry.get("dataset") or not entry.get("input"):
            raise ConfigurationError(
                "every embedding_layout dataset needs a dataset name and an input artifact"
            )
        if entry["dataset"] in declared:
            raise ConfigurationError(
                "embedding_layout names dataset {!r} twice".format(entry["dataset"]))
        declared[entry["dataset"]] = entry["input"]
    missing = [name for name in required if name not in declared]
    if missing:
        raise ConfigurationError(
            "embedding_layout does not cover the datasets this protocol loads: {}".format(
                ", ".join(missing)))
    unknown = sorted(set(declared) - set(required))
    if unknown:
        raise ConfigurationError(
            "embedding_layout declares datasets this protocol never loads: {}".format(
                ", ".join(unknown)))

    inputs_file = plan.get("inputs_file")
    if not inputs_file:
        raise ConfigurationError(
            "the plan declares an embedding_layout but no inputs_file, so no declared "
            "artifact can be matched to it"
        )
    requirements = artifacts.load(os.path.join(checkout_root, inputs_file))
    by_artifact = {item["artifact"]: item for item in requirements["requirements"]}

    root = os.path.join(job_directory, str(layout["root"]))
    if os.path.exists(embedding_root.layout_marker_path(root)):
        raise EmbeddingLayoutError(
            "the job directory already holds a prepared embedding layout at {}; a "
            "verified layout is prepared once per job".format(root)
        )
    pooling_root = os.path.join(root, model_type, pool_id)
    prepared = []
    for dataset in required:
        artifact = declared[dataset]
        requirement = by_artifact.get(artifact)
        if requirement is None:
            raise EmbeddingLayoutError(
                "embedding_layout names input {!r} for dataset {!r}, but the plan's "
                "inputs_file does not declare it".format(artifact, dataset))
        source = os.path.join(job_directory, "inputs", requirement["destination"])
        size = verify_materialized(source, requirement, where="embedding layout")
        files, destination = extract_dataset(source, pooling_root, dataset=dataset)
        prepared.append({
            "dataset": dataset,
            "artifact": artifact,
            "destination": requirement["destination"],
            "sha256": requirement["sha256"],
            "size_bytes": size,
            "files": files,
            "extracted_to": destination,
        })
        if log:
            log("embedding layout | {} <- {} ({} file(s), sha256 {}…)".format(
                dataset, artifact, files, requirement["sha256"][:12]))

    if not pool_id or not model_type:
        raise ImplementationBugError("the loader path components are empty")
    marker = embedding_root.mark_prepared(root, {
        "study": plan["study"],
        "stage": stage,
        "arm": arm,
        "seed": int(seed),
        "commit": str(commit),
        "job_id": str(job_id),
        "upstream_model_type": model_type,
        "frame_pool_id": pool_id,
        "loader_root": root,
        "datasets": sorted(prepared, key=lambda entry: entry["dataset"]),
    })
    return marker


def describe(marker):
    """The manifest-facing summary of one prepared layout."""

    return {
        "root": marker["loader_root"],
        "upstream_model_type": marker["upstream_model_type"],
        "frame_pool_id": marker["frame_pool_id"],
        "datasets": [
            {"dataset": entry["dataset"], "artifact": entry["artifact"],
             "sha256": entry["sha256"], "files": entry["files"]}
            for entry in marker["datasets"]
        ],
    }


def marker_text(marker):
    return json.dumps(marker, sort_keys=True, indent=2) + "\n"
