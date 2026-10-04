"""DG-0008 CPU-only, no-training identity/opportunity manifest generator (DP-0008 clause 3-14).

Reads the restored raw corpora's official metadata only — never an embedding
file and never an embedding wrapper's ``__getitem__`` — emits the canonical
``dg0008.example-identity.v1`` object and every ``dg0008.opportunity-manifest.v2``
object, and freezes their digests plus the clause-12 tuple associations.

Stage 0 is zero-GPU.  This script fabricates nothing: it fails closed if a corpus
is absent, a canonical key duplicates, an official split is unexpected or an
IEMOCAP fold cannot be built.

Per-epoch manifests are produced by the accepted
``IF-MANIFEST-IO=STREAM_CANONICAL_BYTES`` path
(``manifest.iter_opportunity_manifest_bytes``): the canonical bytes are
recomputed batch by batch and hashed while written, so the ``steps x 2048``
``step_keys`` matrix is never held in memory.  Each written file is then
re-hashed by a streaming re-read that must equal the digest frozen into the
index.

Usage::

    python generate_opportunity_manifest.py \
        --root-data-path ~/voice_dataset \
        --out improvements/taskrelation/research/studies/DG-0008/artifacts
"""

import argparse
import hashlib
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
for _path in (_REPO_ROOT, os.path.join(_REPO_ROOT, "downstream")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from utils.constant_mapping import LabelKeywordMapping  # noqa: E402

from improvements.taskrelation.research.dg0008 import CELL_COMPONENTS  # noqa: E402
from improvements.taskrelation.research.dg0008.manifest import (  # noqa: E402
    BATCH_SIZE,
    EPOCHS,
    FOLDS,
    SEEDS,
    ManifestError,
    all_matrix_tuples,
    attach_fold_membership,
    build_identity_object,
    build_iemocap_folds,
    cell_vectors,
    identity_bytes,
    identity_digest,
    iemocap_record,
    iter_opportunity_manifest_bytes,
    length_and_steps,
    matrix_array,
    matrix_digest,
    run_array,
    run_digest,
    speechcommand_record,
    voxceleb_record,
)

SC_SUBSETS = {"training": "training", "validation": "validation", "testing": "testing"}
VOX_SUBSETS = ("train", "dev", "test")


def extract_speechcommand(root, label2index):
    from torchaudio.datasets import SPEECHCOMMANDS

    records = []
    for official_split, subset in SC_SUBSETS.items():
        reader = SPEECHCOMMANDS(
            root=root,
            url="speech_commands_v0.01",
            folder_in_archive="SpeechCommands",
            download=False,
            subset=subset,
        )
        for index in range(len(reader)):
            metadata = reader.get_metadata(index)
            record = speechcommand_record(metadata[0], metadata[2], label2index)
            record["official_split"] = official_split
            records.append(record)
    return records


def extract_voxceleb(root, label2index):
    from torchaudio.datasets import VoxCeleb1Identification

    meta_url = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/iden_split.txt"
    records = []
    for official_split in VOX_SUBSETS:
        reader = VoxCeleb1Identification(root, official_split, meta_url, download=False)
        for index in range(len(reader)):
            metadata = reader.get_metadata(index)
            record = voxceleb_record(metadata[0], metadata[2], label2index)
            record["official_split"] = official_split
            records.append(record)
    return records


def extract_iemocap(root, label2index):
    from torchaudio.datasets import IEMOCAP

    reader = IEMOCAP(root, ("1", "2", "3", "4", "5"))
    records = []
    for wav_stem in reader.data:
        mapping = reader.mapping[wav_stem]
        if mapping["label"] == "fru":
            continue
        records.append(iemocap_record(mapping["path"], mapping["label"], label2index))
    return records


def build_identity(root_data_path, extractors=None):
    label2index_sc = LabelKeywordMapping.LABEL2INDEX_SPEECHCOMMANDv1
    label2index_vox = LabelKeywordMapping.LABEL2INDEX_VOXCELEB1
    label2index_er = LabelKeywordMapping.LABEL2INDEX_IEMOCAP

    if extractors is None:
        extractors = {
            "speechcommand": extract_speechcommand,
            "voxceleb": extract_voxceleb,
            "iemocap": extract_iemocap,
        }
    sc_records = extractors["speechcommand"](
        os.path.join(root_data_path, "speechcommand"), label2index_sc
    )
    vox_records = extractors["voxceleb"](
        os.path.join(root_data_path, "voxceleb"), label2index_vox
    )
    er_records = extractors["iemocap"](
        os.path.join(root_data_path, "iemocap"), label2index_er
    )
    if not sc_records or not vox_records or not er_records:
        raise ManifestError("a restored corpus produced no metadata records")
    folds = build_iemocap_folds(record["speaker_id"] for record in er_records)
    er_records = attach_fold_membership(er_records, folds)
    identity = build_identity_object(sc_records, vox_records, er_records)
    if len(identity["iemocap_folds"]) != len(FOLDS):
        raise ManifestError("IEMOCAP fold construction did not yield ten folds")
    return identity


def _file_sha256(path, chunk_size=1024 * 1024):
    """Streaming SHA-256 of a file's bytes: the bounded-memory re-read check."""

    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate(root_data_path, out):
    identity = build_identity(root_data_path)
    payload = identity_bytes(identity)
    digest = identity_digest(identity)
    os.makedirs(out, exist_ok=True)
    identity_path = os.path.join(out, "identity.json")
    with open(identity_path, "wb") as handle:
        handle.write(payload)

    index = {
        "schema": "dg0008.manifest-index.v1",
        "identity_digest": digest,
        "batch_size": BATCH_SIZE,
        "l": {},
        "s": {},
        "epochs": {},
        "runs": {},
        "run_arrays": {},
        "epoch_paths": {},
    }

    for cell in sorted(CELL_COMPONENTS):
        for fold in FOLDS:
            train_vectors = cell_vectors(identity, cell, fold, "train")
            length, steps = length_and_steps(*[len(keys) for keys, _ in train_vectors])
            if steps < 1:
                raise ManifestError(
                    "cell {} fold {} has no complete batch (L={})".format(cell, fold, length)
                )
            index["l"]["{}|{}".format(cell, fold)] = length
            index["s"]["{}|{}".format(cell, fold)] = steps
            keys_by_component = [keys for keys, _ in train_vectors]
            for seed in SEEDS:
                epoch_digests = []
                for epoch in range(EPOCHS):
                    # Accepted IF-MANIFEST-IO=STREAM_CANONICAL_BYTES: the per-epoch
                    # canonical bytes are recomputed batch by batch and hashed while
                    # written, so the step_keys matrix and the joined JSON document
                    # are never held in memory.  The bytes are the clause-5/11
                    # canonical ones; the test suite pins them to the materializing
                    # oracle object.
                    relative = os.path.join(
                        "manifests", cell, "f{}".format(fold), "s{}".format(seed),
                        "e{}.json".format(epoch),
                    )
                    target = os.path.join(out, relative)
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    hasher = hashlib.sha256()
                    with open(target, "wb") as handle:
                        for chunk in iter_opportunity_manifest_bytes(
                                cell, fold, seed, epoch, keys_by_component):
                            handle.write(chunk)
                            hasher.update(chunk)
                    streamed = hasher.hexdigest()
                    reread = _file_sha256(target)
                    if reread != streamed:
                        raise ManifestError(
                            "streamed manifest digest mismatch after write at "
                            "{}|{}|{}|{}: {} != {}".format(
                                cell, fold, seed, epoch, reread, streamed
                            )
                        )
                    index["epochs"]["{}|{}|{}|{}".format(cell, fold, seed, epoch)] = streamed
                    index["epoch_paths"]["{}|{}|{}|{}".format(cell, fold, seed, epoch)] = relative
                    epoch_digests.append(streamed)
                run = "{}|{}|{}".format(cell, fold, seed)
                index["runs"][run] = run_digest(cell, fold, seed, epoch_digests)
                index["run_arrays"][run] = run_array(cell, fold, seed, epoch_digests)

    entries = [
        [cell, fold, seed, epoch, index["epochs"]["{}|{}|{}|{}".format(cell, fold, seed, epoch)]]
        for (cell, fold, seed, epoch) in all_matrix_tuples()
    ]
    index["matrix_digest"] = matrix_digest(entries)
    index["matrix_array"] = matrix_array(entries)

    index_path = os.path.join(out, "index.json")
    with open(index_path, "wb") as handle:
        handle.write(
            json.dumps(
                index, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        )
    print("IDENTITY_DIGEST", digest)
    print("MATRIX_DIGEST", index["matrix_digest"])
    print("MANIFEST_EPOCHS", len(index["epochs"]))
    return index


def main():
    parser = argparse.ArgumentParser(description="DG-0008 identity/opportunity manifest generator")
    parser.add_argument("--root-data-path", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    generate(os.path.abspath(os.path.expanduser(args.root_data_path)),
             os.path.abspath(os.path.expanduser(args.out)))


if __name__ == "__main__":
    main()
