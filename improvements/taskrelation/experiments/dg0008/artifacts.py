"""Load and verify the frozen DG-0008 identity/opportunity artifacts, fail closed.

Every consumer (training, validity gate, delayed evaluation, analysis) reads the
frozen bytes through here, re-hashing what it reads against the frozen index, so
no consumer can silently use an opportunity schedule or identity that differs
from the one the generator froze.
"""

import json
import os

from .manifest import ManifestError, identity_digest, run_array, run_digest, sha256_hex

INDEX_SCHEMA = "dg0008.manifest-index.v1"


def _read_bytes(path):
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except OSError as exc:
        raise ManifestError("cannot read frozen artifact {}: {}".format(path, exc)) from exc


def load_index(index_path):
    payload = _read_bytes(index_path)
    try:
        index = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ManifestError("manifest index is not UTF-8 JSON: {}".format(exc)) from exc
    if index.get("schema") != INDEX_SCHEMA:
        raise ManifestError("unexpected manifest index schema {!r}".format(index.get("schema")))
    return index


def load_identity(index, artifacts_directory):
    path = os.path.join(artifacts_directory, "identity.json")
    payload = _read_bytes(path)
    try:
        identity = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ManifestError("identity artifact is not UTF-8 JSON: {}".format(exc)) from exc
    actual = sha256_hex(payload)
    if actual != index["identity_digest"]:
        raise ManifestError(
            "identity bytes digest {} != frozen {}".format(actual, index["identity_digest"])
        )
    if identity_digest(identity) != actual:
        raise ManifestError("identity canonical digest does not match its bytes")
    return identity


def load_epoch_manifest(index, artifacts_directory, cell, fold, seed, epoch):
    key = "{}|{}|{}|{}".format(cell, fold, seed, epoch)
    relative = index["epoch_paths"].get(key)
    if relative is None:
        raise ManifestError("index has no frozen opportunity manifest for {}".format(key))
    path = os.path.join(artifacts_directory, relative)
    payload = _read_bytes(path)
    actual = sha256_hex(payload)
    if actual != index["epochs"].get(key):
        raise ManifestError(
            "opportunity manifest {} digest {} != frozen {}".format(
                key, actual, index["epochs"].get(key)
            )
        )
    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ManifestError("opportunity manifest {} is not JSON".format(key)) from exc
    return manifest


def verify_run(index, cell, fold, seed):
    epoch_digests = [index["epochs"]["{}|{}|{}|{}".format(cell, fold, seed, epoch)]
                     for epoch in range(5)]
    expected = run_digest(cell, fold, seed, epoch_digests)
    frozen = index["runs"].get("{}|{}|{}".format(cell, fold, seed))
    if expected != frozen:
        raise ManifestError(
            "run digest {} != frozen {} for {}|{}|{}".format(
                expected, frozen, cell, fold, seed
            )
        )
    frozen_array = index["run_arrays"].get("{}|{}|{}".format(cell, fold, seed))
    if frozen_array != run_array(cell, fold, seed, epoch_digests):
        raise ManifestError("frozen run array does not match the index digests")
    return epoch_digests, expected


def s_envelope(index, cell, fold, minimum, maximum):
    steps = int(index["s"]["{}|{}".format(cell, fold)])
    if not (int(minimum) <= steps <= int(maximum)):
        raise ManifestError(
            "steps {} for {}|{} is outside the DG-0001 sanity envelope".format(
                steps, cell, fold
            )
        )
    return steps
