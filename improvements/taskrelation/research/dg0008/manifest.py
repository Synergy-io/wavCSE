"""Canonical identity/opportunity manifest contract for DG-0008 (DP-0008 clauses 1-12).

This module is deliberately pure: it depends only on the Python standard
library, so the canonical-key, canonical-JSON, digest-association, fold and
permutation rules can be exercised exactly without any corpus, embedding or
torchaudio reader.  Corpus binding lives in
``studies/DG-0008/generate_opportunity_manifest.py``; everything here is the
declarative contract that the approved proposal fixes byte for byte.

Normative references are clause numbers of
``improvements/taskrelation/research/proposals/DG-0008_exact_matched_directed_transfer.md``.
"""

import hashlib
import json
import re

from . import (
    BATCH_SIZE,
    EPOCHS,
    FOLDS,
    SEEDS,
    SCHEMA_IDENTITY,
    SCHEMA_OPPORTUNITY,
)


class ManifestError(ValueError):
    """One fatal manifest-contract violation; execution stops."""


_DRIVE_QUALIFIED = re.compile(r"^[A-Za-z]:")

# Clause 6: abstract split -> each component's own official token.
OFFICIAL_SPLIT_TOKEN = {
    "speechcommand": {"train": "training", "validation": "validation", "test": "testing"},
    "voxceleb": {"train": "train", "validation": "dev", "test": "test"},
    "iemocap": {"train": "train", "validation": "validation", "test": "test"},
}


# ---------------------------------------------------------------------------
# Clause 2 / 5: canonical keys and canonical JSON bytes
# ---------------------------------------------------------------------------
def canonical_manifest_key(path):
    """Canonicalize one dataset reader's relative path string (clause 2).

    Backslashes become ``/``; NUL, absolute, drive-qualified and empty/``.``/
    ``..`` segments are rejected; segments are re-joined unchanged.  There is
    no filesystem resolution, symlink dereference, case fold, percent decode or
    Unicode normalization.
    """

    if not isinstance(path, str) or path == "":
        raise ManifestError("canonical key requires a non-empty string path")
    if "\x00" in path:
        raise ManifestError("canonical key rejects a NUL byte: {!r}".format(path))
    text = path.replace("\\", "/")
    if text.startswith("/"):
        raise ManifestError("canonical key rejects an absolute path: {!r}".format(path))
    if _DRIVE_QUALIFIED.match(text):
        raise ManifestError("canonical key rejects a drive-qualified path: {!r}".format(path))
    segments = text.split("/")
    for segment in segments:
        if segment in ("", ".", ".."):
            raise ManifestError(
                "canonical key rejects empty/. /.. segments: {!r}".format(path)
            )
    return "/".join(segments)


def key_sort_token(key):
    """Clause 2: ascending lexicographic order of the key's UTF-8 bytes."""

    return key.encode("utf-8")


def sort_keys(keys):
    return sorted(keys, key=key_sort_token)


def canonical_json_bytes(value):
    """``J(x)`` of clauses 5 and 11: canonical JSON bytes, no BOM/newline."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(payload):
    return hashlib.sha256(payload).hexdigest()


def _json_string(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


# ---------------------------------------------------------------------------
# Clause 4: identity records and folds
# ---------------------------------------------------------------------------
def speechcommand_record(path, source_label, label2index):
    key = canonical_manifest_key(path)
    source_label = str(source_label)
    effective = source_label if source_label in label2index else "_unknown_"
    return {
        "key": key,
        "source_label": source_label,
        "effective_label": effective,
        "effective_label_index": int(label2index[effective]),
    }


def voxceleb_record(path, speaker_id, label2index):
    key = canonical_manifest_key(path)
    speaker = str(speaker_id)
    index = label2index.get(speaker, -1)
    return {
        "key": key,
        "effective_speaker_id": speaker,
        "effective_label_index": int(index),
    }


def _iemocap_speaker_id(key):
    parts = key.split("/")
    if len(parts) < 4:
        raise ManifestError("IEMOCAP key has no fourth path component: {!r}".format(key))
    return parts[3].split("_")[0]


def iemocap_record(path, source_label, label2index):
    key = canonical_manifest_key(path)
    source_label = str(source_label)
    if source_label == "fru":
        raise ManifestError("fru must be dropped before identity construction")
    effective = "hap" if source_label == "exc" else source_label
    if effective not in label2index:
        raise ManifestError("IEMOCAP label {!r} is not a mapped class".format(source_label))
    return {
        "key": key,
        "source_label": source_label,
        "effective_label": effective,
        "effective_label_index": int(label2index[effective]),
        "speaker_id": _iemocap_speaker_id(key),
    }


def build_iemocap_folds(speakers):
    """Clause 4 ``FOLD`` list: speakers sorted by UTF-8 bytes, fold 0..9.

    Fold ``f``: ``test = speakers[f]``, ``validation = speakers[(f+1) % 10]``,
    matching ``improvements/base/kfold_iemocap.py::build_loso_fold``.
    """

    ordered = sort_keys(set(str(speaker) for speaker in speakers))
    if len(ordered) != len(FOLDS):
        raise ManifestError(
            "expected {} IEMOCAP speakers, found {}: {}".format(
                len(FOLDS), len(ordered), ordered
            )
        )
    return [
        {
            "fold": fold,
            "test_speaker": ordered[fold],
            "validation_speaker": ordered[(fold + 1) % len(FOLDS)],
        }
        for fold in FOLDS
    ]


def fold_membership(speaker_id, folds):
    """Per-record membership for every fold (clause 4 ``ER_RECORD``)."""

    membership = []
    for entry in folds:
        if speaker_id == entry["test_speaker"]:
            split = "test"
        elif speaker_id == entry["validation_speaker"]:
            split = "validation"
        else:
            split = "train"
        membership.append({"fold": int(entry["fold"]), "split": split})
    return membership


def attach_fold_membership(records, folds):
    enriched = []
    for record in records:
        entry = dict(record)
        entry["fold_membership"] = fold_membership(record["speaker_id"], folds)
        enriched.append(entry)
    return enriched


def _sorted_records(records):
    seen = set()
    for record in records:
        key = record["key"]
        if key in seen:
            raise ManifestError("duplicate canonical key within a component: {!r}".format(key))
        seen.add(key)
    return sorted(records, key=lambda record: key_sort_token(record["key"]))


def build_identity_object(speechcommand_records, voxceleb_records, iemocap_records):
    """Clause 4: the exact ``dg0008.example-identity.v1`` object shape."""

    for record in speechcommand_records:
        _require_exact_keys(
            record,
            {"key", "source_label", "effective_label", "effective_label_index",
             "official_split"},
            "SC_RECORD",
        )
    for record in voxceleb_records:
        _require_exact_keys(
            record,
            {"key", "effective_speaker_id", "effective_label_index", "official_split"},
            "VOX_RECORD",
        )
    for record in iemocap_records:
        _require_exact_keys(
            record,
            {"key", "source_label", "effective_label", "effective_label_index",
             "speaker_id", "fold_membership"},
            "ER_RECORD",
        )

    components = [
        {"component": "speechcommand", "records": _sorted_records(speechcommand_records)},
        {"component": "voxceleb", "records": _sorted_records(voxceleb_records)},
        {"component": "iemocap", "records": _sorted_records(iemocap_records)},
    ]
    folds = build_iemocap_folds(record["speaker_id"] for record in iemocap_records)
    return {
        "schema": SCHEMA_IDENTITY,
        "components": components,
        "iemocap_folds": folds,
    }


def _require_exact_keys(value, expected, where):
    if not isinstance(value, dict):
        raise ManifestError("{} must be an object".format(where))
    extra = set(value) - expected
    missing = expected - set(value)
    if extra or missing:
        raise ManifestError(
            "{} keys are wrong: extra={} missing={}".format(
                where, sorted(extra), sorted(missing)
            )
        )


def identity_bytes(identity_object):
    return canonical_json_bytes(identity_object)


def identity_digest(identity_object):
    return sha256_hex(identity_bytes(identity_object))


def assert_identity_bytes(payload, expected_digest):
    """Dual-identity admission, condition (b) (clause 13)."""

    if not isinstance(payload, bytes):
        raise ManifestError("identity payload must be bytes")
    actual = sha256_hex(payload)
    if actual != expected_digest:
        raise ManifestError(
            "identity bytes digest {} != frozen {}".format(actual, expected_digest)
        )
    return actual


# ---------------------------------------------------------------------------
# Clause 6: version-independent ordered vectors
# ---------------------------------------------------------------------------
def component_records(identity_object, component):
    for entry in identity_object["components"]:
        if entry["component"] == component:
            return entry["records"]
    raise ManifestError("identity object has no component {!r}".format(component))


def records_for_split(identity_object, component, fold, split):
    """Ordered ``(keys, label_indices)`` for one component/split (clause 6).

    ``split`` is the abstract ``train`` / ``validation`` / ``test``; each
    component maps it to its own official token (Speech Commands
    ``training``/``testing``, VoxCeleb ``train``/``dev``/``test``, IEMOCAP the
    fold's membership split).  Records are always sorted by canonical-key UTF-8
    bytes, never by reader enumeration order.
    """

    tokens = OFFICIAL_SPLIT_TOKEN.get(component)
    if tokens is None:
        raise ManifestError("unknown component {!r}".format(component))
    if split not in tokens:
        raise ManifestError(
            "unknown split {!r} for component {!r}".format(split, component)
        )
    records = component_records(identity_object, component)
    selected = []
    for record in records:
        if component == "iemocap":
            membership = {
                entry["fold"]: entry["split"] for entry in record["fold_membership"]
            }
            if membership.get(fold) != split:
                continue
        else:
            if record["official_split"] != OFFICIAL_SPLIT_TOKEN[component][split]:
                continue
        selected.append(record)

    selected.sort(key=lambda record: key_sort_token(record["key"]))
    keys = [record["key"] for record in selected]
    labels = [int(record["effective_label_index"]) for record in selected]
    return keys, labels


def cell_vectors(identity_object, cell, fold, split):
    """Ordered vectors for a cell (component order per clause 6)."""

    try:
        components = {
            "ks_er": ("speechcommand", "iemocap"),
            "si_er": ("voxceleb", "iemocap"),
        }[cell]
    except KeyError:
        raise ManifestError("unknown cell {!r}".format(cell))
    return [records_for_split(identity_object, component, fold, split)
            for component in components]


def length_and_steps(*vector_lengths):
    length = int(sum(vector_lengths))
    return length, length // BATCH_SIZE


# ---------------------------------------------------------------------------
# Clause 8 / 9: manifest seed and counter-based keyed Fisher-Yates
# ---------------------------------------------------------------------------
def manifest_seed(cell, fold, seed, epoch):
    return (
        "DG-0008|manifest|" + str(cell) + "|f=" + str(fold)
        + "|s=" + str(seed) + "|e=" + str(epoch)
    )


def permutation(length, seed_text):
    """Counter-based keyed Fisher-Yates over ``range(length)`` (clause 9)."""

    order = list(range(int(length)))
    for i in range(int(length) - 1, 0, -1):
        key = hashlib.sha256(
            ("perm|" + str(seed_text) + "|i=" + str(i)).encode("utf-8")
        ).digest()
        j = int.from_bytes(key[:8], "big") % (i + 1)
        order[i], order[j] = order[j], order[i]
    return order


def _locate(lengths, global_index):
    """Map a global permutation index onto its ``(component, local_index)``."""

    component = 0
    local = global_index
    while local >= lengths[component]:
        local -= lengths[component]
        component += 1
    return component, local


def build_epoch_steps(cell, fold, seed, epoch, keys_by_component):
    """TEST ORACLE ONLY -- clause 10/11 step keys and epoch ER counts.

    This materializes the whole ``steps x 2048`` ``step_keys`` matrix.  It is
    retained purely as the small/medium-input oracle the streaming production
    path is proven against
    (``tests/test_dg0008_opportunity_manifest.py``); the generator and every
    experiment consumer use :func:`iter_opportunity_manifest_bytes` instead.
    Do not put this back on the production path: holding the matrix is exactly
    what the accepted ``IF-MANIFEST-IO=STREAM_CANONICAL_BYTES`` choice forbids.
    """

    lengths = [len(keys) for keys in keys_by_component]
    length, steps = length_and_steps(*lengths)
    order = permutation(length, manifest_seed(cell, fold, seed, epoch))
    step_keys = []
    n_er = []
    iemocap_index = len(keys_by_component) - 1  # iemocap is always the last component
    for step in range(steps):
        batch = []
        er_count = 0
        for global_index in order[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]:
            component, local = _locate(lengths, global_index)
            batch.append([component, keys_by_component[component][local]])
            if component == iemocap_index:
                er_count += 1
        step_keys.append(batch)
        n_er.append(er_count)
    return {
        "length": length,
        "steps": steps,
        "step_keys": step_keys,
        "n_er": n_er,
    }


def build_opportunity_object(cell, fold, seed, epoch, keys_by_component):
    """TEST ORACLE ONLY -- clause 11 ``dg0008.opportunity-manifest.v2`` object.

    Materialized companion of :func:`build_epoch_steps`, kept so the streaming
    bytes can be compared against ``canonical_json_bytes`` of the exact object
    shape on small/medium inputs.  Never used in production.
    """

    built = build_epoch_steps(cell, fold, seed, epoch, keys_by_component)
    return {
        "cell": str(cell),
        "fold": int(fold),
        "seed": int(seed),
        "epoch": int(epoch),
        "length": int(built["length"]),
        "steps": int(built["steps"]),
        "step_keys": built["step_keys"],
        "n_er": built["n_er"],
        "schema": SCHEMA_OPPORTUNITY,
    }


def iter_opportunity_manifest_bytes(cell, fold, seed, epoch, keys_by_component):
    """Stream one epoch's canonical manifest bytes (accepted ``IF-MANIFEST-IO``).

    This is the production producer behind the flexible choice
    ``STREAM_CANONICAL_BYTES``.  It recomputes the frozen clause-9 permutation
    once and maps each batch to its component keys on the fly, so peak memory is
    one batch plus the ``n_er`` vector plus the permutation itself; the
    ``steps x 2048`` ``step_keys`` matrix and the joined JSON document are never
    materialized.

    Chunks are yielded in exactly the canonical key order
    (``cell, epoch, fold, length, n_er, schema, seed, step_keys, steps``), so
    ``b"".join(iter_opportunity_manifest_bytes(...))`` equals
    ``canonical_json_bytes(build_opportunity_object(...))`` byte for byte and
    freezes the same per-epoch digest -- the clause-5/11 equivalence guard the
    flexible choice is required to preserve.
    """

    lengths = [len(keys) for keys in keys_by_component]
    length, steps = length_and_steps(*lengths)
    iemocap_index = len(keys_by_component) - 1  # iemocap is always the last component
    order = permutation(length, manifest_seed(cell, fold, seed, epoch))

    # ``n_er`` precedes ``step_keys`` in canonical key order, so it is computed
    # first in one permutation pass, holding only the step-indexed counts.
    n_er = []
    for step in range(steps):
        count = 0
        for global_index in order[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]:
            if _locate(lengths, global_index)[0] == iemocap_index:
                count += 1
        n_er.append(count)

    yield b"{"
    yield b'"cell":' + _json_string(str(cell))
    yield b',"epoch":' + str(int(epoch)).encode("ascii")
    yield b',"fold":' + str(int(fold)).encode("ascii")
    yield b',"length":' + str(int(length)).encode("ascii")
    yield b',"n_er":['
    yield b",".join(str(int(value)).encode("ascii") for value in n_er)
    yield b']'
    yield b',"schema":' + _json_string(SCHEMA_OPPORTUNITY)
    yield b',"seed":' + str(int(seed)).encode("ascii")
    yield b',"step_keys":['
    first_batch = True
    for step in range(steps):
        if not first_batch:
            yield b","
        first_batch = False
        yield b"["
        first_entry = True
        for global_index in order[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]:
            component, local = _locate(lengths, global_index)
            if not first_entry:
                yield b","
            first_entry = False
            yield b"[" + str(component).encode("ascii") + b"," + _json_string(
                keys_by_component[component][local]
            ) + b"]"
        yield b"]"
    yield b']'
    yield b',"steps":' + str(int(steps)).encode("ascii")
    yield b"}"


def streamed_epoch_digest(cell, fold, seed, epoch, keys_by_component):
    """SHA-256 of the streamed canonical bytes; equals ``per_epoch_digest``."""

    hasher = hashlib.sha256()
    for chunk in iter_opportunity_manifest_bytes(cell, fold, seed, epoch, keys_by_component):
        hasher.update(chunk)
    return hasher.hexdigest()


def per_epoch_digest(opportunity_object):
    """Canonical per-epoch digest ``sha256(J(x))`` (clauses 5 and 12)."""

    return sha256_hex(canonical_json_bytes(opportunity_object))


# ---------------------------------------------------------------------------
# Clause 12: tuple-to-digest association
# ---------------------------------------------------------------------------
def run_array(cell, fold, seed, epoch_digests):
    if len(epoch_digests) != EPOCHS:
        raise ManifestError("a run must carry exactly {} epoch digests".format(EPOCHS))
    return [
        [str(cell), int(fold), int(seed), epoch, str(epoch_digests[epoch])]
        for epoch in range(EPOCHS)
    ]


def run_digest(cell, fold, seed, epoch_digests):
    return sha256_hex(canonical_json_bytes(run_array(cell, fold, seed, epoch_digests)))


def matrix_array(entries):
    """Sort the full ``[cell, fold, seed, epoch, digest]`` array (clause 12)."""

    ordered = sorted(
        entries,
        key=lambda entry: (key_sort_token(str(entry[0])), int(entry[1]), int(entry[2]), int(entry[3])),
    )
    return [
        [str(entry[0]), int(entry[1]), int(entry[2]), int(entry[3]), str(entry[4])]
        for entry in ordered
    ]


def matrix_digest(entries):
    return sha256_hex(canonical_json_bytes(matrix_array(entries)))


def all_matrix_tuples():
    """Every ``(cell, fold, seed, epoch)`` tuple of the fixed 500-epoch matrix."""

    tuples = []
    for cell in sorted(("ks_er", "si_er")):
        for fold in FOLDS:
            for seed in SEEDS:
                for epoch in range(EPOCHS):
                    tuples.append((cell, fold, seed, epoch))
    return tuples
