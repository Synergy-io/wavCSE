"""Manifest-driven exact training/validation datasets for DG-0008.

The opportunity manifest (clause 11) is the sole authority for the order and
membership of every training batch: the loader resolves a ``[component_index,
canonical_key]`` entry to its frozen embedding file, and no torchaudio
enumeration index enters the schedule.  Pair and control arms of a cell consume
the same :class:`ManifestEpochSampler` object, so their ordered batch entries
are identical by construction.

``downstream/`` is frozen; this module only reads its path conventions, which
the identity manifest already records.
"""

import os

import torch

from . import CELL_COMPONENTS
from .manifest import ManifestError, canonical_json_bytes, sha256_hex

COMPONENT_DIRECTORY = {
    "speechcommand": "speechcommand",
    "voxceleb": "voxceleb",
    "iemocap": "iemocap",
}


def embedding_relpath(key, upstream_model_type, frame_pool_id):
    """The loader's own path transform (``downstream/.../preprocess_embedding.py``)."""

    return key.replace(
        ".wav", "_{}_{}.pt".format(upstream_model_type, frame_pool_id)
    )


def component_directories(cell):
    try:
        return [COMPONENT_DIRECTORY[name] for name in CELL_COMPONENTS[cell]]
    except KeyError:
        raise ManifestError("unknown cell {!r}".format(cell))


def index_patterns(component_count):
    """One-hot label patterns for a cell's component order (loader convention)."""

    patterns = []
    for position in range(component_count):
        bits = ["0"] * component_count
        bits[position] = "1"
        patterns.append("".join(bits))
    return patterns


class ManifestDataset(torch.utils.data.Dataset):
    """``__getitem__(i)`` resolves the *i*-th ordered vector entry to its embedding.

    ``entries`` is the concatenation of component 0 followed by component 1
    (clause 6); ``label_indices`` are the effective integer indices frozen in the
    identity manifest, and the label tuple is built with the same one-hot
    ``index_pattern`` the loader uses.
    """

    def __init__(
        self,
        entries,
        label_indices,
        patterns,
        *,
        embedding_root,
        component_dirs,
        upstream_model_type,
        frame_pool_id,
        transformer_layer_array=None,
    ):
        if len(entries) != len(label_indices):
            raise ManifestError("entries/labels length mismatch")
        if len(component_dirs) != len(patterns):
            raise ManifestError("component/pattern length mismatch")
        self.entries = list(entries)
        self.label_indices = [int(value) for value in label_indices]
        self.patterns = list(patterns)
        self.embedding_root = str(embedding_root)
        self.component_dirs = list(component_dirs)
        self.upstream_model_type = str(upstream_model_type)
        self.frame_pool_id = str(frame_pool_id)
        self.transformer_layer_array = transformer_layer_array

    def __len__(self):
        return len(self.entries)

    def label_tuple(self, index):
        component, _key = self.entries[index]
        label_index = self.label_indices[index]
        return tuple(
            label_index if bit == "1" else -1 for bit in self.patterns[component]
        )

    def resolve_path(self, index):
        component, key = self.entries[index]
        relative = embedding_relpath(key, self.upstream_model_type, self.frame_pool_id)
        return os.path.join(
            self.embedding_root, self.component_dirs[component], relative
        )

    def __getitem__(self, index):
        path = self.resolve_path(index)
        try:
            embedding = torch.load(path, map_location="cpu")
        except (FileNotFoundError, RuntimeError, EOFError) as exc:
            raise ManifestError("failed to load embedding: {}".format(path)) from exc
        if self.transformer_layer_array is not None:
            embedding = embedding[self.transformer_layer_array, :]
        label = torch.tensor(self.label_tuple(index), dtype=torch.long)
        return embedding, label


class ManifestEpochSampler(torch.utils.data.Sampler):
    """Yields the exact global indices of one frozen epoch manifest, in order.

    The order is derived from the manifest's ``step_keys`` (component, key)
    entries — never from the dataset's own iteration order.
    """

    def __init__(self, steps, key_to_index):
        order = []
        for batch in steps:
            for component, key in batch:
                try:
                    order.append(key_to_index[component][key])
                except KeyError:
                    raise ManifestError(
                        "manifest key {!r} is absent from component {}".format(key, component)
                    )
        self._order = order

    def __iter__(self):
        return iter(self._order)

    def __len__(self):
        return len(self._order)

    def ordered_indices(self):
        return list(self._order)


def key_to_index(vectors):
    """Per-component ``key -> global index`` maps for the concatenated vectors."""

    maps = []
    offset = 0
    for keys, _labels in vectors:
        maps.append({key: offset + position for position, key in enumerate(keys)})
        offset += len(keys)
    return maps


def plain_dataset(vectors, cell, *, embedding_root, upstream_model_type, frame_pool_id,
                  transformer_layer_array=None):
    """Build a :class:`ManifestDataset` from ordered cell vectors."""

    entries = []
    labels = []
    for component, (keys, label_indices) in enumerate(vectors):
        entries.extend((component, key) for key in keys)
        labels.extend(label_indices)
    return ManifestDataset(
        entries,
        labels,
        index_patterns(len(vectors)),
        embedding_root=embedding_root,
        component_dirs=component_directories(cell),
        upstream_model_type=upstream_model_type,
        frame_pool_id=frame_pool_id,
        transformer_layer_array=transformer_layer_array,
    )


def schedule_fingerprint(steps):
    """Canonical digest of an ordered schedule; pair and control must match."""

    return sha256_hex(canonical_json_bytes(steps))
