"""mtlkit/tasks.py — task/dataset registry + per-(sample, task) boolean masks.

Closes issue #8's mapping half: this module is now the canonical home for the
label/task/dataset mappings that were independently duplicated between
``upstream/utils/constant_mapping.py`` and ``downstream/utils/constant_mapping.py``.
Those files stay as-is until Next Step 5 rewrites `upstream/`/`downstream/` as
thin wrappers over this module.

Task/dataset parsing (``task_type`` strings like ``"ks_si_er"``) and the
one-hot ``index_pattern`` construction below are a byte-for-byte port of
``downstream/dataset/load_embedding.py``'s ``_build_dataset_array_from_task_type``,
``_build_task_name_array_from_task_type``, and its inline index-pattern loop
(same error messages, same ordering/dedup behavior) — this is Next Step 1's
"re-deriving the current 4-task index_pattern masking behavior exactly".

Mask vs. index_pattern (see the design doc's Glossary):
    index_pattern   a bitstring like "0100", one per dataset, built once per
                    task_type call — position i is "1" iff position i in the
                    ordered dataset list is this dataset.
    mask            the *decoded* boolean form of an index_pattern: a
                    length-num_tasks list of bools, one per sample. In the
                    current disjoint 4-task setup every mask has exactly one
                    True; FSC's 3-simultaneous-valid-positions case (Next Step
                    8) just means more than one True — this module's mask
                    representation already supports that without change.

    task_type = "ks_si_er_ic"
                 |    |    |    |
                 v    v    v    v
    dataset_id_array = [speechcommand, voxceleb, iemocap, fluentspeechcommand]
                 |
                 v  (one-hot per position)
    index_pattern_map = {
        speechcommand:        "1000",
        voxceleb:             "0100",
        iemocap:              "0010",
        fluentspeechcommand:  "0001",
    }
                 |
                 v  decode_index_pattern("0100")
    mask = [False, True, False, False]   # this sample is valid for task[1] only
"""

from dataclasses import dataclass, field
from typing import Dict, List

from mtlkit.registry import Registry

# ---------------------------------------------------------------------------
# Label mappings (ported from downstream/utils/constant_mapping.py's
# LabelKeywordMapping — canonical copy now lives here per issue #8).
# ---------------------------------------------------------------------------

LABEL2INDEX_SPEECHCOMMAND = {
    "_silence_": 11,
    "_unknown_": 10,
    "down": 3,
    "go": 9,
    "left": 4,
    "no": 1,
    "off": 7,
    "on": 6,
    "right": 5,
    "stop": 8,
    "up": 2,
    "yes": 0,
}
INDEX2LABEL_SPEECHCOMMAND = {v: k for k, v in LABEL2INDEX_SPEECHCOMMAND.items()}

LABEL2INDEX_VOXCELEB = {str(i): i - 1 for i in range(1, 1252)}
INDEX2LABEL_VOXCELEB = {v: k for k, v in LABEL2INDEX_VOXCELEB.items()}

LABEL2INDEX_IEMOCAP = {"ang": 2, "hap": 1, "neu": 0, "sad": 3}
INDEX2LABEL_IEMOCAP = {v: k for k, v in LABEL2INDEX_IEMOCAP.items()}

LABEL2INDEX_FLUENTSPEECHCOMMAND = {
    "activate_lamp": 0,
    "activate_lights": 1,
    "activate_lights_bedroom": 2,
    "activate_lights_kitchen": 3,
    "activate_lights_washroom": 4,
    "activate_music": 5,
    "deactivate_lamp": 6,
    "deactivate_lights": 7,
    "deactivate_lights_bedroom": 8,
    "deactivate_lights_kitchen": 9,
    "deactivate_lights_washroom": 10,
    "deactivate_music": 11,
    "bring_juice": 12,
    "bring_newspaper": 13,
    "bring_shoes": 14,
    "bring_socks": 15,
    "increase_heat": 16,
    "increase_heat_bedroom": 17,
    "increase_heat_kitchen": 18,
    "increase_heat_washroom": 19,
    "increase_volume": 20,
    "decrease_heat": 21,
    "decrease_heat_bedroom": 22,
    "decrease_heat_kitchen": 23,
    "decrease_heat_washroom": 24,
    "decrease_volume": 25,
    "change_language": 26,
    "change_language_chinese": 27,
    "change_language_english": 28,
    "change_language_german": 29,
    "change_language_korean": 30,
}
INDEX2LABEL_FLUENTSPEECHCOMMAND = {v: k for k, v in LABEL2INDEX_FLUENTSPEECHCOMMAND.items()}


# ---------------------------------------------------------------------------
# Task/dataset registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskSpec:
    """One task: its human-readable name, source dataset key, and label space."""

    key: str
    display_name: str
    dataset_key: str
    label2index: Dict[str, int] = field(repr=False)

    @property
    def num_classes(self) -> int:
        return len(self.label2index)


TASK_REGISTRY: Registry[TaskSpec] = Registry("task")


def register_task(spec: TaskSpec) -> TaskSpec:
    return TASK_REGISTRY.register(spec.key, spec)


def _register_builtin_tasks() -> None:
    register_task(
        TaskSpec("ks", "Keyword Spotting", "speechcommand", LABEL2INDEX_SPEECHCOMMAND)
    )
    register_task(
        TaskSpec("si", "Speaker Identification", "voxceleb", LABEL2INDEX_VOXCELEB)
    )
    register_task(TaskSpec("er", "Emotion Recognition", "iemocap", LABEL2INDEX_IEMOCAP))
    register_task(
        TaskSpec(
            "ic",
            "Intent Classification",
            "fluentspeechcommand",
            LABEL2INDEX_FLUENTSPEECHCOMMAND,
        )
    )


_register_builtin_tasks()


# ---------------------------------------------------------------------------
# task_type parsing — exact port of load_embedding.py's helpers
# ---------------------------------------------------------------------------


def task_name_array_from_task_type(task_type: str) -> List[str]:
    """Ordered, de-duplicated task display names for an underscore-joined
    ``task_type`` string, e.g. ``"ks_si_er"`` -> ["Keyword Spotting", ...].

    Byte-for-byte port of ``LoadEmbedding._build_task_name_array_from_task_type``
    (same tokenization, same de-dup-preserving-order, same error message).
    """
    task_tokens = task_type.split("_")
    task_keys: List[str] = []
    for t in task_tokens:
        spec = TASK_REGISTRY.try_get(t)
        display_name = spec.display_name if spec is not None else None
        if display_name is None:
            raise ValueError(f"Invalid task type: {t}")
        if display_name not in task_keys:
            task_keys.append(display_name)
    return task_keys


def dataset_array_from_task_type(task_type: str) -> List[str]:
    """Ordered, de-duplicated dataset keys for a ``task_type`` string, e.g.
    ``"ks_si_er"`` -> ["speechcommand", "voxceleb", "iemocap"].

    Byte-for-byte port of ``LoadEmbedding._build_dataset_array_from_task_type``.
    """
    task_tokens = task_type.split("_")
    dataset_keys: List[str] = []
    for t in task_tokens:
        spec = TASK_REGISTRY.try_get(t)
        dataset_key = spec.dataset_key if spec is not None else None
        if dataset_key is None:
            raise ValueError(f"Invalid task type: {t}")
        if dataset_key not in dataset_keys:
            dataset_keys.append(dataset_key)
    return dataset_keys


def index_pattern_map_for_task_type(task_type: str) -> Dict[str, str]:
    """One-hot bitstring per dataset, in the order ``task_type`` names them.

    Exact port of ``LoadEmbedding.load_embedding``'s inline index-pattern
    loop (lines 140-146 of the original): position ``i`` in the ordered
    dataset list gets a bitstring of length ``len(dataset_id_array)`` with a
    "1" at position ``i`` and "0" everywhere else.
    """
    dataset_id_array = dataset_array_from_task_type(task_type)
    pattern_len = len(dataset_id_array)
    index_pattern_map: Dict[str, str] = {}
    for i, ds in enumerate(dataset_id_array):
        bits = ["0"] * pattern_len
        bits[i] = "1"
        index_pattern_map[ds] = "".join(bits)
    return index_pattern_map


def decode_index_pattern(index_pattern: str) -> List[bool]:
    """Decode an index_pattern bitstring (e.g. ``"0100"``) into a per-task
    boolean validity mask (``[False, True, False, False]``).

    This is the "pure validity function (source dataset -> task-validity
    vector)" the design doc's Glossary calls for. It generalizes unchanged to
    FSC's several-simultaneously-valid-positions case (Next Step 8): decoding
    never assumes exactly one "1".
    """
    return [bit == "1" for bit in index_pattern]


def task_validity_mask(dataset_key: str, task_type: str) -> List[bool]:
    """Per-(sample, task) boolean mask for a sample from ``dataset_key``,
    given the ``task_type`` of the current run.

    Example: ``task_validity_mask("voxceleb", "ks_si_er")`` -> [False, True, False]
    (this sample is valid for the SI task slot only).
    """
    index_pattern_map = index_pattern_map_for_task_type(task_type)
    try:
        pattern = index_pattern_map[dataset_key]
    except KeyError:
        valid = ", ".join(sorted(index_pattern_map)) or "(none)"
        raise KeyError(
            f"'{dataset_key}' is not one of the datasets in task_type='{task_type}' "
            f"(valid datasets for this task_type: {valid})"
        ) from None
    return decode_index_pattern(pattern)
