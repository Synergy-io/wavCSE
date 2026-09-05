"""Label, task, and dataset keyword mappings.

Thin compatibility wrapper (Next Step 5 / Eng Review decision D1): the
canonical mappings moved to `mtlkit/tasks.py` (closes issue #8's mapping
half). This module rebuilds the exact same `LabelKeywordMapping`,
`TaskKeywordMapping`, and `TaskDatasetMapping` classes — same attribute
names, same classmethods, same values — sourced from `mtlkit.tasks`'s
`TASK_REGISTRY` and label dicts, so every existing consumer
(`from utils.constant_mapping import ...`) keeps working unmodified. See
`mtlkit/tests/test_facade_parity.py` for the parity proof.

Author: Braveenan Sritharan
Created: 2026-01-19
"""

import mtlkit.tasks as _mtlkit_tasks


class LabelKeywordMapping:
    LABEL2INDEX_SPEECHCOMMANDv1 = _mtlkit_tasks.LABEL2INDEX_SPEECHCOMMAND
    INDEX2LABEL_SPEECHCOMMANDv1 = _mtlkit_tasks.INDEX2LABEL_SPEECHCOMMAND

    LABEL2INDEX_VOXCELEB1 = _mtlkit_tasks.LABEL2INDEX_VOXCELEB
    INDEX2LABEL_VOXCELEB1 = _mtlkit_tasks.INDEX2LABEL_VOXCELEB

    LABEL2INDEX_IEMOCAP = _mtlkit_tasks.LABEL2INDEX_IEMOCAP
    INDEX2LABEL_IEMOCAP = _mtlkit_tasks.INDEX2LABEL_IEMOCAP

    LABEL2INDEX_FLUENTSPEECHCOMMAND = _mtlkit_tasks.LABEL2INDEX_FLUENTSPEECHCOMMAND
    INDEX2LABEL_FLUENTSPEECHCOMMAND = _mtlkit_tasks.INDEX2LABEL_FLUENTSPEECHCOMMAND

    # Group the mappings into a tuple
    speechcommand = (LABEL2INDEX_SPEECHCOMMANDv1, INDEX2LABEL_SPEECHCOMMANDv1)
    voxceleb = (LABEL2INDEX_VOXCELEB1, INDEX2LABEL_VOXCELEB1)
    iemocap = (LABEL2INDEX_IEMOCAP, INDEX2LABEL_IEMOCAP)
    fluentspeechcommand = (LABEL2INDEX_FLUENTSPEECHCOMMAND, INDEX2LABEL_FLUENTSPEECHCOMMAND)

    @classmethod
    def get_label_mapping(cls, key):
        return getattr(cls, key)


class TaskKeywordMapping:
    ks = "Keyword Spotting"
    si = "Speaker Identification"
    er = "Emotion Recognition"
    ic = "Intent Classification"

    @classmethod
    def get_task_name(cls, key):
        return getattr(cls, key, None)


class TaskDatasetMapping:
    ks = "speechcommand"
    si = "voxceleb"
    er = "iemocap"
    ic = "fluentspeechcommand"

    @classmethod
    def get_dataset_key(cls, key):
        return getattr(cls, key, None)
