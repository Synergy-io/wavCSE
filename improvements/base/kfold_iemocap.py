"""
Leave-one-speaker-out (LOSO) fold builder for the IEMOCAP/er task.

Framework-agnostic: takes any object exposing IEMOCAPEmbedding's `.data`
(list of wav_stem keys) and `.mapping` (wav_stem -> {"path": ..., ...}),
and returns index lists. No torch/mlflow imports needed.
"""


def build_loso_fold(iemocap_total_data, fold_index, num_folds=10):
    """Groups iemocap_total_data's samples by speaker (parsed the same way
    IEMOCAPEmbedding.__init__ already does internally, via the wav path's
    4th path component), then returns leave-one-speaker-out train/val/test
    indices for the given fold.

    For fold i: test = speakers[i], val = speakers[(i+1) % num_folds],
    train = the remaining num_folds - 2 speakers. Rotating the val speaker
    (rather than reusing the test speaker) keeps validation genuinely
    unseen relative to the fold being scored.
    """
    speaker_to_indices = {}
    for idx, wav_stem in enumerate(iemocap_total_data.data):
        speaker = iemocap_total_data.mapping[wav_stem]["path"].split("/")[3].split("_")[0]
        speaker_to_indices.setdefault(speaker, []).append(idx)

    speakers = sorted(speaker_to_indices)
    if len(speakers) != num_folds:
        raise ValueError(f"Expected {num_folds} speakers, found {len(speakers)}: {speakers}")

    test_speaker = speakers[fold_index]
    val_speaker = speakers[(fold_index + 1) % num_folds]

    test_indices = speaker_to_indices[test_speaker]
    val_indices = speaker_to_indices[val_speaker]
    train_indices = [
        idx for spk, idxs in speaker_to_indices.items()
        if spk not in (test_speaker, val_speaker)
        for idx in idxs
    ]

    return train_indices, val_indices, test_indices, test_speaker, val_speaker
