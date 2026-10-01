#!/usr/bin/env python3
"""Regression test: an embedding's path is derived from the source extension alone.

`process_and_write_data` used to build the `.pt` path with

    os.path.join(dir_emb_path, wavpath).replace(".wav", "_<model>_<pooling>.pt")

which rewrites the *first* ".wav" anywhere in the joined path. An embedding root that
itself contains ".wav" therefore redirected every embedding into a parallel tree while the
CSV still recorded the real file size -- silent data loss. The RunPod root
(`/workspace/wavcse-jobs`) never contained ".wav", so the defect only appeared on Colab,
whose job root is `/content/.wavcse/jobs/<id>`.

Run (the working directory must be `upstream/`, as for `main.py`):

    cd upstream && python tests/test_embedding_io_paths.py
"""

import csv
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.embedding_io import process_and_write_data  # noqa: E402
from utils.get_size import get_file_size  # noqa: E402

WAVPATH = "Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav"
EXPECTED_RELATIVE = f"{WAVPATH[:-len('.wav')]}_wavlm_large_max.pt"


def write_one(root_emb_path, pooling="max"):
    process_and_write_data(
        data=[(torch.zeros(25, 1024), 0, WAVPATH)],
        root_emb_path=root_emb_path,
        upstream_model_type="wavlm_large",
        frame_pooling_type=pooling,
        dataset_name="iemocap",
        limit=None,
        append=False,
        frame_pooling_param=None,
    )
    return os.path.join(root_emb_path, "wavlm_large", pooling, "iemocap")


def check(condition, label):
    if not condition:
        raise SystemExit(f"FAIL: {label}")
    print(f"ok  {label}")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        plain = write_one(os.path.join(tmp, "plain"))
        check(os.path.isfile(os.path.join(plain, EXPECTED_RELATIVE)),
              "embedding is written under a plain root")

        # The regression: a root containing the substring ".wav".
        dotted_root = os.path.join(tmp, "content.wavcse", "outputs")
        dotted = write_one(dotted_root)
        check(os.path.isfile(os.path.join(dotted, EXPECTED_RELATIVE)),
              "embedding is written under a root containing '.wav'")
        loaded = torch.load(os.path.join(dotted, EXPECTED_RELATIVE))
        check(tuple(loaded.shape) == (25, 1024) and loaded.dtype == torch.float32,
              "the written tensor loads with the expected shape and dtype")

        # No parallel tree may be created by rewriting the root's own ".wav".
        parallel = os.path.join(tmp, "content_wavlm_large_max.ptcse")
        check(not os.path.exists(parallel),
              "no parallel directory is created by rewriting the root's '.wav'")

        # The CSV must agree with what is actually on disk.
        csv_path = os.path.join(dotted, "iemocap_wavlm_large_max.csv")
        with open(csv_path, newline="") as handle:
            rows = list(csv.reader(handle))
        check(rows[0] == ["wavpath", "label", "file_size"], "the index has the expected header")
        check(rows[1][0] == WAVPATH, "the index records the source wavpath")
        check(rows[1][2] == get_file_size(os.path.join(dotted, EXPECTED_RELATIVE)),
              "the recorded size is the size of the file on disk")
        check(rows[1][2] != "does not exist", "the recorded size is not 'does not exist'")

        # A wavpath without the source extension keeps the previous behaviour exactly.
        other = os.path.join(tmp, "other")
        process_and_write_data(
            data=[(torch.zeros(25, 1024), 0, "Session1/clip")],
            root_emb_path=other,
            upstream_model_type="wavlm_large",
            frame_pooling_type="max",
            dataset_name="iemocap",
            limit=None,
            append=False,
            frame_pooling_param=None,
        )
        check(os.path.isfile(os.path.join(other, "wavlm_large", "max", "iemocap", "Session1", "clip")),
              "a non-.wav source path is passed through unchanged, as before")

    print("ALL EMBEDDING-IO PATH CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
