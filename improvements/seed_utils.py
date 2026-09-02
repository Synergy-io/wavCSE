"""
Shared random-seed helper for every improvements/ training script.

No code in this pipeline set a random seed before this file existed --
not run_base.py, run_improvements.py, downstream/, or upstream/, and no
config carried a `seed:` key. Every run (baseline, MTRL, GBC/TSM/PMR, the
pooling grid) started from an uncontrolled, unrecorded random state, so
two runs of the identical config could (and did) land noticeably apart
purely from randomness -- see improvements/taskrelation/01-mtrl/README.md's
iteration 4 stability check (0.7812 vs 0.7577 er accuracy, same config).

The only randomness anywhere in the pipeline is (1) nn.Linear/nn.Parameter
weight init at model-construction time and (2) DataLoader shuffling.
CustomEmbDataLoader passes no generator=/worker_init_fn=, so with
num_workers > 0 (used throughout), PyTorch auto-derives each worker's seed
from the main process's torch.manual_seed() -- a single set_seed() call
before model/DataLoader construction is sufficient for full
reproducibility, no extra worker_init_fn plumbing needed.
"""

import random

import numpy as np
import torch


def set_seed(seed):
    """Seed random/numpy/torch (CPU + all CUDA devices) for reproducible
    training. No-op if seed is None, so omitting `seed:` from a config
    preserves whatever behavior existing callers expect."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
