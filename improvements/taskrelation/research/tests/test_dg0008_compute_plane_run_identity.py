"""A DG-0008 member must report a run identity the compute plane can stage.

``improvements/compute/worker_stage.py`` never parses a training run's stdout:
it reads the records the process wrote to ``ARC_RUN_IDENTITY_FILE`` with
``improvements.run_identity.read_identity_file`` and exits non-zero when there
are none, so a member that trains but stays silent stages none of its declared
outputs.  ``run_dg0008.build_member`` must therefore emit its identity as soon
as the trainer owns its directories and before the first training step, with the
same ``research_identity`` study/arm/commit block its sibling
``improvements/base/run_base.py`` publishes.

The test drives the real ``build_member`` over the real committed config with
the frozen-artifact readers and the tracker replaced by offline stubs, and a
stub trainer that refuses to take a second step: the member stops at
``trainer.train()``, so the identity must already be on disk by then.  No GPU,
no corpora, no embeddings and no network are involved.
"""

import contextlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import yaml

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
STUDY_DIR = os.path.join(
    REPO_ROOT, "improvements", "taskrelation", "research", "studies", "DG-0008"
)
CONFIG_PATH = os.path.join(STUDY_DIR, "configs", "dg0008.yaml")
for path in (REPO_ROOT, os.path.join(REPO_ROOT, "downstream")):
    if path not in sys.path:
        sys.path.insert(0, path)

from improvements.run_identity import (  # noqa: E402
    ENV_FILE,
    read_identity_file,
)

CELL = "si_er"
ARM = "pair"
FOLD = 3
SEED = 2
EPOCHS = 5
LENGTH = 4
STEPS = 2
IMPLEMENTATION_COMMIT = "0" * 40
RUN_ID = "2026_10_04_09_15_00"


class _TrainingReached(RuntimeError):
    """Raised by the stub trainer to stop the member at the first step."""


def _load_runner():
    path = os.path.join(STUDY_DIR, "run_dg0008.py")
    spec = importlib.util.spec_from_file_location("dg0008_identity_runner", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_dg0008 = _load_runner()


def _epoch_manifest(epoch):
    return {
        "cell": CELL,
        "fold": FOLD,
        "seed": SEED,
        "epoch": epoch,
        "length": LENGTH,
        "steps": STEPS,
        "step_keys": [[0, 1], [2, 3]],
        "n_er": [1, 1],
    }


class _StubTrainer:
    """The trainer surface the run-identity record reads, and nothing more.

    The run directories are created exactly as
    ``downstream/trainer/trainer_model.py`` creates them -- a per-run child of
    the roots the member handed in -- so the record is checked against the
    trainer's own directories rather than against a path derived independently.
    """

    instances = []

    def __init__(self, *, results_root, checkpoints_root, **kwargs):
        self.results_root = results_root
        self.checkpoints_root = checkpoints_root
        self.results_dir = os.path.join(results_root, RUN_ID)
        self.ckpt_dir = os.path.join(checkpoints_root, RUN_ID)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.events = []
        self.identity_file = os.environ.get(ENV_FILE, "")
        self.identity_at_training = None
        _StubTrainer.instances.append(self)

    def bind_key_index(self, key_to_index_store, epoch_n_er):
        self.events.append("bind_key_index")

    def train(self):
        self.events.append("train")
        # Exactly what the wrapper would read if the member died right here.
        self.identity_at_training = read_identity_file(self.identity_file)
        raise _TrainingReached("the member is stopped at the first training step")


@contextlib.contextmanager
def _offline_tracking_run(member_config, method, cell):
    yield None


class ComputePlaneRunIdentityTests(unittest.TestCase):
    def test_the_wrapper_reads_this_members_own_identity_before_training(self):
        _StubTrainer.instances = []
        out_root = tempfile.mkdtemp(prefix="dg0008-identity-")
        identity_file = os.path.join(out_root, ".run_identity.jsonl")
        with open(CONFIG_PATH, encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        index = {
            "identity_digest": "identity-digest",
            "l": {"{}|{}".format(CELL, FOLD): LENGTH},
            "s": {"{}|{}".format(CELL, FOLD): STEPS},
        }
        patches = [
            mock.patch.dict(os.environ, {
                ENV_FILE: identity_file,
                # The committed root is only expanded by the real resolver; no
                # prepared layout exists here, so no override may be honoured.
                "WAVCSE_ROOT_EMB_PATH": "",
            }),
            mock.patch.object(run_dg0008.frozen, "load_identity", return_value={}),
            mock.patch.object(
                run_dg0008.frozen, "verify_run",
                return_value=(["digest-{}".format(epoch) for epoch in range(EPOCHS)],
                              "run-manifest-digest"),
            ),
            mock.patch.object(
                run_dg0008.frozen, "load_epoch_manifest",
                side_effect=lambda *args, **kwargs: _epoch_manifest(args[-1]),
            ),
            mock.patch.object(run_dg0008, "cell_vectors",
                              return_value=[(["a", "b", "c", "d"], [0, 1, 0, 1])]),
            mock.patch.object(run_dg0008, "plain_dataset",
                              side_effect=lambda vectors, cell, **kwargs: ["dataset"]),
            mock.patch.object(run_dg0008, "DownstreamMultiTaskModel",
                              side_effect=lambda **kwargs: mock.Mock()),
            mock.patch.object(run_dg0008.training, "DG0008Trainer", _StubTrainer),
            mock.patch.object(run_dg0008, "open_member_run", _offline_tracking_run),
            mock.patch.object(run_dg0008.mlflow_utils, "set_standard_tags"),
            mock.patch.object(run_dg0008.mlflow_utils, "log_config_params"),
            mock.patch.object(run_dg0008.mlflow_utils, "resolve_git_commit",
                              return_value=IMPLEMENTATION_COMMIT),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

        with self.assertRaises(_TrainingReached):
            run_dg0008.build_member(cfg, CELL, ARM, FOLD, SEED, out_root, index,
                                    out_root, "cpu")

        self.assertEqual(len(_StubTrainer.instances), 1)
        trainer = _StubTrainer.instances[0]
        self.assertEqual(trainer.events, ["bind_key_index", "train"])

        # The member stopped at its first training step, so whatever the
        # wrapper can read was emitted before training started.
        self.assertEqual(len(trainer.identity_at_training), 1,
                         "the run identity must be reported before training starts")
        records = read_identity_file(identity_file)
        self.assertEqual(len(records), 1,
                         "the wrapper's reader found no run identity to stage")
        with open(identity_file, encoding="utf-8") as handle:
            self.assertEqual(len(handle.read().splitlines()), 1,
                             "a second emission would stage the wrong run")

        record = records[0]
        self.assertEqual(record, trainer.identity_at_training[0])
        self.assertEqual(json.loads(json.dumps(record)), record)
        self.assertEqual(record["results_dir"], os.path.abspath(trainer.results_dir))
        self.assertEqual(record["checkpoints_dir"],
                         os.path.abspath(trainer.ckpt_dir))
        self.assertTrue(os.path.isdir(record["results_dir"]))
        self.assertTrue(os.path.isdir(record["checkpoints_dir"]))
        self.assertEqual(record["run_id"], RUN_ID)
        self.assertEqual(record["checkpoint_run_id"], RUN_ID)

        # The record answers which arm of which study at which commit built the
        # artifact, from the member's own cell/arm identity.
        methods = cfg["research"]["methods"]
        self.assertEqual(record["model"], run_dg0008.MEMBER_MODEL)
        self.assertEqual(record["task_type"], CELL)
        self.assertEqual(record["seed"], SEED)
        self.assertEqual(record["study_id"], cfg["research"]["study_id"])
        self.assertEqual(record["stage"], cfg["research"]["stage"])
        self.assertEqual(record["method"], methods[ARM])
        self.assertNotEqual(record["method"], methods["control"])
        self.assertEqual(record["representation"],
                         cfg["research"]["representation"])
        self.assertEqual(record["git_commit"], IMPLEMENTATION_COMMIT)


if __name__ == "__main__":
    unittest.main()
