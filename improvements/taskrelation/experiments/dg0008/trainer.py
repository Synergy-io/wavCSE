"""Fixed-LR five-epoch diagnostic trainer for DG-0008.

Only the two arms of one cell differ, and only by whether the auxiliary
cross-entropy term contributes to the loss.  Both arms run the identical
forward graph once per step (the model produces every head before the loss-term
switch), share the same ordered manifest, the same target coefficient, the same
AdamW state, the same serialized initialization and the same RNG snapshot, and
the validation-driven scheduler is removed symmetrically so the learning rate is
constant at ``0.0025`` at every step.

The trainer records the deterministic trace the Stage-1 validity gate needs:
per-step loss, the LR sequence, the epoch-indexed ER counts it actually
consumed, the initialization digest and the pre-forward RNG-state sequence.
Held-out endpoints are never read here (see ``evaluate_dg0008.py``).
"""

import hashlib
import os
import random
import sys

import numpy as np
import torch

from .manifest import ManifestError, canonical_json_bytes, sha256_hex
from .sampler import ManifestEpochSampler

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
for _path in (_REPO_ROOT, os.path.join(_REPO_ROOT, "downstream")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dataset.custom_emb_dataloader import CustomEmbDataLoader  # noqa: E402
from trainer.trainer_model import MultiTasksModelTrainer  # noqa: E402
from trainer.trainer_utils import BatchStats, masked_accuracy, masked_ce_loss  # noqa: E402


TARGET_COEFFICIENT = 0.5
FIXED_LR = 0.0025


def apply_determinism():
    """The pinned, fail-closed determinism contract (proposal reproducibility rule)."""

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False


def rng_state_digest():
    """Digest of every RNG the proposal names, captured before a forward pass."""

    hasher = hashlib.sha256()
    hasher.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        for state in torch.cuda.get_rng_state_all():
            hasher.update(state.cpu().numpy().tobytes())
    hasher.update(np.random.get_state()[1].tobytes())
    hasher.update(repr(random.getstate()).encode("utf-8"))
    return hasher.hexdigest()


def rng_sequence_digest(sequence):
    return sha256_hex(canonical_json_bytes(list(sequence)))


def model_state_digest(model):
    """Digest of the serialized parameter state, independent of file layout."""

    hasher = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        hasher.update(name.encode("utf-8"))
        hasher.update(str(tuple(tensor.shape)).encode("ascii"))
        hasher.update(str(tensor.dtype).encode("ascii"))
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()


def file_sha256(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class _NoOpScheduler:
    """A scheduler that never changes the learning rate."""

    def step(self, *args, **kwargs):  # noqa: D401 - deliberate no-op
        return None


class DG0008Trainer(MultiTasksModelTrainer):
    """Five-epoch fixed-LR trainer with a default-off auxiliary-loss hook."""

    def __init__(
        self,
        *args,
        aux_loss_enabled,
        epoch_steps,
        target_task="er",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.num_tasks != 2:
            raise ManifestError("DG-0008 requires a two-task (cell) model")
        self.aux_loss_enabled = bool(aux_loss_enabled)
        self.target_task_index = self.task_array.index(target_task)
        self.aux_task_index = 1 - self.target_task_index
        if abs((1.0 / float(self.num_tasks)) - TARGET_COEFFICIENT) > 1e-12:
            raise ManifestError("the diagnostic weight must match the cell's 1/num_tasks")
        if float(self.optimizer.param_groups[0]["lr"]) != FIXED_LR:
            raise ManifestError("DG-0008 requires the fixed LR {}".format(FIXED_LR))
        self.task_weight = TARGET_COEFFICIENT
        self.epoch_steps = [list(batch) for batch in epoch_steps]
        if len(self.epoch_steps) != self.num_epochs:
            raise ManifestError("epoch_steps must carry one entry per epoch")
        self.scheduler = _NoOpScheduler()

        self.step_loss_trace = []
        self.step_lr_trace = []
        self.step_er_counts = []
        self.rng_trace = []
        self.epoch_bounds = []
        self.consumed_n_er = []
        self.er_counts_match = True
        self.initialization_digest = model_state_digest(self.model)

    # -- loss term switch (forward graph is unchanged) ----------------------
    def _process_batch(self, batch, train_mode):
        if train_mode:
            self.rng_trace.append(rng_state_digest())

        input_seq, labels_list = self._unpack_batch(batch)
        self.optimizer.zero_grad(set_to_none=True)

        outputs = self.model(input_seq=input_seq)
        logits_tuple = outputs.logits
        pred_tuple = outputs.prediction

        loss_weight = self.task_weight
        loss_all = torch.tensor(0.0, device=self.device)
        loss_task = {}
        batch_task = {}

        for task_index in range(self.num_tasks):
            raw, valid = masked_ce_loss(
                logits=logits_tuple[task_index],
                labels=labels_list[task_index],
                loss_fn=self.loss_fn,
                ignore_index=self.ignore_index,
            )
            enabled = task_index == self.target_task_index or (
                self.aux_loss_enabled and task_index == self.aux_task_index
            )
            if raw is not None and enabled:
                loss_all = loss_all + raw * loss_weight
                batch_task[task_index] = 1
            else:
                batch_task[task_index] = 0
            loss_task[task_index] = float(raw.item()) if raw is not None else 0.0

        l1_reg = torch.tensor(0.0, device=self.device)
        l2_reg = torch.tensor(0.0, device=self.device)
        for parameter in self.model.parameters():
            l1_reg = l1_reg + torch.sum(torch.abs(parameter))
            l2_reg = l2_reg + torch.sum(torch.square(parameter))
        loss_all = loss_all + self.l1_lambda * l1_reg + self.l2_lambda * l2_reg

        if train_mode:
            loss_all.backward()
            self.optimizer.step()

        correct_all = 0
        samples_all = 0
        correct_task = {}
        samples_task = {}
        for task_index in range(self.num_tasks):
            correct, samples = masked_accuracy(
                pred_tuple[task_index], labels_list[task_index], ignore_index=self.ignore_index
            )
            correct_task[task_index] = correct
            samples_task[task_index] = samples
            correct_all += correct
            samples_all += samples

        self._save_debug_counts(
            "train" if train_mode else "val",
            {t: samples_task[t] for t in range(self.num_tasks)},
        )

        if train_mode:
            self.step_loss_trace.append(float(loss_all.item()))
            self.step_lr_trace.append(float(self.optimizer.param_groups[0]["lr"]))
            self.step_er_counts.append(int(samples_task[self.target_task_index]))

        return BatchStats(
            loss_all=float(loss_all.item()),
            batch_all=1,
            loss_task=loss_task,
            batch_task=batch_task,
            correct_task=correct_task,
            samples_task=samples_task,
            correct_all=correct_all,
            samples_all=samples_all,
            valid_count_task=samples_task,
        )

    # -- epoch orchestration: manifest order, fixed LR, one final checkpoint
    def train(self):
        train_dataset = getattr(self, "train_dataset", None) or self.train_dataloader.dataset
        for epoch in range(self.num_epochs):
            sampler = ManifestEpochSampler(self.epoch_steps[epoch], self.key_to_index_store)
            loader = CustomEmbDataLoader(
                train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last_train,
                num_workers=self.num_workers,
            )
            before = len(self.step_er_counts)
            train_stats = self._process_data_loader(loader, train_mode=True)
            val_stats = self._process_data_loader(self.val_dataloader, train_mode=False)

            consumed = self.step_er_counts[before:]
            self.consumed_n_er.append(consumed)
            expected = [int(count) for count in self.epoch_n_er[epoch]]
            if consumed != expected:
                self.er_counts_match = False
            self.epoch_bounds.append((before, len(self.step_er_counts)))

            self.train_losses_all.append(train_stats.avg_loss_all)
            self.train_acc_all.append(train_stats.accuracy_all)
            self.val_losses_all.append(val_stats.avg_loss_all)
            self.val_acc_all.append(val_stats.accuracy_all)
            for task_index in range(self.num_tasks):
                self.train_losses_task[task_index].append(train_stats.avg_loss_task[task_index])
                self.train_acc_task[task_index].append(train_stats.accuracy_task[task_index])
                self.val_losses_task[task_index].append(val_stats.avg_loss_task[task_index])
                self.val_acc_task[task_index].append(val_stats.accuracy_task[task_index])

            self.metrics_writer.write_metrics_all(
                self._epoch_report_line(epoch + 1, "train", train_stats)
            )
            self.metrics_writer.write_metrics_all(
                self._epoch_report_line(epoch + 1, "val", val_stats)
            )
            self.learning_rate_array.append(float(self.optimizer.param_groups[0]["lr"]))

        self.final_validation_er_accuracy = float(
            self.val_acc_task[self.target_task_index][-1]
        )
        self.ckpt.save_epoch_checkpoint(self.num_epochs)
        self.fixed_final_checkpoint_path = self.model_checkpoint_path.replace(
            ".pth", "_fixed_final_epoch{}.pth".format(self.num_epochs)
        )
        torch.save(self.model.state_dict(), self.fixed_final_checkpoint_path)
        torch.load(self.fixed_final_checkpoint_path, map_location="cpu")
        self.fixed_final_checkpoint_sha256 = file_sha256(self.fixed_final_checkpoint_path)

    def bind_key_index(self, key_to_index_store, epoch_n_er):
        """Attach the per-component key lookup and the frozen epoch-indexed ER counts."""

        self.key_to_index_store = key_to_index_store
        self.epoch_n_er = [list(epoch) for epoch in epoch_n_er]
        if len(self.epoch_n_er) != self.num_epochs:
            raise ManifestError("epoch_n_er must carry one entry per epoch")


def refuse_resume(run_directory):
    """Fail closed: a scientific member never resumes from a checkpoint."""

    marker = os.path.join(run_directory, "run.json")
    if os.path.exists(marker):
        raise ManifestError(
            "refusing to resume {}: a completed run record exists; an infrastructure "
            "failure must be repeated from the same immutable inputs and identity, "
            "never resumed".format(run_directory)
        )
    return True


def run_identity_record(*, cell, arm, fold, seed, run_manifest_digest, epoch_digests,
                        initialization_digest, implementation_commit, extra=None):
    """The immutable identity a repeated infrastructure-failed job must reuse."""

    record = {
        "cell": str(cell),
        "arm": str(arm),
        "fold": int(fold),
        "seed": int(seed),
        "run_manifest_digest": str(run_manifest_digest),
        "epoch_digests": [str(value) for value in epoch_digests],
        "initialization_sha256": str(initialization_digest),
        "implementation_commit": implementation_commit,
    }
    if extra:
        record.update(extra)
    return record
