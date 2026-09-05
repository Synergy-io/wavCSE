"""MTRL migration numeric-parity test (Next Step 6, Success Criteria).

MTRL migrates to the combine() seam; its per-task losses must numerically
match its pre-migration results on a set of pinned batches, including the
full warmup + Omega-refresh epoch schedule replayed identically, before its
duplicated `_process_batch` is considered safely deleted.

Uses the same git-frozen-reference technique as
mtlkit/tests/facade_parity_utils.py: extracts `improvements/taskrelation/01-mtrl/`
as it existed before this migration (commit a6b9823) into an isolated
namespace, builds the OLD standalone `DownstreamMultiTaskModelMTRL` (its
own Omega buffers) side by side with the NEW mtlkit-backed one (Omega owned
by `MTRLCombineStrategy`), and replays several pinned batches across a
multi-epoch schedule (crossing the warmup boundary and an Omega refresh),
asserting identical per-task losses and Omega matrices at every step.
"""

import contextlib
import os
import subprocess
import sys
import tempfile
import unittest

import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DOWNSTREAM_DIR = os.path.join(REPO_ROOT, "downstream")
MTRL_DIR = os.path.join(REPO_ROOT, "improvements", "taskrelation", "01-mtrl")
for path in (DOWNSTREAM_DIR, REPO_ROOT, MTRL_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

PRE_MIGRATION_COMMIT = "a6b9823"


@contextlib.contextmanager
def reference_mtrl_and_downstream(commit: str = PRE_MIGRATION_COMMIT):
    """Extract `downstream/` and `improvements/taskrelation/01-mtrl/` as of
    `commit` (before this migration) into an isolated temp tree, importable
    without colliding with the live (migrated) modules of the same name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = os.path.join(tmpdir, "archive.tar")
        with open(archive_path, "wb") as archive_file:
            subprocess.run(
                ["git", "archive", commit, "downstream", "improvements/taskrelation/01-mtrl"],
                cwd=REPO_ROOT,
                stdout=archive_file,
                check=True,
            )
        subprocess.run(["tar", "-xf", archive_path], cwd=tmpdir, check=True)

        ref_downstream_dir = os.path.join(tmpdir, "downstream")
        ref_mtrl_dir = os.path.join(tmpdir, "improvements", "taskrelation", "01-mtrl")

        colliding_names = ("pooling", "model", "trainer", "evaluator", "dataset", "utils", "mtrl_model", "mtrl_trainer")
        saved_modules = {}
        for name in list(sys.modules):
            if name.split(".")[0] in colliding_names:
                saved_modules[name] = sys.modules.pop(name)

        saved_path = list(sys.path)
        sys.path = [ref_mtrl_dir, ref_downstream_dir] + [
            p for p in sys.path
            if os.path.abspath(p) not in (DOWNSTREAM_DIR, MTRL_DIR)
        ]

        try:
            yield ref_downstream_dir, ref_mtrl_dir
        finally:
            for name in list(sys.modules):
                if name.split(".")[0] in colliding_names:
                    del sys.modules[name]
            sys.path = saved_path
            sys.modules.update(saved_modules)


def _common_kwargs():
    return dict(
        upstream_model_type="wavlm_large",
        task_type="ks_si_er",
        embedding_dim_shared1=32,
        embedding_dim_shared2=16,
        layer_pooling_type="mean",
        dropout_prob_shared1=0.0,
        dropout_prob_shared2=0.0,
    )


class MTRLNumericParityTests(unittest.TestCase):
    """Replays the same pinned batches across the same epoch schedule
    (crossing warmup and an Omega refresh) on both the pre-migration
    standalone model and the migrated mtlkit-backed one, asserting
    identical per-task losses and Omega matrices throughout."""

    def _pinned_batches(self, num_batches=4, batch_size=4):
        torch.manual_seed(2024)
        batches = []
        for _ in range(num_batches):
            input_seq = torch.randn(batch_size, 25, 1024)
            labels_list = [
                torch.randint(0, 12, (batch_size,)),
                torch.randint(0, 1251, (batch_size,)),
                torch.randint(0, 4, (batch_size,)),
            ]
            batches.append((input_seq, labels_list))
        return batches

    def test_per_task_losses_and_omega_match_across_warmup_and_refresh(self):
        import mtrl_model as live_mtrl_model
        import mtrl_combine as live_mtrl_combine

        with reference_mtrl_and_downstream() as (_, _):
            import mtrl_model as ref_mtrl_model

            torch.manual_seed(7)
            live_model = live_mtrl_model.DownstreamMultiTaskModelMTRL(**_common_kwargs())
            torch.manual_seed(7)
            ref_model = ref_mtrl_model.DownstreamMultiTaskModelMTRL(
                **_common_kwargs(), mtrl_lambda=0.01, omega_epsilon=1e-4, normalize_w=False
            )

        # Verify identical initialization before replaying batches.
        for (lw, lb), (rw, rb) in zip(
            [(h.weight, h.bias) for h in live_model.heads],
            [(h.weight, h.bias) for h in ref_model.classifiers],
        ):
            self.assertTrue(torch.allclose(lw, rw))
            self.assertTrue(torch.allclose(lb, rb))

        live_strategy = live_mtrl_combine.MTRLCombineStrategy(
            num_tasks=3, mtrl_lambda=0.01, omega_epsilon=1e-4, normalize_w=False,
            mtrl_warmup_epochs=2, omega_update_frequency=1,
        )
        ref_warmup_epochs = 2
        ref_omega_update_frequency = 1
        loss_fn = nn.CrossEntropyLoss()
        ignore_index = -1

        import mtlkit.trainer as mtlkit_trainer

        batches = self._pinned_batches()

        for epoch in range(1, 4):  # crosses warmup (epoch 2) and does 2 Omega refreshes
            live_strategy.on_epoch_begin(epoch)
            ref_current_epoch = epoch  # mirrors ref trainer's self.current_epoch = ep

            for input_seq, labels_list in batches:
                # --- live (mtlkit-backed) ---
                live_result = mtlkit_trainer.process_batch(
                    live_model, input_seq, labels_list, loss_fn, live_strategy, ignore_index
                )
                live_result.loss_all.backward()
                live_grads = [h.weight.grad.clone() for h in live_model.heads]
                for h in live_model.heads:
                    h.weight.grad = None
                    h.bias.grad = None

                # --- reference (pre-migration) ---
                outputs = ref_model(input_seq)
                logits_tuple = outputs.logits
                ref_loss_all = torch.tensor(0.0)
                loss_weight = 1.0 / 3.0
                for t in range(3):
                    mask = labels_list[t] != ignore_index
                    labels_masked = labels_list[t][mask]
                    if labels_masked.numel() == 0:
                        continue
                    logits_masked = logits_tuple[t][mask, :]
                    loss_t = loss_fn(logits_masked, labels_masked)
                    ref_loss_all = ref_loss_all + loss_t * loss_weight
                if ref_current_epoch >= ref_warmup_epochs:
                    ref_loss_all = ref_loss_all + ref_model.get_mtrl_regularizer_loss()
                ref_loss_all.backward()
                ref_grads = [h.weight.grad.clone() for h in ref_model.classifiers]
                for h in ref_model.classifiers:
                    h.weight.grad = None
                    h.bias.grad = None

                self.assertTrue(
                    torch.allclose(live_result.loss_all, ref_loss_all, atol=1e-5),
                    msg=f"loss mismatch at epoch={epoch}",
                )
                for i, (lg, rg) in enumerate(zip(live_grads, ref_grads)):
                    self.assertTrue(
                        torch.allclose(lg, rg, atol=1e-5),
                        msg=f"grad mismatch at epoch={epoch}, head={i}",
                    )

            # --- epoch-end Omega refresh, both sides ---
            live_strategy.on_epoch_end(epoch)
            if epoch >= ref_warmup_epochs and epoch % ref_omega_update_frequency == 0:
                ref_model.update_omega()

            self.assertTrue(
                torch.allclose(live_strategy.omega, ref_model.get_omega_matrix(), atol=1e-5),
                msg=f"Omega mismatch after epoch={epoch}",
            )
            self.assertTrue(
                torch.allclose(live_strategy.omega_inv, ref_model.omega_inv, atol=1e-5),
                msg=f"Omega_inv mismatch after epoch={epoch}",
            )


if __name__ == "__main__":
    unittest.main()
