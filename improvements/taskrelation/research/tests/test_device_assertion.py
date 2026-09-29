"""A run that declares CUDA must not silently train on CPU.

`downstream/utils/setup_device.py` falls back to CPU and clamps an out-of-range
index, printing a line nobody reads. For a recorded run that silently changes
the device the configuration names, so `improvements/device_utils.py` asserts at
the entry-point boundary instead. `downstream/` stays frozen.
"""

import os
import sys
import unittest
from unittest import mock

import torch

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from improvements.device_utils import assert_training_device  # noqa: E402


class DeviceAssertionTests(unittest.TestCase):
    def test_cpu_work_is_never_second_guessed(self):
        device = torch.device("cpu")
        self.assertIs(assert_training_device("cpu", None, device), device)
        self.assertIs(assert_training_device("CPU", 3, device), device)

    def test_cuda_configured_without_cuda_fails_loudly(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaises(RuntimeError) as caught:
                assert_training_device("cuda", 0, torch.device("cpu"))
        message = str(caught.exception)
        self.assertIn("Refusing to train on CPU", message)
        self.assertIn("device.type: cuda", message)

    def test_cuda_configured_but_a_non_cuda_device_resolved_fails(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            with self.assertRaises(RuntimeError):
                assert_training_device("cuda", 0, torch.device("cpu"))

    def test_a_substituted_gpu_index_fails(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            with self.assertRaises(RuntimeError) as caught:
                assert_training_device("cuda", 1, torch.device("cuda", 0))
        self.assertIn("cuda:1", str(caught.exception))

    def test_a_matching_cuda_device_passes(self):
        device = torch.device("cuda", 1)
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            self.assertIs(assert_training_device("cuda", 1, device), device)

    def test_context_is_included_for_diagnosis(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaises(RuntimeError) as caught:
                assert_training_device("cuda", 0, torch.device("cpu"),
                                       context="run_base --task_type ks_si_er")
        self.assertIn("run_base", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
