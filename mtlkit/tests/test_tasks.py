"""Unit tests for mtlkit/tasks.py.

Includes the exact-parity check Next Step 1 calls for: re-deriving the
current 4-task `index_pattern` masking behavior from
`downstream/dataset/load_embedding.py` bit-for-bit, not just structurally.
"""

import os
import sys
import unittest

import mtlkit.tasks as tasks

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DOWNSTREAM_DIR = os.path.join(REPO_ROOT, "downstream")
for path in (DOWNSTREAM_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from utils.constant_mapping import TaskDatasetMapping, TaskKeywordMapping  # noqa: E402


class TaskRegistryTests(unittest.TestCase):
    def test_builtin_tasks_registered(self):
        self.assertEqual(tasks.TASK_REGISTRY.list(), ["er", "ic", "ks", "si"])

    def test_task_spec_num_classes(self):
        self.assertEqual(tasks.TASK_REGISTRY.get("ks").num_classes, 12)
        self.assertEqual(tasks.TASK_REGISTRY.get("si").num_classes, 1251)
        self.assertEqual(tasks.TASK_REGISTRY.get("er").num_classes, 4)
        self.assertEqual(tasks.TASK_REGISTRY.get("ic").num_classes, 31)

    def test_duplicate_registration_rejected(self):
        with self.assertRaises(KeyError):
            tasks.register_task(
                tasks.TaskSpec("ks", "dup", "speechcommand", {"a": 0})
            )


class TaskTypeParsingTests(unittest.TestCase):
    def test_dataset_array_orders_and_dedupes(self):
        self.assertEqual(
            tasks.dataset_array_from_task_type("ks_si_er_ic"),
            ["speechcommand", "voxceleb", "iemocap", "fluentspeechcommand"],
        )

    def test_dataset_array_dedupes_repeated_tokens(self):
        self.assertEqual(
            tasks.dataset_array_from_task_type("ks_ks_si"),
            ["speechcommand", "voxceleb"],
        )

    def test_task_name_array(self):
        self.assertEqual(
            tasks.task_name_array_from_task_type("ks_si"),
            ["Keyword Spotting", "Speaker Identification"],
        )

    def test_invalid_token_raises_valueerror_matching_load_embedding_message(self):
        with self.assertRaises(ValueError) as ctx:
            tasks.dataset_array_from_task_type("ks_bogus")
        self.assertEqual(str(ctx.exception), "Invalid task type: bogus")

    def test_invalid_token_in_task_name_array_raises(self):
        with self.assertRaises(ValueError) as ctx:
            tasks.task_name_array_from_task_type("bogus")
        self.assertEqual(str(ctx.exception), "Invalid task type: bogus")


class IndexPatternTests(unittest.TestCase):
    def test_4task_index_pattern_map(self):
        self.assertEqual(
            tasks.index_pattern_map_for_task_type("ks_si_er_ic"),
            {
                "speechcommand": "1000",
                "voxceleb": "0100",
                "iemocap": "0010",
                "fluentspeechcommand": "0001",
            },
        )

    def test_3task_index_pattern_map(self):
        self.assertEqual(
            tasks.index_pattern_map_for_task_type("ks_si_er"),
            {"speechcommand": "100", "voxceleb": "010", "iemocap": "001"},
        )

    def test_single_task_index_pattern_is_trivial(self):
        self.assertEqual(tasks.index_pattern_map_for_task_type("ks"), {"speechcommand": "1"})

    def test_decode_index_pattern(self):
        self.assertEqual(tasks.decode_index_pattern("0100"), [False, True, False, False])

    def test_decode_index_pattern_supports_multiple_true(self):
        # FSC's 3-simultaneous-valid-positions case (Next Step 8) — the
        # decode function makes no "exactly one 1" assumption.
        self.assertEqual(tasks.decode_index_pattern("0111"), [False, True, True, True])

    def test_task_validity_mask(self):
        self.assertEqual(
            tasks.task_validity_mask("voxceleb", "ks_si_er"), [False, True, False]
        )

    def test_task_validity_mask_unknown_dataset_raises(self):
        with self.assertRaises(KeyError) as ctx:
            tasks.task_validity_mask("iemocap", "ks_si")
        self.assertIn("not one of the datasets", str(ctx.exception))


class ExactParityWithLoadEmbeddingTests(unittest.TestCase):
    """Re-derive downstream/dataset/load_embedding.py's current behavior
    exactly, per Next Step 1's validation criterion."""

    def _reference_dataset_array(self, task_type):
        task_tokens = task_type.split("_")
        dataset_keys = []
        for t in task_tokens:
            ds = TaskDatasetMapping.get_dataset_key(t)
            if ds is None:
                raise ValueError(f"Invalid task type: {t}")
            if ds not in dataset_keys:
                dataset_keys.append(ds)
        return dataset_keys

    def _reference_task_name_array(self, task_type):
        task_tokens = task_type.split("_")
        task_keys = []
        for t in task_tokens:
            tk = TaskKeywordMapping.get_task_name(t)
            if tk is None:
                raise ValueError(f"Invalid task type: {t}")
            if tk not in task_keys:
                task_keys.append(tk)
        return task_keys

    def _reference_index_pattern_map(self, task_type):
        dataset_id_array = self._reference_dataset_array(task_type)
        pattern_len = len(dataset_id_array)
        index_pattern_map = {}
        for i, ds in enumerate(dataset_id_array):
            bits = ["0"] * pattern_len
            bits[i] = "1"
            index_pattern_map[ds] = "".join(bits)
        return index_pattern_map

    def test_dataset_array_matches_reference_for_every_permutation(self):
        import itertools

        for perm in itertools.permutations(["ks", "si", "er", "ic"]):
            task_type = "_".join(perm)
            self.assertEqual(
                tasks.dataset_array_from_task_type(task_type),
                self._reference_dataset_array(task_type),
                msg=f"mismatch for task_type={task_type!r}",
            )

    def test_task_name_array_matches_reference(self):
        for task_type in ("ks", "ks_si", "ks_si_er", "ks_si_er_ic", "ic_ks"):
            self.assertEqual(
                tasks.task_name_array_from_task_type(task_type),
                self._reference_task_name_array(task_type),
                msg=f"mismatch for task_type={task_type!r}",
            )

    def test_index_pattern_map_matches_reference(self):
        for task_type in ("ks", "ks_si", "ks_si_er", "ks_si_er_ic", "ic_er_si_ks"):
            self.assertEqual(
                tasks.index_pattern_map_for_task_type(task_type),
                self._reference_index_pattern_map(task_type),
                msg=f"mismatch for task_type={task_type!r}",
            )

    def test_invalid_token_error_message_matches_reference(self):
        with self.assertRaises(ValueError) as mtlkit_ctx:
            tasks.dataset_array_from_task_type("ks_nope")
        with self.assertRaises(ValueError) as reference_ctx:
            self._reference_dataset_array("ks_nope")
        self.assertEqual(str(mtlkit_ctx.exception), str(reference_ctx.exception))


if __name__ == "__main__":
    unittest.main()
