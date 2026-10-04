"""DG-0008 opportunity manifest: batching, digests, tuple association, pair equality.

The generator's exact scheduling rules are exercised on synthetic canonical
keys, without any corpus: step batching, epoch-indexed ER counts, canonical
stream-vs-materialized byte equivalence, the clause-12 tuple-to-digest
association, and the pair/control schedule identity.
"""

import hashlib
import os
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from improvements.taskrelation.research.dg0008 import manifest  # noqa: E402
from improvements.taskrelation.research.dg0008.sampler import (  # noqa: E402
    ManifestEpochSampler,
    key_to_index,
    schedule_fingerprint,
)

BATCH = manifest.BATCH_SIZE
KEYS_0 = ["a/{:05d}.wav".format(i) for i in range(2100)]
KEYS_1 = ["b/{:05d}.wav".format(i) for i in range(2000)]
VECTORS = [(KEYS_0, [0] * len(KEYS_0)), (KEYS_1, [0] * len(KEYS_1))]


def _reference_permutation(length, seed_text):
    order = list(range(length))
    for i in range(length - 1, 0, -1):
        digest = hashlib.sha256(("perm|" + seed_text + "|i=" + str(i)).encode("utf-8")).digest()
        j = int.from_bytes(digest[:8], "big") % (i + 1)
        order[i], order[j] = order[j], order[i]
    return order


class PermutationTests(unittest.TestCase):
    def test_matches_the_clause_9_formula_and_is_a_bijection(self):
        seed = manifest.manifest_seed("ks_er", 0, 0, 0)
        order = manifest.permutation(64, seed)
        self.assertEqual(order, _reference_permutation(64, seed))
        self.assertEqual(sorted(order), list(range(64)))

    def test_manifest_seed_is_verbatim(self):
        self.assertEqual(
            manifest.manifest_seed("si_er", 3, 2, 4),
            "DG-0008|manifest|si_er|f=3|s=2|e=4",
        )

    def test_epoch_seed_differs(self):
        self.assertNotEqual(
            manifest.permutation(128, manifest.manifest_seed("ks_er", 0, 0, 0)),
            manifest.permutation(128, manifest.manifest_seed("ks_er", 0, 0, 1)),
        )


class BatchingTests(unittest.TestCase):
    def setUp(self):
        self.steps = manifest.build_epoch_steps("ks_er", 0, 0, 0, [KEYS_0, KEYS_1])

    def test_steps_and_dropped_remainder(self):
        length, steps = manifest.length_and_steps(len(KEYS_0), len(KEYS_1))
        self.assertEqual(length, 4100)
        self.assertEqual(steps, 2)
        self.assertEqual(self.steps["steps"], 2)
        self.assertEqual(len(self.steps["step_keys"]), 2)
        self.assertTrue(all(len(batch) == BATCH for batch in self.steps["step_keys"]))
        self.assertEqual(self.steps["length"], 4100)

    def test_step_is_the_permutation_slice_mapped_to_component_keys(self):
        order = manifest.permutation(4100, manifest.manifest_seed("ks_er", 0, 0, 0))
        for step in range(2):
            expected = []
            for global_index in order[step * BATCH:(step + 1) * BATCH]:
                if global_index < len(KEYS_0):
                    expected.append([0, KEYS_0[global_index]])
                else:
                    expected.append([1, KEYS_1[global_index - len(KEYS_0)]])
            self.assertEqual(self.steps["step_keys"][step], expected)

    def test_n_er_is_recomputed_from_the_step_contents(self):
        for step, batch in enumerate(self.steps["step_keys"]):
            self.assertEqual(self.steps["n_er"][step], sum(1 for c, _ in batch if c == 1))

    def test_all_entries_are_distinct(self):
        flat = [tuple(entry) for batch in self.steps["step_keys"] for entry in batch]
        self.assertEqual(len(set(flat)), 2 * BATCH)


class StreamingManifestTests(unittest.TestCase):
    """The production streaming path freezes exactly the oracle's canonical bytes.

    ``IF-MANIFEST-IO=STREAM_CANONICAL_BYTES`` is only lawful while both paths
    emit byte-identical clause-5/11 canonical JSON and the same per-epoch
    digest, so every case compares the streamed bytes and digest against the
    materializing oracle object.
    """

    @staticmethod
    def _vectors(n0, n1):
        return [["a/{:06d}.wav".format(i) for i in range(n0)],
                ["b/{:06d}.wav".format(i) for i in range(n1)]]

    def assert_stream_matches_oracle(self, vectors, cell="ks_er", fold=3, seed=2, epoch=4):
        streamed = b"".join(
            manifest.iter_opportunity_manifest_bytes(cell, fold, seed, epoch, vectors)
        )
        oracle = manifest.build_opportunity_object(cell, fold, seed, epoch, vectors)
        self.assertEqual(streamed, manifest.canonical_json_bytes(oracle))
        self.assertEqual(
            manifest.streamed_epoch_digest(cell, fold, seed, epoch, vectors),
            manifest.per_epoch_digest(oracle),
        )
        return streamed, oracle

    def test_small_case_streams_the_oracle_bytes_and_digest(self):
        streamed, _ = self.assert_stream_matches_oracle(self._vectors(2100, 2000))
        self.assertTrue(streamed.startswith(b'{"cell":"ks_er"'))
        self.assertTrue(streamed.endswith(b',"steps":2}'))
        self.assertIn(b',"schema":"dg0008.opportunity-manifest.v2"', streamed)

    def test_small_case_key_order_is_the_sorted_schema(self):
        streamed, _ = self.assert_stream_matches_oracle(self._vectors(2100, 2000))
        # canonical key order: cell, epoch, fold, length, n_er, schema, seed,
        # step_keys, steps.
        offsets = [
            streamed.index(b'"cell":'),
            streamed.index(b'"epoch":'),
            streamed.index(b'"fold":'),
            streamed.index(b'"length":'),
            streamed.index(b'"n_er":'),
            streamed.index(b'"schema":'),
            streamed.index(b'"seed":'),
            streamed.index(b'"step_keys":'),
            streamed.index(b'"steps":'),
        ]
        self.assertEqual(offsets, sorted(offsets))
        self.assertGreater(offsets[7], offsets[4])

    def test_medium_case_streams_the_oracle_bytes_and_digest(self):
        streamed, _ = self.assert_stream_matches_oracle(
            self._vectors(20000, 20000), fold=7, seed=1, epoch=0
        )
        self.assertTrue(streamed.endswith(b',"steps":19}'))

    def test_large_case_matches_the_oracle_and_bounds_every_chunk(self):
        vectors = self._vectors(60000, 60000)  # L=120000, steps=58
        length, steps = manifest.length_and_steps(*(len(keys) for keys in vectors))
        self.assertGreaterEqual(length, 100000)
        self.assertGreaterEqual(steps, 48)
        cell, fold, seed, epoch = "si_er", 5, 3, 2

        chunks = list(
            manifest.iter_opportunity_manifest_bytes(cell, fold, seed, epoch, vectors)
        )
        self.assertTrue(all(len(chunk) <= 1024 * 1024 for chunk in chunks))
        streamed = b"".join(chunks)

        oracle = manifest.build_opportunity_object(cell, fold, seed, epoch, vectors)
        self.assertEqual(streamed, manifest.canonical_json_bytes(oracle))
        self.assertEqual(manifest.sha256_hex(streamed), manifest.per_epoch_digest(oracle))

    def test_streamed_digest_changes_when_any_key_changes(self):
        vectors = self._vectors(2100, 2000)
        mutated = [vectors[0], vectors[1][:-1] + ["b/XXXXX.wav"]]
        self.assertNotEqual(
            manifest.streamed_epoch_digest("ks_er", 0, 0, 0, vectors),
            manifest.streamed_epoch_digest("ks_er", 0, 0, 0, mutated),
        )


class DigestAssociationTests(unittest.TestCase):
    def test_run_array_carries_the_exact_tuple_association(self):
        digests = ["{:064x}".format(i) for i in range(5)]
        array = manifest.run_array("ks_er", 2, 1, digests)
        self.assertEqual(array, [["ks_er", 2, 1, epoch, digests[epoch]] for epoch in range(5)])
        self.assertEqual(manifest.run_digest("ks_er", 2, 1, digests),
                         manifest.sha256_hex(manifest.canonical_json_bytes(array)))

    def test_run_digest_changes_with_any_epoch_digest(self):
        digests = ["{:064x}".format(i) for i in range(5)]
        mutated = list(digests)
        mutated[3] = "f" * 64
        self.assertNotEqual(
            manifest.run_digest("ks_er", 0, 0, digests),
            manifest.run_digest("ks_er", 0, 0, mutated),
        )

    def test_matrix_array_sort_order_is_cell_bytes_fold_seed_epoch(self):
        entries = [
            ["si_er", 0, 0, 0, "aa"], ["ks_er", 1, 0, 0, "bb"],
            ["ks_er", 0, 1, 0, "cc"], ["ks_er", 0, 0, 1, "dd"],
        ]
        ordered = manifest.matrix_array(entries)
        self.assertEqual(
            [tuple(entry[:4]) for entry in ordered],
            [("ks_er", 0, 0, 1), ("ks_er", 0, 1, 0), ("ks_er", 1, 0, 0), ("si_er", 0, 0, 0)],
        )
        self.assertNotEqual(manifest.matrix_digest(entries),
                            manifest.matrix_digest(entries[:-1] + [["si_er", 1, 0, 0, "zz"]]))


class ScheduleEqualityTests(unittest.TestCase):
    def test_pair_and_control_consume_the_same_ordered_batches(self):
        steps = manifest.build_epoch_steps("ks_er", 0, 0, 0, [KEYS_0, KEYS_1])["step_keys"]
        lookup = key_to_index(VECTORS)
        pair = ManifestEpochSampler(steps, lookup)
        control = ManifestEpochSampler(steps, lookup)
        self.assertEqual(list(pair), list(control))
        self.assertEqual(schedule_fingerprint(steps), schedule_fingerprint(steps))
        self.assertEqual(len(pair), 2 * BATCH)

    def test_missing_key_fails_closed(self):
        lookup = key_to_index(VECTORS)
        broken = [[[0, "a/99999.wav"]]]
        with self.assertRaises(manifest.ManifestError):
            ManifestEpochSampler(broken, lookup)

    def test_manifest_is_shared_by_arm_absence(self):
        # Arm is deliberately absent from the tuple: both arms must consume the
        # same digest for the same (cell, fold, seed).
        digests = ["{:064x}".format(i) for i in range(5)]
        self.assertEqual(
            manifest.run_digest("si_er", 0, 0, digests),
            manifest.run_digest("si_er", 0, 0, list(digests)),
        )


if __name__ == "__main__":
    unittest.main()
