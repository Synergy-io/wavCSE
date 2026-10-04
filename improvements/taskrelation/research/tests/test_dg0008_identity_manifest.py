"""DG-0008 identity manifest: canonical keys, labels, splits, folds, exact bytes.

These tests exercise the frozen contract on synthetic metadata, never on a
corpus or embedding, and assert the consumer-visible invariants the Stage-1
gate depends on: byte-identical canonical JSON, UTF-8 key ordering, the
dataset-specific label/split/fold semantics, and dual-identity admission.
"""

import os
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
for path in (REPO_ROOT, os.path.join(REPO_ROOT, "downstream")):
    if path not in sys.path:
        sys.path.insert(0, path)

from improvements.taskrelation.research.dg0008 import manifest  # noqa: E402

IEMOCAP_MAP = {"ang": 2, "hap": 1, "neu": 0, "sad": 3}
SC_MAP = {"bed": 0, "cat": 1, "_unknown_": 10, "_silence_": 11}
VOX_MAP = {"1": 0, "2": 1}


class CanonicalKeyTests(unittest.TestCase):
    def test_backslash_is_normalized_and_segments_are_preserved(self):
        self.assertEqual(
            manifest.canonical_manifest_key("Session1\\sentences\\wav\\a.wav"),
            "Session1/sentences/wav/a.wav",
        )

    def test_rejected_forms(self):
        for path in ("/abs/a.wav", "C:sess/a.wav", "a//b.wav", "a/./b.wav",
                     "a/../b.wav", "a/b/", "\x00a.wav", ""):
            with self.subTest(path=path):
                with self.assertRaises(manifest.ManifestError):
                    manifest.canonical_manifest_key(path)

    def test_no_case_fold_percent_decode_or_unicode_normalization(self):
        # Percent-encoded text stays literal; composed and decomposed forms differ.
        self.assertEqual(manifest.canonical_manifest_key("a%2Fb.wav"), "a%2Fb.wav")
        self.assertNotEqual(
            manifest.canonical_manifest_key("\u00e9.wav"),
            manifest.canonical_manifest_key("e\u0301.wav"),
        )

    def test_ordering_is_utf8_byte_order(self):
        keys = ["a.wav", "Z.wav", "\u00e9.wav", "b.wav"]
        self.assertEqual(
            manifest.sort_keys(keys),
            sorted(keys, key=lambda key: key.encode("utf-8")),
        )
        self.assertEqual(manifest.sort_keys(keys)[0], "Z.wav")


class IdentityRecordTests(unittest.TestCase):
    def test_speechcommand_unknown_maps_to_unknown_class(self):
        known = manifest.speechcommand_record("bed/a.wav", "bed", SC_MAP)
        self.assertEqual(known["effective_label"], "bed")
        self.assertEqual(known["effective_label_index"], 0)
        unknown = manifest.speechcommand_record("bed/b.wav", "bird", SC_MAP)
        self.assertEqual(unknown["effective_label"], "_unknown_")
        self.assertEqual(unknown["effective_label_index"], 10)

    def test_voxceleb_speaker_index_and_missing_id(self):
        record = manifest.voxceleb_record("wav/id1/a.wav", "1", VOX_MAP)
        self.assertEqual(record["effective_speaker_id"], "1")
        self.assertEqual(record["effective_label_index"], 0)
        missing = manifest.voxceleb_record("wav/id9/a.wav", "9", VOX_MAP)
        self.assertEqual(missing["effective_label_index"], -1)

    def test_iemocap_exc_is_hap_fru_is_rejected_and_speaker_parsed(self):
        record = manifest.iemocap_record(
            "Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav", "exc", IEMOCAP_MAP
        )
        self.assertEqual(record["effective_label"], "hap")
        self.assertEqual(record["effective_label_index"], 1)
        self.assertEqual(record["speaker_id"], "Ses01F")
        with self.assertRaises(manifest.ManifestError):
            manifest.iemocap_record(
                "Session1/sentences/wav/Ses01F_impro01/x.wav", "fru", IEMOCAP_MAP
            )

    def test_real_label_mappings_shape(self):
        from utils.constant_mapping import LabelKeywordMapping

        self.assertNotIn("exc", LabelKeywordMapping.LABEL2INDEX_IEMOCAP)
        self.assertEqual(LabelKeywordMapping.LABEL2INDEX_IEMOCAP["hap"], 1)
        self.assertEqual(LabelKeywordMapping.LABEL2INDEX_SPEECHCOMMANDv1["_unknown_"], 10)
        self.assertEqual(LabelKeywordMapping.LABEL2INDEX_VOXCELEB1["1"], 0)
        self.assertEqual(LabelKeywordMapping.LABEL2INDEX_VOXCELEB1["1251"], 1250)


def _iemocap_records():
    speakers = ["Ses01F", "Ses01M", "Ses02F", "Ses02M", "Ses03F",
                "Ses03M", "Ses04F", "Ses04M", "Ses05F", "Ses05M"]
    records = []
    for speaker in speakers:
        records.append(
            manifest.iemocap_record(
                "Session{}/sentences/wav/{}_{}/{}_x.wav".format(
                    speaker[3], speaker, "impro01", speaker
                ),
                "neu",
                IEMOCAP_MAP,
            )
        )
        records[-1]["speaker_id"] = speaker
    return records


class FoldTests(unittest.TestCase):
    def setUp(self):
        self.records = _iemocap_records()
        self.folds = manifest.build_iemocap_folds(
            record["speaker_id"] for record in self.records
        )

    def test_fold_rule_is_test_f_validation_f_plus_one(self):
        speakers = manifest.sort_keys(record["speaker_id"] for record in self.records)
        for entry in self.folds:
            self.assertEqual(entry["test_speaker"], speakers[entry["fold"]])
            self.assertEqual(
                entry["validation_speaker"], speakers[(entry["fold"] + 1) % 10]
            )
            self.assertNotEqual(entry["test_speaker"], entry["validation_speaker"])

    def test_every_speaker_is_test_once_and_validation_once(self):
        self.assertEqual(sorted(e["test_speaker"] for e in self.folds),
                         sorted(e["validation_speaker"] for e in self.folds))
        self.assertEqual(len({e["test_speaker"] for e in self.folds}), 10)

    def test_membership_has_ten_ordered_entries_with_only_known_splits(self):
        membership = manifest.fold_membership("Ses01F", self.folds)
        self.assertEqual([e["fold"] for e in membership], list(range(10)))
        self.assertTrue(all(e["split"] in ("train", "validation", "test") for e in membership))
        self.assertEqual(membership[0]["split"], "test")
        # Ses01F (speakers[0]) is the rotating validation speaker in fold 9.
        self.assertEqual(membership[9]["split"], "validation")


class IdentityObjectTests(unittest.TestCase):
    def _identity(self):
        sc = [
            dict(manifest.speechcommand_record("bed/a.wav", "bed", SC_MAP), official_split="training"),
            dict(manifest.speechcommand_record("bed/b.wav", "bed", SC_MAP), official_split="validation"),
        ]
        vox = [
            dict(manifest.voxceleb_record("wav/id1/a.wav", "1", VOX_MAP), official_split="train"),
        ]
        er = manifest.attach_fold_membership(
            _iemocap_records(),
            manifest.build_iemocap_folds(record["speaker_id"] for record in _iemocap_records()),
        )
        return manifest.build_identity_object(sc, vox, er)

    def test_component_order_record_sort_and_schema(self):
        identity = self._identity()
        self.assertEqual(
            [entry["component"] for entry in identity["components"]],
            ["speechcommand", "voxceleb", "iemocap"],
        )
        self.assertEqual(identity["schema"], "dg0008.example-identity.v1")
        for entry in identity["components"]:
            keys = [record["key"] for record in entry["records"]]
            self.assertEqual(keys, manifest.sort_keys(keys))

    def test_duplicate_key_is_fatal(self):
        sc = [dict(manifest.speechcommand_record("bed/a.wav", "bed", SC_MAP), official_split="training"),
              dict(manifest.speechcommand_record("bed\\a.wav", "bed", SC_MAP), official_split="training")]
        er = manifest.attach_fold_membership(
            _iemocap_records(),
            manifest.build_iemocap_folds(r["speaker_id"] for r in _iemocap_records()),
        )
        with self.assertRaises(manifest.ManifestError):
            manifest.build_identity_object(sc, [], er)

    def test_identity_digest_is_stable_and_matches_bytes(self):
        identity = self._identity()
        digest = manifest.identity_digest(identity)
        self.assertEqual(digest, manifest.sha256_hex(manifest.identity_bytes(identity)))
        self.assertEqual(len(digest), 64)
        self.assertNotEqual(digest, manifest.identity_digest(self._identity_variant(identity)))

    def _identity_variant(self, identity):
        import copy

        variant = copy.deepcopy(identity)
        variant["components"][0]["records"][0]["effective_label"] = "different"
        return variant

    def test_split_membership_uses_official_and_fold_splits(self):
        identity = self._identity()
        sc_keys, _ = manifest.records_for_split(identity, "speechcommand", 0, "train")
        self.assertEqual(sc_keys, ["bed/a.wav"])
        sc_val, _ = manifest.records_for_split(identity, "speechcommand", 0, "validation")
        self.assertEqual(sc_val, ["bed/b.wav"])
        vox_keys, _ = manifest.records_for_split(identity, "voxceleb", 0, "train")
        self.assertEqual(vox_keys, ["wav/id1/a.wav"])
        er_train, _ = manifest.records_for_split(identity, "iemocap", 0, "train")
        er_test, _ = manifest.records_for_split(identity, "iemocap", 0, "test")
        self.assertEqual(len(er_test), 1)
        self.assertEqual(len(er_train), 8)
        self.assertNotIn(er_test[0], er_train)

    def test_official_split_tokens_differ_per_component(self):
        self.assertEqual(manifest.OFFICIAL_SPLIT_TOKEN["speechcommand"]["train"], "training")
        self.assertEqual(manifest.OFFICIAL_SPLIT_TOKEN["voxceleb"]["train"], "train")
        self.assertEqual(manifest.OFFICIAL_SPLIT_TOKEN["voxceleb"]["validation"], "dev")
        self.assertEqual(manifest.OFFICIAL_SPLIT_TOKEN["speechcommand"]["test"], "testing")


class DualIdentityTests(unittest.TestCase):
    def test_bytes_and_digest_must_match(self):
        payload = manifest.canonical_json_bytes({"a": 1})
        digest = manifest.sha256_hex(payload)
        self.assertEqual(manifest.assert_identity_bytes(payload, digest), digest)
        with self.assertRaises(manifest.ManifestError):
            manifest.assert_identity_bytes(payload + b" ", digest)

    def test_canonical_bytes_have_no_spaces_or_escapes(self):
        self.assertEqual(manifest.canonical_json_bytes({"b": 1, "a": "x"}), b'{"a":"x","b":1}')
        self.assertEqual(
            manifest.canonical_json_bytes({"k": "\u00e9"}), '{"k":"\u00e9"}'.encode("utf-8")
        )


def _load_generator():
    import importlib.util

    path = os.path.join(
        REPO_ROOT,
        "improvements", "taskrelation", "research", "studies", "DG-0008",
        "generate_opportunity_manifest.py",
    )
    spec = importlib.util.spec_from_file_location("dg0008_generator_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GeneratorAssemblyTests(unittest.TestCase):
    """The generator's identity assembly must attach fold membership itself."""

    def test_build_identity_attaches_ten_entry_fold_membership(self):
        generator = _load_generator()
        speakers = ["Ses01F", "Ses01M", "Ses02F", "Ses02M", "Ses03F",
                    "Ses03M", "Ses04F", "Ses04M", "Ses05F", "Ses05M"]

        def sc_extractor(root, mapping):
            records = []
            for index in range(3):
                record = manifest.speechcommand_record(
                    "bed/{:03d}.wav".format(index), "bed", {"bed": 0, "_unknown_": 10}
                )
                record["official_split"] = "training"
                records.append(record)
            return records

        def vox_extractor(root, mapping):
            record = manifest.voxceleb_record("wav/id1/a.wav", "1", {"1": 0})
            record["official_split"] = "train"
            return [record]

        def er_extractor(root, mapping):
            return [
                manifest.iemocap_record(
                    "Session{}/sentences/wav/{}_{}/{}__{:03d}.wav".format(
                        speaker[3], speaker, "impro01", speaker, index
                    ),
                    "neu",
                    {"ang": 2, "hap": 1, "neu": 0, "sad": 3},
                )
                for speaker in speakers
                for index in range(2)
            ]

        identity = generator.build_identity(
            "/nonexistent",
            extractors={
                "speechcommand": sc_extractor,
                "voxceleb": vox_extractor,
                "iemocap": er_extractor,
            },
        )
        self.assertEqual(len(identity["iemocap_folds"]), 10)
        er_records = manifest.component_records(identity, "iemocap")
        self.assertEqual(len(er_records), 20)
        for record in er_records:
            self.assertEqual(
                [entry["fold"] for entry in record["fold_membership"]], list(range(10))
            )
            self.assertTrue(
                all(entry["split"] in ("train", "validation", "test")
                    for entry in record["fold_membership"])
            )
            self.assertIn("validation", [entry["split"] for entry in record["fold_membership"]])



if __name__ == "__main__":
    unittest.main()
