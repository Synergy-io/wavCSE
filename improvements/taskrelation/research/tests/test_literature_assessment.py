"""The canonical PaperAssessment registry: one investigation-scoped authority.

Covers validation (identity, vocabularies, gates, anchors), the legacy
private-key normalisation, the real-registry coverage of LT-0001/LT-0002, and the
transitional equivalence with the legacy representations it will replace.
"""

import json
import tempfile
import unittest
from pathlib import Path

from improvements.taskrelation.research import literature_assessment
from improvements.taskrelation.research import literature_assessment_equivalence
from improvements.taskrelation.research import literature_catalog


REPO_ROOT = Path(__file__).resolve().parents[4]
RESEARCH_DIR = REPO_ROOT / "improvements" / "taskrelation" / "research"
LITERATURE_DIR = RESEARCH_DIR / "literature"
STUDIES_PATH = RESEARCH_DIR / "STUDIES.jsonl"
ASSESSMENTS_PATH = LITERATURE_DIR / "assessments.jsonl"


def paper_record(paper_id, title, url):
    return {
        "schema_version": 1,
        "paper_id": paper_id,
        "card_path": "research/literature/{}.md".format(paper_id),
        "title": title,
        "year": 2024,
        "authors": ["A. Author"],
        "venue": "Venue",
        "external_ids": {},
        "source_urls": [url],
        "aliases": [],
    }


def valid_row(**overrides):
    row = {
        "schema_version": 1,
        "investigation_id": "LT-0002",
        "paper_id": "beta-2020-method",
        "role": "family_b_estimator",
        "gates": [],
        "verdict": "pass",
        "reason_summary": "Clean pass.",
        "assessment_anchor": "research/literature/beta-2020-method.md#lt-0002-assessment",
        "assessed_at": "2026-01-02T00:00:00+00:00",
    }
    row.update(overrides)
    return row


class AssessmentFixture(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="literature-assessment-")
        self.repo_root = Path(self.tempdir.name)
        self.research_dir = self.repo_root / "research"
        self.literature_dir = self.research_dir / "literature"
        self.studies_dir = self.research_dir / "studies"
        self.literature_dir.mkdir(parents=True)
        self.studies_dir.mkdir()
        self._write_catalog()
        self._write_studies()
        for paper_id in ("alpha-2024-method", "beta-2020-method", "gamma-2024-method"):
            (self.literature_dir / (paper_id + ".md")).write_text(
                "# {}\n".format(paper_id), encoding="utf-8"
            )
        self.assessments_path = self.literature_dir / "assessments.jsonl"

    def tearDown(self):
        self.tempdir.cleanup()

    def _write_catalog(self):
        records = [
            paper_record("alpha-2024-method", "Alpha Method", "https://example.org/a"),
            paper_record("beta-2020-method", "Beta Method", "https://example.org/b"),
            paper_record("gamma-2024-method", "Gamma Method", "https://example.org/g"),
        ]
        (self.literature_dir / "catalog.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def _write_studies(self):
        studies = [
            {
                "study_id": "DG-0001",
                "type": "diagnostic",
                "title": "Not literature",
                "status": "complete",
                "stage": "analysis",
                "path": "research/studies/DG-0001",
            },
            {
                "study_id": "LT-0001",
                "type": "literature",
                "title": "First literature question",
                "status": "rejected",
                "stage": "primary_source_review",
                "path": "research/studies/LT-0001",
            },
            {
                "study_id": "LT-0002",
                "type": "literature",
                "title": "Second literature question",
                "status": "complete",
                "stage": "verification",
                "path": "research/studies/LT-0002",
            },
        ]
        (self.research_dir / "STUDIES.jsonl").write_text(
            "".join(json.dumps(study) + "\n" for study in studies), encoding="utf-8"
        )
        for study_id in ("DG-0001", "LT-0001", "LT-0002"):
            (self.studies_dir / study_id).mkdir()

    def load(self, rows):
        self.assessments_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return literature_assessment.load_assessments(
            self.assessments_path,
            repo_root=self.repo_root,
            literature_dir=self.literature_dir,
            studies_path=self.research_dir / "STUDIES.jsonl",
        )


class AssessmentValidationTests(AssessmentFixture):
    def test_valid_record_loads_and_exposes_exact_lookup(self):
        registry = self.load([valid_row()])

        record = registry.get_assessment("LT-0002", "beta-2020-method")

        self.assertEqual(record.assessment_ref, "LT-0002#beta-2020-method")
        self.assertEqual(record.role, "family_b_estimator")
        self.assertEqual(record.verdict, "pass")
        self.assertEqual(record.detail_anchor, "lt-0002-assessment")

    def test_unknown_paper_id_fails(self):
        with self.assertRaisesRegex(
            literature_assessment.AssessmentError, "unknown paper_id 'missing'"
        ):
            self.load([valid_row(paper_id="missing")])

    def test_unknown_investigation_fails(self):
        with self.assertRaisesRegex(
            literature_assessment.AssessmentError,
            "not a registered LT-\\* literature Study",
        ):
            self.load([valid_row(investigation_id="LT-9999")])

    def test_non_literature_study_is_not_an_assessment_investigation(self):
        with self.assertRaisesRegex(
            literature_assessment.AssessmentError,
            "not a registered LT-\\* literature Study",
        ):
            self.load([valid_row(investigation_id="DG-0001")])

    def test_duplicate_investigation_paper_pair_fails(self):
        with self.assertRaisesRegex(
            literature_assessment.AssessmentError,
            "duplicate paper assessment 'LT-0002#beta-2020-method'",
        ):
            self.load([valid_row(), valid_row()])

    def test_role_vocabulary_is_validated(self):
        with self.assertRaisesRegex(
            literature_assessment.AssessmentError, "not a known role"
        ):
            self.load([valid_row(role="family_c")])

    def test_verdict_vocabulary_is_validated(self):
        with self.assertRaisesRegex(
            literature_assessment.AssessmentError, "not a known verdict"
        ):
            self.load([valid_row(verdict="maybe")])

    def test_gate_id_vocabulary_and_uniqueness_are_validated(self):
        with self.assertRaisesRegex(
            literature_assessment.AssessmentError, "is not a known gate"
        ):
            self.load([valid_row(gates=[{"gate_id": "vibes", "verdict": "pass"}])])
        with self.assertRaisesRegex(
            literature_assessment.AssessmentError, "repeats gate_id"
        ):
            self.load(
                [
                    valid_row(
                        gates=[
                            {"gate_id": "taxonomy", "verdict": "pass"},
                            {"gate_id": "taxonomy", "verdict": "fail"},
                        ]
                    )
                ]
            )

    def test_gate_verdict_vocabulary_is_validated(self):
        with self.assertRaisesRegex(
            literature_assessment.AssessmentError, "not a known gate verdict"
        ):
            self.load(
                [valid_row(gates=[{"gate_id": "taxonomy", "verdict": "maybe"}])]
            )

    def test_a_global_screening_status_key_is_refused(self):
        with self.assertRaisesRegex(
            literature_assessment.AssessmentError, "unknown key"
        ):
            self.load([valid_row(screening_status="included")])

    def test_anchor_must_point_at_the_papers_own_card(self):
        with self.assertRaisesRegex(
            literature_assessment.AssessmentError, "canonical card"
        ):
            self.load(
                [
                    valid_row(
                        assessment_anchor=(
                            "research/literature/alpha-2024-method.md"
                            "#lt-0002-assessment"
                        )
                    )
                ]
            )

    def test_assessed_at_must_be_a_timezoned_iso_timestamp(self):
        with self.assertRaisesRegex(
            literature_assessment.AssessmentError, "ISO 8601"
        ):
            self.load([valid_row(assessed_at="yesterday")])

    def test_records_must_be_sorted_by_investigation_then_paper(self):
        with self.assertRaisesRegex(
            literature_assessment.AssessmentError, "sorted by investigation_id"
        ):
            self.load(
                [
                    valid_row(
                        paper_id="gamma-2024-method",
                        assessment_anchor=(
                            "research/literature/gamma-2024-method.md"
                            "#lt-0002-assessment"
                        ),
                    ),
                    valid_row(paper_id="beta-2020-method"),
                ]
            )

    def test_registry_queries_are_investigation_scoped(self):
        registry = self.load(
            [
                valid_row(
                    investigation_id="LT-0001",
                    paper_id="beta-2020-method",
                    role="direct_gradient_scale_balancing",
                    verdict="exclude_taxonomy",
                    assessment_anchor=(
                        "research/literature/beta-2020-method.md#lt-0001-decision"
                    ),
                ),
                valid_row(paper_id="beta-2020-method"),
            ]
        )

        self.assertEqual(len(registry.assessments_for_paper("beta-2020-method")), 2)
        self.assertEqual(
            [
                record.paper_id
                for record in registry.assessments_for_investigation("LT-0001")
            ],
            ["beta-2020-method"],
        )
        with self.assertRaisesRegex(
            literature_assessment.AssessmentError, "unknown paper_id"
        ):
            registry.assessments_for_paper("missing")
        with self.assertRaisesRegex(
            literature_assessment.AssessmentError, "unknown literature investigation"
        ):
            registry.assessments_for_investigation("LT-9999")


class RealRegistryTests(unittest.TestCase):
    """The backfilled registry against the real repository state."""

    @classmethod
    def setUpClass(cls):
        cls.registry = literature_assessment.load_assessments()
        cls.catalog = literature_catalog.load_catalog(
            LITERATURE_DIR / "catalog.jsonl",
            repo_root=REPO_ROOT,
            validate_references=False,
        )

    def test_every_assessment_uses_a_canonical_paper_id(self):
        paper_ids = {entry.paper_id for entry in self.catalog.entries}
        for record in self.registry.records:
            with self.subTest(paper_id=record.paper_id):
                self.assertIn(record.paper_id, paper_ids)
                self.assertNotIn("_", record.paper_id)

    def test_every_investigation_is_a_registered_lt_study(self):
        studies = {
            json.loads(line)["study_id"]
            for line in STUDIES_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        for record in self.registry.records:
            with self.subTest(investigation_id=record.investigation_id):
                self.assertIn(record.investigation_id, studies)
                self.assertTrue(record.investigation_id.startswith("LT-"))

    def test_all_lt_0001_assessed_papers_are_represented(self):
        lt0001 = self.registry.assessments_for_investigation("LT-0001")
        reviewed = set(
            literature_assessment_equivalence.private_key_map().values()
        )

        self.assertEqual(len(lt0001), 8)
        self.assertEqual({record.paper_id for record in lt0001}, reviewed)

    def test_all_lt_0002_assessed_papers_are_represented(self):
        lt0002 = self.registry.assessments_for_investigation("LT-0002")
        studies = [
            json.loads(line)
            for line in STUDIES_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        registered = next(
            study["cards"] for study in studies if study["study_id"] == "LT-0002"
        )

        self.assertEqual(len(lt0002), 15)
        self.assertEqual({record.paper_id for record in lt0002}, set(registered))

    def test_catalog_still_carries_no_global_screening_status(self):
        for entry in self.catalog.entries:
            with self.subTest(paper_id=entry.paper_id):
                for forbidden in ("status", "decision", "verdict", "role", "gates"):
                    self.assertFalse(hasattr(entry, forbidden))

    def test_paper_cards_keep_their_legacy_verdict_sections(self):
        # INC-V2-2 does not de-verdict cards; they remain migration witnesses.
        for record in self.registry.records:
            card = REPO_ROOT / record.detail_path
            text = card.read_text(encoding="utf-8")
            heading = "## {}".format(
                "LT-0001 decision"
                if record.investigation_id == "LT-0001"
                else "LT-0002 assessment"
            )
            with self.subTest(paper_id=record.paper_id):
                self.assertIn("## Candidate Study ID", text)
                self.assertIn(heading, text)


class LegacyPrivateKeyTests(unittest.TestCase):
    def test_private_lt_0001_keys_map_to_canonical_paper_ids(self):
        mapping = literature_assessment_equivalence.private_key_map()

        self.assertEqual(
            mapping["zhang_yang_2021_survey"], "zhang-yang-2021-mtl-survey"
        )
        self.assertEqual(mapping["chen_2018_gradnorm"], "chen-et-al-2018-gradnorm")
        self.assertEqual(len(mapping), 8)
        for key, paper_id in mapping.items():
            with self.subTest(key=key):
                self.assertNotIn("_", paper_id)
                self.assertFalse(key == paper_id)


class TransitionalConsistencyTests(unittest.TestCase):
    """While the legacy representations remain, they must agree with the registry.

    This is the INC-V2-2 consistency gate: it fails if a card verdict, a
    ``STUDIES.jsonl`` ``cards`` list or an LT result relationship changes without
    the canonical registry (or the reverse).
    """

    def test_canonical_registry_matches_every_legacy_representation(self):
        rows = literature_assessment_equivalence.check_equivalence()

        self.assertEqual(len(rows), 23)

    def test_a_drifted_registry_is_detected(self):
        # A registry whose verdict disagrees with the card must fail the checker.
        rows = [
            json.loads(line)
            for line in ASSESSMENTS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in rows:
            if row["paper_id"] == "goncalves-2016-mssl":
                row["verdict"] = "fail"
        with tempfile.TemporaryDirectory(prefix="assessment-drift-") as tempdir:
            drifted = Path(tempdir) / "assessments.jsonl"
            drifted.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                literature_assessment_equivalence.AssessmentEquivalenceError,
                "disagrees with the legacy",
            ) as caught:
                literature_assessment_equivalence.check_equivalence(
                    assessments_path=drifted
                )

        self.assertTrue(
            any("goncalves-2016-mssl" in diff for diff in caught.exception.differences),
            caught.exception.differences,
        )


if __name__ == "__main__":
    unittest.main()
