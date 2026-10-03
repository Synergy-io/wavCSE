"""The canonical PaperAssessment registry: one investigation-scoped authority.

Covers validation (identity, vocabularies, gates, investigation reasoning
anchors), the real-registry coverage of LT-0001/LT-0002, the de-verdicting of the
PaperCards, and the steady-state authority checks that replaced the INC-V2-2
transitional equivalence with the legacy witnesses.
"""

import json
import tempfile
import unittest
from pathlib import Path

from improvements.taskrelation.research import literature_assessment
from improvements.taskrelation.research import literature_catalog
from improvements.taskrelation.research import literature_claims


REPO_ROOT = Path(__file__).resolve().parents[4]
RESEARCH_DIR = REPO_ROOT / "improvements" / "taskrelation" / "research"
LITERATURE_DIR = RESEARCH_DIR / "literature"
STUDIES_PATH = RESEARCH_DIR / "STUDIES.jsonl"
ASSESSMENTS_PATH = LITERATURE_DIR / "assessments.jsonl"

# The 23 canonical assessments as a stable semantic snapshot: authority may move,
# meaning may not. (investigation, paper) -> (role, verdict, gates).
EXPECTED_ASSESSMENTS = {
    ("LT-0001", "chang-et-al-2024-informative-relations"): (
        "recent_explicit_relation_method", "exclude_decomposition_boundary", (),
    ),
    ("LT-0001", "chen-et-al-2018-gradnorm"): (
        "direct_gradient_scale_balancing", "exclude_taxonomy", (),
    ),
    ("LT-0001", "feldman-et-al-2014-mta"): (
        "sample_variance_aware_relation",
        "retain_framework_evidence_reject_implementation", (),
    ),
    ("LT-0001", "kendall-et-al-2018-uncertainty-weighting"): (
        "task_reliability_loss_scale", "exclude_taxonomy", (),
    ),
    ("LT-0001", "rakitsch-et-al-2013-structured-residuals"): (
        "signal_noise_relation_separation",
        "retain_diagnostic_principle_reject_implementation", (),
    ),
    ("LT-0001", "zhang-yang-2017-spats"): (
        "sparse_task_covariance", "reject_for_F9", (),
    ),
    ("LT-0001", "zhang-yang-2021-mtl-survey"): ("taxonomy_anchor", "retain", ()),
    ("LT-0001", "zhang-yeung-2010-mtgtp"): (
        "bayesian_relation_uncertainty", "reject_implementation", (),
    ),
    ("LT-0002", "bonilla-2007-mtgp"): (
        "family_b_estimator", "pass_with_documented_deviation", (
            ("explicit_relation_object", "pass"), ("taxonomy", "pass"),
            ("heterogeneous_heads", "fail"), ("fixed_representation", "pass"),
            ("faithful_implementability", "fail"), ("source_verification", "pass"),
        ),
    ),
    ("LT-0002", "fifty-2021-tag"): (
        "family_b_estimator", "fail", (
            ("explicit_relation_object", "pass"), ("taxonomy", "fail"),
            ("heterogeneous_heads", "pass"), ("fixed_representation", "pass"),
            ("faithful_implementability", "fail"), ("source_verification", "pass"),
        ),
    ),
    ("LT-0002", "goncalves-2016-mssl"): (
        "family_b_estimator", "pass", (
            ("explicit_relation_object", "pass"), ("taxonomy", "pass"),
            ("heterogeneous_heads", "pass"), ("fixed_representation", "pass"),
            ("faithful_implementability", "pass"), ("source_verification", "pass"),
        ),
    ),
    ("LT-0002", "graffeuille-2024-self-auxiliaries"): (
        "family_a_directed", "fail", (
            ("explicit_relation_object", "partial"), ("taxonomy", "fail"),
            ("heterogeneous_heads", "fail"), ("fixed_representation", "pass"),
            ("faithful_implementability", "fail"), ("source_verification", "pass"),
        ),
    ),
    ("LT-0002", "lee-2016-asymmetric-mtl"): (
        "family_a_directed", "pass_with_documented_deviation", (
            ("explicit_relation_object", "pass"), ("taxonomy", "pass"),
            ("heterogeneous_heads", "pass_with_deviation"),
            ("fixed_representation", "pass"), ("faithful_implementability", "pass"),
            ("source_verification", "pass"),
        ),
    ),
    ("LT-0002", "lee-2018-deep-asymmetric-mtfl"): (
        "family_a_directed", "fail", (
            ("explicit_relation_object", "partial"), ("taxonomy", "fail"),
            ("heterogeneous_heads", "fail"), ("fixed_representation", "partial"),
            ("faithful_implementability", "fail"), ("source_verification", "pass"),
        ),
    ),
    ("LT-0002", "liu-2017-trace-lasso-gamtl"): (
        "family_a_directed", "fail", (
            ("explicit_relation_object", "pass"), ("taxonomy", "fail"),
            ("heterogeneous_heads", "fail"), ("fixed_representation", "pass"),
            ("faithful_implementability", "fail"), ("source_verification", "pass"),
        ),
    ),
    ("LT-0002", "nguyen-2021-tp-amtl"): (
        "family_a_directed", "fail", (
            ("explicit_relation_object", "partial"), ("taxonomy", "fail"),
            ("heterogeneous_heads", "fail"), ("fixed_representation", "pass"),
            ("faithful_implementability", "fail"), ("source_verification", "pass"),
        ),
    ),
    ("LT-0002", "oliveira-2019-group-lasso-asymmetric"): (
        "family_a_directed", "pass_with_documented_deviation", (
            ("explicit_relation_object", "pass"), ("taxonomy", "pass"),
            ("heterogeneous_heads", "pass_with_deviation"),
            ("fixed_representation", "pass"), ("faithful_implementability", "pass"),
            ("source_verification", "pass"),
        ),
    ),
    ("LT-0002", "yu-2007-t-processes"): (
        "family_b_estimator", "fail", (
            ("explicit_relation_object", "fail"), ("taxonomy", "fail"),
            ("heterogeneous_heads", "fail"), ("fixed_representation", "pass"),
            ("faithful_implementability", "fail"), ("source_verification", "pass"),
        ),
    ),
    ("LT-0002", "yu-2020-graph-adjacency-gamtl"): (
        "family_a_directed", "fail", (
            ("explicit_relation_object", "fail"), ("taxonomy", "pass"),
            ("heterogeneous_heads", "fail"), ("fixed_representation", "pass"),
            ("faithful_implementability", "pass"), ("source_verification", "pass"),
        ),
    ),
    ("LT-0002", "zhang-schneider-2010-sparse-matrix-normal"): (
        "family_b_estimator", "fail", (
            ("explicit_relation_object", "pass"), ("taxonomy", "pass"),
            ("heterogeneous_heads", "fail"), ("fixed_representation", "pass"),
            ("faithful_implementability", "fail"), ("source_verification", "pass"),
        ),
    ),
    ("LT-0002", "zhang-yeung-2014-mtrl-asymmetric"): (
        "family_a_directed", "fail", (
            ("explicit_relation_object", "fail"), ("taxonomy", "pass"),
            ("heterogeneous_heads", "pass"), ("fixed_representation", "pass"),
            ("faithful_implementability", "not_applicable"),
            ("source_verification", "pass"),
        ),
    ),
    ("LT-0002", "zhao-2020-fetr"): (
        "family_b_estimator", "fail", (
            ("explicit_relation_object", "pass"), ("taxonomy", "pass"),
            ("heterogeneous_heads", "fail"), ("fixed_representation", "pass"),
            ("faithful_implementability", "fail"), ("source_verification", "pass"),
        ),
    ),
    ("LT-0002", "zhou-2023-autotr"): (
        "family_a_directed", "pass_with_documented_deviation", (
            ("explicit_relation_object", "pass"), ("taxonomy", "pass"),
            ("heterogeneous_heads", "pass_with_deviation"),
            ("fixed_representation", "pass"), ("faithful_implementability", "pass"),
            ("source_verification", "pass"),
        ),
    ),
}


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
        "assessment_anchor": "research/studies/LT-0002/analysis.md#assessment",
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
        (self.studies_dir / "LT-0001" / "analysis.md").write_text(
            "# LT-0001 analysis\n\n## Decision\n\nlt-0001 body\n", encoding="utf-8"
        )
        (self.studies_dir / "LT-0002" / "analysis.md").write_text(
            "# LT-0002 analysis\n\n## Assessment\n\nlt-0002 body\n", encoding="utf-8"
        )

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

    def check(self):
        return literature_assessment.check_authority(
            self.assessments_path,
            repo_root=self.repo_root,
            literature_dir=self.literature_dir,
            studies_path=self.research_dir / "STUDIES.jsonl",
            index_path=self.literature_dir / "INDEX.md",
        )


class AssessmentValidationTests(AssessmentFixture):
    def test_valid_record_loads_and_exposes_exact_lookup(self):
        registry = self.load([valid_row()])

        record = registry.get_assessment("LT-0002", "beta-2020-method")

        self.assertEqual(record.assessment_ref, "LT-0002#beta-2020-method")
        self.assertEqual(record.role, "family_b_estimator")
        self.assertEqual(record.verdict, "pass")
        self.assertEqual(record.detail_anchor, "assessment")
        self.assertEqual(
            record.detail_path, "research/studies/LT-0002/analysis.md"
        )

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

    def test_anchor_must_point_at_the_investigations_own_analysis(self):
        # A card path is no longer a valid anchor: cards hold no verdict state.
        with self.assertRaisesRegex(
            literature_assessment.AssessmentError, "analysis artifact"
        ):
            self.load(
                [
                    valid_row(
                        assessment_anchor=(
                            "research/literature/beta-2020-method.md"
                            "#lt-0002-assessment"
                        )
                    )
                ]
            )
        with self.assertRaisesRegex(
            literature_assessment.AssessmentError, "analysis artifact"
        ):
            self.load(
                [
                    valid_row(
                        assessment_anchor=(
                            "research/studies/LT-0001/analysis.md#decision"
                        )
                    )
                ]
            )

    def test_anchor_slug_must_be_a_heading_in_the_analysis_artifact(self):
        with self.assertRaisesRegex(
            literature_assessment.AssessmentError, "is not a heading"
        ):
            self.load(
                [
                    valid_row(
                        assessment_anchor=(
                            "research/studies/LT-0002/analysis.md#missing-section"
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
                            "research/studies/LT-0002/analysis.md#assessment"
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
                        "research/studies/LT-0001/analysis.md#decision"
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

    def test_the_registry_holds_the_23_canonical_assessments(self):
        self.assertEqual(len(self.registry.records), 23)
        self.assertEqual(len(self.registry.assessments_for_investigation("LT-0001")), 8)
        self.assertEqual(len(self.registry.assessments_for_investigation("LT-0002")), 15)

    def test_every_assessment_meaning_is_unchanged(self):
        actual = {
            (record.investigation_id, record.paper_id): (
                record.role,
                record.verdict,
                tuple((gate.gate_id, gate.verdict) for gate in record.gates),
            )
            for record in self.registry.records
        }
        self.assertEqual(actual, EXPECTED_ASSESSMENTS)

    def test_catalog_still_carries_no_global_screening_status(self):
        for entry in self.catalog.entries:
            with self.subTest(paper_id=entry.paper_id):
                for forbidden in ("status", "decision", "verdict", "role", "gates"):
                    self.assertFalse(hasattr(entry, forbidden))

    def test_paper_cards_carry_no_investigation_verdict_sections(self):
        for entry in self.catalog.entries:
            card = REPO_ROOT / entry.card_path
            text = card.read_text(encoding="utf-8")
            with self.subTest(paper_id=entry.paper_id):
                self.assertNotIn("## Candidate Study ID", text)
                self.assertNotIn("## LT-0001 decision", text)
                self.assertNotIn("## LT-0002 assessment", text)

    def test_paper_cards_keep_their_paper_scoped_knowledge(self):
        for entry in self.catalog.entries:
            card = REPO_ROOT / entry.card_path
            text = card.read_text(encoding="utf-8")
            with self.subTest(paper_id=entry.paper_id):
                self.assertIn("## Relation representation", text)
                self.assertIn("## Implementation difficulty", text)
        goncalves = (
            LITERATURE_DIR / "goncalves-2016-mssl.md"
        ).read_text(encoding="utf-8")
        self.assertIn("PMR", goncalves)
        zhao = (LITERATURE_DIR / "zhao-2020-fetr.md").read_text(encoding="utf-8")
        self.assertIn("ill-posedness analysis is the cleanest", zhao)

    def test_no_assessment_points_at_a_card_or_a_deleted_anchor(self):
        for record in self.registry.records:
            with self.subTest(paper_id=record.paper_id):
                self.assertNotIn("/literature/", record.detail_path)
                self.assertIn("/studies/", record.detail_path)
                self.assertTrue(record.detail_path.endswith("analysis.md"))
                self.assertNotIn(record.detail_anchor, {"lt-0001-decision",
                                                        "lt-0002-assessment"})

    def test_studies_registry_no_longer_duplicates_assessment_membership(self):
        for line in STUDIES_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                self.assertNotIn("cards", json.loads(line))

    def test_index_assessment_block_matches_the_registry(self):
        block = literature_assessment._index_assessment_block(
            LITERATURE_DIR / "INDEX.md"
        )
        self.assertIsNotNone(block)
        self.assertEqual(
            block, literature_assessment.render_assessment_tables().strip("\n")
        )


class AuthorityTests(AssessmentFixture):
    """The steady-state gate: one active structured authority for assessments."""

    def test_a_clean_repository_passes_the_authority_check(self):
        registry = self.load(
            [
                valid_row(
                    investigation_id="LT-0001",
                    paper_id="beta-2020-method",
                    role="direct_gradient_scale_balancing",
                    verdict="exclude_taxonomy",
                    assessment_anchor="research/studies/LT-0001/analysis.md#decision",
                ),
                valid_row(),
            ]
        )
        self.assertEqual(len(self.check().records), len(registry.records))

    def test_a_card_that_still_carries_a_verdict_section_is_detected(self):
        self.load([valid_row()])
        card = self.literature_dir / "beta-2020-method.md"
        card.write_text(
            card.read_text(encoding="utf-8") + "\n## LT-0002 assessment\n\nverdict\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            literature_assessment.AssessmentError, "still carries an investigation"
        ):
            self.check()

    def test_a_studies_cards_field_is_detected(self):
        self.load([valid_row()])
        (self.research_dir / "STUDIES.jsonl").write_text(
            json.dumps(
                {
                    "study_id": "LT-0002",
                    "type": "literature",
                    "title": "Second literature question",
                    "status": "complete",
                    "stage": "verification",
                    "path": "research/studies/LT-0002",
                    "cards": ["beta-2020-method"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            literature_assessment.AssessmentError, "legacy 'cards'"
        ):
            self.check()

    def test_index_assessment_drift_is_detected(self):
        self.load([valid_row()])
        (self.literature_dir / "INDEX.md").write_text(
            "# Literature\n\n"
            + literature_assessment.INDEX_ASSESSMENTS_BEGIN
            + "\n| Investigation | Paper | Role | Verdict |\n"
            + "| --- | --- | --- | --- |\n"
            + "| `LT-0002` | [beta-2020-method](beta-2020-method.md) | `x` | `y` |\n"
            + literature_assessment.INDEX_ASSESSMENTS_END
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            literature_assessment.AssessmentError, "disagrees with the registry"
        ):
            self.check()


class ClaimProvenanceTests(unittest.TestCase):
    """Every card-sourced claim must still resolve after de-verdicting."""

    def test_every_card_evidence_reference_resolves(self):
        registry = literature_claims.load_claims()
        card_refs = [
            reference
            for record in registry.records
            for reference in record.evidence
            if reference.kind == literature_claims.EVIDENCE_CARD
        ]
        self.assertTrue(card_refs)
        for reference in card_refs:
            with self.subTest(anchor=reference.fields.get("anchor")):
                self.assertNotIn(
                    reference.fields["anchor"],
                    {"LT-0001 decision", "LT-0002 assessment", "Candidate Study ID"},
                )

    def test_the_adaptive_grouping_claim_now_points_at_a_surviving_section(self):
        registry = literature_claims.load_claims()
        record = registry.get_claim(
            "liu-2017-trace-lasso-gamtl", "contribution-is-adaptive-task-grouping"
        )
        self.assertEqual(record.evidence[0].kind, "card")
        self.assertEqual(record.evidence[0].fields["anchor"], "Differences from our setting")


if __name__ == "__main__":
    unittest.main()
