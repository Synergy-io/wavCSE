"""Read-only literature queries join identity and Study state without global verdicts."""

import json
import tempfile
import unittest
from pathlib import Path

from improvements.taskrelation.research import literature_assessment
from improvements.taskrelation.research import literature_query


REPO_ROOT = Path(__file__).resolve().parents[4]


def require_paper(match):
    """Return the matched paper, failing loudly when no paper was matched."""

    if match.paper is None:
        raise AssertionError(
            "expected a known paper match, got status {!r}".format(match.status)
        )
    return match.paper


def paper_record(paper_id, title, year, author, venue, url, doi=None):
    external_ids = {"doi": [doi]} if doi else {}
    return {
        "schema_version": 1,
        "paper_id": paper_id,
        "card_path": "research/literature/{}.md".format(paper_id),
        "title": title,
        "year": year,
        "authors": [author],
        "venue": venue,
        "external_ids": external_ids,
        "source_urls": [url],
        "aliases": [],
    }


class QueryFixture(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="literature-query-")
        self.repo_root = Path(self.tempdir.name)
        self.research_dir = self.repo_root / "research"
        self.literature_dir = self.research_dir / "literature"
        self.studies_dir = self.research_dir / "studies"
        self.literature_dir.mkdir(parents=True)
        self.studies_dir.mkdir()
        self.catalog_path = self.literature_dir / "catalog.jsonl"
        self.records = [
            paper_record(
                "alpha-2024-method",
                "Alpha Method",
                2024,
                "A. Author",
                "Venue A",
                "https://example.org/alpha",
                doi="10.1234/alpha",
            ),
            paper_record(
                "beta-2020-method",
                "Beta Method",
                2020,
                "B. Author",
                "Venue B",
                "https://example.org/beta",
            ),
            paper_record(
                "gamma-2024-method",
                "Gamma Method",
                2024,
                "A. Author",
                "Venue B",
                "https://example.org/gamma",
            ),
        ]
        self.catalog_path.write_text(
            "".join(json.dumps(record) + "\n" for record in self.records),
            encoding="utf-8",
        )
        self.studies_path = self.research_dir / "STUDIES.jsonl"
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
                "study_id": "LT-0002",
                "type": "literature",
                "title": "Second literature question",
                "status": "complete",
                "stage": "source_verification",
                "decision": "CANDIDATES_FOUND",
                "path": "research/studies/LT-0002",
                "cards": ["gamma-2024-method", "beta-2020-method"],
            },
            {
                "study_id": "LT-0001",
                "type": "literature",
                "title": "First literature question",
                "status": "rejected",
                "stage": "primary_source_review",
                "decision": "REJECTED",
                "path": "research/studies/LT-0001",
            },
        ]
        self.studies_path.write_text(
            "".join(json.dumps(study) + "\n" for study in studies),
            encoding="utf-8",
        )
        for study_id in ("DG-0001", "LT-0001", "LT-0002"):
            folder = self.studies_dir / study_id
            folder.mkdir()
            for name in ("PLAN.md", "NOTE.md", "analysis.md"):
                (folder / name).write_text("# {} {}\n".format(study_id, name), encoding="utf-8")
        (self.studies_dir / "LT-0001" / "result.json").write_text(
            json.dumps(
                {
                    "study_id": "LT-0001",
                    "primary_sources_reviewed": [
                        {
                            "key": "beta",
                            "role": "direct_gradient_scale_balancing",
                            "url": "https://example.org/beta",
                            "decision": "exclude_taxonomy",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        for record in self.records:
            (self.literature_dir / (record["paper_id"] + ".md")).write_text(
                "# {}\n".format(record["title"]), encoding="utf-8"
            )
        self.assessments_path = self.literature_dir / "assessments.jsonl"
        assessments = [
            {
                "schema_version": 1,
                "investigation_id": "LT-0001",
                "paper_id": "beta-2020-method",
                "role": "direct_gradient_scale_balancing",
                "gates": [],
                "verdict": "exclude_taxonomy",
                "reason_summary": "Boundary case excluded by taxonomy.",
                "assessment_anchor": "research/literature/beta-2020-method.md#lt-0001-decision",
                "assessed_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "schema_version": 1,
                "investigation_id": "LT-0002",
                "paper_id": "beta-2020-method",
                "role": "family_b_estimator",
                "gates": [],
                "verdict": "fail",
                "reason_summary": "Screened and closed.",
                "assessment_anchor": "research/literature/beta-2020-method.md#lt-0002-assessment",
                "assessed_at": "2026-01-02T00:00:00+00:00",
            },
            {
                "schema_version": 1,
                "investigation_id": "LT-0002",
                "paper_id": "gamma-2024-method",
                "role": "family_b_estimator",
                "gates": [],
                "verdict": "pass",
                "reason_summary": "Clean pass.",
                "assessment_anchor": "research/literature/gamma-2024-method.md#lt-0002-assessment",
                "assessed_at": "2026-01-02T00:00:00+00:00",
            },
        ]
        self.assessments_path.write_text(
            "".join(json.dumps(row) + "\n" for row in assessments), encoding="utf-8"
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def open_query(self):
        return literature_query.LiteratureQuery(
            repo_root=self.repo_root,
            catalog_path=self.catalog_path,
            studies_path=self.studies_path,
            assessments_path=self.assessments_path,
        )


class LiteratureMetadataQueryTests(QueryFixture):
    def test_list_papers_returns_compact_records_in_paper_id_order(self):
        query = self.open_query()

        papers = query.list_papers()

        self.assertEqual(
            [paper.paper_id for paper in papers],
            ["alpha-2024-method", "beta-2020-method", "gamma-2024-method"],
        )
        self.assertEqual(papers[0].card_path, "research/literature/alpha-2024-method.md")
        self.assertFalse(hasattr(papers[0], "status"))
        self.assertFalse(hasattr(papers[0], "decision"))

    def test_list_papers_filters_exact_existing_metadata(self):
        query = self.open_query()

        self.assertEqual(
            [paper.paper_id for paper in query.list_papers(year=2024)],
            ["alpha-2024-method", "gamma-2024-method"],
        )
        self.assertEqual(
            [paper.paper_id for paper in query.list_papers(author="A. Author")],
            ["alpha-2024-method", "gamma-2024-method"],
        )
        self.assertEqual(
            [paper.paper_id for paper in query.list_papers(venue="venue b")],
            ["beta-2020-method", "gamma-2024-method"],
        )

    def test_list_papers_returns_empty_tuple_when_metadata_has_no_match(self):
        query = self.open_query()

        self.assertEqual(query.list_papers(year=1999), ())

    def test_resolve_paper_accepts_catalog_external_alias(self):
        query = self.open_query()

        paper = query.resolve_paper("https://doi.org/10.1234/ALPHA")

        self.assertEqual(paper.paper_id, "alpha-2024-method")
        self.assertEqual(paper.title, "Alpha Method")

    def test_unknown_paper_identity_fails_clearly(self):
        query = self.open_query()

        with self.assertRaisesRegex(
            literature_query.LiteratureQueryError,
            "unknown paper identity 'missing-paper'",
        ):
            query.resolve_paper("missing-paper")

    def test_malformed_metadata_filter_fails_clearly(self):
        query = self.open_query()

        with self.assertRaisesRegex(
            literature_query.LiteratureQueryError,
            "year must be an integer",
        ):
            query.list_papers(year="2024")
        with self.assertRaisesRegex(
            literature_query.LiteratureQueryError,
            "author must be a non-empty string",
        ):
            query.list_papers(author="  ")


class LiteratureStudyRelationshipTests(QueryFixture):
    def test_paper_to_studies_keeps_each_assessment_study_scoped(self):
        query = self.open_query()

        relationships = query.studies_for_paper("beta-2020-method")

        self.assertEqual(
            [relationship.study.study_id for relationship in relationships],
            ["LT-0001", "LT-0002"],
        )
        first, second = relationships
        self.assertEqual(first.assessment.investigation_id, "LT-0001")
        self.assertEqual(first.assessment.paper_id, "beta-2020-method")
        self.assertEqual(first.assessment.verdict, "exclude_taxonomy")
        self.assertEqual(first.assessment.role, "direct_gradient_scale_balancing")
        self.assertEqual(second.assessment.investigation_id, "LT-0002")
        self.assertEqual(second.assessment.verdict, "fail")
        self.assertEqual(second.assessment.role, "family_b_estimator")
        self.assertEqual(second.assessment.detail_anchor, "lt-0002-assessment")
        self.assertEqual(second.study.decision, "CANDIDATES_FOUND")

    def test_relationship_queries_never_open_a_card(self):
        # Queries are backed by the registry, so card *content* is irrelevant;
        # replacing every card with unparseable text must not change an answer.
        query = self.open_query()
        for record in self.records:
            (self.literature_dir / (record["paper_id"] + ".md")).write_text(
                "not a card at all\n", encoding="utf-8"
            )

        relationships = query.papers_for_study("LT-0002")

        self.assertEqual(
            [relationship.paper.paper_id for relationship in relationships],
            ["beta-2020-method", "gamma-2024-method"],
        )

    def test_study_to_papers_returns_deterministic_relationship_order(self):
        query = self.open_query()

        relationships = query.papers_for_study("LT-0002")

        self.assertEqual(
            [relationship.paper.paper_id for relationship in relationships],
            ["beta-2020-method", "gamma-2024-method"],
        )
        self.assertTrue(
            all(
                relationship.assessment.investigation_id == "LT-0002"
                for relationship in relationships
            )
        )

    def test_known_paper_without_study_relationship_returns_empty_tuple(self):
        query = self.open_query()

        self.assertEqual(query.studies_for_paper("alpha-2024-method"), ())

    def test_unknown_study_fails_clearly(self):
        query = self.open_query()

        with self.assertRaisesRegex(
            literature_query.LiteratureQueryError,
            "unknown Study 'LT-9999'",
        ):
            query.papers_for_study("LT-9999")

    def test_non_literature_study_is_not_exposed_as_literature_state(self):
        query = self.open_query()

        with self.assertRaisesRegex(
            literature_query.LiteratureQueryError,
            "Study 'DG-0001' is not a literature Study",
        ):
            query.papers_for_study("DG-0001")

    def test_get_study_returns_artifact_pointers_without_a_result_file(self):
        query = self.open_query()

        study = query.get_study("LT-0002")

        self.assertEqual(study.study_id, "LT-0002")
        self.assertEqual(study.decision, "CANDIDATES_FOUND")
        self.assertEqual(study.analysis_path, "research/studies/LT-0002/analysis.md")
        self.assertIsNone(study.result_path)

    def test_two_records_for_one_paper_in_one_study_are_a_conflict(self):
        # Canonical uniqueness is enforced by the registry: a duplicate
        # (investigation, paper) pair is rejected when the query loads.
        self.assessments_path.write_text(
            self.assessments_path.read_text(encoding="utf-8")
            + json.dumps(
                {
                    "schema_version": 1,
                    "investigation_id": "LT-0002",
                    "paper_id": "gamma-2024-method",
                    "role": "family_b_estimator",
                    "gates": [],
                    "verdict": "fail",
                    "reason_summary": "duplicate",
                    "assessment_anchor": "research/literature/gamma-2024-method.md#lt-0002-assessment",
                    "assessed_at": "2026-01-02T00:00:00+00:00",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            literature_query.LiteratureQueryError,
            "duplicate paper assessment 'LT-0002#gamma-2024-method'",
        ):
            self.open_query()


class CandidateIdentificationTests(QueryFixture):
    def test_candidate_identified_from_each_supported_identity_field(self):
        query = self.open_query()
        cases = {
            "paper_id": ("alpha-2024-method", "paper_id"),
            "title": ("Alpha Method", "title"),
            "doi": ("https://doi.org/10.1234/ALPHA", "doi"),
            "source_url": ("https://example.org/alpha", "source_url"),
        }

        for field, (value, expected_field) in cases.items():
            with self.subTest(field=field):
                match = query.identify_candidate(**{field: value})
                self.assertEqual(match.status, "known")
                self.assertEqual(match.matched_field, expected_field)
                self.assertEqual(
                    require_paper(match).paper_id, "alpha-2024-method"
                )
                self.assertEqual(match.conflicts, ())

    def test_unseen_candidate_reports_new_without_raising(self):
        query = self.open_query()

        match = query.identify_candidate(
            title="Unseen Method",
            doi="10.9999/unseen",
            arxiv="2601.00001",
        )

        self.assertEqual(match.status, "new")
        self.assertIsNone(match.matched_field)
        self.assertIsNone(match.paper)
        self.assertEqual(match.conflicts, ())

    def test_agreeing_fields_resolve_to_one_known_paper(self):
        query = self.open_query()

        match = query.identify_candidate(
            paper_id="alpha-2024-method",
            title="Alpha Method",
            doi="10.1234/alpha",
            source_url="https://example.org/alpha",
        )

        self.assertEqual(match.status, "known")
        self.assertEqual(require_paper(match).paper_id, "alpha-2024-method")

    def test_disagreeing_fields_are_ambiguous_not_silently_resolved(self):
        query = self.open_query()

        match = query.identify_candidate(
            doi="10.1234/alpha",
            source_url="https://example.org/beta",
        )

        self.assertEqual(match.status, "ambiguous")
        self.assertIsNone(match.paper)
        self.assertEqual(
            [paper.paper_id for paper in match.conflicts],
            ["alpha-2024-method", "beta-2020-method"],
        )

    def test_ambiguity_is_deterministic_regardless_of_field_order(self):
        query = self.open_query()

        first = query.identify_candidate(
            doi="10.1234/alpha",
            source_url="https://example.org/beta",
        )
        second = query.identify_candidate(
            source_url="https://example.org/beta",
            doi="10.1234/alpha",
        )

        self.assertEqual(
            [paper.paper_id for paper in first.conflicts],
            [paper.paper_id for paper in second.conflicts],
        )

    def test_candidate_title_requires_an_exact_normalized_match(self):
        query = self.open_query()

        self.assertEqual(
            query.identify_candidate(title="alpha method").status, "known"
        )
        self.assertEqual(
            query.identify_candidate(title="Alpha").status, "new"
        )
        self.assertEqual(
            query.identify_candidate(title="Alpha Method Extended").status, "new"
        )

    def test_candidate_identification_requires_at_least_one_field(self):
        query = self.open_query()

        with self.assertRaisesRegex(
            literature_query.LiteratureQueryError,
            "at least one identity field is required",
        ):
            query.identify_candidate()

    def test_malformed_candidate_field_fails_clearly(self):
        query = self.open_query()

        with self.assertRaisesRegex(
            literature_query.LiteratureQueryError,
            "doi must be a non-empty string",
        ):
            query.identify_candidate(doi="   ")
        with self.assertRaisesRegex(
            literature_query.LiteratureQueryError,
            "title must be a non-empty string",
        ):
            query.identify_candidate(title=42)


class RealRepositoryCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.query = literature_query.LiteratureQuery(repo_root=REPO_ROOT)

    def test_real_doi_candidate_is_recognized_as_already_known(self):
        match = self.query.identify_candidate(
            doi="10.1145/3580305.3599261",
            title="Automatic Temporal Relation in Multi-Task Learning",
        )

        self.assertEqual(match.status, "known")
        self.assertEqual(require_paper(match).paper_id, "zhou-2023-autotr")
        self.assertEqual(
            require_paper(match).card_path,
            "improvements/taskrelation/research/literature/zhou-2023-autotr.md",
        )

    def test_real_arxiv_candidate_is_recognized_as_already_known(self):
        match = self.query.identify_candidate(arxiv="arXiv:2410.15875v1")

        self.assertEqual(match.status, "known")
        self.assertEqual(
            require_paper(match).paper_id, "graffeuille-2024-self-auxiliaries"
        )

    def test_real_unseen_candidate_reports_new(self):
        match = self.query.identify_candidate(
            title="A Paper We Have Never Retained",
            doi="10.5555/not-retained",
        )

        self.assertEqual(match.status, "new")


class RealRepositoryQueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.query = literature_query.LiteratureQuery(repo_root=REPO_ROOT)

    def test_real_repository_lists_all_validated_catalog_papers(self):
        self.assertEqual(len(self.query.list_papers()), 23)

    def test_real_literature_studies_resolve_their_existing_papers(self):
        self.assertEqual(len(self.query.papers_for_study("LT-0001")), 8)
        self.assertEqual(len(self.query.papers_for_study("LT-0002")), 15)

    def test_real_assessments_stay_investigation_scoped(self):
        gradnorm = self.query.studies_for_paper("chen-et-al-2018-gradnorm")
        mssl = self.query.studies_for_paper("goncalves-2016-mssl")

        self.assertEqual(gradnorm[0].assessment.investigation_id, "LT-0001")
        self.assertEqual(gradnorm[0].assessment.verdict, "exclude_taxonomy")
        self.assertEqual(mssl[0].assessment.investigation_id, "LT-0002")
        self.assertEqual(mssl[0].assessment.verdict, "pass")
        self.assertEqual(mssl[0].assessment.role, "family_b_estimator")
        self.assertEqual(mssl[0].assessment.detail_anchor, "lt-0002-assessment")

    def test_real_assessment_operation_returns_the_canonical_record(self):
        record = self.query.get_assessment("LT-0002", "goncalves-2016-mssl")

        self.assertEqual(record.assessment_ref, "LT-0002#goncalves-2016-mssl")
        self.assertEqual(record.verdict, "pass")
        self.assertEqual(len(record.gates), 6)
        self.assertEqual(
            [gate.gate_id for gate in record.gates], list(literature_assessment.GATE_IDS)
        )
        self.assertEqual(
            record.assessment_anchor,
            "improvements/taskrelation/research/literature/goncalves-2016-mssl.md"
            "#lt-0002-assessment",
        )


if __name__ == "__main__":
    unittest.main()
