"""The synthesis layer stays an explicit, provenance-bounded narrative authority.

These tests prove the INC-V2-4 envelope: stable identity, machine-visible
kind/status, bounded provenance, a deterministic 1:1 coverage rule, and the
authority boundaries it must not cross (Claims, PaperAssessment, primary
evidence, research decisions).
"""

import json
import tempfile
import unittest
from pathlib import Path

from improvements.taskrelation.research import literature_assessment
from improvements.taskrelation.research import literature_claims
from improvements.taskrelation.research import literature_query
from improvements.taskrelation.research import literature_read
from improvements.taskrelation.research import literature_synthesis


REPO_ROOT = Path(__file__).resolve().parents[4]
RESEARCH_DIR = REPO_ROOT / "improvements" / "taskrelation" / "research"
SURVEY_DIR = RESEARCH_DIR / "literature_survey"
SURVEY_REL = "improvements/taskrelation/research/literature_survey/"
AGENT_PATH = REPO_ROOT / ".omp" / "agents" / "literature-reviewer.md"


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class SynthesisFixture(unittest.TestCase):
    """A minimal repository whose only registered documents are ``DOC_A``/``DOC_B``."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = Path(self._tmp.name)
        research = self.repo / "improvements" / "taskrelation" / "research"
        self.survey = research / "literature_survey"
        self.literature = research / "literature"
        self.survey.mkdir(parents=True)
        self.literature.mkdir(parents=True)
        _write(
            self.literature / "catalog.jsonl",
            json.dumps(
                {
                    "schema_version": 1,
                    "paper_id": "paper-one",
                    "card_path": SURVEY_REL + "../../literature/paper-one.md",
                    "title": "Paper One",
                    "year": 2020,
                    "authors": ["A. Author"],
                    "venue": "VENUE",
                    "external_ids": {},
                    "source_urls": ["https://example.invalid/one"],
                    "aliases": [],
                }
            )
            + "\n",
        )
        _write(self.literature / "primary_manifest.jsonl", "")
        _write(self.literature / "claims.jsonl", "")
        self.studies = research / "STUDIES.jsonl"
        _write(self.studies, json.dumps({"study_id": "TR-0001"}) + "\n")
        self.findings = research / "FINDINGS.md"
        _write(self.findings, "# Findings\n\n## F1 — x  (ESTABLISHED)\n")
        _write(self.survey / "DOC_A.md", "# A\n")
        _write(self.survey / "DOC_B.md", "# B\n")

    def record(self, synthesis_id, *, path="DOC_A.md", **overrides):
        record = {
            "schema_version": 1,
            "synthesis_id": synthesis_id,
            "kind": "theory",
            "status": "active",
            "path": SURVEY_REL + path,
            "derives_from": [],
        }
        record.update(overrides)
        return record

    def write_registry(self, records):
        ordered = sorted(records, key=lambda item: item.get("synthesis_id", ""))
        self.survey.joinpath("registry.jsonl").write_text(
            "\n".join(json.dumps(item, sort_keys=True) for item in ordered) + "\n",
            encoding="utf-8",
        )

    def load(self, **overrides):
        return literature_synthesis.load_syntheses(
            self.survey / "registry.jsonl",
            repo_root=self.repo,
            literature_dir=self.literature,
            studies_path=self.studies,
            findings_path=self.findings,
            verify_claim_survey_evidence=False,
            **overrides
        )

    def assert_error(self, kind, records):
        self.write_registry(records)
        with self.assertRaises(literature_synthesis.SynthesisError) as caught:
            self.load()
        self.assertEqual(caught.exception.kind, kind)

    def both(self, **overrides):
        return [self.record("doc-a", **overrides), self.record("doc-b", path="DOC_B.md")]


class EnvelopeValidationTests(SynthesisFixture):
    def test_valid_registry_loads(self):
        self.write_registry(self.both())
        registry = self.load()
        self.assertEqual(registry.synthesis_ids, ("doc-a", "doc-b"))

    def test_kind_must_be_in_vocabulary(self):
        self.assert_error(
            "INVALID_KIND",
            [self.record("doc-a", kind="nonsense"), self.record("doc-b", path="DOC_B.md")],
        )

    def test_status_must_be_in_vocabulary(self):
        self.assert_error(
            "INVALID_STATUS",
            [self.record("doc-a", status="done"), self.record("doc-b", path="DOC_B.md")],
        )

    def test_metadata_envelope_rejects_extra_fields(self):
        self.assert_error(
            "UNEXPECTED_KEYS",
            [
                self.record("doc-a", decision="authorized"),
                self.record("doc-b", path="DOC_B.md"),
            ],
        )

    def test_unsupported_schema_version_fails(self):
        self.assert_error(
            "UNSUPPORTED_SCHEMA_VERSION",
            [
                self.record("doc-a", schema_version=2),
                self.record("doc-b", path="DOC_B.md"),
            ],
        )

    def test_synthesis_id_must_be_lowercase_hyphenated(self):
        self.assert_error(
            "INVALID_SYNTHESIS_ID",
            [self.record("Doc_A"), self.record("doc-b", path="DOC_B.md")],
        )

    def test_duplicate_identity_fails(self):
        self.assert_error(
            "DUPLICATE_SYNTHESIS_ID",
            [self.record("doc-a"), self.record("doc-a", path="DOC_B.md")],
        )

    def test_one_path_cannot_be_two_identities(self):
        self.assert_error(
            "DUPLICATE_SYNTHESIS_PATH",
            [self.record("doc-a"), self.record("doc-b")],
        )

    def test_malformed_reference_fails(self):
        self.assert_error(
            "INVALID_REFERENCE",
            [
                self.record("doc-a", derives_from=["paper"]),
                self.record("doc-b", path="DOC_B.md"),
            ],
        )


class SurfaceValidationTests(SynthesisFixture):
    def test_missing_document_fails(self):
        self.assert_error(
            "SYNTHESIS_PATH_MISSING",
            [self.record("doc-a"), self.record("doc-c", path="DOC_C.md")],
        )

    def test_path_outside_surface_fails(self):
        self.assert_error(
            "INVALID_PATH",
            [self.record("doc-a", path="../../DECISIONS.md")],
        )

    def test_unregistered_markdown_fails_coverage(self):
        _write(self.survey / "DOC_C.md", "# C\n")
        self.assert_error(
            "UNREGISTERED_SYNTHESIS",
            self.both(),
        )


class DerivesFromValidationTests(SynthesisFixture):
    def test_unknown_paper_reference_fails(self):
        self.assert_error(
            "UNRESOLVED_REFERENCE",
            [
                self.record("doc-a", derives_from=["paper:not-in-catalog"]),
                self.record("doc-b", path="DOC_B.md"),
            ],
        )

    def test_unknown_claim_reference_fails(self):
        self.assert_error(
            "UNRESOLVED_REFERENCE",
            [
                self.record("doc-a", derives_from=["claim:paper-one#nope"]),
                self.record("doc-b", path="DOC_B.md"),
            ],
        )

    def test_unknown_investigation_reference_fails(self):
        self.assert_error(
            "UNRESOLVED_REFERENCE",
            [
                self.record("doc-a", derives_from=["investigation:TR-9999"]),
                self.record("doc-b", path="DOC_B.md"),
            ],
        )

    def test_unknown_finding_reference_fails(self):
        self.assert_error(
            "UNRESOLVED_REFERENCE",
            [
                self.record("doc-a", derives_from=["finding:F42"]),
                self.record("doc-b", path="DOC_B.md"),
            ],
        )

    def test_unknown_synthesis_reference_fails(self):
        self.assert_error(
            "UNRESOLVED_REFERENCE",
            [
                self.record("doc-a", derives_from=["synthesis:not-registered"]),
                self.record("doc-b", path="DOC_B.md"),
            ],
        )

    def test_paper_and_investigation_references_resolve(self):
        self.write_registry(
            [
                self.record(
                    "doc-a",
                    derives_from=["paper:paper-one", "investigation:TR-0001"],
                ),
                self.record("doc-b", path="DOC_B.md", derives_from=["finding:F1"]),
            ]
        )
        registry = self.load()
        self.assertIn("paper:paper-one", registry.get("doc-a").derives_from)

    def test_self_dependency_fails(self):
        self.assert_error(
            "SELF_DEPENDENCY",
            [
                self.record("doc-a", derives_from=["synthesis:doc-a"]),
                self.record("doc-b", path="DOC_B.md"),
            ],
        )

    def test_synthesis_dependency_cycle_fails(self):
        self.assert_error(
            "SYNTHESIS_CYCLE",
            [
                self.record("doc-a", derives_from=["synthesis:doc-b"]),
                self.record("doc-b", path="DOC_B.md", derives_from=["synthesis:doc-a"]),
            ],
        )


class RealRegistryTests(unittest.TestCase):
    """The committed registry must satisfy every rule against the real repository."""

    @classmethod
    def setUpClass(cls):
        cls.registry = literature_synthesis.load_syntheses()
        cls.records = {record.synthesis_id: record for record in cls.registry.records}

    def test_synthesis_ids_are_unique(self):
        ids = [record.synthesis_id for record in self.registry.records]
        self.assertEqual(len(ids), len(set(ids)))

    def test_kinds_and_statuses_are_machine_visible(self):
        for record in self.registry.records:
            self.assertIn(record.kind, literature_synthesis.KINDS)
            self.assertIn(record.status, literature_synthesis.STATUSES)

    def test_every_survey_markdown_is_registered_exactly_once(self):
        present = {path.name for path in SURVEY_DIR.glob("*.md")}
        registered = [Path(record.path).name for record in self.registry.records]
        self.assertEqual(set(registered), present)
        self.assertEqual(len(registered), len(set(registered)))

    def test_registered_paths_resolve_inside_the_surface(self):
        for record in self.registry.records:
            path = (REPO_ROOT / record.path).resolve()
            self.assertTrue(path.is_file(), record.path)
            self.assertEqual(path.parent, SURVEY_DIR.resolve())

    def test_every_reference_resolves(self):
        # load_syntheses() already validated every derives_from reference; this
        # asserts the resolver actually saw a non-trivial graph.
        self.assertTrue(any(record.derives_from for record in self.registry.records))

    def test_metadata_carries_no_decision_or_assessment_state(self):
        for record in self.registry.records:
            payload = record.as_dict()
            self.assertEqual(
                set(payload),
                {"synthesis_id", "kind", "status", "path", "derives_from"},
            )
            for forbidden in ("decision", "authorization", "verdict", "gates", "role"):
                self.assertNotIn(forbidden, payload)

    def test_prediction_snapshots_are_historical_not_active(self):
        predictions = [r for r in self.registry.records if r.kind == "prediction"]
        self.assertTrue(predictions)
        for record in predictions:
            self.assertEqual(record.status, "historical")

    def test_unique_theory_documents_remain_registered_prose(self):
        record = self.records["mssl-sparsity-analysis"]
        self.assertEqual(record.kind, "theory")
        content = literature_read.LiteratureReader().read_synthesis(
            "mssl-sparsity-analysis", max_chars=120
        )
        self.assertEqual(content.source_kind, "synthesis")
        self.assertEqual(content.evidence_level, "survey-derived")
        self.assertTrue(content.text.startswith("#"))


class AuthorityBoundaryTests(unittest.TestCase):
    def test_check_authority_passes_on_the_real_repository(self):
        registry = literature_synthesis.check_authority()
        self.assertEqual(len(registry.records), 19)

    def test_survey_backed_claim_stays_survey_derived_and_registered(self):
        claims = literature_claims.load_claims()
        survey_claims = [
            record
            for record in claims.records
            if any(ref.kind == "survey" for ref in record.evidence)
        ]
        self.assertTrue(survey_claims)
        registered = {
            record.path for record in literature_synthesis.load_syntheses().records
        }
        for record in survey_claims:
            self.assertEqual(record.source_level, "survey")
            for ref in record.evidence:
                if ref.kind == "survey":
                    self.assertIn(SURVEY_REL + ref.fields["document"], registered)

    def test_paper_assessment_registry_is_unchanged(self):
        assessments = literature_assessment.load_assessments(
            RESEARCH_DIR / "literature" / "assessments.jsonl",
            repo_root=REPO_ROOT,
            literature_dir=RESEARCH_DIR / "literature",
            studies_path=RESEARCH_DIR / "STUDIES.jsonl",
        )
        self.assertEqual(len(assessments.records), 23)
        self.assertEqual(len(assessments.assessments_for_investigation("LT-0002")), 15)
        record = assessments.get_assessment("LT-0002", "goncalves-2016-mssl")
        self.assertEqual(record.verdict, "pass")

    def test_claims_registry_is_unchanged(self):
        claims = literature_claims.load_claims()
        self.assertEqual(len(claims.records), 18)

    def test_synthesis_registry_is_not_an_assessment_authority(self):
        # No card/Study/registry field here; the assessment authority check is the
        # proof that synthesis documents are not consumed as PaperAssessment.
        literature_assessment.check_authority()

    def test_literature_agent_tool_grant_is_unchanged(self):
        text = AGENT_PATH.read_text(encoding="utf-8")
        grant = next(
            line for line in text.splitlines() if line.startswith("tools:")
        )
        self.assertEqual(
            grant,
            "tools: literature_resolve, literature_query, literature_read, "
            "literature_primary",
        )


class QueryAndReadTests(unittest.TestCase):
    def test_list_syntheses_returns_metadata_without_prose(self):
        query = literature_query.LiteratureQuery()
        records = query.list_syntheses(status="historical")
        self.assertEqual(len(records), 8)
        for record in records:
            payload = record.as_dict()
            self.assertNotIn("text", payload)
            self.assertEqual(payload["status"], "historical")

    def test_get_synthesis_returns_one_metadata_record(self):
        query = literature_query.LiteratureQuery()
        record = query.get_synthesis("post-tr0007-synthesis")
        self.assertEqual(record.kind, "synthesis")
        self.assertIn("investigation:TR-0007", record.derives_from)

    def test_unknown_synthesis_id_fails_clearly(self):
        query = literature_query.LiteratureQuery()
        with self.assertRaises(literature_query.LiteratureQueryError):
            query.get_synthesis("does-not-exist")

    def test_bounded_synthesis_read_truncates(self):
        content = literature_read.LiteratureReader().read_synthesis(
            "post-tr0007-synthesis", max_chars=64
        )
        self.assertEqual(content.characters, 64)
        self.assertTrue(content.truncated)
        self.assertEqual(content.reference, "post-tr0007-synthesis")

    def test_read_refuses_unknown_synthesis_identity(self):
        reader = literature_read.LiteratureReader()
        with self.assertRaises(literature_read.LiteratureReadError) as caught:
            reader.read_synthesis("../../DECISIONS.md")
        self.assertEqual(caught.exception.kind, literature_read.UNKNOWN_REFERENCE)


if __name__ == "__main__":
    unittest.main()
