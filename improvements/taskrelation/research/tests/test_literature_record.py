"""The bounded, investigation-scoped literature-write surface (INC-018).

The positive path (a delegated ACTIVE investigation may persist a note, an
assessment, a claim and a synthesis) and — more importantly — the negative
authority: nothing outside the delegated investigation, no other research record
and no rejected write may change state.
"""

import json
import tempfile
import unittest
from pathlib import Path

from improvements.taskrelation.research import literature_investigation as li
from improvements.taskrelation.research import literature_query
from improvements.taskrelation.research import literature_record as lr


class RecordFixture(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="lt-record-")
        self.repo_root = Path(self.tempdir.name)
        self.research_dir = (
            self.repo_root / "improvements" / "taskrelation" / "research"
        )
        self.literature_dir = self.research_dir / "literature"
        self.survey_dir = self.research_dir / "literature_survey"
        self.studies_dir = self.research_dir / "studies"
        self.literature_dir.mkdir(parents=True)
        self.survey_dir.mkdir()
        self.studies_dir.mkdir()
        self.studies_path = self.research_dir / "STUDIES.jsonl"
        self._write_literature()
        self._write_studies()

    def tearDown(self):
        self.tempdir.cleanup()

    def _write_literature(self):
        record = {
            "schema_version": 1,
            "paper_id": "alpha-2024-method",
            "card_path": "improvements/taskrelation/research/literature/alpha-2024-method.md",
            "title": "Alpha Method",
            "year": 2024,
            "authors": ["A. Author"],
            "venue": "Venue A",
            "external_ids": {"doi": ["10.1234/alpha"]},
            "source_urls": ["https://example.org/alpha"],
            "aliases": [],
        }
        (self.literature_dir / "catalog.jsonl").write_text(
            json.dumps(record) + "\n", encoding="utf-8"
        )
        (self.literature_dir / "alpha-2024-method.md").write_text(
            "# Alpha Method\n\n## Summary\n\nalpha body text\n", encoding="utf-8"
        )
        (self.literature_dir / "assessments.jsonl").write_text("", encoding="utf-8")
        (self.literature_dir / "claims.jsonl").write_text("", encoding="utf-8")
        (self.literature_dir / "primary_manifest.jsonl").write_text("", encoding="utf-8")
        (self.literature_dir / "INDEX.md").write_text(
            "\n<!-- BEGIN GENERATED: assessments -->\n"
            "| Investigation | Paper | Role | Verdict |\n"
            "| --- | --- | --- | --- |\n"
            "<!-- END GENERATED: assessments -->\n",
            encoding="utf-8",
        )
        (self.survey_dir / "registry.jsonl").write_text("", encoding="utf-8")
        (self.research_dir / "FINDINGS.md").write_text("", encoding="utf-8")

    def _write_studies(self):
        studies = [
            {
                "study_id": "DG-0001",
                "type": "diagnostic",
                "title": "Not literature",
                "status": "complete",
                "stage": "analysis",
                "path": "improvements/taskrelation/research/studies/DG-0001",
            },
            {
                "study_id": "LT-0001",
                "type": "literature",
                "title": "Closed",
                "status": "complete",
                "stage": "verification",
                "path": "improvements/taskrelation/research/studies/LT-0001",
            },
            {
                "study_id": "LT-0002",
                "type": "literature",
                "status": "active",
                "stage": "investigation_open",
                "title": "Not delegated",
                "question": "q2",
                "scope": "s2",
                "path": "improvements/taskrelation/research/studies/LT-0002",
                "created_at": "2026-10-04T00:00:00+00:00",
                "started_at": "2026-10-04T00:00:00+00:00",
                "delegated": False,
                "assessment_scope": {"roles": ["family_b"], "verdicts": ["pass"]},
            },
            {
                "study_id": "LT-0003",
                "type": "literature",
                "status": "active",
                "stage": "investigation_open",
                "title": "Delegated",
                "question": "q3",
                "scope": "s3",
                "path": "improvements/taskrelation/research/studies/LT-0003",
                "created_at": "2026-10-04T00:00:00+00:00",
                "started_at": "2026-10-04T00:00:00+00:00",
                "delegated": True,
                "assessment_scope": {"roles": ["family_b"], "verdicts": ["pass"]},
            },
        ]
        self.studies_path.write_text(
            "".join(json.dumps(row) + "\n" for row in studies), encoding="utf-8"
        )
        for study_id in ("DG-0001", "LT-0001", "LT-0002", "LT-0003"):
            (self.studies_dir / study_id).mkdir()
        (self.studies_dir / "LT-0003" / "analysis.md").write_text(
            "# LT-0003\n\n## Assessment\n\nbody\n", encoding="utf-8"
        )

    # -- helpers -------------------------------------------------------------

    def assessment(self, investigation="LT-0003", **kwargs):
        params = {
            "paper_id": "alpha-2024-method",
            "role": "family_b",
            "verdict": "pass",
            "reason_summary": "bounded reason",
            "anchor": "assessment",
        }
        params.update(kwargs)
        return lr.put_assessment(
            investigation,
            repo_root=self.repo_root,
            literature_dir=self.literature_dir,
            studies_path=self.studies_path,
            **params
        )

    def claim(self, investigation="LT-0003", **kwargs):
        params = {
            "paper_id": "alpha-2024-method",
            "claim_id": "alpha-claim",
            "claim_type": "method-objective",
            "assertion_kind": "paraphrase",
            "assertion": "Alpha states a bounded objective.",
            "evidence": [{"kind": "card", "anchor": "Summary"}],
        }
        params.update(kwargs)
        return lr.put_claim(
            investigation,
            repo_root=self.repo_root,
            literature_dir=self.literature_dir,
            studies_path=self.studies_path,
            **params
        )

    def synthesis(self, investigation="LT-0003", **kwargs):
        params = {
            "synthesis_id": "alpha-finding",
            "kind": "synthesis",
            "status": "active",
            "document": "ALPHA_FINDING.md",
            "derives_from": ["investigation:{}".format(investigation)],
            "body": "# Alpha finding\n\nOBSERVED: alpha.\n",
        }
        params.update(kwargs)
        return lr.put_synthesis(
            investigation,
            repo_root=self.repo_root,
            literature_dir=self.literature_dir,
            studies_path=self.studies_path,
            **params
        )


class ScopeAuthorityTests(RecordFixture):
    def test_active_delegated_investigation_permits_a_scoped_assessment(self):
        result = self.assessment()

        rows = [
            json.loads(line)
            for line in (self.literature_dir / "assessments.jsonl").read_text().splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["investigation_id"], "LT-0003")
        self.assertEqual(
            rows[0]["assessment_anchor"],
            "improvements/taskrelation/research/studies/LT-0003/analysis.md#assessment",
        )
        self.assertIn("assessment", result)

    def test_another_investigation_cannot_be_written_while_delegated(self):
        with self.assertRaises(lr.RecordError) as raised:
            self.assessment(investigation="LT-0002")
        self.assertEqual(raised.exception.kind, lr.OUTSIDE_DELEGATED_SCOPE)
        self.assertEqual(
            (self.literature_dir / "assessments.jsonl").read_text(), ""
        )

    def test_unknown_investigation_is_rejected(self):
        with self.assertRaises(lr.RecordError) as raised:
            self.assessment(investigation="LT-9999")
        self.assertEqual(raised.exception.kind, lr.INVESTIGATION_NOT_FOUND)

    def test_closed_investigation_is_rejected(self):
        with self.assertRaises(lr.RecordError) as raised:
            self.assessment(investigation="LT-0001")
        self.assertEqual(raised.exception.kind, lr.INVESTIGATION_NOT_ACTIVE)

    def test_non_literature_study_is_rejected(self):
        with self.assertRaises(lr.RecordError) as raised:
            self.assessment(investigation="DG-0001")
        self.assertEqual(raised.exception.kind, lr.INVESTIGATION_NOT_LITERATURE)


class AssessmentTests(RecordFixture):
    def test_assessment_references_a_canonical_paper(self):
        with self.assertRaises(lr.RecordError) as raised:
            self.assessment(paper_id="missing-paper")
        self.assertEqual(raised.exception.kind, lr.PAPER_NOT_FOUND)

    def test_unknown_role_is_rejected_and_nothing_is_written(self):
        with self.assertRaises(lr.RecordError) as raised:
            self.assessment(role="not_a_declared_role")
        self.assertEqual(raised.exception.kind, lr.ASSESSMENT_CONFLICT)
        self.assertEqual((self.literature_dir / "assessments.jsonl").read_text(), "")

    def test_missing_anchor_heading_is_rejected(self):
        with self.assertRaises(lr.RecordError) as raised:
            self.assessment(anchor="no-such-heading")
        self.assertEqual(raised.exception.kind, lr.ASSESSMENT_CONFLICT)

    def test_rewriting_the_same_assessment_updates_in_place(self):
        self.assessment()
        self.assessment(reason_summary="updated reason")

        rows = (self.literature_dir / "assessments.jsonl").read_text().splitlines()
        self.assertEqual(len(rows), 1)
        self.assertEqual(json.loads(rows[0])["reason_summary"], "updated reason")

    def test_index_generated_block_is_regenerated(self):
        self.assessment()
        index = (self.literature_dir / "INDEX.md").read_text(encoding="utf-8")
        self.assertIn("LT-0003", index)
        self.assertIn("family_b", index)


class ClaimTests(RecordFixture):
    def test_valid_card_backed_claim_persists(self):
        result = self.claim()

        self.assertEqual(result["claim_ref"], "alpha-2024-method#alpha-claim")
        rows = [
            json.loads(line)
            for line in (self.literature_dir / "claims.jsonl").read_text().splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 1)
        self.assertNotIn("investigation_id", rows[0])

    def test_invalid_evidence_reference_is_rejected(self):
        with self.assertRaises(lr.RecordError) as raised:
            self.claim(evidence=[{"kind": "card", "anchor": "NoSuchHeading"}])
        self.assertEqual(raised.exception.kind, lr.EVIDENCE_REFERENCE_INVALID)
        self.assertEqual((self.literature_dir / "claims.jsonl").read_text(), "")

    def test_duplicate_claim_is_idempotent(self):
        self.claim()
        self.claim()

        rows = (self.literature_dir / "claims.jsonl").read_text().splitlines()
        self.assertEqual(len(rows), 1)

    def test_claim_from_a_non_delegated_investigation_is_refused(self):
        with self.assertRaises(lr.RecordError) as raised:
            self.claim(investigation="LT-0002")
        self.assertEqual(raised.exception.kind, lr.OUTSIDE_DELEGATED_SCOPE)


class SynthesisTests(RecordFixture):
    def test_synthesis_records_investigation_provenance_and_is_queryable(self):
        self.synthesis()

        query = literature_query.LiteratureQuery(
            repo_root=self.repo_root,
            catalog_path=self.literature_dir / "catalog.jsonl",
            studies_path=self.studies_path,
            assessments_path=self.literature_dir / "assessments.jsonl",
            syntheses_path=self.survey_dir / "registry.jsonl",
        )
        record = query.get_synthesis("alpha-finding")
        self.assertEqual(record.kind, "synthesis")
        self.assertIn("investigation:LT-0003", record.derives_from)

    def test_synthesis_without_investigation_provenance_is_rejected(self):
        with self.assertRaises(lr.RecordError) as raised:
            self.synthesis(derives_from=["paper:alpha-2024-method"])
        self.assertEqual(raised.exception.kind, lr.SYNTHESIS_INVALID)

    def test_synthesis_invalid_document_is_rejected(self):
        with self.assertRaises(lr.RecordError) as raised:
            self.synthesis(document="../escape.md")
        self.assertEqual(raised.exception.kind, lr.INVALID_REQUEST)

    def test_rejected_synthesis_removes_the_unregistered_document(self):
        with self.assertRaises(lr.RecordError):
            self.synthesis(derives_from=["paper:alpha-2024-method"])
        self.assertFalse((self.survey_dir / "ALPHA_FINDING.md").exists())


class NoteTests(RecordFixture):
    def test_note_upserts_a_section_and_supports_an_assessment_anchor(self):
        lr.put_note(
            "LT-0003",
            "Reasoning",
            "The bounded reasoning body.",
            repo_root=self.repo_root,
            studies_path=self.studies_path,
        )
        # A heading written through the bounded note op is a valid anchor target.
        self.assessment(anchor="reasoning")

        analysis = (self.studies_dir / "LT-0003" / "analysis.md").read_text()
        self.assertIn("## Reasoning", analysis)
        self.assertIn("The bounded reasoning body.", analysis)

    def test_note_rewrite_replaces_the_section_body(self):
        for body in ("first body", "second body"):
            lr.put_note(
                "LT-0003",
                "Reasoning",
                body,
                repo_root=self.repo_root,
                studies_path=self.studies_path,
            )
        analysis = (self.studies_dir / "LT-0003" / "analysis.md").read_text()
        self.assertIn("second body", analysis)
        self.assertNotIn("first body", analysis)


class ClosedInvestigationTests(RecordFixture):
    def test_completed_investigation_rejects_further_scoped_writes(self):
        self.assessment()
        li.complete_investigation(
            "LT-0003",
            studies_path=self.studies_path,
            repo_root=self.repo_root,
            literature_dir=self.literature_dir,
        )
        with self.assertRaises(lr.RecordError) as raised:
            self.assessment(reason_summary="after close")
        self.assertEqual(raised.exception.kind, lr.INVESTIGATION_NOT_ACTIVE)


class NegativeAuthorityTests(RecordFixture):
    def test_writes_never_touch_other_research_state(self):
        self.assessment()
        self.claim()
        self.synthesis()

        self.assertEqual((self.research_dir / "FINDINGS.md").read_text(), "")
        for name in ("DECISIONS.md", "FAILURES.md", "BACKLOG.md"):
            self.assertFalse((self.research_dir / name).exists())
        self.assertFalse((self.research_dir / "proposals").exists())
        # The catalog and the paper card are untouched.
        self.assertNotIn(
            "LT-0003",
            (self.literature_dir / "catalog.jsonl").read_text(encoding="utf-8"),
        )

    def test_validate_passes_on_a_valid_state(self):
        self.assessment()
        self.synthesis()
        counts = lr.validate(
            repo_root=self.repo_root,
            literature_dir=self.literature_dir,
            studies_path=self.studies_path,
        )
        self.assertEqual(counts["assessments"], 1)
        self.assertEqual(counts["syntheses"], 1)


if __name__ == "__main__":
    unittest.main()
